"""
Task management for A2A protocol implementation.
"""

import asyncio
from datetime import datetime
from typing import Dict, AsyncIterable, Optional, Any
from swissknife.modules.agents import AgentManager, LocalAgent
from .types import (
    JSONRPCError,
    Task,
    TaskStatus,
    TaskState,
    SendTaskRequest,
    GetTaskRequest,
    CancelTaskRequest,
    JSONRPCResponse,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    TaskNotFoundError,
    TaskNotCancelableError,
)
from .adapters import (
    convert_a2a_message_to_swissknife,
    convert_swissknife_response_to_a2a,
    convert_swissknife_message_to_a2a,
)


class AgentTaskManager:
    """Manages tasks for a specific agent"""

    def __init__(self, agent_name: str, agent_manager: AgentManager):
        self.agent_name = agent_name
        self.agent_manager = agent_manager
        self.tasks: Dict[str, Task] = {}
        self.streaming_tasks: Dict[str, asyncio.Queue] = {}

    async def send_task(self, request: SendTaskRequest) -> JSONRPCResponse:
        """
        Handle tasks/send request for this agent.

        Args:
            request: The task request

        Returns:
            JSON-RPC response with task result
        """
        agent = self.agent_manager.get_agent(self.agent_name)
        if not agent or not isinstance(agent, LocalAgent):
            return JSONRPCResponse(
                id=request.id,
                error=JSONRPCError(
                    code=-32001, message=f"Agent {self.agent_name} not found"
                ),
            )

        # Convert A2A message to SwissKnife format
        message = convert_a2a_message_to_swissknife(request.params.message)

        # Create task with initial state
        task = Task(
            id=request.params.id,
            sessionId=request.params.sessionId,
            status=TaskStatus(state=TaskState.WORKING, timestamp=datetime.now()),
        )
        self.tasks[task.id] = task

        # Process with agent (non-blocking)
        asyncio.create_task(self._process_agent_task(agent, message, task))

        # Return initial task state
        return JSONRPCResponse(id=request.id, result=task)

    async def _process_agent_task(
        self, agent: LocalAgent, message: Dict[str, Any], task: Task
    ):
        """
        Process a task with the agent (background task).

        Args:
            agent: The agent to process the task
            message: The message to process
            task: The task object to update
        """
        try:
            print(message)
            # Add message to agent history
            agent.history.append(message)

            # # Select the agent
            # self.agent_manager.select_agent(self.agent_name)

            # Process with agent
            response_generator = agent.process_messages()

            # Create artifacts from response
            artifacts = []
            current_response = ""

            for response_chunk, chunk_text, thinking_chunk in response_generator:
                # Update current response
                if chunk_text:
                    current_response = response_chunk

                # Update task status
                task.status.state = TaskState.WORKING
                task.status.timestamp = datetime.now()

                # If this is a streaming task, send updates
                if task.id in self.streaming_tasks:
                    queue = self.streaming_tasks[task.id]

                    # Send thinking update if available
                    if thinking_chunk:
                        await queue.put(
                            TaskStatusUpdateEvent(
                                id=task.id,
                                status=TaskStatus(
                                    state=TaskState.WORKING,
                                    message=convert_swissknife_message_to_a2a(
                                        {"role": "agent", "content": thinking_chunk}
                                    ),
                                ),
                                final=False,
                            )
                        )

                    # Send chunk update
                    if chunk_text:
                        artifact = convert_swissknife_response_to_a2a(current_response)
                        await queue.put(
                            TaskArtifactUpdateEvent(id=task.id, artifact=artifact)
                        )

            # Get final result
            tool_uses, input_tokens, output_tokens = agent.get_process_result()

            # Create artifact from final response
            artifact = convert_swissknife_response_to_a2a(current_response, tool_uses)
            artifacts.append(artifact)

            # Update task with final state
            task.status.state = TaskState.COMPLETED
            task.status.timestamp = datetime.now()
            task.artifacts = artifacts

            # If this is a streaming task, send final update
            if task.id in self.streaming_tasks:
                queue = self.streaming_tasks[task.id]

                # Send final artifact
                await queue.put(TaskArtifactUpdateEvent(id=task.id, artifact=artifact))

                # Send final status
                await queue.put(
                    TaskStatusUpdateEvent(id=task.id, status=task.status, final=True)
                )

                # Mark queue as done
                await queue.put(None)

        except Exception as e:
            print(str(e))
            # Handle errors
            task.status.state = TaskState.FAILED
            task.status.timestamp = datetime.now()

            # If this is a streaming task, send error
            if task.id in self.streaming_tasks:
                queue = self.streaming_tasks[task.id]
                await queue.put(
                    TaskStatusUpdateEvent(id=task.id, status=task.status, final=True)
                )
                await queue.put(None)

    async def send_task_subscribe(
        self, request: SendTaskRequest
    ) -> AsyncIterable[JSONRPCResponse]:
        """
        Handle tasks/sendSubscribe request for this agent.

        Args:
            request: The task request

        Yields:
            JSON-RPC responses with task updates
        """
        # Create streaming queue
        queue = asyncio.Queue()
        self.streaming_tasks[request.params.id] = queue

        try:
            # Start the task
            response = await self.send_task(request)

            # If there was an error, yield it and stop
            if response.error:
                yield response
                return

            # Yield events from the queue
            while True:
                event = await queue.get()
                if event is None:  # End of stream
                    break
                yield JSONRPCResponse(id=request.id, result=event)

        finally:
            # Clean up
            self.streaming_tasks.pop(request.params.id, None)

    async def get_task(self, request: GetTaskRequest) -> JSONRPCResponse:
        """
        Handle tasks/get request for this agent.

        Args:
            request: The task request

        Returns:
            JSON-RPC response with task result
        """
        task_id = request.params.id
        if task_id not in self.tasks:
            return JSONRPCResponse(id=request.id, error=TaskNotFoundError())

        return JSONRPCResponse(id=request.id, result=self.tasks[task_id])

    async def cancel_task(self, request: CancelTaskRequest) -> JSONRPCResponse:
        """
        Handle tasks/cancel request for this agent.

        Args:
            request: The task request

        Returns:
            JSON-RPC response with task result
        """
        task_id = request.params.id
        if task_id not in self.tasks:
            return JSONRPCResponse(id=request.id, error=TaskNotFoundError())

        task = self.tasks[task_id]

        # Check if task can be canceled
        if task.status.state in [
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
        ]:
            return JSONRPCResponse(id=request.id, error=TaskNotCancelableError())

        # Update task status
        task.status.state = TaskState.CANCELED
        task.status.timestamp = datetime.now()

        # If this is a streaming task, send cancellation
        if task_id in self.streaming_tasks:
            queue = self.streaming_tasks[task_id]
            await queue.put(
                TaskStatusUpdateEvent(id=task_id, status=task.status, final=True)
            )
            await queue.put(None)

        return JSONRPCResponse(id=request.id, result=task)


class MultiAgentTaskManager:
    """Manages tasks for multiple agents"""

    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.agent_task_managers: Dict[str, AgentTaskManager] = {}

        # Initialize task managers for all agents
        for agent_name in agent_manager.agents:
            self.agent_task_managers[agent_name] = AgentTaskManager(
                agent_name, agent_manager
            )

    def get_task_manager(self, agent_name: str) -> Optional[AgentTaskManager]:
        """
        Get the task manager for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            The task manager if found, None otherwise
        """
        return self.agent_task_managers.get(agent_name)
