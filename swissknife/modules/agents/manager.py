from typing import Dict, Any, List, Optional
from .base import Agent


class AgentManager:
    """Manager for specialized agents."""

    _instance = None

    def __new__(cls):
        """Ensure only one instance is created (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(AgentManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the agent manager."""
        if not self._initialized:
            self.agents = {}
            self.current_agent = None
            self.handoff_history = []
            self._initialized = True

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of AgentManager."""
        if cls._instance is None:
            cls._instance = AgentManager()
        return cls._instance

    def register_agent(self, agent: Agent):
        """
        Register an agent with the manager.

        Args:
            agent: The agent to register
        """
        self.agents[agent.name] = agent
        # Set the first registered agent as the default
        # if not self.current_agent:
        #     self.select_agent(agent.name)
        # else:
        #     # Keep the agent in deactivated state until selected
        #     agent.deactivate()

    def select_agent(self, agent_name: str) -> bool:
        """
        Select an agent by name.

        Args:
            agent_name: The name of the agent to select

        Returns:
            True if the agent was selected, False otherwise
        """
        if agent_name in self.agents:
            # Get the new agent
            new_agent = self.agents[agent_name]

            # If there was a previous agent, deactivate it
            if self.current_agent:
                self.current_agent.deactivate()

            # Set the new agent as current
            self.current_agent = new_agent

            # Activate the new agent
            self.current_agent.activate()

            return True
        return False

    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """
        Get an agent by name.

        Args:
            agent_name: The name of the agent to get

        Returns:
            The agent, or None if not found
        """
        return self.agents.get(agent_name)

    def get_current_agent(self) -> Optional[Agent]:
        """
        Get the current agent.

        Returns:
            The current agent, or None if no agent is selected
        """
        return self.current_agent

    def perform_handoff(
        self, target_agent_name: str, task: str, context_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a handoff to another agent.

        Args:
            target_agent_name: The name of the agent to hand off to
            reason: The reason for the handoff
            context_summary: Optional summary of the conversation context

        Returns:
            A dictionary with the result of the handoff
        """
        if target_agent_name not in self.agents:
            return {
                "success": False,
                "error": f"Agent '{target_agent_name}' not found",
                "available_agents": list(self.agents.keys()),
            }

        source_agent = self.current_agent

        # Record the handoff
        handoff_record = {
            "from": source_agent.name if source_agent else "None",
            "to": target_agent_name,
            "reason": task,
            "context_summary": context_summary,
        }
        self.handoff_history.append(handoff_record)

        # Set the new current agent
        self.select_agent(target_agent_name)

        return {"success": True, "handoff": handoff_record}

    def update_llm_service(self, llm_service):
        """
        Update the LLM service for all agents.

        Args:
            llm_service: The new LLM service to use
        """
        if self.current_agent:
            # Deactivate the current agent
            self.current_agent.deactivate()

            # Update the LLM service for the current agent
            self.current_agent.update_llm_service(llm_service)

            # Reactivate the agent with the new LLM service
            self.current_agent.activate()

            # Update all other agents' LLM service but keep them deactivated
            for name, agent in self.agents.items():
                if agent != self.current_agent:
                    agent.update_llm_service(llm_service)

    def route_message(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Route messages to the current agent.

        Args:
            messages: The messages to route

        Returns:
            The processed messages with the agent's system prompt
        """
        if not self.current_agent:
            raise ValueError("No agent selected")

        return self.current_agent.process_messages(messages)
