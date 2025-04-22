"""
A2A protocol server implementation for SwissKnife.
"""

import json
import logging
from typing import Callable, Optional
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from starlette.requests import Request
from sse_starlette.sse import EventSourceResponse
from pydantic import ValidationError

from swissknife.modules.agents import AgentManager
from .registry import AgentRegistry
from .task_manager import MultiAgentTaskManager
from .types import (
    A2ARequest,
    JSONRPCError,
    JSONRPCResponse,
    InvalidRequestError,
    JSONParseError,
    InternalError,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure debug logs are captured

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class A2AServer:
    """A2A server that exposes multiple agents"""

    def __init__(
        self,
        agent_manager: AgentManager,
        host: str = "0.0.0.0",
        port: int = 41241,
        base_url: Optional[str] = None,
    ):
        logger.info(f"Initializing A2A server with host={host}, port={port}")
        self.agent_manager = agent_manager
        self.host = host
        self.port = port
        self.base_url = base_url or f"http://{host}:{port}"
        logger.debug(f"Using base URL: {self.base_url}")

        # Create agent registry
        self.agent_registry = AgentRegistry(agent_manager, self.base_url)

        # Create task manager
        self.task_manager = MultiAgentTaskManager(agent_manager)

        # Create Starlette app
        self.app = self._create_app()

    def _create_app(self) -> Starlette:
        """
        Create the Starlette application with routes.

        Returns:
            The configured Starlette application
        """
        logger.debug("Creating Starlette application")
        routes = [
            Route("/agents", self._list_agents, methods=["GET"]),
        ]

        # Add routes for each agent
        for agent_name in self.agent_manager.agents:
            logger.debug(f"Creating routes for agent: {agent_name}")
            routes.append(
                Mount(
                    f"/{agent_name}",
                    routes=[
                        Route(
                            "/.well-known/agent.json",
                            self._get_agent_card_factory(agent_name),
                            methods=["GET"],
                        ),
                        # Single JSON-RPC endpoint
                        Route(
                            "/",
                            self._process_jsonrpc_request_factory(agent_name),
                            methods=["POST"],
                        ),
                    ],
                )
            )

        return Starlette(routes=routes)

    def _get_agent_card_factory(self, agent_name: str) -> Callable:
        """
        Factory function to create agent card handler for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Handler function for agent card requests
        """

        async def get_agent_card(request: Request):
            agent_card = self.agent_registry.get_agent_card(agent_name)
            if not agent_card:
                return JSONResponse(
                    {"error": f"Agent {agent_name} not found"}, status_code=404
                )
            return JSONResponse(agent_card.model_dump(exclude_none=True))

        return get_agent_card

    def _process_jsonrpc_request_factory(self, agent_name: str) -> Callable:
        """
        Factory function to create JSON-RPC request handler for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Handler function for JSON-RPC requests
        """

        async def process_jsonrpc_request(request: Request):
            logger.debug(f"Received JSON-RPC request for agent {agent_name}")
            try:
                # Get task manager for this agent
                task_manager = self.task_manager.get_task_manager(agent_name)
                if not task_manager:
                    logger.warning(f"Agent {agent_name} not found")
                    return JSONResponse(
                        {"error": f"Agent {agent_name} not found"}, status_code=404
                    )

                # Parse request body
                body = await request.json()
                logger.debug(f"Request body: {body}")

                # Validate as A2A request
                try:
                    json_rpc_request = A2ARequest.validate_python(body)
                except ValidationError as e:
                    logging.debug(f"cannot validate_python {e} ")
                    error = JSONRPCResponse(
                        id=body.get("id"),
                        error=InvalidRequestError(data=e.errors()),
                    )
                    return JSONResponse(
                        error.model_dump(exclude_none=True), status_code=400
                    )

                # Process based on method
                method = json_rpc_request.method
                logger.debug(f"Processing method: {method}")

                if method == "tasks/send":
                    logger.debug("Handling tasks/send request")
                    result = await task_manager.send_task(json_rpc_request)
                    logger.debug(f"tasks/send result: {result}")
                    return JSONResponse(result.model_dump(exclude_none=True))

                elif method == "tasks/sendSubscribe":
                    result_stream = task_manager.send_task_subscribe(json_rpc_request)

                    async def event_generator():
                        async for item in result_stream:
                            print(item)
                            yield {
                                "data": json.dumps(item.model_dump(exclude_none=True))
                            }

                    return EventSourceResponse(event_generator())

                elif method == "tasks/get":
                    result = await task_manager.get_task(json_rpc_request)
                    return JSONResponse(result.model_dump(exclude_none=True))

                elif method == "tasks/cancel":
                    result = await task_manager.cancel_task(json_rpc_request)
                    return JSONResponse(result.model_dump(exclude_none=True))

                else:
                    logger.error(f"Invalid method requested: {method}")
                    logger.error(f"Request ID: {json_rpc_request.id}")
                    logger.error(f"Request params: {json_rpc_request.params}")
                    error = JSONRPCResponse(
                        id=json_rpc_request.id,
                        error=JSONRPCError(code=-32601, message="Method not found"),
                    )
                    return JSONResponse(
                        error.model_dump(exclude_none=True), status_code=400
                    )

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.error(f"Error position: line {e.lineno}, column {e.colno}")
                logger.error(f"Error document: {e.doc}")
                error = JSONRPCResponse(id=None, error=JSONParseError())
                return JSONResponse(
                    error.model_dump(exclude_none=True), status_code=400
                )

            except ValidationError as e:
                logger.error(f"Validation error: {str(e)}")
                for error in e.errors():
                    logger.error(f"Error type: {error['type']}")
                    logger.error(f"Error message: {error['msg']}")
                error = JSONRPCResponse(
                    id=None, error=InvalidRequestError(data=e.errors())
                )
                return JSONResponse(
                    error.model_dump(exclude_none=True), status_code=400
                )

            except Exception as e:
                logger.exception(f"Error processing request: {e}")
                error = JSONRPCResponse(
                    id=body.get("id") if "body" in locals() else None,
                    error=InternalError(),
                )
                return JSONResponse(
                    error.model_dump(exclude_none=True), status_code=500
                )

        return process_jsonrpc_request

    async def _list_agents(self, request: Request):
        """
        List all available agents.

        Args:
            request: The HTTP request

        Returns:
            JSON response with list of agents
        """
        logger.debug("Handling list_agents request")
        agents = self.agent_registry.list_agents()
        logger.debug(f"Found {len(agents)} agents")
        return JSONResponse([agent.model_dump() for agent in agents])

    def start(self):
        """Start the A2A server"""
        import uvicorn

        logger.info(f"Starting A2A server on {self.host}:{self.port}")
        logger.info(f"Available agents: {list(self.agent_manager.agents.keys())}")
        uvicorn.run(self.app, host=self.host, port=self.port)
