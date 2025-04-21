from contextlib import AsyncExitStack
from typing import Dict, Any, List, Optional, Callable
from mcp import ClientSession, StdioServerParameters
from mcp.types import EmbeddedResource, ImageContent, TextContent
from mcp.client.stdio import stdio_client
from swissknife.modules.agents import AgentManager
from swissknife.modules.tools.registry import ToolRegistry
from .config import MCPServerConfig
import asyncio
import threading


class MCPService:
    """Service for interacting with Model Context Protocol (MCP) servers."""

    def __init__(self):
        """Initialize the MCP service."""
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.connected_servers: Dict[str, bool] = {}
        self.tools_cache: Dict[str, Dict[str, Any]] = {}
        self.stdio_transports: Dict[str, tuple] = {}
        self.loop = asyncio.new_event_loop()

    async def connect_to_server(self, server_config: MCPServerConfig) -> bool:
        """
        Connect to an MCP server.

        Args:
            server_config: Configuration for the server to connect to

        Returns:
            bool: True if connection was successful, False otherwise
        """
        server_id = server_config.name
        try:
            # Set up server parameters
            server_params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env,
            )

            # Connect to the server
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio_transports[server_id] = stdio_transport
            stdio, write = stdio_transport

            # Create and initialize session
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            await session.initialize()

            # Store session
            self.sessions[server_id] = session
            self.connected_servers[server_id] = True
            for agent in server_config.enabledForAgents:
                # Cache available tools
                await self.register_server_tools(server_id, agent)

            return True
        except Exception as e:
            print(f"Error connecting to MCP server '{server_id}': {str(e)}")
            self.connected_servers[server_id] = False
            return False

    async def register_server_tools(self, server_id: str, agent=None) -> None:
        """
        Register all tools from a connected server.

        Args:
            server_id: ID of the server to register tools from
        """
        if server_id not in self.sessions or not self.connected_servers.get(server_id):
            print(f"Cannot register tools: Server '{server_id}' is not connected")
            return

        try:
            # Get tools from server
            response = await self.sessions[server_id].list_tools()

            # Cache tools
            self.tools_cache[server_id] = {tool.name: tool for tool in response.tools}

            if agent:
                agent_manager = AgentManager.get_instance()
                registry = agent_manager.get_local_agent(agent)
            else:
                # Register each tool with the tool registry
                registry = ToolRegistry.get_instance()
            for tool in response.tools:
                # Create namespaced tool definition
                def tool_definition_factory(tool_info=tool, srv_id=server_id):
                    def get_definition(provider=None):
                        return self._format_tool_definition(tool_info, srv_id, provider)

                    return get_definition

                # Create tool handler
                handler_factory = self._create_tool_handler(server_id, tool.name)

                # Register the tool
                if registry:
                    registry.register_tool(
                        tool_definition_factory(), handler_factory, self
                    )
        except Exception as e:
            print(f"Error registering tools from server '{server_id}': {str(e)}")
            self.connected_servers[server_id] = False

    def _format_tool_definition(
        self, tool: Any, server_id: str, provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format a tool definition for the tool registry.

        Args:
            tool: Tool information from the server
            server_id: ID of the server the tool belongs to
            provider: LLM provider to format for (if None, uses default format)

        Returns:
            Formatted tool definition
        """
        # Create namespaced tool name
        namespaced_name = f"{server_id}_{tool.name}"

        # Format for different providers
        if provider == "claude":
            return {
                "name": namespaced_name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
        else:  # Default format (OpenAI-compatible)
            return {
                "type": "function",
                "function": {
                    "name": namespaced_name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }

    def _run_async(self, coro):
        """Helper method to run coroutines in the service's event loop"""
        if self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def _create_tool_handler(self, server_id: str, tool_name: str) -> Callable:
        """
        Create a handler function for a tool.

        Args:
            server_id: ID of the server the tool belongs to
            tool_name: Name of the tool

        Returns:
            Handler function for the tool
        """

        def handler_factory(service_instance=None):
            def handler(
                **params,
            ) -> list[TextContent | ImageContent | EmbeddedResource]:
                print(params)
                if server_id not in self.sessions or not self.connected_servers.get(
                    server_id
                ):
                    print("Not connected to server")
                    raise Exception(
                        f"Cannot call tool: Server '{server_id}' is not connected"
                    )

                try:
                    result = self._run_async(
                        self.sessions[server_id].call_tool(tool_name, params)
                    )
                    return result.content
                except Exception as e:
                    print(str(e))
                    raise e

            return handler

        return handler_factory

    async def list_tools(self, server_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available tools from a connected MCP server or all servers.

        Args:
            server_id: Optional ID of the server to list tools from. If None, lists tools from all servers.

        Returns:
            List of tools with their metadata
        """
        if server_id is not None:
            if server_id not in self.sessions or not self.connected_servers.get(
                server_id
            ):
                return []

            try:
                response = await self.sessions[server_id].list_tools()
                self.tools_cache[server_id] = {
                    tool.name: tool for tool in response.tools
                }

                return [
                    {
                        "name": f"{server_id}.{tool.name}",
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in response.tools
                ]
            except Exception as e:
                print(f"Error listing tools from server '{server_id}': {str(e)}")
                self.connected_servers[server_id] = False
                return []
        else:
            # List tools from all connected servers
            all_tools = []
            for srv_id in list(self.sessions.keys()):
                all_tools.extend(await self.list_tools(srv_id))
            return all_tools

    async def call_tool(
        self, server_id: str, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.

        Args:
            server_id: ID of the server to call the tool on
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool

        Returns:
            Dict containing the tool's response
        """
        if server_id not in self.sessions or not self.connected_servers.get(server_id):
            return {
                "content": f"Cannot call tool: Server '{server_id}' is not connected",
                "status": "error",
            }

        if server_id not in self.tools_cache or tool_name not in self.tools_cache.get(
            server_id, {}
        ):
            # Refresh tools cache
            await self.list_tools(server_id)
            if (
                server_id not in self.tools_cache
                or tool_name not in self.tools_cache.get(server_id, {})
            ):
                return {
                    "content": f"Tool '{tool_name}' is not available on server '{server_id}'",
                    "status": "error",
                }

        try:
            result = await self.sessions[server_id].call_tool(tool_name, tool_args)
            return {"content": result.content, "status": "success"}
        except Exception as e:
            self.connected_servers[server_id] = False
            return {
                "content": f"Error calling tool '{tool_name}' on server '{server_id}': {str(e)}",
                "status": "error",
            }

    def start(self):
        """Start the service's event loop in a separate thread"""

        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()

    def stop(self):
        """Stop the service's event loop"""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if hasattr(self, "loop_thread"):
            self.loop_thread.join()
        if self.loop and not self.loop.is_closed():
            self.loop.close()

    async def cleanup(self):
        """Clean up resources for all connected servers."""
        await self.exit_stack.aclose()
        self.sessions = {}
        self.connected_servers = {}
        self.stdio_transports = {}
