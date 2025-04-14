import asyncio
from typing import Dict
from .config import MCPConfigManager, MCPServerConfig
from .service import MCPService


class MCPSessionManager:
    """Manager for MCP sessions and server connections."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the session manager."""
        if cls._instance is None:
            cls._instance = MCPSessionManager()
        return cls._instance

    def __init__(self):
        """Initialize the session manager."""
        self.config_manager = MCPConfigManager()
        self.mcp_service = MCPService()
        self.initialized = False

    def initialize(self) -> Dict[str, bool]:
        """Synchronous initialization method"""
        if self.initialized:
            return {}

        # Start the MCP service
        self.mcp_service.start()

        # Initialize servers using the service's event loop
        return self.mcp_service._run_async(self.initialize_servers())

    async def initialize_servers(self) -> Dict[str, bool]:
        """
        Initialize connections to all enabled MCP servers.

        Returns:
            Dictionary mapping server IDs to connection status (True if connected, False otherwise)
        """
        # Load server configurations
        self.config_manager.load_config()
        enabled_servers = self.config_manager.get_enabled_servers()

        if not enabled_servers:
            print("No enabled MCP servers found in configuration")
            return {}

        # Connect to each enabled server
        connection_results = {}
        connection_tasks = []

        for server_id, config in enabled_servers.items():
            connection_tasks.append(self._connect_to_server(server_id, config))

        # Wait for all connections to complete
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)

        # Process results
        for i, (server_id, _) in enumerate(enabled_servers.items()):
            result = results[i]
            if isinstance(result, Exception):
                print(f"Error connecting to server '{server_id}': {str(result)}")
                connection_results[server_id] = False
            else:
                connection_results[server_id] = result

        self.initialized = True
        return connection_results

    async def _connect_to_server(self, server_id: str, config: MCPServerConfig) -> bool:
        """
        Connect to a single MCP server.

        Args:
            server_id: ID of the server to connect to
            config: Configuration for the server

        Returns:
            True if connection was successful, False otherwise
        """
        try:
            return await self.mcp_service.connect_to_server(config)
        except Exception as e:
            print(f"Error connecting to server '{server_id}': {str(e)}")
            return False

    def cleanup(self):
        """Clean up all resources."""
        if self.initialized:
            self.mcp_service._run_async(self.mcp_service.cleanup())
            # self.mcp_service.stop()
            self.initialized = False
