import json
from typing import Dict, Any
from swissknife.modules.mcpclient import MCPSessionManager


def get_mcp_connect_tool_definition():
    """Get the tool definition for connecting to an MCP server."""
    return {
        "type": "function",
        "function": {
            "name": "mcp_connect",
            "description": "Connect to a Model Context Protocol (MCP) server",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "ID of the server to connect to (must be defined in configuration)",
                    }
                },
                "required": ["server_id"],
            },
        },
    }


def get_mcp_list_servers_tool_definition():
    """Get the tool definition for listing configured MCP servers."""
    return {
        "type": "function",
        "function": {
            "name": "mcp_list_servers",
            "description": "List all configured MCP servers and their connection status",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def get_mcp_list_tools_tool_definition():
    """Get the tool definition for listing tools available on MCP servers."""
    return {
        "type": "function",
        "function": {
            "name": "mcp_list_tools",
            "description": "List all tools available on connected MCP servers",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Optional ID of the server to list tools from. If not provided, lists tools from all servers.",
                    }
                },
                "required": [],
            },
        },
    }


def get_mcp_call_tool_definition():
    """Get the tool definition for calling a tool on an MCP server."""
    return {
        "type": "function",
        "function": {
            "name": "mcp_call_tool",
            "description": "Call a tool on a connected MCP server",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Full name of the tool to call (format: server_id.tool_name)",
                    },
                    "tool_args": {
                        "type": "object",
                        "description": "Arguments to pass to the tool",
                    },
                },
                "required": ["tool_name", "tool_args"],
            },
        },
    }


def get_mcp_disconnect_tool_definition():
    """Get the tool definition for disconnecting from MCP servers."""
    return {
        "type": "function",
        "function": {
            "name": "mcp_disconnect",
            "description": "Disconnect from all MCP servers or a specific server",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Optional ID of the server to disconnect from. If not provided, disconnects from all servers.",
                    }
                },
                "required": [],
            },
        },
    }


def get_mcp_connect_tool_handler():
    """Get the handler for the MCP connect tool."""

    async def handle_mcp_connect(params: Dict[str, Any]) -> Dict[str, Any]:
        server_id = params.get("server_id")

        # Get session manager
        manager = MCPSessionManager.get_instance()

        # Make sure configuration is loaded
        if not manager.initialized:
            await manager.initialize_servers()

        # Check if server is in configuration
        server_configs = manager.config_manager.get_enabled_servers()
        if server_id not in server_configs:
            return {
                "content": f"Server '{server_id}' is not defined in configuration or is disabled",
                "status": "error",
            }

        # Connect to server
        success = await manager.mcp_service._connect_to_server(
            server_id, server_configs[server_id]
        )

        if success:
            tools = await manager.mcp_service.list_tools(server_id)
            return {
                "content": f"Successfully connected to MCP server '{server_id}'. Available tools: {json.dumps([t['name'] for t in tools])}",
                "status": "success",
            }
        else:
            return {
                "content": f"Failed to connect to MCP server '{server_id}'",
                "status": "error",
            }

    return handle_mcp_connect


def get_mcp_list_servers_tool_handler():
    """Get the handler for the MCP list servers tool."""

    async def handle_mcp_list_servers(params: Dict[str, Any]) -> Dict[str, Any]:
        # Get session manager
        manager = MCPSessionManager.get_instance()

        # Make sure configuration is loaded
        if not manager.initialized:
            await manager.initialize_servers()

        # Get server configurations and connection status
        server_configs = manager.config_manager.configs
        connected_servers = manager.mcp_service.connected_servers

        # Format response
        servers_info = []
        for server_id, config in server_configs.items():
            servers_info.append(
                {
                    "id": server_id,
                    "name": config.name,
                    "enabled": config.enabledForAgents,
                    "connected": connected_servers.get(server_id, False),
                }
            )

        return {"content": json.dumps(servers_info, indent=2), "status": "success"}

    return handle_mcp_list_servers


def get_mcp_list_tools_tool_handler():
    """Get the handler for the MCP list tools tool."""

    async def handle_mcp_list_tools(params: Dict[str, Any]) -> Dict[str, Any]:
        server_id = params.get("server_id")

        # Get session manager
        manager = MCPSessionManager.get_instance()

        # Make sure configuration is loaded
        if not manager.initialized:
            await manager.initialize_servers()

        try:
            tools = await manager.mcp_service.list_tools(server_id)
            if not tools:
                if server_id:
                    return {
                        "content": f"No tools available from server '{server_id}' (server may not be connected)",
                        "status": "error",
                    }
                else:
                    return {
                        "content": "No tools available from any connected servers",
                        "status": "error",
                    }

            return {"content": json.dumps(tools, indent=2), "status": "success"}
        except Exception as e:
            return {"content": f"Error listing tools: {str(e)}", "status": "error"}

    return handle_mcp_list_tools


def get_mcp_call_tool_tool_handler():
    """Get the handler for the MCP call tool tool."""

    async def handle_mcp_call_tool(params: Dict[str, Any]) -> Dict[str, Any]:
        tool_name: str = str(params.get("tool_name"))
        tool_args = params.get("tool_args", {})

        # Parse server_id and tool_name from the full tool name
        if "." not in tool_name:
            return {
                "content": f"Invalid tool name format: '{tool_name}'. Expected format: 'server_id.tool_name'",
                "status": "error",
            }

        server_id, actual_tool_name = tool_name.split(".", 1)

        # Get session manager
        manager = MCPSessionManager.get_instance()

        # Make sure configuration is loaded
        if not manager.initialized:
            await manager.initialize_servers()

        try:
            result = await manager.mcp_service.call_tool(
                server_id, actual_tool_name, tool_args
            )
            return result
        except Exception as e:
            return {
                "content": f"Error calling tool '{tool_name}': {str(e)}",
                "status": "error",
            }

    return handle_mcp_call_tool


def get_mcp_disconnect_tool_handler():
    """Get the handler for the MCP disconnect tool."""

    async def handle_mcp_disconnect(params: Dict[str, Any]) -> Dict[str, Any]:
        server_id = params.get("server_id")

        # Get session manager
        manager = MCPSessionManager.get_instance()

        if server_id:
            # Disconnect from specific server
            if server_id not in manager.mcp_service.sessions:
                return {
                    "content": f"No connection to server '{server_id}'",
                    "status": "error",
                }

            # Mark server as disconnected
            manager.mcp_service.connected_servers[server_id] = False

            return {
                "content": f"Disconnected from MCP server '{server_id}'",
                "status": "success",
            }
        else:
            # Disconnect from all servers
            manager.cleanup()

            return {"content": "Disconnected from all MCP servers", "status": "success"}

    return handle_mcp_disconnect


def register(service_instance=None):
    """
    Register all MCP tools with the tool registry or directly with an agent.

    Args:
        service_instance: Not used for MCP tools, but included for consistency
        agent: Agent instance to register with directly (optional)

    This function should be called during application initialization.
    """
    # from swissknife.modules.tools.registration import register_tool
    #
    # # Register tool definitions and handlers
    # register_tool(
    #     get_mcp_connect_tool_definition, get_mcp_connect_tool_handler, None, agent
    # )
    # register_tool(
    #     get_mcp_list_servers_tool_definition,
    #     get_mcp_list_servers_tool_handler,
    #     None,
    #     agent,
    # )
    # register_tool(
    #     get_mcp_list_tools_tool_definition, get_mcp_list_tools_tool_handler, None, agent
    # )
    # register_tool(
    #     get_mcp_call_tool_definition, get_mcp_call_tool_tool_handler, None, agent
    # )
    # register_tool(
    #     get_mcp_disconnect_tool_definition, get_mcp_disconnect_tool_handler, None, agent
    # )

    # Initialize servers asynchronously
    MCPSessionManager.get_instance().initialize()
