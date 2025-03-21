from typing import Dict, Any, Callable


def get_handoff_tool_definition(provider="claude") -> Dict[str, Any]:
    """
    Get the definition for the handoff tool.

    Args:
        provider: The LLM provider (claude, openai, groq)

    Returns:
        The tool definition
    """
    if provider == "claude":
        return {
            "name": "handoff",
            "description": "Hands off the conversation to a specialized agent when the current task requires different expertise",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_agent": {
                        "type": "string",
                        "enum": ["Architect", "TechLead", "Documentation"],
                        "description": "The name of the agent to hand off to.",
                    },
                    "task": {
                        "type": "string",
                        "description": "The task for handoff agent to handle",
                    },
                    "context_summary": {
                        "type": "string",
                        "description": "A summary of the conversation context to provide to the new agent",
                    },
                },
                "required": ["target_agent", "task"],
            },
        }
    elif provider in ["openai", "groq"]:
        return {
            "type": "function",
            "function": {
                "name": "handoff",
                "description": "Hands off the conversation to a specialized agent when the current task requires different expertise",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_agent": {
                            "type": "string",
                            "enum": ["Architect", "TechLead", "Documentation"],
                            "description": "The name of the agent to hand off to",
                        },
                        "task": {
                            "type": "string",
                            "description": "The task for handoff agent to handle",
                        },
                        "context_summary": {
                            "type": "string",
                            "description": "A summary of the conversation context to provide to the new agent",
                        },
                    },
                    "required": ["target_agent", "task"],
                },
            },
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_handoff_tool_handler(agent_manager) -> Callable:
    """
    Get the handler function for the handoff tool.

    Args:
        agent_manager: The agent manager instance

    Returns:
        The handler function
    """

    def handler(**params) -> str:
        """
        Handle a handoff request.

        Args:
            target_agent: The name of the agent to hand off to
            reason: The reason for the handoff
            context_summary: Optional summary of the conversation context

        Returns:
            A string describing the result of the handoff
        """
        target_agent = params.get("target_agent")
        task = params.get("task")
        context_summary = params.get("context_summary", "")

        if not target_agent:
            return "Error: No target agent specified"

        if not task:
            return "Error: No task specified for the handoff"

        result = agent_manager.perform_handoff(target_agent, task, context_summary)

        if result["success"]:
            return f"Successfully handed off to {target_agent} with {context_summary}. Continue to {task}"
        else:
            available_agents = ", ".join(result.get("available_agents", []))
            return f"Error: {result.get('error')}. Available agents: {available_agents}"

    return handler


def register(agent_manager, agent=None):
    """
    Register the handoff tool with all agents or a specific agent.

    Args:
        agent_manager: The agent manager instance
        agent: Specific agent to register with (optional)
    """

    # Create the tool definition and handler

    from swissknife.modules.tools.registration import register_tool

    register_tool(
        get_handoff_tool_definition, get_handoff_tool_handler, agent_manager, agent
    )
