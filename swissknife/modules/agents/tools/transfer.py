from typing import Dict, Any, Callable


def get_transfer_tool_definition(provider="claude") -> Dict[str, Any]:
    """
    Get the definition for the transfer tool.

    Args:
        provider: The LLM provider (claude, openai, groq)

    Returns:
        The tool definition
    """
    tool_description = "transfer to a specialized agent when the current task requires expertise beyond the current agent's capabilities. Provide a clear explanation to the user why the transfer is necessary. IMPORTANT: transfered agent cannot see your conversation with user"
    tool_arguments = {
        "target_agent": {
            "type": "string",
            "description": "The name of the specialized agent to transfer to. Refer to the ## Agents for list of available agents",
        },
        "task": {
            "type": "string",
            "description": "A precise description of the task the target agent should perform. This description MUST include the triggering keywords that prompted the transfer. Be specific and actionable.",
        },
        "relevant_messages": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "MUST include 0-based index of previous chat messages relates to this task, user messages, tool use results, assistant messages",
        },
        "report_back": {
            "type": "boolean",
            "default": "false",
            "description": "Indicated task need to transfer back to original Agent for further processing",
        },
        "report_result": {
            "type": "string",
            "default": "",
            "description": "When `report_back` is `true`, provide the detailed results of the completed task for the original agent. This should include comprehensive information about what was accomplished and any relevant findings.",
        },
    }
    tool_required = ["target_agent", "task", "relevant_messages"]
    if provider == "claude":
        return {
            "name": "transfer",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": "transfer",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_transfer_tool_handler(agent_manager) -> Callable:
    """
    Get the handler function for the transfer tool.

    Args:
        agent_manager: The agent manager instance

    Returns:
        The handler function
    """

    def handler(**params) -> str:
        """
        Handle a transfer request.

        Args:
            target_agent: The name of the agent to transfer to
            reason: The reason for the transfer
            context_summary: Optional summary of the conversation context

        Returns:
            A string describing the result of the transfer
        """
        target_agent = params.get("target_agent")
        task = params.get("task")
        relevant_messages = params.get("relevant_messages", [])
        report_back = params.get("report_back", True)
        report_result = params.get("report_result", "")

        if not target_agent:
            return "Error: No target agent specified"

        if not task:
            return "Error: No task specified for the transfer"

        if target_agent == agent_manager.current_agent.name:
            return "Error: Cannot transfer to same agent"

        result = agent_manager.perform_transfer(target_agent, task, relevant_messages)
        if target_agent == "None":
            return "Error: Task is completed. This transfer is invalid"

        response = ""

        if result["success"] and result["transfer"]["from"] != "None":
            response = f"Here is your task from {result['transfer']['from']}: {task}"

            if report_back and "transfer" in result:
                response += f"\n\nTransfer back to {result['transfer']['from']} with detail report for further processing."

            if report_result.strip():
                response = response + f"\n\n Task result {report_result}"

            if result["transfer"]["relevant_data"]:
                response += f"\n\nRelevant messages:\n{'\n'.join(result['transfer']['relevant_data'])}"

            return response

        else:
            available_agents = ", ".join(result.get("available_agents", []))
            return f"Error: {result.get('error')}. Available agents: {available_agents}"

    return handler


def register(agent_manager, agent=None):
    """
    Register the transfer tool with all agents or a specific agent.

    Args:
        agent_manager: The agent manager instance
        agent: Specific agent to register with (optional)
    """

    # Create the tool definition and handler

    from swissknife.modules.tools.registration import register_tool

    register_tool(
        get_transfer_tool_definition, get_transfer_tool_handler, agent_manager, agent
    )
