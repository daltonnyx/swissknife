from typing import Dict, Any, Callable
from .service import MemoryService


def get_memory_forget_tool_definition(provider="claude") -> Dict[str, Any]:
    """
    Get the tool definition for forgetting memories based on provider.

    Args:
        provider: The LLM provider ("claude" or "groq")

    Returns:
        Dict containing the tool definition
    """
    tool_description = "Removes memories related to a specific topic from the conversation history. This is useful for clearing sensitive information, irrelevant details, or outdated information that might conflict with the current task. Use this sparingly and only when absolutely necessary to avoid losing valuable context. Provide a clear justification for why the topic is being removed."
    tool_arguments = {
        "topic": {
            "type": "string",
            "description": "Keywords describing the topic to be forgotten. Be precise and comprehensive to ensure all relevant memories are removed. Include any IDs or specific identifiers related to the topic.",
        },
    }
    tool_required = ["topic"]
    if provider == "claude":
        return {
            "name": "forget_memory_topic",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # provider == "groq"
        return {
            "type": "function",
            "function": {
                "name": "forget_memory_topic",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_memory_forget_tool_handler(memory_service: MemoryService) -> Callable:
    """
    Get the handler function for the memory forget tool.

    Args:
        memory_service: The memory service instance

    Returns:
        Function that handles memory forgetting requests
    """

    def handle_memory_forget(**params) -> str:
        topic = params.get("topic")

        if not topic:
            return "Error: Topic is required for forgetting memories."

        try:
            result = memory_service.forget_topic(topic)
            if result["success"]:
                return result["message"]
            else:
                return f"Unable to forget memories: {result['message']}"
        except Exception as e:
            return f"Error forgetting memories: {str(e)}"

    return handle_memory_forget


def get_memory_retrieve_tool_definition(provider="claude") -> Dict[str, Any]:
    """
    Get the tool definition for retrieving memories based on provider.

    Args:
        provider: The LLM provider ("claude" or "groq")

    Returns:
        Dict containing the tool definition
    """
    tool_description = "Retrieves relevant information from past conversations and stored knowledge, based on keywords. ALWAYS use this tool as a primary method for finding relevant information and context. Use specific keywords for best results. Consider the likely topics and key terms from prior interactions when formulating your keywords."
    tool_arguments = {
        "keywords": {
            "type": "string",
            "description": "Keywords used to search for relevant information in memory. Use specific and descriptive terms to narrow the search and retrieve the most useful results. Consider synonyms and related terms to broaden the search.",
        },
    }
    tool_required = ["keywords"]
    if provider == "claude":
        return {
            "name": "retrieve_memory",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # provider == "groq"
        return {
            "type": "function",
            "function": {
                "name": "retrieve_memory",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_memory_retrieve_tool_handler(memory_service: MemoryService) -> Callable:
    """
    Get the handler function for the memory retrieve tool.

    Args:
        memory_service: The memory service instance

    Returns:
        Function that handles memory retrieval requests
    """

    def handle_memory_retrieve(**params) -> str:
        keywords = params.get("keywords")
        limit = params.get("limit", 10)

        if not keywords:
            return "Error: Keywords are required for memory retrieval."

        try:
            result = memory_service.retrieve_memory(keywords, limit)
            return result
        except Exception as e:
            return f"Error retrieving memories: {str(e)}"

    return handle_memory_retrieve


def register(service_instance=None, agent=None):
    """
    Register this tool with the central registry or directly with an agent

    Args:
        service_instance: The memory service instance
        agent: Agent instance to register with directly (optional)
    """
    from swissknife.modules.tools.registration import register_tool

    register_tool(
        get_memory_retrieve_tool_definition,
        get_memory_retrieve_tool_handler,
        service_instance,
        agent,
    )
    register_tool(
        get_memory_forget_tool_definition,
        get_memory_forget_tool_handler,
        service_instance,
        agent,
    )
