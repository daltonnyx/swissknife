from typing import Dict, Any, Callable
import os
from .spec_validation import SpecPromptValidationService
from .service import CodeAssistant


def get_spec_validation_tool_definition(provider="claude") -> Dict[str, Any]:
    """
    Get the tool definition for the spec validation tool.

    Args:
        provider: The LLM provider (claude, openai, groq)

    Returns:
        Tool definition compatible with the specified provider
    """
    tool_description = (
        "Run through a checklist with aider prompt and return refine suggestions"
    )
    tool_arguments = {
        "prompt": {
            "type": "string",
            "description": "The aider prompt to be reviewed",
        },
        "repo_path": {
            "type": "string",
            "description": "The repo directory path to review aider prompt against",
        },
    }
    tool_required = ["prompt", "repo_path"]
    if provider == "claude":
        return {
            "name": "review_aider_prompt",
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
                "name": "review_aider_prompt",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_spec_validation_tool_handler(
    spec_validation_service: SpecPromptValidationService,
) -> Callable:
    """
    Get the handler function for the spec validation tool.

    Args:
        spec_validation_service: The spec validation service instance

    Returns:
        Function that handles spec validation requests
    """

    def handle_spec_validation(**params) -> str:
        prompt = params.get("prompt", "")
        repo_path = params.get("repo_path", "")

        real_path = os.path.expanduser(repo_path)
        # Validate the spec
        validation_result = spec_validation_service.validate(prompt, real_path)

        # Check if validation meets criteria

        # Format the result as a readable string
        return validation_result

    return handle_spec_validation


def get_implement_spec_prompt_tool_definition(provider="claude") -> Dict[str, Any]:
    """
    Get the tool definition for the implement spec prompt tool.

    Args:
        provider: The LLM provider (claude, openai, groq)

    Returns:
        Tool definition compatible with the specified provider
    """
    tool_description = "NEVER run this without user request explicitly. Generate code implementation via aider using a aider prompt"
    tool_arguments = {
        "aider_prompt": {
            "type": "string",
            "description": "The Aider prompt that instructs Aider to implement",
        },
        "repo_path": {
            "type": "string",
            "description": "The repository path where code should be implemented",
        },
    }
    tool_required = ["aider_prompt", "repo_path"]
    if provider == "claude":
        return {
            "name": "implement_aider_prompt",
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
                "name": "implement_aider_prompt",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_implement_spec_prompt_tool_handler(
    code_assistant: CodeAssistant,
) -> Callable:
    """
    Get the handler function for the implement spec prompt tool.

    Returns:
        Function that handles implement spec prompt requests
    """

    def handle_implement_spec_prompt(**params) -> str:
        spec_prompt = params.get("aider_prompt", "")
        repo_path = params.get("repo_path", "")

        if not spec_prompt or not repo_path:
            return "Error: Both aider_prompt and repo_path are required."

        try:
            result = code_assistant.generate_implementation(spec_prompt, repo_path)
            return result
        except Exception as e:
            return f"Error implementing spec prompt: {str(e)}"

    return handle_implement_spec_prompt


def register(service_instance=None, agent=None):
    """
    Register all coder tools with the tool registry or directly with an agent

    Args:
        service_instance: The spec validation service instance (optional)
        agent: Agent instance to register with directly (optional)
    """
    if service_instance is None:
        service_instance = SpecPromptValidationService()

    from swissknife.modules.tools.registration import register_tool

    # Register spec validation tool
    register_tool(
        get_spec_validation_tool_definition,
        get_spec_validation_tool_handler,
        service_instance,
        agent,
    )

    code_assistant = CodeAssistant()
    # Register implement spec prompt tool
    register_tool(
        get_implement_spec_prompt_tool_definition,
        get_implement_spec_prompt_tool_handler,
        code_assistant,
        agent,
    )
