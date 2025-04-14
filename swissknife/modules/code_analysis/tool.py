from os import path as os_path
from typing import Dict, Any, Callable
from .service import CodeAnalysisService


# Tool definition function
def get_code_analysis_tool_definition(provider="claude") -> Dict[str, Any]:
    """
    Return the tool definition for code analysis based on provider.

    Args:
        provider: The LLM provider ("claude", "groq", or "openai")

    Returns:
        Dict containing the tool definition
    """
    description = "Analyzes the structure of source code files within a repository, creating a structural map. This identifies key code elements, enabling code understanding and project organization insights. Explain what insights you are hoping to gain from analyzing the repository before using this tool."

    tool_arguments = {
        "path": {
            "type": "string",
            "description": "The root directory to analyze. Use '.' to analyze all source files in the current directory, or specify a subdirectory (e.g., 'src') to analyze files within that directory. Choose the path that will provide the most relevant information for the task at hand.",
        }
    }
    tool_required = ["path"]

    if provider == "claude":
        return {
            "name": "analyze_repo",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # provider == "openai"
        return {
            "type": "function",
            "function": {
                "name": "analyze_repo",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


# Tool handler function
def get_code_analysis_tool_handler(
    code_analysis_service: CodeAnalysisService,
) -> Callable:
    """Return the handler function for the code analysis tool."""

    def handler(**params) -> str:
        path = params.get("path", ".")
        path = os_path.expanduser(path)
        result = code_analysis_service.analyze_code_structure(path)
        if isinstance(result, dict) and "error" in result:
            raise Exception(f"Failed to analyze code: {result['error']}")

        return result

    return handler


def get_file_content_tool_definition(provider="claude"):
    """
    Return the tool definition for retrieving file content based on provider.

    Args:
        provider: The LLM provider ("claude", "groq", or "openai")

    Returns:
        Dict containing the tool definition
    """
    tool_description = "Reads the content of a code file, or a specific code element within that file (function or class body). Use this to examine the logic of specific functions, the structure of classes, or the overall content of a file. ALWAYS justify why you're reading the file and what you expect to learn from it. Before using, consider if `analyze_repo` would provide sufficient information at a higher level."

    tool_arguments = {
        "file_path": {
            "type": "string",
            "description": "The relative path from the current directory of the agent to the local repository file. Example: 'src/my_module.py'",
        },
        "element_type": {
            "type": "string",
            "description": "The type of code element to extract. ALWAYS use this when targeting a specific element.",
            "enum": ["class", "function"],
        },
        "element_name": {
            "type": "string",
            "description": "The name of the class or function to extract. ALWAYS use this when targeting a specific element. Case-sensitive.",
        },
        "scope_path": {
            "type": "string",
            "description": "A dot-separated path to resolve ambiguity when multiple elements share the same name (e.g., 'ClassName.method_name'). Required only if the element name is ambiguous. Omit if unnecessary.",
        },
    }
    tool_required = ["file_path"]

    if provider == "claude":
        return {
            "name": "read_file",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # provider == "openai"
        return {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_file_content_tool_handler(
    code_analysis_service: CodeAnalysisService,
):
    """Returns a function that handles the get_file_content tool."""

    def handler(**params) -> str:
        file_path = params.get("file_path")
        element_type = params.get("element_type")
        element_name = params.get("element_name")
        scope_path = params.get("scope_path")

        if not file_path:
            raise Exception("File path is required")

        # Validate parameters
        if element_type and element_type not in ["class", "function"]:
            raise Exception("Element type must be 'class' or 'function'")

        if (element_type and not element_name) or (element_name and not element_type):
            raise Exception(
                "Both element_type and element_name must be provided together"
            )

        results = code_analysis_service.get_file_content(
            file_path, element_type, element_name, scope_path
        )

        content = ""

        for path, code in results.items():
            content += f"{path}: {code}\n"

        # If we're getting a specific element, format the output accordingly
        if element_type and element_name:
            scope_info = f" in {scope_path}" if scope_path else ""
            return f"CONTENT OF {element_name} {element_type}{scope_info}: {content}"
        else:
            # If we're getting the whole file content
            return content

    return handler


def register(service_instance=None, agent=None):
    """
    Register this tool with the central registry or directly with an agent

    Args:
        service_instance: The code analysis service instance
        agent: Agent instance to register with directly (optional)
    """
    from swissknife.modules.tools.registration import register_tool

    register_tool(
        get_code_analysis_tool_definition,
        get_code_analysis_tool_handler,
        service_instance,
        agent,
    )

    register_tool(
        get_file_content_tool_definition,
        get_file_content_tool_handler,
        service_instance,
        agent,
    )
