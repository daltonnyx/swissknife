import os
import base64
import mimetypes
from typing import Dict, Any
from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv
from modules.llm.base import BaseLLMService
from ..prompts.constants import (
    EXPLAIN_PROMPT,
    SUMMARIZE_PROMPT,
    CHAT_SYSTEM_PROMPT,
)


INPUT_TOKEN_COST_PER_MILLION = 3.0
OUTPUT_TOKEN_COST_PER_MILLION = 15.0


def read_text_file(file_path):
    """Read and return the contents of a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {str(e)}")
        return None


def read_binary_file(file_path):
    """Read a binary file and return base64 encoded content."""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        return base64.b64encode(content).decode("utf-8")
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {str(e)}")
        return None


class AnthropicService(BaseLLMService):
    """Anthropic-specific implementation of the LLM service."""

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-7-sonnet-20250219"
        self.tools = []  # Initialize empty tools list
        self.tool_handlers = {}  # Map tool names to handler functions

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * INPUT_TOKEN_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * OUTPUT_TOKEN_COST_PER_MILLION
        return input_cost + output_cost

    def _process_content(self, prompt_template, content, max_tokens=2048):
        """Process content with a given prompt template."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_template.format(content=content),
                    }
                ],
            )

            content_block = message.content[0]
            if not isinstance(content_block, TextBlock):
                raise ValueError(
                    "Unexpected response type: message content is not a TextBlock"
                )

            # Calculate and log token usage and cost
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            total_cost = self.calculate_cost(input_tokens, output_tokens)

            print(f"\nToken Usage Statistics:")
            print(f"Input tokens: {input_tokens:,}")
            print(f"Output tokens: {output_tokens:,}")
            print(f"Total tokens: {input_tokens + output_tokens:,}")
            print(f"Estimated cost: ${total_cost:.4f}")

            return content_block.text
        except Exception as e:
            raise Exception(f"Failed to process content: {str(e)}")

    def summarize_content(self, content: str) -> str:
        """Summarize the provided content using Claude."""
        return self._process_content(SUMMARIZE_PROMPT, content, max_tokens=2048)

    def explain_content(self, content: str) -> str:
        """Explain the provided content using Claude."""
        return self._process_content(EXPLAIN_PROMPT, content, max_tokens=1500)

    def process_file_for_message(self, file_path):
        """Process a file and return the appropriate message content."""
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type == "application/pdf":
            pdf_data = read_binary_file(file_path)
            if pdf_data:
                print(f"📄 Including PDF document: {file_path}")
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data,
                    },
                }
        else:
            content = read_text_file(file_path)
            if content:
                print(f"📄 Including text file: {file_path}")
                return {
                    "type": "text",
                    "text": f"Content of {file_path}:\n\n{content}",
                }

        return None

    def handle_file_command(self, file_path):
        """Handle the /file command and return message content."""
        mime_type, _ = mimetypes.guess_type(file_path)
        message_content = []

        if mime_type == "application/pdf":
            pdf_data = read_binary_file(file_path)
            if pdf_data:
                message_content.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        },
                    }
                )
                message_content.append(
                    {
                        "type": "text",
                        "text": "I'm sharing this PDF file with you. Please analyze it.",
                    }
                )
                print(f"📄 Including PDF document: {file_path}")
            else:
                return None
        else:
            content = read_text_file(file_path)
            if content:
                message_content = [
                    {
                        "type": "text",
                        "text": f"I'm sharing this file with you:\n\nContent of {file_path}:\n\n{content}",
                    }
                ]
                print(f"📄 Including text file: {file_path}")
            else:
                return None

        return message_content

    def register_tool(self, tool_definition, handler_function):
        """
        Register a tool with its handler function.

        Args:
            tool_definition (dict): The tool definition following Anthropic's schema
            handler_function (callable): Function to call when tool is used
        """
        self.tools.append(tool_definition)
        self.tool_handlers[tool_definition["name"]] = handler_function
        print(f"🔧 Registered tool: {tool_definition['name']}")

    def execute_tool(self, tool_name, tool_params):
        """
        Execute a registered tool with the given parameters.

        Args:
            tool_name (str): Name of the tool to execute
            tool_params (dict): Parameters to pass to the tool

        Returns:
            dict: Result of the tool execution
        """
        if tool_name not in self.tool_handlers:
            return {"error": f"Tool '{tool_name}' not found"}

        handler = self.tool_handlers[tool_name]
        result = handler(**tool_params)
        return result

    def process_stream_chunk(self, chunk, assistant_response, tool_use):
        """
        Process a single chunk from the Anthropic streaming response.

        Args:
            chunk: The chunk from the stream
            assistant_response: Current accumulated assistant response
            tool_use: Current tool use information

        Returns:
            tuple: (
                updated_assistant_response (str),
                updated_tool_use (dict or None),
                input_tokens (int),
                output_tokens (int),
                chunk_text (str or None) - text to print for this chunk
            )
        """
        chunk_text = None
        input_tokens = 0
        output_tokens = 0

        if chunk.type == "content_block_delta" and hasattr(chunk.delta, "text"):
            chunk_text = chunk.delta.text
            assistant_response += chunk_text
        elif (
            chunk.type == "message_start"
            and hasattr(chunk, "message")
            and hasattr(chunk.message, "usage")
        ):
            if hasattr(chunk.message.usage, "input_tokens"):
                input_tokens = chunk.message.usage.input_tokens
        elif chunk.type == "message_delta" and hasattr(chunk, "usage") and chunk.usage:
            if hasattr(chunk.usage, "output_tokens"):
                output_tokens = chunk.usage.output_tokens
        elif chunk.type == "message_stop" and hasattr(chunk, "message"):
            if (
                hasattr(chunk.message, "stop_reason")
                and chunk.message.stop_reason == "tool_use"
                and hasattr(chunk.message, "content")
            ):
                # Extract tool use information
                for content_block in chunk.message.content:
                    if (
                        hasattr(content_block, "type")
                        and content_block.type == "tool_use"
                    ):
                        tool_use = {
                            "name": content_block.name,
                            "input": content_block.input,
                            "id": content_block.id,
                            "response": content_block,
                        }
                        break

        return assistant_response, tool_use, input_tokens, output_tokens, chunk_text

    def format_tool_result(
        self, tool_use: Dict, tool_result: Any, is_error: bool = False
    ) -> Dict[str, Any]:
        """
        Format a tool result for Claude API.

        Args:
            tool_use_id: The ID of the tool use
            tool_result: The result from the tool execution
            is_error: Whether the result is an error

        Returns:
            A formatted message that can be appended to the messages list
        """
        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result,
                }
            ],
        }

        # Add is_error flag if this is an error
        if is_error:
            message["content"][0]["is_error"] = True

        return message

    def format_assistant_message(self, assistant_response: str, tool_use: Dict = None) -> Dict[str, Any]:
        """Format the assistant's response for Anthropic API."""
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_response}],
        }
        
        # If there's a tool use response, add it to the content array
        if tool_use and "response" in tool_use and tool_use["response"] != "":
            assistant_message["content"].append(tool_use["response"])
            
        return assistant_message

    def stream_assistant_response(self, messages):
        """Stream the assistant's response with tool support."""
        stream_params = {
            "model": self.model,
            "max_tokens": 4096,
            "system": CHAT_SYSTEM_PROMPT,
            "messages": messages,
        }

        # Add tools if available
        if self.tools:
            stream_params["tools"] = self.tools
        return self.client.messages.stream(**stream_params)
