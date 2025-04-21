import os
import contextlib
import json
import mimetypes
import threading
import itertools
import rich
import re
from rich.live import Live
from typing import Dict, Any, List, Optional, Tuple
from groq import Groq
from dotenv import load_dotenv
from groq.types.chat import ChatCompletion
from swissknife.modules.llm.base import BaseLLMService, read_binary_file, read_text_file
from swissknife.modules.llm.model_registry import ModelRegistry
from ..prompts.constants import (
    EXPLAIN_PROMPT,
    SUMMARIZE_PROMPT,
    CHAT_SYSTEM_PROMPT,
)


class GroqService(BaseLLMService):
    """Groq-specific implementation of the LLM service."""

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)
        # Set default model - can be updated based on Groq's available models
        self.model = "qwen-qwq-32b"
        self.tools = []  # Initialize empty tools list
        self.tool_handlers = {}  # Map tool names to handler functions
        self._provider_name = "groq"
        self.current_input_tokens = 0
        self.current_output_tokens = 0
        self.system_prompt = CHAT_SYSTEM_PROMPT
        self._is_stream = False

    def set_think(self, budget_tokens) -> bool:
        """
        Enable or disable thinking mode with the specified token budget.

        Args:
            budget_tokens (int): Token budget for thinking. 0 to disable thinking mode.

        Returns:
            bool: True if thinking mode is supported and successfully set, False otherwise.
        """
        print("Thinking mode is not supported for Groq models.")
        return False

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost based on token usage."""
        current_model = ModelRegistry.get_instance().get_model(self.model)
        if current_model:
            input_cost = (input_tokens / 1_000_000) * current_model.input_token_price_1m
            output_cost = (
                output_tokens / 1_000_000
            ) * current_model.output_token_price_1m
            return input_cost + output_cost
        return 0.0

    def _process_content(self, prompt_template, content, max_tokens=2048):
        """Process content with a given prompt template."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_template.format(content=content),
                    }
                ],
            )

            # Calculate and log token usage and cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_cost = self.calculate_cost(input_tokens, output_tokens)

            print("\nToken Usage Statistics:")
            print(f"Input tokens: {input_tokens:,}")
            print(f"Output tokens: {output_tokens:,}")
            print(f"Total tokens: {input_tokens + output_tokens:,}")
            print(f"Estimated cost: ${total_cost:.4f}")

            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Failed to process content: {str(e)}")

    def summarize_content(self, content: str) -> str:
        """Summarize the provided content using Groq."""
        return self._process_content(SUMMARIZE_PROMPT, content, max_tokens=2048)

    def explain_content(self, content: str) -> str:
        """Explain the provided content using Groq."""
        return self._process_content(EXPLAIN_PROMPT, content, max_tokens=1500)

    def _process_file(self, file_path):
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type and mime_type.startswith("image/"):
            if "vision" not in ModelRegistry.get_model_capabilities(self.model):
                return None
            image_data = read_binary_file(file_path)
            if image_data:
                message_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}",
                        "detail": "high",
                    },
                }
                return message_content
        else:
            content = read_text_file(file_path)
            if content:
                message_content = {
                    "type": "text",
                    "text": f"I'm sharing this file with you:\n\nContent of {file_path}:\n\n{content}",
                }

                print(f"📄 Including text file: {file_path}")
                return message_content
            else:
                return None

    def process_file_for_message(self, file_path):
        """Process a file and return the appropriate message content."""

        return self._process_file(file_path)

    def handle_file_command(self, file_path):
        """Handle the /file command and return message content."""
        content = self._process_file(file_path)
        if content:
            return [content]
        return None

    def _loading_animation(self, stop_event):
        """Display a loading animation in the terminal."""
        spinner = itertools.cycle(["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
        fun_words = [
            "Pondering",
            "Cogitating",
            "Ruminating",
            "Contemplating",
            "Brainstorming",
            "Calculating",
            "Processing",
            "Analyzing",
            "Deciphering",
            "Meditating",
            "Daydreaming",
            "Scheming",
            "Brewing",
            "Conjuring",
            "Inventing",
            "Imagining",
        ]
        import random

        fun_word = random.choice(fun_words)
        console = rich.get_console()
        with Live("", console=console, auto_refresh=True) as live:
            while not stop_event.is_set():
                live.update(f"{fun_word} {next(spinner)}")
        # live.update("")

    def register_tool(self, tool_definition, handler_function):
        """
        Register a tool with its handler function.

        Args:
            tool_definition (dict): The tool definition following OpenAI's function schema
            handler_function (callable): Function to call when tool is used
        """
        self.tools.append(tool_definition)
        self.tool_handlers[tool_definition["function"]["name"]] = handler_function
        print(f"🔧 Registered tool: {tool_definition['function']['name']}")

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

    def stream_assistant_response(self, messages):
        """Stream the assistant's response with tool support."""
        stream_params = {
            "model": self.model,
            "max_completion_tokens": 4096,
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
        }

        # Add system message if provided
        if self.system_prompt:
            system_role = "user" if "deepseek" in self.model else "system"
            stream_params["messages"] = [
                {
                    "role": f"{system_role}",
                    "content": """DO NOT generate Chinese characters.""",
                },
                {"role": f"{system_role}", "content": self.system_prompt},
            ] + messages

        if self.model == "qwen-qwq-32b":
            stream_params["reasoning_format"] = "parsed"
            # stream_params["messages"].append(
            #     {"role": "assistant", "content": "<think>\n"}
            # )

        # Add tools if available
        if self.tools and "tool_use" in ModelRegistry.get_model_capabilities(
            self.model
        ):
            stream_params["tools"] = self.tools

            # Start loading animation for tool-based requests
            stop_animation = threading.Event()
            animation_thread = threading.Thread(
                target=self._loading_animation, args=(stop_animation,)
            )
            animation_thread.daemon = True
            animation_thread.start()

            try:
                # Use non-streaming mode for tool support
                response = self.client.chat.completions.create(**stream_params)
            finally:
                # Stop the animation when response is received
                stop_animation.set()
                animation_thread.join()

            @contextlib.contextmanager
            def simulate_stream(data: ChatCompletion):
                self.current_input_tokens = data.usage.prompt_tokens
                self.current_output_tokens = data.usage.completion_tokens
                yield data.choices

            # Return a list containing the single response to simulate a stream
            return simulate_stream(response)
        else:
            # Use actual streaming when no tools are needed
            return self.client.chat.completions.create(**stream_params, stream=True)

    def process_stream_chunk(
        self, chunk, assistant_response, tool_uses
    ) -> Tuple[str, List[Dict], int, int, Optional[str], Optional[tuple]]:
        """
        Process a single chunk from the streaming response.

        Args:
            chunk: The chunk from the stream
            assistant_response: Current accumulated assistant response
            tool_uses: Current tool use information

        Returns:
            tuple: (
                updated_assistant_response,
                updated_tool_uses,
                input_tokens,
                output_tokens,
                chunk_text,
                thinking_data
            )
        """
        # Check if this is a non-streaming response (for tool use)
        thinking_content = None

        input_tokens = self.current_input_tokens
        self.current_input_tokens = 0
        output_tokens = self.current_output_tokens
        self.current_output_tokens = 0
        if hasattr(chunk, "message"):
            # This is a complete response, not a streaming chunk
            message = chunk.message
            content = message.content or " "
            # Check for tool calls
            if hasattr(message, "reasoning") and message.reasoning:
                thinking_content = (message.reasoning, None)
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    function = tool_call.function

                    tool_uses.append(
                        {
                            "id": tool_call.id,
                            "name": function.name,
                            "input": json.loads(function.arguments),
                            "type": tool_call.type,
                            "response": "",
                        }
                    )

                # Return with tool use information and the full content
                return (
                    content,
                    tool_uses,
                    input_tokens,
                    output_tokens,
                    content,  # Return the full content to be printed
                    thinking_content,
                )

            content, tool_uses = self._parse_tool_calls_from_content(content, tool_uses)

            return (
                content,
                tool_uses,
                input_tokens,
                output_tokens,
                content,  # Return the full content to be printed
                thinking_content,
            )

        # Handle regular streaming chunk
        chunk_text = chunk.choices[0].delta.content or ""
        updated_assistant_response = assistant_response + chunk_text

        return (
            updated_assistant_response,
            tool_uses,
            input_tokens,
            output_tokens,
            chunk_text,
            thinking_content,
        )

    def _parse_tool_calls_from_content(self, content, tool_uses):
        """
        Parse tool calls from content and update tool_uses list.

        Args:
            content (str): The content to parse
            tool_uses (list): Current list of tool uses

        Returns:
            tuple: (updated_content, updated_tool_uses)
        """
        # Parse <tool_call> format
        tool_call_start = "<tool_call>"
        tool_call_end = "<｜tool▁calls▁end｜>"

        start_idx = content.find(tool_call_start)
        end_idx = (
            content.find(tool_call_end) + len(tool_call_end)
            if content.find(tool_call_end) != -1
            else -1
        )
        if start_idx != -1 and end_idx != -1:
            tool_call_content = content[
                start_idx + len(tool_call_start) : end_idx - len(tool_call_end)
            ]

            try:
                tool_data = json.loads(tool_call_content)
                tool_uses.append(
                    {
                        "id": f"call_{len(tool_uses)}",  # Generate an ID
                        "name": tool_data.get("name", ""),
                        "input": tool_data.get("arguments", {}),
                        "type": "function",
                        "response": "",
                    }
                )

                # Remove the tool call from the response
                content = content[:start_idx] + content[end_idx:]
            except json.JSONDecodeError:
                # If we can't parse the JSON, just continue
                pass
            return content, tool_uses

        # Parse <function=name>{...}</function=name> format
        function_pattern = r"<function=([^>]+)>(.*?)</function=\1>"

        matches = re.finditer(function_pattern, content, re.DOTALL)
        for match in matches:
            function_name = match.group(1)
            function_content = match.group(2)

            try:
                # Try to parse the function content as JSON
                function_data = json.loads(function_content)

                tool_uses.append(
                    {
                        "id": f"call_{len(tool_uses)}",
                        "name": function_name,
                        "input": function_data,
                        "type": "function",
                        "response": "",
                    }
                )

                # Remove the function call from the content
                content = content.replace(match.group(0), "")
            except json.JSONDecodeError:
                # If we can't parse the JSON, just continue
                pass

        # Parse <mermaid_render_mermaid_chart>{...}</mermaid_render_mermaid_chart> format
        mermaid_pattern = r"<([a-zA-Z_]+)>(.*?)</\1>"

        matches = re.finditer(mermaid_pattern, content, re.DOTALL)
        for match in matches:
            tool_name = match.group(1)
            tool_content = match.group(2)

            # Check if the tool name is in the registered tools
            if any(
                tool.get("function", {}).get("name") == tool_name for tool in self.tools
            ):
                try:
                    # Try to parse the tool content as JSON
                    tool_data = json.loads(tool_content)

                    tool_uses.append(
                        {
                            "id": f"call_{len(tool_uses)}",
                            "name": tool_name,
                            "input": tool_data,
                            "type": "function",
                            "response": "",
                        }
                    )

                    # Remove the tool call from the content
                    content = content.replace(match.group(0), "")
                except json.JSONDecodeError:
                    # If we can't parse the JSON, just continue
                    pass

        return content, tool_uses

    def format_tool_result(
        self, tool_use: Dict, tool_result: Any, is_error: bool = False
    ) -> Dict[str, Any]:
        """
        Format a tool result for Groq API.

        Args:
            tool_use_id: The ID of the tool use
            tool_result: The result from the tool execution
            is_error: Whether the result is an error

        Returns:
            A formatted message that can be appended to the messages list
        """
        # Groq follows OpenAI format for tool responses
        message = {
            "role": "tool",
            "tool_call_id": tool_use["id"],
            "name": tool_use["name"],
            "content": str(tool_result),  # Groq expects string content
        }

        # Note: OpenAI/Groq format doesn't have a standard way to indicate errors
        # We could potentially prefix the content with "ERROR: " if needed
        if is_error:
            message["content"] = f"ERROR: {message['content']}"

        return message

    def format_assistant_message(
        self, assistant_response: str, tool_uses: list[Dict] | None = None
    ) -> Dict[str, Any]:
        """Format the assistant's response for Groq API."""
        # Groq uses a simpler format with just a string content
        if tool_uses:
            return {
                "role": "assistant",
                "content": assistant_response,
                "tool_calls": [
                    {
                        "id": tool_use["id"],
                        "function": {
                            "name": tool_use["name"],
                            "arguments": json.dumps(tool_use["input"]),
                        },
                        "type": tool_use["type"],
                    }
                    for tool_use in tool_uses
                ],
            }
        else:
            return {
                "role": "assistant",
                "content": assistant_response,
            }

    def format_thinking_message(self, thinking_data) -> Optional[Dict[str, Any]]:
        """
        Format thinking content into the appropriate message format for Groq.

        Args:
            thinking_data: Tuple containing (thinking_content, thinking_signature)
                or None if no thinking data is available

        Returns:
            Dict[str, Any]: A properly formatted message containing thinking blocks
        """
        # Groq doesn't support thinking blocks, so we return None
        return None

    def validate_spec(self, prompt: str) -> str:
        """
        Validate a specification prompt using Groq.

        Args:
            prompt: The specification prompt to validate

        Returns:
            Validation result as a JSON string
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=8192,
                temperature=0.6,
                top_p=0.95,
                reasoning_format="parsed",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                # Groq doesn't support response_format, so we rely on the prompt
            )

            # Calculate and log token usage and cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_cost = self.calculate_cost(input_tokens, output_tokens)

            print("\nSpec Validation Token Usage:")
            print(f"Input tokens: {input_tokens:,}")
            print(f"Output tokens: {output_tokens:,}")
            print(f"Total tokens: {input_tokens + output_tokens:,}")
            print(f"Estimated cost: ${total_cost:.4f}")

            text = response.choices[0].message.content
            if text is None:
                raise ValueError("Cannot validate this spec")
            think_tag = "<think>"
            end_think_tag = "</think>"
            think_start_idx = text.find(think_tag)
            think_end_idx = text.rfind(end_think_tag)
            if think_start_idx > -1 and think_end_idx > -1:
                text = (
                    text[:think_start_idx] + text[think_end_idx + len(end_think_tag) :]
                )
            start_tag = "<SpecificationReview>"
            end_tag = "</SpecificationReview>"
            start_idx = text.rindex(start_tag)
            end_idx = text.rindex(end_tag) + len(end_tag)
            result = text[start_idx:end_idx].strip()
            return result

        except Exception as e:
            raise Exception(f"Failed to validate specification: {str(e)}")

    def set_system_prompt(self, system_prompt: str):
        """
        Set the system prompt for the LLM service.

        Args:
            system_prompt: The system prompt to use
        """
        self.system_prompt = system_prompt

    def clear_tools(self):
        """
        Clear all registered tools from the LLM service.
        """
        self.tools = []
        self.tool_handlers = {}
