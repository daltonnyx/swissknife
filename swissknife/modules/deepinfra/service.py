from swissknife.modules.openai import OpenAIService
from typing import Dict, Any, List, Optional, Tuple
import json
import os
import contextlib
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv


class DeepInfraService(OpenAIService):
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY not found in environment variables")
        super().__init__(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.model = "deepseek-ai/DeepSeek-V3-0324"
        self.current_input_tokens = 0
        self._provider_name = "deepinfra"
        self.current_output_tokens = 0
        self._is_stream = False

    def format_tool_result(
        self, tool_use: Dict, tool_result: Any, is_error: bool = False
    ) -> Dict[str, Any]:
        """
        Format a tool result for OpenAI API.

        Args:
            tool_use: The tool use details
            tool_result: The result from the tool execution
            is_error: Whether the result is an error

        Returns:
            A formatted message for tool response
        """
        # OpenAI format for tool responses
        message = {
            "role": "tool",
            "tool_call_id": tool_use["id"],
            "name": tool_use["name"],
            "content": str(tool_result),  # Groq expects string content
        }

        # Add error indication if needed
        if is_error:
            message["content"] = f"ERROR: {message['content']}"

        return message

    def stream_assistant_response(self, messages):
        """Stream the assistant's response with tool support."""
        stream_params = {
            "model": self.model,
            "messages": messages,
            # "stream_options": {"include_usage": True},
            "max_tokens": 8192,
        }
        stream_params["temperature"] = 0.6
        stream_params["extra_body"] = {"min_p": 0.1}

        # Add system message if provided
        if self.system_prompt:
            stream_params["messages"] = [
                {"role": "system", "content": self.system_prompt}
            ] + messages

        # Add tools if available
        if self.tools:
            stream_params["tools"] = self.tools

        response = self.client.chat.completions.create(**stream_params, stream=False)

        @contextlib.contextmanager
        def simulate_stream(data: ChatCompletion):
            if data.usage:
                self.current_input_tokens = data.usage.prompt_tokens
                self.current_output_tokens = data.usage.completion_tokens
            yield data.choices

        # Return a list containing the single response to simulate a stream
        return simulate_stream(response)

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

            # Check for tool call format in the response
            tool_call_start = "<tool_call>"
            tool_call_end = "<｜tool▁calls▁end｜>"

            if tool_call_start in content and tool_call_end in content:
                start_idx = content.find(tool_call_start)
                end_idx = content.find(tool_call_end) + len(tool_call_end)

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

            # Regular response without tool calls
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

    # def process_stream_chunk(
    #     self, chunk, assistant_response: str, tool_uses: List[Dict]
    # ) -> Tuple[str, List[Dict], int, int, Optional[str], Optional[tuple]]:
    #     """
    #     Process a single chunk from the streaming response.
    #
    #     Args:
    #         chunk: The chunk from the stream
    #         assistant_response: Current accumulated assistant response
    #         tool_uses: Current tool use information
    #
    #     Returns:
    #         tuple: (
    #             updated_assistant_response,
    #             updated_tool_uses,
    #             input_tokens,
    #             output_tokens,
    #             chunk_text,
    #             thinking_data
    #         )
    #     """
    #     chunk_text = None
    #     input_tokens = 0
    #     output_tokens = 0
    #     # thinking_content = None  # OpenAI doesn't support thinking mode
    #
    #     # Handle tool call chunks
    #     if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, "tool_calls"):
    #         delta_tool_calls = chunk.choices[0].delta.tool_calls
    #         if delta_tool_calls:
    #             # Process each tool call in the delta
    #             for tool_call_delta in delta_tool_calls:
    #                 tool_call_index = tool_call_delta.index or 0
    #
    #                 # Check if this is a new tool call
    #                 if tool_call_index >= len(tool_uses):
    #                     # Create a new tool call entry
    #                     tool_uses.append(
    #                         {
    #                             "id": tool_call_delta.id
    #                             if hasattr(tool_call_delta, "id")
    #                             else None,
    #                             "name": getattr(tool_call_delta.function, "name", "")
    #                             if hasattr(tool_call_delta, "function")
    #                             else "",
    #                             "input": {},
    #                             "type": "function",
    #                             "response": "",
    #                         }
    #                     )
    #
    #                 # Update existing tool call with new data
    #                 if hasattr(tool_call_delta, "id") and tool_call_delta.id:
    #                     tool_uses[tool_call_index]["id"] = tool_call_delta.id
    #
    #                 if hasattr(tool_call_delta, "function"):
    #                     if (
    #                         hasattr(tool_call_delta.function, "name")
    #                         and tool_call_delta.function.name
    #                     ):
    #                         tool_uses[tool_call_index]["name"] = (
    #                             tool_call_delta.function.name
    #                         )
    #
    #                     if (
    #                         hasattr(tool_call_delta.function, "arguments")
    #                         and tool_call_delta.function.arguments
    #                     ):
    #                         # Accumulate arguments as they come in chunks
    #                         current_args = tool_uses[tool_call_index].get(
    #                             "args_json", ""
    #                         )
    #                         tool_uses[tool_call_index]["args_json"] = (
    #                             current_args + tool_call_delta.function.arguments
    #                         )
    #
    #                         # Try to parse JSON if it seems complete
    #                         try:
    #                             args_json = tool_uses[tool_call_index]["args_json"]
    #                             tool_uses[tool_call_index]["input"] = json.loads(
    #                                 args_json
    #                             )
    #                             # Keep args_json for accumulation but use input for execution
    #                         except json.JSONDecodeError:
    #                             # Arguments JSON is still incomplete, keep accumulating
    #                             pass
    #
    #             # For tool calls, we don't append to assistant_response as it's handled separately
    #             return (
    #                 assistant_response or " ",
    #                 tool_uses,
    #                 input_tokens,
    #                 output_tokens,
    #                 "",
    #                 None,
    #             )
    #
    #     # Handle regular content chunks
    #     if (
    #         len(chunk.choices) > 0
    #         and hasattr(chunk.choices[0].delta, "content")
    #         and chunk.choices[0].delta.content is not None
    #     ):
    #         chunk_text = chunk.choices[0].delta.content
    #         assistant_response += chunk_text
    #
    #     # Handle final chunk with usage information
    #     if hasattr(chunk, "usage"):
    #         if hasattr(chunk.usage, "prompt_tokens"):
    #             input_tokens = chunk.usage.prompt_tokens
    #         if hasattr(chunk.usage, "completion_tokens"):
    #             output_tokens = chunk.usage.completion_tokens
    #
    #     return (
    #         assistant_response or " ",
    #         tool_uses,
    #         input_tokens,
    #         output_tokens,
    #         chunk_text,
    #         None,
    #     )
