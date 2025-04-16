import re
from typing import Dict, List, Any, Union
from anthropic.types import TextBlockParam, ToolUseBlock
import json


class MessageTransformer:
    """Utility for transforming messages between different provider formats."""

    @staticmethod
    def standardize_messages(
        messages: List[Dict[str, Any]], source_provider: str, agent: str
    ) -> List[Dict[str, Any]]:
        """
        Convert provider-specific messages to a standard format.

        Args:
            messages: The messages to standardize
            source_provider: The provider the messages are from

        Returns:
            Standardized messages
        """
        if source_provider == "claude":
            return MessageTransformer._standardize_claude_messages(messages, agent)
        elif source_provider == "openai":
            return MessageTransformer._standardize_openai_messages(messages, agent)
        elif source_provider == "google":
            return MessageTransformer._standardize_google_messages(messages, agent)
        else:
            return MessageTransformer._standardize_groq_messages(messages, agent)

    @staticmethod
    def convert_messages(
        messages: List[Dict[str, Any]], target_provider: str
    ) -> List[Dict[str, Any]]:
        """
        Convert standardized messages to provider-specific format.

        Args:
            messages: The standardized messages to convert
            target_provider: The provider to convert to

        Returns:
            Provider-specific messages
        """

        if target_provider == "claude":
            return MessageTransformer._convert_to_claude_format(messages)
        elif target_provider == "openai":
            return MessageTransformer._convert_to_openai_format(messages)
        elif target_provider == "google":
            return MessageTransformer._convert_to_google_format(messages)
        else:
            return MessageTransformer._convert_to_groq_format(messages)

    @staticmethod
    def _standardize_claude_messages(
        messages: List[Dict[str, Any]], agent: str
    ) -> List[Dict[str, Any]]:
        """Convert Claude-specific messages to standard format."""
        standardized = []
        for msg in messages:
            std_msg = {"role": msg.get("role", "")}
            std_msg["agent"] = agent

            # Handle content based on type
            content = msg.get("content", [])
            if isinstance(content, str):
                std_msg["content"] = content
            elif isinstance(content, list):
                # For Claude's content array, extract text and other content
                text_parts = []
                tool_calls = []
                image_parts = []

                for item in content:
                    # Check if item is a dictionary or an object
                    if isinstance(item, dict):
                        # Handle dictionary-style items
                        if item.get("type") == "text":
                            text_parts.append(
                                {"type": "text", "text": item.get("text", "")}
                            )
                        elif item.get("type") == "image":
                            image_parts.append(
                                MessageTransformer._standardize_claude_image_content(
                                    item
                                )
                            )
                        elif item.get("type") == "tool_use":
                            tool_calls.append(
                                {
                                    "id": item.get("id", ""),
                                    "name": item.get("name", ""),
                                    "arguments": item.get("input", {}),
                                    "type": "function",
                                }
                            )
                        elif (
                            item.get("type") == "tool_result"
                            and msg.get("role") == "user"
                        ):
                            content = ""
                            try:
                                if isinstance(item["content"], List):
                                    content = [
                                        TextBlockParam(i) for i in item["content"]
                                    ][0]["text"]
                                else:
                                    content = item["content"]
                            except Exception as e:
                                pass
                            std_msg["tool_result"] = {
                                "tool_use_id": item.get("tool_use_id", ""),
                                "content": content,
                                "is_error": item.get("is_error", False),
                            }
                    elif isinstance(item, ToolUseBlock):
                        item_type = item.type
                        tool_calls.append(
                            {
                                "id": item.id,
                                "name": item.name,
                                "arguments": item.input,
                                "type": "function",
                            }
                        )
                    else:
                        # Handle object-style items
                        item_type = getattr(item, "type", None)
                        if item_type == "text":
                            text_parts.append(
                                {"type": "text", "text": item.get("text", "")}
                            )
                        elif item.get("type") == "image":
                            image_parts.append(
                                MessageTransformer._standardize_claude_image_content(
                                    item
                                )
                            )
                        elif item_type == "tool_use":
                            tool_calls.append(
                                {
                                    "id": getattr(item, "id", ""),
                                    "name": getattr(item, "name", ""),
                                    "arguments": getattr(item, "input", {}),
                                    "type": "function",
                                }
                            )
                        elif item_type == "tool_result" and msg.get("role") == "user":
                            item_dict = item.to_dict()
                            content = ""
                            try:
                                if isinstance(item_dict["content"], List):
                                    content = [
                                        TextBlockParam(i) for i in item_dict["content"]
                                    ]
                                else:
                                    content = item_dict["content"]
                            except Exception:
                                pass

                            std_msg["tool_result"] = {
                                "tool_use_id": getattr(item, "tool_use_id", ""),
                                "content": content,
                                "is_error": getattr(item, "is_error", False),
                            }

                if image_parts:
                    std_msg["content"] = image_parts
                elif text_parts:
                    std_msg["content"] = text_parts
                else:
                    std_msg["content"] = " "

                # Add tool calls if present
                if tool_calls:
                    std_msg["tool_calls"] = tool_calls

            standardized.append(std_msg)
        return standardized

    @staticmethod
    def _standardize_claude_image_content(content: Dict[str, Any]):
        if content.get("type", "image"):
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content.get('source', {}).get('media_type', '')};base64,{content.get('source', {}).get('data', '')}"
                },
            }
        return content

    @staticmethod
    def _standardize_openai_messages(
        messages: List[Dict[str, Any]], agent: str
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-specific messages to standard format."""
        standardized = []
        for msg in messages:
            std_msg = {"role": msg.get("role", "")}
            std_msg["agent"] = agent

            # Handle content
            if "content" in msg:
                if (
                    isinstance(msg["content"], str)
                    and msg.get("role", "") == "assistant"
                ):
                    std_msg["content"] = [{"type": "text", "text": msg["content"]}]

                else:
                    std_msg["content"] = msg["content"]

            # Handle tool calls
            if "tool_calls" in msg:
                std_msg["tool_calls"] = []
                for tool_call in msg["tool_calls"]:
                    std_tool_call = {
                        "id": tool_call.get("id"),
                        "name": tool_call.get("function", {}).get("name"),
                        "arguments": json.loads(
                            tool_call.get("function", {}).get("arguments")
                        ),
                        "type": tool_call.get("type", "function"),
                    }
                    std_msg["tool_calls"].append(std_tool_call)

            # Handle tool results
            if msg.get("role") == "tool":
                std_msg["tool_result"] = {
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": msg.get("content"),
                    "is_error": msg.get("content", "").startswith("ERROR:"),
                }

            standardized.append(std_msg)
        return standardized

    @staticmethod
    def _standardize_google_messages(
        messages: List[Dict[str, Any]], agent: str
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-specific messages to standard format."""
        standardized = []
        for msg in messages:
            std_msg = {"role": msg.get("role", "")}
            std_msg["agent"] = agent

            # Handle content
            if "content" in msg:
                if (
                    isinstance(msg["content"], str)
                    and msg.get("role", "") == "assistant"
                ):
                    std_msg["content"] = [{"type": "text", "text": msg["content"]}]

                else:
                    std_msg["content"] = msg["content"]

            # Handle tool calls
            if "tool_calls" in msg:
                std_msg["tool_calls"] = []
                for tool_call in msg["tool_calls"]:
                    arguments = tool_call.get("arguments", {})
                    std_tool_call = {
                        "id": tool_call.get("id"),
                        "name": tool_call.get("name"),
                        "arguments": arguments
                        if isinstance(arguments, dict)
                        else json.loads(arguments),
                        "type": tool_call.get("type", "function"),
                    }
                    std_msg["tool_calls"].append(std_tool_call)

            # Handle tool results
            if msg.get("role") == "tool":
                std_msg["tool_result"] = {
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": msg.get("content"),
                    "is_error": msg.get("content", "").startswith("ERROR:"),
                }

            standardized.append(std_msg)
        return standardized

    @staticmethod
    def _standardize_groq_messages(
        messages: List[Dict[str, Any]], agent: str
    ) -> List[Dict[str, Any]]:
        """Convert Groq-specific messages to standard format."""
        # Groq uses OpenAI format, so we can reuse that
        standardized = []
        for msg in messages:
            std_msg = {"role": msg.get("role", "")}
            std_msg["agent"] = agent

            # Handle content
            if "content" in msg:
                if (
                    isinstance(msg["content"], str)
                    and msg.get("role", "") == "assistant"
                ):
                    std_msg["content"] = [{"type": "text", "text": msg["content"]}]

                else:
                    std_msg["content"] = msg["content"]

            # Handle tool calls
            if "tool_calls" in msg:
                std_msg["tool_calls"] = []
                for tool_call in msg["tool_calls"]:
                    std_tool_call = {
                        "id": tool_call.get("id"),
                        "name": tool_call.get("function", {}).get("name"),
                        "arguments": json.loads(
                            tool_call.get("function", {}).get("arguments")
                        ),
                        "type": tool_call.get("type", "function"),
                    }
                    std_msg["tool_calls"].append(std_tool_call)

            # Handle tool results
            if msg.get("role") == "tool":
                std_msg["tool_result"] = {
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": msg.get("content"),
                    "is_error": msg.get("content", "").startswith("ERROR:"),
                }

            standardized.append(std_msg)
        return standardized

    @staticmethod
    def _convert_to_claude_format(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert standard messages to Claude format."""
        claude_messages = []
        for msg in messages:
            claude_msg = {"role": msg.get("role", "")}
            if claude_msg["role"] == "tool":
                claude_msg["role"] = "user"
            # Handle content
            if "content" in msg:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    # For assistant messages with tool calls, we need to format differently
                    # Fix issue with empty content
                    if isinstance(msg["content"], List):
                        claude_msg["content"] = msg["content"]
                    else:
                        if msg["content"] == "":
                            msg["content"] = " "
                        claude_msg["content"] = [
                            {"type": "text", "text": msg["content"]}
                        ]

                    # Add tool use blocks
                    for tool_call in msg.get("tool_calls", []):
                        tool_use = {
                            "type": "tool_use",
                            "id": tool_call.get("id", ""),
                            "name": tool_call.get("name", ""),
                            "input": tool_call.get("arguments", {}),
                        }
                        claude_msg["content"].append(tool_use)
                else:
                    # Regular content
                    if msg["content"] is str:
                        claude_msg["content"] = [
                            {"type": "text", "text": msg["content"]}
                        ]
                    else:
                        claude_msg["content"] = (
                            MessageTransformer._convert_content_to_claude_format(
                                msg["content"]
                            )
                        )

            # Handle tool results
            if "tool_result" in msg:
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_result"].get("tool_use_id", ""),
                    "content": msg["tool_result"].get("content", ""),
                }

                if msg["tool_result"].get("is_error", False):
                    tool_result["is_error"] = True

                claude_msg["content"] = [tool_result]

            claude_messages.append(claude_msg)
        return claude_messages

    @staticmethod
    def _convert_content_to_claude_format(
        content: Union[Dict[str, Any], List[Dict[str, Any]]],
    ):
        new_content = None

        pattern = r"^data:([^;]+);base64,(.*)$"
        if isinstance(content, Dict):
            if content.get("type", "text") == "image_url":
                data_url = content.get("image_url", {}).get("url", "")
                match = re.match(pattern, data_url, re.DOTALL)

                if match:
                    mime_type = match.group(1)
                    base64_data = match.group(2)
                    new_content = {
                        "type": "image",
                        "source": {
                            "media_type": mime_type,
                            "data": base64_data,
                            "type": "base64",
                        },
                    }
                    return new_content
            else:
                return content
        elif isinstance(content, List):
            new_content = []
            for c in content:
                new_content.append(
                    MessageTransformer._convert_content_to_claude_format(c)
                )
            return new_content
        return content

    @staticmethod
    def _convert_to_openai_format(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert standard messages to OpenAI format."""
        openai_messages = []
        for msg in messages:
            openai_msg = {"role": msg.get("role", "")}

            # Handle content
            if "content" in msg:
                openai_msg["content"] = msg["content"]

            # Handle tool calls
            if "tool_calls" in msg:
                openai_msg["tool_calls"] = []
                for tool_call in msg.get("tool_calls", []):
                    # Convert arguments to JSON string if it's not already a string
                    arguments = tool_call.get("arguments", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)

                    openai_msg["tool_calls"].append(
                        {
                            "id": tool_call.get("id", ""),
                            "type": tool_call.get("type", "function"),
                            "function": {
                                "name": tool_call.get("name", ""),
                                "arguments": arguments,
                            },
                        }
                    )

            if "tool_call_id" in msg:
                openai_msg["role"] = "tool"
                openai_msg["tool_call_id"] = msg.get("tool_call_id", "")
            # Handle tool results
            if "tool_result" in msg:
                openai_msg["role"] = "tool"
                openai_msg["tool_call_id"] = msg["tool_result"].get("tool_use_id", "")
                openai_msg["content"] = msg["tool_result"].get("content", "")

                if msg["tool_result"].get("is_error", False):
                    openai_msg["content"] = f"ERROR: {openai_msg['content']}"

            openai_messages.append(openai_msg)
        return openai_messages

    @staticmethod
    def _convert_to_google_format(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert standard messages to OpenAI format."""
        google_messages = []
        for msg in messages:
            google_msg = {"role": msg.get("role", "")}

            # Handle content
            if "content" in msg:
                if (
                    isinstance(msg["content"], List)
                    and msg["content"]
                    and msg["role"] == "assistant"
                ):
                    google_msg["content"] = msg["content"][0]["text"]
                else:
                    google_msg["content"] = msg["content"]

            # Handle tool calls
            if "tool_calls" in msg:
                google_msg["tool_calls"] = []
                for tool_call in msg.get("tool_calls", []):
                    # Convert arguments to JSON string if it's not already a string
                    arguments = tool_call.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)

                    google_msg["tool_calls"].append(
                        {
                            "id": tool_call.get("id", ""),
                            "type": tool_call.get("type", "function"),
                            "name": tool_call.get("name", ""),
                            "arguments": arguments,
                        }
                    )

            if "tool_call_id" in msg:
                google_msg["tool_call_id"] = msg.get("tool_call_id", "")
            # Handle tool results
            if "tool_result" in msg:
                google_msg["role"] = "tool"
                google_msg["tool_call_id"] = msg["tool_result"].get("tool_use_id", "")
                google_msg["content"] = msg["tool_result"].get("content", "")

                if msg["tool_result"].get("is_error", False):
                    google_msg["content"] = f"ERROR: {google_msg['content']}"

            google_messages.append(google_msg)
        return google_messages

    @staticmethod
    def _convert_to_groq_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert standard messages to Groq format."""
        # Groq uses OpenAI format, so we can reuse that
        groq_messages = []
        for msg in messages:
            groq_msg = {"role": msg.get("role", "")}

            # Handle content
            if "content" in msg:
                print(msg["content"])
                if (
                    isinstance(msg["content"], List)
                    and len(msg["content"]) > 0
                    and msg["role"] == "assistant"
                ):
                    groq_msg["content"] = msg["content"][0]["text"]
                else:
                    groq_msg["content"] = msg["content"]

            # Handle tool calls
            if "tool_calls" in msg:
                groq_msg["tool_calls"] = []
                for tool_call in msg.get("tool_calls", []):
                    # Convert arguments to JSON string if it's not already a string
                    arguments = tool_call.get("arguments", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)

                    groq_msg["tool_calls"].append(
                        {
                            "id": tool_call.get("id", ""),
                            "type": tool_call.get("type", "function"),
                            "function": {
                                "name": tool_call.get("name", ""),
                                "arguments": arguments,
                            },
                        }
                    )

            if "tool_call_id" in msg:
                groq_msg["role"] = "tool"
                groq_msg["tool_call_id"] = msg.get("tool_call_id", "")
            # Handle tool results
            if "tool_result" in msg:
                groq_msg["role"] = "tool"
                groq_msg["tool_call_id"] = msg["tool_result"].get("tool_use_id", "")
                groq_msg["content"] = msg["tool_result"].get("content", "")

                if msg["tool_result"].get("is_error", False):
                    groq_msg["content"] = f"ERROR: {groq_msg['content']}"

            print(groq_msg)
            groq_messages.append(groq_msg)
        return groq_messages
