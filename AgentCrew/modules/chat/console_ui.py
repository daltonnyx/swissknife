import asyncio
import sys
import time
import pyperclip
import re
import threading
import random
import itertools
import queue
from threading import Thread, Event
from typing import Dict, Any, List
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.text import Text
from rich.panel import Panel
from rich.live import Live
from rich.style import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.formatted_text import HTML
import AgentCrew

from AgentCrew.modules.chat.message_handler import MessageHandler, Observer
from AgentCrew.modules import logger
from AgentCrew.modules.chat.completers import ChatCompleter

# Rich styles
RICH_STYLE_YELLOW = Style(color="yellow", bold=False)
RICH_STYLE_GREEN = Style(color="green", bold=False)
RICH_STYLE_BLUE = Style(color="blue", bold=False)
RICH_STYLE_RED = Style(color="red", bold=False)
RICH_STYLE_GRAY = Style(color="grey66", bold=False)

RICH_STYLE_YELLOW_BOLD = Style(color="yellow", bold=True)
RICH_STYLE_GREEN_BOLD = Style(color="green", bold=True)
RICH_STYLE_BLUE_BOLD = Style(color="blue", bold=True)
RICH_STYLE_RED_BOLD = Style(color="red", bold=True)

CODE_THEME = "lightbulb"


class ConsoleUI(Observer):
    """
    A console-based UI for the interactive chat that implements the Observer interface
    to receive updates from the MessageHandler.
    """

    def __init__(self, message_handler: MessageHandler):
        """
        Initialize the ConsoleUI.

        Args:
            message_handler: The MessageHandler instance that this UI will observe.
        """
        self.message_handler = message_handler
        self.message_handler.attach(self)

        self.console = Console()
        self.live = None  # Will be initialized during response streaming
        self._last_ctrl_c_time = 0
        self.latest_assistant_response = ""
        self.session_cost = 0.0
        self._live_text_data = ""
        self._loading_stop_event = None
        self._loading_thread = None

        # Set up key bindings
        self.kb = self._setup_key_bindings()

        # Threading for user input
        self._input_queue = queue.Queue()
        self._input_thread = None
        self._input_stop_event = Event()
        self._current_prompt_session = None

    def listen(self, event: str, data: Any = None):
        """
        Update method required by the Observer interface. Handles events from the MessageHandler.

        Args:
            event: The type of event that occurred.
            data: The data associated with the event.
        """

        if event == "thinking_started":
            self.stop_loading_animation()  # Stop loading on first chunk
            self.display_thinking_started(data)  # data is agent_name
        elif event == "thinking_chunk":
            self.display_thinking_chunk(data)  # data is the thinking chunk
        elif event == "user_message_created":
            self.console.print("\n")
        elif event == "response_chunk":
            self.stop_loading_animation()  # Stop loading on first chunk
            _, assistant_response = data
            self.update_live_display(assistant_response)  # data is the response chunk
        elif event == "tool_use":
            self.stop_loading_animation()  # Stop loading on first chunk
            self.display_tool_use(data)  # data is the tool use object
        elif event == "tool_result":
            pass
            # self.display_tool_result(data)  # data is dict with tool_use and tool_result
        elif event == "tool_error":
            self.display_tool_error(data)  # data is dict with tool_use and error
        elif event == "tool_confirmation_required":
            self.stop_loading_animation()  # Stop loading on first chunk
            self.display_tool_confirmation_request(
                data
            )  # data is the tool use with confirmation ID
        elif event == "tool_denied":
            self.display_tool_denied(data)  # data is the tool use that was denied
        elif event == "response_completed" or event == "assistant_message_added":
            # pass
            self.finish_response(data)  # data is the complete response
        elif event == "error":
            self.display_error(data)  # data is the error message or dict
        elif event == "clear_requested":
            self.display_message(
                Text("🎮 Chat history cleared.", style=RICH_STYLE_YELLOW_BOLD)
            )
        elif event == "exit_requested":
            self.display_message(
                Text("🎮 Ending chat session. Goodbye!", style=RICH_STYLE_YELLOW_BOLD)
            )
            sys.exit(0)
        elif event == "copy_requested":
            self.copy_to_clipboard(data)  # data is the text to copy
        elif event == "debug_requested":
            self.display_debug_info(data)  # data is the debug information
        elif event == "think_budget_set":
            thinking_text = Text("Thinking budget set to ", style=RICH_STYLE_YELLOW)
            thinking_text.append(f"{data} tokens.")
            self.display_message(thinking_text)
        elif event == "models_listed":
            self.display_models(data)  # data is dict of models by provider
        elif event == "model_changed":
            model_text = Text("Switched to ", style=RICH_STYLE_YELLOW)
            model_text.append(f"{data['name']} ({data['id']})")
            self.display_message(model_text)
        elif event == "agents_listed":
            self.display_agents(data)  # data is dict of agent info
        elif event == "agent_changed":
            agent_text = Text("Switched to ", style=RICH_STYLE_YELLOW)
            agent_text.append(f"{data} agent")
            self.display_message(agent_text)
        elif event == "system_message":
            self.display_message(data)
        elif event == "mcp_prompt":
            self.display_mcp_prompt_confirmation(data)
        elif event == "agent_changed_by_transfer":
            transfer_text = Text("Transfered to ", style=RICH_STYLE_YELLOW)
            transfer_text.append(
                f"{data['agent_name'] if 'agent_name' in data else 'other'} agent"
            )
            self.display_message(transfer_text)
        elif event == "agent_continue":
            self.display_message(
                Text(f"\n🤖 {data.upper()}:", style=RICH_STYLE_GREEN_BOLD)
            )
        elif event == "jump_performed":
            jump_text = Text(
                f"🕰️ Jumping to turn {data['turn_number']}...\n",
                style=RICH_STYLE_YELLOW_BOLD,
            )
            preview_text = Text("Conversation rewound to: ", style=RICH_STYLE_YELLOW)
            preview_text.append(data["preview"])

            self.display_message(jump_text)
            self.display_message(preview_text)
        elif event == "thinking_completed":
            self.display_divider()
        elif event == "file_processed":
            self.stop_loading_animation()  # Stop loading on first chunk
            file_text = Text("Processed file: ", style=RICH_STYLE_YELLOW)
            file_text.append(data["file_path"])
            self.display_message(file_text)
        elif event == "consolidation_completed":
            self.display_consolidation_result(data)
        elif event == "conversations_listed":
            self.display_conversations(data)  # data is list of conversation metadata
        elif event == "conversation_loaded":
            loaded_text = Text("Loaded conversation: ", style=RICH_STYLE_YELLOW)
            loaded_text.append(data.get("id", "N/A"))
            self.display_message(loaded_text)
        elif event == "conversation_saved":
            logger.info(f"Conversation saved: {data.get('id', 'N/A')}")
            # self.display_message(
            #     f"{YELLOW}Conversation saved: {data.get('id', 'N/A')}{RESET}"
            # )
        elif event == "clear_requested":
            self.session_cost = 0.0
        elif event == "update_token_usage":
            self._calculate_token_usage(data["input_tokens"], data["output_tokens"])

    def display_thinking_started(self, agent_name: str):
        """Display the start of the thinking process."""
        self.console.print(
            Text(
                f"\n💭 {agent_name.upper()}'s thinking process:",
                style=RICH_STYLE_YELLOW,
            )
        )

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
        fun_word = random.choice(fun_words)

        with Live(
            "", console=self.console, auto_refresh=True, refresh_per_second=10
        ) as live:
            while not stop_event.is_set():
                live.update(f"{fun_word} {next(spinner)}")
                time.sleep(0.1)  # Control animation speed

    def start_loading_animation(self):
        """Start the loading animation."""
        if self._loading_thread and self._loading_thread.is_alive():
            return  # Already running

        self._loading_stop_event = threading.Event()
        self._loading_thread = threading.Thread(
            target=self._loading_animation, args=(self._loading_stop_event,)
        )
        self._loading_thread.daemon = True
        self._loading_thread.start()

    def stop_loading_animation(self):
        """Stop the loading animation."""
        if self._loading_stop_event:
            if self.console._live:
                self.console._live.update("")
                self.console._live.stop()
            self._loading_stop_event.set()
        if self._loading_thread and self._loading_thread.is_alive():
            self._loading_thread.join(timeout=0.5)
        self._loading_stop_event = None
        self._loading_thread = None

    def display_thinking_chunk(self, chunk: str):
        """Display a chunk of the thinking process."""
        self.console.print(Text(chunk, style=RICH_STYLE_GRAY), end="", soft_wrap=True)

    def update_live_display(self, chunk: str):
        """Update the live display with a new chunk of the response."""
        if not self.live:
            self.start_streaming_response(self.message_handler.agent.name)

        updated_text = chunk

        self._live_text_data = updated_text

        # Only show the last part that fits in the console
        lines = updated_text.split("\n")
        height_limit = (
            self.console.size.height - 10
        )  # leave some space for other elements
        if len(lines) > height_limit:
            lines = lines[-height_limit:]

        if self.live:
            self.live.update(Markdown("\n".join(lines), code_theme=CODE_THEME))

    def display_tool_use(self, tool_use: Dict):
        """Display information about a tool being used."""
        self.finish_live_update()

        # Tool icons mapping
        tool_icons = {
            "web_search": "🔍",
            "fetch_webpage": "🌐",
            "transfer": "↗️",
            "adapt": "🧠",
            "retrieve_memory": "💭",
            "forget_memory_topic": "🗑️",
            "analyze_repo": "📂",
            "read_file": "📄",
        }

        # Get tool icon or default
        tool_icon = tool_icons.get(tool_use["name"], "🔧")

        # Display tool header with better formatting
        header = Text(f"\n┌───── {tool_icon} Tool: ", style=RICH_STYLE_YELLOW)
        header.append(tool_use["name"], style=RICH_STYLE_YELLOW_BOLD)
        header.append(" ─────", style=RICH_STYLE_YELLOW)
        self.console.print(header)

        # Format tool input parameters
        if isinstance(tool_use.get("input"), dict):
            self.console.print(Text("│ Parameters:", style=RICH_STYLE_YELLOW))
            for key, value in tool_use["input"].items():
                # Format value based on type
                if isinstance(value, dict) or isinstance(value, list):
                    import json

                    formatted_value = json.dumps(value, indent=2)
                    # Add indentation to all lines after the first
                    formatted_lines = formatted_value.split("\n")
                    param_text = Text("│ • ", style=RICH_STYLE_YELLOW)
                    param_text.append(key, style=RICH_STYLE_BLUE)
                    param_text.append(": " + formatted_lines[0])
                    self.console.print(param_text)

                    for line in formatted_lines[1:]:
                        indent_text = Text("│     ", style=RICH_STYLE_YELLOW)
                        indent_text.append(line)
                        self.console.print(indent_text)
                else:
                    param_text = Text("│ • ", style=RICH_STYLE_YELLOW)
                    param_text.append(key, style=RICH_STYLE_BLUE)
                    param_text.append(f": {value}")
                    self.console.print(param_text)
        else:
            input_text = Text("│ Input: ", style=RICH_STYLE_YELLOW)
            input_text.append(str(tool_use.get("input", "")))
            self.console.print(input_text)

        self.console.print(Text("└", style=RICH_STYLE_YELLOW))

    def display_tool_result(self, data: Dict):
        """Display the result of a tool execution."""
        tool_use = data["tool_use"]
        tool_result = data["tool_result"]

        # Tool icons mapping
        tool_icons = {
            "web_search": "🔍",
            "fetch_webpage": "🌐",
            "transfer": "↗️",
            "adapt": "🧠",
            "retrieve_memory": "💭",
            "forget_memory_topic": "🗑️",
            "analyze_repo": "📂",
            "read_file": "📄",
        }

        # Get tool icon or default
        tool_icon = tool_icons.get(tool_use["name"], "🔧")

        # Display tool result with better formatting
        header = Text(f"\n┌───── {tool_icon} Tool Result: ", style=RICH_STYLE_GREEN)
        header.append(tool_use["name"], style=RICH_STYLE_GREEN_BOLD)
        header.append(" ─────", style=RICH_STYLE_GREEN)
        self.console.print(header)

        # Format the result based on type
        result_str = str(tool_result)
        # If result is very long, try to format it
        if len(result_str) > 500:
            result_line = Text("│ ", style=RICH_STYLE_GREEN)
            result_line.append(result_str[:500] + "...")
            self.console.print(result_line)

            truncated_line = Text("│ ", style=RICH_STYLE_GREEN)
            truncated_line.append(
                f"(Output truncated, total length: {len(result_str)} characters)"
            )
            self.console.print(truncated_line)
        else:
            # Split by lines to add prefixes
            for line in result_str.split("\n"):
                result_line = Text("│ ", style=RICH_STYLE_GREEN)
                result_line.append(line)
                self.console.print(result_line)

        self.console.print(Text("└", style=RICH_STYLE_GREEN))

    def display_tool_error(self, data: Dict):
        """Display an error that occurred during tool execution."""
        self.finish_live_update()
        tool_use = data["tool_use"]
        error = data["error"]

        # Tool icons mapping
        tool_icons = {
            "web_search": "🔍",
            "fetch_webpage": "🌐",
            "transfer": "↗️",
            "adapt": "🧠",
            "retrieve_memory": "💭",
            "forget_memory_topic": "🗑️",
            "analyze_repo": "📂",
            "read_file": "📄",
        }

        # Get tool icon or default
        tool_icon = tool_icons.get(tool_use["name"], "🔧")

        # Display tool error with better formatting
        header = Text(f"\n┌───── {tool_icon} Tool Error: ", style=RICH_STYLE_RED)
        header.append(tool_use["name"], style=RICH_STYLE_RED_BOLD)
        header.append(" ─────", style=RICH_STYLE_RED)
        self.console.print(header)

        error_line = Text("│ ", style=RICH_STYLE_RED)
        error_line.append(str(error))
        self.console.print(error_line)

        self.console.print(Text("└", style=RICH_STYLE_RED))

    def display_tool_confirmation_request(self, tool_info):
        """Display tool confirmation request and get user response."""
        self.finish_live_update()

        tool_use = tool_info.copy()
        confirmation_id = tool_use.pop("confirmation_id")

        self.console.print(
            Text(
                "\n🔧 Tool execution requires your permission:", style=RICH_STYLE_YELLOW
            )
        )
        tool_name = Text("Tool: ", style=RICH_STYLE_YELLOW)
        tool_name.append(tool_use["name"])
        self.console.print(tool_name)

        # Display tool parameters
        if isinstance(tool_use["input"], dict):
            self.console.print(Text("Parameters:", style=RICH_STYLE_YELLOW))
            for key, value in tool_use["input"].items():
                param_text = Text(f"  - {key}: ", style=RICH_STYLE_YELLOW)
                param_text.append(str(value))
                self.console.print(param_text)
        else:
            input_text = Text("Input: ", style=RICH_STYLE_YELLOW)
            input_text.append(str(tool_use["input"]))
            self.console.print(input_text)

        # Get user response
        self._stop_input_thread()
        while True:
            # Use Rich to print the prompt but still need to use input() for user interaction
            self.console.print(
                Text(
                    "\nAllow this tool to run? [y]es/[n]o/[a]lways in this session/[f]orever (this and future sessions): ",
                    style=RICH_STYLE_YELLOW,
                ),
                end="",
            )
            try:
                response = input().lower()
            except KeyboardInterrupt:
                response = "no"

            if response in ["y", "yes"]:
                self.message_handler.resolve_tool_confirmation(
                    confirmation_id, {"action": "approve"}
                )
                break
            elif response in ["n", "no"]:
                self.message_handler.resolve_tool_confirmation(
                    confirmation_id, {"action": "deny"}
                )
                break
            elif response in ["a", "all"]:
                self.message_handler.resolve_tool_confirmation(
                    confirmation_id, {"action": "approve_all"}
                )
                approved_text = Text(
                    "✓ Approved all future calls to '", style=RICH_STYLE_YELLOW
                )
                approved_text.append(tool_use["name"])
                approved_text.append("' for this session.")
                self.console.print(approved_text)
                break
            elif response in ["f", "forever"]:
                from AgentCrew.modules.config import ConfigManagement
                config_manager = ConfigManagement()
                config_manager.write_auto_approval_tools(tool_use["name"], add=True)

                self.message_handler.resolve_tool_confirmation(
                    confirmation_id, {"action": "approve"}
                )
                saved_text = Text(
                    "✓ Tool '", style=RICH_STYLE_YELLOW
                )
                saved_text.append(tool_use["name"])
                saved_text.append("' will be auto-approved forever.")
                self.console.print(saved_text)
                break
            else:
                self.console.print(
                    Text("Please enter 'y', 'n', 'a', or 'f'.", style=RICH_STYLE_YELLOW)
                )
        self._start_input_thread()

    def display_tool_denied(self, data):
        """Display information about a denied tool execution."""
        tool_use = data["tool_use"]
        denied_text = Text("\n❌ Tool execution denied: ", style=RICH_STYLE_RED)
        denied_text.append(tool_use["name"])
        self.console.print(denied_text)

    def display_mcp_prompt_confirmation(self, prompt_data):
        """Display MCP prompt confirmation request and get user response."""
        self.finish_live_update()

        self.console.print(
            Text("\n🤖 MCP Tool wants to execute a prompt:", style=RICH_STYLE_YELLOW)
        )

        # Display the prompt content
        if isinstance(prompt_data, dict):
            if "name" in prompt_data:
                prompt_name = Text("Prompt: ", style=RICH_STYLE_YELLOW)
                prompt_name.append(prompt_data["name"])
                self.console.print(prompt_name)

            if "content" in prompt_data:
                self.console.print(Text("Content:", style=RICH_STYLE_YELLOW))
                # Display content with proper formatting
                content = str(prompt_data["content"])
                if len(content) > 1000:
                    self.console.print(f"  {content[:1000]}...")
                    self.console.print(
                        Text(
                            f"  (Content truncated, total length: {len(content)} characters)",
                            style=RICH_STYLE_GRAY,
                        )
                    )
                else:
                    self.console.print(f"  {content}")

        # Get user response
        self._stop_input_thread()
        while True:
            self.console.print(
                Text(
                    "\nAllow this prompt to be executed? [y]es/[n]o: ",
                    style=RICH_STYLE_YELLOW,
                ),
                end="",
            )
            response = input().lower()

            if response in ["y", "yes"]:
                # User approved, put the prompt data in the input queue
                self.console.print(
                    Text(
                        "✓ MCP prompt approved and queued for execution.",
                        style=RICH_STYLE_GREEN,
                    )
                )

                self._input_queue.put(prompt_data["content"])
                break
            elif response in ["n", "no"]:
                # User denied, don't queue the prompt
                self.console.print(
                    Text("❌ MCP prompt execution denied.", style=RICH_STYLE_RED)
                )
                break
            else:
                self.console.print(
                    Text(
                        "Please enter 'y' for yes or 'n' for no.",
                        style=RICH_STYLE_YELLOW,
                    )
                )

        self._start_input_thread()

    def finish_live_update(self):
        """stop the live update display."""
        if self.live:
            self.console.print(self.live.get_renderable())
            self.live.update("")
            self.live.stop()
            self.live = None

    def finish_response(self, response: str):
        """Finalize and display the complete response."""
        if self.live:
            self.live.update("")
            self.live.stop()
            self.live = None

        # Replace \n with two spaces followed by \n for proper Markdown line breaks
        markdown_formatted_response = response.replace("\n", "  \n")
        self.console.print(Markdown(markdown_formatted_response, code_theme=CODE_THEME))

        # Store the latest response
        self.latest_assistant_response = response

    def display_error(self, error):
        """Display an error message."""
        self.stop_loading_animation()  # Stop loading on error
        if isinstance(error, dict):
            error_text = Text("\n❌ Error: ", style=RICH_STYLE_RED)
            error_text.append(error["message"])
            self.console.print(error_text)
            if "traceback" in error:
                self.console.print(Text(error["traceback"], style=RICH_STYLE_GRAY))
        else:
            error_text = Text("\n❌ Error: ", style=RICH_STYLE_RED)
            error_text.append(str(error))
            self.console.print(error_text)
        if self.live:
            self.live.update("")
            self.live.stop()
            self.live = None

    def display_message(self, message: Text):
        """Display a generic message."""
        # Check if message contains ANSI color codes and convert to Rich styling
        self.console.print(message)

    def display_divider(self):
        """Display a divider line."""
        pass
        # divider = "─" * self.console.size.width
        # self.console.print(divider, style=RICH_STYLE_GRAY)

    def copy_to_clipboard(self, text: str):
        """Copy text to clipboard and show confirmation."""
        if text:
            pyperclip.copy(text)
            self.console.print(
                Text("\n✓ Text copied to clipboard!", style=RICH_STYLE_YELLOW)
            )
        else:
            self.console.print(Text("\n! No text to copy.", style=RICH_STYLE_YELLOW))

    def display_debug_info(self, debug_info):
        """Display debug information."""
        import json

        self.console.print(Text("Current messages:", style=RICH_STYLE_YELLOW))
        try:
            self.console.print(json.dumps(debug_info, indent=2))
        except Exception:
            self.console.print(debug_info)

    def display_models(self, models_by_provider: Dict):
        """Display available models grouped by provider."""
        self.console.print(Text("Available models:", style=RICH_STYLE_YELLOW))
        for provider, models in models_by_provider.items():
            self.console.print(
                Text(f"\n{provider.capitalize()} models:", style=RICH_STYLE_YELLOW)
            )
            for model in models:
                current = " (current)" if model["current"] else ""
                self.console.print(f"  - {model['id']}: {model['name']}{current}")
                self.console.print(f"    {model['description']}")
                self.console.print(
                    f"    Capabilities: {', '.join(model['capabilities'])}"
                )

    def display_agents(self, agents_info: Dict):
        """Display available agents."""
        self.console.print(
            Text(f"Current agent: {agents_info['current']}", style=RICH_STYLE_YELLOW)
        )
        self.console.print(Text("Available agents:", style=RICH_STYLE_YELLOW))

        for agent_name, agent_data in agents_info["available"].items():
            current = " (current)" if agent_data["current"] else ""
            self.console.print(
                f"  - {agent_name}{current}: {agent_data['description']}"
            )

    def display_conversations(self, conversations: List[Dict[str, Any]]):
        """Display available conversations."""
        if not conversations:
            self.console.print(
                Text("No saved conversations found.", style=RICH_STYLE_YELLOW)
            )
            return

        self.console.print(Text("Available conversations:", style=RICH_STYLE_YELLOW))
        for i, convo in enumerate(conversations[:30], 1):
            # Format timestamp for better readability
            timestamp = convo.get("timestamp", "Unknown")
            if isinstance(timestamp, (int, float)):
                from datetime import datetime

                timestamp = datetime.fromtimestamp(timestamp).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            title = convo.get("title", "Untitled")
            convo_id = convo.get("id", "unknown")

            # Display conversation with index for easy selection
            self.console.print(f"  {i}. {title} [{convo_id}]")
            self.console.print(f"     Created: {timestamp}")

            # Show a preview if available
            if "preview" in convo:
                self.console.print(f"     Preview: {convo['preview']}")
            self.console.print("")

    def handle_load_conversation(self, load_arg: str):
        """
        Handle loading a conversation by number or ID.

        Args:
            load_arg: Either a conversation number (from the list) or a conversation ID
        """
        # First check if we have a list of conversations cached
        if not hasattr(self, "_cached_conversations"):
            # If not, get the list first
            self._cached_conversations = self.message_handler.list_conversations()

        try:
            # Check if the argument is a number (index in the list)
            if load_arg.isdigit():
                index = int(load_arg) - 1  # Convert to 0-based index
                if 0 <= index < len(self._cached_conversations):
                    convo_id = self._cached_conversations[index].get("id")
                    if convo_id:
                        self.console.print(
                            Text(
                                f"Loading conversation #{load_arg}...",
                                style=RICH_STYLE_YELLOW,
                            )
                        )
                        messages = self.message_handler.load_conversation(convo_id)
                        if messages:
                            self.display_loaded_conversation(messages)
                        return
                self.console.print(
                    Text(
                        "Invalid conversation number. Use '/list' to see available conversations.",
                        style=RICH_STYLE_RED,
                    )
                )
            else:
                # Assume it's a conversation ID
                self.console.print(
                    Text(
                        f"Loading conversation with ID: {load_arg}...",
                        style=RICH_STYLE_YELLOW,
                    )
                )
                messages = self.message_handler.load_conversation(load_arg)
                if messages:
                    self.display_loaded_conversation(messages)
        except Exception as e:
            self.console.print(
                Text(f"Error loading conversation: {str(e)}", style=RICH_STYLE_RED)
            )

    def display_consolidation_result(self, result: Dict[str, Any]):
        """
        Display information about a consolidation operation.

        Args:
            result: Dictionary containing consolidation results
        """
        self.console.print(
            Text("\n🔄 Conversation Consolidated:", style=RICH_STYLE_YELLOW)
        )
        self.console.print(f"  • {result['messages_consolidated']} messages summarized")
        self.console.print(
            f"  • {result['messages_preserved']} recent messages preserved"
        )
        self.console.print(
            f"  • ~{result['original_token_count'] - result['consolidated_token_count']} tokens saved"
        )
        self.display_loaded_conversation(self.message_handler.streamline_messages)

    def display_loaded_conversation(self, messages):
        """Display all messages from a loaded conversation.

        Args:
            messages: List of message dictionaries from the loaded conversation
        """
        self.console.print(
            Text("\nDisplaying conversation history:", style=RICH_STYLE_YELLOW)
        )
        self.display_divider()

        last_consolidated_idx = 0

        for i, msg in reversed(list(enumerate(messages))):
            if msg.get("role") == "consolidated":
                last_consolidated_idx = i
                break

        # Display each message in the conversation
        for msg in messages[last_consolidated_idx:]:
            role = msg.get("role")
            if role == "user":
                self.console.print(Text("\n👤 YOU:", style=RICH_STYLE_BLUE_BOLD))
                content = self._extract_message_content(msg)
                self.console.print(content)
                self.display_divider()
            elif role == "assistant":
                agent_name = self.message_handler.agent.name
                self.console.print(
                    Text(f"\n🤖 {agent_name.upper()}:", style=RICH_STYLE_GREEN_BOLD)
                )
                content = self._extract_message_content(msg)
                # Format as markdown for better display
                self.console.print(Markdown(content, code_theme=CODE_THEME))
                self.display_divider()
                if "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        self.display_tool_use(tool_call)
                self.display_divider()
            elif role == "consolidated":
                self.console.print(
                    Text("\n📝 CONVERSATION SUMMARY:", style=RICH_STYLE_YELLOW)
                )
                content = self._extract_message_content(msg)

                # Display metadata if available
                metadata = msg.get("metadata", {})
                if metadata:
                    consolidated_count = metadata.get(
                        "messages_consolidated", "unknown"
                    )
                    token_savings = metadata.get(
                        "original_token_count", 0
                    ) - metadata.get("consolidated_token_count", 0)
                    self.console.print(
                        Text(
                            f"({consolidated_count} messages consolidated, ~{token_savings} tokens saved)",
                            style=RICH_STYLE_YELLOW,
                        )
                    )

                # Format the summary with markdown
                self.console.print(Markdown(content, code_theme=CODE_THEME))
                self.display_divider()

        self.console.print(
            Text("End of conversation history\n", style=RICH_STYLE_YELLOW)
        )

    def _extract_message_content(self, message):
        """Extract the content from a message, handling different formats.

        Args:
            message: A message dictionary

        Returns:
            The extracted content as a string
        """
        content = message.get("content", "")

        # Handle different content structures
        if isinstance(content, str):
            pass
        elif isinstance(content, list) and content:
            # For content in the format of a list of content parts
            result = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        result.append(item.get("text", ""))
                    # Handle other content types if needed
            return "\n".join(result)

        content = re.sub(
            r"(?:```(?:json)?)?\s*<user_context_summary>.*?</user_context_summary>\s*(?:```)?",
            "",
            str(content),
            flags=re.DOTALL | re.IGNORECASE,
        )
        return str(content)

    def _input_thread_worker(self):
        """Worker thread for handling user input."""
        while not self._input_stop_event.is_set():
            try:
                session = PromptSession(
                    key_bindings=self.kb,
                    completer=ChatCompleter(self.message_handler),
                )
                self._current_prompt_session = session

                # Create a dynamic prompt that includes agent and model info using HTML formatting
                prompt_text = HTML("<ansiblue>👤 YOU:</ansiblue> ")

                user_input = session.prompt(prompt_text)

                # Reset history position after submission
                self.message_handler.history_manager.reset_position()

                # Put the input in the queue
                self._input_queue.put(user_input)

            except KeyboardInterrupt:
                # Handle Ctrl+C in input thread
                current_time = time.time()
                if (
                    hasattr(self, "_last_ctrl_c_time")
                    and current_time - self._last_ctrl_c_time < 2
                ):
                    self._input_queue.put("__EXIT__")
                    break
                else:
                    self._last_ctrl_c_time = current_time
                    self._input_queue.put("__INTERRUPT__")
                    continue
            except Exception as e:
                self._input_queue.put(f"__ERROR__:{str(e)}")
                break

    def _stop_input_thread(self):
        """Stop the input thread cleanly."""
        if self._input_thread and self._input_thread.is_alive():
            # Don't try to join if we're in the same thread
            if threading.current_thread() == self._input_thread:
                # We're in the input thread, just set the stop event
                self._input_stop_event.set()
                return

            self._input_stop_event.set()
            if self._current_prompt_session:
                # Try to interrupt the current prompt session
                try:
                    if (
                        hasattr(self._current_prompt_session, "app")
                        and self._current_prompt_session.app
                    ):
                        self._current_prompt_session.app.exit()
                except Exception:
                    pass
            self._input_thread.join(timeout=1.0)

    def _handle_keyboard_interrupt(self):
        """Handle Ctrl+C pressed during streaming or other operations."""
        self.stop_loading_animation()
        self.message_handler.stop_streaming = True

        current_time = time.time()
        if (
            hasattr(self, "_last_ctrl_c_time")
            and current_time - self._last_ctrl_c_time < 2
        ):
            self.console.print(
                Text(
                    "\n🎮 Confirmed exit. Goodbye!",
                    style=RICH_STYLE_YELLOW_BOLD,
                )
            )
            self._stop_input_thread()

            sys.exit(0)
        else:
            self._last_ctrl_c_time = current_time
            self.console.print(
                Text(
                    "\n🎮 Chat interrupted. Press Ctrl+C again within 2 seconds to exit.",
                    style=RICH_STYLE_YELLOW_BOLD,
                )
            )

    def _start_input_thread(self):
        if self._input_thread is None or not self._input_thread.is_alive():
            self._input_stop_event.clear()
            self._input_thread = Thread(target=self._input_thread_worker, daemon=True)
            self._input_thread.start()

    def get_user_input(self):
        """
        Get multiline input from the user with support for command history.
        Now runs in a separate thread to allow events to display during input.

        Args:
            conversation_turns: Optional list of conversation turns for completions.

        Returns:
            The user input as a string.
        """

        # Start input thread if not already running
        if self._input_thread is None or not self._input_thread.is_alive():
            title = Text(f"\n[{self.message_handler.agent.name}", style=RICH_STYLE_RED)
            title.append(":")
            title.append(
                f"{self.message_handler.agent.get_model()}]",
                style=RICH_STYLE_BLUE,
            )
            title.append(
                "\n(Press Enter for new line, Ctrl+S/Alt+Enter to submit, Up/Down for history)\n",
                style=RICH_STYLE_YELLOW,
            )
            self.console.print(title)
            self._start_input_thread()
        else:
            self._print_prompt_prefix()

        # Wait for input while allowing events to be processed
        while True:
            try:
                # Check for input with a short timeout to allow event processing
                user_input = self._input_queue.get(timeout=0.2)

                # Add None check here
                if user_input is None:
                    continue

                if user_input == "__EXIT__":
                    self.console.print(
                        Text(
                            "\n🎮 Confirmed exit. Goodbye!",
                            style=RICH_STYLE_YELLOW_BOLD,
                        )
                    )
                    self._stop_input_thread()
                    sys.exit(0)
                elif user_input == "__INTERRUPT__":
                    self.console.print(
                        Text(
                            "\n🎮 Chat interrupted. Press Ctrl+C again within 2 seconds to exit.",
                            style=RICH_STYLE_YELLOW_BOLD,
                        )
                    )
                    return ""
                elif user_input.startswith("__ERROR__:"):
                    error_msg = user_input[10:]  # Remove "__ERROR__:" prefix
                    self.console.print(
                        Text(f"\nInput error: {error_msg}", style=RICH_STYLE_RED)
                    )
                    return ""
                else:
                    self.display_divider()
                    return user_input

            except queue.Empty:
                # No input yet, continue waiting
                continue
            except KeyboardInterrupt:
                # Handle KeyboardInterrupt from the prompt session exit
                self.console.print(
                    Text(
                        "\n🎮 Confirmed exit. Goodbye!",
                        style=RICH_STYLE_YELLOW_BOLD,
                    )
                )
                sys.exit(0)

    def start_streaming_response(self, agent_name: str):
        """
        Start streaming the assistant's response.

        Args:
            agent_name: The name of the agent providing the response.
        """
        self.console.print(
            Text(f"🤖 {agent_name.upper()}:", style=RICH_STYLE_GREEN_BOLD)
        )
        self.live = Live(
            "", console=self.console, refresh_per_second=24, vertical_overflow="crop"
        )
        self.live.start()

    def _print_prompt_prefix(self):
        title = Text(f"\n[{self.message_handler.agent.name}", style=RICH_STYLE_RED)
        title.append(":")
        title.append(
            f"{self.message_handler.agent.get_model()}]",
            style=RICH_STYLE_BLUE,
        )
        title.append(
            "\n(Press Enter for new line, Ctrl+S/Alt+Enter to submit, Up/Down for history)\n",
            style=RICH_STYLE_YELLOW,
        )
        self.console.print(title)
        prompt = Text("👤 YOU: ", style=RICH_STYLE_BLUE_BOLD)
        self.console.print(prompt, end="")

    def _setup_key_bindings(self):
        """Set up key bindings for multiline input."""
        kb = KeyBindings()

        @kb.add(Keys.ControlS)
        @kb.add("escape", "enter")
        def _(event):
            """Submit on Ctrl+S."""
            if event.current_buffer.text.strip():
                event.current_buffer.validate_and_handle()

        @kb.add(Keys.Enter)
        def _(event):
            """Insert newline on Enter."""
            event.current_buffer.insert_text("\n")

        @kb.add("escape", "c")  # Alt+C
        def _(event):
            """Copy latest assistant response to clipboard."""
            self.copy_to_clipboard(self.latest_assistant_response)
            self._print_prompt_prefix()

        @kb.add(Keys.ControlC)
        def _(event):
            """Handle Ctrl+C with confirmation for exit."""
            current_time = time.time()
            if (
                hasattr(self, "_last_ctrl_c_time")
                and current_time - self._last_ctrl_c_time < 2
            ):
                self.console.print(
                    Text("\n🎮 Confirmed exit. Goodbye!", style=RICH_STYLE_YELLOW_BOLD)
                )
                # Don't try to join from within the same thread - just exit
                event.app.exit("__EXIT__")
            else:
                self._last_ctrl_c_time = current_time
                if self.live:
                    if self.message_handler.stream_generator:
                        try:
                            asyncio.run(self.message_handler.stream_generator.aclose())
                        except RuntimeError as e:
                            logger.warning(f"Error closing stream generator: {e}")
                        except Exception as e:
                            logger.warning(f"Exception closing stream generator: {e}")
                        finally:
                            self.message_handler.stop_streaming = True
                            self.message_handler.stream_generator = None

                self.console.print(
                    Text(
                        "\nPress Ctrl+C again within 2 seconds to exit.",
                        style=RICH_STYLE_YELLOW,
                    )
                )

                self._print_prompt_prefix()

        @kb.add(Keys.Up)
        def _(event):
            """Navigate to previous history entry."""
            buffer = event.current_buffer
            document = buffer.document

            # Check if cursor is at the first line's start
            cursor_position = document.cursor_position
            if document.cursor_position_row == 0 and cursor_position <= len(
                document.current_line
            ):
                # Get previous history entry
                prev_entry = self.message_handler.history_manager.get_previous()
                if prev_entry is not None:
                    # Replace current text with history entry
                    buffer.text = prev_entry
                    # Move cursor to end of text
                    buffer.cursor_position = len(prev_entry)
            else:
                # Regular up arrow behavior - move cursor up
                buffer.cursor_up()

        @kb.add(Keys.Down)
        def _(event):
            """Navigate to next history entry if cursor is at last line."""
            buffer = event.current_buffer
            document = buffer.document

            # Check if cursor is at the last line
            if document.cursor_position_row == document.line_count - 1:
                # Get next history entry
                next_entry = self.message_handler.history_manager.get_next()
                if next_entry is not None:
                    # Replace current text with history entry
                    buffer.text = next_entry
                    # Move cursor to end of text
                    buffer.cursor_position = len(next_entry)
            else:
                # Regular down arrow behavior - move cursor down
                buffer.cursor_down()

        return kb

    def print_welcome_message(self):
        """Print the welcome message for the chat."""
        # Get version information
        version = getattr(AgentCrew, "__version__", "Unknown")

        welcome_messages = Group(
            Text(
                "🎮 Welcome to AgentCrew v" + version + " interactive chat!",
                style=RICH_STYLE_YELLOW_BOLD,
            ),
            Text("Press Ctrl+C twice to exit.", style=RICH_STYLE_YELLOW),
            Text("Type 'exit' or 'quit' to end the session.", style=RICH_STYLE_YELLOW),
            Text(
                "Use '/file <file_path>' to include a file in your message.",
                style=RICH_STYLE_YELLOW,
            ),
            Text(
                "Use '/clear' to clear the conversation history.",
                style=RICH_STYLE_YELLOW,
            ),
            Text(
                "Use '/think <budget>' to enable Claude's thinking mode (min 1024 tokens).",
                style=RICH_STYLE_YELLOW,
            ),
            Text("Use '/think 0' to disable thinking mode.", style=RICH_STYLE_YELLOW),
            Text(
                "Use '/model [model_id]' to switch models or list available models.",
                style=RICH_STYLE_YELLOW,
            ),
            Text(
                "Use '/jump <turn_number>' to rewind the conversation to a previous turn.",
                style=RICH_STYLE_YELLOW,
            ),
            Text(
                "Use '/copy' to copy the latest assistant response to clipboard.",
                style=RICH_STYLE_YELLOW,
            ),
            Text(
                "Use '/agent [agent_name]' to switch agents or list available agents.",
                style=RICH_STYLE_YELLOW,
            ),
            Text("Use '/list' to list saved conversations.", style=RICH_STYLE_YELLOW),
            Text(
                "Use '/load <id>' or '/load <number>' to load a conversation.",
                style=RICH_STYLE_YELLOW,
            ),
            Text(
                "Use '/consolidate [count]' to summarize older messages (default: 10 recent messages preserved).",
                style=RICH_STYLE_YELLOW,
            ),
            Text(
                "Tool calls require confirmation before execution.",
                style=RICH_STYLE_YELLOW,
            ),
            Text(
                "Use 'y' to approve once, 'n' to deny, 'all' to approve future calls to the same tool.",
                style=RICH_STYLE_YELLOW,
            ),
        )

        self.console.print(Panel(welcome_messages))
        self.display_divider()

    def display_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        total_cost: float,
        session_cost: float,
    ):
        """Display token usage and cost information."""
        self.console.print("\n")
        self.display_divider()
        token_info = Text("📊 Token Usage: ", style=RICH_STYLE_YELLOW)
        token_info.append(
            f"Input: {input_tokens:,} | Output: {output_tokens:,} | ",
            style=RICH_STYLE_YELLOW,
        )
        token_info.append(
            f"Total: {input_tokens + output_tokens:,} | Cost: ${total_cost:.4f} | Total: {session_cost:.4f}",
            style=RICH_STYLE_YELLOW,
        )
        self.console.print(Panel(token_info))
        self.display_divider()

    def _calculate_token_usage(self, input_tokens: int, output_tokens: int):
        total_cost = self.message_handler.agent.calculate_usage_cost(
            input_tokens, output_tokens
        )
        self.session_cost += total_cost
        return total_cost

    def start(self):
        self.print_welcome_message()

        self.session_cost = 0.0
        self._cached_conversations = []  # Add this to cache conversation list

        try:
            while True:
                try:
                    # Get user input (now in separate thread)
                    self.stop_loading_animation()  # Stop if any
                    user_input = self.get_user_input()

                    # Handle list command directly
                    if user_input.strip() == "/list":
                        self._cached_conversations = (
                            self.message_handler.list_conversations()
                        )
                        self.display_conversations(self._cached_conversations)
                        continue

                    # Handle load command directly
                    if user_input.strip().startswith("/load "):
                        load_arg = user_input.strip()[
                            6:
                        ].strip()  # Extract argument after "/load "
                        if load_arg:
                            self.handle_load_conversation(load_arg)
                        else:
                            self.console.print(
                                Text(
                                    "Usage: /load <conversation_id> or /load <number>",
                                    style=RICH_STYLE_YELLOW,
                                )
                            )
                        continue

                    # Handle help command directly
                    if user_input.strip() == "/help":
                        self.print_welcome_message()
                        continue

                    # Start loading animation while waiting for response
                    if not user_input.startswith("/") or user_input.startswith(
                        "/file "
                    ):
                        self.start_loading_animation()

                    # Process user input and commands
                    should_exit, was_cleared = asyncio.run(
                        self.message_handler.process_user_input(user_input)
                    )

                    # Exit if requested
                    if should_exit:
                        break

                    # Skip to next iteration if messages were cleared
                    if was_cleared:
                        continue

                    # Skip to next iteration if no messages to process
                    if not self.message_handler.agent.history:
                        continue

                    # Get assistant response
                    assistant_response, input_tokens, output_tokens = asyncio.run(
                        self.message_handler.get_assistant_response()
                    )

                    # Ensure loading animation is stopped
                    self.stop_loading_animation()

                    total_cost = self._calculate_token_usage(
                        input_tokens, output_tokens
                    )

                    if assistant_response:
                        # Calculate and display token usage
                        self.display_token_usage(
                            input_tokens, output_tokens, total_cost, self.session_cost
                        )
                except KeyboardInterrupt:
                    self._handle_keyboard_interrupt()
                    continue  # Continue the loop instead of breaking
        finally:
            # Clean up input thread when exiting
            self._stop_input_thread()
