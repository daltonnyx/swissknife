from abc import abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import traceback
import os
import time
import json
from swissknife.modules.chat.history import ChatHistoryManager, ConversationTurn
from swissknife.modules.agents import AgentManager
from swissknife.modules.chat.file_handler import FileHandler
from swissknife.modules.llm.models import ModelRegistry
from swissknife.modules.llm.service_manager import ServiceManager
from swissknife.modules.llm.message import MessageTransformer
from swissknife.modules.memory import MemoryService, ContextPersistenceService


class Observable:
    """Base class for observables, implementing the observer pattern."""

    def __init__(self):
        self._observers: List["Observer"] = []

    def attach(self, observer: "Observer"):
        """Attaches an observer to the observable."""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: "Observer"):
        """Detaches an observer from the observable."""
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify(self, event: str, data: Any = None):
        """Notifies all attached observers of a new event."""
        for observer in self._observers:
            observer.listen(event, data)


class Observer:
    """Abstract base class for observers."""

    @abstractmethod
    def listen(self, event: str, data: Any = None):
        """Updates the observer with new data from the observable."""
        pass


class MessageHandler(Observable):
    """
    Handles message processing, interaction with the LLM service, and manages
    conversation history. Uses the Observer pattern to notify UI components
    about relevant events.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        context_persistent_service: ContextPersistenceService,
    ):
        """
        Initializes the MessageHandler.

        Args:
            llm_service: The LLM service to use for generating responses.
            memory_service: Optional memory service for storing conversations.
        """
        super().__init__()
        self.agent_manager = AgentManager.get_instance()
        self.llm = self.agent_manager.get_current_agent().llm
        self.agent_name = self.agent_manager.get_current_agent().name
        self.memory_service = memory_service
        self.persistent_service = context_persistent_service
        self.history_manager = ChatHistoryManager()
        self.latest_assistant_response = ""
        self.conversation_turns = []
        self.current_user_input = None
        self.current_user_input_idx = -1
        self.last_assisstant_response_idx = 0
        self.file_handler = FileHandler()
        self.messages = []  # Initialize empty messages list
        self.current_conversation_id = None  # ID for persistence
        self.start_new_conversation()  # Initialize first conversation

    def process_user_input(
        self,
        user_input: str,
        message_content: Optional[List[Dict]] = None,
        files: Optional[List[str]] = None,
    ) -> Tuple[bool, bool]:
        """
        Processes user input, handles commands, and updates message history.

        Args:
            user_input: The input string from the user.
            message_content: Optional content to prepend.
            files: Optional files to include.

        Returns:
            Tuple of (exit_flag, clear_flag)
        """
        # Handle exit command
        if self._handle_exit_command(user_input):
            self._notify("exit_requested")
            return True, False

        # Handle clear command
        if user_input.lower() == "/clear":
            # Now handled by start_new_conversation
            self.start_new_conversation()
            return False, True  # exit_flag, clear_flag

        # Handle copy command
        if user_input.lower() == "/copy":
            self._notify("copy_requested", self.latest_assistant_response)
            return False, True  # Skip to next iteration

        # Handle debug command
        if user_input.lower() == "/debug":
            self._notify("debug_requested", self.messages)
            return False, True

        # Handle think command
        if user_input.lower().startswith("/think "):
            try:
                budget = int(user_input[7:].strip())
                self.llm.set_think(budget)
                self._notify("think_budget_set", budget)
            except ValueError:
                self._notify("error", "Invalid budget value. Please provide a number.")
            return False, True

        # Handle jump command
        if user_input.lower().startswith("/jump "):
            jumped = self._handle_jump_command(user_input)
            if jumped:
                return False, True
            return False, True

        # Handle agent command
        if user_input.lower().startswith("/agent"):
            success, message = self._handle_agent_command(user_input)
            self._notify(
                "agent_command_result", {"success": success, "message": message}
            )
            return False, True

        # Handle model command
        if user_input.lower().startswith("/model"):
            return self._handle_model_command(user_input)

        # Store non-command messages in history
        if not user_input.startswith("/"):
            self.history_manager.add_entry(user_input)

        if len(self.messages) == 0:
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Use user_context_summary to tailor your response:
                            <user_context_summary>{self.persistent_service.get_user_context_json(2, 3)}</user_context_summary>""",
                        }
                    ],
                }
            )

        # Handle files that were loaded but not yet sent
        if files and not self.messages:
            combined_content = message_content.copy() if message_content else []
            combined_content.append({"type": "text", "text": user_input})
            self.messages.append({"role": "user", "content": combined_content})
            self.current_user_input = self.messages[-1]
            self.current_user_input_idx = len(self.messages) - 1
            self._notify(
                "user_message_created",
                {"message": self.messages[-1], "with_files": True},
            )
        # Handle file command
        elif user_input.startswith("/file "):
            file_path = user_input[6:].strip()
            file_path = os.path.expanduser(file_path)

            # Process file with the file handling service
            file_content = self.file_handler.process_file(file_path)
            # Fallback to llm handle
            if not file_content:
                file_content = self.llm.process_file_for_message(file_path)

            if file_content:
                self.messages.append({"role": "user", "content": [file_content]})
                self._notify(
                    "file_processed",
                    {"file_path": file_path, "message": self.messages[-1]},
                )
                return False, True
            else:
                self._notify(
                    "error",
                    f"Failed to process file {file_path} Or Model is not supported",
                )
                return False, True
        else:
            # Add regular text message
            self.messages.append(
                {"role": "user", "content": [{"type": "text", "text": user_input}]}
            )
            self.current_user_input = self.messages[-1]
            self.current_user_input_idx = len(self.messages) - 1
            self._notify(
                "user_message_created",
                {"message": self.messages[-1], "with_files": False},
            )

        return False, False

    def start_new_conversation(self):
        """Starts a new persistent conversation, clears history, and gets a new ID."""
        try:
            # Ensure the service instance is available
            if (
                not hasattr(self, "persistent_service")
                or self.persistent_service is None
            ):
                raise RuntimeError(
                    "ContextPersistenceService not initialized in MessageHandler."
                )

            self.current_conversation_id = self.persistent_service.start_conversation()
            self.messages = []  # Clear in-memory message list
            self.conversation_turns = []  # Clear jump history
            self.current_user_input = None
            self.current_user_input_idx = -1
            # Notify UI about the new conversation
            self._notify(
                "system_message",
                f"Started new conversation: {self.current_conversation_id}",
            )
            # Re-use existing signal to clear UI display, ensures UI is reset
            self._notify("clear_requested")
            print(
                f"INFO: Started new persistent conversation {self.current_conversation_id}"
            )
        except Exception as e:
            error_message = f"Failed to start new persistent conversation: {str(e)}"
            print(f"ERROR: {error_message}")
            self._notify("error", {"message": error_message})
            self.current_conversation_id = (
                None  # Ensure saving fails safely if start fails
            )

    def _handle_exit_command(self, user_input: str) -> bool:
        """Check if the user wants to exit the chat."""
        return user_input.lower() in ["exit", "quit"]

    def _handle_jump_command(self, command: str) -> bool:
        """Handle the /jump command to rewind conversation to a previous turn."""
        try:
            # Extract the turn number from the command
            parts = command.split()
            if len(parts) != 2:
                self._notify("error", "Usage: /jump <turn_number>")
                return False

            turn_number = int(parts[1])

            # Validate the turn number
            if turn_number < 1 or turn_number > len(self.conversation_turns):
                self._notify(
                    "error",
                    f"Invalid turn number. Available turns: 1-{len(self.conversation_turns)}",
                )
                return False

            # Get the selected turn
            selected_turn = self.conversation_turns[turn_number - 1]

            # Provide feedback via notification

            # Truncate messages to the index from the selected turn
            self.messages = self.messages[: selected_turn.message_index]
            self.conversation_turns = self.conversation_turns[: turn_number - 1]
            if self.current_conversation_id:
                self.persistent_service.append_conversation_messages(
                    self.current_conversation_id,
                    MessageTransformer.standardize_messages(
                        self.messages, self.llm.provider_name
                    ),
                    True,
                )

            self._notify(
                "jump_performed",
                {"turn_number": turn_number, "preview": selected_turn.get_preview(100)},
            )

            # Return success
            return True

        except ValueError:
            self._notify("error", "Invalid turn number. Please provide a number.")
            return False

    def _handle_model_command(self, command: str) -> Tuple[bool, bool]:
        """
        Handle the /model command to switch models or list available models.

        Args:
            command: The model command string

        Returns:
            Tuple of (exit_flag, clear_flag)
        """
        model_id = command[7:].strip()
        registry = ModelRegistry.get_instance()
        manager = ServiceManager.get_instance()

        # If no model ID is provided, list available models
        if not model_id:
            models_by_provider = {}
            for provider in ["claude", "openai", "groq", "google"]:
                models = registry.get_models_by_provider(provider)
                if models:
                    models_by_provider[provider] = []
                    for model in models:
                        current = (
                            registry.current_model
                            and registry.current_model.id == model.id
                        )
                        models_by_provider[provider].append(
                            {
                                "id": model.id,
                                "name": model.name,
                                "description": model.description,
                                "capabilities": model.capabilities,
                                "current": current,
                            }
                        )

            self._notify("models_listed", models_by_provider)
            return False, True

        # Try to switch to the specified model
        if registry.set_current_model(model_id):
            model = registry.get_current_model()
            if model:
                # Get the current provider
                current_provider = self.llm.provider_name

                # If we're switching providers, convert messages
                if current_provider != model.provider:
                    # Standardize messages from current provider
                    std_messages = MessageTransformer.standardize_messages(
                        self.messages, current_provider
                    )
                    # Convert to new provider format
                    self.messages = MessageTransformer.convert_messages(
                        std_messages, model.provider
                    )

                # Update the LLM service
                manager.set_model(model.provider, model.id)

                # Get the new LLM service
                new_llm_service = manager.get_service(model.provider)

                # Update the agent manager with the new LLM service
                self.agent_manager.update_llm_service(new_llm_service)

                # Update our reference to the LLM
                self.llm = self.agent_manager.get_current_agent().llm

                self._notify(
                    "model_changed",
                    {"id": model.id, "name": model.name, "provider": model.provider},
                )
            else:
                self._notify("error", "Failed to switch model.")
        else:
            self._notify("error", f"Unknown model: {model_id}")

        return False, True

    def _handle_agent_command(self, command: str) -> Tuple[bool, str]:
        """
        Handle the /agent command to switch agents or list available agents.

        Args:
            command: The agent command string

        Returns:
            Tuple of (success, message)
        """
        parts = command.split()

        # If no agent name is provided, list available agents
        if len(parts) == 1:
            available_agents = list(self.agent_manager.agents.keys())
            current_agent = self.agent_manager.get_current_agent()
            current_agent_name = current_agent.name if current_agent else "None"

            agents_info = {"current": current_agent_name, "available": {}}

            for agent_name, agent in self.agent_manager.agents.items():
                agents_info["available"][agent_name] = {
                    "description": agent.description,
                    "current": (current_agent and current_agent.name == agent_name),
                }

            self._notify("agents_listed", agents_info)
            return True, "Listed available agents"

        # If an agent name is provided, try to switch to that agent
        agent_name = parts[1]
        if self.agent_manager.select_agent(agent_name):
            # Update the LLM reference to the new agent's LLM
            self.llm = self.agent_manager.get_current_agent().llm
            self.agent_name = agent_name
            self._notify("agent_changed", agent_name)
            return True, f"Switched to {agent_name} agent"
        else:
            available_agents = ", ".join(self.agent_manager.agents.keys())
            self._notify(
                "error",
                f"Unknown agent: {agent_name}. Available agents: {available_agents}",
            )
            return (
                False,
                f"Unknown agent: {agent_name}. Available agents: {available_agents}",
            )

    def get_assistant_response(
        self, input_tokens=0, output_tokens=0
    ) -> Tuple[Optional[str], int, int]:
        """
        Stream the assistant's response and return the response and token usage.

        Returns:
            Tuple of (assistant_response, input_tokens, output_tokens)
        """
        assistant_response = ""
        tool_uses = []
        thinking_content = ""  # Reset thinking content for new response
        thinking_signature = ""  # Store the signature
        start_thinking = False
        end_thinking = False
        context_data_processed = False
        try:
            with self.llm.stream_assistant_response(self.messages) as stream:
                for chunk in stream:
                    # Process the chunk using the LLM service
                    (
                        assistant_response,
                        tool_uses,
                        chunk_input_tokens,
                        chunk_output_tokens,
                        chunk_text,
                        thinking_chunk,
                    ) = self.llm.process_stream_chunk(
                        chunk, assistant_response, tool_uses
                    )

                    clean_response = assistant_response
                    context_data, clean_response = self.llm.parse_user_context_summary(
                        assistant_response
                    )
                    if context_data and not context_data_processed:
                        # self._notify("debug_requested", context_data)
                        self.persistent_service.store_user_context(context_data)
                        context_data_processed = True
                        # self.messages.append(
                        #     self.llm.format_assistant_message(
                        #         f"""Need to tailor response bases on this <user_context_summary>{json.dumps(context_data)}</user_context_summary>"""
                        #     )
                        # )

                    # Update token counts
                    if chunk_input_tokens > 0:
                        input_tokens = chunk_input_tokens
                    if chunk_output_tokens > 0:
                        output_tokens = chunk_output_tokens

                    # Accumulate thinking content if available
                    if thinking_chunk:
                        think_text_chunk, signature = thinking_chunk

                        if not start_thinking:
                            # Notify about thinking process
                            self._notify("thinking_started", self.agent_name)
                            if not self.llm.is_stream:
                                # Delays it a bit when using without stream
                                time.sleep(0.5)
                            start_thinking = True
                        if think_text_chunk:
                            thinking_content += think_text_chunk
                            self._notify("thinking_chunk", think_text_chunk)
                        if signature:
                            thinking_signature += signature
                    if chunk_text:
                        # End thinking when chunk_text start
                        if not end_thinking and start_thinking:
                            self._notify("thinking_completed")
                            end_thinking = True
                        # Notify about response progress
                        if not self.llm.is_stream:
                            # Delays it a bit when using without stream
                            time.sleep(0.5)
                        self._notify("response_chunk", (chunk_text, clean_response))

            # Handle tool use if needed
            if tool_uses and len(tool_uses) > 0:
                # Add thinking content as a separate message if available
                thinking_data = (
                    (thinking_content, thinking_signature) if thinking_content else None
                )
                thinking_message = self.llm.format_thinking_message(thinking_data)
                if thinking_message:
                    self.messages.append(thinking_message)
                    self._notify("thinking_message_added", thinking_message)

                # Format assistant message with the response and tool uses
                assistant_message = self.llm.format_assistant_message(
                    assistant_response, tool_uses
                )
                self.messages.append(assistant_message)
                self._notify("assistant_message_added", assistant_message)

                # Process each tool use
                for tool_use in tool_uses:
                    self._notify("response_completed", assistant_response)
                    self._notify("tool_use", tool_use)

                    try:
                        if tool_use["name"] == "transfer":
                            self.agent_manager.get_current_agent().history = (
                                MessageTransformer.standardize_messages(
                                    self.messages, self.llm.provider_name
                                )
                            )
                        tool_result = self.llm.execute_tool(
                            tool_use["name"], tool_use["input"]
                        )
                        tool_result_message = self.llm.format_tool_result(
                            tool_use, tool_result
                        )
                        self.messages.append(tool_result_message)
                        self._notify(
                            "tool_result",
                            {
                                "tool_use": tool_use,
                                "tool_result": tool_result,
                                "message": tool_result_message,
                            },
                        )

                        # Update llm service when transfer agent
                        if tool_use["name"] == "transfer":
                            self.llm = self.agent_manager.get_current_agent().llm
                            self.agent_name = (
                                self.agent_manager.get_current_agent().name
                            )
                            if self.current_conversation_id:
                                self.persistent_service.append_conversation_messages(
                                    self.current_conversation_id,
                                    MessageTransformer.standardize_messages(
                                        self.messages[
                                            self.last_assisstant_response_idx :
                                        ],
                                        self.llm.provider_name,
                                    ),
                                )
                            self.messages = MessageTransformer.convert_messages(
                                self.agent_manager.get_current_agent().history,
                                self.llm.provider_name,
                            )

                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": tool_result}],
                                }
                            )
                            self.last_assisstant_response_idx = len(self.messages)

                            self._notify("agent_changed_by_transfer", self.agent_name)

                    except Exception as e:
                        error_message = self.llm.format_tool_result(
                            tool_use, str(e), is_error=True
                        )
                        self.messages.append(error_message)
                        self._notify(
                            "tool_error",
                            {
                                "tool_use": tool_use,
                                "error": str(e),
                                "message": error_message,
                            },
                        )

                return self.get_assistant_response(input_tokens, output_tokens)

            if thinking_content:
                # self._notify("thinking_completed")
                self._notify("agent_continue", self.agent_name)

            # Final assistant message
            self._notify("response_completed", assistant_response)

            # Store the latest response
            self.latest_assistant_response = assistant_response

            # Add assistant response to messages
            if assistant_response:
                self.messages.append(
                    self.llm.format_assistant_message(assistant_response)
                )

            if self.current_user_input and self.current_user_input_idx >= 0:
                if self.memory_service:
                    user_input = ""
                    user_message = self.current_user_input  # Get the user message
                    if (
                        isinstance(user_message["content"], list)
                        and len(user_message["content"]) > 0
                    ):
                        for content_item in user_message["content"]:
                            if content_item.get("type") == "text":
                                user_input += content_item.get("text", "")
                    elif isinstance(user_message["content"], str):
                        user_input = user_message["content"]

                    try:
                        self.memory_service.store_conversation(
                            user_input, assistant_response
                        )
                    except Exception as e:
                        self._notify(
                            "error", f"Failed to store conversation in memory: {str(e)}"
                        )
                # Store the conversation turn reference for /jump command
                turn = ConversationTurn(
                    self.current_user_input,  # User message for preview
                    self.current_user_input_idx,  # Index of the *start* of this turn's messages
                )
                self.conversation_turns.append(turn)
                self.current_user_input = None
                self.current_user_input_idx = -1

            # --- Start of Persistence Logic ---
            if self.current_conversation_id and self.last_assisstant_response_idx >= 0:
                try:
                    # Get all messages added since the user input for this turn
                    current_provider = self.llm.provider_name
                    messages_for_this_turn = MessageTransformer.standardize_messages(
                        self.messages[self.last_assisstant_response_idx :],
                        current_provider,
                    )
                    if (
                        messages_for_this_turn
                    ):  # Only save if there are messages for the turn
                        self.persistent_service.append_conversation_messages(
                            self.current_conversation_id, messages_for_this_turn
                        )
                        self._notify(
                            "conversation_saved", {"id": self.current_conversation_id}
                        )
                except Exception as e:
                    error_message = f"Failed to save conversation turn to {self.current_conversation_id}: {str(e)}"
                    print(f"ERROR: {error_message}")
                    self._notify("error", {"message": error_message})
            self.last_assisstant_response_idx = len(self.messages)
            # --- End of Persistence Logic ---

            return assistant_response, input_tokens, output_tokens

        except Exception as e:
            error_message = str(e)
            traceback_str = traceback.format_exc()
            self._notify(
                "error",
                {
                    "message": error_message,
                    "traceback": traceback_str,
                    "messages": self.messages,
                },
            )
            return None, 0, 0

    # --- Add these new methods ---
    def list_conversations(self) -> List[Dict[str, Any]]:
        """Lists available conversations from the persistence service."""
        try:
            return self.persistent_service.list_conversations()
        except Exception as e:
            print(f"Error listing conversations: {e}")
            self._notify("error", f"Failed to list conversations: {e}")
            return []

    def load_conversation(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """Loads a specific conversation history and sets it as active."""
        try:
            history = self.persistent_service.get_conversation_history(conversation_id)
            if history:
                self.current_conversation_id = conversation_id
                current_provider = self.llm.provider_name
                self.messages = MessageTransformer.convert_messages(
                    history, current_provider
                )
                for i, message in enumerate(self.messages):
                    role = message.get("role")
                    if role == "user":
                        content = message.get("content", "")
                        message_content = ""

                        # Handle different content structures (standardized format)
                        if isinstance(content, str):
                            message_content = content
                        elif isinstance(content, list) and content:
                            # Assuming the first item in the list contains the primary text
                            first_item = content[0]
                            if (
                                isinstance(first_item, dict)
                                and first_item.get("type") == "text"
                            ):
                                message_content = first_item.get("text", "")
                        if (
                            message_content
                            and not message_content.startswith(
                                "Use user_context_summary to tailor your response"
                            )
                            and not message_content.startswith("Content of ")
                        ):
                            turn = ConversationTurn(message_content, i)
                            self.conversation_turns.append(turn)

                print(f"Loaded conversation {conversation_id}")  # Optional: Debugging
                self._notify(
                    "conversation_loaded", {"id": conversation_id, "history": history}
                )
                return history
            else:
                self._notify(
                    "error", f"Conversation {conversation_id} not found or empty."
                )
                return []
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {e}")
            self._notify("error", f"Failed to load conversation {conversation_id}: {e}")
