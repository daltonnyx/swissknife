from typing import Tuple, Optional
import traceback
import time

from AgentCrew.modules import logger
from AgentCrew.modules.agents.base import MessageType
from AgentCrew.modules.chat.history import ChatHistoryManager
from AgentCrew.modules.agents import AgentManager
from AgentCrew.modules.chat.file_handler import FileHandler
from AgentCrew.modules.llm.message import MessageTransformer
from AgentCrew.modules.config import ConfigManagement
from AgentCrew.modules.memory import (
    BaseMemoryService,
    ContextPersistenceService,
)
from AgentCrew.modules.mcpclient import MCPSessionManager

from AgentCrew.modules.chat.message import (
    CommandProcessor,
    ToolManager,
    ConversationManager,
    Observable,
)


class MessageHandler(Observable):
    """
    Handles message processing, interaction with the LLM service, and manages
    conversation history. Uses the Observer pattern to notify UI components
    about relevant events.
    """

    def __init__(
        self,
        memory_service: BaseMemoryService,
        context_persistent_service: ContextPersistenceService,
    ):
        """
        Initializes the MessageHandler.

        Args:
            memory_service: Memory service for storing conversations.
            context_persistent_service: Service for persistent conversation storage.
        """
        super().__init__()
        self.agent_manager = AgentManager.get_instance()
        self.mcp_manager = MCPSessionManager.get_instance()
        self.agent = self.agent_manager.get_current_agent()
        self.memory_service = memory_service
        self.persistent_service = context_persistent_service
        self.history_manager = ChatHistoryManager()
        self.latest_assistant_response = ""
        self.conversation_turns = []
        self.current_user_input = None
        self.current_user_input_idx = -1
        self.last_assisstant_response_idx = -1
        self.file_handler: Optional[FileHandler] = None
        self.stop_streaming = False
        self.streamline_messages = []
        self.current_conversation_id: Optional[str] = None  # ID for persistence

        # Initialize components
        self.command_processor = CommandProcessor(self)
        self.tool_manager = ToolManager(self)
        self.conversation_manager = ConversationManager(self)

        self.conversation_manager.start_new_conversation()  # Initialize first conversation

    def _messages_append(self, message):
        """Append a message to the agent history and streamline messages."""
        self.agent.history.append(message)

        std_msg = MessageTransformer.standardize_messages(
            [message], self.agent.get_provider(), self.agent.name
        )
        self.streamline_messages.extend(std_msg)

    async def process_user_input(
        self,
        user_input: str,
    ) -> Tuple[bool, bool]:
        """
        Processes user input, handles commands, and updates message history.

        Args:
            user_input: The input string from the user.

        Returns:
            Tuple of (exit_flag, clear_flag)
        """
        self.history_manager.add_entry(user_input)

        # Process commands first
        command_result = await self.command_processor.process_command(user_input)
        if command_result.handled:
            return command_result.exit_flag, command_result.clear_flag

        # Handle regular user input (non-commands)
        # RAG base on user query
        # IMPORTANT: this actually add more problems than it solves, so it's disabled for now
        #         if await self.memory_service.need_generate_user_context(user_input):
        #             self._notify("user_context_request", None)
        #             self._messages_append(
        #                 {
        #                     "role": "user",
        #                     "content": [
        #                         {
        #                             "type": "text",
        #                             "text": f"""Memories related to the user request:
        # ---
        # {await self.memory_service.generate_user_context(user_input, self.agent.name)}
        # ---
        # INSTRUCTIONS:
        # 1. EXTRACT relevant facts and context from memories
        # 2. PRIORITIZE recent memories when information conflicts
        # 3. INCORPORATE memory insights to enhance your response
        # 4. PRESERVE the user's original intent - memories should supplement, not override
        # 5. RESPOND directly to the current request first and foremost
        #
        # IMPORTANT: Memory serves to enhance responses, never to reinterpret or redirect the user's explicit request.""",
        #                         }
        #                     ],
        #                 }
        #             )
        # Add regular text message
        self._messages_append(
            {"role": "user", "content": [{"type": "text", "text": user_input}]}
        )
        self.current_user_input = self.agent.history[-1]
        self.current_user_input_idx = len(self.streamline_messages) - 1
        self._notify(
            "user_message_created",
            {"message": self.agent.history[-1], "with_files": False},
        )

        return False, False

    def start_new_conversation(self):
        """Starts a new persistent conversation."""
        # Reset approved tools for the new conversation
        self.tool_manager.reset_approved_tools()
        self.conversation_manager.start_new_conversation()

    def resolve_tool_confirmation(self, confirmation_id, result):
        """
        Resolve a pending tool confirmation future with the user's decision.
        """
        self.tool_manager.resolve_tool_confirmation(confirmation_id, result)

    async def get_assistant_response(
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
        has_stop_interupted = False

        # Create a reference to the streaming generator
        self.stream_generator = None

        try:
            # Store the generator in a variable so we can properly close it if needed
            self.stream_generator = self.agent.process_messages()

            async for (
                assistant_response,
                chunk_text,
                thinking_chunk,
            ) in self.stream_generator:
                # Check if stop was requested
                if self.stop_streaming:
                    # Properly close the generator instead of breaking
                    self.stop_streaming = False  # Reset flag
                    has_stop_interupted = True
                    self._notify("streaming_stopped", assistant_response)
                    break

                # Accumulate thinking content if available
                if thinking_chunk:
                    think_text_chunk, signature = thinking_chunk

                    if not start_thinking:
                        # Notify about thinking process
                        self._notify("thinking_started", self.agent.name)
                        if not self.agent.is_streaming():
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
                        self._notify("thinking_completed", thinking_content)
                        end_thinking = True
                    # Notify about response progress
                    if not self.agent.is_streaming():
                        # Delays it a bit when using without stream
                        time.sleep(0.5)
                    self._notify("response_chunk", (chunk_text, assistant_response))

            tool_uses, input_tokens_in_turn, output_tokens_in_turn = (
                self.agent.get_process_result()
            )
            input_tokens += input_tokens_in_turn
            output_tokens += output_tokens_in_turn

            # Handle tool use if needed
            if not has_stop_interupted and tool_uses and len(tool_uses) > 0:
                # Add thinking content as a separate message if available
                thinking_data = (
                    (thinking_content, thinking_signature) if thinking_content else None
                )
                thinking_message = self.agent.format_message(
                    MessageType.Thinking, {"thinking": thinking_data}
                )
                if thinking_message:
                    self._messages_append(thinking_message)
                    self._notify("thinking_message_added", thinking_message)

                # Format assistant message with the response and tool uses
                tool_uses_without_transfer = [
                    t for t in tool_uses if t["name"] != "transfer"
                ]
                # only append message if there are tool uses other than transfer
                if len(tool_uses_without_transfer) > 0:
                    assistant_message = self.agent.format_message(
                        MessageType.Assistant,
                        {
                            "message": assistant_response,
                            "tool_uses": tool_uses_without_transfer,
                        },
                    )
                    self._messages_append(assistant_message)
                # ignore if message is empty
                elif assistant_response.strip():
                    assistant_message = self.agent.format_message(
                        MessageType.Assistant,
                        {
                            "message": assistant_response,
                        },
                    )
                    self._messages_append(assistant_message)
                self._notify("assistant_message_added", assistant_response)

                # This should allows YOLO can be configured on-the-fly without recalled to config too many times
                config_management = ConfigManagement()
                global_config = config_management.read_global_config_data()

                self.tool_manager.yolo_mode = global_config.get(
                    "global_settings", {}
                ).get("yolo_mode", False)

                # Process each tool use
                for tool_use in tool_uses:
                    await self.tool_manager.execute_tool(tool_use)

                self._notify(
                    "update_token_usage",
                    {"input_tokens": input_tokens, "output_tokens": output_tokens},
                )

                if has_stop_interupted:
                    # return as soon as possible
                    self._notify("response_completed", assistant_response)
                    return assistant_response, input_tokens, output_tokens

                return await self.get_assistant_response()

            self.stream_generator = None

            if thinking_content:
                self._notify("agent_continue", self.agent.name)

            # Add assistant response to messages
            if assistant_response.strip():
                # Store the latest response
                self.latest_assistant_response = assistant_response

                self._messages_append(
                    self.agent.format_message(
                        MessageType.Assistant, {"message": assistant_response}
                    )
                )
            # Final assistant message
            self._notify("response_completed", assistant_response)

            # --- Start of Persistence Logic ---
            if self.current_conversation_id and self.last_assisstant_response_idx >= 0:
                try:
                    # Get all messages added since the user input for this turn
                    current_provider = self.agent.get_provider()
                    messages_for_this_turn = MessageTransformer.standardize_messages(
                        self.agent.history[self.last_assisstant_response_idx :],
                        current_provider,
                        self.agent.name,
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
                    logger.error(f"ERROR: {error_message}")
                    self._notify("error", {"message": error_message})

            self.last_assisstant_response_idx = len(self.agent.history)
            # --- End of Persistence Logic ---

            if self.current_user_input and self.current_user_input_idx >= 0:
                self.conversation_manager.store_conversation_turn(
                    self.current_user_input, self.current_user_input_idx
                )
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
                        await self.memory_service.store_conversation(
                            user_input, assistant_response, self.agent.name
                        )
                    except Exception as e:
                        self._notify(
                            "error", f"Failed to store conversation in memory: {str(e)}"
                        )
                # Store the conversation turn reference for /jump command
                self.current_user_input = None
                self.current_user_input_idx = -1

            return assistant_response, input_tokens, output_tokens

        except GeneratorExit:
            return assistant_response, input_tokens, output_tokens
        except Exception as e:
            if self.current_user_input:
                self.conversation_manager.store_conversation_turn(
                    self.current_user_input, self.current_user_input_idx
                )
                self.current_user_input = None
                self.current_user_input_idx = -1
            error_message = str(e)
            traceback_str = traceback.format_exc()
            self._notify(
                "error",
                {
                    "message": error_message,
                    "traceback": traceback_str,
                    "messages": self.agent.history,
                },
            )
            return None, 0, 0

    # Delegate conversation management methods
    def list_conversations(self):
        """Lists available conversations from the persistence service."""
        return self.conversation_manager.list_conversations()

    def load_conversation(self, conversation_id: str):
        """Loads a specific conversation history and sets it as active."""
        # Reset approved tools for the loaded conversation
        self.tool_manager.reset_approved_tools()
        return self.conversation_manager.load_conversation(conversation_id)

    def delete_conversation_by_id(self, conversation_id: str) -> bool:
        """Deletes a conversation by its ID."""
        return self.conversation_manager.delete_conversation_by_id(conversation_id)
