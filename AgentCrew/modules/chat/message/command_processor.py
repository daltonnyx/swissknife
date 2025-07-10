from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import os

from AgentCrew.modules.agents.local_agent import LocalAgent
from AgentCrew.modules.chat.file_handler import FileHandler
from AgentCrew.modules.llm.model_registry import ModelRegistry
from AgentCrew.modules.llm.service_manager import ServiceManager
from AgentCrew.modules.chat.consolidation import ConversationConsolidator
from AgentCrew.modules.config import ConfigManagement
from AgentCrew.modules.mcpclient import MCPService
import shlex


@dataclass
class CommandResult:
    """Result of command processing."""

    handled: bool = False
    exit_flag: bool = False
    clear_flag: bool = False
    message: str = ""


class CommandProcessor:
    """Handles command processing for the message handler."""

    def __init__(self, message_handler):
        from AgentCrew.modules.chat.message import MessageHandler

        if isinstance(message_handler, MessageHandler):
            self.message_handler = message_handler

    async def process_command(self, user_input: str) -> CommandResult:
        """Process a command and return the result."""
        # Exit command
        if self._is_exit_command(user_input):
            self.message_handler._notify("exit_requested")
            return CommandResult(handled=True, exit_flag=True)

        # Clear command
        elif user_input.lower() == "/clear":
            self.message_handler.start_new_conversation()
            return CommandResult(handled=True, clear_flag=True)

        # Copy command
        elif user_input.lower() == "/copy":
            self.message_handler._notify(
                "copy_requested", self.message_handler.latest_assistant_response
            )
            return CommandResult(handled=True, clear_flag=True)

        # Debug command
        elif user_input.lower() == "/debug":
            self.message_handler._notify(
                "debug_requested", self.message_handler.agent.history
            )
            self.message_handler._notify(
                "debug_requested", self.message_handler.streamline_messages
            )
            return CommandResult(handled=True, clear_flag=True)

        # Think command
        elif user_input.lower().startswith("/think "):
            try:
                budget = user_input[7:].strip()
                self.message_handler.agent.configure_think(budget)
                self.message_handler._notify("think_budget_set", budget)
            except ValueError:
                self.message_handler._notify(
                    "error", "Invalid budget value. Please provide a number."
                )
            return CommandResult(handled=True, clear_flag=True)

        # Consolidate command
        elif user_input.lower().startswith("/consolidate"):
            return await self._handle_consolidate_command(user_input)

        # Jump command
        elif user_input.lower().startswith("/jump "):
            jumped = self._handle_jump_command(user_input)
            return CommandResult(handled=jumped, clear_flag=True)

        # Agent command
        elif user_input.lower().startswith("/agent"):
            success, message = self._handle_agent_command(user_input)
            self.message_handler._notify(
                "agent_command_result", {"success": success, "message": message}
            )
            return CommandResult(handled=True, clear_flag=True)

        # Model command
        elif user_input.lower().startswith("/model"):
            exit_flag, clear_flag = self._handle_model_command(user_input)
            return CommandResult(
                handled=True, exit_flag=exit_flag, clear_flag=clear_flag
            )
        elif user_input.lower().startswith("/mcp"):
            exit_flag, clear_flag = await self._handle_mcp_command(user_input)
            return CommandResult(
                handled=True, exit_flag=exit_flag, clear_flag=clear_flag
            )

        # File command
        elif user_input.startswith("/file "):
            return self._handle_file_command(user_input)

        # Not a command
        return CommandResult(handled=False)

    def _is_exit_command(self, user_input: str) -> bool:
        return user_input.lower() in ["/exit", "/quit"]

    async def _handle_consolidate_command(self, user_input: str) -> CommandResult:
        """Handle consolidate command."""
        try:
            # Extract the parameter (number of messages to preserve)
            parts = user_input.split()
            if len(parts) == 1:
                preserve_count = 10
            else:
                preserve_count = int(parts[1])

            if isinstance(self.message_handler.agent, LocalAgent):
                consolidator = ConversationConsolidator(self.message_handler.agent.llm)

                # Consolidate messages
                result = await consolidator.consolidate(
                    self.message_handler.streamline_messages, preserve_count
                )

                if result["success"]:
                    # Rebuild agent messages from consolidated messages
                    self.message_handler.agent_manager.rebuild_agents_messages(
                        self.message_handler.streamline_messages
                    )

                    # Notify UI about consolidation
                    self.message_handler._notify("consolidation_completed", result)

                    # If we have a conversation ID, update the persistent storage
                    if self.message_handler.current_conversation_id:
                        try:
                            self.message_handler.persistent_service.append_conversation_messages(
                                self.message_handler.current_conversation_id,
                                self.message_handler.streamline_messages,
                                True,  # Replace all messages with the consolidated version
                            )
                        except Exception as e:
                            self.message_handler._notify(
                                "error",
                                f"Failed to save consolidated conversation: {str(e)}",
                            )

                    message = (
                        f"Consolidated {result['messages_consolidated']} messages, "
                        f"preserving {result['messages_preserved']} recent messages. "
                        f"Token savings: ~{result['original_token_count'] - result['consolidated_token_count']}"
                    )
                    self.message_handler._notify("system_message", message)
                else:
                    self.message_handler._notify(
                        "system_message",
                        f"Consolidation skipped: {result['reason']}",
                    )

                return CommandResult(handled=True, clear_flag=True)
            else:
                self.message_handler._notify(
                    "error",
                    "Consolidation is only supported with LocalAgent.",
                )
                return CommandResult(handled=False, clear_flag=False)
        except ValueError as e:
            self.message_handler._notify(
                "error",
                f"Invalid consolidation parameter: {str(e)}. Use /consolidate [number]",
            )
            return CommandResult(handled=True, clear_flag=True)
        except Exception as e:
            self.message_handler._notify(
                "error", f"Error during consolidation: {str(e)}"
            )
            return CommandResult(handled=True, clear_flag=True)

    async def _handle_mcp_command(self, command: str) -> Tuple[bool, bool]:
        """
        Handle the /mcp command: list prompts or fetch a specific prompt content.

        Returns:
            Tuple of (exit_flag, clear_flag)
        """
        parts = command.strip().split()
        mcp_service: Optional[MCPService] = self.message_handler.mcp_manager.mcp_service
        # /mcp with no args: list all prompts
        if len(parts) == 1:
            # Aggregate prompts from all servers
            prompts = []
            if mcp_service:
                for server_id, prompt_list in mcp_service.server_prompts.items():
                    for prompt in prompt_list:
                        prompt_name = prompt.name
                        if prompt_name:
                            prompts.append(f"{server_id}/{prompt_name}")
            msg = (
                "Available MCP prompts:\n" + "\n".join(prompts)
                if prompts
                else "No MCP prompts found."
            )
            self.message_handler._notify("system_message", msg)
            return False, True
        # /mcp <server_id.prompt_name>: fetch and show the prompt
        elif len(parts) == 2:
            full_name = parts[1]
            if "/" not in full_name:
                self.message_handler._notify(
                    "error", "Please use format: /mcp server_id/prompt_name"
                )
                return False, True
            server_id, prompt_name = full_name.split("/", 1)
            try:
                prompt = await mcp_service.get_prompt(server_id, prompt_name)
                prompt_content = prompt.get("content", [])
                if len(prompt_content) > 0:
                    prompt_text = prompt_content[0].content.text
                    self.message_handler._notify(
                        "mcp_prompt",
                        {"name": prompt_name, "content": f"{prompt_text}"},
                    )
                else:
                    self.message_handler._notify(
                        "error", f"Prompt {server_id}.{prompt_name} not found."
                    )
            except Exception as e:
                self.message_handler._notify(
                    "error", f"Error fetching prompt: {str(e)}"
                )
            return False, True
        else:
            self.message_handler._notify("error", "Usage: /mcp [server_id.prompt_name]")
            return False, True

    def _handle_jump_command(self, command: str) -> bool:
        """Handle the /jump command to rewind conversation to a previous turn."""
        try:
            # Extract the turn number from the command
            parts = command.split()
            if len(parts) != 2:
                self.message_handler._notify("error", "Usage: /jump <turn_number>")
                return False

            turn_number = int(parts[1])

            # Validate the turn number
            if turn_number < 1 or turn_number > len(
                self.message_handler.conversation_turns
            ):
                self.message_handler._notify(
                    "error",
                    f"Invalid turn number. Available turns: 1-{len(self.message_handler.conversation_turns)}",
                )
                return False

            # Get the selected turn
            selected_turn = self.message_handler.conversation_turns[turn_number - 1]

            # Truncate messages to the index from the selected turn
            self.message_handler.streamline_messages = (
                self.message_handler.streamline_messages[: selected_turn.message_index]
            )
            if self.message_handler.current_conversation_id:
                self.message_handler.persistent_service.append_conversation_messages(
                    self.message_handler.current_conversation_id,
                    self.message_handler.streamline_messages,
                    True,
                )

            # Get the last assistant message from the streamline messages
            last_assistant_message = next(
                (
                    msg
                    for msg in reversed(self.message_handler.streamline_messages)
                    if msg.get("role") == "assistant"
                ),
                None,
            )
            if last_assistant_message and last_assistant_message.get("agent", ""):
                self._handle_agent_command(f"/agent {last_assistant_message['agent']}")

            self.message_handler.agent_manager.rebuild_agents_messages(
                self.message_handler.streamline_messages
            )
            self.message_handler.conversation_turns = (
                self.message_handler.conversation_turns[: turn_number - 1]
            )
            self.message_handler.last_assisstant_response_idx = len(
                self.message_handler.agent.history
            )

            self.message_handler._notify(
                "jump_performed",
                {"turn_number": turn_number, "preview": selected_turn.get_preview(100)},
            )

            return True

        except ValueError:
            self.message_handler._notify(
                "error", "Invalid turn number. Please provide a number."
            )
            return False

    def _handle_model_command(self, command: str) -> Tuple[bool, bool]:
        """
        Handle the /model command to switch models or list available models.

        Returns:
            Tuple of (exit_flag, clear_flag)
        """
        model_id = command[7:].strip()
        registry = ModelRegistry.get_instance()
        manager = ServiceManager.get_instance()

        # If no model ID is provided, list available models
        if not model_id:
            models_by_provider = {}
            for provider in registry.get_providers():
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
                                "id": f"{model.provider}/{model.id}",
                                "name": model.name,
                                "description": model.description,
                                "capabilities": model.capabilities,
                                "current": current,
                            }
                        )

            self.message_handler._notify("models_listed", models_by_provider)
            return False, True

        # Try to switch to the specified model
        if registry.set_current_model(model_id):
            model = registry.get_current_model()
            if model:
                # Update the LLM service
                manager.set_model(model.provider, model.id)

                # Get the new LLM service
                new_llm_service = manager.get_service(model.provider)

                # Update the agent manager with the new LLM service
                self.message_handler.agent_manager.update_llm_service(new_llm_service)

                # NEW: Persist the last used model to global config
                try:
                    config_manager = ConfigManagement()
                    config_manager.set_last_used_model(model_id, model.provider)
                except Exception as e:
                    # Don't fail the command if config save fails, just log it
                    print(f"Warning: Failed to save last used model: {e}")

                self.message_handler._notify(
                    "model_changed",
                    {"id": model.id, "name": model.name, "provider": model.provider},
                )
            else:
                self.message_handler._notify("error", "Failed to switch model.")
        else:
            self.message_handler._notify("error", f"Unknown model: {model_id}")

        return False, True

    def _handle_agent_command(self, command: str) -> Tuple[bool, str]:
        """
        Handle the /agent command to switch agents or list available agents.

        Returns:
            Tuple of (success, message)
        """
        parts = command.split()

        # If no agent name is provided, list available agents
        if len(parts) == 1:
            agents_info = {"current": self.message_handler.agent.name, "available": {}}

            for agent_name, agent in self.message_handler.agent_manager.agents.items():
                agents_info["available"][agent_name] = {
                    "description": agent.description,
                    "current": (
                        self.message_handler.agent
                        and self.message_handler.agent.name == agent_name
                    ),
                }

            self.message_handler._notify("agents_listed", agents_info)
            return True, "Listed available agents"

        # If an agent name is provided, try to switch to that agent
        agent_name = parts[1]
        old_agent_name = self.message_handler.agent_manager.get_current_agent().name
        if self.message_handler.agent_manager.select_agent(agent_name):
            self.message_handler.agent = (
                self.message_handler.agent_manager.get_current_agent()
            )
            old_agent = self.message_handler.agent_manager.get_agent(old_agent_name)
            if old_agent:
                self.message_handler.agent.history = list(old_agent.history)
                old_agent.history = []

            # NEW: Persist the last used agent to global config
            try:
                config_manager = ConfigManagement()
                config_manager.set_last_used_agent(agent_name)
            except Exception as e:
                # Don't fail the command if config save fails, just log it
                print(f"Warning: Failed to save last used agent: {e}")

            self.message_handler._notify("agent_changed", agent_name)
            return True, f"Switched to {agent_name} agent"
        else:
            available_agents = ", ".join(
                self.message_handler.agent_manager.agents.keys()
            )
            self.message_handler._notify(
                "error",
                f"Unknown agent: {agent_name}. Available agents: {available_agents}",
            )
            return (
                False,
                f"Unknown agent: {agent_name}. Available agents: {available_agents}",
            )

    def _handle_file_command(self, user_input: str) -> CommandResult:
        """Handle file command with support for multiple files."""
        # Extract file paths from user input (space-separated)
        file_paths_str: str = user_input[6:].strip()
        file_paths: List[str] = [
            os.path.expanduser(path.strip())
            for path in shlex.split(file_paths_str)
            if path.strip()
        ]

        if not file_paths:
            self.message_handler._notify("error", "No file paths provided")
            return CommandResult(handled=True, clear_flag=True)

        processed_files: List[str] = []
        failed_files: List[str] = []
        all_file_contents: List[Dict[str, str]] = []

        # Process each file
        for file_path in file_paths:
            self.message_handler._notify("file_processing", {"file_path": file_path})

            # Process file with the file handling service
            if self.message_handler.file_handler is None:
                self.message_handler.file_handler = FileHandler()
            file_content = self.message_handler.file_handler.process_file(file_path)

            # Fallback to llm handle
            if not file_content:
                from AgentCrew.modules.agents.base import MessageType

                file_content = self.message_handler.agent.format_message(
                    MessageType.FileContent, {"file_uri": file_path}
                )

            if file_content:
                all_file_contents.append(file_content)
                processed_files.append(file_path)
                self.message_handler._notify(
                    "file_processed",
                    {
                        "file_path": file_path,
                        "message": file_content,
                    },
                )
            else:
                failed_files.append(file_path)
                self.message_handler._notify(
                    "error",
                    f"Failed to process file {file_path} Or Model is not supported",
                )

        # Add all successfully processed file contents to messages
        if all_file_contents:
            self.message_handler._messages_append(
                {"role": "user", "content": all_file_contents}
            )

            # Notify about overall processing results
            if failed_files:
                self.message_handler._notify(
                    "error", f"Failed to process: {', '.join(failed_files)}"
                )
            self.message_handler._notify(
                "system_message",
                f"Successfully processed {len(processed_files)} files: {', '.join(processed_files)}",
            )

        return CommandResult(handled=True, clear_flag=True)
