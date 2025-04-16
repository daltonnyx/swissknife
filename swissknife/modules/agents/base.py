from abc import ABC
from typing import Dict, Any, List
from swissknife.modules.prompts.constants import ANALYSIS_PROMPT


class Agent(ABC):
    """Base class for all specialized agents."""

    def __init__(
        self,
        name: str,
        description: str,
        llm_service,
        services: Dict[str, Any],
        tools: List[str],
    ):
        """
        Initialize a new agent.

        Args:
            name: The name of the agent
            description: A description of the agent's capabilities
            llm_service: The LLM service to use for this agent
            services: Dictionary of available services
        """
        self.name = name
        self.description = description
        self.llm = llm_service
        self.services = services
        self.tools: List[str] = tools  # List of tool names that the agent needs
        self.system_prompt = None
        self.custom_system_prompt = None
        self.history = []
        self.shared_context_pool: Dict[str, List[int]] = {}
        # Store tool definitions in the same format as ToolRegistry
        self.tool_definitions = {}  # {tool_name: (definition_func, handler_factory, service_instance)}
        self.registered_tools = (
            set()
        )  # Set of tool names that are registered with the LLM
        self.is_active = False

        self.register_tools()

    def _extract_tool_name(self, tool_def: Dict) -> str:
        """
        Extract tool name from definition regardless of format.

        Args:
            tool_def: The tool definition

        Returns:
            The name of the tool

        Raises:
            ValueError: If the tool name cannot be extracted
        """
        if "name" in tool_def:
            return tool_def["name"]
        elif "function" in tool_def and "name" in tool_def["function"]:
            return tool_def["function"]["name"]
        else:
            raise ValueError("Could not extract tool name from definition")

    def register_tools(self):
        """
        Register tools for this agent using the services dictionary.
        """

        if self.services.get("agent_manager"):
            from swissknife.modules.agents.tools.transfer import (
                register as register_transfer,
            )

            register_transfer(self.services["agent_manager"], self)
        for tool_name in self.tools:
            if self.services and tool_name in self.services:
                service = self.services[tool_name]
                if service:
                    if tool_name == "llm":
                        self.register_tool(
                            service.register_tool, service.register_tool
                        )  # Example: register LLM tools
                    elif tool_name == "memory":
                        from swissknife.modules.memory.tool import (
                            register as register_memory,
                        )

                        register_memory(service, self)
                    elif tool_name == "clipboard":
                        from swissknife.modules.clipboard.tool import (
                            register as register_clipboard,
                        )

                        register_clipboard(service, self)
                    elif tool_name == "code_analysis":
                        from swissknife.modules.code_analysis.tool import (
                            register as register_code_analysis,
                        )

                        register_code_analysis(service, self)
                    elif tool_name == "web_search":
                        from swissknife.modules.web_search.tool import (
                            register as register_web_search,
                        )

                        register_web_search(service, self)
                    elif tool_name == "aider":
                        from swissknife.modules.coder.tool import (
                            register as register_spec_validator,
                        )

                        register_spec_validator(service, self)

                    else:
                        print(f"⚠️ Tool {tool_name} not found in services")
            else:
                print(f"⚠️ Service {tool_name} not available for tool registration")

    def register_tool(self, definition_func, handler_factory, service_instance=None):
        """
        Register a tool with this agent.

        Args:
            definition_func: Function that returns tool definition given a provider or direct definition
            handler_factory: Function that creates a handler function or direct handler
            service_instance: Service instance needed by the handler (optional)
        """
        # Get the tool definition to extract the name
        tool_def = definition_func() if callable(definition_func) else definition_func
        tool_name = self._extract_tool_name(tool_def)

        # Store the definition function, handler factory, and service instance
        self.tool_definitions[tool_name] = (
            definition_func,
            handler_factory,
            service_instance,
        )

        # If the agent is active, register the tool with the LLM immediately
        if self.is_active and self.llm:
            # Get provider-specific definition
            provider = getattr(self.llm, "provider_name", None)
            if callable(definition_func) and provider:
                try:
                    tool_def = definition_func(provider)
                except TypeError:
                    # If definition_func doesn't accept provider argument
                    tool_def = definition_func()
            else:
                tool_def = definition_func

            # Get handler function
            if callable(handler_factory):
                handler = (
                    handler_factory(service_instance)
                    if service_instance
                    else handler_factory()
                )
            else:
                handler = handler_factory

            # Register with LLM
            self.llm.register_tool(tool_def, handler)
            self.registered_tools.add(tool_name)

    def set_system_prompt(self, prompt: str):
        """
        Set the system prompt for this agent.

        Args:
            prompt: The system prompt
        """
        self.system_prompt = prompt

    def set_custom_system_prompt(self, prompt: str):
        """
        Set the system prompt for this agent.

        Args:
            prompt: The system prompt
        """
        self.custom_system_prompt = prompt

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Returns:
            The system prompt
        """
        return self.system_prompt

    def activate(self):
        """
        Activate this agent by registering all tools with the LLM service.

        Returns:
            True if activation was successful, False otherwise
        """
        if not self.llm:
            return False

        self.register_tools_with_llm()
        system_prompt = self.get_system_prompt()
        if self.custom_system_prompt:
            system_prompt = (
                self.get_system_prompt() + "\n---\n\n" + self.custom_system_prompt
                # + "\n---\n\n"
                # + ANALYSIS_PROMPT
            )

        self.llm.set_system_prompt(system_prompt)
        self.is_active = True
        return True

    def deactivate(self):
        """
        Deactivate this agent by clearing all tools from the LLM service.

        Returns:
            True if deactivation was successful, False otherwise
        """
        if not self.llm:
            return False

        self.clear_tools_from_llm()
        self.is_active = False
        return True

    def register_tools_with_llm(self):
        """
        Register all of this agent's tools with the LLM service.
        """
        if not self.llm:
            return

        # Clear existing tools first to avoid duplicates
        self.clear_tools_from_llm()

        # Get the provider name if available
        provider = getattr(self.llm, "provider_name", None)

        for tool_name, (
            definition_func,
            handler_factory,
            service_instance,
        ) in self.tool_definitions.items():
            try:
                # Get provider-specific definition if possible
                if callable(definition_func) and provider:
                    try:
                        tool_def = definition_func(provider)
                    except TypeError:
                        # If definition_func doesn't accept provider argument
                        tool_def = definition_func()
                else:
                    tool_def = definition_func

                # Get handler function
                if callable(handler_factory):
                    handler = (
                        handler_factory(service_instance)
                        if service_instance
                        else handler_factory()
                    )
                else:
                    handler = handler_factory

                # Register with LLM
                self.llm.register_tool(tool_def, handler)
                self.registered_tools.add(tool_name)
                # print(f"Resgitered tool {tool_name}")
            except Exception as e:
                print(f"Error registering tool {tool_name}: {e}")

    def clear_tools_from_llm(self):
        """
        Clear all tools from the LLM service.
        """
        if self.llm:
            self.llm.clear_tools()
            self.registered_tools.clear()
            # Note: We don't clear self.tool_definitions as we want to keep the definitions

    def update_llm_service(self, new_llm_service):
        """
        Update the LLM service used by this agent.

        Args:
            new_llm_service: The new LLM service to use

        Returns:
            True if the update was successful, False otherwise
        """
        was_active = self.is_active

        # Deactivate with the current LLM if active
        if was_active:
            self.deactivate()

        # Update the LLM service
        self.llm = new_llm_service

        # Reactivate with the new LLM if it was active before
        if was_active:
            self.activate()

        return True

    def process_messages(self, messages: List[Dict[str, Any]]):
        """
        Process messages using this agent.

        Args:
            messages: The messages to process

        Returns:
            The processed messages with the agent's response
        """
        # Ensure the first message is a system message with the agent's prompt
        if not messages or messages[0].get("role") != "system":
            system_message = {"role": "system", "content": self.get_system_prompt()}
            messages = [system_message] + messages
        elif messages[0].get("role") == "system":
            # Update the system message with this agent's prompt
            messages[0]["content"] = self.get_system_prompt()

        return messages
