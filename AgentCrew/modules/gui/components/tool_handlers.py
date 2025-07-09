from typing import Any, Dict
from PySide6.QtWidgets import QMessageBox, QTextEdit, QGridLayout
from PySide6.QtCore import Qt
from AgentCrew.modules.gui.widgets.tool_widget import ToolWidget


class ToolEventHandler:
    """Handles tool-related events in the chat UI."""

    def __init__(self, chat_window):
        from AgentCrew.modules.gui import ChatWindow

        if isinstance(chat_window, ChatWindow):
            self.chat_window = chat_window

    def handle_event(self, event: str, data: Any):
        """Handle a tool-related event."""
        if event == "tool_use":
            self.handle_tool_use(data)
        elif event == "tool_result":
            self.handle_tool_result(data)
        elif event == "agent_changed_by_transfer":
            data["tool_result"] = f"Transfered to Agent: {data['agent_name']}"
            self.handle_tool_result(data)
            self.chat_window.status_indicator.setText(
                f"Agent: {data['agent_name']} | Model: {self.chat_window.message_handler.agent.get_model()}"
            )
        elif event == "tool_error":
            self.handle_tool_error(data)
        elif event == "tool_confirmation_required":
            self.handle_tool_confirmation_required(data)
        elif event == "tool_denied":
            self.handle_tool_denied(data)

    def handle_tool_use(self, tool_use: Dict):
        """Display information about a tool being used."""
        # Create tool widget
        tool_widget = ToolWidget(tool_use["name"], tool_use)

        # Store reference to the tool widget for later result update
        tool_use["tool_widget_ref"] = tool_widget

        # Add to chat using convenience method
        self.chat_window.chat_components.add_tool_widget(tool_widget)

        # Display status message
        self.chat_window.display_status_message(f"Using tool: {tool_use['name']}")

    def handle_tool_result(self, data: Dict):
        """Display the result of a tool execution."""
        tool_use = data["tool_use"]
        tool_result = data["tool_result"]

        # If we have a reference to the tool widget, update it
        if "tool_widget_ref" in tool_use:
            tool_widget = tool_use["tool_widget_ref"]
            tool_widget.update_with_result(tool_result)
        else:
            # Create a new tool widget with both use and result
            tool_widget = ToolWidget(tool_use["name"], tool_use, tool_result)

            # Add to chat using convenience method
            self.chat_window.chat_components.add_tool_widget(tool_widget)
        self.chat_window.message_event_handler.chunk_buffer_queue = []
        self.chat_window.message_event_handler.think_buffer_queue = []
        self.chat_window.current_response_bubble = None
        self.chat_window.current_response_container = None

        # Reset the current response bubble so the next agent message starts in a new bubble

    def handle_tool_error(self, data: Dict):
        """Display an error that occurred during tool execution."""
        tool_use = data["tool_use"]
        error = data["error"]

        # If we have a reference to the tool widget, update it
        if "tool_widget_ref" in tool_use:
            tool_widget = tool_use["tool_widget_ref"]
            tool_widget.update_with_result(error, is_error=True)
        else:
            # Create a new tool widget with both use and error result
            tool_widget = ToolWidget(tool_use["name"], tool_use, error, is_error=True)

            # Add to chat using convenience method
            self.chat_window.chat_components.add_tool_widget(tool_widget)

        self.chat_window.display_status_message(f"Error in tool {tool_use['name']}")

        # Reset the current response bubble so the next agent message starts in a new bubble
        self.chat_window.current_response_bubble = None
        self.chat_window.current_response_container = None

        if self.chat_window.current_thinking_bubble:
            self.chat_window.current_thinking_bubble.raw_text_buffer = (
                self.chat_window.thinking_content
            )
            self.chat_window.current_thinking_bubble._finalize_streaming()
            self.chat_window.current_thinking_bubble = None
        self.chat_window.thinking_content = ""

    def handle_tool_confirmation_required(self, tool_info):
        """Display a dialog for tool confirmation request."""
        tool_use = tool_info.copy()
        confirmation_id = tool_use.pop("confirmation_id")

        # Create dialog
        dialog = QMessageBox(self.chat_window)
        dialog.setWindowTitle("Tool Execution Confirmation")
        dialog.setIcon(QMessageBox.Icon.Question)

        # Format tool information for display
        tool_description = f"The assistant wants to use the '{tool_use['name']}' tool."
        params_text = ""

        if isinstance(tool_use["input"], dict):
            params_text = "Parameters:"
            for key, value in tool_use["input"].items():
                params_text += f"\n• {key}: {value}"
        else:
            params_text = f"\n\nInput: {tool_use['input']}"

        dialog.setInformativeText("Do you want to allow this tool to run?")
        dialog.setText(tool_description)

        lt = dialog.layout()
        text_edit = QTextEdit()
        text_edit.setMinimumWidth(500)
        text_edit.setMinimumHeight(300)
        text_edit.setReadOnly(True)
        text_edit.setText(params_text)

        # Style the text edit to match the main theme
        text_edit.setStyleSheet(
            self.chat_window.style_provider.get_tool_dialog_text_edit_style()
        )

        if isinstance(lt, QGridLayout):
            lt.addWidget(
                text_edit,
                lt.rowCount() - 2,
                2,
                1,
                lt.columnCount() - 2,
                Qt.AlignmentFlag.AlignLeft,
            )

        # Add buttons
        yes_button = dialog.addButton("Yes (Once)", QMessageBox.ButtonRole.YesRole)
        no_button = dialog.addButton("No", QMessageBox.ButtonRole.NoRole)
        all_button = dialog.addButton("Yes to All", QMessageBox.ButtonRole.AcceptRole)
        forever_button = dialog.addButton("Forever", QMessageBox.ButtonRole.AcceptRole)

        # Style the buttons with Catppuccin colors
        yes_button.setStyleSheet(
            self.chat_window.style_provider.get_tool_dialog_yes_button_style()
        )

        all_button.setStyleSheet(
            self.chat_window.style_provider.get_tool_dialog_all_button_style()
        )

        forever_button.setStyleSheet(
            self.chat_window.style_provider.get_tool_dialog_all_button_style()
        )

        no_button.setStyleSheet(
            self.chat_window.style_provider.get_tool_dialog_no_button_style()
        )

        # Execute dialog and get result
        dialog.exec()
        clicked_button = dialog.clickedButton()

        # Process result
        if clicked_button == yes_button:
            self.chat_window.message_handler.resolve_tool_confirmation(
                confirmation_id, {"action": "approve"}
            )
            self.chat_window.display_status_message(
                f"Approved tool: {tool_use['name']}"
            )
        elif clicked_button == all_button:
            self.chat_window.message_handler.resolve_tool_confirmation(
                confirmation_id, {"action": "approve_all"}
            )
            self.chat_window.display_status_message(
                f"Approved all future calls to tool: {tool_use['name']}"
            )
        elif clicked_button == forever_button:
            from AgentCrew.modules.config import ConfigManagement

            config_manager = ConfigManagement()
            config_manager.write_auto_approval_tools(tool_use["name"], add=True)

            self.chat_window.message_handler.resolve_tool_confirmation(
                confirmation_id, {"action": "approve_all"}
            )
            self.chat_window.display_status_message(
                f"Tool '{tool_use['name']}' will be auto-approved forever"
            )
        else:  # No or dialog closed
            self.chat_window.message_handler.resolve_tool_confirmation(
                confirmation_id, {"action": "deny"}
            )
            self.chat_window.display_status_message(f"Denied tool: {tool_use['name']}")

    def handle_tool_denied(self, data):
        """Display a message about a denied tool execution."""
        tool_use = data["tool_use"]
        self.chat_window.chat_components.add_system_message(
            f"❌ Tool execution rejected: {tool_use['name']}"
        )
        self.chat_window.display_status_message(
            f"Tool execution rejected: {tool_use['name']}"
        )
