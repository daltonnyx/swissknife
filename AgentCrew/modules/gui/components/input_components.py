import re
import os
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QCompleter,
    QFileDialog,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QStringListModel
from PySide6.QtGui import QTextCursor
import qtawesome as qta
from AgentCrew.modules.chat.completers import DirectoryListingCompleter
from .completers import GuiChatCompleter


class InputComponents:
    """Handles input-related UI components and file completion."""

    def __init__(self, chat_window):
        from AgentCrew.modules.gui import ChatWindow

        if isinstance(chat_window, ChatWindow):
            self.chat_window = chat_window
        self._setup_input_area()
        self._setup_file_completion()

    def _setup_input_area(self):
        """Set up the input area with text input and buttons."""
        # Input area
        self.chat_window.message_input = QTextEdit()
        input_font = self.chat_window.message_input.font()
        input_font.setPixelSize(16)
        self.chat_window.message_input.setFont(input_font)
        self.chat_window.message_input.setReadOnly(False)
        self.chat_window.message_input.setMaximumHeight(120)
        self.chat_window.message_input.setPlaceholderText(
            "Type your message here... (Ctrl+Enter to send)"
        )
        self.chat_window.message_input.setAcceptRichText(False)
        self.chat_window.message_input.setStyleSheet(
            self.chat_window.style_provider.get_input_style()
        )
        self.chat_window.message_input.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding
        )

        # Create buttons layout
        buttons_layout = QVBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 5, 0)

        # Create Send button
        send_icon = qta.icon("fa6s.paper-plane", color="white")
        self.chat_window.send_button = QPushButton(send_icon, "")
        self.chat_window.send_button.setFont(input_font)
        self.chat_window.send_button.setStyleSheet(
            self.chat_window.style_provider.get_button_style("primary")
        )

        # Create File button
        upload_icon = qta.icon("fa6s.arrow-up-from-bracket", color="white")
        self.chat_window.file_button = QPushButton(upload_icon, "")
        self.chat_window.file_button.setFont(input_font)
        self.chat_window.file_button.setStyleSheet(
            self.chat_window.style_provider.get_button_style("secondary")
        )

        # Add buttons to layout
        buttons_layout.addWidget(self.chat_window.send_button)
        buttons_layout.addWidget(self.chat_window.file_button)
        buttons_layout.addStretch(1)

        # Store the buttons layout for use in main window
        self.buttons_layout = buttons_layout

    def _setup_file_completion(self):
        """Set up file path completion for the input field."""
        # Set up file path completion
        self.chat_window.file_completer = QCompleter(self.chat_window)
        self.chat_window.file_completer.setCompletionMode(
            QCompleter.CompletionMode.PopupCompletion
        )
        self.chat_window.file_completer.setCaseSensitivity(
            Qt.CaseSensitivity.CaseSensitive
        )
        self.chat_window.file_completer.setWidget(self.chat_window.message_input)
        self.chat_window.file_completer.activated.connect(self.insert_completion)

        self.directory_completer = DirectoryListingCompleter()
        self.path_prefix = ""

        # Add chat command completion
        self.chat_completer = GuiChatCompleter(
            getattr(self.chat_window, "message_handler", None)
        )
        self.chat_window.command_completer = QCompleter(self.chat_window)
        self.chat_window.command_completer.setCompletionMode(
            QCompleter.CompletionMode.PopupCompletion
        )
        self.chat_window.command_completer.setCaseSensitivity(
            Qt.CaseSensitivity.CaseSensitive
        )
        self.chat_window.command_completer.setWidget(self.chat_window.message_input)
        self.chat_window.command_completer.activated.connect(
            self.insert_command_completion
        )

        self.chat_window.message_input.textChanged.connect(
            self.check_for_path_completion
        )

    def check_for_path_completion(self):
        """Check if the current text contains a path that should trigger completion."""
        if self.chat_window.file_completer.popup().isVisible():
            self.chat_window.file_completer.popup().hide()
        if self.chat_window.command_completer.popup().isVisible():
            self.chat_window.command_completer.popup().hide()
        text = self.chat_window.message_input.toPlainText()
        cursor_position = self.chat_window.message_input.textCursor().position()

        # Get the text up to the cursor position
        text_to_cursor = text[:cursor_position]

        # First check for command completion
        if text_to_cursor.startswith("/") and not text_to_cursor.startswith("/file "):
            self.check_for_command_completion()
            return

        # Look for path patterns that should trigger completion
        path_match = re.search(r"((~|\.{1,2})?/[^\s]*|~)$", text_to_cursor)

        if path_match:
            path = path_match.group(0)
            completions = self.directory_completer.get_path_completions(path)

            if completions:
                # Create a model for the completer
                model = QStringListModel(completions)
                self.chat_window.file_completer.setModel(model)

                # Calculate the prefix length to determine what part to complete
                prefix = os.path.basename(path) if "/" in path else path
                self.chat_window.file_completer.setCompletionPrefix(prefix)

                # Store the path prefix (everything before the basename)
                self.path_prefix = path[: len(path) - len(prefix)]

                # Show the completion popup
                popup = self.chat_window.file_completer.popup()
                popup.setCurrentIndex(
                    self.chat_window.file_completer.completionModel().index(0, 0)
                )

                # Calculate position for the popup
                rect = self.chat_window.message_input.cursorRect()
                rect.setWidth(400)

                # Show the popup
                self.chat_window.file_completer.complete(rect)
            else:
                # Hide the popup if no completions
                self.chat_window.file_completer.popup().hide()

    def insert_completion(self, completion):
        """Insert the selected completion into the text input."""
        cursor = self.chat_window.message_input.textCursor()
        text = self.chat_window.message_input.toPlainText()
        position = cursor.position()

        if text.startswith("/") and not text.startswith("/file "):
            self.insert_command_completion(completion)
            return
        # Find the start of the path
        text_to_cursor = text[:position]
        path_match = re.search(r"((~|\.{1,2})?/[^\s]*|~)$", text_to_cursor)

        if path_match:
            path_start = path_match.start()
            path = path_match.group(0)

            # Calculate what part of the path to replace
            prefix = os.path.basename(path) if "/" in path else path
            prefix_start = path_start + len(path) - len(prefix)

            # Replace the prefix with the completion
            cursor.setPosition(prefix_start)
            cursor.setPosition(position, QTextCursor.MoveMode.KeepAnchor)

            cursor.insertText(completion.replace(" ", "\\ "))

    def browse_file(self):
        """Open file dialog and process selected file."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.chat_window,
            "Select File",
            "",
            "All Files (*);;Text Files (*.txt);;PDF Files (*.pdf);;Word Files (*.docx)",
        )

        for file_path in file_paths:
            if file_path and os.path.isfile(file_path):
                # Disable input controls while processing file
                self.chat_window.ui_state_manager.set_input_controls_enabled(False)

                # Process the file using the /file command
                file_command = f'/file "{file_path}"'
                self.chat_window.display_status_message(f"Processing file: {file_path}")

                # Send the file command to the worker thread
                self.chat_window.llm_worker.process_request.emit(file_command)

    def check_for_command_completion(self):
        """Check if the current text should trigger command completion."""
        text = self.chat_window.message_input.toPlainText()
        cursor_position = self.chat_window.message_input.textCursor().position()

        # Get the text up to the cursor position
        text_to_cursor = text[:cursor_position]

        # Check if we're typing a command
        if text_to_cursor.startswith("/"):
            completions = self.chat_completer.get_completions(text_to_cursor)

            if completions:
                model = QStringListModel(completions)
                self.chat_window.command_completer.setModel(model)

                popup = self.chat_window.command_completer.popup()
                popup.setCurrentIndex(
                    self.chat_window.command_completer.completionModel().index(0, 0)
                )

                # Show completion popup
                rect = self.chat_window.message_input.cursorRect()
                rect.setWidth(400)
                self.chat_window.command_completer.complete(rect)
            else:
                self.chat_window.command_completer.popup().hide()

    def insert_command_completion(self, completion: str):
        """Insert the selected command completion."""
        cursor = self.chat_window.message_input.textCursor()
        text = self.chat_window.message_input.toPlainText()
        position = cursor.position()

        # Find the start of the command
        text_to_cursor = text[:position]
        if text_to_cursor.startswith("/"):
            # Replace the current command with the completion
            for word in text_to_cursor.split():
                if completion.find(word) != -1:
                    position = text_to_cursor.rfind(word)
            cursor.setPosition(position, QTextCursor.MoveMode.KeepAnchor)
            cursor.insertText(completion)

    def get_input_layout(self):
        """Get the input row layout for integration with main window."""
        input_row = QHBoxLayout()
        input_row.addWidget(self.chat_window.message_input, 1)
        input_row.addLayout(self.buttons_layout)
        return input_row
