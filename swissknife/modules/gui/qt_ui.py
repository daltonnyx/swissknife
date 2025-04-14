import re
import os
from typing import Any, Dict
import pyperclip

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QMessageBox,
    QMainWindow,
    QStatusBar,
    QLabel,
    QScrollArea,
    QMenu,
    QMenuBar,
    QFileDialog,
    QSplitter,
)
from PySide6.QtCore import (
    Qt,
    Slot,
    QThread,
    Signal,
    QStringListModel,
)
from swissknife.modules.llm.models import ModelRegistry
from swissknife.modules.agents.manager import AgentManager
from PySide6.QtGui import (
    QKeySequence,
    QShortcut,
    QFont,
    QAction,
    QTextCursor,
)
from swissknife.modules.chat.message_handler import MessageHandler, Observer
from swissknife.modules.chat.completers import DirectoryListingCompleter
from PySide6.QtWidgets import QCompleter
from swissknife.modules.gui.widgets import (
    TokenUsageWidget,
    SystemMessageWidget,
    MessageBubble,
    ConversationSidebar,
    ConversationLoader,
)

from .worker import LLMWorker


class ChatWindow(QMainWindow, Observer):
    # Signal for thread-safe event handling
    event_received = Signal(str, object)

    def __init__(self, message_handler: MessageHandler):
        super().__init__()
        self.setWindowTitle("Interactive Chat")
        self.setGeometry(100, 100, 1000, 700)  # Adjust size for sidebar

        # Set application-wide style
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #F9FAEF;
            }
            QScrollArea {
                border: none;
                background-color: #FFFFFF;
            }
            QSplitter::handle {
                background-color: #E1E4D5;
            }
            QStatusBar {
                background-color: #F3F4E9;
                color: #1A1C16;
            }
            """
        )

        # Create menu bar with styling
        self.create_menu_bar()

        # Initialize MessageHandler - kept in main thread
        self.message_handler = message_handler
        self.message_handler.attach(self)

        # Track if we're waiting for a response
        self.waiting_for_response = False
        self.loading_conversation = False  # Track conversation loading state

        # --- Create Chat Area Widgets ---
        # Create widget for chat messages
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_layout.setSpacing(10)

        # Create a scroll area for messages
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setWidget(self.chat_container)
        self.chat_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.chat_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # Create token usage widget
        self.token_usage = TokenUsageWidget()

        # Create the status indicator (showing current agent and model)
        self.status_indicator = QLabel(
            f"Agent: {self.message_handler.agent_name} | Model: {self.message_handler.llm.model}"
        )
        self.status_indicator.setStyleSheet(
            """
            background-color: #E1E4D5; 
            color: #4C662B;
            padding: 8px; 
            border-radius: 4px;
            font-weight: bold;
            """
        )

        # Input area
        self.message_input = QTextEdit()  # Use QTextEdit for multi-line input
        self.message_input.setFont(QFont("Arial", 12))
        self.message_input.setReadOnly(False)
        self.message_input.setMaximumHeight(100)  # Limit input height
        self.message_input.setPlaceholderText(
            "Type your message here... (Ctrl+Enter to send)"
        )
        self.message_input.setStyleSheet(
            """
            QTextEdit {
                border: 1px solid #C5C8BA;
                border-radius: 4px;
                background-color: white;
                padding: 8px;
            }
            QTextEdit:focus {
                border: 1px solid #4C662B;
            }
            """
        )

        # Set up file path completion
        self.file_completer = QCompleter(self)
        self.file_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.file_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseSensitive)
        self.file_completer.setWidget(self.message_input)
        self.file_completer.activated.connect(self.insert_completion)
        # Set completer to use Enter and Tab keys
        self.file_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.directory_completer = DirectoryListingCompleter()
        self.path_prefix = ""  # Store the path prefix for completions
        self.message_input.textChanged.connect(self.check_for_path_completion)

        # Create buttons layout
        buttons_layout = QVBoxLayout()  # Change to vertical layout for stacking buttons

        # Create Send button
        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Arial", 12))
        self.send_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4C662B; 
                color: white; 
                border: none;
                border-radius: 4px; 
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #354E16;
            }
            QPushButton:pressed {
                background-color: #102000;
            }
            QPushButton:disabled {
                background-color: #B1D18A;
            }
            """
        )

        # Create File button
        self.file_button = QPushButton("File")
        self.file_button.setFont(QFont("Arial", 12))
        self.file_button.setStyleSheet(
            """
            QPushButton {
                background-color: #586249; 
                color: white; 
                border: none;
                border-radius: 4px; 
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #404A33;
            }
            QPushButton:pressed {
                background-color: #2A331E;
            }
            QPushButton:disabled {
                background-color: #BFCBAD;
            }
            """
        )

        # Add buttons to layout
        buttons_layout.addWidget(self.send_button)
        buttons_layout.addWidget(self.file_button)
        buttons_layout.addStretch(1)  # Add stretch to keep buttons at the top

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # --- Assemble Chat Area Layout ---
        chat_area_widget = QWidget()  # Container for everything right of the sidebar
        chat_area_layout = QVBoxLayout(chat_area_widget)
        chat_area_layout.setContentsMargins(
            0, 0, 0, 0
        )  # No margins for this inner container
        chat_area_layout.addWidget(self.chat_scroll, 1)  # Give chat area more space
        chat_area_layout.addWidget(self.status_indicator)

        # Create horizontal layout for input and buttons
        input_row = QHBoxLayout()
        input_row.addWidget(self.message_input, 1)  # Give input area stretch priority
        input_row.addLayout(buttons_layout)  # Add buttons layout to the right

        chat_area_layout.addLayout(
            input_row
        )  # Add the horizontal layout to main layout
        chat_area_layout.addWidget(self.token_usage)

        # --- Create Sidebar ---
        self.sidebar = ConversationSidebar(self.message_handler, self)
        self.sidebar.conversation_selected.connect(self.load_conversation)
        self.sidebar.error_occurred.connect(self.display_error)  # Connect error signal
        self.sidebar.new_conversation_requested.connect(
            self.start_new_conversation
        )  # Connect new conversation signal

        # --- Create Splitter and Set Central Widget ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(chat_area_widget)
        self.splitter.setStretchFactor(0, 0)  # Sidebar doesn't stretch
        self.splitter.setStretchFactor(1, 1)  # Chat area stretches
        self.splitter.setSizes([250, 750])  # Initial sizes

        self.setCentralWidget(self.splitter)

        # --- Connect signals and slots (rest of the setup) ---
        self.send_button.clicked.connect(self.send_message)
        self.file_button.clicked.connect(self.browse_file)

        # Setup context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        # Connect event handling signal
        self.event_received.connect(self.handle_event)

        # Ctrl+Enter shortcut
        self.send_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.send_shortcut.activated.connect(self.send_message)

        # Ctrl+C shortcut (copy last response)
        self.copy_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        self.copy_shortcut.activated.connect(self.copy_last_response)

        # Ctrl+L shortcut (clear chat)
        self.clear_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self.clear_shortcut.activated.connect(
            lambda: self.clear_chat(requested=False)
        )  # Pass requested=False

        # Override key press event
        self.message_input.keyPressEvent = self.input_key_press_event

        # Thread and worker for LLM interaction
        self.llm_thread = QThread()
        self.llm_worker = LLMWorker()  # No message_handler passed to worker

        # Connect worker signals to UI slots
        self.llm_worker.response_ready.connect(self.handle_response)
        self.llm_worker.response_chunk.connect(self.display_response_chunk)
        self.llm_worker.error.connect(self.display_error)
        self.llm_worker.status_message.connect(self.display_status_message)
        self.llm_worker.token_usage.connect(self.update_token_usage)
        self.llm_worker.request_exit.connect(self.handle_exit_request)
        self.llm_worker.request_clear.connect(self.handle_clear_request)

        # Connect message handler to worker in the main thread
        self.llm_worker.connect_handler(self.message_handler)

        # Move worker to thread and start it
        self.llm_worker.moveToThread(self.llm_thread)
        self.llm_thread.start()

        # Initialize history position
        self.history_position = len(self.message_handler.history_manager.history)
        self.message_input.setFocus()

        # Track current response bubble for chunked responses
        self.current_response_bubble = None
        self.current_user_bubble = None
        self.current_response_container = None
        self.current_thinking_bubble = None
        self.thinking_content = ""
        self.expecting_response = False

        # Track session cost
        self.session_cost = 0.0

        # Add welcome message
        self.add_system_message(
            "Welcome! Select a past conversation or start a new one."
        )
        self.add_system_message(
            "Press Ctrl+Enter to send, Ctrl+C to copy, Ctrl+L to clear chat."
        )

    def closeEvent(self, event):
        """Handle window close event to clean up threads properly"""
        # Terminate worker thread properly
        self.llm_thread.quit()
        self.llm_thread.wait(1000)  # Wait up to 1 second for thread to finish
        # If the thread didn't quit cleanly, terminate it
        if self.llm_thread.isRunning():
            self.llm_thread.terminate()
            self.llm_thread.wait()
        super().closeEvent(event)

    def input_key_press_event(self, event):
        """Custom key press event for the message input."""
        # Handle Tab key for completion
        if event.key() == Qt.Key.Key_Tab and self.file_completer.popup().isVisible():
            # Select the current completion
            current_index = self.file_completer.popup().currentIndex()
            if current_index.isValid():
                completion = self.file_completer.completionModel().data(
                    current_index, Qt.ItemDataRole.DisplayRole
                )
                self.insert_completion(completion)
                self.file_completer.popup().hide()
                event.accept()
                return
        # Handle Enter key for completion
        elif (
            event.key() == Qt.Key.Key_Return and self.file_completer.popup().isVisible()
        ):
            # Select the current completion
            current_index = self.file_completer.popup().currentIndex()
            if current_index.isValid():
                completion = self.file_completer.completionModel().data(
                    current_index, Qt.ItemDataRole.DisplayRole
                )
                self.insert_completion(completion)
                self.file_completer.popup().hide()
                event.accept()
                return
        # Ctrl+Enter to send
        elif (
            event.key() == Qt.Key.Key_Return
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            self.send_message()
            event.accept()
            return
        # Up arrow to navigate history
        elif (
            event.key() == Qt.Key.Key_Up
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and not self.file_completer.popup().isVisible()
        ):
            self.history_navigate(-1)
            event.accept()
            return
        # Down arrow to navigate history
        elif (
            event.key() == Qt.Key.Key_Down
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and not self.file_completer.popup().isVisible()
        ):
            self.history_navigate(1)
            event.accept()
            return
        # Default behavior for other keys
        else:
            QTextEdit.keyPressEvent(self.message_input, event)

    def set_input_controls_enabled(self, enabled: bool):
        """Enable or disable input controls."""
        # Keep controls disabled if loading a conversation, regardless of 'enabled' argument
        actual_enabled = enabled and not self.loading_conversation

        self.message_input.setEnabled(actual_enabled)
        self.send_button.setEnabled(actual_enabled)
        self.file_button.setEnabled(actual_enabled)
        self.sidebar.setEnabled(actual_enabled)

        # Update cursor and appearance for visual feedback
        if actual_enabled:
            self.message_input.setFocus()
            self.send_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #4C662B; 
                    color: white; 
                    border: none;
                    border-radius: 4px; 
                    padding: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #354E16;
                }
                QPushButton:pressed {
                    background-color: #102000;
                }
                """
            )
            self.file_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #586249; 
                    color: white; 
                    border: none;
                    border-radius: 4px; 
                    padding: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #404A33;
                }
                QPushButton:pressed {
                    background-color: #2A331E;
                }
                """
            )
        else:
            # Use a different style if disabled due to loading vs. waiting for response
            if self.loading_conversation:
                self.send_button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #B1D18A; 
                        color: white; 
                        border: none;
                        border-radius: 4px; 
                        padding: 8px;
                        font-weight: bold;
                    }
                    """
                )
                self.file_button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #BFCBAD; 
                        color: white; 
                        border: none;
                        border-radius: 4px; 
                        padding: 8px;
                        font-weight: bold;
                    }
                    """
                )
            else:
                self.send_button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #CDEDA3; 
                        color: #354E16; 
                        border: none;
                        border-radius: 4px; 
                        padding: 8px;
                        font-weight: bold;
                    }
                    """
                )
                self.file_button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #DCE7C8; 
                        color: #404A33; 
                        border: none;
                        border-radius: 4px; 
                        padding: 8px;
                        font-weight: bold;
                    }
                    """
                )

        # Update waiting state (only relevant for LLM responses)
        if not self.loading_conversation:
            self.waiting_for_response = not enabled

    @Slot()
    def send_message(self):
        user_input = self.message_input.toPlainText().strip()  # Get text from QTextEdit
        if not user_input:  # Skip if empty
            return

        # Disable input controls while waiting for response
        self.set_input_controls_enabled(False)

        self.message_input.clear()

        # Process commands locally that don't need LLM processing
        if user_input.startswith("/"):
            # Clear command
            if user_input.startswith("/clear"):
                self.clear_chat()
                self.set_input_controls_enabled(True)  # Re-enable controls
                return
            # Copy command
            elif user_input.startswith("/copy"):
                self.copy_last_response()
                self.set_input_controls_enabled(True)  # Re-enable controls
                return
            # Debug command
            elif user_input.startswith("/debug"):
                self.display_debug_info()
                self.set_input_controls_enabled(True)  # Re-enable controls
                return
            # Exit command
            elif user_input in ["/exit", "/quit"]:
                QApplication.quit()
                return

        # Add user message to chat
        self.append_message(
            user_input, True, self.message_handler.current_user_input_idx
        )  # True = user message

        # Set flag to expect a response (for chunking)
        self.expecting_response = True
        self.current_response_bubble = None
        self.current_response_container = None

        # Update status bar
        self.display_status_message("Processing your message...")

        # Send the request to worker thread via signal
        # This is thread-safe and doesn't require QMetaObject.invokeMethod
        self.llm_worker.process_request.emit(user_input)

    def add_system_message(self, text):
        """Add a system message to the chat."""
        system_widget = SystemMessageWidget(text)
        self.chat_layout.addWidget(system_widget)

        # Scroll to show the new message
        QApplication.processEvents()
        self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        )

    def append_file(self, file_path, is_base64=False):
        # Create container for message alignment (similar to append_message)
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Create the message bubble
        message_bubble = MessageBubble(None, True, "You")

        # Add the file display to the message bubble
        if is_base64:
            message_bubble.display_base64_img(file_path)
        else:
            message_bubble.display_file(file_path)

        container_layout.addWidget(message_bubble)
        container_layout.addStretch(1)  # Push to left

        # Add the container to the chat layout
        self.chat_layout.addWidget(container)

        # Process events and scroll to show the new message
        QApplication.processEvents()
        self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        )

    def append_message(self, text, is_user=True, message_index=None, agent_name=None):
        """Adds a message bubble to the chat container."""
        # Create container for message alignment
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Create the message bubble with agent name for non-user messages
        agent_name = (
            agent_name
            if agent_name
            else self.message_handler.agent_name
            if not is_user
            else "YOU"
        )

        message_bubble = MessageBubble(
            text, is_user, agent_name, message_index=message_index
        )

        # Add bubble to container with appropriate alignment
        if message_bubble.rollback_button:
            message_bubble.rollback_button.clicked.connect(
                lambda: self.rollback_to_message(message_bubble)
            )
        if is_user:
            container_layout.addWidget(message_bubble)
            container_layout.addStretch(1)  # Push to left
        else:
            container_layout.addStretch(1)  # Push to right
            container_layout.addWidget(message_bubble)

        # Add the container to the chat layout
        self.chat_layout.addWidget(container)

        # If this is an assistant message, store references for potential future chunks
        if not is_user:
            self.current_response_bubble = message_bubble
            self.current_response_container = container
        else:
            self.current_user_bubble = message_bubble

        # Process events to ensure UI updates immediately
        QApplication.processEvents()

        # Scroll to the bottom to show new message
        self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        )

        return message_bubble

    @Slot(str, int, int)
    def handle_response(self, response, input_tokens, output_tokens):
        """Handle the full response from the LLM worker"""
        # self.display_response_chunk(response)

        # Calculate cost
        total_cost = self.message_handler.llm.calculate_cost(
            input_tokens, output_tokens
        )

        # Update token usage
        self.update_token_usage(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": total_cost,
            }
        )

        # Re-enable input controls
        self.set_input_controls_enabled(True)

    @Slot(str)
    def display_response_chunk(self, chunk: str):
        """Display a response chunk from the assistant."""

        # If we're expecting a response and don't have a bubble yet, create one
        if self.expecting_response and self.current_response_bubble is None:
            self.current_response_bubble = self.append_message(
                chunk, False
            )  # False = assistant message
        # If we already have a response bubble, append to it
        elif self.expecting_response and self.current_response_bubble is not None:
            self.current_response_bubble.append_text(chunk)
            # Force update and scroll
            QApplication.processEvents()
            self.chat_scroll.verticalScrollBar().setValue(
                self.chat_scroll.verticalScrollBar().maximum()
            )
        # Otherwise, create a new message (should not happen in normal operation)
        else:
            self.current_response_bubble = self.append_message(chunk, False)

    @Slot(str)
    def display_error(self, error):
        """Display an error message.

        Args:
            error: Either a string error message or a dictionary with error details
        """
        # Handle both string and dictionary error formats
        if isinstance(error, dict):
            # Extract error message from dictionary
            error_message = error.get("message", str(error))
        else:
            error_message = str(error)

        QMessageBox.critical(self, "Error", error_message)
        self.status_bar.showMessage(
            f"Error: {error_message}", 5000
        )  # Display error in status bar
        self.expecting_response = False

        # Re-enable input controls
        self.set_input_controls_enabled(True)

    @Slot(str)
    def display_status_message(self, message):
        self.status_bar.showMessage(message, 5000)

    @Slot(dict)
    def update_token_usage(self, usage_data):
        """Update token usage display."""
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        total_cost = usage_data.get("total_cost", 0.0)

        # Update session cost
        self.session_cost += total_cost

        # Update the token usage widget
        self.token_usage.update_token_info(
            input_tokens, output_tokens, total_cost, self.session_cost
        )

        # Reset response expectation
        self.expecting_response = False

        # Re-enable input controls
        self.set_input_controls_enabled(True)

    @Slot()
    def copy_last_response(self):
        """Copy the last assistant response to clipboard."""
        text = self.message_handler.latest_assistant_response
        if text:
            pyperclip.copy(text)
            self.status_bar.showMessage("Last response copied to clipboard!", 3000)
        else:
            self.status_bar.showMessage("No response to copy", 3000)

    @Slot()
    def handle_exit_request(self):
        """Handle exit request from worker thread"""
        QApplication.quit()

    @Slot()
    def handle_clear_request(self):
        """Handle clear request from worker thread"""
        self.clear_chat(True)

    @Slot()
    def clear_chat(self, requested=False):
        """Clear the chat history and UI."""
        # Only ask for confirmation if triggered by user (e.g., Ctrl+L), not programmatically
        if not requested:
            reply = QMessageBox.question(
                self,
                "Clear Chat",
                "Are you sure you want to start new conversation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return  # User cancelled

        # Clear the UI immediately
        self._clear_chat_ui()

        # Reset session cost display
        self.session_cost = 0.0
        self.token_usage.update_token_info(0, 0, 0.0, 0.0)

        # If the clear was initiated by the user (not loading a conversation),
        # tell the message handler to clear its state.
        if not requested:
            self.llm_worker.process_request.emit("/clear")
            # Add a confirmation message after clearing
            self.add_system_message("Chat history cleared.")
            self.display_status_message("Chat history cleared")

        # Ensure input controls are enabled after clearing
        self.set_input_controls_enabled(True)
        self.loading_conversation = False  # Ensure loading flag is reset

    def _clear_chat_ui(self):
        """Clears only the chat message widgets from the UI."""
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        # Reset tracking variables related to response streaming
        self.current_response_bubble = None
        self.current_response_container = None
        self.current_thinking_bubble = None
        self.thinking_content = ""
        self.expecting_response = False

    def history_navigate(self, direction):
        if not self.message_handler.history_manager.history:
            return

        new_position = self.history_position + direction

        if 0 <= new_position < len(self.message_handler.history_manager.history):
            self.history_position = new_position
            history_entry = self.message_handler.history_manager.history[
                self.history_position
            ]
            self.message_input.setText(history_entry)  # Set text in input
        elif new_position < 0:
            self.history_position = -1
            self.message_input.clear()
        elif new_position >= len(self.message_handler.history_manager.history):
            self.history_position = len(self.message_handler.history_manager.history)
            self.message_input.clear()

    def display_tool_use(self, tool_use: Dict):
        """Display information about a tool being used."""
        tool_message = f"TOOL: Using {tool_use['name']}\n\n```\n{tool_use}\n```"
        self.add_system_message(tool_message)
        self.display_status_message(f"Using tool: {tool_use['name']}")

    def display_tool_result(self, data: Dict):
        """Display the result of a tool execution."""
        tool_use = data["tool_use"]
        tool_result = data["tool_result"]
        result_message = f"RESULT: Tool {tool_use['name']}:\n\n```\n{tool_result}\n```"
        self.add_system_message(result_message)

        # Reset the current response bubble so the next agent message starts in a new bubble
        self.current_response_bubble = None
        self.current_response_container = None

    def display_tool_error(self, data: Dict):
        """Display an error that occurred during tool execution."""
        tool_use = data["tool_use"]
        error = data["error"]
        error_message = f"ERROR: Tool {tool_use['name']}: {error}"
        self.add_system_message(error_message)
        self.display_status_message(f"Error in tool {tool_use['name']}")

        # Reset the current response bubble so the next agent message starts in a new bubble
        self.current_response_bubble = None
        self.current_response_container = None

    def browse_file(self):
        """Open file dialog and process selected file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "All Files (*);;Text Files (*.txt);;PDF Files (*.pdf);;Word Files (*.docx)",
        )

        if file_path:
            # Disable input controls while processing file
            self.set_input_controls_enabled(False)

            # Process the file using the /file command
            file_command = f"/file {file_path}"
            self.display_status_message(f"Processing file: {file_path}")

            # Send the file command to the worker thread
            self.llm_worker.process_request.emit(file_command)

    def show_context_menu(self, position):
        """Show context menu with options."""
        context_menu = QMenu(self)

        # Add menu actions
        copy_action = context_menu.addAction("Copy Last Response")
        clear_action = context_menu.addAction("Clear Chat")

        # Connect actions to slots
        copy_action.triggered.connect(self.copy_last_response)
        clear_action.triggered.connect(self.clear_chat)

        # Show the menu at the cursor position
        context_menu.exec(self.mapToGlobal(position))

    def rollback_to_message(self, message_bubble):
        """Rollback the conversation to the selected message."""
        if message_bubble.message_index is None:
            self.display_status_message("Cannot rollback: no message index available")
            return

        current_text = message_bubble.message_label.text()

        # Find the turn number for this message
        # We need to find which conversation turn corresponds to this message
        turn_number = None

        for i, turn in enumerate(self.message_handler.conversation_turns):
            print(turn.message_index)
            print(message_bubble.message_index)
            if turn.message_index == message_bubble.message_index:
                turn_number = i + 1  # Turn numbers are 1-indexed
                break

        if turn_number is None:
            self.display_status_message(
                "Cannot rollback: message not found in conversation history"
            )
            return

        # Execute the jump command
        self.llm_worker.process_request.emit(f"/jump {turn_number}")

        # Find and remove all widgets after this message in the UI
        self.remove_messages_after(message_bubble)
        self.message_input.setText(current_text)

    def remove_messages_after(self, message_bubble):
        """Remove all message widgets that appear after the given message bubble, including the message bubble itself."""
        # Find the index of the container widget that holds the message bubble
        container_index = -1
        for i in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(i)
            if item and item.widget():
                # Check if this widget contains our message bubble
                if message_bubble in item.widget().findChildren(MessageBubble):
                    container_index = i
                    break

        if container_index == -1:
            return  # Message bubble not found

        # Remove the container with the message bubble and all widgets after it
        while self.chat_layout.count() > container_index:
            item = self.chat_layout.takeAt(container_index)
            if item.widget():
                item.widget().deleteLater()

        # Reset current response tracking
        self.current_response_bubble = None
        self.current_response_container = None
        self.expecting_response = False

    @Slot(str)
    def load_conversation(self, conversation_id):
        """Initiates loading a conversation asynchronously."""
        if self.loading_conversation:
            self.display_status_message("Already loading a conversation.")
            return

        self.loading_conversation = True
        self.set_input_controls_enabled(False)  # Disable input during load
        self.display_status_message(f"Loading conversation: {conversation_id}...")

        # Use the ConversationLoader thread
        self.loader_thread = ConversationLoader(self.message_handler, conversation_id)
        self.loader_thread.loaded.connect(self.display_conversation)
        self.loader_thread.error.connect(self.handle_load_error)
        # Clean up thread when finished
        self.loader_thread.finished.connect(self.loader_thread.deleteLater)
        self.loader_thread.start()

    @Slot(list, str)
    def display_conversation(self, messages, conversation_id):
        """Displays the loaded conversation messages in the UI."""
        self._clear_chat_ui()  # Clear existing messages first

        # Reset session cost when loading a new conversation
        self.session_cost = 0.0
        self.token_usage.update_token_info(0, 0, 0.0, 0.0)

        # Add messages from the loaded conversation, filtering for user/assistant roles
        msg_idx = 0
        for msg in messages:
            role = msg.get("role")
            if role == "user" or role == "assistant":
                content = msg.get("content", "")
                message_content = ""
                is_user = role == "user"

                # Handle different content structures (standardized format)
                if isinstance(content, str):
                    message_content = content
                    # self.append_message(content, is_user=is_user)
                elif isinstance(content, list) and content:
                    # Assuming the first item in the list contains the primary text
                    first_item = content[0]
                    if (
                        isinstance(first_item, dict)
                        and first_item.get("type") == "text"
                    ):
                        message_content = first_item.get("text", "")
                    elif (
                        isinstance(first_item, dict)
                        and first_item.get("type") == "image_url"
                    ):
                        self.append_file(
                            first_item.get("image_url", {}).get("url", ""), True
                        )
                        msg_idx += 1
                        continue
                        # self.append_message(first_item.get("text", ""), is_user=is_user)
                    # Add more specific handling here if other content types need display

                ## Striped out the user context summary
                message_content = re.sub(
                    r"(?:```(?:json)?)?\s*<user_context_summary>.*?</user_context_summary>\s*(?:```)?",
                    "",
                    message_content,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if (
                    message_content.strip()
                    and not message_content.startswith(
                        "Use user_context_summary to tailor your response"
                    )
                    and not message_content.startswith("Content of ")
                    and not message_content.startswith(
                        "Need to tailor response bases on this"
                    )
                ):
                    self.append_message(
                        message_content,
                        is_user,
                        msg_idx if is_user else None,
                        msg.get("agent", None),
                    )
                # Add handling for other potential content formats if necessary
            msg_idx += 1

        # Update status bar and re-enable controls
        self.display_status_message(f"Loaded conversation: {conversation_id}")
        self.loading_conversation = False
        self.set_input_controls_enabled(True)
        # Optionally, update the agent/model indicator if that info is part of the loaded convo metadata
        # self.status_indicator.setText(...)

        # Refresh sidebar in case title/timestamp changed (optional)
        # self.sidebar.update_conversation_list()

    @Slot(str)
    def handle_load_error(self, error_message):
        """Handles errors during conversation loading."""
        self.display_error(error_message)
        self.loading_conversation = False
        self.set_input_controls_enabled(True)  # Re-enable controls on error

    @Slot()
    def start_new_conversation(self):
        """Start a new conversation by clearing the current one."""
        # Check if there are unsaved changes or ongoing operations
        if self.waiting_for_response:
            QMessageBox.warning(
                self,
                "Operation in Progress",
                "Please wait for the current operation to complete before starting a new conversation.",
            )
            return

        self.clear_chat()

    def check_for_path_completion(self):
        """Check if the current text contains a path that should trigger completion."""
        text = self.message_input.toPlainText()
        cursor_position = self.message_input.textCursor().position()

        # Get the text up to the cursor position
        text_to_cursor = text[:cursor_position]

        # Look for path patterns that should trigger completion
        path_match = re.search(r"((~|\.{1,2})?/[^\s]*|~)$", text_to_cursor)

        if path_match:
            path = path_match.group(0)
            completions = self.directory_completer.get_path_completions(path)

            if completions:
                # Create a model for the completer
                model = QStringListModel(completions)
                self.file_completer.setModel(model)

                # Calculate the prefix length to determine what part to complete
                prefix = os.path.basename(path) if "/" in path else path
                self.file_completer.setCompletionPrefix(prefix)

                # Store the path prefix (everything before the basename)
                self.path_prefix = path[: len(path) - len(prefix)]

                # Show the completion popup
                popup = self.file_completer.popup()
                popup.setCurrentIndex(self.file_completer.completionModel().index(0, 0))

                # Calculate position for the popup
                rect = self.message_input.cursorRect()
                rect.setWidth(300)  # Set a reasonable width for the popup

                # Show the popup
                self.file_completer.complete(rect)
            else:
                # Hide the popup if no completions
                self.file_completer.popup().hide()

    def insert_completion(self, completion):
        """Insert the selected completion into the text input."""
        cursor = self.message_input.textCursor()
        text = self.message_input.toPlainText()
        position = cursor.position()

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

            # If the completion is a directory, add a trailing slash
            # full_path = os.path.join(os.path.expanduser(self.path_prefix), completion)
            cursor.insertText(completion)

    def create_menu_bar(self):
        """Create the application menu bar with Agents, Models, and Settings menus"""
        menu_bar = QMenuBar(self)
        menu_bar.setStyleSheet(
            """
            QMenuBar {
                background-color: #4C662B;
                color: white;
                padding: 2px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #354E16;
            }
            QMenu {
                background-color: white;
                border: 1px solid #C5C8BA;
            }
            QMenu::item {
                padding: 6px 24px 6px 12px;
            }
            QMenu::item:selected {
                background-color: #CDEDA3;
                color: #354E16;
            }
            """
        )
        self.setMenuBar(menu_bar)

        # Create Agents menu
        agents_menu = menu_bar.addMenu("Agents")

        # Get agent manager instance
        agent_manager = AgentManager.get_instance()

        # Get available agents
        available_agents = agent_manager.agents

        # Add agent options to menu
        for agent_name in available_agents:
            agent_action = QAction(agent_name, self)
            agent_action.triggered.connect(
                lambda checked, name=agent_name: self.change_agent(name)
            )
            agents_menu.addAction(agent_action)

        # Create Models menu
        models_menu = menu_bar.addMenu("Models")

        # Get model registry instance
        model_registry = ModelRegistry.get_instance()

        # Add provider submenus
        for provider in ["claude", "openai", "groq", "google", "deepinfra"]:
            provider_menu = models_menu.addMenu(provider.capitalize())

            # Get models for this provider
            models = model_registry.get_models_by_provider(provider)

            # Add model options to submenu
            for model in models:
                model_action = QAction(f"{model.name} ({model.id})", self)
                model_action.triggered.connect(
                    lambda checked, model_id=model.id: self.change_model(model_id)
                )
                provider_menu.addAction(model_action)

        # Create Settings menu
        settings_menu = menu_bar.addMenu("Settings")

        # Add Agents configuration option
        agents_config_action = QAction("Agents", self)
        agents_config_action.triggered.connect(self.open_agents_config)
        settings_menu.addAction(agents_config_action)

        # Add MCPs configuration option
        mcps_config_action = QAction("MCP Servers", self)
        mcps_config_action.triggered.connect(self.open_mcps_config)
        settings_menu.addAction(mcps_config_action)

    def change_agent(self, agent_name):
        """Change the current agent"""
        # Process the agent change command
        self.llm_worker.process_request.emit(f"/agent {agent_name}")

    def change_model(self, model_id):
        """Change the current model"""
        # Process the model change command
        self.llm_worker.process_request.emit(f"/model {model_id}")

    def open_agents_config(self):
        """Open the agents configuration window."""
        from swissknife.modules.gui.widgets.config_window import ConfigWindow

        config_window = ConfigWindow(self)
        config_window.tab_widget.setCurrentIndex(0)  # Show Agents tab
        config_window.exec()

        # Refresh agent list in case changes were made
        self.refresh_agent_menu()

    def open_mcps_config(self):
        """Open the MCP servers configuration window."""
        from swissknife.modules.gui.widgets.config_window import ConfigWindow

        config_window = ConfigWindow(self)
        config_window.tab_widget.setCurrentIndex(1)  # Show MCPs tab
        config_window.exec()

    def refresh_agent_menu(self):
        """Refresh the agents menu after configuration changes."""
        # Get the menu bar
        menu_bar = self.menuBar()

        # Find the Agents menu
        agents_menu = None
        for action in menu_bar.actions():
            if action.text() == "Agents":
                agents_menu = action.menu()
                break

        if agents_menu:
            # Clear existing actions
            agents_menu.clear()

            # Get agent manager instance
            agent_manager = AgentManager.get_instance()

            # Get available agents
            available_agents = agent_manager.agents

            # Add agent options to menu
            for agent_name in available_agents:
                agent_action = QAction(agent_name, self)
                agent_action.triggered.connect(
                    lambda checked, name=agent_name: self.change_agent(name)
                )
                agents_menu.addAction(agent_action)

    def display_debug_info(self):
        """Display debug information about the current messages."""
        import json

        try:
            # Format the messages for display
            debug_info = json.dumps(self.message_handler.messages, indent=2)
        except Exception as e:
            debug_info = str(self.message_handler.messages)
        # Add as a system message
        self.add_system_message(f"DEBUG INFO:\n\n```json\n{debug_info}\n```")

        # Update status bar
        self.display_status_message("Debug information displayed")

    def listen(self, event: str, data: Any = None):
        """Handle events from the message handler."""
        # Use a signal to ensure thread-safety
        self.event_received.emit(event, data)

    def display_thinking_started(self, agent_name: str):
        """Display the start of the thinking process."""
        self.add_system_message(f"💭 {agent_name.upper()}'s thinking process started")

        # Create a new thinking bubble
        self.current_thinking_bubble = self.append_thinking_message(" ", agent_name)
        self.thinking_content = ""  # Initialize thinking content

    def display_thinking_chunk(self, chunk: str):
        """Display a chunk of the thinking process."""
        if hasattr(self, "current_thinking_bubble") and self.current_thinking_bubble:
            # Append to the thinking content
            self.thinking_content += chunk
            self.current_thinking_bubble.append_text(chunk)

            # Force update and scroll
            QApplication.processEvents()
            self.chat_scroll.verticalScrollBar().setValue(
                self.chat_scroll.verticalScrollBar().maximum()
            )

    def append_thinking_message(self, text, agent_name):
        """Adds a thinking message bubble to the chat container."""
        # Create container for message alignment
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Create the message bubble with agent name and thinking flag
        message_bubble = MessageBubble(text, False, agent_name, is_thinking=True)

        # Add bubble to container with appropriate alignment (same as assistant messages)
        container_layout.addStretch(1)  # Push to right
        container_layout.addWidget(message_bubble)

        # Add the container to the chat layout
        self.chat_layout.addWidget(container)

        # Process events to ensure UI updates immediately
        QApplication.processEvents()

        # Scroll to the bottom to show new message
        self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        )

        return message_bubble

    @Slot(str, object)
    def handle_event(self, event: str, data: Any):
        if event == "response_chunk":
            chunk, assistant_response = data
            if assistant_response.strip():
                self.display_response_chunk(assistant_response)
        elif event == "error":
            # If an error occurs during LLM processing, ensure loading flag is false
            self.loading_conversation = False
            self.set_input_controls_enabled(True)
            self.display_error(data)
        elif event == "user_message_created":
            if self.current_user_bubble:
                self.current_user_bubble.message_index = (
                    self.message_handler.current_user_input_idx
                )
                self.current_user_bubble = None
        elif event == "thinking_started":
            self.display_thinking_started(data)  # data is agent_name
        elif event == "thinking_chunk":
            self.display_thinking_chunk(data)  # data is the thinking chunk
        elif event == "thinking_completed":
            self.display_status_message("Thinking completed.")
            # Reset thinking bubble reference
            self.current_thinking_bubble = None
        elif event == "clear_requested":
            # This is likely triggered by the worker after /clear command
            # The UI clear might have already happened if user initiated via Ctrl+L
            # Ensure UI is clear and state is reset.
            self._clear_chat_ui()
            self.session_cost = 0.0
            self.token_usage.update_token_info(0, 0, 0.0, 0.0)
            self.add_system_message("Chat history cleared by command.")
            self.loading_conversation = False
            self.set_input_controls_enabled(True)
            # Refresh sidebar as the current conversation is gone
            self.sidebar.update_conversation_list()
        elif event == "exit_requested":
            QApplication.quit()
        elif event == "copy_requested":
            if isinstance(data, str):
                pyperclip.copy(data)
                self.display_status_message("Text copied to clipboard!")
        elif event == "debug_requested":
            # Format the debug data and display it
            import json

            try:
                debug_info = json.dumps(data, indent=2)
                self.add_system_message(f"DEBUG INFO:\n\n```json\n{debug_info}\n```")
            except Exception:
                # Fallback for non-JSON serializable data
                self.add_system_message(f"DEBUG INFO:\n\n{str(data)}")
        elif event == "file_processed":
            # Create a message bubble for the file
            file_path = data["file_path"]
            self.append_file(file_path)
            # Re-enable controls only if not loading a conversation
            if not self.loading_conversation:
                self.set_input_controls_enabled(True)
        elif event == "tool_use":
            self.display_tool_use(data)
        elif event == "tool_result":
            self.display_tool_result(data)
        elif event == "tool_error":
            self.display_tool_error(data)
        elif event == "jump_performed":
            self.add_system_message(
                f"🕰️ Jumped to turn {data['turn_number']}: {data['preview']}"
            )
        elif event == "agent_changed":
            self.add_system_message(f"Switched to {data} agent")
            self.status_indicator.setText(
                f"Agent: {data} | Model: {self.message_handler.llm.model}"
            )
        elif event == "model_changed":
            self.add_system_message(f"Switched to {data['name']} ({data['id']})")
            self.status_indicator.setText(
                f"Agent: {self.message_handler.agent_name} | Model: {data['id']}"
            )
        elif event == "agent_changed_by_transfer":
            self.add_system_message(f"Transfered to {data} agent")
            self.status_indicator.setText(
                f"Agent: {data} | Model: {self.message_handler.llm.model}"
            )
        elif event == "think_budget_set":
            self.add_system_message(f"Set thinking budget at {data}")
            self.set_input_controls_enabled(True)
        elif event == "conversation_saved":  # Add handler for this event
            self.display_status_message(f"Conversation saved: {data.get('id', 'N/A')}")
            self.sidebar.update_conversation_list()  # Refresh sidebar
        elif (
            event == "conversation_loaded"
        ):  # Add handler for this event (if loaded via command)
            self.display_status_message(f"Conversation loaded: {data.get('id', 'N/A')}")

        # --- Ensure controls are re-enabled after most events if not loading ---
        # Place this check strategically if needed, or rely on specific event handlers
        # if not self.loading_conversation and not self.waiting_for_response:
        #    self.set_input_controls_enabled(True)
