import traceback
from swissknife.modules.chat.message_handler import MessageHandler

from PySide6.QtCore import (
    Slot,
    QObject,
    Signal,
)


class LLMWorker(QObject):
    """Worker object that processes LLM requests in a separate thread"""

    # Signals for thread communication
    response_ready = Signal(str, int, int)  # response, input_tokens, output_tokens
    response_chunk = Signal(str)
    error = Signal(str)
    status_message = Signal(str)
    token_usage = Signal(dict)
    request_exit = Signal()
    request_clear = Signal()
    thinking_started = Signal(str)  # agent_name
    thinking_chunk = Signal(str)  # thinking text chunk
    thinking_completed = Signal()

    # Signal to request processing - passing the user input as a string
    process_request = Signal(str)

    def __init__(self):
        super().__init__()
        self.user_input = None
        self.message_handler = None  # Will be set in connect_handler

    def connect_handler(self, message_handler: MessageHandler):
        """Connect to the message handler - called from main thread before processing begins"""
        self.message_handler = message_handler
        # Connect the process_request signal to our processing slot
        self.process_request.connect(self.process_input)

    @Slot(str)
    def process_input(self, user_input):
        """Process the user input with the message handler"""
        try:
            if not self.message_handler:
                self.error.emit("Message handler not connected")
                return

            if not user_input:
                return

            # Process user input (commands, etc.)
            exit_flag, clear_flag = self.message_handler.process_user_input(user_input)

            # Handle command results
            if exit_flag:
                self.status_message.emit("Exiting...")
                self.request_exit.emit()
                return

            if clear_flag:
                # self.request_clear.emit()
                return  # Skip further processing if chat was cleared

            # Get assistant response
            assistant_response, input_tokens, output_tokens = (
                self.message_handler.get_assistant_response()
            )

            # Emit the response
            if assistant_response:
                self.response_ready.emit(
                    assistant_response, input_tokens, output_tokens
                )

                # Calculate cost
                total_cost = self.message_handler.agent.llm.calculate_cost(
                    input_tokens, output_tokens
                )

                # Emit token usage information
                self.token_usage.emit(
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_cost": total_cost,
                    }
                )
            else:
                print("No response received from assistant")
                self.status_message.emit("No response received")

        except Exception as e:
            traceback_str = traceback.format_exc()
            error_msg = f"{str(e)}\n{traceback_str}"
            print(f"Error in LLMWorker: {error_msg}")  # Console debug
            self.error.emit(error_msg)
