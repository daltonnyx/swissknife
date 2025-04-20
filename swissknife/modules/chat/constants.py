# Terminal formatting
COLORS_ENABLED = True  # Set to False to disable colors
BLUE = "\033[94m" if COLORS_ENABLED else ""
GREEN = "\033[92m" if COLORS_ENABLED else ""
YELLOW = "\033[93m" if COLORS_ENABLED else ""
GRAY = "\033[90m" if COLORS_ENABLED else ""
RESET = "\033[0m" if COLORS_ENABLED else ""
BOLD = "\033[1m" if COLORS_ENABLED else ""
RED = "\033[91m" if COLORS_ENABLED else ""
RICH_YELLOW = "yellow"
RICH_GRAY = "grey66"
# History settings
DEFAULT_HISTORY_FILE = "~/.chat_histories"
DEFAULT_HISTORY_LIMIT = 1000
HISTORY_ENTRY_DELIMITER = "\n---ENTRY---\n"

OBSERVABLE_EVENTS = {}
