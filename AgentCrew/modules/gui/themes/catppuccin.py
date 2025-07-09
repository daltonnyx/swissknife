"""Catppuccin theme styles for AgentCrew GUI."""


class CatppuccinTheme:
    """Static class containing Catppuccin theme styles."""

    # Main application styles
    MAIN_STYLE = """
QMainWindow {
    background-color: #1e1e2e; /* Catppuccin Base */
}
QScrollArea {
    border: none;
    background-color: #181825; /* Catppuccin Mantle */
}
QWidget#chatContainer { /* Specific ID for chat_container */
    background-color: #1e1e2e; /* Catppuccin Mantle */
}
QSplitter::handle {
    background-color: #313244; /* Catppuccin Surface0 */
}
QSplitter::handle:hover {
    background-color: #45475a; /* Catppuccin Surface1 */
}
QSplitter::handle:pressed {
    background-color: #585b70; /* Catppuccin Surface2 */
}
QStatusBar {
    background-color: #11111b; /* Catppuccin Crust */
    color: #cdd6f4; /* Catppuccin Text */
}
QToolTip {
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    padding: 4px;
}
QMessageBox {
    background-color: #181825; /* Catppuccin Mantle */
}
QMessageBox QLabel { /* For message text in QMessageBox */
    color: #cdd6f4; /* Catppuccin Text */
    background-color: transparent; /* Ensure no overriding background */
}
/* QCompleter's popup is often a QListView */
QListView { /* General style for QListView, affects completer */
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    padding: 2px;
    outline: 0px; /* Remove focus outline if not desired */
}
QListView::item {
    padding: 4px 8px;
    border-radius: 2px; /* Optional: rounded corners for items */
}
QListView::item:selected {
    background-color: #585b70; /* Catppuccin Surface2 */
    color: #b4befe; /* Catppuccin Lavender */
}
QListView::item:hover {
    background-color: #45475a; /* Catppuccin Surface1 */
}

/* Modern Scrollbar Styles */
QScrollBar:vertical {
    border: none;
    background: #181825; /* Catppuccin Mantle - Track background */
    width: 10px; /* Adjust width for a thinner scrollbar */
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #45475a; /* Catppuccin Surface1 - Handle color */
    min-height: 20px; /* Minimum handle size */
    border-radius: 5px; /* Rounded corners for the handle */
}
QScrollBar::handle:vertical:hover {
    background: #585b70; /* Catppuccin Surface2 - Handle hover color */
}
QScrollBar::handle:vertical:pressed {
    background: #6c7086; /* Catppuccin Overlay0 - Handle pressed color */
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none; /* Hide arrow buttons */
    height: 0px;
    subcontrol-position: top;
    subcontrol-origin: margin;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none; /* Track area above/below handle */
}

QScrollBar:horizontal {
    border: none;
    background: #181825; /* Catppuccin Mantle - Track background */
    height: 10px; /* Adjust height for a thinner scrollbar */
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:horizontal {
    background: #45475a; /* Catppuccin Surface1 - Handle color */
    min-width: 20px; /* Minimum handle size */
    border-radius: 5px; /* Rounded corners for the handle */
}
QScrollBar::handle:horizontal:hover {
    background: #585b70; /* Catppuccin Surface2 - Handle hover color */
}
QScrollBar::handle:horizontal:pressed {
    background: #6c7086; /* Catppuccin Overlay0 - Handle pressed color */
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    border: none;
    background: none; /* Hide arrow buttons */
    width: 0px;
    subcontrol-position: left;
    subcontrol-origin: margin;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none; /* Track area left/right of handle */
}

/* Context menu styling for QLabel widgets */
QLabel QMenu {
    background-color: #181825; /* Catppuccin Mantle */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    padding: 4px;
    border-radius: 6px;
}
QLabel QMenu::item {
    color: #f8f8f2; /* Brighter text color */
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    margin: 2px;
}
QLabel QMenu::item:selected {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #ffffff; /* Pure white for selected items */
}
QLabel QMenu::item:pressed {
    background-color: #585b70; /* Catppuccin Surface2 */
}
QLabel QMenu::separator {
    height: 1px;
    background: #45475a; /* Catppuccin Surface1 */
    margin: 4px 8px;
}
"""

    # Button styles
    PRIMARY_BUTTON = """
QPushButton {
    background-color: #89b4fa; /* Catppuccin Blue */
    color: #1e1e2e; /* Catppuccin Base (for contrast) */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #74c7ec; /* Catppuccin Sapphire */
}
QPushButton:pressed {
    background-color: #b4befe; /* Catppuccin Lavender */
}
QPushButton:disabled {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #6c7086; /* Catppuccin Overlay0 */
}
"""

    SECONDARY_BUTTON = """
QPushButton {
    background-color: #585b70; /* Catppuccin Surface2 */
    color: #cdd6f4; /* Catppuccin Text */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #6c7086; /* Catppuccin Overlay0 */
}
QPushButton:pressed {
    background-color: #7f849c; /* Catppuccin Overlay1 */
}
QPushButton:disabled {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #6c7086; /* Catppuccin Overlay0 */
}
"""

    STOP_BUTTON = """
QPushButton {
    background-color: #f38ba8; /* Catppuccin Red */
    color: #1e1e2e; /* Catppuccin Base (for contrast) */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #eba0ac; /* Catppuccin Maroon */
}
QPushButton:pressed {
    background-color: #f5c2e7; /* Catppuccin Pink */
}
"""

    RED_BUTTON = """
QPushButton {
    background-color: #f38ba8; /* Catppuccin Red */
    color: #1e1e2e; /* Catppuccin Base (for contrast) */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #eba0ac; /* Catppuccin Maroon (lighter red for hover) */
}
QPushButton:pressed {
    background-color: #e67e8a; /* A slightly darker/more intense red for pressed */
}
QPushButton:disabled {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #6c7086; /* Catppuccin Overlay0 */
}
"""

    GREEN_BUTTON = """
QPushButton {
    background-color: #a6e3a1; /* Catppuccin Green */
    color: #1e1e2e; /* Catppuccin Base */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #94e2d5; /* Catppuccin Teal - lighter green for hover */
}
QPushButton:pressed {
    background-color: #8bd5ca; /* Slightly darker for pressed state */
}
QPushButton:disabled {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #6c7086; /* Catppuccin Overlay0 */
}
"""

    API_KEYS_GROUP = """
QGroupBox {
    background-color: #1e1e2e; /* Catppuccin Base */
}
QLabel {
    background-color: #1e1e2e; 
}
"""

    EDITOR_CONTAINER_WIDGET = """
QWidget {
    background-color: #181825; /* Catppuccin Mantle */
}
"""

    MENU_BUTTON = """
QPushButton {
    background-color: #89b4fa; /* Catppuccin Blue */
    color: #1e1e2e; /* Catppuccin Base */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
    padding-left: 12px; /* Add some padding for text */
}
QPushButton:hover {
    background-color: #74c7ec; /* Catppuccin Sapphire */
}
QPushButton:pressed {
    background-color: #b4befe; /* Catppuccin Lavender */
}
QPushButton:disabled {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #6c7086; /* Catppuccin Overlay0 */
}
QPushButton::menu-indicator {
    /* image: url(myindicator.png); Adjust if using a custom image */
    subcontrol-origin: padding;
    subcontrol-position: right center;
    right: 5px; /* Adjust as needed to position from the right edge */
    width: 16px; /* Ensure there's enough space for the indicator */
}
"""

    DISABLED_BUTTON = """
QPushButton {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #6c7086; /* Catppuccin Overlay0 */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
"""

    STOP_BUTTON_STOPPING = """
QPushButton {
    background-color: #6c7086; /* Catppuccin Overlay0 - More grey/disabled look */
    color: #9399b2; /* Catppuccin Subtext1 - Muted text */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
"""

    COMBO_BOX = """
QComboBox {
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    border-radius: 4px;
    padding: 6px 8px;
    min-width: 120px;
}
QComboBox:hover {
    border: 1px solid #585b70; /* Catppuccin Surface2 */
}
QComboBox:focus {
    border: 1px solid #89b4fa; /* Catppuccin Blue */
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: #45475a; /* Catppuccin Surface1 */
    border-left-style: solid;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
    background-color: #45475a; /* Catppuccin Surface1 */
}
QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #cdd6f4; /* Catppuccin Text */
}
QComboBox QAbstractItemView {
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    border-radius: 4px;
    selection-background-color: #89b4fa; /* Catppuccin Blue */
    selection-color: #1e1e2e; /* Catppuccin Base */
    outline: 0px;
}
"""

    # Input styles
    TEXT_INPUT = """
QTextEdit {
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    border-radius: 4px;
    padding: 8px;
}
QTextEdit:focus {
    border: 1px solid #89b4fa; /* Catppuccin Blue */
}
"""

    LINE_EDIT = """
QLineEdit {
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    border-radius: 4px;
    padding: 8px;
}
QLineEdit:focus {
    border: 1px solid #89b4fa; /* Catppuccin Blue */
}
"""

    # Menu styles
    MENU_BAR = """
QMenuBar {
    background-color: #181825; /* Catppuccin Base */
    color: #cdd6f4; /* Catppuccin Text */
    padding: 2px;
}
QMenuBar::item {
    background-color: transparent;
    color: #cdd6f4; /* Catppuccin Text */
    padding: 4px 12px;
    border-radius: 4px;
}
QMenuBar::item:selected { /* When menu is open or item is hovered */
    background-color: #313244; /* Catppuccin Surface0 */
}
QMenuBar::item:pressed { /* When menu item is pressed to open the menu */
    background-color: #45475a; /* Catppuccin Surface1 */
}
QMenu {
    background-color: #181825; /* Catppuccin Mantle */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 4px; /* Add border-radius to menu items */
}
QMenu::item:selected {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #b4befe; /* Catppuccin Lavender */
}
QMenu::separator {
    height: 1px;
    background: #45475a; /* Catppuccin Surface1 */
    margin-left: 10px;
    margin-right: 5px;
}
"""

    CONTEXT_MENU = """
QMenu {
    background-color: #181825; /* Catppuccin Mantle */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    padding: 4px;
    border-radius: 6px;
}
QMenu::item {
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    margin: 2px;
    color: #f8f8f2; /* Brighter text color for better visibility */
}
QMenu::item:selected {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #ffffff; /* Pure white for selected items */
}
QMenu::item:pressed {
    background-color: #585b70; /* Catppuccin Surface2 */
}
QMenu::item:disabled {
    background-color: transparent;
    color: #6c7086; /* Catppuccin Overlay0 - muted color for disabled items */
}
QMenu::separator {
    height: 1px;
    background: #45475a; /* Catppuccin Surface1 */
    margin: 4px 8px;
}
"""

    AGENT_MENU = """
QMenu {
    background-color: #181825; /* Catppuccin Mantle */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #313244; /* Catppuccin Surface0 */
    border-radius: 4px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 20px;
    background-color: transparent;
}
QMenu::item:selected {
    background-color: #89b4fa; /* Catppuccin Blue */
    color: #1e1e2e; /* Catppuccin Base */
    border-radius: 3px;
}
QMenu::separator {
    height: 1px;
    background-color: #313244; /* Catppuccin Surface0 */
    margin-left: 5px;
    margin-right: 5px;
}
"""

    # Label styles
    STATUS_INDICATOR = """
QLabel {
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
    padding: 8px; 
    border-radius: 5px;
    font-weight: bold;
}
"""

    VERSION_LABEL = """
QLabel {
    color: #a6adc8; /* Catppuccin Subtext0 */
    padding: 2px 8px;
    font-size: 11px;
}
"""

    SYSTEM_MESSAGE_TOGGLE_BUTTON = """
QPushButton {
    background-color: transparent;
    color: #94e2d5; /* Catppuccin Teal */
    border: none;
    text-align: left;
    padding: 2px;
}
QPushButton:hover {
    color: #89dceb; /* Catppuccin Sky */
}
"""

    # Widget-specific styles
    SIDEBAR = """
background-color: #181825; /* Catppuccin Mantle */
"""

    CONVERSATION_LIST = """
QListWidget {
    background-color: #1e1e2e; /* Catppuccin Base */
    border: 1px solid #313244; /* Catppuccin Surface0 */
    border-radius: 4px;
}
QListWidget::item {
    color: #cdd6f4; /* Catppuccin Text */
    background-color: #1e1e2e; /* Catppuccin Base */
    border: none; /* Remove individual item borders if not desired */
    border-bottom: 1px solid #313244; /* Surface0 for separator */
    margin: 0px; /* Remove margin if using border for separation */
    padding: 8px;
}
/* QListWidget::item:alternate { */ /* Remove if not using alternating */
/*     background-color: #1a1a2e; */ /* Slightly different base if needed */
/* } */
QListWidget::item:selected {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #b4befe; /* Catppuccin Lavender */
}
QListWidget::item:hover:!selected {
    background-color: #313244; /* Catppuccin Surface0 */
}
"""

    SEARCH_BOX = """
QLineEdit {
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    border-radius: 4px;
    padding: 8px;
}
QLineEdit:focus {
    border: 1px solid #89b4fa; /* Catppuccin Blue */
}
"""

    TOKEN_USAGE = """
QLabel {
    color: #bac2de; /* Catppuccin Subtext1 */
    font-weight: bold;
    padding: 8px;
    background-color: #1e1e2e; /* Catppuccin Crust */
    border-top: 1px solid #313244; /* Catppuccin Surface0 for a subtle separator */
}
"""

    TOKEN_USAGE_WIDGET = """
background-color: #11111b; /* Catppuccin Crust */
"""

    # Message bubble styles
    USER_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #89b4fa; /* Catppuccin Blue */
    border: none;
    padding: 2px;
}
"""

    ASSISTANT_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #313244; /* Catppuccin Surface0 */
    border: none;
    padding: 2px;
}
"""

    THINKING_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #45475a; /* Catppuccin Surface1 */
    border: none;
    padding: 2px;
}
"""

    CONSOLIDATED_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #313244; /* Catppuccin Surface0 */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    padding: 2px;
}
"""

    ROLLBACK_BUTTON = """
QPushButton {
    background-color: #b4befe; /* Catppuccin Lavender */
    color: #1e1e2e; /* Catppuccin Base */
    border: none;
    border-radius: 15px;
    padding: 8px;
    font-size: 24px;
    font-weight: bold;
    width: 30px;
    height: 30px;
}
QPushButton:hover {
    background-color: #cba6f7; /* Catppuccin Mauve */
}
QPushButton:pressed {
    background-color: #f5c2e7; /* Catppuccin Pink */
}
"""

    CONSOLIDATED_BUTTON = """
QPushButton {
    background-color: #b4befe; /* Catppuccin Lavender */
    color: #1e1e2e; /* Catppuccin Base */
    border: none;
    border-radius: 15px;
    padding: 8px;
    font-size: 16px;
    font-weight: bold;
    width: 30px;
    height: 30px;
}
QPushButton:hover {
    background-color: #cba6f7; /* Catppuccin Mauve */
}
QPushButton:pressed {
    background-color: #f5c2e7; /* Catppuccin Pink */
}
"""

    # Tool dialog styles
    TOOL_DIALOG_TEXT_EDIT = """
QTextEdit {
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    border-radius: 4px;
    padding: 8px;
    font-family: monospace;
}
QTextEdit:focus {
    border: 1px solid #89b4fa; /* Catppuccin Blue */
}
"""

    TOOL_DIALOG_YES_BUTTON = """
QPushButton {
    background-color: #a6e3a1; /* Catppuccin Green */
    color: #1e1e2e; /* Catppuccin Base */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #94e2d5; /* Catppuccin Teal */
}
"""

    TOOL_DIALOG_ALL_BUTTON = """
QPushButton {
    background-color: #89b4fa; /* Catppuccin Blue */
    color: #1e1e2e; /* Catppuccin Base */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #74c7ec; /* Catppuccin Sapphire */
}
"""

    TOOL_DIALOG_NO_BUTTON = """
QPushButton {
    background-color: #f38ba8; /* Catppuccin Red */
    color: #1e1e2e; /* Catppuccin Base */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #eba0ac; /* Catppuccin Maroon */
}
"""

    # Additional button styles
    RED_BUTTON = """
QPushButton {
    background-color: #f38ba8; /* Catppuccin Red */
    color: #1e1e2e; /* Catppuccin Base (for contrast) */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #eba0ac; /* Catppuccin Maroon (lighter red for hover) */
}
QPushButton:pressed {
    background-color: #e67e8a; /* A slightly darker/more intense red for pressed */
}
QPushButton:disabled {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #6c7086; /* Catppuccin Overlay0 */
}
"""

    GREEN_BUTTON = """
QPushButton {
    background-color: #a6e3a1; /* Catppuccin Green */
    color: #1e1e2e; /* Catppuccin Base */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #94e2d5; /* Catppuccin Teal - lighter green for hover */
}
QPushButton:pressed {
    background-color: #8bd5ca; /* Slightly darker for pressed state */
}
QPushButton:disabled {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #6c7086; /* Catppuccin Overlay0 */
}
"""

    AGENT_MENU_BUTTON = """
QPushButton {
    background-color: #89b4fa; /* Catppuccin Blue */
    color: #1e1e2e; /* Catppuccin Base */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
    padding-left: 12px; /* Add some padding for text */
}
QPushButton:hover {
    background-color: #74c7ec; /* Catppuccin Sapphire */
}
QPushButton:pressed {
    background-color: #b4befe; /* Catppuccin Lavender */
}
QPushButton:disabled {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #6c7086; /* Catppuccin Overlay0 */
}
QPushButton::menu-indicator {
    subcontrol-origin: padding;
    subcontrol-position: right center;
    right: 5px;
    width: 16px;
}
"""

    # Widget-specific styles
    SYSTEM_MESSAGE_LABEL = """
color: #a6adc8; /* Catppuccin Subtext0 */
padding: 2px;
"""

    SYSTEM_MESSAGE_TOGGLE = """
QPushButton {
    background-color: transparent;
    color: #94e2d5; /* Catppuccin Teal */
    border: none;
    text-align: left;
    padding: 2px;
}
QPushButton:hover {
    color: #89dceb; /* Catppuccin Sky */
}
"""

    # Config window styles
    CONFIG_DIALOG = """
QDialog {
    background-color: #1e1e2e; /* Catppuccin Base */
    color: #cdd6f4; /* Catppuccin Text */
}
QTabWidget::pane {
    border: 1px solid #313244; /* Catppuccin Surface0 */
    background-color: #181825; /* Catppuccin Mantle */
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
    padding: 8px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #181825; /* Catppuccin Mantle (same as pane) */
    border-bottom-color: #181825; /* Catppuccin Mantle */
    color: #b4befe; /* Catppuccin Lavender for selected tab text */
}
QTabBar::tab:hover:!selected {
    background-color: #45475a; /* Catppuccin Surface1 */
}
QPushButton {
    background-color: #89b4fa; /* Catppuccin Blue */
    color: #1e1e2e; /* Catppuccin Base */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #74c7ec; /* Catppuccin Sapphire */
}
QPushButton:pressed {
    background-color: #b4befe; /* Catppuccin Lavender */
}
QPushButton:disabled {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #6c7086; /* Catppuccin Overlay0 */
}
QListWidget {
    background-color: #1e1e2e; /* Catppuccin Base */
    border: 1px solid #313244; /* Catppuccin Surface0 */
    border-radius: 4px;
    padding: 4px;
    color: #cdd6f4; /* Catppuccin Text */
}
QListWidget::item {
    padding: 6px;
    border-radius: 2px;
    color: #cdd6f4; /* Catppuccin Text */
    background-color: #1e1e2e; /* Catppuccin Base */
}
QListWidget::item:selected {
    background-color: #45475a; /* Catppuccin Surface1 */
    color: #b4befe; /* Catppuccin Lavender */
}
QListWidget::item:hover:!selected {
    background-color: #313244; /* Catppuccin Surface0 */
}
QLineEdit, QTextEdit {
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    border-radius: 4px;
    padding: 6px;
    background-color: #313244; /* Catppuccin Surface0 */
    color: #cdd6f4; /* Catppuccin Text */
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #89b4fa; /* Catppuccin Blue */
}
QCheckBox {
    spacing: 8px;
    color: #cdd6f4; /* Catppuccin Text */
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 2px;
    background-color: #45475a; /* Catppuccin Surface1 */
    border: 1px solid #585b70; /* Catppuccin Surface2 */
}
QCheckBox::indicator:checked {
    background-color: #89b4fa; /* Catppuccin Blue */
    border: 1px solid #89b4fa; /* Catppuccin Blue */
}
QCheckBox::indicator:hover {
    border: 1px solid #b4befe; /* Catppuccin Lavender */
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #45475a; /* Catppuccin Surface1 */
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px; /* Ensure space for title */
    background-color: #181825; /* Catppuccin Mantle for groupbox background */
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left; /* position at the top left */
    padding: 0 4px 4px 4px; /* padding for title */
    color: #b4befe; /* Catppuccin Lavender for title */
    background-color: #181825; /* Match groupbox background */
    left: 10px; /* Adjust to align with content */
}
QScrollArea {
    background-color: #181825; /* Catppuccin Mantle */
    border: none;
}
/* Style for the QWidget inside QScrollArea if needed */
QScrollArea > QWidget > QWidget { /* Target the editor_widget */
     background-color: #181825; /* Catppuccin Mantle */
}
QLabel {
    color: #cdd6f4; /* Catppuccin Text */
    padding: 2px; /* Add some padding to labels */
}
QSplitter::handle {
    background-color: #313244; /* Catppuccin Surface0 */
}
QSplitter::handle:hover {
    background-color: #45475a; /* Catppuccin Surface1 */
}
QSplitter::handle:pressed {
    background-color: #585b70; /* Catppuccin Surface2 */
}
"""

    PANEL = """
background-color: #181825; /* Catppuccin Mantle */
"""

    SCROLL_AREA = """
background-color: #181825; /* Catppuccin Mantle */
border: none;
"""

    EDITOR_WIDGET = """
background-color: #181825; /* Catppuccin Mantle */
"""

    GROUP_BOX = """
background-color: #1e1e2e; /* Catppuccin Base */
"""

    SPLITTER_COLOR = """
QSplitter::handle {
    background-color: #181825; /* Darker color (Catppuccin Mantle) */
}
QSplitter::handle:hover {
    background-color: #313244; /* Catppuccin Surface0 */
}
QSplitter::handle:pressed {
    background-color: #45475a; /* Catppuccin Surface1 */
}
"""

    METADATA_HEADER_LABEL = """
QLabel {
    color: #a6adc8; /* Catppuccin Subtext0 */
    font-style: italic;
    padding-bottom: 5px;
}
"""

    # Message label styles
    USER_MESSAGE_LABEL = """
QLabel {
    color: #1e1e2e; /* Catppuccin Base */
}
"""

    ASSISTANT_MESSAGE_LABEL = """
QLabel {
    color: #cdd6f4; /* Catppuccin Text */
}
"""

    THINKING_MESSAGE_LABEL = """
QLabel {
    color: #bac2de; /* Catppuccin Subtext1 */
}
"""

    # Sender label styles
    USER_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #1e1e2e; /* Catppuccin Base */
    padding: 2px;
}
"""

    ASSISTANT_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #cdd6f4; /* Catppuccin Text */
    padding: 2px;
}
"""

    THINKING_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #bac2de; /* Catppuccin Subtext1 */
    padding: 2px;
}
"""

    # File display label styles
    USER_FILE_NAME_LABEL = """
QLabel {
    font-weight: bold;
    color: #1e1e2e; /* Catppuccin Base */
    padding-left: 10px;
}
"""

    ASSISTANT_FILE_NAME_LABEL = """
QLabel {
    font-weight: bold;
    color: #cdd6f4; /* Catppuccin Text */
    padding-left: 10px;
}
"""

    USER_FILE_INFO_LABEL = """
QLabel {
    color: #313244; /* Catppuccin Surface0 - better contrast on blue background */
}
"""

    ASSISTANT_FILE_INFO_LABEL = """
QLabel {
    color: #6c7086; /* Catppuccin Overlay0 */
}
"""

    CODE_CSS = """
pre { line-height: 1; background-color: #181825; border-radius: 8px; padding: 12px; color: #cdd6f4; white-space: pre-wrap; word-wrap: break-word; } /* Mantle, Text */
td.linenos .normal { color: #6c7086; background-color: transparent; padding-left: 5px; padding-right: 5px; } /* Overlay0 */
span.linenos { color: #6c7086; background-color: transparent; padding-left: 5px; padding-right: 5px; } /* Overlay0 */
td.linenos .special { color: #cdd6f4; background-color: #313244; padding-left: 5px; padding-right: 5px; } /* Text, Surface0 */
span.linenos.special { color: #cdd6f4; background-color: #313244; padding-left: 5px; padding-right: 5px; } /* Text, Surface0 */
.codehilite .hll { background-color: #313244 } /* Surface0 */
.codehilite { background: #181825; border-radius: 8px; padding: 10px; color: #cdd6f4; } /* Mantle, Text */
.codehilite .c { color: #6c7086; font-style: italic } /* Comment -> Overlay0 */
.codehilite .err { border: 1px solid #f38ba8; color: #f38ba8; } /* Error -> Red */
.codehilite .k { color: #cba6f7; font-weight: bold } /* Keyword -> Mauve */
.codehilite .o { color: #94e2d5 } /* Operator -> Teal */
.codehilite .ch { color: #6c7086; font-style: italic } /* Comment.Hashbang -> Overlay0 */
.codehilite .cm { color: #6c7086; font-style: italic } /* Comment.Multiline -> Overlay0 */
.codehilite .cp { color: #f9e2af } /* Comment.Preproc -> Yellow */
.codehilite .cpf { color: #6c7086; font-style: italic } /* Comment.PreprocFile -> Overlay0 */
.codehilite .c1 { color: #6c7086; font-style: italic } /* Comment.Single -> Overlay0 */
.codehilite .cs { color: #6c7086; font-style: italic } /* Comment.Special -> Overlay0 */
.codehilite .gd { color: #f38ba8 } /* Generic.Deleted -> Red */
.codehilite .ge { font-style: italic } /* Generic.Emph */
.codehilite .ges { font-weight: bold; font-style: italic } /* Generic.EmphStrong */
.codehilite .gr { color: #f38ba8 } /* Generic.Error -> Red */
.codehilite .gh { color: #89b4fa; font-weight: bold } /* Generic.Heading -> Blue */
.codehilite .gi { color: #a6e3a1 } /* Generic.Inserted -> Green */
.codehilite .go { color: #cdd6f4 } /* Generic.Output -> Text */
.codehilite .gp { color: #89b4fa; font-weight: bold } /* Generic.Prompt -> Blue */
.codehilite .gs { font-weight: bold } /* Generic.Strong */
.codehilite .gu { color: #89b4fa; font-weight: bold } /* Generic.Subheading -> Blue */
.codehilite .gt { color: #f38ba8 } /* Generic.Traceback -> Red */
.codehilite .kc { color: #cba6f7; font-weight: bold } /* Keyword.Constant -> Mauve */
.codehilite .kd { color: #cba6f7; font-weight: bold } /* Keyword.Declaration -> Mauve */
.codehilite .kn { color: #cba6f7; font-weight: bold } /* Keyword.Namespace -> Mauve */
.codehilite .kp { color: #cba6f7 } /* Keyword.Pseudo -> Mauve */
.codehilite .kr { color: #cba6f7; font-weight: bold } /* Keyword.Reserved -> Mauve */
.codehilite .kt { color: #fab387; font-weight: bold } /* Keyword.Type -> Peach */
.codehilite .m { color: #fab387 } /* Literal.Number -> Peach */
.codehilite .s { color: #a6e3a1 } /* Literal.String -> Green */
.codehilite .na { color: #89dceb } /* Name.Attribute -> Sky */
.codehilite .nb { color: #89b4fa } /* Name.Builtin -> Blue */
.codehilite .nc { color: #f9e2af; font-weight: bold } /* Name.Class -> Yellow */
.codehilite .no { color: #fab387 } /* Name.Constant -> Peach */
.codehilite .nd { color: #cba6f7 } /* Name.Decorator -> Mauve */
.codehilite .ni { color: #cdd6f4; font-weight: bold } /* Name.Entity -> Text */
.codehilite .ne { color: #f38ba8; font-weight: bold } /* Name.Exception -> Red */
.codehilite .nf { color: #89b4fa; font-weight: bold } /* Name.Function -> Blue */
.codehilite .nl { color: #cdd6f4 } /* Name.Label -> Text */
.codehilite .nn { color: #f9e2af; font-weight: bold } /* Name.Namespace -> Yellow */
.codehilite .nt { color: #cba6f7; font-weight: bold } /* Name.Tag -> Mauve */
.codehilite .nv { color: #f5e0dc } /* Name.Variable -> Rosewater */
.codehilite .ow { color: #94e2d5; font-weight: bold } /* Operator.Word -> Teal */
.codehilite .w { color: #45475a } /* Text.Whitespace -> Surface1 */
.codehilite .mb { color: #fab387 } /* Literal.Number.Bin -> Peach */
.codehilite .mf { color: #fab387 } /* Literal.Number.Float -> Peach */
.codehilite .mh { color: #fab387 } /* Literal.Number.Hex -> Peach */
.codehilite .mi { color: #fab387 } /* Literal.Number.Integer -> Peach */
.codehilite .mo { color: #fab387 } /* Literal.Number.Oct -> Peach */
.codehilite .sa { color: #a6e3a1 } /* Literal.String.Affix -> Green */
.codehilite .sb { color: #a6e3a1 } /* Literal.String.Backtick -> Green */
.codehilite .sc { color: #a6e3a1 } /* Literal.String.Char -> Green */
.codehilite .dl { color: #a6e3a1 } /* Literal.String.Delimiter -> Green */
.codehilite .sd { color: #6c7086; font-style: italic } /* Literal.String.Doc -> Overlay0 */
.codehilite .s2 { color: #a6e3a1 } /* Literal.String.Double -> Green */
.codehilite .se { color: #fab387; font-weight: bold } /* Literal.String.Escape -> Peach */
.codehilite .sh { color: #a6e3a1 } /* Literal.String.Heredoc -> Green */
.codehilite .si { color: #a6e3a1; font-weight: bold } /* Literal.String.Interpol -> Green */
.codehilite .sx { color: #a6e3a1 } /* Literal.String.Other -> Green */
.codehilite .sr { color: #a6e3a1 } /* Literal.String.Regex -> Green */
.codehilite .s1 { color: #a6e3a1 } /* Literal.String.Single -> Green */
.codehilite .ss { color: #a6e3a1 } /* Literal.String.Symbol -> Green */
.codehilite .bp { color: #89b4fa } /* Name.Builtin.Pseudo -> Blue */
.codehilite .fm { color: #89b4fa; font-weight: bold } /* Name.Function.Magic -> Blue */
.codehilite .vc { color: #f5e0dc } /* Name.Variable.Class -> Rosewater */
.codehilite .vg { color: #f5e0dc } /* Name.Variable.Global -> Rosewater */
.codehilite .vi { color: #f5e0dc } /* Name.Variable.Instance -> Rosewater */
.codehilite .vm { color: #f5e0dc } /* Name.Variable.Magic -> Rosewater */
.codehilite .il { color: #fab387 } /* Literal.Number.Integer.Long -> Peach */
"""

    # Tool widget styles
    TOOL_WIDGET = """
QWidget {
    background-color: transparent;
}
"""

    TOOL_CARD = """
QFrame#toolCard {
    border-radius: 6px;  /* Reduced from 12px for subtlety */
    background-color: rgba(137, 180, 250, 0.08); /* More subtle transparency */
    border: 1px solid rgba(137, 180, 250, 0.4); /* Softer border */
    padding: 1px; /* Minimal padding */
}
"""

    TOOL_CARD_ERROR = """
QFrame#toolCard {
    border-radius: 6px;  /* Reduced from 12px for subtlety */
    background-color: rgba(243, 139, 168, 0.08); /* More subtle transparency */
    border: 1px solid rgba(243, 139, 168, 0.4); /* Softer border */
    padding: 1px; /* Minimal padding */
}
"""

    TOOL_HEADER = """
QLabel {
    font-weight: 500; /* Lighter weight for subtlety */
    color: #a6adc8; /* Subtext0 - more muted than full text */
}
"""

    TOOL_TOGGLE_BUTTON = """
QPushButton {
    background-color: transparent;
    border: none;
    color: #6c7086; /* More muted overlay color */
    font-weight: normal; /* Reduced from bold */
    font-size: 11px; /* Smaller font */
}
QPushButton:hover {
    color: #89b4fa; /* Blue */
}
"""

    TOOL_STATUS = """
QLabel {
    font-weight: 500; /* Lighter than bold */
    padding: 2px; /* Reduced padding */
    font-size: 11px; /* Smaller status text */
}
QLabel[status="running"] {
    color: #f9e2af; /* Yellow */
}
QLabel[status="complete"] {
    color: #a6e3a1; /* Green */
}
QLabel[status="error"] {
    color: #f38ba8; /* Red */
}
"""

    TOOL_CONTENT = """
QLabel {
    color: #bac2de; /* Subtext1 - more muted */
    padding: 1px; /* Minimal padding */
    font-size: 11px; /* Smaller content text */
}
QLabel[role="title"] {
    font-weight: 500; /* Lighter than bold */
    color: #bac2de; /* Subtext1 */
    font-size: 12px;
}
QLabel[role="key"] {
    color: #74c7ec; /* Sapphire - more subtle than blue */
    font-weight: 500;
    font-size: 11px;
}
QLabel[role="value"] {
    color: #9399b2; /* Subtext0 - more muted */
    font-size: 11px;
}
QLabel[role="error"] {
    color: #f38ba8; /* Red for error messages */
    font-size: 11px;
}
"""

    TOOL_PROGRESS = """
QProgressBar {
    border: none;
    background-color: rgba(69, 71, 90, 0.4); /* Surface1 with transparency */
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #89b4fa; /* Blue */
    border-radius: 5px;
}
"""

    TOOL_SEPARATOR = """
QFrame {
    background-color: rgba(137, 180, 250, 0.3); /* Blue with transparency */
    border: none;
}
"""

    # Tool icons (consistent across all themes)
    TOOL_ICONS = {
        "web_search": "🔍",
        "fetch_webpage": "🌐",
        "transfer": "↗️",
        "adapt": "🧠",
        "retrieve_memory": "💭",
        "forget_memory_topic": "🗑️",
        "analyze_repo": "📂",
        "read_file": "📄",
        "default": "🔧",  # Default icon for unspecified tools
    }

    # JSON Editor styles
    JSON_EDITOR_COLORS = {
        "background": "#313244",  # Surface0
        "text": "#cdd6f4",  # Text
        "border": "#45475a",  # Surface1
        "string": "#a6e3a1",  # Green
        "number": "#fab387",  # Peach
        "keyword": "#cba6f7",  # Mauve
        "punctuation": "#cdd6f4",  # Text
        "error": "#f38ba8",  # Red
    }

    JSON_EDITOR_STYLE = """
QPlainTextEdit {
    background-color: #313244; /* Surface0 */
    color: #cdd6f4; /* Text */
    border: 1px solid #45475a; /* Surface1 */
    border-radius: 4px;
    padding: 8px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.4;
}
QPlainTextEdit:focus {
    border: 1px solid #89b4fa; /* Blue focus border */
}
"""

    # Markdown Editor styles
    MARKDOWN_EDITOR_COLORS = {
        "background": "#313244",  # Surface0
        "text": "#cdd6f4",  # Text
        "border": "#45475a",  # Surface1
        "header": "#89b4fa",  # Blue - for headers
        "bold": "#fab387",  # Peach - for bold text
        "italic": "#a6e3a1",  # Green - for italic text
        "code": "#f5c2e7",  # Pink - for code blocks
        "code_background": "#45475a",  # Surface1 - code background
        "link": "#74c7ec",  # Sapphire - for links
        "image": "#cba6f7",  # Mauve - for images
        "list": "#f9e2af",  # Yellow - for list markers
        "blockquote": "#94e2d5",  # Teal - for blockquotes
        "hr": "#6c7086",  # Overlay0 - for horizontal rules
        "strikethrough": "#eba0ac",  # Maroon - for strikethrough text
        "error": "#f38ba8",  # Red - for errors
    }

    MARKDOWN_EDITOR_STYLE = """
QPlainTextEdit {
    background-color: #313244; /* Surface0 */
    color: #cdd6f4; /* Text */
    border: 1px solid #45475a; /* Surface1 */
    border-radius: 4px;
    padding: 8px;
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
    line-height: 1.6;
    selection-background-color: #585b70; /* Surface2 */
    selection-color: #cdd6f4; /* Text */
}
QPlainTextEdit:focus {
    border: 1px solid #89b4fa; /* Blue focus border */
}
"""
