"""Atom Light theme styles for AgentCrew GUI."""


class AtomLightTheme:
    """Static class containing Atom Light theme styles."""

    # Main application styles
    MAIN_STYLE = """
QMainWindow {
    background-color: #ffffff; /* White background */
}
QScrollArea {
    border: none;
    background-color: #f8f8f8; /* Light gray */
}
QWidget#chatContainer { /* Specific ID for chat_container */
    background-color: #ffffff; /* Light gray */
}
QSplitter::handle {
    background-color: #e0e0e0; /* Light gray */
}
QSplitter::handle:hover {
    background-color: #d0d0d0; /* Slightly darker gray */
}
QSplitter::handle:pressed {
    background-color: #c0c0c0; /* Even darker gray */
}
QStatusBar {
    background-color: #f0f0f0; /* Very light gray */
    color: #333333; /* Dark gray text */
}
QToolTip {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #cccccc; /* Light gray border */
    padding: 4px;
}
QMessageBox {
    background-color: #ffffff; /* White */
}
QMessageBox QLabel { /* For message text in QMessageBox */
    color: #333333; /* Dark gray text */
    background-color: transparent;
}
QListView { /* General style for QListView, affects completer */
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #cccccc; /* Light gray border */
    padding: 2px;
    outline: 0px;
}
QListView::item {
    padding: 4px 8px;
    border-radius: 2px;
}
QListView::item:selected {
    background-color: #4285f4; /* Blue selection */
    color: #ffffff; /* White text */
}
QListView::item:hover {
    background-color: #e8f0fe; /* Light blue hover */
}

/* Modern Scrollbar Styles */
QScrollBar:vertical {
    border: none;
    background: #f8f8f8; /* Light gray track */
    width: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #cccccc; /* Gray handle */
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #999999; /* Darker gray on hover */
}
QScrollBar::handle:vertical:pressed {
    background: #666666; /* Even darker when pressed */
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
    height: 0px;
    subcontrol-position: top;
    subcontrol-origin: margin;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

QScrollBar:horizontal {
    border: none;
    background: #f8f8f8; /* Light gray track */
    height: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:horizontal {
    background: #cccccc; /* Gray handle */
    min-width: 20px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: #999999; /* Darker gray on hover */
}
QScrollBar::handle:horizontal:pressed {
    background: #666666; /* Even darker when pressed */
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    border: none;
    background: none;
    width: 0px;
    subcontrol-position: left;
    subcontrol-origin: margin;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

/* Context menu styling for QLabel widgets */
QLabel QMenu {
    background-color: #ffffff; /* White */
    border: 1px solid #cccccc; /* Light gray border */
    padding: 4px;
    border-radius: 6px;
}
QLabel QMenu::item {
    color: #333333; /* Dark gray text */
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    margin: 2px;
}
QLabel QMenu::item:selected {
    background-color: #4285f4; /* Blue selection */
    color: #ffffff; /* White text */
}
QLabel QMenu::item:pressed {
    background-color: #3367d6; /* Darker blue */
}
QLabel QMenu::separator {
    height: 1px;
    background: #cccccc; /* Light gray */
    margin: 4px 8px;
}
"""

    # Button styles
    PRIMARY_BUTTON = """
QPushButton {
    background-color: #4285f4; /* Blue */
    color: #ffffff; /* White text */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3367d6; /* Darker blue */
}
QPushButton:pressed {
    background-color: #1557b0; /* Even darker blue */
}
QPushButton:disabled {
    background-color: #e0e0e0; /* Light gray */
    color: #999999; /* Gray text */
}
"""

    SECONDARY_BUTTON = """
QPushButton {
    background-color: #a0a0a0; /* Light gray */
    color: #ffffff; /* Dark gray text */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #d0d0d0; /* Darker gray */
}
QPushButton:pressed {
    background-color: #c0c0c0; /* Even darker gray */
}
QPushButton:disabled {
    background-color: #f0f0f0; /* Very light gray */
    color: #cccccc; /* Light gray text */
}
"""

    STOP_BUTTON = """
QPushButton {
    background-color: #ea4335; /* Red */
    color: #ffffff; /* White text */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #d93025; /* Darker red */
}
QPushButton:pressed {
    background-color: #b52d20; /* Even darker red */
}
"""

    RED_BUTTON = """
QPushButton {
    background-color: #ea4335; /* Red */
    color: #ffffff; /* White text */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #d93025; /* Darker red */
}
QPushButton:pressed {
    background-color: #b52d20; /* Even darker red */
}
QPushButton:disabled {
    background-color: #e0e0e0; /* Light gray */
    color: #999999; /* Gray text */
}
"""

    GREEN_BUTTON = """
QPushButton {
    background-color: #34a853; /* Green */
    color: #ffffff; /* White text */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2d8f47; /* Darker green */
}
QPushButton:pressed {
    background-color: #1e5f31; /* Even darker green */
}
QPushButton:disabled {
    background-color: #e0e0e0; /* Light gray */
    color: #999999; /* Gray text */
}
"""

    API_KEYS_GROUP = """
QGroupBox {
    background-color: #ffffff; /* White */
}
QLabel {
    background-color: #ffffff;
}
"""

    EDITOR_CONTAINER_WIDGET = """
QWidget {
    background-color: #f8f8f8; /* Light gray */
}
"""

    MENU_BUTTON = """
QPushButton {
    background-color: #4285f4; /* Blue */
    color: #ffffff; /* White text */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
    padding-left: 12px;
}
QPushButton:hover {
    background-color: #3367d6; /* Darker blue */
}
QPushButton:pressed {
    background-color: #1557b0; /* Even darker blue */
}
QPushButton:disabled {
    background-color: #e0e0e0; /* Light gray */
    color: #999999; /* Gray text */
}
QPushButton::menu-indicator {
    subcontrol-origin: padding;
    subcontrol-position: right center;
    right: 5px;
    width: 16px;
}
"""

    DISABLED_BUTTON = """
QPushButton {
    background-color: #e0e0e0; /* Light gray */
    color: #999999; /* Gray text */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
"""

    STOP_BUTTON_STOPPING = """
QPushButton {
    background-color: #cccccc; /* Light gray */
    color: #666666; /* Darker gray text */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
"""

    COMBO_BOX = """
QComboBox {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #cccccc; /* Light gray border */
    border-radius: 4px;
    padding: 6px 8px;
    min-width: 120px;
}
QComboBox:hover {
    border: 1px solid #999999; /* Darker gray border */
}
QComboBox:focus {
    border: 1px solid #4285f4; /* Blue border */
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: #cccccc; /* Light gray border */
    border-left-style: solid;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
    background-color: #f0f0f0; /* Light gray */
}
QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #333333; /* Dark gray arrow */
}
QComboBox QAbstractItemView {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #cccccc; /* Light gray border */
    border-radius: 4px;
    selection-background-color: #4285f4; /* Blue selection */
    selection-color: #ffffff; /* White text */
    outline: 0px;
}
"""

    # Input styles
    TEXT_INPUT = """
QTextEdit {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #cccccc; /* Light gray border */
    border-radius: 4px;
    padding: 8px;
}
QTextEdit:focus {
    border: 1px solid #4285f4; /* Blue border on focus */
}
"""

    LINE_EDIT = """
QLineEdit {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #cccccc; /* Light gray border */
    border-radius: 4px;
    padding: 8px;
}
QLineEdit:focus {
    border: 1px solid #4285f4; /* Blue border on focus */
}
"""

    # Menu styles
    MENU_BAR = """
QMenuBar {
    background-color: #f8f8f8; /* White */
    color: #333333; /* Dark gray text */
    padding: 2px;
}
QMenuBar::item {
    background-color: transparent;
    color: #333333; /* Dark gray text */
    padding: 4px 12px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #e8f0fe; /* Light blue */
}
QMenuBar::item:pressed {
    background-color: #d2e3fc; /* Slightly darker blue */
}
QMenu {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #cccccc; /* Light gray border */
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #4285f4; /* Blue selection */
    color: #ffffff; /* White text */
}
QMenu::separator {
    height: 1px;
    background: #cccccc; /* Light gray */
    margin-left: 10px;
    margin-right: 5px;
}
"""

    CONTEXT_MENU = """
QMenu {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #cccccc; /* Light gray border */
    padding: 4px;
    border-radius: 6px;
}
QMenu::item {
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    margin: 2px;
    color: #333333; /* Dark text for better visibility */
}
QMenu::item:selected {
    background-color: #4285f4; /* Blue selection */
    color: #ffffff; /* White text for selected items */
}
QMenu::item:pressed {
    background-color: #3367d6; /* Darker blue */
}
QMenu::item:disabled {
    background-color: transparent;
    color: #999999; /* Light gray - muted color for disabled items */
}
QMenu::separator {
    height: 1px;
    background: #cccccc; /* Light gray */
    margin: 4px 8px;
}
"""

    AGENT_MENU = """
QMenu {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #e0e0e0; /* Light gray border */
    border-radius: 4px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 20px;
    background-color: transparent;
}
QMenu::item:selected {
    background-color: #4285f4; /* Blue selection */
    color: #ffffff; /* White text */
    border-radius: 3px;
}
QMenu::separator {
    height: 1px;
    background-color: #e0e0e0; /* Light gray */
    margin-left: 5px;
    margin-right: 5px;
}
"""

    # Label styles
    STATUS_INDICATOR = """
QLabel {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    padding: 8px; 
    border-radius: 5px;
    font-weight: bold;
}
"""

    VERSION_LABEL = """
QLabel {
    color: #666666; /* Medium gray */
    padding: 2px 8px;
    font-size: 11px;
}
"""

    SYSTEM_MESSAGE_TOGGLE_BUTTON = """
QPushButton {
    background-color: transparent;
    color: #0066cc; /* Blue */
    border: none;
    text-align: left;
    padding: 2px;
}
QPushButton:hover {
    color: #004499; /* Darker blue */
}
"""

    # Widget-specific styles
    SIDEBAR = """
background-color: #f8f8f8; /* Light gray */
"""

    CONVERSATION_LIST = """
QListWidget {
    background-color: #ffffff; /* White */
    border: 1px solid #e0e0e0; /* Light gray border */
    border-radius: 4px;
}
QListWidget::item {
    color: #333333; /* Dark gray text */
    background-color: #ffffff; /* White */
    border: none;
    border-bottom: 1px solid #e0e0e0; /* Light gray separator */
    margin: 0px;
    padding: 8px;
}
QListWidget::item:selected {
    background-color: #4285f4; /* Blue selection */
    color: #ffffff; /* White text */
}
QListWidget::item:hover:!selected {
    background-color: #e8f0fe; /* Light blue hover */
}
"""

    SEARCH_BOX = """
QLineEdit {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #cccccc; /* Light gray border */
    border-radius: 4px;
    padding: 8px;
}
QLineEdit:focus {
    border: 1px solid #4285f4; /* Blue border on focus */
}
"""

    TOKEN_USAGE = """
QLabel {
    color: #666666; /* Medium gray */
    font-weight: bold;
    padding: 8px;
    background-color: #ffffff; /* Very light gray */
    border-top: 1px solid #e0e0e0; /* Light gray separator */
}
"""

    TOKEN_USAGE_WIDGET = """
background-color: #f0f0f0; /* Very light gray */
"""

    # Message bubble styles
    USER_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #4285f4; /* Blue */
    border: none;
    padding: 2px;
}
QLabel {
    border-width: 0px;
}
"""

    ASSISTANT_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #ffffff; /* White */
    border: 1px solid #e0e0e0; /* Light gray border */
    padding: 2px;
}
QLabel {
    border-width: 0px;
}
"""

    THINKING_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #f8f8f8; /* Light gray */
    border: 1px solid #cccccc; /* Gray border */
    padding: 2px;
}
QLabel {
    border-width: 0px;
}
"""

    CONSOLIDATED_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #ffffff; /* White */
    border: 1px solid #cccccc; /* Gray border */
    padding: 2px;
}
"""

    ROLLBACK_BUTTON = """
QPushButton {
    background-color: #9c27b0; /* Purple */
    color: #ffffff; /* White text */
    border: none;
    border-radius: 15px;
    padding: 8px;
    font-size: 24px;
    font-weight: bold;
    width: 30px;
    height: 30px;
}
QPushButton:hover {
    background-color: #7b1fa2; /* Darker purple */
}
QPushButton:pressed {
    background-color: #4a148c; /* Even darker purple */
}
"""

    CONSOLIDATED_BUTTON = """
QPushButton {
    background-color: #9c27b0; /* Purple */
    color: #ffffff; /* White text */
    border: none;
    border-radius: 15px;
    padding: 8px;
    font-size: 16px;
    font-weight: bold;
    width: 30px;
    height: 30px;
}
QPushButton:hover {
    background-color: #7b1fa2; /* Darker purple */
}
QPushButton:pressed {
    background-color: #4a148c; /* Even darker purple */
}
"""

    # Tool dialog styles
    TOOL_DIALOG_TEXT_EDIT = """
QTextEdit {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    border: 1px solid #cccccc; /* Light gray border */
    border-radius: 4px;
    padding: 8px;
    font-family: monospace;
}
QTextEdit:focus {
    border: 1px solid #4285f4; /* Blue border on focus */
}
"""

    TOOL_DIALOG_YES_BUTTON = """
QPushButton {
    background-color: #34a853; /* Green */
    color: #ffffff; /* White text */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #2d8f47; /* Darker green */
}
"""

    TOOL_DIALOG_ALL_BUTTON = """
QPushButton {
    background-color: #4285f4; /* Blue */
    color: #ffffff; /* White text */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #3367d6; /* Darker blue */
}
"""

    TOOL_DIALOG_NO_BUTTON = """
QPushButton {
    background-color: #ea4335; /* Red */
    color: #ffffff; /* White text */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #d93025; /* Darker red */
}
"""

    AGENT_MENU_BUTTON = """
QPushButton {
    background-color: #4285f4; /* Blue */
    color: #ffffff; /* White text */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
    padding-left: 12px;
}
QPushButton:hover {
    background-color: #3367d6; /* Darker blue */
}
QPushButton:pressed {
    background-color: #1557b0; /* Even darker blue */
}
QPushButton:disabled {
    background-color: #e0e0e0; /* Light gray */
    color: #999999; /* Gray text */
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
color: #666666; /* Medium gray */
padding: 2px;
"""

    SYSTEM_MESSAGE_TOGGLE = """
QPushButton {
    background-color: transparent;
    color: #0066cc; /* Blue */
    border: none;
    text-align: left;
    padding: 2px;
}
QPushButton:hover {
    color: #004499; /* Darker blue */
}
"""

    # Config window styles
    CONFIG_DIALOG = """
QDialog {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
}
QTabWidget::pane {
    border: 1px solid #e0e0e0; /* Light gray border */
    background-color: #f8f8f8; /* Light gray */
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
    padding: 8px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #cccccc; /* Light gray border */
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #f8f8f8; /* Light gray (same as pane) */
    border-bottom-color: #f8f8f8; /* Light gray */
    color: #4285f4; /* Blue for selected tab text */
}
QTabBar::tab:hover:!selected {
    background-color: #e8f0fe; /* Light blue */
}
QPushButton {
    background-color: #4285f4; /* Blue */
    color: #ffffff; /* White text */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3367d6; /* Darker blue */
}
QPushButton:pressed {
    background-color: #1557b0; /* Even darker blue */
}
QPushButton:disabled {
    background-color: #e0e0e0; /* Light gray */
    color: #999999; /* Gray text */
}
QListWidget {
    background-color: #ffffff; /* White */
    border: 1px solid #e0e0e0; /* Light gray border */
    border-radius: 4px;
    padding: 4px;
    color: #333333; /* Dark gray text */
}
QListWidget::item {
    padding: 6px;
    border-radius: 2px;
    color: #333333; /* Dark gray text */
    background-color: #ffffff; /* White */
}
QListWidget::item:selected {
    background-color: #4285f4; /* Blue selection */
    color: #ffffff; /* White text */
}
QListWidget::item:hover:!selected {
    background-color: #e8f0fe; /* Light blue hover */
}
QLineEdit, QTextEdit {
    border: 1px solid #cccccc; /* Light gray border */
    border-radius: 4px;
    padding: 6px;
    background-color: #ffffff; /* White */
    color: #333333; /* Dark gray text */
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #4285f4; /* Blue border on focus */
}
QCheckBox {
    spacing: 8px;
    color: #333333; /* Dark gray text */
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 2px;
    background-color: #ffffff; /* White */
    border: 1px solid #cccccc; /* Light gray border */
}
QCheckBox::indicator:checked {
    background-color: #4285f4; /* Blue */
    border: 1px solid #4285f4; /* Blue border */
}
QCheckBox::indicator:hover {
    border: 1px solid #3367d6; /* Darker blue */
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #cccccc; /* Light gray border */
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    background-color: #f8f8f8; /* Light gray background */
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px 4px 4px;
    color: #4285f4; /* Blue for title */
    background-color: #f8f8f8; /* Match groupbox background */
    left: 10px;
}
QScrollArea {
    background-color: #f8f8f8; /* Light gray */
    border: none;
}
QScrollArea > QWidget > QWidget {
     background-color: #f8f8f8; /* Light gray */
}
QLabel {
    color: #333333; /* Dark gray text */
    padding: 2px;
}
QSplitter::handle {
    background-color: #e0e0e0; /* Light gray */
}
QSplitter::handle:hover {
    background-color: #d0d0d0; /* Darker gray */
}
QSplitter::handle:pressed {
    background-color: #c0c0c0; /* Even darker gray */
}
"""

    PANEL = """
background-color: #f8f8f8; /* Light gray */
"""

    SCROLL_AREA = """
background-color: #f8f8f8; /* Light gray */
border: none;
"""

    EDITOR_WIDGET = """
background-color: #f8f8f8; /* Light gray */
"""

    GROUP_BOX = """
background-color: #ffffff; /* White */
"""

    SPLITTER_COLOR = """
QSplitter::handle {
    background-color: #e0e0e0; /* Light gray */
}
QSplitter::handle:hover {
    background-color: #d0d0d0; /* Darker gray */
}
QSplitter::handle:pressed {
    background-color: #c0c0c0; /* Even darker gray */
}
"""

    METADATA_HEADER_LABEL = """
QLabel {
    color: #666666; /* Medium gray */
    font-style: italic;
    padding-bottom: 5px;
}
"""

    # Message label styles
    USER_MESSAGE_LABEL = """
QLabel {
    color: #ffffff; /* White text on blue background */
}
"""

    ASSISTANT_MESSAGE_LABEL = """
QLabel {
    color: #333333; /* Dark gray text */
}
"""

    THINKING_MESSAGE_LABEL = """
QLabel {
    color: #666666; /* Medium gray text */
}
"""

    # Sender label styles
    USER_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #ffffff; /* White text on blue background */
    padding: 2px;
}
"""

    ASSISTANT_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #333333; /* Dark gray text */
    padding: 2px;
}
"""

    THINKING_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #666666; /* Medium gray text */
    padding: 2px;
}
"""

    # File display label styles
    USER_FILE_NAME_LABEL = """
QLabel {
    font-weight: bold;
    color: #ffffff; /* White text on blue background */
    padding-left: 10px;
}
"""

    ASSISTANT_FILE_NAME_LABEL = """
QLabel {
    font-weight: bold;
    color: #333333; /* Dark gray text */
    padding-left: 10px;
}
"""

    USER_FILE_INFO_LABEL = """
QLabel {
    color: #e8f0fe; /* Light blue on blue background */
}
"""

    ASSISTANT_FILE_INFO_LABEL = """
QLabel {
    color: #999999; /* Gray text */
}
"""

    CODE_CSS = """
pre { line-height: 1; background-color: #f8f8f8; border-radius: 8px; padding: 12px; color: #333333; white-space: pre-wrap; word-wrap: break-word; } /* Light gray background, dark text */
td.linenos .normal { color: #999999; background-color: transparent; padding-left: 5px; padding-right: 5px; } /* Gray line numbers */
span.linenos { color: #999999; background-color: transparent; padding-left: 5px; padding-right: 5px; } /* Gray line numbers */
td.linenos .special { color: #333333; background-color: #e0e0e0; padding-left: 5px; padding-right: 5px; } /* Dark text, light gray background */
span.linenos.special { color: #333333; background-color: #e0e0e0; padding-left: 5px; padding-right: 5px; } /* Dark text, light gray background */
.codehilite .hll { background-color: #e0e0e0 } /* Light gray highlight */
.codehilite { background: #f8f8f8; border-radius: 8px; padding: 10px; color: #333333; } /* Light gray background, dark text */
.codehilite .c { color: #999999; font-style: italic } /* Comment -> Gray */
.codehilite .err { border: 1px solid #ea4335; color: #ea4335; } /* Error -> Red */
.codehilite .k { color: #9c27b0; font-weight: bold } /* Keyword -> Purple */
.codehilite .o { color: #0066cc } /* Operator -> Blue */
.codehilite .ch { color: #999999; font-style: italic } /* Comment.Hashbang -> Gray */
.codehilite .cm { color: #999999; font-style: italic } /* Comment.Multiline -> Gray */
.codehilite .cp { color: #ff6600 } /* Comment.Preproc -> Orange */
.codehilite .cpf { color: #999999; font-style: italic } /* Comment.PreprocFile -> Gray */
.codehilite .c1 { color: #999999; font-style: italic } /* Comment.Single -> Gray */
.codehilite .cs { color: #999999; font-style: italic } /* Comment.Special -> Gray */
.codehilite .gd { color: #ea4335 } /* Generic.Deleted -> Red */
.codehilite .ge { font-style: italic } /* Generic.Emph */
.codehilite .ges { font-weight: bold; font-style: italic } /* Generic.EmphStrong */
.codehilite .gr { color: #ea4335 } /* Generic.Error -> Red */
.codehilite .gh { color: #4285f4; font-weight: bold } /* Generic.Heading -> Blue */
.codehilite .gi { color: #34a853 } /* Generic.Inserted -> Green */
.codehilite .go { color: #333333 } /* Generic.Output -> Dark text */
.codehilite .gp { color: #4285f4; font-weight: bold } /* Generic.Prompt -> Blue */
.codehilite .gs { font-weight: bold } /* Generic.Strong */
.codehilite .gu { color: #4285f4; font-weight: bold } /* Generic.Subheading -> Blue */
.codehilite .gt { color: #ea4335 } /* Generic.Traceback -> Red */
.codehilite .kc { color: #9c27b0; font-weight: bold } /* Keyword.Constant -> Purple */
.codehilite .kd { color: #9c27b0; font-weight: bold } /* Keyword.Declaration -> Purple */
.codehilite .kn { color: #9c27b0; font-weight: bold } /* Keyword.Namespace -> Purple */
.codehilite .kp { color: #9c27b0 } /* Keyword.Pseudo -> Purple */
.codehilite .kr { color: #9c27b0; font-weight: bold } /* Keyword.Reserved -> Purple */
.codehilite .kt { color: #ff6600; font-weight: bold } /* Keyword.Type -> Orange */
.codehilite .m { color: #ff6600 } /* Literal.Number -> Orange */
.codehilite .s { color: #34a853 } /* Literal.String -> Green */
.codehilite .na { color: #0066cc } /* Name.Attribute -> Blue */
.codehilite .nb { color: #4285f4 } /* Name.Builtin -> Blue */
.codehilite .nc { color: #ff6600; font-weight: bold } /* Name.Class -> Orange */
.codehilite .no { color: #ff6600 } /* Name.Constant -> Orange */
.codehilite .nd { color: #9c27b0 } /* Name.Decorator -> Purple */
.codehilite .ni { color: #333333; font-weight: bold } /* Name.Entity -> Dark text */
.codehilite .ne { color: #ea4335; font-weight: bold } /* Name.Exception -> Red */
.codehilite .nf { color: #4285f4; font-weight: bold } /* Name.Function -> Blue */
.codehilite .nl { color: #333333 } /* Name.Label -> Dark text */
.codehilite .nn { color: #ff6600; font-weight: bold } /* Name.Namespace -> Orange */
.codehilite .nt { color: #9c27b0; font-weight: bold } /* Name.Tag -> Purple */
.codehilite .nv { color: #e91e63 } /* Name.Variable -> Pink */
.codehilite .ow { color: #0066cc; font-weight: bold } /* Operator.Word -> Blue */
.codehilite .w { color: #cccccc } /* Text.Whitespace -> Light gray */
.codehilite .mb { color: #ff6600 } /* Literal.Number.Bin -> Orange */
.codehilite .mf { color: #ff6600 } /* Literal.Number.Float -> Orange */
.codehilite .mh { color: #ff6600 } /* Literal.Number.Hex -> Orange */
.codehilite .mi { color: #ff6600 } /* Literal.Number.Integer -> Orange */
.codehilite .mo { color: #ff6600 } /* Literal.Number.Oct -> Orange */
.codehilite .sa { color: #34a853 } /* Literal.String.Affix -> Green */
.codehilite .sb { color: #34a853 } /* Literal.String.Backtick -> Green */
.codehilite .sc { color: #34a853 } /* Literal.String.Char -> Green */
.codehilite .dl { color: #34a853 } /* Literal.String.Delimiter -> Green */
.codehilite .sd { color: #999999; font-style: italic } /* Literal.String.Doc -> Gray */
.codehilite .s2 { color: #34a853 } /* Literal.String.Double -> Green */
.codehilite .se { color: #ff6600; font-weight: bold } /* Literal.String.Escape -> Orange */
.codehilite .sh { color: #34a853 } /* Literal.String.Heredoc -> Green */
.codehilite .si { color: #34a853; font-weight: bold } /* Literal.String.Interpol -> Green */
.codehilite .sx { color: #34a853 } /* Literal.String.Other -> Green */
.codehilite .sr { color: #34a853 } /* Literal.String.Regex -> Green */
.codehilite .s1 { color: #34a853 } /* Literal.String.Single -> Green */
.codehilite .ss { color: #34a853 } /* Literal.String.Symbol -> Green */
.codehilite .bp { color: #4285f4 } /* Name.Builtin.Pseudo -> Blue */
.codehilite .fm { color: #4285f4; font-weight: bold } /* Name.Function.Magic -> Blue */
.codehilite .vc { color: #e91e63 } /* Name.Variable.Class -> Pink */
.codehilite .vg { color: #e91e63 } /* Name.Variable.Global -> Pink */
.codehilite .vi { color: #e91e63 } /* Name.Variable.Instance -> Pink */
.codehilite .vm { color: #e91e63 } /* Name.Variable.Magic -> Pink */
.codehilite .il { color: #ff6600 } /* Literal.Number.Integer.Long -> Orange */
"""

    # Tool widget styles
    # Tool widget styles
    TOOL_WIDGET = """
QWidget {
    background-color: transparent;
}
"""

    TOOL_CARD = """
QFrame#toolCard {
    border-radius: 6px; /* Reduced from 12px for subtlety */
    background-color: rgba(66, 133, 244, 0.06); /* More subtle transparency */
    border: 1px solid rgba(66, 133, 244, 0.3); /* Softer border */
    padding: 1px; /* Minimal padding */
}
"""

    TOOL_CARD_ERROR = """
QFrame#toolCard {
    border-radius: 6px; /* Reduced from 12px for subtlety */
    background-color: rgba(234, 67, 53, 0.06); /* More subtle transparency */
    border: 1px solid rgba(234, 67, 53, 0.3); /* Softer border */
    padding: 1px; /* Minimal padding */
}
"""

    TOOL_HEADER = """
QLabel {
    font-weight: 500; /* Lighter weight for subtlety */
    color: #666666; /* More muted gray */
}
"""

    TOOL_TOGGLE_BUTTON = """
QPushButton {
    background-color: transparent;
    border: none;
    color: #999999; /* More muted gray */
    font-weight: normal; /* Reduced from bold */
    font-size: 11px; /* Smaller font */
}
QPushButton:hover {
    color: #4285f4; /* Blue */
}
"""

    TOOL_STATUS = """
QLabel {
    font-weight: 500; /* Lighter than bold */
    padding: 2px; /* Reduced padding */
    font-size: 11px; /* Smaller status text */
}
QLabel[status="running"] {
    color: #ff9800; /* Orange */
}
QLabel[status="complete"] {
    color: #34a853; /* Green */
}
QLabel[status="error"] {
    color: #ea4335; /* Red */
}
"""

    TOOL_CONTENT = """
QLabel {
    color: #666666; /* More muted text */
    padding: 1px; /* Minimal padding */
    font-size: 11px; /* Smaller content text */
}
QLabel[role="title"] {
    font-weight: 500; /* Lighter than bold */
    color: #666666;
    font-size: 12px;
}
QLabel[role="key"] {
    color: #3367d6; /* Slightly muted blue */
    font-weight: 500;
    font-size: 11px;
}
QLabel[role="value"] {
    color: #777777; /* More muted */
    font-size: 11px;
}
QLabel[role="error"] {
    color: #ea4335; /* Red for error messages */
    font-size: 11px;
}
"""

    TOOL_PROGRESS = """
QProgressBar {
    border: none;
    background-color: rgba(224, 224, 224, 0.4); /* Light gray with transparency */
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #4285f4; /* Blue */
    border-radius: 5px;
}
"""

    TOOL_SEPARATOR = """
QFrame {
    background-color: rgba(66, 133, 244, 0.3); /* Blue with transparency */
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
        "background": "#fafafa",
        "text": "#383a42",
        "border": "#e5e5e6",
        "string": "#50a14f",  # Green
        "number": "#986801",  # Orange
        "keyword": "#a626a4",  # Purple
        "punctuation": "#383a42",  # Dark gray
        "error": "#e45649",  # Red
    }

    JSON_EDITOR_STYLE = """
QPlainTextEdit {
    background-color: #fafafa;
    color: #383a42;
    border: 1px solid #e5e5e6;
    border-radius: 4px;
    padding: 8px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.4;
}
QPlainTextEdit:focus {
    border: 1px solid #4285f4; /* Blue focus border */
}
"""

    # Markdown Editor styles
    MARKDOWN_EDITOR_COLORS = {
        "background": "#fafafa",  # Light background
        "text": "#383a42",  # Dark text
        "border": "#e5e5e6",  # Light border
        "header": "#4285f4",  # Blue - for headers
        "bold": "#986801",  # Orange - for bold text
        "italic": "#50a14f",  # Green - for italic text
        "code": "#a626a4",  # Purple - for code blocks
        "code_background": "#f0f0f0",  # Very light gray - code background
        "link": "#0066cc",  # Blue - for links
        "image": "#9c27b0",  # Purple - for images
        "list": "#ff6600",  # Orange - for list markers
        "blockquote": "#666666",  # Gray - for blockquotes
        "hr": "#cccccc",  # Light gray - for horizontal rules
        "strikethrough": "#ea4335",  # Red - for strikethrough text
        "error": "#ea4335",  # Red - for errors
    }

    MARKDOWN_EDITOR_STYLE = """
QPlainTextEdit {
    background-color: #fafafa; /* Light background */
    color: #383a42; /* Dark text */
    border: 1px solid #e5e5e6; /* Light border */
    border-radius: 4px;
    padding: 8px;
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
    line-height: 1.6;
    selection-background-color: #e8f0fe; /* Light blue selection */
    selection-color: #383a42; /* Dark text */
}
QPlainTextEdit:focus {
    border: 1px solid #4285f4; /* Blue focus border */
}
"""
