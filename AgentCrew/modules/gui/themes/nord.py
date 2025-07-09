"""Nord theme styles for AgentCrew GUI."""


class NordTheme:
    """Static class containing Nord theme styles.

    Based on the Nord color palette: https://www.nordtheme.com/

    Nord Palette:
    - Polar Night: #2e3440, #3b4252, #434c5e, #4c566a
    - Snow Storm: #d8dee9, #e5e9f0, #eceff4
    - Frost: #8fbcbb, #88c0d0, #81a1c1, #5e81ac
    - Aurora: #bf616a, #d08770, #ebcb8b, #a3be8c, #b48ead
    """

    # Main application styles
    MAIN_STYLE = """
QMainWindow {
    background-color: #2e3440; /* Nord Polar Night 0 */
}
QScrollArea {
    border: none;
    background-color: #3b4252; /* Nord Polar Night 1 */
}
QWidget#chatContainer { /* Specific ID for chat_container */
    background-color: #2e3440; /* Nord Polar Night 1 */
}
QSplitter::handle {
    background-color: #434c5e; /* Nord Polar Night 2 */
}
QSplitter::handle:hover {
    background-color: #4c566a; /* Nord Polar Night 3 */
}
QSplitter::handle:pressed {
    background-color: #5e81ac; /* Nord Frost 3 */
}
QStatusBar {
    background-color: #3b4252; /* Nord Polar Night 0 */
    color: #d8dee9; /* Nord Snow Storm 0 */
}
QToolTip {
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    padding: 4px;
}
QMessageBox {
    background-color: #3b4252; /* Nord Polar Night 1 */
}
QMessageBox QLabel { /* For message text in QMessageBox */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    background-color: transparent;
}
/* QCompleter's popup is often a QListView */
QListView { /* General style for QListView, affects completer */
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    padding: 2px;
    outline: 0px;
}
QListView::item {
    padding: 4px 8px;
    border-radius: 2px;
}
QListView::item:selected {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
}
QListView::item:hover {
    background-color: #4c566a; /* Nord Polar Night 3 */
}

/* Modern Scrollbar Styles */
QScrollBar:vertical {
    border: none;
    background: #3b4252; /* Nord Polar Night 1 - Track background */
    width: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #4c566a; /* Nord Polar Night 3 - Handle color */
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #5e81ac; /* Nord Frost 3 - Handle hover color */
}
QScrollBar::handle:vertical:pressed {
    background: #81a1c1; /* Nord Frost 2 - Handle pressed color */
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
    background: #3b4252; /* Nord Polar Night 1 - Track background */
    height: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:horizontal {
    background: #4c566a; /* Nord Polar Night 3 - Handle color */
    min-width: 20px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: #5e81ac; /* Nord Frost 3 - Handle hover color */
}
QScrollBar::handle:horizontal:pressed {
    background: #81a1c1; /* Nord Frost 2 - Handle pressed color */
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
    background-color: #3b4252; /* Nord Polar Night 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    padding: 4px;
    border-radius: 6px;
}
QLabel QMenu::item {
    color: #e5e9f0; /* Nord Snow Storm 1 */
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    margin: 2px;
}
QLabel QMenu::item:selected {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
}
QLabel QMenu::item:pressed {
    background-color: #81a1c1; /* Nord Frost 2 */
}
QLabel QMenu::separator {
    height: 1px;
    background: #4c566a; /* Nord Polar Night 3 */
    margin: 4px 8px;
}
"""

    # Button styles
    PRIMARY_BUTTON = """
QPushButton {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #81a1c1; /* Nord Frost 2 */
}
QPushButton:pressed {
    background-color: #88c0d0; /* Nord Frost 1 */
}
QPushButton:disabled {
    background-color: #4c566a; /* Nord Polar Night 3 */
    color: #434c5e; /* Nord Polar Night 2 */
}
"""

    SECONDARY_BUTTON = """
QPushButton {
    background-color: #4c566a; /* Nord Polar Night 3 */
    color: #d8dee9; /* Nord Snow Storm 0 */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #434c5e; /* Nord Polar Night 2 */
}
QPushButton:pressed {
    background-color: #5e81ac; /* Nord Frost 3 */
}
QPushButton:disabled {
    background-color: #3b4252; /* Nord Polar Night 1 */
    color: #434c5e; /* Nord Polar Night 2 */
}
"""

    STOP_BUTTON = """
QPushButton {
    background-color: #bf616a; /* Nord Aurora Red */
    color: #eceff4; /* Nord Snow Storm 2 */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #d08770; /* Nord Aurora Orange */
}
QPushButton:pressed {
    background-color: #bf616a; /* Nord Aurora Red (darker) */
}
"""

    RED_BUTTON = """
QPushButton {
    background-color: #bf616a; /* Nord Aurora Red */
    color: #eceff4; /* Nord Snow Storm 2 */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #d08770; /* Nord Aurora Orange */
}
QPushButton:pressed {
    background-color: #bf616a; /* Nord Aurora Red (darker) */
}
QPushButton:disabled {
    background-color: #4c566a; /* Nord Polar Night 3 */
    color: #434c5e; /* Nord Polar Night 2 */
}
"""

    GREEN_BUTTON = """
QPushButton {
    background-color: #a3be8c; /* Nord Aurora Green */
    color: #2e3440; /* Nord Polar Night 0 */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #8fbcbb; /* Nord Frost 0 */
}
QPushButton:pressed {
    background-color: #a3be8c; /* Nord Aurora Green (darker) */
}
QPushButton:disabled {
    background-color: #4c566a; /* Nord Polar Night 3 */
    color: #434c5e; /* Nord Polar Night 2 */
}
"""

    API_KEYS_GROUP = """
QGroupBox {
    background-color: #3b4252; /* Nord Polar Night 1 */
}
QLabel {
    background-color: #3b4252; 
}
"""

    EDITOR_CONTAINER_WIDGET = """
QWidget {
    background-color: #3b4252; /* Nord Polar Night 1 */
}
"""

    MENU_BUTTON = """
QPushButton {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
    padding-left: 12px;
}
QPushButton:hover {
    background-color: #81a1c1; /* Nord Frost 2 */
}
QPushButton:pressed {
    background-color: #88c0d0; /* Nord Frost 1 */
}
QPushButton:disabled {
    background-color: #4c566a; /* Nord Polar Night 3 */
    color: #434c5e; /* Nord Polar Night 2 */
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
    background-color: #4c566a; /* Nord Polar Night 3 */
    color: #434c5e; /* Nord Polar Night 2 */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
"""

    STOP_BUTTON_STOPPING = """
QPushButton {
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #4c566a; /* Nord Polar Night 3 */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
"""

    COMBO_BOX = """
QComboBox {
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    border-radius: 4px;
    padding: 6px 8px;
    min-width: 120px;
}
QComboBox:hover {
    border: 1px solid #5e81ac; /* Nord Frost 3 */
}
QComboBox:focus {
    border: 1px solid #88c0d0; /* Nord Frost 1 */
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: #4c566a; /* Nord Polar Night 3 */
    border-left-style: solid;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
    background-color: #4c566a; /* Nord Polar Night 3 */
}
QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #e5e9f0; /* Nord Snow Storm 1 */
}
QComboBox QAbstractItemView {
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    border-radius: 4px;
    selection-background-color: #5e81ac; /* Nord Frost 3 */
    selection-color: #eceff4; /* Nord Snow Storm 2 */
    outline: 0px;
}
"""

    # Input styles
    TEXT_INPUT = """
QTextEdit {
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    border-radius: 4px;
    padding: 8px;
}
QTextEdit:focus {
    border: 1px solid #88c0d0; /* Nord Frost 1 */
}
"""

    LINE_EDIT = """
QLineEdit {
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    border-radius: 4px;
    padding: 8px;
}
QLineEdit:focus {
    border: 1px solid #88c0d0; /* Nord Frost 1 */
}
"""

    # Menu styles
    MENU_BAR = """
QMenuBar {
    background-color: #3b4252; /* Nord Polar Night 1 */
    color: #d8dee9; /* Nord Snow Storm 0 */
    padding: 2px;
}
QMenuBar::item {
    background-color: transparent;
    color: #d8dee9; /* Nord Snow Storm 0 */
    padding: 4px 12px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #434c5e; /* Nord Polar Night 2 */
}
QMenuBar::item:pressed {
    background-color: #4c566a; /* Nord Polar Night 3 */
}
QMenu {
    background-color: #3b4252; /* Nord Polar Night 1 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
}
QMenu::separator {
    height: 1px;
    background: #4c566a; /* Nord Polar Night 3 */
    margin-left: 10px;
    margin-right: 5px;
}
"""

    CONTEXT_MENU = """
QMenu {
    background-color: #3b4252; /* Nord Polar Night 1 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    padding: 4px;
    border-radius: 6px;
}
QMenu::item {
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    margin: 2px;
    color: #eceff4; /* Brighter text color for better visibility */
}
QMenu::item:selected {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #ffffff; /* Pure white for selected items */
}
QMenu::item:pressed {
    background-color: #81a1c1; /* Nord Frost 2 */
}
QMenu::item:disabled {
    background-color: transparent;
    color: #4c566a; /* Nord Polar Night 3 - muted color for disabled items */
}
QMenu::separator {
    height: 1px;
    background: #4c566a; /* Nord Polar Night 3 */
    margin: 4px 8px;
}
"""

    AGENT_MENU = """
QMenu {
    background-color: #3b4252; /* Nord Polar Night 1 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #434c5e; /* Nord Polar Night 2 */
    border-radius: 4px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 20px;
    background-color: transparent;
}
QMenu::item:selected {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
    border-radius: 3px;
}
QMenu::separator {
    height: 1px;
    background-color: #434c5e; /* Nord Polar Night 2 */
    margin-left: 5px;
    margin-right: 5px;
}
"""

    # Label styles
    STATUS_INDICATOR = """
QLabel {
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    padding: 8px; 
    border-radius: 5px;
    font-weight: bold;
}
"""

    VERSION_LABEL = """
QLabel {
    color: #d8dee9; /* Nord Snow Storm 0 */
    padding: 2px 8px;
    font-size: 11px;
}
"""

    SYSTEM_MESSAGE_TOGGLE_BUTTON = """
QPushButton {
    background-color: transparent;
    color: #8fbcbb; /* Nord Frost 0 */
    border: none;
    text-align: left;
    padding: 2px;
}
QPushButton:hover {
    color: #88c0d0; /* Nord Frost 1 */
}
"""

    # Widget-specific styles
    SIDEBAR = """
background-color: #3b4252; /* Nord Polar Night 1 */
"""

    CONVERSATION_LIST = """
QListWidget {
    background-color: #2e3440; /* Nord Polar Night 0 */
    border: 1px solid #434c5e; /* Nord Polar Night 2 */
    border-radius: 4px;
}
QListWidget::item {
    color: #e5e9f0; /* Nord Snow Storm 1 */
    background-color: #2e3440; /* Nord Polar Night 0 */
    border: none;
    border-bottom: 1px solid #434c5e; /* Nord Polar Night 2 */
    margin: 0px;
    padding: 8px;
}
QListWidget::item:selected {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
}
QListWidget::item:hover:!selected {
    background-color: #434c5e; /* Nord Polar Night 2 */
}
"""

    SEARCH_BOX = """
QLineEdit {
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    border-radius: 4px;
    padding: 8px;
}
QLineEdit:focus {
    border: 1px solid #88c0d0; /* Nord Frost 1 */
}
"""

    TOKEN_USAGE = """
QLabel {
    color: #d8dee9; /* Nord Snow Storm 0 */
    font-weight: bold;
    padding: 8px;
    background-color: #2e3440; /* Nord Polar Night 0 */
    border-top: 1px solid #434c5e; /* Nord Polar Night 2 */
}
"""

    TOKEN_USAGE_WIDGET = """
background-color: #2e3440; /* Nord Polar Night 0 */
"""

    # Message bubble styles
    USER_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #5e81ac; /* Nord Frost 3 */
    border: none;
    padding: 2px;
}
"""

    ASSISTANT_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #434c5e; /* Nord Polar Night 2 */
    border: none;
    padding: 2px;
}
"""

    THINKING_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #4c566a; /* Nord Polar Night 3 */
    border: none;
    padding: 2px;
}
"""

    CONSOLIDATED_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #434c5e; /* Nord Polar Night 2 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    padding: 2px;
}
"""

    ROLLBACK_BUTTON = """
QPushButton {
    background-color: #b48ead; /* Nord Aurora Purple */
    color: #eceff4; /* Nord Snow Storm 2 */
    border: none;
    border-radius: 15px;
    padding: 8px;
    font-size: 24px;
    font-weight: bold;
    width: 30px;
    height: 30px;
}
QPushButton:hover {
    background-color: #8fbcbb; /* Nord Frost 0 */
}
QPushButton:pressed {
    background-color: #b48ead; /* Nord Aurora Purple (darker) */
}
"""

    CONSOLIDATED_BUTTON = """
QPushButton {
    background-color: #b48ead; /* Nord Aurora Purple */
    color: #eceff4; /* Nord Snow Storm 2 */
    border: none;
    border-radius: 15px;
    padding: 8px;
    font-size: 16px;
    font-weight: bold;
    width: 30px;
    height: 30px;
}
QPushButton:hover {
    background-color: #8fbcbb; /* Nord Frost 0 */
}
QPushButton:pressed {
    background-color: #b48ead; /* Nord Aurora Purple (darker) */
}
"""

    # Tool dialog styles
    TOOL_DIALOG_TEXT_EDIT = """
QTextEdit {
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    border-radius: 4px;
    padding: 8px;
    font-family: monospace;
}
QTextEdit:focus {
    border: 1px solid #88c0d0; /* Nord Frost 1 */
}
"""

    TOOL_DIALOG_YES_BUTTON = """
QPushButton {
    background-color: #a3be8c; /* Nord Aurora Green */
    color: #2e3440; /* Nord Polar Night 0 */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #8fbcbb; /* Nord Frost 0 */
}
"""

    TOOL_DIALOG_ALL_BUTTON = """
QPushButton {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #81a1c1; /* Nord Frost 2 */
}
"""

    TOOL_DIALOG_NO_BUTTON = """
QPushButton {
    background-color: #bf616a; /* Nord Aurora Red */
    color: #eceff4; /* Nord Snow Storm 2 */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #d08770; /* Nord Aurora Orange */
}
"""

    AGENT_MENU_BUTTON = """
QPushButton {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
    padding-left: 12px;
}
QPushButton:hover {
    background-color: #81a1c1; /* Nord Frost 2 */
}
QPushButton:pressed {
    background-color: #88c0d0; /* Nord Frost 1 */
}
QPushButton:disabled {
    background-color: #4c566a; /* Nord Polar Night 3 */
    color: #434c5e; /* Nord Polar Night 2 */
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
color: #d8dee9; /* Nord Snow Storm 0 */
padding: 2px;
"""

    SYSTEM_MESSAGE_TOGGLE = """
QPushButton {
    background-color: transparent;
    color: #8fbcbb; /* Nord Frost 0 */
    border: none;
    text-align: left;
    padding: 2px;
}
QPushButton:hover {
    color: #88c0d0; /* Nord Frost 1 */
}
"""

    # Config window styles
    CONFIG_DIALOG = """
QDialog {
    background-color: #2e3440; /* Nord Polar Night 0 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
}
QTabWidget::pane {
    border: 1px solid #434c5e; /* Nord Polar Night 2 */
    background-color: #3b4252; /* Nord Polar Night 1 */
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
    padding: 8px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #3b4252; /* Nord Polar Night 1 (same as pane) */
    border-bottom-color: #3b4252; /* Nord Polar Night 1 */
    color: #88c0d0; /* Nord Frost 1 for selected tab text */
}
QTabBar::tab:hover:!selected {
    background-color: #4c566a; /* Nord Polar Night 3 */
}
QPushButton {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #81a1c1; /* Nord Frost 2 */
}
QPushButton:pressed {
    background-color: #88c0d0; /* Nord Frost 1 */
}
QPushButton:disabled {
    background-color: #4c566a; /* Nord Polar Night 3 */
    color: #434c5e; /* Nord Polar Night 2 */
}
QListWidget {
    background-color: #2e3440; /* Nord Polar Night 0 */
    border: 1px solid #434c5e; /* Nord Polar Night 2 */
    border-radius: 4px;
    padding: 4px;
    color: #e5e9f0; /* Nord Snow Storm 1 */
}
QListWidget::item {
    padding: 6px;
    border-radius: 2px;
    color: #e5e9f0; /* Nord Snow Storm 1 */
    background-color: #2e3440; /* Nord Polar Night 0 */
}
QListWidget::item:selected {
    background-color: #5e81ac; /* Nord Frost 3 */
    color: #eceff4; /* Nord Snow Storm 2 */
}
QListWidget::item:hover:!selected {
    background-color: #434c5e; /* Nord Polar Night 2 */
}
QLineEdit, QTextEdit {
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    border-radius: 4px;
    padding: 6px;
    background-color: #434c5e; /* Nord Polar Night 2 */
    color: #e5e9f0; /* Nord Snow Storm 1 */
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #88c0d0; /* Nord Frost 1 */
}
QCheckBox {
    spacing: 8px;
    color: #e5e9f0; /* Nord Snow Storm 1 */
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 2px;
    background-color: #4c566a; /* Nord Polar Night 3 */
    border: 1px solid #434c5e; /* Nord Polar Night 2 */
}
QCheckBox::indicator:checked {
    background-color: #5e81ac; /* Nord Frost 3 */
    border: 1px solid #5e81ac; /* Nord Frost 3 */
}
QCheckBox::indicator:hover {
    border: 1px solid #88c0d0; /* Nord Frost 1 */
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    background-color: #3b4252; /* Nord Polar Night 1 for groupbox background */
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px 4px 4px;
    color: #88c0d0; /* Nord Frost 1 for title */
    background-color: #3b4252; /* Match groupbox background */
    left: 10px;
}
QScrollArea {
    background-color: #3b4252; /* Nord Polar Night 1 */
    border: none;
}
QScrollArea > QWidget > QWidget {
     background-color: #3b4252; /* Nord Polar Night 1 */
}
QLabel {
    color: #e5e9f0; /* Nord Snow Storm 1 */
    padding: 2px;
}
QSplitter::handle {
    background-color: #434c5e; /* Nord Polar Night 2 */
}
QSplitter::handle:hover {
    background-color: #4c566a; /* Nord Polar Night 3 */
}
QSplitter::handle:pressed {
    background-color: #5e81ac; /* Nord Frost 3 */
}
"""

    PANEL = """
background-color: #3b4252; /* Nord Polar Night 1 */
"""

    SCROLL_AREA = """
background-color: #3b4252; /* Nord Polar Night 1 */
border: none;
"""

    EDITOR_WIDGET = """
background-color: #3b4252; /* Nord Polar Night 1 */
"""

    GROUP_BOX = """
background-color: #2e3440; /* Nord Polar Night 0 */
"""

    SPLITTER_COLOR = """
QSplitter::handle {
    background-color: #434c5e; /* Nord Polar Night 2 */
}
QSplitter::handle:hover {
    background-color: #4c566a; /* Nord Polar Night 3 */
}
QSplitter::handle:pressed {
    background-color: #5e81ac; /* Nord Frost 3 */
}
"""

    METADATA_HEADER_LABEL = """
QLabel {
    color: #d8dee9; /* Nord Snow Storm 0 */
    font-style: italic;
    padding-bottom: 5px;
}
"""

    # Message label styles
    USER_MESSAGE_LABEL = """
QLabel {
    color: #eceff4; /* Nord Snow Storm 2 */
}
"""

    ASSISTANT_MESSAGE_LABEL = """
QLabel {
    color: #e5e9f0; /* Nord Snow Storm 1 */
}
"""

    THINKING_MESSAGE_LABEL = """
QLabel {
    color: #d8dee9; /* Nord Snow Storm 0 */
}
"""

    # Sender label styles
    USER_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #eceff4; /* Nord Snow Storm 2 */
    padding: 2px;
}
"""

    ASSISTANT_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #e5e9f0; /* Nord Snow Storm 1 */
    padding: 2px;
}
"""

    THINKING_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #d8dee9; /* Nord Snow Storm 0 */
    padding: 2px;
}
"""

    # File display label styles
    USER_FILE_NAME_LABEL = """
QLabel {
    font-weight: bold;
    color: #eceff4; /* Nord Snow Storm 2 */
    padding-left: 10px;
}
"""

    ASSISTANT_FILE_NAME_LABEL = """
QLabel {
    font-weight: bold;
    color: #e5e9f0; /* Nord Snow Storm 1 */
    padding-left: 10px;
}
"""

    USER_FILE_INFO_LABEL = """
QLabel {
    color: #81a1c1; /* Nord Frost 2 - good contrast on user bubble */
}
"""

    ASSISTANT_FILE_INFO_LABEL = """
QLabel {
    color: #4c566a; /* Nord Polar Night 3 */
}
"""

    CODE_CSS = """
pre { line-height: 1; background-color: #3b4252; border-radius: 8px; padding: 12px; color: #e5e9f0; white-space: pre-wrap; word-wrap: break-word; } /* Nord Polar Night 1, Snow Storm 1 */
td.linenos .normal { color: #4c566a; background-color: transparent; padding-left: 5px; padding-right: 5px; } /* Nord Polar Night 3 */
span.linenos { color: #4c566a; background-color: transparent; padding-left: 5px; padding-right: 5px; } /* Nord Polar Night 3 */
td.linenos .special { color: #e5e9f0; background-color: #434c5e; padding-left: 5px; padding-right: 5px; } /* Snow Storm 1, Polar Night 2 */
span.linenos.special { color: #e5e9f0; background-color: #434c5e; padding-left: 5px; padding-right: 5px; } /* Snow Storm 1, Polar Night 2 */
.codehilite .hll { background-color: #434c5e } /* Nord Polar Night 2 */
.codehilite { background: #3b4252; border-radius: 8px; padding: 10px; color: #e5e9f0; } /* Nord Polar Night 1, Snow Storm 1 */
.codehilite .c { color: #4c566a; font-style: italic } /* Comment -> Polar Night 3 */
.codehilite .err { border: 1px solid #bf616a; color: #bf616a; } /* Error -> Aurora Red */
.codehilite .k { color: #81a1c1; font-weight: bold } /* Keyword -> Frost 2 */
.codehilite .o { color: #8fbcbb } /* Operator -> Frost 0 */
.codehilite .ch { color: #4c566a; font-style: italic } /* Comment.Hashbang -> Polar Night 3 */
.codehilite .cm { color: #4c566a; font-style: italic } /* Comment.Multiline -> Polar Night 3 */
.codehilite .cp { color: #ebcb8b } /* Comment.Preproc -> Aurora Yellow */
.codehilite .cpf { color: #4c566a; font-style: italic } /* Comment.PreprocFile -> Polar Night 3 */
.codehilite .c1 { color: #4c566a; font-style: italic } /* Comment.Single -> Polar Night 3 */
.codehilite .cs { color: #4c566a; font-style: italic } /* Comment.Special -> Polar Night 3 */
.codehilite .gd { color: #bf616a } /* Generic.Deleted -> Aurora Red */
.codehilite .ge { font-style: italic } /* Generic.Emph */
.codehilite .ges { font-weight: bold; font-style: italic } /* Generic.EmphStrong */
.codehilite .gr { color: #bf616a } /* Generic.Error -> Aurora Red */
.codehilite .gh { color: #5e81ac; font-weight: bold } /* Generic.Heading -> Frost 3 */
.codehilite .gi { color: #a3be8c } /* Generic.Inserted -> Aurora Green */
.codehilite .go { color: #e5e9f0 } /* Generic.Output -> Snow Storm 1 */
.codehilite .gp { color: #5e81ac; font-weight: bold } /* Generic.Prompt -> Frost 3 */
.codehilite .gs { font-weight: bold } /* Generic.Strong */
.codehilite .gu { color: #5e81ac; font-weight: bold } /* Generic.Subheading -> Frost 3 */
.codehilite .gt { color: #bf616a } /* Generic.Traceback -> Aurora Red */
.codehilite .kc { color: #81a1c1; font-weight: bold } /* Keyword.Constant -> Frost 2 */
.codehilite .kd { color: #81a1c1; font-weight: bold } /* Keyword.Declaration -> Frost 2 */
.codehilite .kn { color: #81a1c1; font-weight: bold } /* Keyword.Namespace -> Frost 2 */
.codehilite .kp { color: #81a1c1 } /* Keyword.Pseudo -> Frost 2 */
.codehilite .kr { color: #81a1c1; font-weight: bold } /* Keyword.Reserved -> Frost 2 */
.codehilite .kt { color: #d08770; font-weight: bold } /* Keyword.Type -> Aurora Orange */
.codehilite .m { color: #b48ead } /* Literal.Number -> Aurora Purple */
.codehilite .s { color: #a3be8c } /* Literal.String -> Aurora Green */
.codehilite .na { color: #88c0d0 } /* Name.Attribute -> Frost 1 */
.codehilite .nb { color: #5e81ac } /* Name.Builtin -> Frost 3 */
.codehilite .nc { color: #ebcb8b; font-weight: bold } /* Name.Class -> Aurora Yellow */
.codehilite .no { color: #d08770 } /* Name.Constant -> Aurora Orange */
.codehilite .nd { color: #81a1c1 } /* Name.Decorator -> Frost 2 */
.codehilite .ni { color: #e5e9f0; font-weight: bold } /* Name.Entity -> Snow Storm 1 */
.codehilite .ne { color: #bf616a; font-weight: bold } /* Name.Exception -> Aurora Red */
.codehilite .nf { color: #5e81ac; font-weight: bold } /* Name.Function -> Frost 3 */
.codehilite .nl { color: #e5e9f0 } /* Name.Label -> Snow Storm 1 */
.codehilite .nn { color: #ebcb8b; font-weight: bold } /* Name.Namespace -> Aurora Yellow */
.codehilite .nt { color: #81a1c1; font-weight: bold } /* Name.Tag -> Frost 2 */
.codehilite .nv { color: #eceff4 } /* Name.Variable -> Snow Storm 2 */
.codehilite .ow { color: #8fbcbb; font-weight: bold } /* Operator.Word -> Frost 0 */
.codehilite .w { color: #4c566a } /* Text.Whitespace -> Polar Night 3 */
.codehilite .mb { color: #b48ead } /* Literal.Number.Bin -> Aurora Purple */
.codehilite .mf { color: #b48ead } /* Literal.Number.Float -> Aurora Purple */
.codehilite .mh { color: #b48ead } /* Literal.Number.Hex -> Aurora Purple */
.codehilite .mi { color: #b48ead } /* Literal.Number.Integer -> Aurora Purple */
.codehilite .mo { color: #b48ead } /* Literal.Number.Oct -> Aurora Purple */
.codehilite .sa { color: #a3be8c } /* Literal.String.Affix -> Aurora Green */
.codehilite .sb { color: #a3be8c } /* Literal.String.Backtick -> Aurora Green */
.codehilite .sc { color: #a3be8c } /* Literal.String.Char -> Aurora Green */
.codehilite .dl { color: #a3be8c } /* Literal.String.Delimiter -> Aurora Green */
.codehilite .sd { color: #4c566a; font-style: italic } /* Literal.String.Doc -> Polar Night 3 */
.codehilite .s2 { color: #a3be8c } /* Literal.String.Double -> Aurora Green */
.codehilite .se { color: #d08770; font-weight: bold } /* Literal.String.Escape -> Aurora Orange */
.codehilite .sh { color: #a3be8c } /* Literal.String.Heredoc -> Aurora Green */
.codehilite .si { color: #a3be8c; font-weight: bold } /* Literal.String.Interpol -> Aurora Green */
.codehilite .sx { color: #a3be8c } /* Literal.String.Other -> Aurora Green */
.codehilite .sr { color: #a3be8c } /* Literal.String.Regex -> Aurora Green */
.codehilite .s1 { color: #a3be8c } /* Literal.String.Single -> Aurora Green */
.codehilite .ss { color: #a3be8c } /* Literal.String.Symbol -> Aurora Green */
.codehilite .bp { color: #5e81ac } /* Name.Builtin.Pseudo -> Frost 3 */
.codehilite .fm { color: #5e81ac; font-weight: bold } /* Name.Function.Magic -> Frost 3 */
.codehilite .vc { color: #eceff4 } /* Name.Variable.Class -> Snow Storm 2 */
.codehilite .vg { color: #eceff4 } /* Name.Variable.Global -> Snow Storm 2 */
.codehilite .vi { color: #eceff4 } /* Name.Variable.Instance -> Snow Storm 2 */
.codehilite .vm { color: #eceff4 } /* Name.Variable.Magic -> Snow Storm 2 */
.codehilite .il { color: #b48ead } /* Literal.Number.Integer.Long -> Aurora Purple */
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
    background-color: rgba(94, 129, 172, 0.08); /* More subtle transparency */
    border: 1px solid rgba(94, 129, 172, 0.4); /* Softer border */
    padding: 1px; /* Minimal padding */
}
"""

    TOOL_CARD_ERROR = """
QFrame#toolCard {
    border-radius: 6px; /* Reduced from 12px for subtlety */
    background-color: rgba(191, 97, 106, 0.08); /* More subtle transparency */
    border: 1px solid rgba(191, 97, 106, 0.4); /* Softer border */
    padding: 1px; /* Minimal padding */
}
"""

    TOOL_HEADER = """
QLabel {
    font-weight: 500; /* Lighter weight for subtlety */
    color: #d8dee9; /* Nord4 - more muted than eceff4 */
}
"""

    TOOL_TOGGLE_BUTTON = """
QPushButton {
    background-color: transparent;
    border: none;
    color: #9ca0a4; /* More muted */
    font-weight: normal; /* Reduced from bold */
    font-size: 11px; /* Smaller font */
}
QPushButton:hover {
    color: #5e81ac; /* Nord Frost 3 */
}
"""

    TOOL_STATUS = """
QLabel {
    font-weight: 500; /* Lighter than bold */
    padding: 2px; /* Reduced padding */
    font-size: 11px; /* Smaller status text */
}
QLabel[status="running"] {
    color: #ebcb8b; /* Nord Aurora Yellow */
}
QLabel[status="complete"] {
    color: #a3be8c; /* Nord Aurora Green */
}
QLabel[status="error"] {
    color: #bf616a; /* Nord Aurora Red */
}
"""

    TOOL_CONTENT = """
QLabel {
    color: #d8dee9; /* Nord4 - more muted */
    padding: 1px; /* Minimal padding */
    font-size: 11px; /* Smaller content text */
}
QLabel[role="title"] {
    font-weight: 500; /* Lighter than bold */
    color: #d8dee9;
    font-size: 12px;
}
QLabel[role="key"] {
    color: #81a1c1; /* Nord10 - slightly muted */
    font-weight: 500;
    font-size: 11px;
}
QLabel[role="value"] {
    color: #9ca0a4; /* More muted */
    font-size: 11px;
}
QLabel[role="error"] {
    color: #bf616a; /* Nord Aurora Red for error messages */
    font-size: 11px;
}
"""

    TOOL_PROGRESS = """
QProgressBar {
    border: none;
    background-color: rgba(76, 86, 106, 0.4); /* Nord Polar Night with transparency */
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #5e81ac; /* Nord Frost 3 */
    border-radius: 5px;
}
"""

    TOOL_SEPARATOR = """
QFrame {
    background-color: rgba(94, 129, 172, 0.3); /* Nord Frost Blue with transparency */
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
        "background": "#2e3440",
        "text": "#d8dee9",
        "border": "#4c566a",
        "string": "#a3be8c",  # Green
        "number": "#d08770",  # Orange
        "keyword": "#b48ead",  # Purple
        "punctuation": "#d8dee9",  # Text
        "error": "#bf616a",  # Red
    }

    JSON_EDITOR_STYLE = """
QPlainTextEdit {
    background-color: #2e3440; /* Nord Polar Night */
    color: #d8dee9; /* Nord Snow Storm */
    border: 1px solid #4c566a; /* Nord Polar Night */
    border-radius: 4px;
    padding: 8px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.4;
}
QPlainTextEdit:focus {
    border: 1px solid #5e81ac; /* Nord Frost Blue focus border */
}
"""

    # Markdown Editor styles
    MARKDOWN_EDITOR_COLORS = {
        "background": "#2e3440",  # Nord Polar Night 0
        "text": "#d8dee9",  # Nord Snow Storm 0
        "border": "#4c566a",  # Nord Polar Night 3
        "header": "#5e81ac",  # Nord Frost 3 - for headers
        "bold": "#d08770",  # Nord Aurora Orange - for bold text
        "italic": "#a3be8c",  # Nord Aurora Green - for italic text
        "code": "#b48ead",  # Nord Aurora Purple - for code blocks
        "code_background": "#3b4252",  # Nord Polar Night 1 - code background
        "link": "#88c0d0",  # Nord Frost 1 - for links
        "image": "#81a1c1",  # Nord Frost 2 - for images
        "list": "#ebcb8b",  # Nord Aurora Yellow - for list markers
        "blockquote": "#8fbcbb",  # Nord Frost 0 - for blockquotes
        "hr": "#4c566a",  # Nord Polar Night 3 - for horizontal rules
        "strikethrough": "#bf616a",  # Nord Aurora Red - for strikethrough text
        "error": "#bf616a",  # Nord Aurora Red - for errors
    }

    MARKDOWN_EDITOR_STYLE = """
QPlainTextEdit {
    background-color: #2e3440; /* Nord Polar Night 0 */
    color: #d8dee9; /* Nord Snow Storm 0 */
    border: 1px solid #4c566a; /* Nord Polar Night 3 */
    border-radius: 4px;
    padding: 8px;
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
    line-height: 1.6;
    selection-background-color: #434c5e; /* Nord Polar Night 2 */
    selection-color: #eceff4; /* Nord Snow Storm 2 */
}
QPlainTextEdit:focus {
    border: 1px solid #5e81ac; /* Nord Frost 3 focus border */
}
"""
