"""Dracula theme styles for AgentCrew GUI."""


class DraculaTheme:
    """Static class containing Dracula theme styles.

    Based on the Dracula color palette: https://draculatheme.com/contribute

    Dracula Palette:
    - Background: #282A36
    - Current Line: #44475A
    - Foreground: #F8F8F2
    - Comment: #6272A4
    - Cyan: #8BE9FD
    - Green: #50FA7B
    - Orange: #FFB86C
    - Pink: #FF79C6
    - Purple: #BD93F9
    - Red: #FF5555
    - Yellow: #F1FA8C
    """

    # Main application styles
    MAIN_STYLE = """
QMainWindow {
    background-color: #282A36; /* Dracula Background */
}
QScrollArea {
    border: none;
    background-color: #282A36; /* Dracula Background */
}
QWidget#chatContainer { /* Specific ID for chat_container */
    background-color: #282A36; /* Dracula Background */
}
QSplitter::handle {
    background-color: #44475A; /* Dracula Current Line */
}
QSplitter::handle:hover {
    background-color: #6272A4; /* Dracula Comment */
}
QSplitter::handle:pressed {
    background-color: #BD93F9; /* Dracula Purple */
}
QStatusBar {
    background-color: #282A36; /* Dracula Background */
    color: #F8F8F2; /* Dracula Foreground */
}
QToolTip {
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #6272A4; /* Dracula Comment */
    padding: 4px;
}
QMessageBox {
    background-color: #282A36; /* Dracula Background */
}
QMessageBox QLabel { /* For message text in QMessageBox */
    color: #F8F8F2; /* Dracula Foreground */
    background-color: transparent;
}
/* QCompleter's popup is often a QListView */
QListView { /* General style for QListView, affects completer */
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #6272A4; /* Dracula Comment */
    padding: 2px;
    outline: 0px;
}
QListView::item {
    padding: 4px 8px;
    border-radius: 2px;
}
QListView::item:selected {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
}
QListView::item:hover {
    background-color: #6272A4; /* Dracula Comment */
}

/* Modern Scrollbar Styles */
QScrollBar:vertical {
    border: none;
    background: #282A36; /* Dracula Background - Track background */
    width: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #44475A; /* Dracula Current Line - Handle color */
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #BD93F9; /* Dracula Purple - Handle hover color */
}
QScrollBar::handle:vertical:pressed {
    background: #FF79C6; /* Dracula Pink - Handle pressed color */
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
    background: #282A36; /* Dracula Background - Track background */
    height: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:horizontal {
    background: #44475A; /* Dracula Current Line - Handle color */
    min-width: 20px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: #BD93F9; /* Dracula Purple - Handle hover color */
}
QScrollBar::handle:horizontal:pressed {
    background: #FF79C6; /* Dracula Pink - Handle pressed color */
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
    background-color: #282A36; /* Dracula Background */
    border: 1px solid #6272A4; /* Dracula Comment */
    padding: 4px;
    border-radius: 6px;
}
QLabel QMenu::item {
    color: #F8F8F2; /* Dracula Foreground */
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    margin: 2px;
}
QLabel QMenu::item:selected {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
}
QLabel QMenu::item:pressed {
    background-color: #FF79C6; /* Dracula Pink */
}
QLabel QMenu::separator {
    height: 1px;
    background: #6272A4; /* Dracula Comment */
    margin: 4px 8px;
}
"""

    # Button styles
    PRIMARY_BUTTON = """
QPushButton {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #FF79C6; /* Dracula Pink */
}
QPushButton:pressed {
    background-color: #8BE9FD; /* Dracula Cyan */
}
QPushButton:disabled {
    background-color: #44475A; /* Dracula Current Line */
    color: #6272A4; /* Dracula Comment */
}
"""

    SECONDARY_BUTTON = """
QPushButton {
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #6272A4; /* Dracula Comment */
}
QPushButton:pressed {
    background-color: #BD93F9; /* Dracula Purple */
}
QPushButton:disabled {
    background-color: #282A36; /* Dracula Background */
    color: #44475A; /* Dracula Current Line */
}
"""

    STOP_BUTTON = """
QPushButton {
    background-color: #FF5555; /* Dracula Red */
    color: #F8F8F2; /* Dracula Foreground */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #FFB86C; /* Dracula Orange */
}
QPushButton:pressed {
    background-color: #FF5555; /* Dracula Red */
}
"""

    RED_BUTTON = """
QPushButton {
    background-color: #FF5555; /* Dracula Red */
    color: #F8F8F2; /* Dracula Foreground */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #FFB86C; /* Dracula Orange */
}
QPushButton:pressed {
    background-color: #FF5555; /* Dracula Red */
}
QPushButton:disabled {
    background-color: #44475A; /* Dracula Current Line */
    color: #6272A4; /* Dracula Comment */
}
"""

    GREEN_BUTTON = """
QPushButton {
    background-color: #50FA7B; /* Dracula Green */
    color: #282A36; /* Dracula Background */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #8BE9FD; /* Dracula Cyan */
}
QPushButton:pressed {
    background-color: #50FA7B; /* Dracula Green */
}
QPushButton:disabled {
    background-color: #44475A; /* Dracula Current Line */
    color: #6272A4; /* Dracula Comment */
}
"""

    API_KEYS_GROUP = """
QGroupBox {
    background-color: #282A36; /* Dracula Background */
}
QLabel {
    background-color: #282A36; /* Dracula Background */
}
"""

    EDITOR_CONTAINER_WIDGET = """
QWidget {
    background-color: #282A36; /* Dracula Background */
}
"""

    MENU_BUTTON = """
QPushButton {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
    padding-left: 12px;
}
QPushButton:hover {
    background-color: #FF79C6; /* Dracula Pink */
}
QPushButton:pressed {
    background-color: #8BE9FD; /* Dracula Cyan */
}
QPushButton:disabled {
    background-color: #44475A; /* Dracula Current Line */
    color: #6272A4; /* Dracula Comment */
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
    background-color: #44475A; /* Dracula Current Line */
    color: #6272A4; /* Dracula Comment */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
"""

    STOP_BUTTON_STOPPING = """
QPushButton {
    background-color: #44475A; /* Dracula Current Line */
    color: #6272A4; /* Dracula Comment */
    border: none;
    border-radius: 4px; 
    padding: 8px;
    font-weight: bold;
}
"""

    COMBO_BOX = """
QComboBox {
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #6272A4; /* Dracula Comment */
    border-radius: 4px;
    padding: 6px 8px;
    min-width: 120px;
}
QComboBox:hover {
    border: 1px solid #BD93F9; /* Dracula Purple */
}
QComboBox:focus {
    border: 1px solid #8BE9FD; /* Dracula Cyan */
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: #6272A4; /* Dracula Comment */
    border-left-style: solid;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
    background-color: #44475A; /* Dracula Current Line */
}
QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #F8F8F2; /* Dracula Foreground */
}
QComboBox QAbstractItemView {
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #6272A4; /* Dracula Comment */
    border-radius: 4px;
    selection-background-color: #BD93F9; /* Dracula Purple */
    selection-color: #F8F8F2; /* Dracula Foreground */
    outline: 0px;
}
"""

    # Input styles
    TEXT_INPUT = """
QTextEdit {
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #6272A4; /* Dracula Comment */
    border-radius: 4px;
    padding: 8px;
}
QTextEdit:focus {
    border: 1px solid #8BE9FD; /* Dracula Cyan */
}
"""

    LINE_EDIT = """
QLineEdit {
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #6272A4; /* Dracula Comment */
    border-radius: 4px;
    padding: 8px;
}
QLineEdit:focus {
    border: 1px solid #8BE9FD; /* Dracula Cyan */
}
"""

    # Menu styles
    MENU_BAR = """
QMenuBar {
    background-color: #282A36; /* Dracula Background */
    color: #F8F8F2; /* Dracula Foreground */
    padding: 2px;
}
QMenuBar::item {
    background-color: transparent;
    color: #F8F8F2; /* Dracula Foreground */
    padding: 4px 12px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #44475A; /* Dracula Current Line */
}
QMenuBar::item:pressed {
    background-color: #6272A4; /* Dracula Comment */
}
QMenu {
    background-color: #282A36; /* Dracula Background */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #6272A4; /* Dracula Comment */
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
}
QMenu::separator {
    height: 1px;
    background: #6272A4; /* Dracula Comment */
    margin-left: 10px;
    margin-right: 5px;
}
"""

    CONTEXT_MENU = """
QMenu {
    background-color: #282A36; /* Dracula Background */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #6272A4; /* Dracula Comment */
    padding: 4px;
    border-radius: 6px;
}
QMenu::item {
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    margin: 2px;
    color: #F8F8F2; /* Brighter text color for better visibility */
}
QMenu::item:selected {
    background-color: #BD93F9; /* Dracula Purple */
    color: #ffffff; /* Pure white for selected items */
}
QMenu::item:pressed {
    background-color: #FF79C6; /* Dracula Pink */
}
QMenu::item:disabled {
    background-color: transparent;
    color: #6272A4; /* Dracula Comment - muted color for disabled items */
}
QMenu::separator {
    height: 1px;
    background: #6272A4; /* Dracula Comment */
    margin: 4px 8px;
}
"""

    AGENT_MENU = """
QMenu {
    background-color: #282A36; /* Dracula Background */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #44475A; /* Dracula Current Line */
    border-radius: 4px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 20px;
    background-color: transparent;
}
QMenu::item:selected {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
    border-radius: 3px;
}
QMenu::separator {
    height: 1px;
    background-color: #44475A; /* Dracula Current Line */
    margin-left: 5px;
    margin-right: 5px;
}
"""

    # Label styles
    STATUS_INDICATOR = """
QLabel {
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    padding: 8px; 
    border-radius: 5px;
    font-weight: bold;
}
"""

    VERSION_LABEL = """
QLabel {
    color: #F8F8F2; /* Dracula Foreground */
    padding: 2px 8px;
    font-size: 11px;
}
"""

    SYSTEM_MESSAGE_TOGGLE_BUTTON = """
QPushButton {
    background-color: transparent;
    color: #8BE9FD; /* Dracula Cyan */
    border: none;
    text-align: left;
    padding: 2px;
}
QPushButton:hover {
    color: #FF79C6; /* Dracula Pink */
}
"""

    # Widget-specific styles
    SIDEBAR = """
background-color: #282A36; /* Dracula Background */
"""

    CONVERSATION_LIST = """
QListWidget {
    background-color: #282A36; /* Dracula Background */
    border: 1px solid #44475A; /* Dracula Current Line */
    border-radius: 4px;
}
QListWidget::item {
    color: #F8F8F2; /* Dracula Foreground */
    background-color: #282A36; /* Dracula Background */
    border: none;
    border-bottom: 1px solid #44475A; /* Dracula Current Line */
    margin: 0px;
    padding: 8px;
}
QListWidget::item:selected {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
}
QListWidget::item:hover:!selected {
    background-color: #44475A; /* Dracula Current Line */
}
"""

    SEARCH_BOX = """
QLineEdit {
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #6272A4; /* Dracula Comment */
    border-radius: 4px;
    padding: 8px;
}
QLineEdit:focus {
    border: 1px solid #8BE9FD; /* Dracula Cyan */
}
"""

    TOKEN_USAGE = """
QLabel {
    color: #F8F8F2; /* Dracula Foreground */
    font-weight: bold;
    padding: 8px;
    background-color: #282A36; /* Dracula Background */
    border-top: 1px solid #44475A; /* Dracula Current Line */
}
"""

    TOKEN_USAGE_WIDGET = """
background-color: #282A36; /* Dracula Background */
"""

    # Message bubble styles
    USER_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #BD93F9; /* Dracula Purple */
    border: none;
    padding: 2px;
}
"""

    ASSISTANT_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #44475A; /* Dracula Current Line */
    border: none;
    padding: 2px;
}
"""

    THINKING_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #6272A4; /* Dracula Comment */
    border: none;
    padding: 2px;
}
"""

    CONSOLIDATED_BUBBLE = """
QFrame { 
    border-radius: 5px; 
    background-color: #44475A; /* Dracula Current Line */
    border: 1px solid #6272A4; /* Dracula Comment */
    padding: 2px;
}
"""

    ROLLBACK_BUTTON = """
QPushButton {
    background-color: #FF79C6; /* Dracula Pink */
    color: #F8F8F2; /* Dracula Foreground */
    border: none;
    border-radius: 15px;
    padding: 8px;
    font-size: 24px;
    font-weight: bold;
    width: 30px;
    height: 30px;
}
QPushButton:hover {
    background-color: #8BE9FD; /* Dracula Cyan */
}
QPushButton:pressed {
    background-color: #FF79C6; /* Dracula Pink */
}
"""

    CONSOLIDATED_BUTTON = """
QPushButton {
    background-color: #FF79C6; /* Dracula Pink */
    color: #F8F8F2; /* Dracula Foreground */
    border: none;
    border-radius: 15px;
    padding: 8px;
    font-size: 16px;
    font-weight: bold;
    width: 30px;
    height: 30px;
}
QPushButton:hover {
    background-color: #8BE9FD; /* Dracula Cyan */
}
QPushButton:pressed {
    background-color: #FF79C6; /* Dracula Pink */
}
"""

    # Tool dialog styles
    TOOL_DIALOG_TEXT_EDIT = """
QTextEdit {
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    border: 1px solid #6272A4; /* Dracula Comment */
    border-radius: 4px;
    padding: 8px;
    font-family: monospace;
}
QTextEdit:focus {
    border: 1px solid #8BE9FD; /* Dracula Cyan */
}
"""

    TOOL_DIALOG_YES_BUTTON = """
QPushButton {
    background-color: #50FA7B; /* Dracula Green */
    color: #282A36; /* Dracula Background */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #8BE9FD; /* Dracula Cyan */
}
"""

    TOOL_DIALOG_ALL_BUTTON = """
QPushButton {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #FF79C6; /* Dracula Pink */
}
"""

    TOOL_DIALOG_NO_BUTTON = """
QPushButton {
    background-color: #FF5555; /* Dracula Red */
    color: #F8F8F2; /* Dracula Foreground */
    font-weight: bold;
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #FFB86C; /* Dracula Orange */
}
"""

    AGENT_MENU_BUTTON = """
QPushButton {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
    padding-left: 12px;
}
QPushButton:hover {
    background-color: #FF79C6; /* Dracula Pink */
}
QPushButton:pressed {
    background-color: #8BE9FD; /* Dracula Cyan */
}
QPushButton:disabled {
    background-color: #44475A; /* Dracula Current Line */
    color: #6272A4; /* Dracula Comment */
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
color: #F8F8F2; /* Dracula Foreground */
padding: 2px;
"""

    SYSTEM_MESSAGE_TOGGLE = """
QPushButton {
    background-color: transparent;
    color: #8BE9FD; /* Dracula Cyan */
    border: none;
    text-align: left;
    padding: 2px;
}
QPushButton:hover {
    color: #FF79C6; /* Dracula Pink */
}
"""

    # Config window styles
    CONFIG_DIALOG = """
QDialog {
    background-color: #282A36; /* Dracula Background */
    color: #F8F8F2; /* Dracula Foreground */
}
QTabWidget::pane {
    border: 1px solid #44475A; /* Dracula Current Line */
    background-color: #282A36; /* Dracula Background */
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
    padding: 8px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #6272A4; /* Dracula Comment */
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #282A36; /* Dracula Background (same as pane) */
    border-bottom-color: #282A36; /* Dracula Background */
    color: #8BE9FD; /* Dracula Cyan for selected tab text */
}
QTabBar::tab:hover:!selected {
    background-color: #6272A4; /* Dracula Comment */
}
QPushButton {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #FF79C6; /* Dracula Pink */
}
QPushButton:pressed {
    background-color: #8BE9FD; /* Dracula Cyan */
}
QPushButton:disabled {
    background-color: #44475A; /* Dracula Current Line */
    color: #6272A4; /* Dracula Comment */
}
QListWidget {
    background-color: #282A36; /* Dracula Background */
    border: 1px solid #44475A; /* Dracula Current Line */
    border-radius: 4px;
    padding: 4px;
    color: #F8F8F2; /* Dracula Foreground */
}
QListWidget::item {
    padding: 6px;
    border-radius: 2px;
    color: #F8F8F2; /* Dracula Foreground */
    background-color: #282A36; /* Dracula Background */
}
QListWidget::item:selected {
    background-color: #BD93F9; /* Dracula Purple */
    color: #F8F8F2; /* Dracula Foreground */
}
QListWidget::item:hover:!selected {
    background-color: #44475A; /* Dracula Current Line */
}
QLineEdit, QTextEdit {
    border: 1px solid #6272A4; /* Dracula Comment */
    border-radius: 4px;
    padding: 6px;
    background-color: #44475A; /* Dracula Current Line */
    color: #F8F8F2; /* Dracula Foreground */
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #8BE9FD; /* Dracula Cyan */
}
QCheckBox {
    spacing: 8px;
    color: #F8F8F2; /* Dracula Foreground */
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 2px;
    background-color: #44475A; /* Dracula Current Line */
    border: 1px solid #6272A4; /* Dracula Comment */
}
QCheckBox::indicator:checked {
    background-color: #BD93F9; /* Dracula Purple */
    border: 1px solid #BD93F9; /* Dracula Purple */
}
QCheckBox::indicator:hover {
    border: 1px solid #8BE9FD; /* Dracula Cyan */
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #6272A4; /* Dracula Comment */
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    background-color: #282A36; /* Dracula Background for groupbox background */
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px 4px 4px;
    color: #8BE9FD; /* Dracula Cyan for title */
    background-color: #282A36; /* Match groupbox background */
    left: 10px;
}
QScrollArea {
    background-color: #282A36; /* Dracula Background */
    border: none;
}
QScrollArea > QWidget > QWidget {
     background-color: #282A36; /* Dracula Background */
}
QLabel {
    color: #F8F8F2; /* Dracula Foreground */
    padding: 2px;
}
QSplitter::handle {
    background-color: #44475A; /* Dracula Current Line */
}
QSplitter::handle:hover {
    background-color: #6272A4; /* Dracula Comment */
}
QSplitter::handle:pressed {
    background-color: #BD93F9; /* Dracula Purple */
}
"""

    PANEL = """
background-color: #282A36; /* Dracula Background */
"""

    SCROLL_AREA = """
background-color: #282A36; /* Dracula Background */
border: none;
"""

    EDITOR_WIDGET = """
background-color: #282A36; /* Dracula Background */
"""

    GROUP_BOX = """
background-color: #282A36; /* Dracula Background */
"""

    SPLITTER_COLOR = """
QSplitter::handle {
    background-color: #44475A; /* Dracula Current Line */
}
QSplitter::handle:hover {
    background-color: #6272A4; /* Dracula Comment */
}
QSplitter::handle:pressed {
    background-color: #BD93F9; /* Dracula Purple */
}
"""

    METADATA_HEADER_LABEL = """
QLabel {
    color: #F8F8F2; /* Dracula Foreground */
    font-style: italic;
    padding-bottom: 5px;
}
"""

    # Message label styles
    USER_MESSAGE_LABEL = """
QLabel {
    color: #F8F8F2; /* Dracula Foreground */
}
"""

    ASSISTANT_MESSAGE_LABEL = """
QLabel {
    color: #F8F8F2; /* Dracula Foreground */
}
"""

    THINKING_MESSAGE_LABEL = """
QLabel {
    color: #F8F8F2; /* Dracula Foreground */
}
"""

    # Sender label styles
    USER_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #F8F8F2; /* Dracula Foreground */
    padding: 2px;
}
"""

    ASSISTANT_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #F8F8F2; /* Dracula Foreground */
    padding: 2px;
}
"""

    THINKING_SENDER_LABEL = """
QLabel {
    font-weight: bold;
    color: #F8F8F2; /* Dracula Foreground */
    padding: 2px;
}
"""

    # File display label styles
    USER_FILE_NAME_LABEL = """
QLabel {
    font-weight: bold;
    color: #F8F8F2; /* Dracula Foreground */
    padding-left: 10px;
}
"""

    ASSISTANT_FILE_NAME_LABEL = """
QLabel {
    font-weight: bold;
    color: #F8F8F2; /* Dracula Foreground */
    padding-left: 10px;
}
"""

    USER_FILE_INFO_LABEL = """
QLabel {
    color: #F1FA8C; /* Dracula Yellow - good contrast on user bubble */
}
"""

    ASSISTANT_FILE_INFO_LABEL = """
QLabel {
    color: #6272A4; /* Dracula Comment */
}
"""

    CODE_CSS = """
pre { line-height: 1; background-color: #282A36; border-radius: 8px; padding: 12px; color: #F8F8F2; white-space: pre-wrap; word-wrap: break-word; } /* Dracula Background, Foreground */
td.linenos .normal { color: #6272A4; background-color: transparent; padding-left: 5px; padding-right: 5px; } /* Dracula Comment */
span.linenos { color: #6272A4; background-color: transparent; padding-left: 5px; padding-right: 5px; } /* Dracula Comment */
td.linenos .special { color: #F8F8F2; background-color: #44475A; padding-left: 5px; padding-right: 5px; } /* Dracula Foreground, Current Line */
span.linenos.special { color: #F8F8F2; background-color: #44475A; padding-left: 5px; padding-right: 5px; } /* Dracula Foreground, Current Line */
.codehilite .hll { background-color: #44475A } /* Dracula Current Line */
.codehilite { background: #282A36; border-radius: 8px; padding: 10px; color: #F8F8F2; } /* Dracula Background, Foreground */
.codehilite .c { color: #6272A4; font-style: italic } /* Comment -> Dracula Comment */
.codehilite .err { border: 1px solid #FF5555; color: #FF5555; } /* Error -> Dracula Red */
.codehilite .k { color: #FF79C6; font-weight: bold } /* Keyword -> Dracula Pink */
.codehilite .o { color: #8BE9FD } /* Operator -> Dracula Cyan */
.codehilite .ch { color: #6272A4; font-style: italic } /* Comment.Hashbang -> Dracula Comment */
.codehilite .cm { color: #6272A4; font-style: italic } /* Comment.Multiline -> Dracula Comment */
.codehilite .cp { color: #F1FA8C } /* Comment.Preproc -> Dracula Yellow */
.codehilite .cpf { color: #6272A4; font-style: italic } /* Comment.PreprocFile -> Dracula Comment */
.codehilite .c1 { color: #6272A4; font-style: italic } /* Comment.Single -> Dracula Comment */
.codehilite .cs { color: #6272A4; font-style: italic } /* Comment.Special -> Dracula Comment */
.codehilite .gd { color: #FF5555 } /* Generic.Deleted -> Dracula Red */
.codehilite .ge { font-style: italic } /* Generic.Emph */
.codehilite .ges { font-weight: bold; font-style: italic } /* Generic.EmphStrong */
.codehilite .gr { color: #FF5555 } /* Generic.Error -> Dracula Red */
.codehilite .gh { color: #BD93F9; font-weight: bold } /* Generic.Heading -> Dracula Purple */
.codehilite .gi { color: #50FA7B } /* Generic.Inserted -> Dracula Green */
.codehilite .go { color: #F8F8F2 } /* Generic.Output -> Dracula Foreground */
.codehilite .gp { color: #BD93F9; font-weight: bold } /* Generic.Prompt -> Dracula Purple */
.codehilite .gs { font-weight: bold } /* Generic.Strong */
.codehilite .gu { color: #BD93F9; font-weight: bold } /* Generic.Subheading -> Dracula Purple */
.codehilite .gt { color: #FF5555 } /* Generic.Traceback -> Dracula Red */
.codehilite .kc { color: #FF79C6; font-weight: bold } /* Keyword.Constant -> Dracula Pink */
.codehilite .kd { color: #FF79C6; font-weight: bold } /* Keyword.Declaration -> Dracula Pink */
.codehilite .kn { color: #FF79C6; font-weight: bold } /* Keyword.Namespace -> Dracula Pink */
.codehilite .kp { color: #FF79C6 } /* Keyword.Pseudo -> Dracula Pink */
.codehilite .kr { color: #FF79C6; font-weight: bold } /* Keyword.Reserved -> Dracula Pink */
.codehilite .kt { color: #FFB86C; font-weight: bold } /* Keyword.Type -> Dracula Orange */
.codehilite .m { color: #BD93F9 } /* Literal.Number -> Dracula Purple */
.codehilite .s { color: #50FA7B } /* Literal.String -> Dracula Green */
.codehilite .na { color: #8BE9FD } /* Name.Attribute -> Dracula Cyan */
.codehilite .nb { color: #BD93F9 } /* Name.Builtin -> Dracula Purple */
.codehilite .nc { color: #F1FA8C; font-weight: bold } /* Name.Class -> Dracula Yellow */
.codehilite .no { color: #FFB86C } /* Name.Constant -> Dracula Orange */
.codehilite .nd { color: #FF79C6 } /* Name.Decorator -> Dracula Pink */
.codehilite .ni { color: #F8F8F2; font-weight: bold } /* Name.Entity -> Dracula Foreground */
.codehilite .ne { color: #FF5555; font-weight: bold } /* Name.Exception -> Dracula Red */
.codehilite .nf { color: #BD93F9; font-weight: bold } /* Name.Function -> Dracula Purple */
.codehilite .nl { color: #F8F8F2 } /* Name.Label -> Dracula Foreground */
.codehilite .nn { color: #F1FA8C; font-weight: bold } /* Name.Namespace -> Dracula Yellow */
.codehilite .nt { color: #FF79C6; font-weight: bold } /* Name.Tag -> Dracula Pink */
.codehilite .nv { color: #F8F8F2 } /* Name.Variable -> Dracula Foreground */
.codehilite .ow { color: #8BE9FD; font-weight: bold } /* Operator.Word -> Dracula Cyan */
.codehilite .w { color: #6272A4 } /* Text.Whitespace -> Dracula Comment */
.codehilite .mb { color: #BD93F9 } /* Literal.Number.Bin -> Dracula Purple */
.codehilite .mf { color: #BD93F9 } /* Literal.Number.Float -> Dracula Purple */
.codehilite .mh { color: #BD93F9 } /* Literal.Number.Hex -> Dracula Purple */
.codehilite .mi { color: #BD93F9 } /* Literal.Number.Integer -> Dracula Purple */
.codehilite .mo { color: #BD93F9 } /* Literal.Number.Oct -> Dracula Purple */
.codehilite .sa { color: #50FA7B } /* Literal.String.Affix -> Dracula Green */
.codehilite .sb { color: #50FA7B } /* Literal.String.Backtick -> Dracula Green */
.codehilite .sc { color: #50FA7B } /* Literal.String.Char -> Dracula Green */
.codehilite .dl { color: #50FA7B } /* Literal.String.Delimiter -> Dracula Green */
.codehilite .sd { color: #6272A4; font-style: italic } /* Literal.String.Doc -> Dracula Comment */
.codehilite .s2 { color: #50FA7B } /* Literal.String.Double -> Dracula Green */
.codehilite .se { color: #FFB86C; font-weight: bold } /* Literal.String.Escape -> Dracula Orange */
.codehilite .sh { color: #50FA7B } /* Literal.String.Heredoc -> Dracula Green */
.codehilite .si { color: #50FA7B; font-weight: bold } /* Literal.String.Interpol -> Dracula Green */
.codehilite .sx { color: #50FA7B } /* Literal.String.Other -> Dracula Green */
.codehilite .sr { color: #50FA7B } /* Literal.String.Regex -> Dracula Green */
.codehilite .s1 { color: #50FA7B } /* Literal.String.Single -> Dracula Green */
.codehilite .ss { color: #50FA7B } /* Literal.String.Symbol -> Dracula Green */
.codehilite .bp { color: #BD93F9 } /* Name.Builtin.Pseudo -> Dracula Purple */
.codehilite .fm { color: #BD93F9; font-weight: bold } /* Name.Function.Magic -> Dracula Purple */
.codehilite .vc { color: #F8F8F2 } /* Name.Variable.Class -> Dracula Foreground */
.codehilite .vg { color: #F8F8F2 } /* Name.Variable.Global -> Dracula Foreground */
.codehilite .vi { color: #F8F8F2 } /* Name.Variable.Instance -> Dracula Foreground */
.codehilite .vm { color: #F8F8F2 } /* Name.Variable.Magic -> Dracula Foreground */
.codehilite .il { color: #BD93F9 } /* Literal.Number.Integer.Long -> Dracula Purple */
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
    background-color: rgba(139, 233, 253, 0.08); /* More subtle transparency */
    border: 1px solid rgba(139, 233, 253, 0.4); /* Softer border */
    padding: 1px; /* Minimal padding */
}
"""

    TOOL_CARD_ERROR = """
QFrame#toolCard {
    border-radius: 6px; /* Reduced from 12px for subtlety */
    background-color: rgba(255, 85, 85, 0.08); /* More subtle transparency */
    border: 1px solid rgba(255, 85, 85, 0.4); /* Softer border */
    padding: 1px; /* Minimal padding */
}
"""

    TOOL_HEADER = """
QLabel {
    font-weight: 500; /* Lighter weight for subtlety */
    color: #c7c9d1; /* More muted color */
}
"""

    TOOL_TOGGLE_BUTTON = """
QPushButton {
    background-color: transparent;
    border: none;
    color: #6272a4; /* Comment color - more muted */
    font-weight: normal; /* Reduced from bold */
    font-size: 11px; /* Smaller font */
}
QPushButton:hover {
    color: #8BE9FD; /* Dracula Cyan */
}
"""

    TOOL_STATUS = """
QLabel {
    font-weight: 500; /* Lighter than bold */
    padding: 2px; /* Reduced padding */
    font-size: 11px; /* Smaller status text */
}
QLabel[status="running"] {
    color: #FFB86C; /* Dracula Orange */
}
QLabel[status="complete"] {
    color: #50FA7B; /* Dracula Green */
}
QLabel[status="error"] {
    color: #FF5555; /* Dracula Red */
}
"""

    TOOL_CONTENT = """
QLabel {
    color: #c7c9d1; /* More muted foreground */
    padding: 1px; /* Minimal padding */
    font-size: 11px; /* Smaller content text */
}
QLabel[role="title"] {
    font-weight: 500; /* Lighter than bold */
    color: #c7c9d1;
    font-size: 12px;
}
QLabel[role="key"] {
    color: #8BE9FD; /* Dracula Cyan for key labels */
    font-weight: 500;
    font-size: 11px;
}
QLabel[role="value"] {
    color: #a6a9b8; /* More muted */
    font-size: 11px;
}
QLabel[role="error"] {
    color: #FF5555; /* Dracula Red for error messages */
    font-size: 11px;
}
"""

    TOOL_PROGRESS = """
QProgressBar {
    border: none;
    background-color: rgba(68, 71, 90, 0.4); /* Dark gray with transparency */
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #8BE9FD; /* Dracula Cyan */
    border-radius: 5px;
}
"""

    TOOL_SEPARATOR = """
QFrame {
    background-color: rgba(139, 233, 253, 0.3); /* Dracula Cyan with transparency */
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
        "background": "#282a36",
        "text": "#f8f8f2",
        "border": "#44475a",
        "string": "#50fa7b",  # Green
        "number": "#ffb86c",  # Orange
        "keyword": "#bd93f9",  # Purple
        "punctuation": "#f8f8f2",  # Text
        "error": "#ff5555",  # Red
    }

    JSON_EDITOR_STYLE = """
QPlainTextEdit {
    background-color: #282a36; /* Dracula Background */
    color: #f8f8f2; /* Dracula Foreground */
    border: 1px solid #44475a; /* Dracula Comment */
    border-radius: 4px;
    padding: 8px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.4;
}
QPlainTextEdit:focus {
    border: 1px solid #8be9fd; /* Dracula Cyan focus border */
}
"""

    # Markdown Editor styles
    MARKDOWN_EDITOR_COLORS = {
        "background": "#282a36",  # Dracula Background
        "text": "#f8f8f2",  # Dracula Foreground
        "border": "#44475a",  # Dracula Current Line
        "header": "#bd93f9",  # Dracula Purple - for headers
        "bold": "#ffb86c",  # Dracula Orange - for bold text
        "italic": "#50fa7b",  # Dracula Green - for italic text
        "code": "#ff79c6",  # Dracula Pink - for code blocks
        "code_background": "#44475a",  # Dracula Current Line - code background
        "link": "#8be9fd",  # Dracula Cyan - for links
        "image": "#bd93f9",  # Dracula Purple - for images
        "list": "#f1fa8c",  # Dracula Yellow - for list markers
        "blockquote": "#6272a4",  # Dracula Comment - for blockquotes
        "hr": "#44475a",  # Dracula Current Line - for horizontal rules
        "strikethrough": "#ff5555",  # Dracula Red - for strikethrough text
        "error": "#ff5555",  # Dracula Red - for errors
    }

    MARKDOWN_EDITOR_STYLE = """
QPlainTextEdit {
    background-color: #282a36; /* Dracula Background */
    color: #f8f8f2; /* Dracula Foreground */
    border: 1px solid #44475a; /* Dracula Current Line */
    border-radius: 4px;
    padding: 8px;
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
    line-height: 1.6;
    selection-background-color: #44475a; /* Dracula Current Line */
    selection-color: #f8f8f2; /* Dracula Foreground */
}
QPlainTextEdit:focus {
    border: 1px solid #8be9fd; /* Dracula Cyan focus border */
}
"""
