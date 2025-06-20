from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QScrollArea,
    QComboBox,
    QCheckBox,
)
from PySide6.QtCore import Signal

from AgentCrew.modules.config import ConfigManagement
from AgentCrew.modules.gui.themes import StyleProvider


class SettingsTab(QWidget):
    """Tab for configuring global settings like API keys."""

    config_changed = Signal()

    API_KEY_DEFINITIONS = [
        {
            "label": "Anthropic API Key:",
            "key_name": "ANTHROPIC_API_KEY",
            "placeholder": "e.g., sk-ant-...",
        },
        {
            "label": "Gemini API Key:",
            "key_name": "GEMINI_API_KEY",
            "placeholder": "e.g., AIzaSy...",
        },
        {
            "label": "OpenAI API Key:",
            "key_name": "OPENAI_API_KEY",
            "placeholder": "e.g., sk-...",
        },
        {
            "label": "Groq API Key:",
            "key_name": "GROQ_API_KEY",
            "placeholder": "e.g., gsk_...",
        },
        {
            "label": "DeepInfra API Key:",
            "key_name": "DEEPINFRA_API_KEY",
            "placeholder": "e.g., ...",
        },
        {
            "label": "Github Copilot API Key:",
            "key_name": "GITHUB_COPILOT_API_KEY",
            "placeholder": "e.g., ...",
        },
        {
            "label": "Tavily API Key:",
            "key_name": "TAVILY_API_KEY",
            "placeholder": "e.g., tvly-...",
        },
        {
            "label": "Voyage API Key:",
            "key_name": "VOYAGE_API_KEY",
            "placeholder": "e.g., pa-...",
        },
    ]

    def __init__(self, config_manager: ConfigManagement):
        super().__init__()
        self.config_manager = config_manager
        self.global_config = self.config_manager.read_global_config_data()

        self.api_key_inputs = {}
        self.theme_dropdown = None
        self.yolo_mode_checkbox = None

        self.init_ui()
        self.load_api_keys()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        style_provider = StyleProvider()
        scroll_area.setStyleSheet(style_provider.get_sidebar_style())

        editor_widget = QWidget()
        editor_widget.setStyleSheet(style_provider.get_sidebar_style())
        form_layout = QFormLayout(editor_widget)
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setSpacing(10)

        # Global Settings Group
        global_settings_group = QGroupBox("Global Settings")
        global_settings_group.setStyleSheet(style_provider.get_api_keys_group_style())
        global_settings_form_layout = QFormLayout()

        # Theme dropdown
        theme_label = QLabel("Theme:")
        self.theme_dropdown = QComboBox()
        self.theme_dropdown.addItems(["dark", "light", "nord", "dracula"])
        self.theme_dropdown.setCurrentText("dark")  # Default to dark
        self.theme_dropdown.setStyleSheet(style_provider.get_combo_box_style())
        global_settings_form_layout.addRow(theme_label, self.theme_dropdown)

        # YOLO Mode checkbox
        yolo_label = QLabel("YOLO Mode:")
        self.yolo_mode_checkbox = QCheckBox()
        self.yolo_mode_checkbox.setChecked(False)  # Default to unchecked
        global_settings_form_layout.addRow(yolo_label, self.yolo_mode_checkbox)

        global_settings_group.setLayout(global_settings_form_layout)
        form_layout.addWidget(global_settings_group)

        api_keys_group = QGroupBox("API Keys")
        api_keys_group.setStyleSheet(style_provider.get_api_keys_group_style())
        api_keys_form_layout = QFormLayout()

        for item in self.API_KEY_DEFINITIONS:
            label = QLabel(item["label"])
            line_edit = QLineEdit()
            line_edit.setEchoMode(QLineEdit.EchoMode.Password)
            line_edit.setPlaceholderText(item["placeholder"])
            self.api_key_inputs[item["key_name"]] = line_edit
            api_keys_form_layout.addRow(label, line_edit)

        api_keys_group.setLayout(api_keys_form_layout)
        form_layout.addWidget(api_keys_group)

        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setStyleSheet(style_provider.get_button_style("primary"))
        self.save_btn.clicked.connect(self.save_settings)

        form_layout.addWidget(self.save_btn)
        editor_widget.setLayout(form_layout)
        scroll_area.setWidget(editor_widget)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)

    def load_api_keys(self):
        api_keys_data = self.global_config.get("api_keys", {})
        for key_name, line_edit in self.api_key_inputs.items():
            line_edit.setText(api_keys_data.get(key_name, ""))

        self.load_global_settings()

    def load_global_settings(self):
        global_settings_data = self.global_config.get("global_settings", {})

        # Load theme setting
        theme = global_settings_data.get("theme", "dark")
        if self.theme_dropdown:
            self.theme_dropdown.setCurrentText(theme)

        # Load YOLO mode setting
        yolo_mode = global_settings_data.get("yolo_mode", False)
        if self.yolo_mode_checkbox:
            self.yolo_mode_checkbox.setChecked(yolo_mode)

    def save_settings(self):
        if "api_keys" not in self.global_config:
            self.global_config["api_keys"] = {}

        for key_name, line_edit in self.api_key_inputs.items():
            self.global_config["api_keys"][key_name] = line_edit.text().strip()

        # Save global settings
        if "global_settings" not in self.global_config:
            self.global_config["global_settings"] = {}

        self.global_config["global_settings"]["theme"] = (
            self.theme_dropdown.currentText() if self.theme_dropdown else "dark"
        )
        self.global_config["global_settings"]["yolo_mode"] = (
            self.yolo_mode_checkbox.isChecked() if self.yolo_mode_checkbox else False
        )

        try:
            # Save the configuration
            self.config_manager.write_global_config_data(self.global_config)

            # Get the style provider and update the theme
            style_provider = StyleProvider()
            theme_changed = style_provider.update_theme()

            # Show different message based on whether theme changed
            self.config_changed.emit()
            if theme_changed:
                QMessageBox.information(
                    self,
                    "Settings Saved",
                    "Settings saved successfully. The theme has been updated immediately.\n\n"
                    "Some components may require a restart to fully apply the new theme.",
                )
            else:
                QMessageBox.information(
                    self,
                    "Settings Saved",
                    "Settings saved successfully.",
                )

        except Exception as e:
            QMessageBox.critical(
                self, "Error Saving Settings", f"Could not save settings: {str(e)}"
            )
