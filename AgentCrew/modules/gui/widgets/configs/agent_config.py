from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QMenu,
    QStackedWidget,
    QFileDialog,
)
import os
import toml
import json
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator

from AgentCrew.modules.config import ConfigManagement
from AgentCrew.modules.agents import AgentManager

from AgentCrew.modules.gui.themes import StyleProvider


class AgentsConfigTab(QWidget):
    """Tab for configuring agents."""

    # Add signal for configuration changes
    config_changed = Signal()

    def __init__(self, config_manager: ConfigManagement):
        super().__init__()
        self.config_manager = config_manager
        self.agent_manager = AgentManager.get_instance()
        self.available_tools = [
            "memory",
            "clipboard",
            "code_analysis",
            "web_search",
            "image_generation",
        ]

        # Load agents configuration
        self.agents_config = self.config_manager.read_agents_config()
        self._is_dirty = False

        self.init_ui()
        self.load_agents()

    @staticmethod
    def _determine_file_format_and_path(file_path: str, selected_filter: str) -> tuple[str, str]:
        """
        Determine file format and ensure correct file extension.
        
        Args:
            file_path: The selected file path
            selected_filter: The filter selected in the file dialog
            
        Returns:
            Tuple of (final_file_path, file_format)
        """
        # Prioritize existing extension if present
        if file_path.lower().endswith('.toml'):
            return file_path, "toml"
        elif file_path.lower().endswith('.json'):
            return file_path, "json"
        
        # If no extension, use filter preference or default to JSON
        if "toml" in selected_filter.lower():
            return file_path + '.toml', "toml"
        else:
            return file_path + '.json', "json"

    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QHBoxLayout()

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Agent list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.agents_list = QListWidget()
        self.agents_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # Enable multi-select
        self.agents_list.currentItemChanged.connect(self.on_agent_selected)
        self.agents_list.itemSelectionChanged.connect(self.on_selection_changed)

        # Buttons for agent list management
        list_buttons_layout = QHBoxLayout()

        self.add_agent_menu_btn = QPushButton("Add Agent")
        style_provider = StyleProvider()
        self.add_agent_menu_btn.setStyleSheet(
            style_provider.get_button_style("agent_menu")
        )
        add_agent_menu = QMenu(self)
        add_agent_menu.setStyleSheet(style_provider.get_agent_menu_style())
        add_local_action = add_agent_menu.addAction("Add Local Agent")
        add_remote_action = add_agent_menu.addAction("Add Remote Agent")
        self.add_agent_menu_btn.setMenu(add_agent_menu)

        add_local_action.triggered.connect(self.add_new_local_agent)
        add_remote_action.triggered.connect(self.add_new_remote_agent)

        self.import_agents_btn = QPushButton("Import")
        self.import_agents_btn.setStyleSheet(style_provider.get_button_style("green"))
        self.import_agents_btn.clicked.connect(self.import_agents)

        self.export_agents_btn = QPushButton("Export")
        self.export_agents_btn.setStyleSheet(style_provider.get_button_style("primary"))
        self.export_agents_btn.clicked.connect(self.export_agents)
        self.export_agents_btn.setEnabled(False)  # Disable until selection

        self.remove_agent_btn = QPushButton("Remove")
        self.remove_agent_btn.setStyleSheet(style_provider.get_button_style("red"))
        self.remove_agent_btn.clicked.connect(self.remove_agent)
        self.remove_agent_btn.setEnabled(False)  # Disable until selection

        list_buttons_layout.addWidget(
            self.add_agent_menu_btn
        )
        list_buttons_layout.addWidget(self.import_agents_btn)
        list_buttons_layout.addWidget(self.export_agents_btn)
        list_buttons_layout.addWidget(self.remove_agent_btn)

        left_layout.addWidget(QLabel("Agents:"))
        left_layout.addWidget(self.agents_list)
        left_layout.addLayout(list_buttons_layout)

        # Right panel - Agent editor
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        # right_panel.setStyleSheet("background-color: #181825;") # Set by QDialog stylesheet

        editor_container_widget = (
            QWidget()
        )  # Container for stacked widget and save button
        editor_container_widget.setStyleSheet(
            style_provider.get_editor_container_widget_style()
        )
        self.editor_layout = QVBoxLayout(
            editor_container_widget
        )  # editor_layout now on container

        self.editor_stacked_widget = QStackedWidget()

        # Local Agent Editor Widget
        self.local_agent_editor_widget = QWidget()
        local_agent_layout = QVBoxLayout(self.local_agent_editor_widget)
        local_form_layout = QFormLayout()

        self.name_input = QLineEdit()  # This is for Local Agent Name
        local_form_layout.addRow("Name:", self.name_input)
        self.description_input = QLineEdit()
        local_form_layout.addRow("Description:", self.description_input)
        self.temperature_input = QLineEdit()
        self.temperature_input.setValidator(QDoubleValidator(0.0, 2.0, 1))
        self.temperature_input.setPlaceholderText("0.0 - 2.0")
        local_form_layout.addRow("Temperature:", self.temperature_input)

        self.enabled_checkbox = QCheckBox("Enabled")
        self.enabled_checkbox.setChecked(True)  # Default to enabled
        local_form_layout.addRow("", self.enabled_checkbox)

        tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout()
        self.tool_checkboxes = {}
        for tool in self.available_tools:
            checkbox = QCheckBox(tool)
            self.tool_checkboxes[tool] = checkbox
            tools_layout.addWidget(checkbox)
        tools_group.setLayout(tools_layout)

        self.system_prompt_input = QTextEdit()
        self.system_prompt_input.setMinimumHeight(200)

        local_agent_layout.addLayout(local_form_layout)
        local_agent_layout.addWidget(tools_group)
        local_agent_layout.addWidget(QLabel("System Prompt:"))
        local_agent_layout.addWidget(self.system_prompt_input)
        local_agent_layout.addStretch()

        # Remote Agent Editor Widget
        self.remote_agent_editor_widget = QWidget()
        remote_agent_layout = QVBoxLayout(self.remote_agent_editor_widget)
        remote_form_layout = QFormLayout()

        self.remote_name_input = QLineEdit()
        remote_form_layout.addRow("Name:", self.remote_name_input)
        self.remote_base_url_input = QLineEdit()
        self.remote_base_url_input.setPlaceholderText("e.g., http://localhost:8000")
        remote_form_layout.addRow("Base URL:", self.remote_base_url_input)

        self.remote_enabled_checkbox = QCheckBox("Enabled")
        self.remote_enabled_checkbox.setChecked(True)  # Default to enabled
        remote_form_layout.addRow("", self.remote_enabled_checkbox)

        remote_agent_layout.addLayout(remote_form_layout)
        remote_agent_layout.addStretch()

        self.editor_stacked_widget.addWidget(self.local_agent_editor_widget)
        self.editor_stacked_widget.addWidget(self.remote_agent_editor_widget)

        # Save button (common to both editors)
        self.save_btn = QPushButton("Save")
        # ... (save_btn styling and connect remains the same)
        self.save_btn.setStyleSheet(style_provider.get_button_style("primary"))
        self.save_btn.clicked.connect(self.save_agent)
        self.save_btn.setEnabled(False)

        self.editor_layout.addWidget(self.editor_stacked_widget)  # Changed
        self.editor_layout.addWidget(self.save_btn)
        # self.editor_layout.addStretch() # Removed, stretch is within individual editors

        # Connect signals for editor fields to handle changes
        # Local agent fields
        self.name_input.textChanged.connect(self._on_editor_field_changed)
        self.description_input.textChanged.connect(self._on_editor_field_changed)
        self.temperature_input.textChanged.connect(self._on_editor_field_changed)
        self.system_prompt_input.textChanged.connect(self._on_editor_field_changed)
        self.enabled_checkbox.stateChanged.connect(self._on_editor_field_changed)
        for checkbox in self.tool_checkboxes.values():
            checkbox.stateChanged.connect(self._on_editor_field_changed)
        # Remote agent fields
        self.remote_name_input.textChanged.connect(self._on_editor_field_changed)
        self.remote_base_url_input.textChanged.connect(self._on_editor_field_changed)
        self.remote_enabled_checkbox.stateChanged.connect(self._on_editor_field_changed)

        right_panel.setWidget(editor_container_widget)  # Set the container widget

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 600])  # Initial sizes

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        self.set_editor_enabled(False)

    def load_agents(self):
        """Load agents from configuration."""
        self.agents_list.clear()

        local_agents = self.agents_config.get("agents", [])
        for agent_conf in local_agents:
            item_data = agent_conf.copy()
            item_data["agent_type"] = "local"
            item = QListWidgetItem(item_data.get("name", "Unnamed Local Agent"))
            item.setData(Qt.ItemDataRole.UserRole, item_data)
            self.agents_list.addItem(item)

        remote_agents = self.agents_config.get("remote_agents", [])
        for agent_conf in remote_agents:
            item_data = agent_conf.copy()
            item_data["agent_type"] = "remote"
            item = QListWidgetItem(item_data.get("name", "Unnamed Remote Agent"))
            item.setData(Qt.ItemDataRole.UserRole, item_data)
            self.agents_list.addItem(item)

    def on_selection_changed(self):
        """Handle selection changes to update button states."""
        selected_items = self.agents_list.selectedItems()
        has_selection = len(selected_items) > 0
        
        # Enable/disable export and remove buttons based on selection
        self.export_agents_btn.setEnabled(has_selection)
        self.remove_agent_btn.setEnabled(has_selection)

    def on_agent_selected(self, current, previous):
        """Handle agent selection."""
        if current is None:
            self.set_editor_enabled(False)
            # Optionally hide both editors or show a placeholder
            # self.editor_stacked_widget.setCurrentIndex(-1) # or a placeholder widget index
            return

        self.set_editor_enabled(True)

        agent_data = current.data(Qt.ItemDataRole.UserRole)
        agent_type = agent_data.get(
            "agent_type", "local"
        )

        all_editor_widgets = [
            self.name_input,
            self.description_input,
            self.temperature_input,
            self.system_prompt_input,
            self.enabled_checkbox,
            self.remote_name_input,
            self.remote_base_url_input,
            self.remote_enabled_checkbox,
        ] + list(self.tool_checkboxes.values())
        for widget in all_editor_widgets:
            widget.blockSignals(True)

        if agent_type == "local":
            self.editor_stacked_widget.setCurrentWidget(self.local_agent_editor_widget)
            self.name_input.setText(agent_data.get("name", ""))
            self.description_input.setText(agent_data.get("description", ""))
            self.temperature_input.setText(str(agent_data.get("temperature", "0.5")))
            self.enabled_checkbox.setChecked(agent_data.get("enabled", True))
            tools = agent_data.get("tools", [])
            for tool, checkbox in self.tool_checkboxes.items():
                checkbox.setChecked(tool in tools)
            self.system_prompt_input.setText(agent_data.get("system_prompt", ""))
            # Clear remote fields just in case
            self.remote_name_input.clear()
            self.remote_base_url_input.clear()
            self.remote_enabled_checkbox.setChecked(True)  # Default for clearing
        elif agent_type == "remote":
            self.editor_stacked_widget.setCurrentWidget(self.remote_agent_editor_widget)
            self.remote_name_input.setText(agent_data.get("name", ""))
            self.remote_base_url_input.setText(agent_data.get("base_url", ""))
            self.remote_enabled_checkbox.setChecked(agent_data.get("enabled", True))
            # Clear local fields
            self.name_input.clear()
            self.description_input.clear()
            self.temperature_input.clear()
            self.system_prompt_input.clear()
            self.enabled_checkbox.setChecked(True)  # Default for clearing
            for checkbox in self.tool_checkboxes.values():
                checkbox.setChecked(False)

        for widget in all_editor_widgets:
            widget.blockSignals(False)

        self._is_dirty = False
        self.save_btn.setEnabled(False)

    def _find_agent_index_by_name(self, agent_name):
        """Find the index of an agent in the agents_list by name."""
        for i in range(self.agents_list.count()):
            item = self.agents_list.item(i)
            agent_data = item.data(Qt.ItemDataRole.UserRole)
            if agent_data.get("name", "") == agent_name:
                return i
        return -1

    def _on_editor_field_changed(self):
        """Mark configuration as dirty and enable save if an agent is selected and editor is active."""
        if self.agents_list.currentItem():
            current_editor_widget = self.editor_stacked_widget.currentWidget()
            is_editor_active = False
            if (
                current_editor_widget == self.local_agent_editor_widget
                and self.name_input.isEnabled()
            ):
                is_editor_active = True
            elif (
                current_editor_widget == self.remote_agent_editor_widget
                and self.remote_name_input.isEnabled()
            ):
                is_editor_active = True

            if is_editor_active:
                if not self._is_dirty:
                    self._is_dirty = True
                self.save_btn.setEnabled(True)

    def set_editor_enabled(self, enabled: bool):
        """Enable or disable all editor form fields."""
        # Local agent fields
        self.name_input.setEnabled(enabled)
        self.description_input.setEnabled(enabled)
        self.temperature_input.setEnabled(enabled)
        self.system_prompt_input.setEnabled(enabled)
        self.enabled_checkbox.setEnabled(enabled)
        for checkbox in self.tool_checkboxes.values():
            checkbox.setEnabled(enabled)

        # Remote agent fields
        self.remote_name_input.setEnabled(enabled)
        self.remote_base_url_input.setEnabled(enabled)
        self.remote_enabled_checkbox.setEnabled(enabled)

        if not enabled:
            # Clear all fields when disabling
            self.name_input.clear()
            self.description_input.clear()
            self.temperature_input.clear()
            self.system_prompt_input.clear()
            self.enabled_checkbox.setChecked(True)
            for checkbox in self.tool_checkboxes.values():
                checkbox.setChecked(False)

            self.remote_name_input.clear()
            self.remote_base_url_input.clear()
            self.remote_enabled_checkbox.setChecked(True)

            self.save_btn.setEnabled(False)
            self._is_dirty = False
            # self.editor_stacked_widget.setCurrentIndex(-1) # Optionally hide content

    def add_new_local_agent(self):
        """Add a new local agent to the configuration."""
        new_agent_data = {
            "name": "NewLocalAgent",
            "description": "Description for the new local agent",
            "temperature": 0.5,
            "tools": ["memory", "clipboard"],
            "system_prompt": "You are a helpful assistant. Today is {current_date}.",
            "enabled": True,
            "agent_type": "local",
        }

        item = QListWidgetItem(new_agent_data["name"])
        item.setData(Qt.ItemDataRole.UserRole, new_agent_data)
        self.agents_list.addItem(item)
        self.agents_list.setCurrentItem(item)  # Triggers on_agent_selected

        # on_agent_selected will switch to local editor and populate.
        self._is_dirty = True
        self.save_btn.setEnabled(True)
        self.name_input.setFocus()
        self.name_input.selectAll()

    def add_new_remote_agent(self):
        """Add a new remote agent to the configuration."""
        new_agent_data = {
            "name": "NewRemoteAgent",
            "base_url": "http://localhost:8000",
            "enabled": True,
            "agent_type": "remote",
        }

        item = QListWidgetItem(new_agent_data["name"])
        item.setData(Qt.ItemDataRole.UserRole, new_agent_data)
        self.agents_list.addItem(item)
        self.agents_list.setCurrentItem(item)  # Triggers on_agent_selected

        # on_agent_selected will switch to remote editor and populate.
        self._is_dirty = True
        self.save_btn.setEnabled(True)
        self.remote_name_input.setFocus()
        self.remote_name_input.selectAll()

    def remove_agent(self):
        """Remove the selected agent(s)."""
        selected_items = self.agents_list.selectedItems()
        if not selected_items:
            return

        if len(selected_items) == 1:
            agent_data = selected_items[0].data(Qt.ItemDataRole.UserRole)
            agent_name = agent_data.get("name", "this agent")
            message = f"Are you sure you want to delete the agent '{agent_name}'?"
        else:
            agent_names = [item.data(Qt.ItemDataRole.UserRole).get("name", "unnamed") for item in selected_items]
            message = f"Are you sure you want to delete {len(selected_items)} agents?\n\n• " + "\n• ".join(agent_names)

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Remove items in reverse order to maintain valid row indices
            rows_to_remove = sorted([self.agents_list.row(item) for item in selected_items], reverse=True)
            for row in rows_to_remove:
                self.agents_list.takeItem(row)

            # set_editor_enabled(False) is called by on_agent_selected when currentItem becomes None
            # or when a new item is selected. If list becomes empty, on_agent_selected(None, old_item) is called.
            if self.agents_list.count() == 0:
                self.set_editor_enabled(False)  # Explicitly disable if list is empty

            self.save_all_agents()

    def save_agent(self):
        """Save the current agent configuration."""
        current_item = self.agents_list.currentItem()
        if not current_item:
            return

        agent_data_from_list = current_item.data(Qt.ItemDataRole.UserRole)
        agent_type = agent_data_from_list.get("agent_type", "local")

        updated_agent_data = {}

        if agent_type == "local":
            name = self.name_input.text().strip()
            description = self.description_input.text().strip()
            system_prompt = self.system_prompt_input.toPlainText().strip()
            try:
                temperature = float(self.temperature_input.text().strip() or "0.5")
                temperature = max(0.0, min(2.0, temperature))
            except ValueError:
                temperature = 0.5

            if not name:
                QMessageBox.warning(
                    self, "Validation Error", "Local Agent name cannot be empty."
                )
                return

            tools = [t for t, cb in self.tool_checkboxes.items() if cb.isChecked()]
            updated_agent_data = {
                "name": name,
                "description": description,
                "temperature": temperature,
                "tools": tools,
                "system_prompt": system_prompt,
                "enabled": self.enabled_checkbox.isChecked(),
                "agent_type": "local",
            }
            current_item.setText(name)
        elif agent_type == "remote":
            name = self.remote_name_input.text().strip()
            base_url = self.remote_base_url_input.text().strip()

            if not name:
                QMessageBox.warning(
                    self, "Validation Error", "Remote Agent name cannot be empty."
                )
                return
            if not base_url:  # Basic validation for URL
                QMessageBox.warning(
                    self, "Validation Error", "Remote Agent Base URL cannot be empty."
                )
                return

            updated_agent_data = {
                "name": name,
                "base_url": base_url,
                "enabled": self.remote_enabled_checkbox.isChecked(),
                "agent_type": "remote",
            }
            current_item.setText(name)

        current_item.setData(Qt.ItemDataRole.UserRole, updated_agent_data)
        self.save_all_agents()
        self._is_dirty = False
        self.save_btn.setEnabled(False)

    def import_agents(self):
        """Import agent configurations from a file."""
        # Open file dialog to select a TOML or JSON file
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Import Agent Configuration")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Agent Configuration (*.toml *.json)")

        if not file_dialog.exec():
            # User canceled the dialog
            return

        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            return

        import_file_path = selected_files[0]

        # Check if file exists
        if not os.path.exists(import_file_path):
            QMessageBox.critical(
                self,
                "Import Error",
                f"The selected file does not exist: {import_file_path}",
            )
            return

        # Load the configuration file
        try:
            temp_config = ConfigManagement(import_file_path)
            imported_config = temp_config.get_config()

            # Validate the configuration structure
            local_agents = imported_config.get("agents", [])
            remote_agents = imported_config.get("remote_agents", [])

            if not local_agents and not remote_agents:
                QMessageBox.warning(
                    self,
                    "Invalid Configuration",
                    "No agent configurations found in the selected file.",
                )
                return

        except Exception as e:
            QMessageBox.critical(
                self, "Import Error", f"Failed to load agent configuration: {str(e)}"
            )
            return

        # Check for conflicts
        existing_agent_names = set()
        for i in range(self.agents_list.count()):
            item = self.agents_list.item(i)
            agent_data = item.data(Qt.ItemDataRole.UserRole)
            existing_agent_names.add(agent_data.get("name", ""))

        # Find conflicts
        conflict_names = []
        imported_names = []

        for agent in local_agents:
            name = agent.get("name", "")
            if name:
                imported_names.append(name)
                if name in existing_agent_names:
                    conflict_names.append(name)

        for agent in remote_agents:
            name = agent.get("name", "")
            if name:
                imported_names.append(name)
                if name in existing_agent_names:
                    conflict_names.append(name)

        # If there are conflicts, show warning dialog
        user_choice = "import_all"  # Default: import all
        if conflict_names:
            conflict_list = "\n".join([f"• {name}" for name in conflict_names])

            message_box = QMessageBox(self)
            message_box.setWindowTitle("Agent Name Conflicts")
            message_box.setIcon(QMessageBox.Icon.Warning)
            message_box.setText(
                f"The following agent(s) already exist and will be overridden:\n\n"
                f"{conflict_list}\n\n"
                f"How would you like to proceed?"
            )

            override_btn = message_box.addButton(
                "Override", QMessageBox.ButtonRole.AcceptRole
            )
            skip_btn = message_box.addButton(
                "Skip Conflicts", QMessageBox.ButtonRole.ActionRole
            )
            _ = message_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

            message_box.exec()

            clicked_button = message_box.clickedButton()
            if clicked_button == override_btn:
                user_choice = "import_all"
            elif clicked_button == skip_btn:
                user_choice = "skip_conflicts"
            else:
                user_choice = "cancel"

        if user_choice == "cancel":
            return

        # Process the import based on user's choice
        imported_count = 0
        skipped_count = 0

        # Get current config to update
        current_local_agents = self.agents_config.get("agents", [])
        current_remote_agents = self.agents_config.get("remote_agents", [])

        # Process local agents
        for imported_agent in local_agents:
            name = imported_agent.get("name", "")
            if not name:
                continue

            is_conflict = name in existing_agent_names

            if is_conflict and user_choice == "skip_conflicts":
                skipped_count += 1
                continue

            if is_conflict:
                current_local_agents = [
                    a for a in current_local_agents if a.get("name") != name
                ]
                current_remote_agents = [
                    a for a in current_remote_agents if a.get("name") != name
                ]

            if "enabled" not in imported_agent:
                imported_agent["enabled"] = True

            current_local_agents.append(imported_agent)
            imported_count += 1

        for imported_agent in remote_agents:
            name = imported_agent.get("name", "")
            if not name:
                continue

            is_conflict = name in existing_agent_names

            if is_conflict and user_choice == "skip_conflicts":
                skipped_count += 1
                continue

            if is_conflict:
                current_local_agents = [
                    a for a in current_local_agents if a.get("name") != name
                ]
                current_remote_agents = [
                    a for a in current_remote_agents if a.get("name") != name
                ]

            if "enabled" not in imported_agent:
                imported_agent["enabled"] = True

            current_remote_agents.append(imported_agent)
            imported_count += 1

        # Update the configuration
        self.agents_config["agents"] = current_local_agents
        self.agents_config["remote_agents"] = current_remote_agents

        # Save the updated configuration and refresh the UI
        self.config_manager.write_agents_config(self.agents_config)
        self.load_agents()

        # Select the first imported agent in the list if any were imported
        if imported_count > 0 and imported_names:
            index = self._find_agent_index_by_name(imported_names[0])
            if index >= 0:
                self.agents_list.setCurrentRow(index)

        status_message = f"Successfully imported {imported_count} agent(s)."
        if skipped_count > 0:
            status_message += f" Skipped {skipped_count} agent(s) due to conflicts."

        QMessageBox.information(self, "Import Complete", status_message)

    def export_agents(self):
        """Export selected agents to a file."""
        selected_items = self.agents_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select one or more agents to export."
            )
            return

        selected_agents_data = []
        selected_remote_agents_data = []
        
        for item in selected_items:
            agent_data = item.data(Qt.ItemDataRole.UserRole)
            agent_type = agent_data.get("agent_type", "local")
            
            export_data = agent_data.copy()
            export_data.pop("agent_type", None)
            
            if agent_type == "local":
                selected_agents_data.append(export_data)
            elif agent_type == "remote":
                selected_remote_agents_data.append(export_data)

        if len(selected_items) == 1:
            agent_name = selected_items[0].data(Qt.ItemDataRole.UserRole).get("name", "agent")
            default_filename = f"{agent_name}_export"
        else:
            default_filename = f"agents_export_{len(selected_items)}_agents"

        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Export Agent Configuration")
        file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setNameFilter("TOML Files (*.toml);;JSON Files (*.json)")
        file_dialog.selectFile(default_filename)

        if not file_dialog.exec():
            return

        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            return

        export_file_path = selected_files[0]
        selected_filter = file_dialog.selectedNameFilter()

        export_file_path, file_format = self._determine_file_format_and_path(
            export_file_path, selected_filter
        )

        try:
            export_config = {}
            if selected_agents_data:
                export_config["agents"] = selected_agents_data
            if selected_remote_agents_data:
                export_config["remote_agents"] = selected_remote_agents_data

            with open(export_file_path, 'w', encoding='utf-8') as f:
                if file_format == "toml":
                    toml.dump(export_config, f)
                else:
                    json.dump(export_config, f, indent=2, ensure_ascii=False)

            agent_count = len(selected_items)
            agent_word = "agent" if agent_count == 1 else "agents"
            QMessageBox.information(
                self,
                "Export Successful",
                f"Successfully exported {agent_count} {agent_word} to:\n{export_file_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export agents:\n{str(e)}"
            )

    def save_all_agents(self):
        """Save all agents to the configuration file."""
        local_agents_list = []
        remote_agents_list = []

        for i in range(self.agents_list.count()):
            item = self.agents_list.item(i)
            agent_data = item.data(Qt.ItemDataRole.UserRole)

            config_data = agent_data.copy()
            agent_type_for_sorting = config_data.pop(
                "agent_type", "local"
            )

            if agent_type_for_sorting == "local":
                local_agents_list.append(config_data)
            elif agent_type_for_sorting == "remote":
                remote_agents_list.append(config_data)

        self.agents_config["agents"] = local_agents_list
        self.agents_config["remote_agents"] = remote_agents_list

        self.config_manager.write_agents_config(self.agents_config)
        self.config_changed.emit()
