"""
Preprocessing panel for data cleaning and transformation operations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGridLayout, QTabWidget, QLineEdit, QCheckBox,
    QTableWidget, QTableWidgetItem, QMessageBox,
    QScrollArea, QGroupBox, QProgressDialog, QApplication,
    QMenu, QInputDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from copy import deepcopy

class PreprocessingPanel(QWidget):
    """Panel for data preprocessing operations."""
    
    # Signal to notify when preprocessing is complete
    preprocessing_complete = pyqtSignal()
    
    def __init__(self, data_manager):
        """Initialize the preprocessing panel."""
        super().__init__()
        self.data_manager = data_manager
        self.history = []  # Stack for undo
        self.redo_stack = []  # Stack for redo
        self.max_history = 20  # Maximum number of operations to store
        self.current_outliers = None  # Store current outlier detection results
        self.data_loaded_flag = False  # Flag to track whether data has been loaded
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create Transform tab
        transform_tab = QWidget()
        transform_layout = QVBoxLayout(transform_tab)
        
        # Create transform ribbon
        ribbon = QHBoxLayout()
        
        # Column Operations Group
        column_group = QGroupBox("Column")
        column_layout = QHBoxLayout(column_group)
        
        # Data type dropdown
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItem("")  # Empty item for when no column is selected
        self.dtype_combo.addItems(["int64", "float64", "string", "datetime", "boolean"])
        self.dtype_combo.setToolTip("Change Data Type")
        self.dtype_combo.setEnabled(False)  # Disable initially until data is loaded
        column_layout.addWidget(self.dtype_combo)
        
        # Add common column operations
        self.rename_btn = QPushButton("Rename")
        self.rename_btn.setEnabled(False)  # Disable initially until data is loaded
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setEnabled(False)  # Disable initially until data is loaded
        column_layout.addWidget(self.rename_btn)
        column_layout.addWidget(self.remove_btn)
        ribbon.addWidget(column_group)
        
        # Transform Group
        transform_group = QGroupBox("Transform")
        transform_layout_group = QHBoxLayout(transform_group)
        
        # Add transform operations
        self.transform_combo = QComboBox()
        self.transform_combo.addItems([
            "Standard Scale",
            "Min-Max Scale",
            "Robust Scale",
            "Log Transform",
            "Square Root",
            "Box-Cox"
        ])
        self.transform_combo.setEnabled(False)  # Disable initially until data is loaded
        transform_layout_group.addWidget(self.transform_combo)
        self.apply_transform_btn = QPushButton("Apply")
        self.apply_transform_btn.setEnabled(False)  # Disable initially until data is loaded
        transform_layout_group.addWidget(self.apply_transform_btn)
        ribbon.addWidget(transform_group)
        
        # Filter Group
        filter_group = QGroupBox("Filter")
        filter_layout_group = QHBoxLayout(filter_group)
        
        self.filter_column = QComboBox()
        self.filter_column.setEnabled(False)  # Disable initially until data is loaded
        self.filter_condition = QComboBox()
        self.filter_condition.addItems(["equals", "not equals", "greater than", "less than", "contains"])
        self.filter_condition.setEnabled(False)  # Disable initially until data is loaded
        self.filter_value = QLineEdit()
        self.filter_value.setPlaceholderText("Value")
        self.filter_value.setEnabled(False)  # Disable initially until data is loaded
        
        filter_layout_group.addWidget(self.filter_column)
        filter_layout_group.addWidget(self.filter_condition)
        filter_layout_group.addWidget(self.filter_value)
        self.apply_filter_btn = QPushButton("Apply")
        filter_layout_group.addWidget(self.apply_filter_btn)
        ribbon.addWidget(filter_group)
        
        # Replace Group
        replace_group = QGroupBox("Replace")
        replace_layout_group = QVBoxLayout(replace_group)
        
        # Find and replace inputs
        input_layout = QHBoxLayout()
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find")
        self.find_edit.setEnabled(False)  # Disable initially until data is loaded
        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("Replace")
        self.replace_edit.setEnabled(False)  # Disable initially until data is loaded
        input_layout.addWidget(self.find_edit)
        input_layout.addWidget(self.replace_edit)
        replace_layout_group.addLayout(input_layout)
        
        # Options and button
        options_layout = QHBoxLayout()
        self.exact_match_check = QCheckBox("Exact Match")
        self.exact_match_check.setEnabled(False)  # Disable initially until data is loaded
        options_layout.addWidget(self.exact_match_check)
        
        self.replace_btn = QPushButton("Replace")
        self.replace_btn.setEnabled(False)  # Disable initially until data is loaded
        options_layout.addWidget(self.replace_btn)
        replace_layout_group.addLayout(options_layout)
        
        ribbon.addWidget(replace_group)
        
        transform_layout.addLayout(ribbon)
        
        # Add second row of tools
        second_row = QHBoxLayout()
        
        # Rounding Tool
        rounding_group = QGroupBox("Rounding")
        rounding_layout = QHBoxLayout(rounding_group)
        
        self.rounding_column = QComboBox()
        self.rounding_column.setEnabled(False)  # Disable initially until data is loaded
        # Set a fixed width for the combo box
        self.rounding_column.setFixedWidth(120)
        
        self.rounding_digits = QSpinBox()
        self.rounding_digits.setRange(0, 10)
        self.rounding_digits.setValue(2)
        self.rounding_digits.setEnabled(False)  # Disable initially until data is loaded
        
        self.apply_rounding_btn = QPushButton("Round")
        self.apply_rounding_btn.setEnabled(False)  # Disable initially until data is loaded
        
        rounding_layout.addWidget(QLabel("Column:"))
        rounding_layout.addWidget(self.rounding_column)
        rounding_layout.addWidget(QLabel("Digits:"))
        rounding_layout.addWidget(self.rounding_digits)
        rounding_layout.addWidget(self.apply_rounding_btn)
        
        second_row.addWidget(rounding_group)
        
        # Split Column Tool
        split_group = QGroupBox("Split Column")
        split_layout = QHBoxLayout(split_group)
        
        self.split_column = QComboBox()
        self.split_column.setEnabled(False)  # Disable initially until data is loaded
        # Set a fixed width for the delimiter input
        self.split_column.setFixedWidth(120)
        
        self.split_delimiter = QLineEdit()
        self.split_delimiter.setPlaceholderText("Delimiter")
        self.split_delimiter.setText(",")
        self.split_delimiter.setEnabled(False)  # Disable initially until data is loaded
        
        self.apply_split_btn = QPushButton("Split")
        self.apply_split_btn.setEnabled(False)  # Disable initially until data is loaded
        
        split_layout.addWidget(QLabel("Column:"))
        split_layout.addWidget(self.split_column)
        split_layout.addWidget(QLabel("Delimiter:"))
        split_layout.addWidget(self.split_delimiter)
        split_layout.addWidget(self.apply_split_btn)
        
        second_row.addWidget(split_group)
        
        transform_layout.addLayout(second_row)
        
        # Add third row of tools
        third_row = QHBoxLayout()
        
        # Unpivot Column Tool
        unpivot_group = QGroupBox("Unpivot Columns")
        unpivot_layout = QHBoxLayout(unpivot_group)
        
        self.unpivot_id_column = QComboBox()
        self.unpivot_id_column.setEnabled(False)  # Disable initially until data is loaded
        
        self.apply_unpivot_btn = QPushButton("Unpivot")
        self.apply_unpivot_btn.setEnabled(False)  # Disable initially until data is loaded
        
        unpivot_layout.addWidget(QLabel("ID Column:"))
        unpivot_layout.addWidget(self.unpivot_id_column)
        unpivot_layout.addWidget(self.apply_unpivot_btn)
        
        third_row.addWidget(unpivot_group)
        
        # Group By Tool
        groupby_group = QGroupBox("Group By")
        groupby_layout = QHBoxLayout(groupby_group)
        
        self.groupby_column = QComboBox()
        self.groupby_column.setEnabled(False)  # Disable initially until data is loaded
        
        self.groupby_agg = QComboBox()
        self.groupby_agg.addItems(["Count", "Sum", "Mean", "Min", "Max"])
        self.groupby_agg.setEnabled(False)  # Disable initially until data is loaded
        
        self.apply_groupby_btn = QPushButton("Group")
        self.apply_groupby_btn.setEnabled(False)  # Disable initially until data is loaded
        
        groupby_layout.addWidget(QLabel("Column:"))
        groupby_layout.addWidget(self.groupby_column)
        groupby_layout.addWidget(QLabel("Aggregation:"))
        groupby_layout.addWidget(self.groupby_agg)
        groupby_layout.addWidget(self.apply_groupby_btn)
        
        third_row.addWidget(groupby_group)
        
        transform_layout.addLayout(third_row)
        
        # Create main data view for transform tab
        self.data_view = QTableWidget()
        self.data_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.data_view.customContextMenuRequested.connect(self.show_context_menu)
        # Make cells not editable
        self.data_view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Add pagination controls and Apply Changes button
        pagination = QHBoxLayout()
        self.page_label = QLabel("Page:")
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.rows_per_page_label = QLabel("Rows per page:")
        self.rows_per_page_combo = QComboBox()
        self.rows_per_page_combo.addItems(["50", "100", "500", "1000"])
        
        pagination.addWidget(self.page_label)
        pagination.addWidget(self.page_spin)
        pagination.addWidget(self.rows_per_page_label)
        pagination.addWidget(self.rows_per_page_combo)
        pagination.addStretch()
        
        # Add Apply Changes button
        self.apply_changes_btn = QPushButton("Apply Changes to Main View")
        self.apply_changes_btn.setEnabled(False)  # Disable initially until data is loaded
        self.apply_changes_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        pagination.addWidget(self.apply_changes_btn)
        
        transform_layout.addWidget(self.data_view)
        transform_layout.addLayout(pagination)
        
        # Create Preprocessing tab
        preprocess_tab = QWidget()
        preprocess_layout = QVBoxLayout(preprocess_tab)
        
        # Outlier Detection Group
        outlier_group = QGroupBox("Outlier Detection")
        outlier_layout = QGridLayout(outlier_group)
        
        # Column selection
        outlier_layout.addWidget(QLabel("Column:"), 0, 0)
        self.outlier_column_combo = QComboBox()
        outlier_layout.addWidget(self.outlier_column_combo, 0, 1)
        
        # Method selection
        outlier_layout.addWidget(QLabel("Method:"), 1, 0)
        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems(["IQR Method", "Z-Score Method", "Modified Z-Score"])
        outlier_layout.addWidget(self.outlier_method_combo, 1, 1)
        
        # Threshold
        outlier_layout.addWidget(QLabel("Threshold:"), 2, 0)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 10)
        self.threshold_spin.setValue(3)
        outlier_layout.addWidget(self.threshold_spin, 2, 1)
        
        # Detect button
        self.detect_outliers_btn = QPushButton("Detect Outliers")
        outlier_layout.addWidget(self.detect_outliers_btn, 3, 0, 1, 2)
        
        preprocess_layout.addWidget(outlier_group)
        
        # Outlier View Group
        view_group = QGroupBox("Outlier View")
        view_layout = QVBoxLayout(view_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.show_only_outliers = QCheckBox("Show Only Outliers")
        controls_layout.addWidget(self.show_only_outliers)
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Value", "Outlier Status"])
        controls_layout.addWidget(QLabel("Sort by:"))
        controls_layout.addWidget(self.sort_combo)
        controls_layout.addStretch()
        
        view_layout.addLayout(controls_layout)
        
        # Outlier table
        self.outlier_table = QTableWidget()
        self.outlier_table.setColumnCount(2)
        self.outlier_table.setHorizontalHeaderLabels(["Value", "Is Outlier"])
        # Make outlier table cells not editable
        self.outlier_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        view_layout.addWidget(self.outlier_table)
        
        preprocess_layout.addWidget(view_group)
        
        # Outlier Handling Group
        handling_group = QGroupBox("Outlier Handling")
        handling_layout = QHBoxLayout(handling_group)
        
        self.handling_method_combo = QComboBox()
        self.handling_method_combo.addItems([
            "Remove outliers",
            "Cap outliers",
            "Replace with mean",
            "Replace with median"
        ])
        handling_layout.addWidget(self.handling_method_combo)
        
        self.apply_handling_btn = QPushButton("Apply")
        handling_layout.addWidget(self.apply_handling_btn)
        
        preprocess_layout.addWidget(handling_group)
        
        # Add tabs
        self.tabs.addTab(transform_tab, "Transform")
        self.tabs.addTab(preprocess_tab, "Preprocessing")
        
        layout.addWidget(self.tabs)
        
        # Add undo/redo buttons at the bottom
        button_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        button_layout.addWidget(self.undo_btn)
        button_layout.addWidget(self.redo_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

    def setup_connections(self):
        """Setup signal connections."""
        # Data view connections
        self.page_spin.valueChanged.connect(self.update_data_view)
        self.rows_per_page_combo.currentTextChanged.connect(self.update_data_view)
        
        # Connect data view selection change to update data type dropdown
        self.data_view.currentCellChanged.connect(lambda row, col, prev_row, prev_col: self.update_dtype_dropdown())
        
        # Column operations
        self.rename_btn.clicked.connect(self.handle_rename_click)
        self.remove_btn.clicked.connect(self.handle_remove_click)
        
        # Connect data type change signal after initialization
        # This prevents the signal from being triggered during initialization
        self.dtype_combo.currentTextChanged.connect(self.handle_type_change)
        
        # Transform operations
        self.apply_transform_btn.clicked.connect(self.handle_transform_click)
        
        # Filter operations
        self.apply_filter_btn.clicked.connect(self.handle_filter_click)
        
        # Replace operations
        self.replace_btn.clicked.connect(self.handle_replace_click)
        
        # New tool connections
        self.apply_rounding_btn.clicked.connect(self.handle_rounding_click)
        self.apply_split_btn.clicked.connect(self.handle_split_click)
        self.apply_unpivot_btn.clicked.connect(self.handle_unpivot_click)
        self.apply_groupby_btn.clicked.connect(self.handle_groupby_click)
        
        # Apply Changes button
        self.apply_changes_btn.clicked.connect(self.apply_changes_to_main_view)
        
        # Outlier detection
        self.detect_outliers_btn.clicked.connect(lambda: self.detect_outliers(True))
        self.show_only_outliers.stateChanged.connect(self.update_outlier_view)
        self.sort_combo.currentTextChanged.connect(self.update_outlier_view)
        self.apply_handling_btn.clicked.connect(self.apply_outlier_handling)
        
        # Update data when loaded
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        
        # Undo/Redo connections
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)

    def check_data_loaded(self):
        """Check if data is loaded and show error message if not."""
        if self.data_manager.data is None:
            QMessageBox.warning(self, "No Data", 
                              "Please load a dataset before performing operations.")
            return False
        return True

    def get_selected_column(self):
        """Get the currently selected column name with error handling."""
        try:
            current_col = self.data_view.currentColumn()
            if current_col < 0:
                # No column is selected, return None without showing an error message
                return None
            header_item = self.data_view.horizontalHeaderItem(current_col)
            if header_item is None:
                # Invalid column, return None without showing an error message
                return None
            return header_item.text()
        except Exception as e:
            # Only show error message for unexpected exceptions
            QMessageBox.critical(self, "Error", 
                               f"Error getting column name: {str(e)}\n"
                               "Please make sure a valid column is selected.")
            return None

    def handle_rename_click(self):
        """Handle rename button click with error checking."""
        # First check if data is loaded
        if self.data_manager.data is None:
            # Silently return if no data is loaded - this prevents errors when initializing the UI
            return
            
        # Only show warning if user explicitly tries to rename a column
        if not self.check_data_loaded():
            return
            
        column_name = self.get_selected_column()
        if column_name:
            self.rename_column_dialog(column_name)
        else:
            QMessageBox.warning(self, "No Column Selected", 
                              "Please select a column before performing this operation.")

    def handle_remove_click(self):
        """Handle remove button click with error checking."""
        # First check if data is loaded
        if self.data_manager.data is None:
            # Silently return if no data is loaded - this prevents errors when initializing the UI
            return
            
        # Only show warning if user explicitly tries to remove a column
        if not self.check_data_loaded():
            return
            
        column_name = self.get_selected_column()
        if column_name:
            self.remove_column(column_name)
        else:
            QMessageBox.warning(self, "No Column Selected", 
                              "Please select a column before performing this operation.")

    def handle_type_change(self, new_type):
        """Handle data type change with error checking."""
        # If empty type is selected, just update the dropdown to match the current column
        if not new_type:
            self.update_dtype_dropdown()
            return
            
        # Check if data has been loaded
        if not self.data_loaded_flag:
            # Data hasn't been loaded yet, silently return
            return
            
        # First check if data is loaded
        if self.data_manager.data is None:
            # Silently return if no data is loaded - this prevents errors when initializing the UI
            return
            
        # Check if a column is selected in the data view
        current_col = self.data_view.currentColumn()
        if current_col < 0:
            # No column is selected, silently return
            # This prevents errors when the dropdown is changed but no column is selected
            return
            
        # Only show warning if user explicitly tries to change the type
        if not self.check_data_loaded():
            return
            
        column_name = self.get_selected_column()
        if column_name:
            # Get current data type to check if it's different
            df = self.data_manager.data
            current_dtype = str(df[column_name].dtype)
            
            # Map current dtype to our dropdown options
            if 'int' in current_dtype and new_type == 'int64':
                # Already the right type, no need to change
                return
            elif 'float' in current_dtype and new_type == 'float64':
                # Already the right type, no need to change
                return
            elif 'datetime' in current_dtype and new_type == 'datetime':
                # Already the right type, no need to change
                return
            elif 'bool' in current_dtype and new_type == 'boolean':
                # Already the right type, no need to change
                return
            elif ('object' in current_dtype or 'string' in current_dtype) and new_type == 'string':
                # Already the right type, no need to change
                return
                
            # Apply the type change
            self.change_column_type(column_name, new_type)

    def handle_transform_click(self):
        """Handle transform button click."""
        if not self.check_data_loaded():
            return
            
        column = self.get_selected_column()
        if column is None:
            QMessageBox.warning(self, "No Column Selected", 
                              "Please select a column to transform.")
            return
            
        transform = self.transform_combo.currentText()
        
        progress = QProgressDialog(f"Applying {transform}...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        try:
            # Save current state for undo functionality
            self.save_state()
            
            df = self.data_manager.data.copy()
            
            # Get the data
            data = df[column]
            
            progress.setValue(20)
            
            # Apply transformation
            if transform == "Standard Scale":
                scaler = StandardScaler()
                df[column] = scaler.fit_transform(data.values.reshape(-1, 1))
            elif transform == "Min-Max Scale":
                scaler = MinMaxScaler()
                df[column] = scaler.fit_transform(data.values.reshape(-1, 1))
            elif transform == "Robust Scale":
                scaler = RobustScaler()
                df[column] = scaler.fit_transform(data.values.reshape(-1, 1))
            elif transform == "Log Transform":
                # Handle negative or zero values
                min_val = data.min()
                if min_val <= 0:
                    offset = abs(min_val) + 1
                    df[column] = np.log(data + offset)
                else:
                    df[column] = np.log(data)
            elif transform == "Square Root":
                # Handle negative values
                min_val = data.min()
                if min_val < 0:
                    offset = abs(min_val)
                    df[column] = np.sqrt(data + offset)
                else:
                    df[column] = np.sqrt(data)
            elif transform == "Box-Cox":
                # Box-Cox requires positive values
                min_val = data.min()
                if min_val <= 0:
                    offset = abs(min_val) + 1
                    transformed, lambda_val = stats.boxcox(data + offset)
                else:
                    transformed, lambda_val = stats.boxcox(data)
                df[column] = transformed
            
            progress.setValue(80)
            
            # Update the data manager with the new dataframe
            self.data_manager._data = df
            
            # Update only the local view without emitting data_loaded signal
            self.update_data_view()
            
            progress.setValue(100)
            QMessageBox.information(self, "Success", 
                                  "Transformation applied successfully! Click 'Apply Changes to Main View' to update the main data preview.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying transformation: {str(e)}")
        finally:
            progress.close()

    def handle_filter_click(self):
        """Handle filter button click."""
        if not self.check_data_loaded():
            return
            
        column = self.filter_column.currentText()
        condition = self.filter_condition.currentText()
        value = self.filter_value.text()
        
        if not column or not value:
            QMessageBox.warning(self, "Missing Information", 
                              "Please select a column and enter a value.")
            return
            
        try:
            # Save current state for undo functionality
            self.save_state()
            
            df = self.data_manager.data.copy()
            
            # Apply filter based on condition
            if condition == "equals":
                try:
                    # Try to convert value to numeric if column is numeric
                    if pd.api.types.is_numeric_dtype(df[column]):
                        df = df[df[column] == float(value)]
                    else:
                        df = df[df[column] == value]
                except ValueError:
                    # If conversion fails, use string comparison
                    df = df[df[column] == value]
            elif condition == "not equals":
                try:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        df = df[df[column] != float(value)]
                    else:
                        df = df[df[column] != value]
                except ValueError:
                    df = df[df[column] != value]
            elif condition == "greater than":
                try:
                    df = df[df[column] > float(value)]
                except ValueError:
                    QMessageBox.warning(self, "Invalid Value", 
                                      "Please enter a numeric value for 'greater than' comparison.")
                    return
            elif condition == "less than":
                try:
                    df = df[df[column] < float(value)]
                except ValueError:
                    QMessageBox.warning(self, "Invalid Value", 
                                      "Please enter a numeric value for 'less than' comparison.")
                    return
            elif condition == "contains":
                df = df[df[column].astype(str).str.contains(value, case=False, na=False)]
            
            if len(df) == 0:
                QMessageBox.warning(self, "No Data", 
                                  "The filter returned no results. Please try a different filter.")
                return
                
            # Update the data manager with the new dataframe
            self.data_manager._data = df
            
            # Update only the local view without emitting data_loaded signal
            self.update_data_view()
            
            QMessageBox.information(self, "Success", 
                                  "Filter applied successfully! Click 'Apply Changes to Main View' to update the main data preview.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying filter: {str(e)}")

    def handle_replace_click(self):
        """Handle replace button click."""
        if not self.check_data_loaded():
            return
            
        find_value = self.find_edit.text()
        replace_value = self.replace_edit.text()
        exact_match = self.exact_match_check.isChecked()
        
        if not find_value:
            QMessageBox.warning(self, "Missing Information", 
                              "Please enter a value to find.")
            return
            
        try:
            # Save current state for undo functionality
            self.save_state()
            
            df = self.data_manager.data.copy()
            
            # Get selected column if any
            column = self.get_selected_column()
            
            # Replace in selected column or all columns
            if column:
                # Try to convert values based on column type
                if pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        find_numeric = float(find_value)
                        replace_numeric = float(replace_value) if replace_value else np.nan
                        
                        if exact_match:
                            # Only replace exact matches
                            mask = df[column] == find_numeric
                            df.loc[mask, column] = replace_numeric
                        else:
                            # Use pandas replace which can handle partial matches
                            df[column] = df[column].replace(find_numeric, replace_numeric)
                    except ValueError:
                        QMessageBox.warning(self, "Type Mismatch", 
                                          "Cannot convert values to match column type.")
                        return
                else:
                    if exact_match:
                        # Only replace exact matches for string columns
                        mask = df[column].astype(str) == find_value
                        df.loc[mask, column] = replace_value
                    else:
                        # Use pandas replace which can handle partial matches
                        df[column] = df[column].replace(find_value, replace_value)
            else:
                # Replace in all columns
                if exact_match:
                    # For exact match, we need to iterate through columns
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            try:
                                find_numeric = float(find_value)
                                replace_numeric = float(replace_value) if replace_value else np.nan
                                mask = df[col] == find_numeric
                                df.loc[mask, col] = replace_numeric
                            except ValueError:
                                # Skip columns where conversion fails
                                continue
                        else:
                            # For non-numeric columns, compare as strings
                            mask = df[col].astype(str) == find_value
                            df.loc[mask, col] = replace_value
                else:
                    # Use pandas replace for non-exact matches
                    df = df.replace(find_value, replace_value)
                
            # Update the data manager with the new dataframe
            self.data_manager._data = df
            
            # Update only the local view without emitting data_loaded signal
            self.update_data_view()
            
            QMessageBox.information(self, "Success", 
                                  "Replace operation completed successfully! Click 'Apply Changes to Main View' to update the main data preview.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error replacing values: {str(e)}")

    def update_data_view(self):
        """Update the main data view with current page of data."""
        if self.data_manager.data is None:
            return
            
        df = self.data_manager.data
        rows_per_page = int(self.rows_per_page_combo.currentText())
        current_page = self.page_spin.value() - 1
        
        start_idx = current_page * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(df))
        
        # Update page spinner maximum
        total_pages = (len(df) + rows_per_page - 1) // rows_per_page
        self.page_spin.setMaximum(total_pages)
        
        # Get current page of data
        page_data = df.iloc[start_idx:end_idx]
        
        # Update table
        self.data_view.setRowCount(len(page_data))
        self.data_view.setColumnCount(len(df.columns))
        self.data_view.setHorizontalHeaderLabels(df.columns)
        
        # Set vertical header labels to start from 1 instead of 0
        self.data_view.setVerticalHeaderLabels([str(i + 1) for i in range(start_idx, end_idx)])
        
        for i in range(len(page_data)):
            for j in range(len(df.columns)):
                value = str(page_data.iloc[i, j])
                self.data_view.setItem(i, j, QTableWidgetItem(value))
                
        self.data_view.resizeColumnsToContents()
        
        # Update the data type dropdown to reflect the currently selected column
        self.update_dtype_dropdown()

    def show_context_menu(self, pos):
        """Show context menu for column operations."""
        column = self.data_view.horizontalHeader().logicalIndexAt(pos.x())
        if column >= 0:
            # Select the column
            self.data_view.selectColumn(column)
            
            # Update the data type dropdown
            self.update_dtype_dropdown()
            
            menu = QMenu(self)
            
            # Get column name
            column_name = self.data_view.horizontalHeaderItem(column).text()
            
            # Add column operations
            rename_action = menu.addAction("Rename")
            change_type_menu = menu.addMenu("Change Type")
            for dtype in ["int64", "float64", "string", "datetime", "boolean"]:
                change_type_menu.addAction(dtype)
            
            remove_action = menu.addAction("Remove")
            menu.addSeparator()
            
            # Add transform operations
            transform_menu = menu.addMenu("Transform")
            for transform in ["Standard Scale", "Min-Max Scale", "Robust Scale", 
                            "Log Transform", "Square Root", "Box-Cox"]:
                transform_menu.addAction(transform)
            
            # Add filter operations
            filter_menu = menu.addMenu("Filter")
            for condition in ["equals", "not equals", "greater than", "less than", "contains"]:
                filter_menu.addAction(condition)
            
            # Show menu and handle actions
            action = menu.exec(self.data_view.mapToGlobal(pos))
            if action:
                if action == rename_action:
                    self.rename_column_dialog(column_name)
                elif action == remove_action:
                    self.remove_column(column_name)
                elif action.parent() == change_type_menu:
                    self.change_column_type(column_name, action.text())
                elif action.parent() == transform_menu:
                    self.handle_transform_click()
                elif action.parent() == filter_menu:
                    self.show_filter_dialog(column_name, action.text())

    def rename_column_dialog(self, column_name):
        """Show dialog to rename a column."""
        new_name, ok = QInputDialog.getText(
            self, "Rename Column",
            f"Enter new name for column '{column_name}':",
            text=column_name
        )
        
        if ok and new_name:
            self.rename_column(column_name, new_name)

    def rename_column(self, old_name, new_name):
        """Rename a column."""
        if new_name in self.data_manager.data.columns and new_name != old_name:
            QMessageBox.warning(self, "Warning", "Column name already exists.")
            return
            
        try:
            df = self.data_manager.data.copy()
            df = df.rename(columns={old_name: new_name})
            
            self.save_state()
            
            # Update the data manager with the new dataframe
            self.data_manager._data = df
            
            # Update only the local view without emitting data_loaded signal
            self.update_data_view()
            
            QMessageBox.information(self, "Success", 
                                  f"Column '{old_name}' renamed to '{new_name}' successfully! Click 'Apply Changes to Main View' to update the main data preview.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error renaming column: {str(e)}")

    def remove_column(self, column_name):
        """Remove a column from the dataset."""
        try:
            df = self.data_manager.data.copy()
            df = df.drop(columns=[column_name])
            
            self.save_state()
            
            # Update the data manager with the new dataframe
            self.data_manager._data = df
            
            # Update only the local view without emitting data_loaded signal
            self.update_data_view()
            
            QMessageBox.information(self, "Success", 
                                  f"Column '{column_name}' removed successfully! Click 'Apply Changes to Main View' to update the main data preview.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error removing column: {str(e)}")

    def show_filter_dialog(self, column_name, condition):
        """Show dialog for filter value input."""
        value, ok = QInputDialog.getText(
            self,
            "Filter Value",
            f"Enter value to filter where {column_name} {condition}:"
        )
        
        if ok and value:
            self.filter_column.setCurrentText(column_name)
            self.filter_condition.setCurrentText(condition)
            self.filter_value.setText(value)
            self.handle_filter_click()

    def on_data_loaded(self, df):
        """Handle when new data is loaded."""
        # Set the data loaded flag
        self.data_loaded_flag = True
        
        # Update column selectors
        self.filter_column.clear()
        self.filter_column.addItems(df.columns)
        self.filter_column.setEnabled(True)  # Enable now that data is loaded
        self.filter_condition.setEnabled(True)  # Enable now that data is loaded
        self.filter_value.setEnabled(True)  # Enable now that data is loaded
        
        # Update column selectors for new tools
        self.rounding_column.clear()
        self.rounding_column.addItems(df.select_dtypes(include=[np.number]).columns)
        self.rounding_column.setEnabled(True)
        self.rounding_digits.setEnabled(True)
        self.apply_rounding_btn.setEnabled(True)
        
        self.split_column.clear()
        # Use all columns instead of just object/string types
        self.split_column.addItems(df.columns)
        self.split_column.setEnabled(True)
        self.split_delimiter.setEnabled(True)
        self.apply_split_btn.setEnabled(True)
        
        self.unpivot_id_column.clear()
        self.unpivot_id_column.addItems(df.columns)
        self.unpivot_id_column.setEnabled(True)
        self.apply_unpivot_btn.setEnabled(True)
        
        self.groupby_column.clear()
        self.groupby_column.addItems(df.columns)
        self.groupby_column.setEnabled(True)
        self.groupby_agg.setEnabled(True)
        self.apply_groupby_btn.setEnabled(True)
        
        self.outlier_column_combo.clear()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.outlier_column_combo.addItems(numeric_columns)
        
        # Update data type combo - temporarily disconnect signal to prevent triggering handle_type_change
        self.dtype_combo.blockSignals(True)
        self.dtype_combo.clear()
        self.dtype_combo.addItem("")  # Empty item for when no column is selected
        self.dtype_combo.addItems(["int64", "float64", "string", "datetime", "boolean"])
        self.dtype_combo.setCurrentIndex(0)  # Set to empty initially
        self.dtype_combo.setEnabled(True)  # Enable the dropdown now that data is loaded
        self.dtype_combo.blockSignals(False)
        
        # Reset pagination
        self.page_spin.setValue(1)
        
        # Update views
        self.update_data_view()
        if self.current_outliers is not None:
            self.update_outlier_view()
        
        # Enable transform operations
        self.transform_combo.setEnabled(True)
        self.apply_transform_btn.setEnabled(True)
        
        # Enable column operations
        self.rename_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
        
        # Enable replace operations
        self.find_edit.setEnabled(True)
        self.replace_edit.setEnabled(True)
        self.replace_btn.setEnabled(True)
        self.exact_match_check.setEnabled(True)
        
        # Enable Apply Changes button
        self.apply_changes_btn.setEnabled(True)
        
        # Update undo/redo buttons state
        self.update_undo_redo_buttons()

    def update_missing_values_info(self):
        """Update the missing values information table."""
        # This method is no longer used in the new UI design
        pass
        
    def update_transform_columns_table(self):
        """Update the transformation columns table."""
        # This method is no longer used in the new UI design
        pass

    def apply_transformation(self):
        """Apply the selected transformation to the selected columns."""
        if self.data_manager.data is None:
            return
            
        progress = QProgressDialog("Applying transformation...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        try:
            df = self.data_manager.data.copy()
            method = self.transform_combo.currentText()
            
            progress.setValue(20)
            
            # Get selected columns
            selected_columns = []
            for i in range(self.transform_columns_table.rowCount()):
                checkbox = self.transform_columns_table.cellWidget(i, 1)
                if checkbox.isChecked():
                    column = self.transform_columns_table.item(i, 0).text()
                    selected_columns.append(column)
                    
            if not selected_columns:
                QMessageBox.warning(self, "Warning", "Please select at least one column to transform.")
                return
                
            progress.setValue(40)
            
            # Apply transformation
            if method == "Standard Scale":
                scaler = StandardScaler()
                df[selected_columns] = scaler.fit_transform(df[selected_columns])
            elif method == "Min-Max Scale":
                scaler = MinMaxScaler()
                df[selected_columns] = scaler.fit_transform(df[selected_columns])
            elif method == "Robust Scale":
                scaler = RobustScaler()
                df[selected_columns] = scaler.fit_transform(df[selected_columns])
            elif method == "Log Transform":
                for col in selected_columns:
                    if (df[col] <= 0).any():
                        QMessageBox.warning(self, "Warning", 
                                         f"Column {col} contains non-positive values. "
                                         "Log transform requires positive values.")
                        continue
                    df[col] = np.log(df[col])
            elif method == "Square Root":
                for col in selected_columns:
                    if (df[col] < 0).any():
                        QMessageBox.warning(self, "Warning",
                                         f"Column {col} contains negative values. "
                                         "Square root transform requires non-negative values.")
                        continue
                    df[col] = np.sqrt(df[col])
            elif method == "Box-Cox":
                for col in selected_columns:
                    if (df[col] <= 0).any():
                        QMessageBox.warning(self, "Warning",
                                         f"Column {col} contains non-positive values. "
                                         "Box-Cox transform requires positive values.")
                        continue
                    df[col] = stats.boxcox(df[col])[0]
                    
            progress.setValue(80)
            
            self.data_manager._data = df
            QTimer.singleShot(100, lambda: self.data_manager.data_loaded.emit(df))
            
            progress.setValue(100)
            QMessageBox.information(self, "Success", "Transformation applied successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying transformation: {str(e)}")
        finally:
            progress.close()

    def save_state(self):
        """Save current state to history."""
        if self.data_manager.data is not None:
            self.data_manager.save_state()
            self.update_undo_redo_buttons()
            
    def update_undo_redo_buttons(self):
        """Update the enabled state of undo/redo buttons."""
        self.undo_btn.setEnabled(len(self.data_manager.history) > 0)
        self.redo_btn.setEnabled(len(self.data_manager.redo_stack) > 0)
        
    def undo(self):
        """Undo the last operation."""
        # Use the DataManager's undo method
        self.data_manager.undo()
        # Update undo/redo buttons
        self.update_undo_redo_buttons()
        
    def redo(self):
        """Redo the last undone operation."""
        # Use the DataManager's redo method
        self.data_manager.redo()
        # Update undo/redo buttons
        self.update_undo_redo_buttons()

    def apply_operation(self, operation_func):
        """Apply an operation with proper state management."""
        try:
            # Save current state before operation
            if self.data_manager.data is not None:
                self.save_state()
            
            # Apply the operation
            operation_func()
            
        except Exception as e:
            # Restore previous state if operation fails
            if self.history:
                self.data_manager._data = self.history[-1]
                self.data_manager.data_loaded.emit(self.history[-1])
            QMessageBox.critical(self, "Error", str(e))
            
    def export_to_csv(self):
        """Export the current dataset to a CSV file."""
        if self.data_manager.data is None:
            return
            
        from PyQt6.QtWidgets import QFileDialog
        
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_name:
            try:
                self.data_manager.data.to_csv(file_name, index=False)
                QMessageBox.information(self, "Success", "Data exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting data: {str(e)}")

    def change_column_type(self, column_name, new_type):
        """Change the data type of the selected column."""
        if self.data_manager.data is None:
            return
            
        try:
            df = self.data_manager.data.copy()
            
            if new_type == "datetime":
                df[column_name] = pd.to_datetime(df[column_name])
            elif new_type == "boolean":
                df[column_name] = df[column_name].astype(bool)
            else:
                df[column_name] = df[column_name].astype(new_type)
            
            self.save_state()
            
            # Update the data manager with the new dataframe
            self.data_manager._data = df
            
            # Update only the local view without emitting data_loaded signal
            self.update_data_view()
            
            QMessageBox.information(self, "Success", 
                                  f"Column '{column_name}' type changed to {new_type} successfully! Click 'Apply Changes to Main View' to update the main data preview.")
                                  
            # Update the dropdown to reflect the new type
            self.update_dtype_dropdown()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error changing data type: {str(e)}")

    def find_and_replace(self):
        """Implement the find and replace functionality."""
        if not self.check_data_loaded():
            return
            
        find_value = self.find_edit.text().strip()
        replace_value = self.replace_edit.text()
        
        try:
            df = self.data_manager.data.copy()
            
            # Apply find and replace to all columns
            for column in df.columns:
                df[column] = df[column].astype(str).replace(find_value, replace_value)
            
            self.save_state()
            self.data_manager._data = df
            self.data_manager.data_loaded.emit(df)
            
            QMessageBox.information(self, "Success", 
                                  f"Replaced all occurrences of '{find_value}' "
                                  f"with '{replace_value}'")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Error during find and replace: {str(e)}")

    def update_outlier_view(self):
        """Update the outlier table view based on current filters."""
        if self.current_outliers is None or len(self.current_outliers) == 0:
            return
            
        # Get the data and outlier status
        display_data = self.current_outliers['data']
        is_outlier = self.current_outliers['is_outlier']
        total_outliers = self.current_outliers.get('total_outliers', is_outlier.sum())
        total_rows = self.current_outliers.get('total_rows', len(display_data))
        
        # Create a DataFrame for easier manipulation
        df_view = pd.DataFrame({
            'value': display_data,
            'is_outlier': is_outlier
        })
        
        # Filter if show only outliers is checked
        if self.show_only_outliers.isChecked():
            df_view = df_view[df_view['is_outlier']]
        
        # Sort based on selection
        sort_by = self.sort_combo.currentText()
        if sort_by == "Value":
            df_view = df_view.sort_values('value')
        else:  # Sort by outlier status
            df_view = df_view.sort_values('is_outlier', ascending=False)
        
        # Update table
        self.outlier_table.setRowCount(len(df_view))
        self.outlier_table.setUpdatesEnabled(False)
        
        try:
            for i in range(len(df_view)):
                self.outlier_table.setItem(i, 0, QTableWidgetItem(f"{df_view['value'].iloc[i]:.2f}"))
                self.outlier_table.setItem(i, 1, QTableWidgetItem("Yes" if df_view['is_outlier'].iloc[i] else "No"))
                
            # Add a note about total outliers in the window title
            self.outlier_table.setToolTip(
                f"Showing first 1000 rows. Total outliers in dataset: {total_outliers} out of {total_rows} rows"
            )
        finally:
            self.outlier_table.setUpdatesEnabled(True)
            self.outlier_table.resizeColumnsToContents()

    def detect_outliers(self, show_info=True):
        """Detect outliers using the selected method."""
        if self.data_manager.data is None:
            return
            
        progress = QProgressDialog("Detecting outliers...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        try:
            df = self.data_manager.data
            column = self.outlier_column_combo.currentText()
            method = self.outlier_method_combo.currentText()
            threshold = self.threshold_spin.value()
            
            progress.setValue(10)
            
            # Get the data and handle empty case
            data = df[column].dropna()
            if len(data) == 0:
                self.outlier_table.setRowCount(0)
                self.current_outliers = None
                return
                
            progress.setValue(20)
            
            # Calculate statistics based on the entire dataset
            if method == "IQR Method":
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                if IQR == 0:  # Handle case where IQR is zero
                    is_outlier = pd.Series(False, index=data.index)
                else:
                    lower_bound = Q1 - threshold * IQR  # Use threshold instead of fixed 1.5
                    upper_bound = Q3 + threshold * IQR
                    is_outlier = (data < lower_bound) | (data > upper_bound)
            elif method == "Z-Score Method":
                mean = data.mean()
                std = data.std()
                if std == 0:  # Handle case where std is zero
                    is_outlier = pd.Series(False, index=data.index)
                else:
                    z_scores = np.abs((data - mean) / std)
                    is_outlier = z_scores > threshold
            elif method == "Modified Z-Score":
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                if mad == 0:  # Handle case where MAD is zero
                    is_outlier = pd.Series(False, index=data.index)
                else:
                    modified_z_scores = 0.6745 * np.abs(data - median) / mad
                    is_outlier = modified_z_scores > threshold
            
            progress.setValue(60)
            
            # Store current outliers - keep track of full dataset results
            total_outliers = is_outlier.sum()
            
            # For display, take first 1000 rows but maintain the outlier status
            display_data = data.head(1000)
            display_outliers = is_outlier[display_data.index]
            
            self.current_outliers = {
                'data': display_data,
                'is_outlier': display_outliers,
                'total_outliers': total_outliers,
                'total_rows': len(data),
                'bounds': {
                    'lower': lower_bound if method == "IQR Method" else None,
                    'upper': upper_bound if method == "IQR Method" else None,
                    'threshold': threshold,
                    'method': method
                }
            }
            
            # Update the view
            self.update_outlier_view()
            
            # Show summary only when explicitly requested
            if show_info:
                if total_outliers > 0:
                    display_outliers_count = display_outliers.sum()
                    QMessageBox.information(self, "Outlier Detection", 
                                          f"Found {total_outliers} outliers in total.\n"
                                          f"Showing {display_outliers_count} outliers in the first 1000 rows.")
                else:
                    QMessageBox.information(self, "Outlier Detection", 
                                          "No outliers detected in the dataset.")
            
            progress.setValue(100)
            
        except Exception as e:
            if not progress.wasCanceled() and show_info:
                QMessageBox.critical(self, "Error", f"Error detecting outliers: {str(e)}")
            self.outlier_table.setRowCount(0)
            self.current_outliers = None
        finally:
            progress.close()
            
    def apply_outlier_handling(self):
        """Apply the selected outlier handling method."""
        if self.data_manager.data is None or self.current_outliers is None:
            return
            
        progress = QProgressDialog("Processing outliers...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        try:
            # Save current state for undo functionality
            self.save_state()
            
            df = self.data_manager.data.copy()
            column = self.outlier_column_combo.currentText()
            method = self.handling_method_combo.currentText()
            
            # Get the data and create a Series with the original index
            data = pd.Series(df[column].dropna().values, index=df[column].dropna().index)
            
            progress.setValue(20)
            
            # Get the outlier detection method and threshold from stored results
            bounds = self.current_outliers['bounds']
            detection_method = bounds['method']
            threshold = bounds['threshold']
            
            # Re-detect outliers on the entire dataset to ensure we handle all outliers
            if detection_method == "IQR Method":
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                if IQR == 0:  # Handle case where IQR is zero
                    is_outlier = pd.Series(False, index=data.index)
                else:
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    is_outlier = (data < lower_bound) | (data > upper_bound)
            elif detection_method == "Z-Score Method":
                mean = data.mean()
                std = data.std()
                if std == 0:  # Handle case where std is zero
                    is_outlier = pd.Series(False, index=data.index)
                else:
                    z_scores = np.abs((data - mean) / std)
                    is_outlier = z_scores > threshold
            elif detection_method == "Modified Z-Score":
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                if mad == 0:  # Handle case where MAD is zero
                    is_outlier = pd.Series(False, index=data.index)
                else:
                    modified_z_scores = 0.6745 * np.abs(data - median) / mad
                    is_outlier = modified_z_scores > threshold
            
            # Create mask for all outliers in the dataset
            outlier_mask = is_outlier
            
            progress.setValue(50)
            
            # Handle outliers using the selected method
            if method == "Remove outliers":
                df = df[~outlier_mask]
            else:
                if method == "Cap outliers":
                    if detection_method == "IQR Method":
                        data[outlier_mask] = data[outlier_mask].clip(lower_bound, upper_bound)
                    else:
                        # For other methods, use percentiles for capping
                        lower_bound = np.percentile(data[~outlier_mask], 1)
                        upper_bound = np.percentile(data[~outlier_mask], 99)
                        data[outlier_mask] = data[outlier_mask].clip(lower_bound, upper_bound)
                elif method == "Replace with mean":
                    replacement = data[~outlier_mask].mean()
                    data[outlier_mask] = replacement
                elif method == "Replace with median":
                    replacement = data[~outlier_mask].median()
                    data[outlier_mask] = replacement
                
                df.loc[data.index, column] = data
            
            progress.setValue(90)
            
            # Update the data manager with the new dataframe
            self.data_manager._data = df
            
            # Update only the local view without emitting data_loaded signal
            self.update_data_view()
            
            progress.setValue(100)
            QMessageBox.information(self, "Success", 
                                  f"Outliers handled successfully! "
                                  f"Handled {outlier_mask.sum()} outliers from the entire dataset. "
                                  f"Click 'Apply Changes to Main View' to update the main data preview.")
            
        except Exception as e:
            if not progress.wasCanceled():
                QMessageBox.critical(self, "Error", f"Error handling outliers: {str(e)}")
        finally:
            progress.close()

    def update_dtype_dropdown(self):
        """Update the data type dropdown based on the selected column."""
        if not self.data_loaded_flag or self.data_manager.data is None:
            return
            
        column_name = self.get_selected_column()
        if column_name is None:
            # No column selected, set to empty item
            self.dtype_combo.blockSignals(True)
            self.dtype_combo.setCurrentIndex(0)  # Select empty item
            self.dtype_combo.blockSignals(False)
            return
            
        # Get the data type of the selected column
        df = self.data_manager.data
        dtype = str(df[column_name].dtype)
        
        # Map pandas dtype to our dropdown options
        if 'int' in dtype:
            dtype_str = 'int64'
        elif 'float' in dtype:
            dtype_str = 'float64'
        elif 'datetime' in dtype:
            dtype_str = 'datetime'
        elif 'bool' in dtype:
            dtype_str = 'boolean'
        else:
            dtype_str = 'string'
            
        # Set the dropdown to the current data type
        self.dtype_combo.blockSignals(True)
        index = self.dtype_combo.findText(dtype_str)
        if index >= 0:
            self.dtype_combo.setCurrentIndex(index)
        else:
            self.dtype_combo.setCurrentIndex(0)  # Default to empty if not found
        self.dtype_combo.blockSignals(False) 

    def apply_changes_to_main_view(self):
        """Apply the changes made in the Transform tab to the main data preview."""
        if self.data_manager.data is None:
            return
        
        try:
            # Save current state for undo functionality
            self.save_state()
            
            # Get the current data from the data manager
            df = self.data_manager.data.copy()
            
            # Create a progress dialog
            progress = QProgressDialog("Applying changes to main view...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            progress.setValue(10)
            
            # Update the data manager with the current dataframe
            self.data_manager._data = df
            
            progress.setValue(50)
            
            # Emit the data_loaded signal to update all panels
            self.data_manager.data_loaded.emit(df)
            
            progress.setValue(100)
            
            # Show success message
            QMessageBox.information(self, "Success", "Changes applied to main view successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying changes: {str(e)}")
        finally:
            if 'progress' in locals():
                progress.close() 

    def handle_rounding_click(self):
        """Handle rounding button click."""
        if not self.check_data_loaded():
            return
            
        column = self.rounding_column.currentText()
        digits = self.rounding_digits.value()
        
        if not column:
            QMessageBox.warning(self, "Missing Information", 
                              "Please select a column to round.")
            return
            
        try:
            # Save current state for undo functionality
            self.save_state()
            
            df = self.data_manager.data.copy()
            
            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(df[column]):
                QMessageBox.warning(self, "Invalid Column Type", 
                                  "Rounding can only be applied to numeric columns.")
                return
                
            # Apply rounding
            df[column] = df[column].round(digits)
            
            # Update the data manager with the new dataframe
            self.data_manager._data = df
            
            # Update only the local view without emitting data_loaded signal
            self.update_data_view()
            
            QMessageBox.information(self, "Success", 
                                  f"Column '{column}' rounded to {digits} decimal places successfully! "
                                  f"Click 'Apply Changes to Main View' to update the main data preview.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying rounding: {str(e)}")

    def handle_split_click(self):
        """Handle split column button click."""
        if not self.check_data_loaded():
            return
            
        column = self.split_column.currentText()
        delimiter = self.split_delimiter.text()
        
        if not column:
            QMessageBox.warning(self, "Missing Information", 
                              "Please select a column to split.")
            return
            
        if not delimiter:
            QMessageBox.warning(self, "Missing Information", 
                              "Please enter a delimiter.")
            return
            
        try:
            # Save current state for undo functionality
            self.save_state()
            
            df = self.data_manager.data.copy()
            
            # Create a progress dialog
            progress = QProgressDialog("Splitting column...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            progress.setValue(10)
            
            # Split the column into multiple columns
            split_df = df[column].str.split(delimiter, expand=True)
            
            # If split was successful, add new columns to the dataframe
            if split_df is not None and not split_df.empty:
                # Generate new column names
                new_columns = [f"{column}_{i+1}" for i in range(split_df.shape[1])]
                
                # Rename the split columns
                split_df.columns = new_columns
                
                progress.setValue(50)
                
                # Add the new columns to the original dataframe
                for new_col in new_columns:
                    df[new_col] = split_df[new_col]
                
                progress.setValue(90)
                
                # Update the data manager with the new dataframe
                self.data_manager._data = df
                
                # Update only the local view without emitting data_loaded signal
                self.update_data_view()
                
                progress.setValue(100)
                
                QMessageBox.information(self, "Success", 
                                      f"Column '{column}' split into {len(new_columns)} new columns successfully! "
                                      f"Click 'Apply Changes to Main View' to update the main data preview.")
            else:
                QMessageBox.warning(self, "Split Failed", 
                                  "The split operation did not produce any new columns. "
                                  "Please check your delimiter and try again.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error splitting column: {str(e)}")
        finally:
            if 'progress' in locals():
                progress.close()

    def handle_unpivot_click(self):
        """Handle unpivot columns button click."""
        if not self.check_data_loaded():
            return
            
        id_column = self.unpivot_id_column.currentText()
        
        if not id_column:
            QMessageBox.warning(self, "Missing Information", 
                              "Please select an ID column.")
            return
            
        try:
            # Save current state for undo functionality
            self.save_state()
            
            df = self.data_manager.data.copy()
            
            # Create a progress dialog
            progress = QProgressDialog("Unpivoting columns...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            progress.setValue(10)
            
            # Get all columns except the ID column
            value_columns = [col for col in df.columns if col != id_column]
            
            if not value_columns:
                QMessageBox.warning(self, "Invalid Selection", 
                                  "There must be at least one column to unpivot.")
                return
                
            progress.setValue(30)
            
            # Create a new dataframe for the unpivoted data
            unpivoted_data = []
            
            # For each row in the original dataframe
            for idx, row in df.iterrows():
                id_value = row[id_column]
                # For each value column, create a new row
                for col in value_columns:
                    unpivoted_data.append({
                        id_column: id_value,
                        'Variable': col,
                        'Value': row[col]
                    })
            
            progress.setValue(70)
            
            # Create the new unpivoted dataframe
            unpivoted_df = pd.DataFrame(unpivoted_data)
            
            # Update the data manager with the new dataframe
            self.data_manager._data = unpivoted_df
            
            # Update only the local view without emitting data_loaded signal
            self.update_data_view()
            
            progress.setValue(100)
            
            QMessageBox.information(self, "Success", 
                                  f"Unpivoted {len(value_columns)} columns successfully! "
                                  f"Click 'Apply Changes to Main View' to update the main data preview.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error unpivoting columns: {str(e)}")
        finally:
            if 'progress' in locals():
                progress.close()

    def handle_groupby_click(self):
        """Handle group by button click."""
        if not self.check_data_loaded():
            return
            
        column = self.groupby_column.currentText()
        aggregation = self.groupby_agg.currentText().lower()
        
        if not column:
            QMessageBox.warning(self, "Missing Information", 
                              "Please select a column to group by.")
            return
            
        try:
            # Save current state for undo functionality
            self.save_state()
            
            df = self.data_manager.data.copy()
            
            # Create a progress dialog
            progress = QProgressDialog("Grouping data...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            progress.setValue(10)
            
            # Get numeric columns for aggregation (except the groupby column)
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if column in numeric_columns:
                numeric_columns.remove(column)
                
            if not numeric_columns and aggregation != 'count':
                QMessageBox.warning(self, "Invalid Selection", 
                                  f"There are no numeric columns to apply '{aggregation}' aggregation. "
                                  f"Only 'count' can be used with non-numeric data.")
                return
                
            progress.setValue(30)
            
            # Apply the groupby operation
            if aggregation == 'count':
                # Count can be applied to any column
                grouped_df = df.groupby(column).size().reset_index(name='count')
            else:
                # For other aggregations, apply to numeric columns
                agg_dict = {col: aggregation for col in numeric_columns}
                grouped_df = df.groupby(column).agg(agg_dict).reset_index()
            
            progress.setValue(70)
            
            # Update the data manager with the new dataframe
            self.data_manager._data = grouped_df
            
            # Update only the local view without emitting data_loaded signal
            self.update_data_view()
            
            progress.setValue(100)
            
            QMessageBox.information(self, "Success", 
                                  f"Data grouped by '{column}' with '{aggregation}' aggregation successfully! "
                                  f"Click 'Apply Changes to Main View' to update the main data preview.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error grouping data: {str(e)}")
        finally:
            if 'progress' in locals():
                progress.close() 