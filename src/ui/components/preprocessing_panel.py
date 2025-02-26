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
        self.dtype_combo.addItems(["int64", "float64", "string", "datetime", "boolean"])
        self.dtype_combo.setToolTip("Change Data Type")
        column_layout.addWidget(self.dtype_combo)
        
        # Add common column operations
        self.rename_btn = QPushButton("Rename")
        self.remove_btn = QPushButton("Remove")
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
        transform_layout_group.addWidget(self.transform_combo)
        self.apply_transform_btn = QPushButton("Apply")
        transform_layout_group.addWidget(self.apply_transform_btn)
        ribbon.addWidget(transform_group)
        
        # Filter Group
        filter_group = QGroupBox("Filter")
        filter_layout_group = QHBoxLayout(filter_group)
        
        self.filter_column = QComboBox()
        self.filter_condition = QComboBox()
        self.filter_condition.addItems(["equals", "not equals", "greater than", "less than", "contains"])
        self.filter_value = QLineEdit()
        self.filter_value.setPlaceholderText("Value")
        
        filter_layout_group.addWidget(self.filter_column)
        filter_layout_group.addWidget(self.filter_condition)
        filter_layout_group.addWidget(self.filter_value)
        self.apply_filter_btn = QPushButton("Apply")
        filter_layout_group.addWidget(self.apply_filter_btn)
        ribbon.addWidget(filter_group)
        
        # Replace Group
        replace_group = QGroupBox("Replace")
        replace_layout_group = QHBoxLayout(replace_group)
        
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find")
        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("Replace")
        replace_layout_group.addWidget(self.find_edit)
        replace_layout_group.addWidget(self.replace_edit)
        self.replace_btn = QPushButton("Replace")
        replace_layout_group.addWidget(self.replace_btn)
        ribbon.addWidget(replace_group)
        
        transform_layout.addLayout(ribbon)
        
        # Create main data view for transform tab
        self.data_view = QTableWidget()
        self.data_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.data_view.customContextMenuRequested.connect(self.show_context_menu)
        
        # Add pagination controls
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
        
        # Column operations
        self.rename_btn.clicked.connect(self.handle_rename_click)
        self.remove_btn.clicked.connect(self.handle_remove_click)
        self.dtype_combo.currentTextChanged.connect(self.handle_type_change)
        
        # Transform operations
        self.apply_transform_btn.clicked.connect(self.handle_transform_click)
        
        # Filter operations
        self.apply_filter_btn.clicked.connect(self.handle_filter_click)
        
        # Replace operations
        self.replace_btn.clicked.connect(self.handle_replace_click)
        
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
                QMessageBox.warning(self, "No Column Selected", 
                                  "Please select a column before performing this operation.")
                return None
            header_item = self.data_view.horizontalHeaderItem(current_col)
            if header_item is None:
                QMessageBox.warning(self, "Invalid Column", 
                                  "The selected column is not valid.")
                return None
            return header_item.text()
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Error getting column name: {str(e)}\n"
                               "Please make sure a valid column is selected.")
            return None

    def handle_rename_click(self):
        """Handle rename button click with error checking."""
        if not self.check_data_loaded():
            return
        column_name = self.get_selected_column()
        if column_name:
            self.rename_column_dialog(column_name)

    def handle_remove_click(self):
        """Handle remove button click with error checking."""
        if not self.check_data_loaded():
            return
        column_name = self.get_selected_column()
        if column_name:
            self.remove_column(column_name)

    def handle_type_change(self, new_type):
        """Handle data type change with error checking."""
        if not self.check_data_loaded():
            return
        column_name = self.get_selected_column()
        if column_name:
            self.change_column_type(column_name, new_type)

    def handle_transform_click(self):
        """Handle transform button click with error checking."""
        if not self.check_data_loaded():
            return
        column_name = self.get_selected_column()
        if column_name:
            self.apply_transformation_to_column(column_name, self.transform_combo.currentText())

    def handle_filter_click(self):
        """Handle filter button click with error checking."""
        if not self.check_data_loaded():
            return
        if not self.filter_column.currentText():
            QMessageBox.warning(self, "No Column Selected", 
                              "Please select a column to filter.")
            return
        if not self.filter_value.text().strip():
            QMessageBox.warning(self, "No Value", 
                              "Please enter a value to filter by.")
            return
        self.apply_filter()

    def handle_replace_click(self):
        """Handle replace button click with error checking."""
        if not self.check_data_loaded():
            return
        if not self.find_edit.text().strip():
            QMessageBox.warning(self, "No Search Term", 
                              "Please enter a value to find.")
            return
        self.find_and_replace()

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
        
        for i in range(len(page_data)):
            for j in range(len(df.columns)):
                value = str(page_data.iloc[i, j])
                self.data_view.setItem(i, j, QTableWidgetItem(value))
                
        self.data_view.resizeColumnsToContents()

    def show_context_menu(self, pos):
        """Show context menu for column operations."""
        column = self.data_view.horizontalHeader().logicalIndexAt(pos.x())
        if column >= 0:
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
                    self.apply_transformation_to_column(column_name, action.text())
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
            self.data_manager._data = df
            self.data_manager.data_loaded.emit(df)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error renaming column: {str(e)}")

    def remove_column(self, column_name):
        """Remove a column from the dataset."""
        try:
            df = self.data_manager.data.copy()
            df = df.drop(columns=[column_name])
            
            self.save_state()
            self.data_manager._data = df
            self.data_manager.data_loaded.emit(df)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error removing column: {str(e)}")

    def apply_filter(self):
        """Apply filter to the selected column."""
        if self.data_manager.data is None:
            return
            
        column = self.filter_column.currentText()
        condition = self.filter_condition.currentText()
        value = self.filter_value.text()
        
        try:
            df = self.data_manager.data.copy()
            
            # Convert value based on column type
            col_type = df[column].dtype
            if pd.api.types.is_numeric_dtype(col_type):
                value = float(value)
            
            # Apply filter
            if condition == "equals":
                mask = df[column] == value
            elif condition == "not equals":
                mask = df[column] != value
            elif condition == "greater than":
                mask = df[column] > value
            elif condition == "less than":
                mask = df[column] < value
            elif condition == "contains":
                mask = df[column].astype(str).str.contains(str(value), case=False, na=False)
            
            filtered_df = df[mask]
            
            self.save_state()
            self.data_manager._data = filtered_df
            self.data_manager.data_loaded.emit(filtered_df)
            
            QMessageBox.information(self, "Success", 
                                  f"Filter applied. {len(filtered_df)} rows remaining.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying filter: {str(e)}")

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
            self.apply_filter()

    def apply_transformation_to_column(self, column_name, transform_type):
        """Apply transformation to a single column."""
        try:
            df = self.data_manager.data.copy()
            
            if transform_type == "Standard Scale":
                scaler = StandardScaler()
                df[column_name] = scaler.fit_transform(df[[column_name]])
            elif transform_type == "Min-Max Scale":
                scaler = MinMaxScaler()
                df[column_name] = scaler.fit_transform(df[[column_name]])
            elif transform_type == "Robust Scale":
                scaler = RobustScaler()
                df[column_name] = scaler.fit_transform(df[[column_name]])
            elif transform_type == "Log Transform":
                if (df[column_name] <= 0).any():
                    QMessageBox.warning(self, "Warning",
                                     "Column contains non-positive values. "
                                     "Log transform requires positive values.")
                    return
                df[column_name] = np.log(df[column_name])
            elif transform_type == "Square Root":
                if (df[column_name] < 0).any():
                    QMessageBox.warning(self, "Warning",
                                     "Column contains negative values. "
                                     "Square root transform requires non-negative values.")
                    return
                df[column_name] = np.sqrt(df[column_name])
            elif transform_type == "Box-Cox":
                if (df[column_name] <= 0).any():
                    QMessageBox.warning(self, "Warning",
                                     "Column contains non-positive values. "
                                     "Box-Cox transform requires positive values.")
                    return
                df[column_name] = stats.boxcox(df[column_name])[0]
            
            self.save_state()
            self.data_manager._data = df
            self.data_manager.data_loaded.emit(df)
            
            QMessageBox.information(self, "Success", 
                                  f"Transformation '{transform_type}' applied to '{column_name}'")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying transformation: {str(e)}")

    def on_data_loaded(self, df):
        """Handle when new data is loaded."""
        # Update column selectors
        self.filter_column.clear()
        self.filter_column.addItems(df.columns)
        
        self.outlier_column_combo.clear()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.outlier_column_combo.addItems(numeric_columns)
        
        # Update data type combo
        self.dtype_combo.clear()
        self.dtype_combo.addItems(["int64", "float64", "string", "datetime", "boolean"])
        
        # Reset pagination
        self.page_spin.setValue(1)
        
        # Update views
        self.update_data_view()
        if self.current_outliers is not None:
            self.update_outlier_view()

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
            self.history.append(self.data_manager.data.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            self.redo_stack.clear()  # Clear redo stack when new action is performed
            self.update_undo_redo_buttons()
            
    def update_undo_redo_buttons(self):
        """Update the enabled state of undo/redo buttons."""
        self.undo_btn.setEnabled(len(self.history) > 0)
        self.redo_btn.setEnabled(len(self.redo_stack) > 0)
        
    def undo(self):
        """Undo the last operation."""
        if self.history:
            # Save current state to redo stack
            if self.data_manager.data is not None:
                self.redo_stack.append(self.data_manager.data.copy())
            # Restore previous state
            previous_state = self.history.pop()
            self.data_manager._data = previous_state
            self.data_manager.data_loaded.emit(previous_state)
            self.update_undo_redo_buttons()
            
    def redo(self):
        """Redo the last undone operation."""
        if self.redo_stack:
            # Save current state to history
            if self.data_manager.data is not None:
                self.history.append(self.data_manager.data.copy())
            # Restore redo state
            redo_state = self.redo_stack.pop()
            self.data_manager._data = redo_state
            self.data_manager.data_loaded.emit(redo_state)
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
            
            self.data_manager._data = df
            self.data_manager.data_loaded.emit(df)
            
            QMessageBox.information(self, "Success", 
                                  f"Column '{column_name}' type changed to {new_type}")
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
            df = self.data_manager.data.copy()
            column = self.outlier_column_combo.currentText()
            method = self.handling_method_combo.currentText()
            
            # Get the data and create a Series with the original index
            data = pd.Series(df[column].dropna().values, index=df[column].dropna().index)
            
            progress.setValue(20)
            
            # Create a DataFrame for the current view
            df_view = pd.DataFrame({
                'value': self.current_outliers['data'],
                'is_outlier': self.current_outliers['is_outlier']
            })
            
            # Apply current view filters
            if self.show_only_outliers.isChecked():
                df_view = df_view[df_view['is_outlier']]
            
            # Sort based on selection
            sort_by = self.sort_combo.currentText()
            if sort_by == "Value":
                df_view = df_view.sort_values('value')
            else:  # Sort by outlier status
                df_view = df_view.sort_values('is_outlier', ascending=False)
            
            # Get indices of outliers to handle based on current view
            outlier_indices = df_view[df_view['is_outlier']].index
            
            progress.setValue(50)
            
            # Create mask for outliers to handle
            outlier_mask = pd.Series(False, index=data.index)
            outlier_mask[outlier_indices] = True
            
            # Handle outliers using the selected method
            if method == "Remove outliers":
                df = df[~outlier_mask]
            else:
                if method == "Cap outliers":
                    bounds = self.current_outliers['bounds']
                    if bounds['method'] == "IQR Method":
                        data[outlier_mask] = data[outlier_mask].clip(bounds['lower'], bounds['upper'])
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
            
            self.data_manager._data = df
            QTimer.singleShot(100, lambda: self.data_manager.data_loaded.emit(df))
            
            progress.setValue(100)
            QMessageBox.information(self, "Success", 
                                  f"Outliers handled successfully! "
                                  f"Handled {outlier_mask.sum()} outliers from the current view.")
            
        except Exception as e:
            if not progress.wasCanceled():
                QMessageBox.critical(self, "Error", f"Error handling outliers: {str(e)}")
        finally:
            progress.close() 