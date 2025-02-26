"""
Preprocessing panel for data cleaning and transformation operations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGridLayout, QTabWidget, QLineEdit, QCheckBox,
    QTableWidget, QTableWidgetItem, QMessageBox,
    QScrollArea, QGroupBox, QProgressDialog, QApplication
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
        
        # Create tab widget for different preprocessing operations
        tabs = QTabWidget()
        
        # Missing Values Tab
        missing_tab = QWidget()
        missing_layout = QVBoxLayout(missing_tab)
        
        # Missing values info group
        missing_info_group = QGroupBox("Missing Values Information")
        missing_info_layout = QGridLayout(missing_info_group)
        
        self.missing_table = QTableWidget()
        self.missing_table.setColumnCount(3)
        self.missing_table.setHorizontalHeaderLabels(["Column", "Missing Count", "Missing %"])
        missing_info_layout.addWidget(self.missing_table, 0, 0)
        
        missing_layout.addWidget(missing_info_group)
        
        # Missing values handling group
        missing_handling_group = QGroupBox("Handle Missing Values")
        missing_handling_layout = QGridLayout(missing_handling_group)
        
        # Column selector
        column_label = QLabel("Select Column:")
        self.missing_column_combo = QComboBox()
        missing_handling_layout.addWidget(column_label, 0, 0)
        missing_handling_layout.addWidget(self.missing_column_combo, 0, 1)
        
        # Method selector
        method_label = QLabel("Fill Method:")
        self.missing_method_combo = QComboBox()
        self.missing_method_combo.addItems([
            "Drop rows",
            "Fill with mean",
            "Fill with median",
            "Fill with mode",
            "Fill with value",
            "Forward fill",
            "Backward fill"
        ])
        missing_handling_layout.addWidget(method_label, 1, 0)
        missing_handling_layout.addWidget(self.missing_method_combo, 1, 1)
        
        # Custom value input
        value_label = QLabel("Custom Value:")
        self.missing_value_edit = QLineEdit()
        self.missing_value_edit.setEnabled(False)
        missing_handling_layout.addWidget(value_label, 2, 0)
        missing_handling_layout.addWidget(self.missing_value_edit, 2, 1)
        
        # Apply button
        self.apply_missing_btn = QPushButton("Apply")
        missing_handling_layout.addWidget(self.apply_missing_btn, 3, 0, 1, 2)
        
        missing_layout.addWidget(missing_handling_group)
        tabs.addTab(missing_tab, "Missing Values")
        
        # Outliers Tab
        outliers_tab = QWidget()
        outliers_layout = QVBoxLayout(outliers_tab)
        
        # Outlier detection group
        outlier_detection_group = QGroupBox("Detect Outliers")
        outlier_layout = QGridLayout(outlier_detection_group)
        
        # Column selector
        outlier_col_label = QLabel("Select Column:")
        self.outlier_column_combo = QComboBox()
        outlier_layout.addWidget(outlier_col_label, 0, 0)
        outlier_layout.addWidget(self.outlier_column_combo, 0, 1)
        
        # Method selector
        outlier_method_label = QLabel("Detection Method:")
        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems([
            "IQR Method",
            "Z-Score Method",
            "Modified Z-Score"
        ])
        outlier_layout.addWidget(outlier_method_label, 1, 0)
        outlier_layout.addWidget(self.outlier_method_combo, 1, 1)
        
        # Threshold input
        threshold_label = QLabel("Threshold:")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 10)
        self.threshold_spin.setValue(3)
        outlier_layout.addWidget(threshold_label, 2, 0)
        outlier_layout.addWidget(self.threshold_spin, 2, 1)
        
        # Add view options
        view_options_layout = QHBoxLayout()
        self.show_only_outliers = QCheckBox("Show Only Outliers")
        self.show_only_outliers.setChecked(False)
        view_options_layout.addWidget(self.show_only_outliers)
        
        # Add sort options
        sort_label = QLabel("Sort By:")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Value", "Status"])
        view_options_layout.addWidget(sort_label)
        view_options_layout.addWidget(self.sort_combo)
        
        outlier_layout.addLayout(view_options_layout, 3, 0, 1, 2)
        
        # Results table
        self.outlier_table = QTableWidget()
        self.outlier_table.setColumnCount(2)
        self.outlier_table.setHorizontalHeaderLabels(["Value", "Is Outlier"])
        outlier_layout.addWidget(self.outlier_table, 4, 0, 1, 2)
        
        # Handle outliers group
        outlier_handling_group = QGroupBox("Handle Outliers")
        outlier_handling_layout = QGridLayout(outlier_handling_group)
        
        # Method selector
        handling_method_label = QLabel("Handling Method:")
        self.handling_method_combo = QComboBox()
        self.handling_method_combo.addItems([
            "Remove outliers",
            "Cap outliers",
            "Replace with mean",
            "Replace with median"
        ])
        outlier_handling_layout.addWidget(handling_method_label, 0, 0)
        outlier_handling_layout.addWidget(self.handling_method_combo, 0, 1)
        
        # Apply button
        self.apply_outlier_btn = QPushButton("Apply")
        outlier_handling_layout.addWidget(self.apply_outlier_btn, 1, 0, 1, 2)
        
        outliers_layout.addWidget(outlier_detection_group)
        outliers_layout.addWidget(outlier_handling_group)
        tabs.addTab(outliers_tab, "Outliers")
        
        # Transformation Tab
        transform_tab = QWidget()
        transform_layout = QVBoxLayout(transform_tab)
        
        # Column selection group
        transform_col_group = QGroupBox("Select Columns")
        transform_col_layout = QGridLayout(transform_col_group)
        
        self.transform_columns_table = QTableWidget()
        self.transform_columns_table.setColumnCount(2)
        self.transform_columns_table.setHorizontalHeaderLabels(["Column", "Transform"])
        transform_col_layout.addWidget(self.transform_columns_table, 0, 0)
        
        transform_layout.addWidget(transform_col_group)
        
        # Transformation options group
        transform_options_group = QGroupBox("Transformation Options")
        transform_options_layout = QGridLayout(transform_options_group)
        
        # Method selector
        transform_method_label = QLabel("Transform Method:")
        self.transform_method_combo = QComboBox()
        self.transform_method_combo.addItems([
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling",
            "Log Transform",
            "Square Root Transform",
            "Box-Cox Transform"
        ])
        transform_options_layout.addWidget(transform_method_label, 0, 0)
        transform_options_layout.addWidget(self.transform_method_combo, 0, 1)
        
        # Apply button
        self.apply_transform_btn = QPushButton("Apply Transformation")
        transform_options_layout.addWidget(self.apply_transform_btn, 1, 0, 1, 2)
        
        transform_layout.addWidget(transform_options_group)
        tabs.addTab(transform_tab, "Transformations")
        
        layout.addWidget(tabs)
        
        # Add undo/redo buttons
        button_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        button_layout.addWidget(self.undo_btn)
        button_layout.addWidget(self.redo_btn)
        layout.addLayout(button_layout)
        
    def setup_connections(self):
        """Setup signal connections."""
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        self.missing_method_combo.currentTextChanged.connect(self.on_missing_method_changed)
        self.apply_missing_btn.clicked.connect(self.apply_missing_values_handling)
        self.apply_outlier_btn.clicked.connect(self.apply_outlier_handling)
        self.apply_transform_btn.clicked.connect(self.apply_transformation)
        self.outlier_column_combo.currentTextChanged.connect(lambda: self.detect_outliers(show_info=False))
        self.outlier_method_combo.currentTextChanged.connect(lambda: self.detect_outliers(show_info=False))
        self.threshold_spin.valueChanged.connect(lambda: self.detect_outliers(show_info=False))
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        self.show_only_outliers.stateChanged.connect(self.update_outlier_view)
        self.sort_combo.currentTextChanged.connect(self.update_outlier_view)
        
    def on_data_loaded(self, df):
        """Handle when new data is loaded."""
        # Update column selectors
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.missing_column_combo.clear()
        self.outlier_column_combo.clear()
        self.missing_column_combo.addItems(df.columns)
        self.outlier_column_combo.addItems(numeric_columns)
        
        # Update missing values table
        self.update_missing_values_info()
        
        # Update transformation columns table
        self.update_transform_columns_table()
        
    def update_missing_values_info(self):
        """Update the missing values information table."""
        df = self.data_manager.data
        if df is None:
            return
            
        missing_info = df.isnull().sum()
        missing_percent = (missing_info / len(df)) * 100
        
        self.missing_table.setRowCount(len(df.columns))
        
        for i, (col, count) in enumerate(missing_info.items()):
            self.missing_table.setItem(i, 0, QTableWidgetItem(col))
            self.missing_table.setItem(i, 1, QTableWidgetItem(str(count)))
            self.missing_table.setItem(i, 2, QTableWidgetItem(f"{missing_percent[col]:.2f}%"))
            
        self.missing_table.resizeColumnsToContents()
        
    def on_missing_method_changed(self, method):
        """Enable/disable custom value input based on selected method."""
        self.missing_value_edit.setEnabled(method == "Fill with value")
        
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
            
    def apply_missing_values_handling(self):
        """Apply the selected missing values handling method."""
        if self.data_manager.data is None:
            return
            
        progress = QProgressDialog("Processing...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        def operation():
            df = self.data_manager.data.copy()
            column = self.missing_column_combo.currentText()
            method = self.missing_method_combo.currentText()
            
            progress.setValue(20)
            
            if method == "Drop rows":
                df = df.dropna(subset=[column])
            elif method == "Fill with mean":
                df[column] = df[column].fillna(df[column].mean())
            elif method == "Fill with median":
                df[column] = df[column].fillna(df[column].median())
            elif method == "Fill with mode":
                df[column] = df[column].fillna(df[column].mode()[0])
            elif method == "Fill with value":
                value = float(self.missing_value_edit.text())
                df[column] = df[column].fillna(value)
            elif method == "Forward fill":
                df[column] = df[column].fillna(method='ffill')
            elif method == "Backward fill":
                df[column] = df[column].fillna(method='bfill')
                
            progress.setValue(80)
            
            self.data_manager._data = df
            QTimer.singleShot(100, lambda: self.data_manager.data_loaded.emit(df))
            
            progress.setValue(100)
            QMessageBox.information(self, "Success", "Missing values handled successfully!")
            
        try:
            self.apply_operation(operation)
        finally:
            progress.close()
            
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
            
    def update_transform_columns_table(self):
        """Update the transformation columns table."""
        df = self.data_manager.data
        if df is None:
            return
            
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.transform_columns_table.setRowCount(len(numeric_columns))
        
        for i, col in enumerate(numeric_columns):
            # Column name
            self.transform_columns_table.setItem(i, 0, QTableWidgetItem(col))
            
            # Checkbox for selection
            checkbox = QCheckBox()
            self.transform_columns_table.setCellWidget(i, 1, checkbox)
            
        self.transform_columns_table.resizeColumnsToContents()
        
    def apply_transformation(self):
        """Apply the selected transformation to the selected columns."""
        if self.data_manager.data is None:
            return
            
        progress = QProgressDialog("Applying transformation...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        try:
            df = self.data_manager.data.copy()
            method = self.transform_method_combo.currentText()
            
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
            if method == "Standard Scaling":
                scaler = StandardScaler()
                df[selected_columns] = scaler.fit_transform(df[selected_columns])
            elif method == "Min-Max Scaling":
                scaler = MinMaxScaler()
                df[selected_columns] = scaler.fit_transform(df[selected_columns])
            elif method == "Robust Scaling":
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
            elif method == "Square Root Transform":
                for col in selected_columns:
                    if (df[col] < 0).any():
                        QMessageBox.warning(self, "Warning",
                                         f"Column {col} contains negative values. "
                                         "Square root transform requires non-negative values.")
                        continue
                    df[col] = np.sqrt(df[col])
            elif method == "Box-Cox Transform":
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