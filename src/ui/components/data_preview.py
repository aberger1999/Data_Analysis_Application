"""
Data preview panel for displaying and basic manipulation of loaded data.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QHeaderView, QHBoxLayout, QPushButton, QSpinBox,
    QComboBox, QLineEdit, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer

class DataPreviewPanel(QWidget):
    """Panel for previewing loaded data."""

    MAX_ROWS_PER_PAGE = 100  # Number of rows to show per page
    
    def __init__(self, data_manager):
        """Initialize the data preview panel."""
        super().__init__()
        self.data_manager = data_manager
        self.current_page = 0
        self.total_pages = 0
        self.filtered_data = None
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Filter group
        filter_group = QGroupBox("Filter Data")
        filter_layout = QGridLayout(filter_group)
        
        # Column filter
        col_filter_label = QLabel("Filter Column:")
        self.filter_column_combo = QComboBox()
        filter_layout.addWidget(col_filter_label, 0, 0)
        filter_layout.addWidget(self.filter_column_combo, 0, 1)
        
        # Filter condition
        condition_label = QLabel("Condition:")
        self.filter_condition_combo = QComboBox()
        self.filter_condition_combo.addItems([
            "equals",
            "not equals",
            "greater than",
            "less than",
            "contains",
            "starts with",
            "ends with"
        ])
        filter_layout.addWidget(condition_label, 0, 2)
        filter_layout.addWidget(self.filter_condition_combo, 0, 3)
        
        # Filter value
        value_label = QLabel("Value:")
        self.filter_value_edit = QLineEdit()
        filter_layout.addWidget(value_label, 0, 4)
        filter_layout.addWidget(self.filter_value_edit, 0, 5)
        
        # Filter buttons
        button_layout = QHBoxLayout()
        self.apply_filter_btn = QPushButton("Apply Filter")
        self.clear_filter_btn = QPushButton("Clear Filter")
        button_layout.addWidget(self.apply_filter_btn)
        button_layout.addWidget(self.clear_filter_btn)
        filter_layout.addLayout(button_layout, 1, 0, 1, 6)
        
        layout.addWidget(filter_group)
        
        # Info and navigation layout
        top_layout = QHBoxLayout()
        
        # Info label
        self.info_label = QLabel("No data loaded")
        top_layout.addWidget(self.info_label)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setEnabled(False)
        self.next_btn = QPushButton("Next")
        self.next_btn.setEnabled(False)
        
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setEnabled(False)
        
        self.total_pages_label = QLabel("of 1")
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.page_spin)
        nav_layout.addWidget(self.total_pages_label)
        nav_layout.addWidget(self.next_btn)
        
        top_layout.addLayout(nav_layout)
        layout.addLayout(top_layout)
        
        # Data table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        
    def setup_connections(self):
        """Setup signal connections."""
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        self.prev_btn.clicked.connect(self.previous_page)
        self.next_btn.clicked.connect(self.next_page)
        self.page_spin.valueChanged.connect(self.go_to_page)
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        self.clear_filter_btn.clicked.connect(self.clear_filter)
        
    def on_data_loaded(self, df):
        """Handle when new data is loaded."""
        if df is None:
            self.info_label.setText("No data loaded")
            self.table.clear()
            self.update_navigation(0)
            self.filter_column_combo.clear()
            return
            
        # Update filter columns
        self.filter_column_combo.clear()
        self.filter_column_combo.addItems(df.columns)
        
        self.filtered_data = df  # Reset filtered data
        self.update_table_view()
        
    def apply_filter(self):
        """Apply the filter to the data."""
        if self.data_manager.data is None:
            return
            
        df = self.data_manager.data
        column = self.filter_column_combo.currentText()
        condition = self.filter_condition_combo.currentText()
        value = self.filter_value_edit.text()
        
        try:
            # Convert value based on column type
            if df[column].dtype in ['int64', 'float64']:
                value = float(value)
            
            # Apply filter condition
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
            elif condition == "starts with":
                mask = df[column].astype(str).str.startswith(str(value), na=False)
            elif condition == "ends with":
                mask = df[column].astype(str).str.endswith(str(value), na=False)
            
            self.filtered_data = df[mask]
            self.current_page = 0
            self.update_table_view()
            
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Filter Error", f"Error applying filter: {str(e)}")
        
    def clear_filter(self):
        """Clear the current filter."""
        self.filtered_data = self.data_manager.data
        self.filter_value_edit.clear()
        self.current_page = 0
        self.update_table_view()
        
    def update_table_view(self):
        """Update the table view with current data."""
        if self.filtered_data is None:
            self.info_label.setText("No data loaded")
            self.table.clear()
            self.update_navigation(0)
            return
            
        # Calculate total pages
        self.total_pages = (len(self.filtered_data) + self.MAX_ROWS_PER_PAGE - 1) // self.MAX_ROWS_PER_PAGE
        
        # Update info label
        total_rows = len(self.data_manager.data)
        filtered_rows = len(self.filtered_data)
        if filtered_rows < total_rows:
            self.info_label.setText(f"Showing {filtered_rows} of {total_rows} rows × {self.filtered_data.shape[1]} columns")
        else:
            self.info_label.setText(f"Loaded data: {filtered_rows} rows × {self.filtered_data.shape[1]} columns")
        
        # Set up table structure
        self.table.setColumnCount(self.filtered_data.shape[1])
        self.table.setHorizontalHeaderLabels(self.filtered_data.columns)
        
        # Update navigation controls
        self.update_navigation(self.total_pages)
        
        # Show current page
        self.page_spin.setValue(self.current_page + 1)
        self.update_current_page()
        
    def update_navigation(self, total_pages):
        """Update navigation controls based on total pages."""
        self.total_pages = total_pages
        self.page_spin.setEnabled(total_pages > 0)
        self.page_spin.setMaximum(max(1, total_pages))
        self.total_pages_label.setText(f"of {max(1, total_pages)}")
        self.update_navigation_buttons()
        
    def update_navigation_buttons(self):
        """Update the enabled state of navigation buttons."""
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < self.total_pages - 1)
        
    def previous_page(self):
        """Show the previous page of data."""
        if self.current_page > 0:
            self.current_page -= 1
            self.page_spin.setValue(self.current_page + 1)
            self.update_current_page()
        
    def next_page(self):
        """Show the next page of data."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.page_spin.setValue(self.current_page + 1)
            self.update_current_page()
            
    def go_to_page(self, page_number):
        """Go to a specific page number."""
        if page_number != self.current_page + 1:
            self.current_page = page_number - 1
            self.update_current_page()
        
    def update_current_page(self):
        """Update the table with the current page of data."""
        if self.filtered_data is None:
            return
            
        start_idx = self.current_page * self.MAX_ROWS_PER_PAGE
        end_idx = min(start_idx + self.MAX_ROWS_PER_PAGE, len(self.filtered_data))
        
        # Update table
        self.table.setRowCount(end_idx - start_idx)
        self.table.setVerticalHeaderLabels([str(i) for i in range(start_idx, end_idx)])
        
        # Disable table updates for better performance
        self.table.setUpdatesEnabled(False)
        
        try:
            # Populate data for current page
            for row_idx, df_idx in enumerate(range(start_idx, end_idx)):
                for col_idx in range(self.filtered_data.shape[1]):
                    value = str(self.filtered_data.iloc[df_idx, col_idx])
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only
                    self.table.setItem(row_idx, col_idx, item)
                    
            # Adjust column widths if this is the first page
            if self.current_page == 0:
                header = self.table.horizontalHeader()
                for i in range(self.filtered_data.shape[1]):
                    header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        finally:
            # Re-enable table updates
            self.table.setUpdatesEnabled(True)
            
        self.update_navigation_buttons() 