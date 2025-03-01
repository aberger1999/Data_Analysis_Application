"""
Analysis panel for data analysis operations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QComboBox, QPushButton, QTableWidget,
    QTableWidgetItem, QStackedWidget, QMessageBox,
    QGridLayout, QSplitter, QSizePolicy, QStyledItemDelegate
)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QColor, QFont, QPainter, QBrush
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import matplotlib.pyplot as plt

# Add this new class for custom cell rendering
class StatisticsItemDelegate(QStyledItemDelegate):
    """Custom delegate for rendering statistics table cells."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.header_rows = {}
        self.separator_rows = []
        
    def set_header_rows(self, header_rows):
        """Set which rows contain headers."""
        self.header_rows = header_rows
        
    def set_separator_rows(self, separator_rows):
        """Set which rows are separators."""
        self.separator_rows = separator_rows
        
    def paint(self, painter, option, index):
        """Custom painting for cells."""
        # Get the row and column
        row = index.row()
        
        # Set background color based on row type
        if row in self.header_rows:
            # Header row - blue background
            painter.fillRect(option.rect, QBrush(QColor(40, 80, 120)))
            
            # Set text color to white and make it bold
            painter.setPen(QColor(255, 255, 255))
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
        elif row in self.separator_rows:
            # Separator row - dark gray background
            painter.fillRect(option.rect, QBrush(QColor(50, 50, 50)))
            return  # No text for separators
        else:
            # Regular row - default background
            painter.fillRect(option.rect, QBrush(QColor(30, 30, 30)))
            painter.setPen(QColor(255, 255, 255))
        
        # Draw the text
        text = index.data()
        if text:
            # Add some padding
            text_rect = QRect(option.rect)
            text_rect.setLeft(text_rect.left() + 5)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, text)

class AnalysisPanel(QWidget):
    """Panel for data analysis operations."""
    
    def __init__(self, data_manager):
        """Initialize the analysis panel."""
        super().__init__()
        self.data_manager = data_manager
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        
        # Create a more compact control panel at the top
        control_panel = QFrame()
        control_panel.setStyleSheet("background-color: #2a2a2a; border-radius: 5px; padding: 10px; color: white;")
        control_layout = QHBoxLayout(control_panel)
        
        # Column selection with label
        control_layout.addWidget(QLabel("Select Column:"))
        self.column_combo = QComboBox()
        self.column_combo.setMinimumWidth(200)
        self.column_combo.setStyleSheet("background-color: #3a3a3a; color: white; selection-background-color: #505050;")
        control_layout.addWidget(self.column_combo)
        
        control_layout.addSpacing(20)
        
        # View selection buttons
        self.stats_btn = QPushButton("Statistics View")
        self.stats_btn.setCheckable(True)
        self.stats_btn.setChecked(True)
        self.stats_btn.setStyleSheet("""
            QPushButton { background-color: #3a3a3a; color: white; border: 1px solid #505050; }
            QPushButton:checked { background-color: #4CAF50; color: white; }
        """)
        
        self.viz_btn = QPushButton("Visualization View")
        self.viz_btn.setCheckable(True)
        self.viz_btn.setStyleSheet("""
            QPushButton { background-color: #3a3a3a; color: white; border: 1px solid #505050; }
            QPushButton:checked { background-color: #4CAF50; color: white; }
        """)
        
        control_layout.addWidget(self.stats_btn)
        control_layout.addWidget(self.viz_btn)
        
        # Add run button
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setStyleSheet("background-color: #2196F3; color: white; border: none;")
        control_layout.addWidget(self.run_btn)
        
        main_layout.addWidget(control_panel)
        
        # Create a stacked widget to switch between statistics and visualization
        self.analysis_stack = QStackedWidget()
        self.analysis_stack.setStyleSheet("background-color: #1e1e1e;")
        
        # Create statistics frame with better styling
        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet("background-color: #1e1e1e; color: white;")
        stats_layout = QVBoxLayout(self.stats_frame)
        stats_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins to use full space
        
        # Results table for statistics with better styling
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Statistic", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Make table expand to fill space
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #505050;
                gridline-color: #505050;
                selection-background-color: #3a3a3a;
            }
            QHeaderView::section {
                background-color: #2a2a2a;
                color: white;
                padding: 5px;
                border: 1px solid #505050;
                font-weight: bold;
            }
            QTableWidget::item:alternate {
                background-color: #262626;
            }
        """)
        stats_layout.addWidget(self.results_table)
        
        # Add stats frame to stack
        self.analysis_stack.addWidget(self.stats_frame)
        
        # Create visualization frame
        self.viz_frame = QFrame()
        self.viz_frame.setStyleSheet("background-color: #1e1e1e; color: white;")
        viz_layout = QVBoxLayout(self.viz_frame)
        viz_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins to use full space
        
        # Add visualization options in a styled panel
        viz_control = QFrame()
        viz_control.setStyleSheet("background-color: #2a2a2a; border-radius: 5px; padding: 5px; color: white;")
        viz_control_layout = QHBoxLayout(viz_control)
        
        self.viz_type_label = QLabel("Visualization Type:")
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.setMinimumWidth(150)
        self.viz_type_combo.setStyleSheet("background-color: #3a3a3a; color: white; selection-background-color: #505050;")
        
        viz_control_layout.addWidget(self.viz_type_label)
        viz_control_layout.addWidget(self.viz_type_combo)
        viz_control_layout.addStretch()
        
        viz_layout.addWidget(viz_control)
        
        # Add matplotlib canvas for visualization with a border
        canvas_frame = QFrame()
        canvas_frame.setStyleSheet("border: 1px solid #505050; background-color: #1e1e1e;")
        canvas_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Make canvas expand to fill space
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins to use full space
        
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Make canvas expand to fill space
        self.figure.patch.set_facecolor('#1e1e1e')
        canvas_layout.addWidget(self.canvas)
        
        viz_layout.addWidget(canvas_frame)
        
        # Add viz frame to stack
        self.analysis_stack.addWidget(self.viz_frame)
        
        # Add stack to main layout - make it take up all available space
        main_layout.addWidget(self.analysis_stack, 1)  # Add stretch factor of 1 to make it expand
        
    def setup_connections(self):
        """Setup signal connections."""
        self.run_btn.clicked.connect(self.run_analysis)
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        self.stats_btn.clicked.connect(self.toggle_view)
        self.viz_btn.clicked.connect(self.toggle_view)
        self.viz_type_combo.currentIndexChanged.connect(self.update_visualization)
        self.column_combo.currentTextChanged.connect(self.on_column_changed)
        
    def toggle_view(self):
        """Toggle between statistics and visualization views."""
        if self.sender() == self.stats_btn:
            self.stats_btn.setChecked(True)
            self.viz_btn.setChecked(False)
            self.analysis_stack.setCurrentIndex(0)
        else:
            self.stats_btn.setChecked(False)
            self.viz_btn.setChecked(True)
            self.analysis_stack.setCurrentIndex(1)
            self.update_viz_options()
        
    def on_data_loaded(self, df):
        """Handle when new data is loaded."""
        # Update column dropdown
        self.column_combo.clear()
        self.column_combo.addItems(df.columns)
        
        # Enable run button
        self.run_btn.setEnabled(True)
        
    def on_column_changed(self, column):
        """Handle column selection change."""
        if self.analysis_stack.currentIndex() == 1:  # Visualization view
            self.update_viz_options()
            
    def update_viz_options(self):
        """Update visualization options based on column data type."""
        if self.data_manager.data is None or not self.column_combo.currentText():
            return
            
        column = self.column_combo.currentText()
        df = self.data_manager.data
        
        self.viz_type_combo.clear()
        
        # Check column data type and add appropriate visualization options
        if pd.api.types.is_numeric_dtype(df[column]):
            self.viz_type_combo.addItems(["Box Plot", "Histogram", "Density Plot"])
        else:
            self.viz_type_combo.addItems(["Bar Chart", "Pie Chart"])
            
        # Update the visualization
        self.update_visualization()
        
    def update_visualization(self):
        """Update the visualization based on selected options."""
        if self.data_manager.data is None or not self.column_combo.currentText():
            return
            
        column = self.column_combo.currentText()
        df = self.data_manager.data
        
        if self.viz_type_combo.count() == 0:
            return
            
        viz_type = self.viz_type_combo.currentText()
        
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Set dark theme for plots
        plt.style.use('dark_background')
        ax.set_facecolor('#1e1e1e')
        
        try:
            if pd.api.types.is_numeric_dtype(df[column]):
                if viz_type == "Box Plot":
                    sns.boxplot(y=df[column], ax=ax, color='#4CAF50')
                    ax.set_title(f"Box Plot of {column}", color='white')
                elif viz_type == "Histogram":
                    sns.histplot(df[column], kde=True, ax=ax, color='#2196F3')
                    ax.set_title(f"Histogram of {column}", color='white')
                elif viz_type == "Density Plot":
                    sns.kdeplot(df[column], ax=ax, color='#FF9800')
                    ax.set_title(f"Density Plot of {column}", color='white')
            else:
                # For categorical data
                value_counts = df[column].value_counts()
                if viz_type == "Bar Chart":
                    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette='viridis')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', color='white')
                    ax.set_title(f"Bar Chart of {column}", color='white')
                elif viz_type == "Pie Chart":
                    ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', colors=plt.cm.viridis(np.linspace(0, 1, len(value_counts))))
                    ax.set_title(f"Pie Chart of {column}", color='white')
            
            # Set labels color to white
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white')
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating visualization: {str(e)}")
            
    def run_analysis(self):
        """Run the selected analysis on the selected column."""
        if self.data_manager.data is None:
            return
            
        column = self.column_combo.currentText()
        
        if not column:
            return
            
        if self.analysis_stack.currentIndex() == 0:  # Statistics view
            self.run_basic_statistics(column)
        else:  # Visualization view
            self.update_visualization()
            
    def run_basic_statistics(self, column):
        """Run basic statistics on the selected column."""
        df = self.data_manager.data
        
        # Clear previous results
        self.results_table.setRowCount(0)
        
        try:
            # Calculate statistics
            stats = []
            
            # Define all section headers
            section_headers = [
                "Basic Information", 
                "Central Tendency", 
                "Dispersion",
                "Quartiles", 
                "Distribution Shape", 
                "Outlier Boundaries (IQR method)", 
                "Correlations", 
                "Frequency Analysis"
            ]
            
            # Basic count statistics (for all data types)
            stats.append(("Basic Information", ""))
            stats.append(("Count", len(df)))
            stats.append(("Missing Values", df[column].isna().sum()))
            stats.append(("Missing Percentage", f"{df[column].isna().sum() / len(df) * 100:.2f}%"))
            stats.append(("Unique Values", df[column].nunique()))
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(df[column]):
                # Central tendency
                stats.append(("", ""))  # Empty row as separator
                stats.append(("Central Tendency", ""))
                stats.append(("Mean", f"{df[column].mean():.4f}"))
                stats.append(("Median", f"{df[column].median():.4f}"))
                stats.append(("Mode", f"{df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A'}"))
                
                # Dispersion
                stats.append(("", ""))  # Empty row as separator
                stats.append(("Dispersion", ""))
                stats.append(("Standard Deviation", f"{df[column].std():.4f}"))
                stats.append(("Variance", f"{df[column].var():.4f}"))
                stats.append(("Range", f"{df[column].max() - df[column].min():.4f}"))
                stats.append(("Min", f"{df[column].min():.4f}"))
                stats.append(("Max", f"{df[column].max():.4f}"))
                
                # Quartiles
                stats.append(("", ""))  # Empty row as separator
                stats.append(("Quartiles", ""))
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                stats.append(("Q1 (25%)", f"{q1:.4f}"))
                stats.append(("Q2 (50%)", f"{df[column].quantile(0.5):.4f}"))
                stats.append(("Q3 (75%)", f"{q3:.4f}"))
                stats.append(("IQR", f"{iqr:.4f}"))
                
                # Shape
                stats.append(("", ""))  # Empty row as separator
                stats.append(("Distribution Shape", ""))
                stats.append(("Skewness", f"{df[column].skew():.4f}"))
                stats.append(("Kurtosis", f"{df[column].kurtosis():.4f}"))
                
                # Outlier boundaries
                stats.append(("", ""))  # Empty row as separator
                stats.append(("Outlier Boundaries (IQR method)", ""))
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                stats.append(("Lower Bound", f"{lower_bound:.4f}"))
                stats.append(("Upper Bound", f"{upper_bound:.4f}"))
                stats.append(("Potential Outliers", f"{((df[column] < lower_bound) | (df[column] > upper_bound)).sum()}"))
                
                # Correlations
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:  # Only if there are other numeric columns
                    stats.append(("", ""))  # Empty row as separator
                    stats.append(("Correlations", ""))
                    for col in numeric_cols:
                        if col != column:
                            corr = df[column].corr(df[col])
                            stats.append((f"Correlation with {col}", f"{corr:.4f}"))
            
            # Categorical statistics
            else:
                # Frequency analysis
                stats.append(("", ""))  # Empty row as separator
                stats.append(("Frequency Analysis", ""))
                value_counts = df[column].value_counts()
                top_n = min(5, len(value_counts))
                
                for i in range(top_n):
                    value = value_counts.index[i]
                    count = value_counts.iloc[i]
                    percentage = count / len(df) * 100
                    stats.append((f"Top {i+1}: {value}", f"{count} ({percentage:.2f}%)"))
            
            # Display results with formatting
            self.results_table.setRowCount(len(stats))
            
            # Create a dictionary to track header rows and separator rows
            header_rows = {}
            separator_rows = []
            
            # First pass: populate the table and track header rows
            for i, (key, value) in enumerate(stats):
                item_key = QTableWidgetItem(str(key))
                item_value = QTableWidgetItem(str(value))
                
                if key in section_headers:
                    # Track this as a header row
                    header_rows[i] = key
                elif key == "":
                    # Track separator rows
                    separator_rows.append(i)
                
                self.results_table.setItem(i, 0, item_key)
                self.results_table.setItem(i, 1, item_value)
            
            # Create and set the custom delegate
            delegate = StatisticsItemDelegate(self.results_table)
            delegate.set_header_rows(header_rows)
            delegate.set_separator_rows(separator_rows)
            
            # Remove any existing delegate
            if hasattr(self, 'stats_delegate'):
                self.results_table.setItemDelegate(None)
            
            # Set the new delegate and store a reference
            self.stats_delegate = delegate
            self.results_table.setItemDelegate(delegate)
            
            # Force a repaint
            self.results_table.viewport().update()
            
            # Set column widths to prevent text cutoff
            self.results_table.setColumnWidth(0, 250)  # Increase width of first column
            self.results_table.setColumnWidth(1, 200)  # Set width of second column
            
            # Make the table stretch to fill available space
            self.results_table.horizontalHeader().setStretchLastSection(True)
            
            # Ensure the table takes up the full available space
            self.results_table.setMinimumWidth(self.stats_frame.width())
            self.results_table.setMinimumHeight(self.stats_frame.height() - 20)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error calculating statistics: {str(e)}") 