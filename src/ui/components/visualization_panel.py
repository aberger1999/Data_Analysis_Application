"""
Visualization panel for creating and customizing data visualizations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QComboBox, QPushButton, QScrollArea,
    QGridLayout, QSpinBox, QColorDialog, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

class VisualizationPanel(QWidget):
    """Panel for creating and customizing visualizations."""
    
    def __init__(self, data_manager):
        """Initialize the visualization panel."""
        super().__init__()
        self.data_manager = data_manager
        
        # Set up matplotlib figure
        plt.style.use('seaborn-v0_8')  # Use a more specific style name
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Set up update timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_visualization)
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Controls container
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        
        # Left controls (Chart type and axes)
        left_frame = QFrame()
        left_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        left_layout = QGridLayout(left_frame)
        
        # Chart type selector
        chart_label = QLabel("Chart Type:")
        self.chart_combo = QComboBox()
        self.chart_combo.addItems([
            "Line Chart",
            "Bar Chart",
            "Scatter Plot",
            "Histogram",
            "Box Plot",
            "Violin Plot",
            "Heatmap",
            "KDE Plot"
        ])
        left_layout.addWidget(chart_label, 0, 0)
        left_layout.addWidget(self.chart_combo, 0, 1)
        
        # Data selection
        x_axis_label = QLabel("X-Axis:")
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.setEnabled(False)
        left_layout.addWidget(x_axis_label, 1, 0)
        left_layout.addWidget(self.x_axis_combo, 1, 1)
        
        y_axis_label = QLabel("Y-Axis:")
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.setEnabled(False)
        left_layout.addWidget(y_axis_label, 2, 0)
        left_layout.addWidget(self.y_axis_combo, 2, 1)
        
        controls_layout.addWidget(left_frame)
        
        # Right controls (Customization)
        right_frame = QFrame()
        right_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        right_layout = QGridLayout(right_frame)
        
        # Title customization
        title_label = QLabel("Chart Title:")
        self.title_edit = QComboBox()
        self.title_edit.setEditable(True)
        right_layout.addWidget(title_label, 0, 0)
        right_layout.addWidget(self.title_edit, 0, 1)
        
        # Style selection
        style_label = QLabel("Plot Style:")
        self.style_combo = QComboBox()
        self.style_combo.addItems([
            "seaborn-v0_8",
            "seaborn-v0_8-darkgrid",
            "seaborn-v0_8-whitegrid",
            "ggplot",
            "default",
            "dark_background"
        ])
        right_layout.addWidget(style_label, 1, 0)
        right_layout.addWidget(self.style_combo, 1, 1)
        
        # Additional options
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        right_layout.addWidget(self.grid_check, 2, 0)
        
        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(True)
        right_layout.addWidget(self.legend_check, 2, 1)
        
        controls_layout.addWidget(right_frame)
        layout.addWidget(controls_container)
        
        # Preview area with matplotlib canvas
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.addWidget(self.canvas)
        
        # Placeholder message
        self.placeholder = QLabel("Import data to create visualizations")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #666; font-style: italic; padding: 20px;")
        preview_layout.addWidget(self.placeholder)
        
        layout.addWidget(preview_frame)
        layout.setStretchFactor(preview_frame, 1)
        
    def setup_connections(self):
        """Setup signal connections."""
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        self.chart_combo.currentTextChanged.connect(self.schedule_update)
        self.x_axis_combo.currentTextChanged.connect(self.schedule_update)
        self.y_axis_combo.currentTextChanged.connect(self.schedule_update)
        self.title_edit.editTextChanged.connect(self.schedule_update)
        self.style_combo.currentTextChanged.connect(self.schedule_update)
        self.grid_check.stateChanged.connect(self.schedule_update)
        self.legend_check.stateChanged.connect(self.schedule_update)
        
    def schedule_update(self):
        """Schedule a visualization update with a delay to prevent rapid updates."""
        self.update_timer.start(300)  # 300ms delay
        
    def on_data_loaded(self, df):
        """Handle when new data is loaded."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        
        self.x_axis_combo.addItems(numeric_columns)
        self.y_axis_combo.addItems(numeric_columns)
        
        if len(numeric_columns) >= 2:
            self.x_axis_combo.setCurrentIndex(0)
            self.y_axis_combo.setCurrentIndex(1)
        
        self.x_axis_combo.setEnabled(True)
        self.y_axis_combo.setEnabled(True)
        
        self.placeholder.setVisible(False)
        self.canvas.setVisible(True)
        
        self.schedule_update()
        
    def update_visualization(self):
        """Update the visualization with current settings."""
        if self.data_manager.data is None:
            return
            
        # Clear the figure
        self.figure.clear()
        
        # Get the data
        df = self.data_manager.data
        x_col = self.x_axis_combo.currentText()
        y_col = self.y_axis_combo.currentText()
        
        if not x_col or not y_col:
            return
            
        # Set the style
        plt.style.use(self.style_combo.currentText())
        
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        try:
            chart_type = self.chart_combo.currentText()
            
            if chart_type == "Line Chart":
                ax.plot(df[x_col], df[y_col], marker='o', label=y_col)
            elif chart_type == "Bar Chart":
                ax.bar(df[x_col], df[y_col], alpha=0.8, label=y_col)
            elif chart_type == "Scatter Plot":
                ax.scatter(df[x_col], df[y_col], alpha=0.6, label=y_col)
            elif chart_type == "Histogram":
                ax.hist(df[x_col], bins=30, alpha=0.7, density=True, label='Density')
                if len(df[x_col].dropna()) > 1:  # Only add KDE if we have enough data points
                    density = stats.gaussian_kde(df[x_col].dropna())
                    xs = np.linspace(df[x_col].min(), df[x_col].max(), 200)
                    ax.plot(xs, density(xs), 'r-', label='KDE')
                ax.set_ylabel('Density')
            elif chart_type == "Box Plot":
                sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
            elif chart_type == "Violin Plot":
                sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)
            elif chart_type == "Heatmap":
                corr = df[[x_col, y_col]].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, center=0)
            elif chart_type == "KDE Plot":
                sns.kdeplot(data=df, x=x_col, y=y_col, ax=ax, cmap='viridis', label='KDE')
                
            # Set title
            title = self.title_edit.currentText() or f"{chart_type}: {x_col} vs {y_col}"
            ax.set_title(title)
            
            # Set labels
            ax.set_xlabel(x_col)
            if chart_type not in ["Histogram", "KDE Plot"]:
                ax.set_ylabel(y_col)
            
            # Configure grid
            ax.grid(self.grid_check.isChecked())
            
            # Configure legend if applicable and checked
            if self.legend_check.isChecked():
                # Only show legend for plots that have labeled elements
                if chart_type not in ["Heatmap", "Box Plot", "Violin Plot"]:
                    handles, labels = ax.get_legend_handles_labels()
                    if len(handles) > 0:  # Only show legend if we have labeled elements
                        ax.legend()
                
            # Adjust layout to prevent label cutoff
            self.figure.tight_layout()
            
        except Exception as e:
            # Clear the figure and show error
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', color='red',
                   transform=ax.transAxes)
            
        # Redraw the canvas
        self.canvas.draw() 