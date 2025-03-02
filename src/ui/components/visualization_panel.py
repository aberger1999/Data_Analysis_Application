"""
Visualization panel for creating and customizing data visualizations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QComboBox, QPushButton, QScrollArea,
    QGridLayout, QSpinBox, QColorDialog, QCheckBox,
    QFileDialog, QListWidget, QToolButton, QSlider,
    QSizePolicy, QGroupBox, QTabWidget, QSplitter,
    QLineEdit
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import os

class VisualizationPanel(QWidget):
    """Panel for creating and customizing visualizations."""
    
    def __init__(self, data_manager):
        """Initialize the visualization panel."""
        super().__init__()
        self.data_manager = data_manager
        
        # Set up matplotlib figure
        plt.style.use('seaborn-v0_8')  # Use a more specific style name
        self.figure = Figure(figsize=(10, 8), dpi=100)  # Increased figure size
        self.canvas = FigureCanvas(self.figure)
        
        # Set up navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Store selected colors for series
        self.series_colors = {}
        self.custom_color = None
        self.color_map = None
        
        # Store selected series for multi-series plots
        self.selected_series = []
        
        # Set up update timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_visualization)
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        
        # Create a vertical splitter to allow resizing
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        
        # Controls container using tabs for better organization
        controls_container = QWidget()
        controls_tabs = QTabWidget(controls_container)
        controls_tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Data Selection Tab
        data_tab = QWidget()
        data_layout = QGridLayout(data_tab)
        
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
            "KDE Plot",
            "Pie Chart",
            "Area Chart"
        ])
        data_layout.addWidget(chart_label, 0, 0)
        data_layout.addWidget(self.chart_combo, 0, 1)
        
        # Data selection
        x_axis_label = QLabel("X-Axis:")
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.setEnabled(False)
        data_layout.addWidget(x_axis_label, 1, 0)
        data_layout.addWidget(self.x_axis_combo, 1, 1)
        
        y_axis_label = QLabel("Y-Axis:")
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.setEnabled(False)
        data_layout.addWidget(y_axis_label, 2, 0)
        data_layout.addWidget(self.y_axis_combo, 2, 1)
        
        # Color by selector for scatter plots
        color_by_label = QLabel("Color By:")
        self.color_by_combo = QComboBox()
        self.color_by_combo.setEnabled(False)
        data_layout.addWidget(color_by_label, 3, 0)
        data_layout.addWidget(self.color_by_combo, 3, 1)
        
        # Multiple series selection
        series_label = QLabel("Data Series:")
        self.series_list = QListWidget()
        self.series_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.series_list.setMaximumHeight(100)
        self.series_list.setEnabled(False)
        data_layout.addWidget(series_label, 4, 0)
        data_layout.addWidget(self.series_list, 4, 1)
        
        controls_tabs.addTab(data_tab, "Data Selection")
        
        # Style Tab
        style_tab = QWidget()
        style_layout = QGridLayout(style_tab)
        
        # Title customization - changed from QComboBox to QLineEdit
        title_label = QLabel("Chart Title:")
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Enter chart title")
        style_layout.addWidget(title_label, 0, 0)
        style_layout.addWidget(self.title_edit, 0, 1)
        
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
        style_layout.addWidget(style_label, 1, 0)
        style_layout.addWidget(self.style_combo, 1, 1)
        
        # Color selection
        color_theme_label = QLabel("Color Theme:")
        self.color_combo = QComboBox()
        self.color_combo.addItems([
            "Default",
            "Viridis",
            "Plasma",
            "Inferno",
            "Magma",
            "Cividis",
            "Blues",
            "Reds",
            "Greens",
            "Custom"
        ])
        style_layout.addWidget(color_theme_label, 2, 0)
        style_layout.addWidget(self.color_combo, 2, 1)
        
        # Custom color button
        self.color_button = QPushButton("Choose Color")
        self.color_button.setEnabled(False)
        style_layout.addWidget(self.color_button, 3, 1)
        
        # Additional options
        options_group = QGroupBox("Display Options")
        options_layout = QGridLayout(options_group)
        
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        options_layout.addWidget(self.grid_check, 0, 0)
        
        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(True)
        options_layout.addWidget(self.legend_check, 0, 1)
        
        style_layout.addWidget(options_group, 4, 0, 1, 2)
        
        controls_tabs.addTab(style_tab, "Style")
        
        # Advanced Tab
        advanced_tab = QWidget()
        advanced_layout = QGridLayout(advanced_tab)
        
        # Transparency slider
        alpha_label = QLabel("Transparency:")
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(10, 100)
        self.alpha_slider.setValue(80)
        advanced_layout.addWidget(alpha_label, 0, 0)
        advanced_layout.addWidget(self.alpha_slider, 0, 1)
        
        # Marker size for scatter plots
        marker_label = QLabel("Marker Size:")
        self.marker_size = QSpinBox()
        self.marker_size.setRange(1, 20)
        self.marker_size.setValue(6)
        advanced_layout.addWidget(marker_label, 1, 0)
        advanced_layout.addWidget(self.marker_size, 1, 1)
        
        # Number of bins for histograms
        bins_label = QLabel("Histogram Bins:")
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(5, 100)
        self.bins_spinbox.setValue(30)
        advanced_layout.addWidget(bins_label, 2, 0)
        advanced_layout.addWidget(self.bins_spinbox, 2, 1)
        
        # Export buttons
        export_group = QGroupBox("Export Options")
        export_layout = QHBoxLayout(export_group)
        
        self.export_png_btn = QPushButton("Export PNG")
        self.export_pdf_btn = QPushButton("Export PDF")
        self.export_svg_btn = QPushButton("Export SVG")
        
        export_layout.addWidget(self.export_png_btn)
        export_layout.addWidget(self.export_pdf_btn)
        export_layout.addWidget(self.export_svg_btn)
        
        advanced_layout.addWidget(export_group, 3, 0, 1, 2)
        
        controls_tabs.addTab(advanced_tab, "Advanced")
        
        # Set up tab container layout
        controls_container_layout = QVBoxLayout(controls_container)
        controls_container_layout.addWidget(controls_tabs)
        controls_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Preview area with matplotlib canvas
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add toolbar for interactive navigation
        preview_layout.addWidget(self.toolbar)
        
        # Create a frame for the canvas
        canvas_frame = QFrame()
        canvas_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        canvas_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.addWidget(self.canvas)
        
        # Placeholder message
        self.placeholder = QLabel("Import data to create visualizations")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #666; font-style: italic; padding: 20px;")
        canvas_layout.addWidget(self.placeholder)
        
        preview_layout.addWidget(canvas_frame)
        
        # Add widgets to splitter and set initial sizes
        splitter.addWidget(controls_container)
        splitter.addWidget(preview_container)
        
        # Set initial splitter sizes (30% controls, 70% chart)
        splitter.setSizes([300, 700])
        
        main_layout.addWidget(splitter)
        
    def setup_connections(self):
        """Setup signal connections."""
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        self.chart_combo.currentTextChanged.connect(self.on_chart_type_changed)
        self.x_axis_combo.currentTextChanged.connect(self.schedule_update)
        self.y_axis_combo.currentTextChanged.connect(self.schedule_update)
        self.color_by_combo.currentTextChanged.connect(self.schedule_update)
        self.series_list.itemSelectionChanged.connect(self.on_series_selection_changed)
        self.title_edit.textChanged.connect(self.schedule_update)  # Changed from editTextChanged to textChanged
        self.style_combo.currentTextChanged.connect(self.schedule_update)
        self.color_combo.currentTextChanged.connect(self.on_color_theme_changed)
        self.color_button.clicked.connect(self.choose_custom_color)
        self.grid_check.stateChanged.connect(self.schedule_update)
        self.legend_check.stateChanged.connect(self.schedule_update)
        self.alpha_slider.valueChanged.connect(self.schedule_update)
        self.marker_size.valueChanged.connect(self.schedule_update)
        self.bins_spinbox.valueChanged.connect(self.schedule_update)
        
        # Export connections
        self.export_png_btn.clicked.connect(lambda: self.export_plot('png'))
        self.export_pdf_btn.clicked.connect(lambda: self.export_plot('pdf'))
        self.export_svg_btn.clicked.connect(lambda: self.export_plot('svg'))
        
    def on_chart_type_changed(self, chart_type):
        """Handle chart type changes by updating UI components."""
        # Enable/disable controls based on chart type
        is_pie_chart = chart_type == "Pie Chart"
        is_histogram = chart_type == "Histogram"
        is_scatter_plot = chart_type == "Scatter Plot"
        
        # Pie charts only need one variable (Y-axis)
        self.x_axis_combo.setEnabled(not is_pie_chart)
        
        # Enable color by option only for scatter plots
        self.color_by_combo.setEnabled(is_scatter_plot)
        
        # Update controls and schedule visualization update
        self.schedule_update()
        
    def on_color_theme_changed(self, theme):
        """Handle color theme selection."""
        self.color_button.setEnabled(theme == "Custom")
        self.color_map = theme.lower() if theme != "Default" and theme != "Custom" else None
        self.schedule_update()
        
    def choose_custom_color(self):
        """Open color dialog for custom color selection."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.custom_color = (color.red()/255, color.green()/255, color.blue()/255)
            self.schedule_update()
            
    def schedule_update(self):
        """Schedule a visualization update with a delay to prevent rapid updates."""
        self.update_timer.start(300)  # 300ms delay
        
    def on_data_loaded(self, df):
        """Handle when new data is loaded."""
        # Get all columns
        all_columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        self.color_by_combo.clear()
        self.series_list.clear()
        
        # Add "None" option to dropdowns
        self.x_axis_combo.addItem("None")
        self.y_axis_combo.addItem("None")
        self.color_by_combo.addItem("None")
        
        # For X-axis, offer all columns for categorical options
        self.x_axis_combo.addItems(all_columns)
        
        # For Y-axis, only offer numeric columns
        self.y_axis_combo.addItems(numeric_columns)
        
        # For Color By, offer all columns
        self.color_by_combo.addItems(all_columns)
        
        # Add numeric columns to the series list for multi-series selection
        self.series_list.addItems(numeric_columns)
        
        if len(numeric_columns) >= 2:
            self.x_axis_combo.setCurrentIndex(1)  # Skip "None"
            self.y_axis_combo.setCurrentIndex(2)  # Skip "None" and first option
        
        self.x_axis_combo.setEnabled(True)
        self.y_axis_combo.setEnabled(True)
        self.color_by_combo.setEnabled(self.chart_combo.currentText() == "Scatter Plot")
        self.series_list.setEnabled(True)
        
        self.placeholder.setVisible(False)
        self.canvas.setVisible(True)
        
        self.schedule_update()
        
    def on_series_selection_changed(self):
        """Handle changes in selected series."""
        self.selected_series = [item.text() for item in self.series_list.selectedItems()]
        self.schedule_update()
        
    def export_plot(self, format_type):
        """Export the plot to a file."""
        if self.data_manager.data is None:
            return
            
        # Get default filename based on chart title
        chart_title = self.title_edit.text() or "chart"  # Changed from currentText() to text()
        filename = f"{chart_title.replace(' ', '_')}.{format_type}"
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            f"Save {format_type.upper()} File", 
            filename,
            f"{format_type.upper()} Files (*.{format_type})"
        )
        
        if file_path:
            # Save the figure
            try:
                self.figure.savefig(file_path, format=format_type, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {file_path}")
            except Exception as e:
                print(f"Error saving plot: {str(e)}")
                
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
        color_by = self.color_by_combo.currentText()
        
        # Handle "None" selection
        if x_col == "None":
            x_col = None
        if y_col == "None":
            y_col = None
        if color_by == "None":
            color_by = None
            
        # Get selected series for multi-series plots
        selected_series = self.selected_series if self.selected_series else ([y_col] if y_col else [])
        
        # Skip if no data to plot
        if (not x_col and not y_col) or not selected_series:
            self.canvas.draw()
            return
            
        # Get alpha (transparency) value
        alpha = self.alpha_slider.value() / 100.0
        
        # Set the style
        plt.style.use(self.style_combo.currentText())
        
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        try:
            chart_type = self.chart_combo.currentText()
            
            if chart_type == "Line Chart":
                if not x_col:
                    # Use index if x is None
                    for i, series in enumerate(selected_series):
                        if self.color_combo.currentText() == "Custom" and self.custom_color:
                            ax.plot(df.index, df[series], marker='o', alpha=alpha, 
                                    label=series, color=self.custom_color)
                        else:
                            ax.plot(df.index, df[series], marker='o', alpha=alpha, label=series)
                else:
                    for i, series in enumerate(selected_series):
                        if self.color_combo.currentText() == "Custom" and self.custom_color:
                            ax.plot(df[x_col], df[series], marker='o', alpha=alpha, 
                                    label=series, color=self.custom_color)
                        else:
                            ax.plot(df[x_col], df[series], marker='o', alpha=alpha, label=series)
                        
            elif chart_type == "Bar Chart":
                # Handle multiple series for bar chart
                if not x_col:
                    # Use index if x is None
                    index = np.arange(len(df))
                    if len(selected_series) == 1:
                        if self.color_combo.currentText() == "Custom" and self.custom_color:
                            ax.bar(index, df[selected_series[0]], alpha=alpha, 
                                   label=selected_series[0], color=self.custom_color)
                        else:
                            ax.bar(index, df[selected_series[0]], alpha=alpha, label=selected_series[0])
                        ax.set_xticks(index)
                        ax.set_xticklabels(df.index)
                    else:
                        # Create grouped bar chart for multiple series
                        width = 0.8 / len(selected_series)
                        for i, series in enumerate(selected_series):
                            offset = i - len(selected_series)/2 + 0.5
                            ax.bar(index + offset*width, df[series], width, alpha=alpha, label=series)
                        ax.set_xticks(index)
                        ax.set_xticklabels(df.index)
                else:
                    if len(selected_series) == 1:
                        if self.color_combo.currentText() == "Custom" and self.custom_color:
                            ax.bar(df[x_col], df[selected_series[0]], alpha=alpha, 
                                   label=selected_series[0], color=self.custom_color)
                        else:
                            ax.bar(df[x_col], df[selected_series[0]], alpha=alpha, label=selected_series[0])
                    else:
                        # Create grouped bar chart for multiple series
                        x = np.arange(len(df[x_col]))
                        width = 0.8 / len(selected_series)
                        
                        for i, series in enumerate(selected_series):
                            offset = i - len(selected_series)/2 + 0.5
                            ax.bar(x + offset*width, df[series], width, alpha=alpha, label=series)
                        
                        ax.set_xticks(x)
                        ax.set_xticklabels(df[x_col])
                    
            elif chart_type == "Scatter Plot":
                marker_size = self.marker_size.value()
                
                if not x_col or not y_col:
                    # Skip if missing an axis for scatter plot
                    pass
                elif color_by:
                    # Use the color_by column for point colors
                    scatter = ax.scatter(
                        df[x_col], 
                        df[y_col], 
                        c=df[color_by] if pd.api.types.is_numeric_dtype(df[color_by]) else None,
                        alpha=alpha, 
                        s=marker_size*10,
                        cmap=self.color_map or 'viridis'
                    )
                    
                    # Add colorbar if using numeric column for colors
                    if pd.api.types.is_numeric_dtype(df[color_by]):
                        cbar = plt.colorbar(scatter, ax=ax)
                        cbar.set_label(color_by)
                    else:
                        # For categorical data, create a legend
                        categories = df[color_by].unique()
                        for category in categories:
                            mask = df[color_by] == category
                            ax.scatter(
                                df[x_col][mask], 
                                df[y_col][mask], 
                                alpha=alpha, 
                                s=marker_size*10,
                                label=category
                            )
                else:
                    # Regular scatter plot without color mapping
                    if self.color_combo.currentText() == "Custom" and self.custom_color:
                        ax.scatter(df[x_col], df[y_col], alpha=alpha, s=marker_size*10, 
                                  color=self.custom_color)
                    else:
                        ax.scatter(df[x_col], df[y_col], alpha=alpha, s=marker_size*10)
                        
            elif chart_type == "Histogram":
                bins = self.bins_spinbox.value()
                for series in selected_series:
                    if self.color_combo.currentText() == "Custom" and self.custom_color:
                        ax.hist(df[series], bins=bins, alpha=alpha, density=True, 
                               label=series, color=self.custom_color)
                    else:
                        ax.hist(df[series], bins=bins, alpha=alpha, density=True, label=series)
                        
                    if len(df[series].dropna()) > 1:  # Only add KDE if we have enough data points
                        density = stats.gaussian_kde(df[series].dropna())
                        xs = np.linspace(df[series].min(), df[series].max(), 200)
                        ax.plot(xs, density(xs), label=f'{series} KDE')
                ax.set_ylabel('Density')
                
            elif chart_type == "Box Plot":
                data = []
                labels = []
                for series in selected_series:
                    data.append(df[series].dropna())
                    labels.append(series)
                    
                if self.color_combo.currentText() == "Custom" and self.custom_color:
                    ax.boxplot(data, labels=labels, patch_artist=True, 
                              boxprops=dict(alpha=alpha, color=self.custom_color))
                else:
                    ax.boxplot(data, labels=labels, patch_artist=True, 
                              boxprops=dict(alpha=alpha))
                
            elif chart_type == "Violin Plot":
                # For violin plots, use seaborn to handle the data
                plot_data = pd.DataFrame()
                for series in selected_series:
                    temp_df = pd.DataFrame({
                        'value': df[series].dropna(),
                        'variable': series
                    })
                    plot_data = pd.concat([plot_data, temp_df])
                
                if self.color_map:
                    sns.violinplot(x='variable', y='value', data=plot_data, ax=ax, 
                                  alpha=alpha, palette=self.color_map)
                else:
                    sns.violinplot(x='variable', y='value', data=plot_data, ax=ax, alpha=alpha)
                
            elif chart_type == "Heatmap":
                if len(selected_series) > 1:
                    # Calculate correlation matrix for selected columns
                    corr = df[selected_series].corr()
                    sns.heatmap(
                        corr, 
                        annot=True, 
                        cmap=self.color_map or 'coolwarm', 
                        ax=ax, center=0, 
                        vmin=-1, vmax=1
                    )
                else:
                    # Single series, show correlation with all numeric columns
                    corr = df.corr()[selected_series[0]].sort_values(ascending=False)
                    # Convert to matrix for heatmap
                    corr_matrix = pd.DataFrame({selected_series[0]: corr})
                    sns.heatmap(
                        corr_matrix, 
                        annot=True, 
                        cmap=self.color_map or 'coolwarm', 
                        ax=ax, center=0, 
                        vmin=-1, vmax=1
                    )
                    
            elif chart_type == "KDE Plot":
                for series in selected_series:
                    if self.color_combo.currentText() == "Custom" and self.custom_color:
                        sns.kdeplot(
                            data=df, x=series, ax=ax, alpha=alpha, 
                            fill=True, label=series, color=self.custom_color
                        )
                    else:
                        sns.kdeplot(
                            data=df, x=series, ax=ax, alpha=alpha, 
                            fill=True, label=series
                        )
                    
            elif chart_type == "Pie Chart":
                # Pie charts work best with categorical x and numeric y
                if len(df) > 10:  # Limit to first 10 entries for readability
                    data = df.iloc[:10]
                    title_suffix = " (first 10 entries)"
                else:
                    data = df
                    title_suffix = ""
                
                if not x_col:
                    labels = data.index
                else:
                    labels = data[x_col]
                    
                # Use the selected series (y-value) for the pie chart values
                if len(selected_series) > 0:
                    values = data[selected_series[0]]
                    
                    if self.color_combo.currentText() == "Custom" and self.custom_color:
                        ax.pie(values, labels=labels, autopct='%1.1f%%', 
                               alpha=alpha, startangle=90, colors=[self.custom_color])
                    elif self.color_map:
                        # Get a colormap from matplotlib
                        cmap = plt.cm.get_cmap(self.color_map)
                        colors = [cmap(i/len(values)) for i in range(len(values))]
                        ax.pie(values, labels=labels, autopct='%1.1f%%', 
                               alpha=alpha, startangle=90, colors=colors)
                    else:
                        ax.pie(values, labels=labels, autopct='%1.1f%%', 
                               alpha=alpha, startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                
            elif chart_type == "Area Chart":
                if not x_col:
                    # Use index if x is None
                    for series in selected_series:
                        if self.color_combo.currentText() == "Custom" and self.custom_color:
                            ax.fill_between(
                                df.index, df[series], alpha=alpha, 
                                label=series, color=self.custom_color
                            )
                        else:
                            ax.fill_between(df.index, df[series], alpha=alpha, label=series)
                else:
                    for series in selected_series:
                        if self.color_combo.currentText() == "Custom" and self.custom_color:
                            ax.fill_between(
                                df[x_col], df[series], alpha=alpha, 
                                label=series, color=self.custom_color
                            )
                        else:
                            ax.fill_between(df[x_col], df[series], alpha=alpha, label=series)
                    
            # Set title
            title = self.title_edit.text()  # Changed from currentText() to text()
            if not title:
                if chart_type == "Pie Chart":
                    title = f"{chart_type}: {selected_series[0] if selected_series else ''}"
                elif not x_col:
                    title = f"{chart_type}: {', '.join(selected_series)}"
                else:
                    title = f"{chart_type}: {x_col} vs {', '.join(selected_series)}"
            
            # Add suffix for truncated data in pie charts
            if chart_type == "Pie Chart" and len(df) > 10:
                title += title_suffix
                
            ax.set_title(title)
            
            # Set labels
            if x_col and chart_type != "Pie Chart":
                ax.set_xlabel(x_col)
            elif chart_type != "Pie Chart" and chart_type not in ["Box Plot", "Violin Plot"]:
                ax.set_xlabel("Index")
            
            if chart_type not in ["Histogram", "KDE Plot", "Pie Chart"]:
                if len(selected_series) == 1:
                    ax.set_ylabel(selected_series[0])
                else:
                    ax.set_ylabel("Value")
            
            # Configure grid
            ax.grid(self.grid_check.isChecked())
            
            # Configure legend if applicable and checked
            if self.legend_check.isChecked():
                # Only show legend for plots that have labeled elements
                if chart_type not in ["Heatmap"]:
                    handles, labels = ax.get_legend_handles_labels()
                    if len(handles) > 0:  # Only show legend if we have labeled elements
                        ax.legend(loc='best')
                
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