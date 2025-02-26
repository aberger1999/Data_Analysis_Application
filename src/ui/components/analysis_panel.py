"""
Analysis panel for displaying statistical information about the data.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel,
    QFrame, QGridLayout, QComboBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
import pandas as pd
import numpy as np
from functools import partial

class AnalysisPanel(QWidget):
    """Panel for displaying statistical analysis of the data."""
    
    def __init__(self, data_manager):
        """Initialize the analysis panel."""
        super().__init__()
        self.data_manager = data_manager
        self.analysis_timer = QTimer()
        self.analysis_timer.setSingleShot(True)
        self.analysis_timer.timeout.connect(self._delayed_analysis)
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Controls section
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_layout = QGridLayout(controls_frame)
        
        # Analysis type selector
        analysis_label = QLabel("Analysis Type:")
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems([
            "Basic Statistics",
            "Correlation Analysis",
            "Distribution Analysis",
            "Outlier Detection"
        ])
        controls_layout.addWidget(analysis_label, 0, 0)
        controls_layout.addWidget(self.analysis_combo, 0, 1)
        
        # Column selector
        column_label = QLabel("Select Column:")
        self.column_combo = QComboBox()
        self.column_combo.setEnabled(False)
        controls_layout.addWidget(column_label, 1, 0)
        controls_layout.addWidget(self.column_combo, 1, 1)
        
        # Run analysis button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.schedule_analysis)
        controls_layout.addWidget(self.run_button, 2, 0, 1, 2)
        
        layout.addWidget(controls_frame)
        
        # Results section
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(QFrame.Shape.NoFrame)
        
        results_widget = QWidget()
        self.results_layout = QVBoxLayout(results_widget)
        
        # Results table with optimized settings
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setVisible(False)
        
        # Optimize table performance
        self.results_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.results_table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.results_table.setCornerButtonEnabled(False)
        
        self.results_layout.addWidget(self.results_table)
        
        # Error message for invalid operations
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red; font-weight: bold;")
        self.error_label.setVisible(False)
        self.results_layout.addWidget(self.error_label)
        
        # Placeholder message
        self.placeholder = QLabel("Import data to begin analysis")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #666; font-style: italic; padding: 20px;")
        self.results_layout.addWidget(self.placeholder)
        
        scroll_area.setWidget(results_widget)
        layout.addWidget(scroll_area)
        layout.setStretchFactor(scroll_area, 1)
        
    def setup_connections(self):
        """Setup signal connections."""
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        self.column_combo.currentTextChanged.connect(self.on_column_selected)
        self.analysis_combo.currentTextChanged.connect(self.on_analysis_type_changed)
    
    def schedule_analysis(self):
        """Schedule the analysis to run after a short delay to prevent UI freezing."""
        self.run_button.setEnabled(False)
        self.analysis_timer.start(100)  # 100ms delay
        
    def _delayed_analysis(self):
        """Run the analysis after a short delay."""
        try:
            self.run_analysis()
        finally:
            self.run_button.setEnabled(True)
            
    def update_table_batch(self, data_items):
        """Update table in batches for better performance."""
        self.results_table.setUpdatesEnabled(False)
        try:
            self.results_table.setRowCount(len(data_items))
            
            # Process items in batches
            BATCH_SIZE = 50
            for start in range(0, len(data_items), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(data_items))
                batch = data_items[start:end]
                
                for i, (metric, value) in enumerate(batch, start=start):
                    metric_item = QTableWidgetItem(str(metric))
                    value_item = QTableWidgetItem(str(value))
                    self.results_table.setItem(i, 0, metric_item)
                    self.results_table.setItem(i, 1, value_item)
                    
        finally:
            self.results_table.setUpdatesEnabled(True)
            
    def run_analysis(self):
        """Run the selected analysis on the selected column."""
        analysis_type = self.analysis_combo.currentText()
        column_name = self.column_combo.currentText()
        
        if not column_name:
            return
            
        # Clear previous results
        self.results_table.setRowCount(0)
        self.results_table.setVisible(True)
        self.placeholder.setVisible(False)
        self.error_label.setVisible(False)
        
        try:
            if analysis_type == "Basic Statistics":
                stats = self.data_manager.get_basic_stats(column_name)
                if stats:
                    self.display_basic_stats(stats)
            
            elif analysis_type == "Correlation Analysis":
                correlations = self.data_manager.get_correlation_analysis(column_name)
                if correlations:
                    if 'error' in correlations:
                        self.show_error(correlations['error'])
                    else:
                        self.display_correlation_analysis(correlations)
            
            elif analysis_type == "Distribution Analysis":
                distribution = self.data_manager.get_distribution_analysis(column_name)
                if distribution:
                    if 'error' in distribution:
                        self.show_error(distribution['error'])
                    else:
                        self.display_distribution_analysis(distribution)
            
            elif analysis_type == "Outlier Detection":
                outliers = self.data_manager.get_outlier_detection(column_name)
                if outliers:
                    if 'error' in outliers:
                        self.show_error(outliers['error'])
                    else:
                        self.display_outlier_detection(outliers)
                        
        except Exception as e:
            self.show_error(str(e))
            
    def display_basic_stats(self, stats):
        """Display basic statistics in the results table."""
        data_items = [
            (metric.replace('_', ' ').title(), 
             f"{value:.4f}" if isinstance(value, float) else str(value))
            for metric, value in stats.items()
        ]
        self.update_table_batch(data_items)
        
    def display_correlation_analysis(self, correlations):
        """Display correlation analysis results."""
        self.results_table.clear()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(['Column', 'Correlation Coefficient'])
        
        data_items = [
            (column, f"{corr:.4f}")
            for column, corr in correlations.items()
        ]
        self.update_table_batch(data_items)
        
    def display_distribution_analysis(self, distribution):
        """Display distribution analysis results."""
        data_items = []
        
        # Basic distribution metrics
        metrics = {
            'Skewness': distribution['skewness'],
            'Kurtosis': distribution['kurtosis'],
            'Range': distribution['range'],
            'IQR': distribution['iqr']
        }
        
        for metric, value in metrics.items():
            data_items.append((metric, f"{value:.4f}" if isinstance(value, float) else str(value)))
            
        # Add normality test results if available
        if distribution.get('normality_p_value') is not None:
            data_items.append(('Normality p-value', f"{distribution['normality_p_value']:.4f}"))
            data_items.append(('Is Normal Distribution', 'Yes' if distribution['is_normal'] else 'No'))
            
        # Add percentiles
        if 'percentiles' in distribution:
            for percentile, value in distribution['percentiles'].items():
                data_items.append((f"{percentile}th Percentile", f"{value:.4f}"))
                
        self.update_table_batch(data_items)
        
    def display_outlier_detection(self, outliers):
        """Display outlier detection results."""
        data_items = [
            ('Total Outliers', str(outliers['count'])),
            ('Percentage', f"{outliers['percentage']:.2f}%"),
            ('Lower Bound', f"{outliers['lower_bound']:.4f}"),
            ('Upper Bound', f"{outliers['upper_bound']:.4f}")
        ]
        self.update_table_batch(data_items)
        
    def show_error(self, message):
        """Display error message."""
        self.error_label.setText(f"Error: {message}")
        self.error_label.setVisible(True)
        self.results_table.setVisible(False)

    def on_data_loaded(self, df):
        """Handle when new data is loaded."""
        self.column_combo.clear()
        self.column_combo.addItems(self.data_manager.columns)
        self.column_combo.setEnabled(True)
        self.run_button.setEnabled(True)
        self.placeholder.setText("Select a column and analysis type to begin")
        
    def on_column_selected(self, column_name):
        """Handle when a column is selected."""
        if column_name:
            self.run_button.setEnabled(True)
    
    def on_analysis_type_changed(self, analysis_type):
        """Handle when analysis type changes."""
        if analysis_type:
            self.run_button.setEnabled(True)
        
    def get_correlation_description(self, corr):
        """Get a human-readable description of correlation strength."""
        abs_corr = abs(corr)
        direction = "positive" if corr > 0 else "negative"
        
        if abs_corr > 0.9:
            strength = "Very strong"
        elif abs_corr > 0.7:
            strength = "Strong"
        elif abs_corr > 0.5:
            strength = "Moderate"
        elif abs_corr > 0.3:
            strength = "Weak"
        else:
            strength = "Very weak or no"
            
        return f"{strength} {direction} correlation (r = {corr:.4f})" 