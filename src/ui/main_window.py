"""
Main window for the Data Analysis Application.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QSplitter,
    QTabWidget, QLabel, QFrame
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
import os
from .components.data_preview import DataPreviewPanel
from .components.analysis_panel import AnalysisPanel
from .components.visualization_panel import VisualizationPanel
from .components.preprocessing_panel import PreprocessingPanel
from .components.feature_engineering_panel import FeatureEngineeringPanel
from .components.machine_learning_panel import MachineLearningPanel
from .components.report_generator_panel import ReportGeneratorPanel
from .data_manager import DataManager

class MainWindow(QMainWindow):
    """Main window of the application."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.data_manager = DataManager()
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Data Analysis Application")
        self.setMinimumSize(1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add application title
        title_layout = QHBoxLayout()
        title_label = QLabel("Data Analysis Application")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 10px;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # Create toolbar with a frame for better visual separation
        toolbar_frame = QFrame()
        toolbar_frame.setFrameShape(QFrame.Shape.StyledPanel)
        toolbar_frame.setFrameShadow(QFrame.Shadow.Raised)
        toolbar_layout = QVBoxLayout(toolbar_frame)
        
        toolbar = QHBoxLayout()
        
        # Import button with icon
        self.import_btn = QPushButton("Import CSV")
        self.import_btn.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogOpenButton))
        self.import_btn.setIconSize(QSize(16, 16))
        self.import_btn.clicked.connect(self.import_csv)
        toolbar.addWidget(self.import_btn)
        
        # Export button with icon
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton))
        self.export_btn.setIconSize(QSize(16, 16))
        self.export_btn.clicked.connect(self.export_csv)
        self.export_btn.setEnabled(False)  # Disable until data is loaded
        toolbar.addWidget(self.export_btn)
        
        toolbar.addStretch()
        toolbar_layout.addLayout(toolbar)
        layout.addWidget(toolbar_frame)
        
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Add data preview to left side
        self.data_preview = DataPreviewPanel(self.data_manager)
        splitter.addWidget(self.data_preview)
        
        # Create tab widget for right side
        self.tabs = QTabWidget()
        
        # Data Preprocessing tab
        self.preprocessing_panel = PreprocessingPanel(self.data_manager)
        self.tabs.addTab(self.preprocessing_panel, "Data Preprocessing")
        
        # Analysis tab
        self.analysis_panel = AnalysisPanel(self.data_manager)
        self.tabs.addTab(self.analysis_panel, "Analysis")
        
        # Visualization tab
        self.visualization_panel = VisualizationPanel(self.data_manager)
        self.tabs.addTab(self.visualization_panel, "Visualization")
        
        # Feature Engineering tab
        self.feature_engineering_panel = FeatureEngineeringPanel(self.data_manager)
        self.tabs.addTab(self.feature_engineering_panel, "Feature Engineering")
        
        # Machine Learning tab
        self.machine_learning_panel = MachineLearningPanel(self.data_manager)
        self.tabs.addTab(self.machine_learning_panel, "Machine Learning")
        
        # Report Generator tab
        self.report_generator_panel = ReportGeneratorPanel(self.data_manager)
        self.tabs.addTab(self.report_generator_panel, "Report Generator")
        
        splitter.addWidget(self.tabs)
        
        # Set initial splitter sizes (40% left, 60% right)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
    def setup_connections(self):
        """Setup signal connections."""
        self.data_manager.data_error.connect(self.show_error)
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        
    def on_data_loaded(self, df):
        """Handle when data is loaded."""
        self.export_btn.setEnabled(True)  # Enable export button when data is loaded
        
    def import_csv(self):
        """Handle CSV file import."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.data_manager.load_csv(file_path)
            
    def export_csv(self):
        """Handle CSV file export."""
        if self.data_manager.data is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset before exporting.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.data_manager.data.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Data exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting data: {str(e)}")
            
    def show_error(self, message):
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message) 