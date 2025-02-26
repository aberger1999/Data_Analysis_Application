"""
Main window for the Data Analysis Application.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QSplitter,
    QTabWidget
)
from PyQt6.QtCore import Qt
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
        
        # Create toolbar
        toolbar = QHBoxLayout()
        
        # Import button
        self.import_btn = QPushButton("Import CSV")
        self.import_btn.clicked.connect(self.import_csv)
        toolbar.addWidget(self.import_btn)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
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
            
    def show_error(self, message):
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message) 