"""
Main entry point for the Data Analysis Application.
This module initializes the PyQt6 application and launches the main window.
"""

import sys
import os
import warnings
import platform

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QLibraryInfo
from ui.main_window import MainWindow

# Suppress macOS-specific Qt warnings
if platform.system() == 'Darwin':  # macOS
    warnings.filterwarnings("ignore", ".*overrides the method identifier.*")
    warnings.filterwarnings("ignore", ".*chose.*")

def main():
    """Initialize and run the application."""
    # Fix for platform plugin issue
    plugins_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins_path
    
    app = QApplication(sys.argv)
    
    # Set application-wide style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 