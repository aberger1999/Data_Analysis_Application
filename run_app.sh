#!/bin/bash

# Get the absolute path to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment
source $SCRIPT_DIR/.venv/bin/activate

# Set the Qt platform plugin path explicitly
export QT_PLUGIN_PATH=$SCRIPT_DIR/.venv/lib/python3.13/site-packages/PyQt6/Qt6/plugins
export QT_QPA_PLATFORM_PLUGIN_PATH=$SCRIPT_DIR/.venv/lib/python3.13/site-packages/PyQt6/Qt6/plugins/platforms

# Run the application
python $SCRIPT_DIR/src/main.py

# Deactivate the virtual environment
deactivate 