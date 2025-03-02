#!/usr/bin/env python3
"""
Clean PyCaches Script

This script removes all __pycache__ directories and .pyc/.pyo files from the project.
Run this script when you want to clean up compiled Python files.
"""

import os
import shutil
import fnmatch

def clean_pycache(directory='.'):
    """
    Remove __pycache__ directories and compiled Python files recursively.
    
    Args:
        directory (str): The directory to start searching from (default: current directory)
    """
    # Counter for removed items
    pycache_dirs_removed = 0
    pyc_files_removed = 0
    
    print(f"Cleaning Python cache files from '{os.path.abspath(directory)}'...")
    
    # Walk through all directories
    for root, dirs, files in os.walk(directory):
        # Skip the virtual environment directory
        if '.venv' in root or 'venv' in root:
            continue
            
        # Remove __pycache__ directories
        for dir_name in dirs:
            if dir_name == '__pycache__':
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    pycache_dirs_removed += 1
                    print(f"Removed: {dir_path}")
                except Exception as e:
                    print(f"Error removing {dir_path}: {e}")
        
        # Remove .pyc, .pyo, and .pyd files
        for pattern in ['*.pyc', '*.pyo', '*.pyd']:
            for filename in fnmatch.filter(files, pattern):
                file_path = os.path.join(root, filename)
                try:
                    os.unlink(file_path)
                    pyc_files_removed += 1
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    
    # Print summary
    print("\nCleanup complete!")
    print(f"Removed {pycache_dirs_removed} __pycache__ directories")
    print(f"Removed {pyc_files_removed} compiled Python files")

if __name__ == '__main__':
    clean_pycache() 