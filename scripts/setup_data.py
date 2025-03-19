"""
TeleChurn Predictor - Data Setup Script

This script copies the raw telecom data files to the project's data/raw directory
and downloads additional data if needed.

Usage:
    python setup_data.py
"""

import os
import shutil
import sys
from pathlib import Path

# Try to import kagglehub for alternative data source
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("kagglehub not installed. To install: pip install kagglehub")

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DATA_DIR = Path("C:/Users/alex5/Documents/Projects/telecom_churn/Raw Data")

# Create raw data directory if it doesn't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def copy_local_data():
    """Copy data from external directory to project raw data directory."""
    source_files = {
        "cell2celltrain.csv": EXTERNAL_DATA_DIR / "cell2celltrain.csv",
        "cell2cellholdout.csv": EXTERNAL_DATA_DIR / "cell2cellholdout.csv"
    }
    
    success = True
    for filename, source_path in source_files.items():
        dest_path = RAW_DATA_DIR / filename
        
        if source_path.exists():
            try:
                shutil.copy2(source_path, dest_path)
                print(f"Copied {filename} to {dest_path}")
            except Exception as e:
                print(f"Error copying {filename}: {str(e)}")
                success = False
        else:
            print(f"Source file not found: {source_path}")
            success = False
    
    return success

def download_kaggle_data():
    """Download telecom churn data from Kaggle."""
    if not KAGGLE_AVAILABLE:
        print("kagglehub not available. Skipping Kaggle download.")
        return False
    
    try:
        # Download dataset
        path = kagglehub.dataset_download("jpacse/datasets-for-churn-telecom")
        print(f"Downloaded dataset to: {path}")
        
        # Copy files to raw data directory
        for file_path in Path(path).glob("*.csv"):
            dest_path = RAW_DATA_DIR / file_path.name
            shutil.copy2(file_path, dest_path)
            print(f"Copied {file_path.name} to {dest_path}")
        
        return True
    except Exception as e:
        print(f"Error downloading from Kaggle: {str(e)}")
        return False

def main():
    """Main function to set up data."""
    print("Setting up data for TeleChurn Predictor...")
    
    # First try to copy local data
    local_success = copy_local_data()
    
    # If local copy failed and Kaggle is available, try downloading from Kaggle
    if not local_success and KAGGLE_AVAILABLE:
        print("Local data not found or copy failed. Trying Kaggle download...")
        kaggle_success = download_kaggle_data()
        
        if not kaggle_success:
            print("Failed to set up data from both local and Kaggle sources.")
            return 1
    
    # Check if data files exist in raw data directory
    required_files = ["cell2celltrain.csv", "cell2cellholdout.csv"]
    missing_files = [f for f in required_files if not (RAW_DATA_DIR / f).exists()]
    
    if missing_files:
        print(f"Warning: The following required files are missing: {', '.join(missing_files)}")
        print("You may need to manually copy these files to the raw data directory.")
        return 1
    
    print("\nData setup complete!")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print("The following files are available:")
    for file_path in RAW_DATA_DIR.glob("*.csv"):
        file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"  - {file_path.name} ({file_size:.2f} MB)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())