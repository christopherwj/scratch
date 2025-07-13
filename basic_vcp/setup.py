"""
Setup script for VCP Detection System
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def check_data_directory():
    """Check if data directory exists and has files"""
    data_dir = Path("../data")
    
    if not data_dir.exists():
        print("✗ Data directory not found at ../data")
        print("Please ensure your stock data CSV files are in the ../data directory")
        return False
    
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("✗ No CSV files found in data directory")
        print("Please add stock data CSV files to the ../data directory")
        return False
    
    print(f"✓ Found {len(csv_files)} CSV files in data directory")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "results", "charts"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        print("✓ Core libraries imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        from data.stock_data import StockDataLoader
        from analysis.vcp_detector import VCPDetector
        from config import Config
        print("✓ Local modules imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

def run_quick_test():
    """Run a quick test to verify the system works"""
    print("Running quick test...")
    
    try:
        from data.stock_data import StockDataLoader
        
        # Test data loading
        data_loader = StockDataLoader()
        available_tickers = data_loader.get_available_tickers()
        
        if not available_tickers:
            print("✗ No tickers found - check data directory")
            return False
        
        # Test loading one stock
        test_ticker = available_tickers[0]
        df = data_loader.load_stock_data(test_ticker)
        
        if df is None:
            print(f"✗ Could not load data for {test_ticker}")
            return False
        
        print(f"✓ Successfully loaded data for {test_ticker} ({len(df)} data points)")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("VCP Detection System - Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check data directory
    if not check_data_directory():
        return False
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        return False
    
    # Run quick test
    if not run_quick_test():
        return False
    
    print("\n" + "=" * 40)
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python test_vcp_system.py' to test the system")
    print("2. Run 'python main.py' to scan all stocks")
    print("3. Check the README.md for more information")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 