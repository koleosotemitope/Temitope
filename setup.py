"""
Quick Start Script for EUR/JPY LSTM Predictor
This script helps you get started with the application
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"✗ Python 3.8+ required. You have Python {version.major}.{version.minor}")
        return False

def create_venv():
    """Create virtual environment"""
    print_header("Creating Virtual Environment")
    venv_path = "venv"
    
    if os.path.exists(venv_path):
        print(f"✓ Virtual environment already exists at {venv_path}")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print(f"✓ Virtual environment created at {venv_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False

def get_activate_command():
    """Get the appropriate activation command"""
    venv_path = "venv"
    
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "activate.bat")
    else:
        return f"source {os.path.join(venv_path, 'bin', 'activate')}"

def install_requirements():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    requirements_file = "requirements_new.txt"
    
    if not os.path.exists(requirements_file):
        print(f"✗ {requirements_file} not found")
        return False
    
    try:
        # Use the Python executable from venv
        if platform.system() == "Windows":
            python_exe = os.path.join("venv", "Scripts", "python.exe")
        else:
            python_exe = os.path.join("venv", "bin", "python")
        
        subprocess.run([python_exe, "-m", "pip", "install", "-r", requirements_file], check=True)
        print(f"✓ Dependencies installed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def train_model():
    """Train the LSTM model"""
    print_header("Training LSTM Model")
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    if os.path.exists("models/lstm_eurjpy.h5"):
        print("✓ Trained model already exists")
        response = input("Do you want to retrain? (y/n): ").lower()
        if response != 'y':
            return True
    
    try:
        if platform.system() == "Windows":
            python_exe = os.path.join("venv", "Scripts", "python.exe")
        else:
            python_exe = os.path.join("venv", "bin", "python")
        
        print("Starting model training (this may take 5-10 minutes)...")
        subprocess.run([python_exe, "lstm_model.py"], check=True)
        print("✓ Model training completed")
        return True
    except Exception as e:
        print(f"✗ Failed to train model: {e}")
        return False

def start_application():
    """Start the Flask application"""
    print_header("Starting Web Application")
    
    print("\nThe application will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        if platform.system() == "Windows":
            python_exe = os.path.join("venv", "Scripts", "python.exe")
        else:
            python_exe = os.path.join("venv", "bin", "python")
        
        subprocess.run([python_exe, "app.py"])
    except KeyboardInterrupt:
        print("\n✓ Application stopped")
    except Exception as e:
        print(f"✗ Failed to start application: {e}")

def main():
    """Main setup flow"""
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║     EUR/JPY LSTM Price Predictor - Quick Start         ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create virtual environment
    if not create_venv():
        return
    
    print("\nTo activate the virtual environment, run:")
    print(f"  {get_activate_command()}")
    
    # Install dependencies
    if not install_requirements():
        return
    
    # Train model
    if not train_model():
        print("\n⚠ Warning: Model training failed. Please run 'python lstm_model.py' manually")
    
    # Ask to start application
    print_header("Ready to Start")
    response = input("Start the web application now? (y/n): ").lower()
    if response == 'y':
        start_application()
    else:
        print("\nTo start the application later, run:")
        print("  python app.py")

if __name__ == "__main__":
    main()
