#!/usr/bin/env python3
"""
Setup and test script for the Nuclei Viewer application.

This script:
1. Generates test data
2. Sets up the backend environment
3. Starts the backend server
4. Provides instructions for frontend setup
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=check, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        raise


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error("Python 3.10 or higher is required")
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def setup_backend(project_root):
    """Set up the backend environment."""
    logger.info("Setting up backend environment...")
    
    backend_dir = project_root / "backend"
    
    # Check if virtual environment exists
    venv_dir = backend_dir / "venv"
    if not venv_dir.exists():
        logger.info("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", "venv"], cwd=backend_dir)
    
    # Determine pip executable
    if os.name == 'nt':  # Windows
        pip_exe = venv_dir / "Scripts" / "pip.exe"
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:  # Unix-like
        pip_exe = venv_dir / "bin" / "pip"
        python_exe = venv_dir / "bin" / "python"
    
    # Install requirements
    logger.info("Installing backend dependencies...")
    run_command([str(pip_exe), "install", "--upgrade", "pip"], cwd=backend_dir)
    run_command([str(pip_exe), "install", "-r", "requirements.txt"], cwd=backend_dir)
    
    return python_exe


def generate_test_data(project_root, data_dir):
    """Generate synthetic test data."""
    logger.info("Generating test data...")
    
    scripts_dir = project_root / "scripts"
    
    # Run the test data generator
    run_command([
        sys.executable, 
        "generate_test_data.py",
        "--output-dir", str(data_dir),
        "--shape", "15", "128", "128",
        "--num-nuclei", "30",
        "--epitope-channels", "488nm", "561nm", "640nm"
    ], cwd=scripts_dir)
    
    return data_dir


def update_config(project_root, data_dir):
    """Update the configuration file with test data path."""
    logger.info("Updating configuration...")
    
    config_file = project_root / "config" / "default_config.yaml"
    
    # Read current config
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    # Update base_path
    updated_config = config_content.replace(
        'base_path: ""',
        f'base_path: "{data_dir.absolute()}"'
    )
    
    # Update array_key to null for direct zarr arrays
    updated_config = updated_config.replace(
        'array_key: "labels"',
        'array_key: null'
    )
    updated_config = updated_config.replace(
        'array_key: "raw"',
        'array_key: null'
    )
    
    # Write updated config
    with open(config_file, 'w') as f:
        f.write(updated_config)
    
    logger.info(f"Updated config with data path: {data_dir.absolute()}")


def start_backend_server(project_root, python_exe):
    """Start the backend server."""
    logger.info("Starting backend server...")
    
    backend_dir = project_root / "backend"
    
    # Start server in background
    cmd = [str(python_exe), "run.py", "--debug"]
    
    logger.info("Starting backend server (this will run in the background)...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("Server will be available at: http://localhost:8000")
    logger.info("API documentation at: http://localhost:8000/docs")
    
    # Start the server
    process = subprocess.Popen(
        cmd,
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Wait a bit and check if server started
    time.sleep(3)
    
    if process.poll() is None:
        logger.info("Backend server started successfully!")
        return process
    else:
        logger.error("Failed to start backend server")
        output, _ = process.communicate()
        logger.error(f"Server output: {output}")
        return None


def setup_frontend(project_root):
    """Set up the frontend environment."""
    logger.info("Setting up frontend environment...")
    
    frontend_dir = project_root / "frontend"
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        logger.info("Installing frontend dependencies...")
        try:
            run_command(["npm", "install"], cwd=frontend_dir)
        except subprocess.CalledProcessError:
            logger.error("Failed to install frontend dependencies")
            logger.error("Please make sure Node.js and npm are installed")
            return False
    
    return True


def print_instructions(project_root, data_dir):
    """Print instructions for running the application."""
    print("\n" + "="*60)
    print("NUCLEI VIEWER SETUP COMPLETE!")
    print("="*60)
    
    print(f"\n📁 Test data generated in: {data_dir.absolute()}")
    print(f"📊 Dataset contains ~30 synthetic nuclei with 3 epitope channels")
    
    print(f"\n🔧 Backend server should be running at: http://localhost:8000")
    print(f"📖 API documentation available at: http://localhost:8000/docs")
    
    print(f"\n🌐 To start the frontend:")
    print(f"   cd {project_root / 'frontend'}")
    print(f"   npm start")
    print(f"   Then open: http://localhost:3000")
    
    print(f"\n🧪 Testing the application:")
    print(f"   1. Open http://localhost:3000 in your browser")
    print(f"   2. The data should load automatically")
    print(f"   3. Browse nuclei using pagination")
    print(f"   4. Click on nuclei to see detailed views")
    print(f"   5. Adjust channel colors and opacity")
    
    print(f"\n🔍 Troubleshooting:")
    print(f"   - Check backend logs for any errors")
    print(f"   - Verify data files exist in: {data_dir}")
    print(f"   - Check browser console for frontend errors")
    print(f"   - Ensure both servers are running on different ports")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Setup and test Nuclei Viewer")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="test_data",
        help="Directory for test data"
    )
    parser.add_argument(
        "--skip-backend",
        action="store_true",
        help="Skip backend setup"
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip frontend setup"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip test data generation"
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start the backend server"
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory: {data_dir}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    try:
        # Generate test data
        if not args.skip_data:
            generate_test_data(project_root, data_dir)
            update_config(project_root, data_dir)
        
        # Setup backend
        python_exe = None
        if not args.skip_backend:
            python_exe = setup_backend(project_root)
        
        # Setup frontend
        if not args.skip_frontend:
            setup_frontend(project_root)
        
        # Start backend server if requested
        server_process = None
        if args.start_server and python_exe:
            server_process = start_backend_server(project_root, python_exe)
            if not server_process:
                logger.error("Failed to start server")
                sys.exit(1)
        
        # Print instructions
        print_instructions(project_root, data_dir)
        
        # Keep server running if started
        if server_process:
            try:
                logger.info("Server is running. Press Ctrl+C to stop.")
                server_process.wait()
            except KeyboardInterrupt:
                logger.info("Stopping server...")
                server_process.terminate()
                server_process.wait()
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()