# VolAlign Installation Guide

This guide provides comprehensive instructions for installing VolAlign with its streamlined dependency management system.

## Overview

VolAlign provides a streamlined installation process with all dependencies managed through standard Python packaging tools.


## Quick Start

### Automated Installation (Recommended)

```bash
git clone https://github.com/XpressAI/VolAlign
cd VolAlign
./install.sh dev
```

This will install VolAlign in development mode with all dependencies.

## Installation Methods

### Method 1: Installation Script

The `install.sh` script provides several installation options:

```bash
# Development installation (recommended for most users)
./install.sh dev

# Production installation
./install.sh prod

# Install in a new virtual environment
./install.sh venv [environment_name]

# Show help
./install.sh help
```

### Method 2: Direct pip Installation

```bash
git clone https://github.com/XpressAI/VolAlign
cd VolAlign

# Install all dependencies (including git-based ones)
pip install -e .
```

### Method 3: From requirements.txt

```bash
git clone https://github.com/XpressAI/VolAlign
cd VolAlign

# Install from requirements file
pip install -r requirements.txt

# Install VolAlign in development mode
pip install -e .
```

## Virtual Environment Installation

### Using the Installation Script

```bash
# Create and install in virtual environment
./install.sh venv my_volalign_env

# Activate the environment
source my_volalign_env/bin/activate
```

### Manual Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv volalign_env

# Activate environment
source volalign_env/bin/activate  # Linux/Mac
# or
volalign_env\Scripts\activate     # Windows

# Install VolAlign
pip install -e .
```


## Troubleshooting

### Common Issues

1. **Python version compatibility**
   ```bash
   # Check Python version
   python --version
   
   # VolAlign requires Python 3.10+
   ```

2. **Permission errors**
   ```bash
   # Use user installation
   pip install --user -e .
   ```

3. **Git dependency installation fails**
   ```bash
   # Check if git is installed
   git --version
   
   # Ensure you have internet access to GitHub
   ping github.com
   ```

### Verification

```python
# Test basic installation
import VolAlign
print(f"VolAlign version: {VolAlign.__version__}")
print("VolAlign installed successfully!")

# Test core modules
try:
    from VolAlign import utils, alignment_tools, distributed_processing
    from VolAlign import MicroscopyProcessingPipeline
    print("All core modules imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")

# Test Git dependencies (may not be available in all environments)
try:
    import exm  # ExSeq-Toolbox
    print("ExSeq-Toolbox available!")
except ImportError:
    print("ExSeq-Toolbox not available (expected in some environments)")

try:
    import cellpose
    print("Cellpose available!")
except ImportError:
    print("Cellpose not available (expected in some environments)")
```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your Python version (3.10+ required)
3. Ensure Git is installed and accessible
4. Check network connectivity for Git repositories
5. Open an issue on the VolAlign GitHub repository

## File Structure

After installation, your VolAlign directory should contain:

```
VolAlign/
├── requirements.txt          # All dependencies
├── install.sh               # Installation script
├── setup.py                 # Setup configuration
├── pyproject.toml           # Project metadata
└── VolAlign/                # Package source code
```