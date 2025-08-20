#!/bin/bash
# Installation script for VolAlign with Git dependencies
# This script provides different installation options for various use cases

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi
    
    if ! command_exists pip; then
        print_error "pip is required but not installed."
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version >= 3.10" | bc -l) -eq 0 ]]; then
        print_error "Python 3.10 or higher is required. Current version: $python_version"
        exit 1
    fi
    
    print_success "Prerequisites check passed (Python $python_version)"
}

# Install VolAlign in development mode
install_dev() {
    print_status "Installing VolAlign in development mode..."
    pip install -e .
    print_success "VolAlign installed in development mode"
}

# Install VolAlign in production mode
install_prod() {
    print_status "Installing VolAlign in production mode..."
    pip install .
    print_success "VolAlign installed in production mode"
}

# Install with virtual environment
install_with_venv() {
    local venv_name=${1:-"volalign_env"}
    
    print_status "Creating virtual environment: $venv_name"
    python3 -m venv "$venv_name"
    
    print_status "Activating virtual environment..."
    source "$venv_name/bin/activate"
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    install_dev
    
    print_success "Installation completed in virtual environment: $venv_name"
    print_status "To activate the environment, run: source $venv_name/bin/activate"
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Installation options:"
    echo "  dev                 Install in development mode (pip install -e .)"
    echo "  prod                Install in production mode (pip install .)"
    echo "  venv [name]         Install in a new virtual environment (default: volalign_env)"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev              # Development installation"
    echo "  $0 venv my_env      # Install in virtual environment named 'my_env'"
    echo ""
}

# Main installation logic
main() {
    local option=${1:-"help"}
    
    case $option in
        "dev")
            check_prerequisites
            install_dev
            ;;
        "prod")
            check_prerequisites
            install_prod
            ;;
        "venv")
            check_prerequisites
            install_with_venv "$2"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown option: $option"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"