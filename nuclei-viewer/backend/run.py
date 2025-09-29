#!/usr/bin/env python3
"""
Startup script for the Nuclei Viewer backend server.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

try:
    import uvicorn
    from app.core.config import get_config, init_config
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install the required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nuclei Viewer Backend Server")

    parser.add_argument(
        "--host", type=str, default=None, help="Host to bind to (default: from config)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to bind to (default: from config)"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (development mode)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Log level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    try:
        # Initialize configuration
        config_path = args.config
        if config_path is None:
            # Try to find config file
            possible_paths = [
                backend_dir / "config" / "default_config.yaml",
                backend_dir.parent / "config" / "default_config.yaml",
                Path("config/default_config.yaml"),
            ]

            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break

        if config_path and Path(config_path).exists():
            init_config(config_path)
            logger.info(f"Configuration loaded from: {config_path}")
        else:
            logger.warning("Configuration file not found, using defaults")
            init_config()

        # Get server configuration
        config = get_config()

        # Override with command line arguments
        host = args.host or config.server.host
        port = args.port or config.server.port
        debug = args.debug or config.server.debug
        reload = args.reload or debug

        logger.info(f"Starting Nuclei Viewer server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        logger.info(f"Auto-reload: {reload}")
        logger.info("Using full file paths for each channel")

        # Validate configured file paths exist
        def validate_file_path(path_str: str, name: str):
            if path_str:
                path_obj = Path(path_str)
                if not path_obj.exists():
                    logger.warning(f"{name} file does not exist: {path_obj}")
                else:
                    logger.info(f"{name} file is valid: {path_obj}")

        # Initialize appropriate data loader from factory
        from app.core.data_loader_factory import create_data_loader
        try:
            data_loader = create_data_loader(config)
            logger.info(f"Initialized data loader: {data_loader.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize data loader: {e}")
            sys.exit(1)

        # Start the server
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=args.log_level,
            access_log=True,
        )

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
