"""
Main FastAPI application for the nuclei viewer.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import config, data, nuclei
from .core.config import get_config, init_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("nuclei_viewer.log"),
    ],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Nuclei Viewer application...")

    try:
        # Check if configuration is already initialized (e.g., by run.py)
        try:
            config = get_config()
            logger.info("Using existing configuration")
        except RuntimeError:
            # Configuration not initialized, load it
            import os
            config_path = os.environ.get("NUCLEI_VIEWER_CONFIG")
            
            if not config_path:
                # Try default locations
                possible_paths = [
                    Path("config/default_config.yaml"),
                    Path(__file__).parent.parent.parent / "config" / "default_config.yaml",
                    Path(__file__).parent.parent.parent / "config" / "pipeline_config_example.yaml"
                ]
                
                for path in possible_paths:
                    if path.exists():
                        config_path = str(path)
                        break

            if config_path and Path(config_path).exists():
                init_config(str(config_path))
                logger.info(f"Configuration loaded from: {config_path}")
            else:
                logger.warning("Configuration file not found, using defaults")
                init_config()

            config = get_config()

        # Log configuration mode
        if hasattr(config.data, 'pipeline') and config.data.pipeline:
            logger.info("Application configured in pipeline mode")
        else:
            logger.info("Application configured with manual file paths")

        # Initialize shared state early to ensure dependencies work
        try:
            from .core.shared_state import get_shared_data_loader, get_shared_nuclei_processor
            
            # Initialize data loader
            data_loader = get_shared_data_loader()
            logger.info(f"Shared data loader initialized: {type(data_loader).__name__}")
            
            # Initialize nuclei processor
            nuclei_processor = get_shared_nuclei_processor()
            logger.info(f"Shared nuclei processor initialized: {type(nuclei_processor).__name__}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize shared state during startup: {e}")
            # Don't fail startup, but log the warning

    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Nuclei Viewer application...")


# Create FastAPI application
app = FastAPI(
    title="Nuclei Viewer API",
    description="API for visualizing segmented nuclei and epitope tags",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS - must be added before including routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

logger.info("CORS configured for frontend development")


# Include API routers
app.include_router(data.router)
app.include_router(nuclei.router)
app.include_router(config.router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    try:
        config = get_config()
        return {
            "message": "Nuclei Viewer API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "data": "/api/data",
                "nuclei": "/api/nuclei",
                "config": "/api/config",
                "docs": "/docs",
                "openapi": "/openapi.json",
            },
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        return {
            "message": "Nuclei Viewer API",
            "version": "1.0.0",
            "status": "running",
            "error": "Configuration not initialized",
        }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        from datetime import datetime

        config = get_config()

        # Check if configured file paths exist
        file_checks = {}

        try:
            # Handle different configuration modes
            if hasattr(config.data, 'pipeline') and config.data.pipeline:
                # Pipeline mode - check pipeline-specific paths
                working_dir = Path(config.data.pipeline.pipeline_working_directory)
                file_checks["working_directory_exists"] = working_dir.exists()
                
                if config.data.pipeline.epitope_analysis_file:
                    epitope_file = working_dir / config.data.pipeline.epitope_analysis_file
                    file_checks["epitope_analysis_exists"] = epitope_file.exists()
                
                # Check for segmentation directory
                seg_dir = working_dir / "segmentation"
                file_checks["segmentation_dir_exists"] = seg_dir.exists()
                
                # Check for zarr volumes directory
                zarr_dir = working_dir / "zarr_volumes"
                file_checks["zarr_volumes_dir_exists"] = zarr_dir.exists()
                
            else:
                # Manual mode - check individual file paths
                if hasattr(config.data, 'segmentation') and config.data.segmentation:
                    seg_path = Path(config.data.segmentation.file_path)
                    file_checks["segmentation_file_exists"] = seg_path.exists()

                if hasattr(config.data, 'dapi_channel') and config.data.dapi_channel:
                    dapi_path = Path(config.data.dapi_channel.file_path)
                    file_checks["dapi_file_exists"] = dapi_path.exists()

                if hasattr(config.data, 'epitope_channels') and config.data.epitope_channels:
                    for epitope in config.data.epitope_channels:
                        epitope_path = Path(epitope.file_path)
                        file_checks[f"{epitope.name}_file_exists"] = epitope_path.exists()
        
        except Exception as e:
            logger.warning(f"Error checking file paths in health check: {e}")
            file_checks["file_check_error"] = str(e)

        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": {"configuration": "ok", **file_checks},
        }

        # Determine overall status - if any critical files are missing, mark as degraded
        # In pipeline mode, check for segmentation directory instead of file
        if hasattr(config.data, 'pipeline') and config.data.pipeline:
            if not file_checks.get("segmentation_dir_exists", False):
                health_status["status"] = "degraded"
        else:
            if not file_checks.get("segmentation_file_exists", False):
                health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        from datetime import datetime

        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception in {request.url}: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please check the server logs.",
            "path": str(request.url),
        },
    )


# Custom 404 handler
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    # Get available routes from the app
    available_routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            available_routes.append(route.path)
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "detail": f"The requested resource {request.url.path} was not found.",
            "available_endpoints": ["/api/data", "/api/nuclei", "/api/config", "/docs"],
            "registered_routes": available_routes[:20],  # Limit to first 20 routes
        },
    )


if __name__ == "__main__":
    import uvicorn

    # Load configuration for server settings
    try:
        config = get_config()
        host = config.server.host
        port = config.server.port
        debug = config.server.debug
    except:
        # Fallback to defaults
        host = "127.0.0.1"
        port = 8000
        debug = True

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=debug, log_level="info")
