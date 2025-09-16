"""
Main FastAPI application for the nuclei viewer.
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import init_config, get_config
from .api import data, nuclei, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nuclei_viewer.log')
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Nuclei Viewer application...")
    
    try:
        # Initialize configuration
        config_path = Path("config/default_config.yaml")
        if not config_path.exists():
            # Try relative to backend directory
            config_path = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
        
        if config_path.exists():
            init_config(str(config_path))
            logger.info(f"Configuration loaded from: {config_path}")
        else:
            logger.warning("Configuration file not found, using defaults")
            init_config()
        
        config = get_config()
        logger.info("Application configured with full file paths for each channel")
        
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
    lifespan=lifespan
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
                "openapi": "/openapi.json"
            }
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        return {
            "message": "Nuclei Viewer API",
            "version": "1.0.0",
            "status": "running",
            "error": "Configuration not initialized"
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
        
        # Check segmentation file
        seg_path = Path(config.data.segmentation.file_path)
        file_checks["segmentation_file_exists"] = seg_path.exists()
        
        # Check DAPI file
        dapi_path = Path(config.data.dapi_channel.file_path)
        file_checks["dapi_file_exists"] = dapi_path.exists()
        
        # Check epitope channel files
        for epitope in config.data.epitope_channels:
            epitope_path = Path(epitope.file_path)
            file_checks[f"{epitope.name}_file_exists"] = epitope_path.exists()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": {
                "configuration": "ok",
                **file_checks
            }
        }
        
        # Determine overall status - if any critical files are missing, mark as degraded
        if not file_checks.get("segmentation_file_exists", False):
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        from datetime import datetime
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
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
            "path": str(request.url)
        }
    )


# Custom 404 handler
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "detail": f"The requested resource {request.url.path} was not found.",
            "available_endpoints": [
                "/api/data",
                "/api/nuclei",
                "/api/config",
                "/docs"
            ]
        }
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
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )