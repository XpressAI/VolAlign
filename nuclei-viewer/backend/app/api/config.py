"""
Configuration management API endpoints.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..core.config import (
    AppConfig,
    EpitopeChannelConfig,
    get_config,
    save_config,
    update_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigResponse(BaseModel):
    """Response model for configuration."""

    data: Dict[str, Any]
    processing: Dict[str, Any]
    server: Dict[str, Any]
    ui: Dict[str, Any]


class UpdateConfigRequest(BaseModel):
    """Request model for updating configuration."""

    data: Optional[Dict[str, Any]] = None
    processing: Optional[Dict[str, Any]] = None
    server: Optional[Dict[str, Any]] = None
    ui: Optional[Dict[str, Any]] = None


class EpitopeChannelRequest(BaseModel):
    """Request model for epitope channel configuration."""

    name: str
    file_path: str
    array_key: Optional[str] = None
    default_color: str = "#ffffff"


@router.get("/", response_model=ConfigResponse)
async def get_configuration():
    """
    Get current configuration.
    """
    try:
        config = get_config()

        return ConfigResponse(
            data=config.data.dict(),
            processing=config.processing.dict(),
            server=config.server.dict(),
            ui=config.ui.dict(),
        )

    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get configuration: {str(e)}"
        )


@router.put("/", response_model=ConfigResponse)
async def update_configuration(request: UpdateConfigRequest):
    """
    Update configuration with new values.
    """
    try:
        # Prepare update dict
        updates = {}
        if request.data is not None:
            updates["data"] = request.data
        if request.processing is not None:
            updates["processing"] = request.processing
        if request.server is not None:
            updates["server"] = request.server
        if request.ui is not None:
            updates["ui"] = request.ui

        # Update configuration
        updated_config = update_config(**updates)

        return ConfigResponse(
            data=updated_config.data.dict(),
            processing=updated_config.processing.dict(),
            server=updated_config.server.dict(),
            ui=updated_config.ui.dict(),
        )

    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update configuration: {str(e)}"
        )


@router.post("/save")
async def save_configuration(
    config_path: Optional[str] = Query(
        None, description="Path to save configuration file"
    )
):
    """
    Save current configuration to file.
    """
    try:
        config = get_config()

        if config_path is None:
            config_path = "config/current_config.yaml"

        save_config(config, config_path)

        return {
            "status": "success",
            "message": f"Configuration saved to {config_path}",
            "config_path": config_path,
        }

    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save configuration: {str(e)}"
        )


@router.get("/data-paths")
async def get_data_paths():
    """
    Get current data path configuration.
    """
    try:
        config = get_config()

        return {
            "segmentation": {
                "file_path": config.data.segmentation.file_path,
                "array_key": config.data.segmentation.array_key,
            },
            "dapi_channel": {
                "file_path": config.data.dapi_channel.file_path,
                "array_key": config.data.dapi_channel.array_key,
            },
            "epitope_channels": [
                {
                    "name": ch.name,
                    "file_path": ch.file_path,
                    "array_key": ch.array_key,
                    "default_color": ch.default_color,
                }
                for ch in config.data.epitope_channels
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get data paths: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get data paths: {str(e)}"
        )


@router.put("/data-paths/base-path")
async def update_base_path(base_path: str):
    """
    Update the base data path.
    """
    try:
        # Validate path exists
        path_obj = Path(base_path)
        if not path_obj.exists():
            raise HTTPException(
                status_code=400, detail=f"Path does not exist: {base_path}"
            )

        if not path_obj.is_dir():
            raise HTTPException(
                status_code=400, detail=f"Path is not a directory: {base_path}"
            )

        # Update configuration
        update_config(data={"base_path": str(path_obj.absolute())})

        return {
            "status": "success",
            "message": f"Base path updated to {base_path}",
            "base_path": str(path_obj.absolute()),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update base path: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update base path: {str(e)}"
        )


@router.post("/epitope-channels")
async def add_epitope_channel(request: EpitopeChannelRequest):
    """
    Add a new epitope channel configuration.
    """
    try:
        config = get_config()

        # Check if channel name already exists
        existing_names = [ch.name for ch in config.data.epitope_channels]
        if request.name in existing_names:
            raise HTTPException(
                status_code=400,
                detail=f"Epitope channel '{request.name}' already exists",
            )

        # Create new epitope channel config
        new_channel = EpitopeChannelConfig(
            name=request.name,
            file_path=request.file_path,
            array_key=request.array_key,
            default_color=request.default_color,
        )

        # Update configuration
        epitope_channels = config.data.epitope_channels + [new_channel]
        update_config(data={"epitope_channels": [ch.dict() for ch in epitope_channels]})

        return {
            "status": "success",
            "message": f"Added epitope channel: {request.name}",
            "channel": new_channel.dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add epitope channel: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to add epitope channel: {str(e)}"
        )


@router.put("/epitope-channels/{channel_name}")
async def update_epitope_channel(channel_name: str, request: EpitopeChannelRequest):
    """
    Update an existing epitope channel configuration.
    """
    try:
        config = get_config()

        # Find the channel to update
        epitope_channels = list(config.data.epitope_channels)
        channel_index = None
        for i, ch in enumerate(epitope_channels):
            if ch.name == channel_name:
                channel_index = i
                break

        if channel_index is None:
            raise HTTPException(
                status_code=404, detail=f"Epitope channel '{channel_name}' not found"
            )

        # Update the channel
        epitope_channels[channel_index] = EpitopeChannelConfig(
            name=request.name,
            file_path=request.file_path,
            array_key=request.array_key,
            default_color=request.default_color,
        )

        # Update configuration
        update_config(data={"epitope_channels": [ch.dict() for ch in epitope_channels]})

        return {
            "status": "success",
            "message": f"Updated epitope channel: {channel_name}",
            "channel": epitope_channels[channel_index].dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update epitope channel: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update epitope channel: {str(e)}"
        )


@router.delete("/epitope-channels/{channel_name}")
async def remove_epitope_channel(channel_name: str):
    """
    Remove an epitope channel configuration.
    """
    try:
        config = get_config()

        # Filter out the channel to remove
        epitope_channels = [
            ch for ch in config.data.epitope_channels if ch.name != channel_name
        ]

        if len(epitope_channels) == len(config.data.epitope_channels):
            raise HTTPException(
                status_code=404, detail=f"Epitope channel '{channel_name}' not found"
            )

        # Update configuration
        update_config(data={"epitope_channels": [ch.dict() for ch in epitope_channels]})

        return {
            "status": "success",
            "message": f"Removed epitope channel: {channel_name}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove epitope channel: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to remove epitope channel: {str(e)}"
        )


@router.get("/processing")
async def get_processing_config():
    """
    Get processing configuration parameters.
    """
    try:
        config = get_config()
        return config.processing.dict()

    except Exception as e:
        logger.error(f"Failed to get processing config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get processing config: {str(e)}"
        )


@router.put("/processing")
async def update_processing_config(
    min_object_size: Optional[int] = Query(
        None, ge=1, description="Minimum object size in voxels"
    ),
    pad_xy: Optional[int] = Query(None, ge=0, description="XY padding around objects"),
    max_objects_per_page: Optional[int] = Query(
        None, ge=1, le=100, description="Max objects per page"
    ),
    mip_chunk_size: Optional[int] = Query(None, ge=1, description="MIP chunk size"),
    cache_mips: Optional[bool] = Query(None, description="Enable MIP caching"),
    auto_contrast: Optional[bool] = Query(None, description="Enable auto contrast"),
    percentile_min: Optional[float] = Query(
        None, ge=0, le=100, description="Min percentile for contrast"
    ),
    percentile_max: Optional[float] = Query(
        None, ge=0, le=100, description="Max percentile for contrast"
    ),
):
    """
    Update processing configuration parameters.
    """
    try:
        updates = {}

        if min_object_size is not None:
            updates["min_object_size"] = min_object_size
        if pad_xy is not None:
            updates["pad_xy"] = pad_xy
        if max_objects_per_page is not None:
            updates["max_objects_per_page"] = max_objects_per_page
        if mip_chunk_size is not None:
            updates["mip_chunk_size"] = mip_chunk_size
        if cache_mips is not None:
            updates["cache_mips"] = cache_mips
        if auto_contrast is not None:
            updates["auto_contrast"] = auto_contrast

        if percentile_min is not None or percentile_max is not None:
            config = get_config()
            current_range = config.processing.percentile_range
            new_min = percentile_min if percentile_min is not None else current_range[0]
            new_max = percentile_max if percentile_max is not None else current_range[1]

            if new_min >= new_max:
                raise HTTPException(
                    status_code=400,
                    detail="percentile_min must be less than percentile_max",
                )

            updates["percentile_range"] = [new_min, new_max]

        if updates:
            update_config(processing=updates)

        # Return updated config
        updated_config = get_config()
        return {
            "status": "success",
            "message": "Processing configuration updated",
            "processing": updated_config.processing.dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update processing config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update processing config: {str(e)}"
        )


@router.get("/ui")
async def get_ui_config():
    """
    Get UI configuration parameters.
    """
    try:
        config = get_config()
        return config.ui.dict()

    except Exception as e:
        logger.error(f"Failed to get UI config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get UI config: {str(e)}"
        )


@router.put("/ui")
async def update_ui_config(
    default_opacity: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Default channel opacity"
    ),
    color_palette: Optional[List[str]] = Query(
        None, description="Color palette for channels"
    ),
):
    """
    Update UI configuration parameters.
    """
    try:
        updates = {}

        if default_opacity is not None:
            updates["default_opacity"] = default_opacity
        if color_palette is not None:
            # Validate color format
            for color in color_palette:
                if not (color.startswith("#") and len(color) == 7):
                    raise HTTPException(
                        status_code=400, detail=f"Invalid color format: {color}"
                    )
            updates["color_palette"] = color_palette

        if updates:
            update_config(ui=updates)

        # Return updated config
        updated_config = get_config()
        return {
            "status": "success",
            "message": "UI configuration updated",
            "ui": updated_config.ui.dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update UI config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update UI config: {str(e)}"
        )
