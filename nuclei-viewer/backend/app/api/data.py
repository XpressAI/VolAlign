"""
Data management API endpoints.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..core.config import AppConfig, get_config
from ..core.data_loader import DataLoader
from ..core.shared_state import get_shared_data_loader, reset_shared_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


def get_data_loader():
    """Dependency to get the shared data loader instance."""
    return get_shared_data_loader()


class DatasetDiscoveryResponse(BaseModel):
    """Response model for dataset discovery."""

    discovered_files: Dict[str, List[str]]
    total_files: int


class DatasetInfoResponse(BaseModel):
    """Response model for dataset information."""

    name: str
    path: str
    shape: List[int]
    dtype: str
    array_key: Optional[str]


class DataSourceRequest(BaseModel):
    """Request model for a data source."""

    file_path: str
    array_key: Optional[str] = None


class EpitopeChannelRequest(BaseModel):
    """Request model for an epitope channel."""

    name: str
    file_path: str
    array_key: Optional[str] = None
    default_color: str = "#ffffff"


class LoadDatasetRequest(BaseModel):
    """Request model for loading datasets."""

    segmentation: DataSourceRequest
    dapi_channel: DataSourceRequest
    epitope_channels: List[EpitopeChannelRequest] = []


@router.get("/discover", response_model=DatasetDiscoveryResponse)
async def discover_datasets(
    base_path: Optional[str] = Query(
        None, description="Override base path for discovery"
    ),
    data_loader: DataLoader = Depends(get_data_loader),
):
    """
    Discover available datasets in the configured directory.
    """
    try:
        # Update base path if provided
        if base_path:
            data_loader.base_path = Path(base_path)

        discovered = data_loader.discover_datasets()

        # Convert Path objects to strings for JSON serialization
        discovered_str = {}
        total_files = 0
        for dataset_type, files in discovered.items():
            discovered_str[dataset_type] = [str(f) for f in files]
            total_files += len(files)

        return DatasetDiscoveryResponse(
            discovered_files=discovered_str,
            base_path=str(data_loader.base_path),
            total_files=total_files,
        )

    except Exception as e:
        logger.error(f"Dataset discovery failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Dataset discovery failed: {str(e)}"
        )


@router.post("/load")
async def load_datasets(
    request: LoadDatasetRequest, data_loader: DataLoader = Depends(get_data_loader)
):
    """
    Load datasets for processing using full file paths.
    """
    try:
        results = {}

        # Load segmentation
        try:
            seg_file = Path(request.segmentation.file_path)
            segmentation = data_loader.load_segmentation(
                seg_file, request.segmentation.array_key
            )
            results["segmentation"] = DatasetInfoResponse(
                name=segmentation.name,
                path=str(segmentation.path),
                shape=list(segmentation.shape),
                dtype=segmentation.dtype,
                array_key=segmentation.array_key,
            )
        except Exception as e:
            logger.error(f"Failed to load segmentation: {e}")
            results["segmentation"] = {"error": str(e)}

        # Load DAPI channel
        try:
            dapi_file = Path(request.dapi_channel.file_path)
            dapi = data_loader.load_dapi_channel(
                dapi_file, request.dapi_channel.array_key
            )
            results["dapi"] = DatasetInfoResponse(
                name=dapi.name,
                path=str(dapi.path),
                shape=list(dapi.shape),
                dtype=dapi.dtype,
                array_key=dapi.array_key,
            )
        except Exception as e:
            logger.error(f"Failed to load DAPI channel: {e}")
            results["dapi"] = {"error": str(e)}

        # Load epitope channels
        try:
            epitope_channels = {}
            for channel_req in request.epitope_channels:
                try:
                    channel_file = Path(channel_req.file_path)
                    dataset = data_loader.load_epitope_channel(
                        channel_req.name, channel_file, channel_req.array_key
                    )
                    epitope_channels[channel_req.name] = dataset
                except Exception as e:
                    logger.error(
                        f"Failed to load epitope channel {channel_req.name}: {e}"
                    )
                    epitope_channels[channel_req.name] = {"error": str(e)}

            results["epitope_channels"] = {}
            for name, dataset in epitope_channels.items():
                if isinstance(dataset, dict) and "error" in dataset:
                    results["epitope_channels"][name] = dataset
                else:
                    results["epitope_channels"][name] = DatasetInfoResponse(
                        name=dataset.name,
                        path=str(dataset.path),
                        shape=list(dataset.shape),
                        dtype=dataset.dtype,
                        array_key=dataset.array_key,
                    )
        except Exception as e:
            logger.error(f"Failed to load epitope channels: {e}")
            results["epitope_channels"] = {"error": str(e)}

        return {"status": "success", "loaded_datasets": results}

    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset loading failed: {str(e)}")


@router.post("/load-all")
async def load_all_datasets(
    base_path: Optional[str] = Query(None, description="Override base path"),
    data_loader: DataLoader = Depends(get_data_loader),
):
    """
    Auto-discover and load all datasets.
    """
    try:
        # Update base path if provided
        if base_path:
            data_loader.base_path = Path(base_path)

        # Load all datasets
        data_loader.load_all_datasets()

        # Return information about loaded datasets
        results = {}

        if data_loader.segmentation:
            results["segmentation"] = DatasetInfoResponse(
                name=data_loader.segmentation.name,
                path=str(data_loader.segmentation.path),
                shape=list(data_loader.segmentation.shape),
                dtype=data_loader.segmentation.dtype,
                array_key=data_loader.segmentation.array_key,
            )

        if data_loader.dapi_channel:
            results["dapi"] = DatasetInfoResponse(
                name=data_loader.dapi_channel.name,
                path=str(data_loader.dapi_channel.path),
                shape=list(data_loader.dapi_channel.shape),
                dtype=data_loader.dapi_channel.dtype,
                array_key=data_loader.dapi_channel.array_key,
            )

        results["epitope_channels"] = {}
        for name, dataset in data_loader.epitope_channels.items():
            results["epitope_channels"][name] = DatasetInfoResponse(
                name=dataset.name,
                path=str(dataset.path),
                shape=list(dataset.shape),
                dtype=dataset.dtype,
                array_key=dataset.array_key,
            )

        return {"status": "success", "loaded_datasets": results}

    except Exception as e:
        logger.error(f"Auto-loading datasets failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Auto-loading datasets failed: {str(e)}"
        )


@router.get("/status")
async def get_data_status(data_loader: DataLoader = Depends(get_data_loader)):
    """
    Get current data loading status.
    """
    try:
        status = {
            "datasets_loaded": {
                "segmentation": data_loader.segmentation is not None,
                "dapi": data_loader.dapi_channel is not None,
                "epitope_channels": len(data_loader.epitope_channels),
            },
            "nuclei_extracted": data_loader._nuclei_loaded,
            "total_nuclei": (
                len(data_loader.nuclei) if data_loader._nuclei_loaded else 0
            ),
        }

        if data_loader._nuclei_loaded:
            config = get_config()
            status["pagination"] = {
                "total_nuclei": len(data_loader.nuclei),
                "nuclei_per_page": config.processing.max_objects_per_page,
                "total_pages": data_loader.get_total_pages(),
            }

        return status

    except Exception as e:
        logger.error(f"Failed to get data status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get data status: {str(e)}"
        )


@router.get("/channels")
async def get_available_channels(data_loader: DataLoader = Depends(get_data_loader)):
    """
    Get information about available channels.
    """
    try:
        channels = {}

        # Add DAPI channel if available
        if data_loader.dapi_channel:
            channels["dapi"] = {
                "name": "DAPI",
                "type": "dapi",
                "shape": list(data_loader.dapi_channel.shape),
                "dtype": data_loader.dapi_channel.dtype,
                "default_color": "#0080ff",  # Blue
            }

        # Add epitope channels
        config = get_config()
        for name, dataset in data_loader.epitope_channels.items():
            # Find corresponding config for default color
            default_color = "#ffffff"
            for epitope_config in config.data.epitope_channels:
                if epitope_config.name == name:
                    default_color = epitope_config.default_color
                    break

            channels[name] = {
                "name": name,
                "type": "epitope",
                "shape": list(dataset.shape),
                "dtype": dataset.dtype,
                "default_color": default_color,
            }

        return {"channels": channels, "total_channels": len(channels)}

    except Exception as e:
        logger.error(f"Failed to get channel information: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get channel information: {str(e)}"
        )


@router.delete("/clear")
async def clear_data(data_loader: DataLoader = Depends(get_data_loader)):
    """
    Clear all loaded data and reset the data loader.
    """
    try:
        # Reset shared state
        reset_shared_state()

        return {"status": "success", "message": "All data cleared"}

    except Exception as e:
        logger.error(f"Failed to clear data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")
