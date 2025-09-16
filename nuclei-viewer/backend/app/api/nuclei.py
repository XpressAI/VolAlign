"""
Nuclei processing and visualization API endpoints.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..core.config import AppConfig, get_config
from ..core.nuclei_processor import MIPResult
from ..core.shared_state import get_shared_data_loader, get_shared_nuclei_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/nuclei", tags=["nuclei"])


def get_data_loader():
    """Dependency to get the shared data loader instance."""
    return get_shared_data_loader()


def get_nuclei_processor():
    """Dependency to get the shared nuclei processor instance."""
    return get_shared_nuclei_processor()


class NucleusSummaryResponse(BaseModel):
    """Response model for nucleus summary."""

    label: int
    area: int
    centroid: List[float]
    bbox: List[int]
    has_cached_mip: bool
    available_channels: Optional[List[str]] = None
    mip_shape: Optional[List[int]] = None


class NucleiPageResponse(BaseModel):
    """Response model for paginated nuclei."""

    nuclei: List[NucleusSummaryResponse]
    page: int
    page_size: int
    total_nuclei: int
    total_pages: int
    has_next: bool
    has_previous: bool


class ChannelSettings(BaseModel):
    """Channel visualization settings."""

    enabled: bool = True
    color: str = "#ffffff"
    opacity: float = Field(default=0.8, ge=0.0, le=1.0)
    auto_contrast: bool = True
    contrast_range: Optional[List[float]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class MIPRequest(BaseModel):
    """Request model for MIP computation."""

    nucleus_label: int
    channels: Optional[List[str]] = None
    force_recompute: bool = False
    channel_settings: Optional[Dict[str, ChannelSettings]] = None
    return_composite: bool = False
    return_individual: bool = True


class MIPResponse(BaseModel):
    """Response model for MIP computation."""

    nucleus_label: int
    bbox: List[int]
    padded_bbox: List[int]
    metadata: Dict[str, Any]
    channels: List[str]
    individual_mips: Optional[Dict[str, str]] = None  # Base64 encoded images
    composite_mip: Optional[str] = None  # Base64 encoded composite image


@router.get("/extract")
async def extract_nuclei_info(
    force_reload: bool = Query(False, description="Force re-extraction of nuclei"),
    data_loader=Depends(get_data_loader),
):
    """
    Extract nuclei information from segmentation mask.
    """
    try:
        nuclei = data_loader.extract_nuclei_info(force_reload=force_reload)

        return {
            "status": "success",
            "total_nuclei": len(nuclei),
            "nuclei_extracted": True,
            "min_object_size": get_config().processing.min_object_size,
        }

    except Exception as e:
        logger.error(f"Nuclei extraction failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Nuclei extraction failed: {str(e)}"
        )


@router.get("/list", response_model=NucleiPageResponse)
async def list_nuclei(
    page: int = Query(0, ge=0, description="Page number (0-based)"),
    page_size: Optional[int] = Query(
        None, ge=1, le=100, description="Number of nuclei per page"
    ),
    data_loader=Depends(get_data_loader),
    processor=Depends(get_nuclei_processor),
):
    """
    Get paginated list of nuclei.
    """
    try:
        # Ensure nuclei are extracted
        if not data_loader._nuclei_loaded:
            data_loader.extract_nuclei_info()

        # Get page of nuclei
        nuclei_page = data_loader.get_nuclei_page(page, page_size)
        total_pages = data_loader.get_total_pages(page_size)

        if page_size is None:
            page_size = get_config().processing.max_objects_per_page

        # Convert to response format
        nuclei_summaries = []
        for nucleus in nuclei_page:
            summary = processor.get_nucleus_summary(nucleus.label)
            nuclei_summaries.append(NucleusSummaryResponse(**summary))

        return NucleiPageResponse(
            nuclei=nuclei_summaries,
            page=page,
            page_size=page_size,
            total_nuclei=len(data_loader.nuclei),
            total_pages=total_pages,
            has_next=page < total_pages - 1,
            has_previous=page > 0,
        )

    except Exception as e:
        logger.error(f"Failed to list nuclei: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list nuclei: {str(e)}")


@router.get("/{nucleus_label}/summary", response_model=NucleusSummaryResponse)
async def get_nucleus_summary(
    nucleus_label: int, processor=Depends(get_nuclei_processor)
):
    """
    Get summary information for a specific nucleus.
    """
    try:
        summary = processor.get_nucleus_summary(nucleus_label)
        return NucleusSummaryResponse(**summary)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get nucleus summary: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get nucleus summary: {str(e)}"
        )


@router.post("/mip", response_model=MIPResponse)
async def compute_mip(request: MIPRequest, processor=Depends(get_nuclei_processor)):
    """
    Compute maximum intensity projection for a nucleus.
    """
    try:
        # Compute MIP
        mip_result = processor.compute_nucleus_mip(
            nucleus_label=request.nucleus_label,
            channels=request.channels,
            force_recompute=request.force_recompute,
        )

        response = MIPResponse(
            nucleus_label=mip_result.nucleus_label,
            bbox=list(mip_result.bbox),
            padded_bbox=list(mip_result.padded_bbox),
            metadata=mip_result.metadata,
            channels=list(mip_result.mip_data.keys()),
        )

        # Generate individual channel images if requested
        if request.return_individual:
            response.individual_mips = {}
            for channel_name, mip_data in mip_result.mip_data.items():
                base64_image = processor.mip_to_base64_png(
                    mip_data, apply_contrast=True
                )
                response.individual_mips[channel_name] = base64_image

        # Generate composite image if requested
        if request.return_composite and request.channel_settings:
            # Convert Pydantic models to dicts
            channel_settings_dict = {}
            for channel_name, settings in request.channel_settings.items():
                channel_settings_dict[channel_name] = settings.dict()

            composite_image = processor.create_composite_image(
                mip_result, channel_settings_dict
            )
            response.composite_mip = processor.mip_to_base64_png(
                composite_image, apply_contrast=False
            )

        return response

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to compute MIP: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compute MIP: {str(e)}")


@router.post("/batch-mip")
async def compute_batch_mips(
    nucleus_labels: List[int],
    channels: Optional[List[str]] = None,
    return_individual: bool = Query(
        True, description="Return individual channel images"
    ),
    processor=Depends(get_nuclei_processor),
):
    """
    Compute MIPs for multiple nuclei in batch.
    """
    try:
        # Limit batch size to prevent memory issues
        max_batch_size = 50
        if len(nucleus_labels) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size too large. Maximum allowed: {max_batch_size}",
            )

        # Compute batch MIPs
        results = processor.compute_batch_mips(nucleus_labels, channels)

        # Convert to response format
        batch_response = {}
        for label, mip_result in results.items():
            response = MIPResponse(
                nucleus_label=mip_result.nucleus_label,
                bbox=list(mip_result.bbox),
                padded_bbox=list(mip_result.padded_bbox),
                metadata=mip_result.metadata,
                channels=list(mip_result.mip_data.keys()),
            )

            if return_individual:
                response.individual_mips = {}
                for channel_name, mip_data in mip_result.mip_data.items():
                    base64_image = processor.mip_to_base64_png(
                        mip_data, apply_contrast=True
                    )
                    response.individual_mips[channel_name] = base64_image

            batch_response[str(label)] = response

        return {
            "status": "success",
            "processed_nuclei": len(results),
            "failed_nuclei": len(nucleus_labels) - len(results),
            "results": batch_response,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compute batch MIPs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to compute batch MIPs: {str(e)}"
        )


@router.get("/cache/info")
async def get_cache_info(processor=Depends(get_nuclei_processor)):
    """
    Get information about the MIP cache.
    """
    try:
        cache_info = processor.get_cache_info()
        return cache_info

    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache info: {str(e)}"
        )


@router.delete("/cache/clear")
async def clear_cache(processor=Depends(get_nuclei_processor)):
    """
    Clear the MIP cache.
    """
    try:
        processor.clear_cache()
        return {"status": "success", "message": "Cache cleared"}

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/stats")
async def get_nuclei_stats(
    data_loader=Depends(get_data_loader), processor=Depends(get_nuclei_processor)
):
    """
    Get statistics about the nuclei dataset.
    """
    try:
        # Ensure nuclei are extracted
        if not data_loader._nuclei_loaded:
            data_loader.extract_nuclei_info()

        nuclei = data_loader.nuclei

        if not nuclei:
            return {
                "total_nuclei": 0,
                "area_stats": {},
                "cache_info": processor.get_cache_info(),
            }

        # Compute area statistics
        areas = [nucleus.area for nucleus in nuclei]
        area_stats = {
            "min": min(areas),
            "max": max(areas),
            "mean": sum(areas) / len(areas),
            "median": sorted(areas)[len(areas) // 2],
        }

        # Compute centroid statistics
        centroids = [nucleus.centroid for nucleus in nuclei]
        centroid_stats = {
            "z_range": [min(c[0] for c in centroids), max(c[0] for c in centroids)],
            "y_range": [min(c[1] for c in centroids), max(c[1] for c in centroids)],
            "x_range": [min(c[2] for c in centroids), max(c[2] for c in centroids)],
        }

        return {
            "total_nuclei": len(nuclei),
            "area_stats": area_stats,
            "centroid_stats": centroid_stats,
            "cache_info": processor.get_cache_info(),
            "processing_params": {
                "min_object_size": get_config().processing.min_object_size,
                "pad_xy": get_config().processing.pad_xy,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get nuclei stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get nuclei stats: {str(e)}"
        )
