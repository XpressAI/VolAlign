"""
Nuclei processing utilities for MIP generation and image analysis.
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure

from .config import AppConfig
from .data_loader import DataLoader, DatasetInfo
from .models import NucleusInfo, EnhancedNucleusInfo, EpitopeAnalysisData

logger = logging.getLogger(__name__)


class MIPResult:
    """Result of maximum intensity projection computation."""

    def __init__(
        self,
        nucleus_label: int,
        mip_data: Dict[str, np.ndarray],
        bbox: Tuple[int, ...],
        padded_bbox: Tuple[int, ...],
        metadata: Dict[str, Any],
    ):
        self.nucleus_label = nucleus_label
        self.mip_data = mip_data  # Dict mapping channel name to MIP array (uint16)
        self.bbox = bbox
        self.padded_bbox = padded_bbox
        self.metadata = metadata
        self.nucleus_mask_2d: Optional[np.ndarray] = None  # 2D nucleus mask for contours

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "nucleus_label": self.nucleus_label,
            "bbox": self.bbox,
            "padded_bbox": self.padded_bbox,
            "metadata": self.metadata,
            "channels": list(self.mip_data.keys()),
            "has_nucleus_mask": self.nucleus_mask_2d is not None,
        }

    def get_channel_with_contours(self, channel_name: str, processor: 'NucleiProcessor') -> Optional[np.ndarray]:
        """
        Get a channel MIP with red contours added.
        
        Args:
            channel_name: Name of the channel
            processor: NucleiProcessor instance for contour generation
            
        Returns:
            RGB image with contours or None if channel not found
        """
        if channel_name not in self.mip_data or self.nucleus_mask_2d is None:
            return None
        
        return processor._add_red_contour(self.mip_data[channel_name], self.nucleus_mask_2d)


class NucleiProcessor:
    """Main processor for nuclei analysis and MIP generation."""

    def __init__(self, config: AppConfig, data_loader: Union[DataLoader, 'PipelineDataLoader']):
        self.config = config
        self.data_loader = data_loader
        self._mip_cache: Dict[int, MIPResult] = {}
        
        # Check if we're using pipeline data loader
        self._is_pipeline_mode = hasattr(data_loader, 'pipeline_config') or hasattr(data_loader, 'epitope_parser')

    def compute_nucleus_mip(
        self,
        nucleus_label: int,
        channels: Optional[List[str]] = None,
        force_recompute: bool = False,
    ) -> MIPResult:
        """
        Compute maximum intensity projection for a specific nucleus.

        Args:
            nucleus_label: Label of the nucleus to process
            channels: List of channel names to include (None for all available)
            force_recompute: Force recomputation even if cached

        Returns:
            MIPResult containing the computed projections

        Raises:
            ValueError: If nucleus not found or required data not loaded
        """
        # Check cache first
        if not force_recompute and nucleus_label in self._mip_cache:
            cached_result = self._mip_cache[nucleus_label]
            if channels is None or all(ch in cached_result.mip_data for ch in channels):
                logger.debug(f"Using cached MIP for nucleus {nucleus_label}")
                return cached_result

        # Get nucleus information
        nucleus = self.data_loader.get_nucleus_by_label(nucleus_label)
        if nucleus is None:
            raise ValueError(f"Nucleus with label {nucleus_label} not found")

        # Validate required datasets are loaded
        if self.data_loader.segmentation is None:
            raise ValueError("Segmentation dataset not loaded")

        # Determine which channels to process
        available_channels = {}

        # Add DAPI channel if available
        if self.data_loader.dapi_channel is not None:
            available_channels["dapi"] = self.data_loader.dapi_channel

        # Add epitope channels
        available_channels.update(self.data_loader.epitope_channels)

        if channels is None:
            channels_to_process = list(available_channels.keys())
        else:
            channels_to_process = [ch for ch in channels if ch in available_channels]
            missing_channels = [ch for ch in channels if ch not in available_channels]
            if missing_channels:
                logger.warning(f"Requested channels not available: {missing_channels}")

        if not channels_to_process:
            raise ValueError("No valid channels available for processing")

        logger.info(
            f"Computing MIP for nucleus {nucleus_label}, channels: {channels_to_process}"
        )

        # Get padded bounding box
        seg_shape = self.data_loader.segmentation.shape
        padded_bbox = nucleus.get_padded_bbox(self.config.processing.pad_xy, seg_shape)
        min_z, min_y, min_x, max_z, max_y, max_x = padded_bbox

        # Load segmentation mask for this region
        seg_array = self.data_loader.segmentation.get_dask_array()
        seg_crop = seg_array[min_z:max_z, min_y:max_y, min_x:max_x]

        # Create binary mask for this specific nucleus
        nucleus_mask = seg_crop == nucleus_label

        # Compute MIP for each channel
        mip_data = {}
        channel_stats = {}

        for channel_name in channels_to_process:
            try:
                channel_dataset = available_channels[channel_name]
                channel_array = channel_dataset.get_dask_array()

                # Crop channel data to same region
                channel_crop = channel_array[min_z:max_z, min_y:max_y, min_x:max_x]

                # Compute maximum intensity projection along Z-axis without nucleus masking
                # This shows the full cropped region including background and neighboring structures
                mip = da.max(channel_crop, axis=0)
                mip_np = mip.compute()

                # Store MIP data (without contours for now)
                mip_data[channel_name] = mip_np

                # Compute channel statistics for the full cropped region (no nucleus masking)
                channel_data = channel_crop.compute()
                nonzero_voxels = channel_data[channel_data > 0]
                
                if len(nonzero_voxels) > 0:
                    channel_stats[channel_name] = {
                        "min": float(nonzero_voxels.min()),
                        "max": float(nonzero_voxels.max()),
                        "mean": float(nonzero_voxels.mean()),
                        "std": float(nonzero_voxels.std()),
                        "nonzero_voxels": int(len(nonzero_voxels)),
                        "total_voxels": int(channel_data.size),
                        "region_shape": channel_data.shape,
                    }
                else:
                    channel_stats[channel_name] = {
                        "min": 0.0,
                        "max": 0.0,
                        "mean": 0.0,
                        "std": 0.0,
                        "nonzero_voxels": 0,
                        "total_voxels": int(channel_data.size),
                        "region_shape": channel_data.shape,
                    }

                logger.debug(
                    f"Computed MIP for {channel_name}: shape={mip_np.shape}, "
                    f"range=[{channel_stats[channel_name]['min']:.1f}, "
                    f"{channel_stats[channel_name]['max']:.1f}], "
                    f"total_voxels={channel_stats[channel_name]['total_voxels']}"
                )

            except Exception as e:
                logger.error(f"Failed to compute MIP for channel {channel_name}: {e}")
                continue

        if not mip_data:
            raise RuntimeError("Failed to compute MIP for any channel")

        # Keep MIP data in original uint16 format - no automatic contour addition
        # The frontend can handle color mapping and contours as needed

        # Create 2D nucleus mask for potential contour generation
        nucleus_mask_2d = da.max(nucleus_mask, axis=0).compute()
        
        # Create metadata
        metadata = {
            "nucleus_area": nucleus.area,
            "nucleus_centroid": nucleus.centroid,
            "crop_shape": (max_z - min_z, max_y - min_y, max_x - min_x),
            "mip_shape": next(iter(mip_data.values())).shape,
            "pad_xy": self.config.processing.pad_xy,
            "channel_stats": channel_stats,
            "processing_params": {
                "min_object_size": self.config.processing.min_object_size,
                "auto_contrast": self.config.processing.auto_contrast,
                "percentile_range": self.config.processing.percentile_range,
            },
            "has_contours": False,  # Contours can be added on demand
            "nucleus_mask_available": True,  # We have the mask for contour generation
            "nucleus_masking_applied": False,  # No nucleus masking applied to MIP data
        }

        # Create result with clean uint16 MIP data
        result = MIPResult(
            nucleus_label=nucleus_label,
            mip_data=mip_data,  # Use clean MIP data without contours
            bbox=nucleus.bbox,
            padded_bbox=padded_bbox,
            metadata=metadata,
        )
        
        # Store the 2D nucleus mask in the result for potential contour generation
        result.nucleus_mask_2d = nucleus_mask_2d

        # Cache result if enabled
        if self.config.processing.cache_mips:
            self._mip_cache[nucleus_label] = result

        return result

    def compute_batch_mips(
        self, nucleus_labels: List[int], channels: Optional[List[str]] = None
    ) -> Dict[int, MIPResult]:
        """
        Compute MIPs for multiple nuclei in batch.

        Args:
            nucleus_labels: List of nucleus labels to process
            channels: List of channel names to include

        Returns:
            Dict mapping nucleus label to MIPResult
        """
        results = {}

        for label in nucleus_labels:
            try:
                result = self.compute_nucleus_mip(label, channels)
                results[label] = result
            except Exception as e:
                logger.error(f"Failed to compute MIP for nucleus {label}: {e}")
                continue

        logger.info(f"Computed MIPs for {len(results)}/{len(nucleus_labels)} nuclei")
        return results

    def apply_contrast_enhancement(
        self, image: np.ndarray, percentile_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Apply contrast enhancement to an image using percentile-based scaling.

        Args:
            image: Input image array
            percentile_range: (min_percentile, max_percentile) for scaling

        Returns:
            Contrast-enhanced image scaled to [0, 255] uint8
        """
        if percentile_range is None:
            percentile_range = self.config.processing.percentile_range

        # Handle completely empty images
        if image.max() == 0:
            return np.zeros_like(image, dtype=np.uint8)

        # Handle flat images (all pixels have same non-zero value)
        if image.max() == image.min():
            # For flat non-zero images, return a mid-gray level
            return np.full_like(image, 128, dtype=np.uint8)

        # Compute percentiles only on non-zero pixels to avoid bias from empty regions
        nonzero_pixels = image[image > 0]
        if len(nonzero_pixels) == 0:
            return np.zeros_like(image, dtype=np.uint8)

        p_min, p_max = np.percentile(nonzero_pixels, percentile_range)

        if p_max <= p_min:
            # If percentiles are the same, use the full range of non-zero data
            p_min = nonzero_pixels.min()
            p_max = nonzero_pixels.max()
            
            if p_max <= p_min:
                return np.full_like(image, 128, dtype=np.uint8)

        # Scale to [0, 1]
        scaled = np.clip((image.astype(np.float32) - p_min) / (p_max - p_min), 0, 1)

        # Convert to uint8
        return (scaled * 255).astype(np.uint8)

    def create_composite_image(
        self, mip_result: MIPResult, channel_settings: Dict[str, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Create a composite RGB image from multiple channel MIPs.

        Args:
            mip_result: MIP result containing channel data
            channel_settings: Dict mapping channel name to settings
                             (color, opacity, enabled, contrast_range)

        Returns:
            RGB composite image as uint8 array
        """
        if not mip_result.mip_data:
            raise ValueError("No MIP data available")

        # Get image dimensions from first channel
        first_channel = next(iter(mip_result.mip_data.values()))
        height, width = first_channel.shape

        # Initialize composite as float32 for accumulation
        composite = np.zeros((height, width, 3), dtype=np.float32)
        
        # Track which channels actually contribute to avoid bias from empty channels
        channels_with_signal = []

        for channel_name, mip_data in mip_result.mip_data.items():
            if channel_name not in channel_settings:
                continue

            settings = channel_settings[channel_name]
            if not settings.get("enabled", True):
                continue

            # Check if channel has any signal (skip completely empty channels)
            if mip_data.max() == 0:
                logger.debug(f"Skipping empty channel {channel_name} in composite")
                continue
                
            channels_with_signal.append(channel_name)

            # Apply contrast enhancement
            if settings.get("auto_contrast", self.config.processing.auto_contrast):
                contrast_range = settings.get(
                    "contrast_range", self.config.processing.percentile_range
                )
                enhanced = self.apply_contrast_enhancement(mip_data, contrast_range)
            else:
                # Manual scaling
                min_val = settings.get("min_value", mip_data.min())
                max_val = settings.get("max_value", mip_data.max())
                if max_val > min_val:
                    scaled = np.clip(
                        (mip_data.astype(np.float32) - min_val) / (max_val - min_val),
                        0,
                        1,
                    )
                    enhanced = (scaled * 255).astype(np.uint8)
                else:
                    # Channel has uniform intensity, create a flat image
                    enhanced = np.full_like(mip_data, min_val, dtype=np.uint8)

            # Get color and opacity
            color = settings.get("color", "#ffffff")
            opacity = settings.get("opacity", self.config.ui.default_opacity)

            # Convert hex color to RGB
            if color.startswith("#"):
                color = color[1:]
            rgb = tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

            # Apply color and opacity
            enhanced_float = enhanced.astype(np.float32) / 255.0
            for i, color_component in enumerate(rgb):
                composite[:, :, i] += enhanced_float * color_component * opacity

        # If no channels had signal, return a black image
        if not channels_with_signal:
            logger.warning("No channels with signal found for composite generation")
            return np.zeros((height, width, 3), dtype=np.uint8)

        # Clip and convert to uint8
        composite = np.clip(composite, 0, 1)
        return (composite * 255).astype(np.uint8)

    def mip_to_base64_png(
        self, mip_array: np.ndarray, apply_contrast: bool = True
    ) -> str:
        """
        Convert MIP array to base64-encoded PNG for web display.

        Args:
            mip_array: MIP image array (uint16 or uint8)
            apply_contrast: Whether to apply contrast enhancement

        Returns:
            Base64-encoded PNG string
        """
        if apply_contrast:
            # Apply contrast enhancement (converts to uint8)
            processed = self.apply_contrast_enhancement(mip_array)
        else:
            # Convert uint16 to uint8 if needed
            if mip_array.dtype == np.uint16:
                # Scale uint16 to uint8 range
                if mip_array.max() > 255:
                    # Scale from 16-bit to 8-bit range
                    processed = (mip_array.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
                else:
                    # Already in 8-bit range
                    processed = mip_array.astype(np.uint8)
            elif mip_array.dtype != np.uint8:
                processed = np.clip(mip_array, 0, 255).astype(np.uint8)
            else:
                processed = mip_array

        # Convert to PIL Image
        if len(processed.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(processed, mode="L")
        elif len(processed.shape) == 3 and processed.shape[2] == 3:
            # RGB
            pil_image = Image.fromarray(processed, mode="RGB")
        else:
            raise ValueError(f"Unsupported image shape: {processed.shape}")

        # Convert to PNG bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode as base64
        png_bytes = buffer.getvalue()
        base64_string = base64.b64encode(png_bytes).decode("utf-8")

        return f"data:image/png;base64,{base64_string}"

    def get_nucleus_summary(self, nucleus_label: int) -> Dict[str, Any]:
        """
        Get summary information for a nucleus.

        Args:
            nucleus_label: Label of the nucleus

        Returns:
            Dictionary containing nucleus summary
        """
        nucleus = self.data_loader.get_nucleus_by_label(nucleus_label)
        if nucleus is None:
            raise ValueError(f"Nucleus with label {nucleus_label} not found")

        # Check if MIP is cached
        has_cached_mip = nucleus_label in self._mip_cache

        summary = {
            "label": nucleus.label,
            "area": nucleus.area,
            "centroid": nucleus.centroid,
            "bbox": nucleus.bbox,
            "has_cached_mip": has_cached_mip,
        }

        if has_cached_mip:
            mip_result = self._mip_cache[nucleus_label]
            summary["available_channels"] = list(mip_result.mip_data.keys())
            summary["mip_shape"] = mip_result.metadata["mip_shape"]

        # Add epitope analysis data if available (pipeline mode)
        if self._is_pipeline_mode and hasattr(nucleus, 'epitope_analysis') and nucleus.epitope_analysis is not None:
            summary["has_epitope_analysis"] = True
            summary["epitope_calls"] = nucleus.epitope_analysis.epitope_calls or {}
            summary["confidence_scores"] = nucleus.epitope_analysis.confidence_scores or {}
            summary["quality_score"] = nucleus.epitope_analysis.quality_score
        else:
            summary["has_epitope_analysis"] = False
            summary["epitope_calls"] = None
            summary["confidence_scores"] = None
            summary["quality_score"] = None

        return summary

    def clear_cache(self) -> None:
        """Clear the MIP cache."""
        self._mip_cache.clear()
        logger.info("Cleared MIP cache")

    def _add_red_contour(self, mip_array: np.ndarray, nucleus_mask_2d: np.ndarray) -> np.ndarray:
        """
        Add red contour around nucleus mask edges to MIP image.
        
        Args:
            mip_array: Original MIP image (grayscale)
            nucleus_mask_2d: 2D binary mask of the nucleus
            
        Returns:
            RGB image with red contours
        """
        # Convert grayscale MIP to RGB
        if len(mip_array.shape) == 2:
            # Convert to RGB by duplicating grayscale values
            height, width = mip_array.shape
            rgb_image = np.zeros((height, width, 3), dtype=mip_array.dtype)
            rgb_image[:, :, 0] = mip_array  # Red channel
            rgb_image[:, :, 1] = mip_array  # Green channel
            rgb_image[:, :, 2] = mip_array  # Blue channel
        else:
            rgb_image = mip_array.copy()
        
        # Find contours using skimage
        try:
            # Convert mask to binary if needed
            binary_mask = nucleus_mask_2d.astype(bool)
            
            # Find contours
            contours = measure.find_contours(binary_mask, 0.5)
            
            # Draw red contours on the RGB image
            for contour in contours:
                # Convert contour coordinates to integer indices
                coords = np.round(contour).astype(int)
                
                # Ensure coordinates are within image bounds
                height, width = rgb_image.shape[:2]
                valid_coords = (
                    (coords[:, 0] >= 0) & (coords[:, 0] < height) &
                    (coords[:, 1] >= 0) & (coords[:, 1] < width)
                )
                coords = coords[valid_coords]
                
                if len(coords) > 0:
                    # Set contour pixels to red (max intensity in red channel, zero in others)
                    max_val = np.iinfo(rgb_image.dtype).max if np.issubdtype(rgb_image.dtype, np.integer) else 1.0
                    rgb_image[coords[:, 0], coords[:, 1], 0] = max_val  # Red channel
                    rgb_image[coords[:, 0], coords[:, 1], 1] = 0        # Green channel
                    rgb_image[coords[:, 0], coords[:, 1], 2] = 0        # Blue channel
                    
        except Exception as e:
            logger.warning(f"Failed to add red contours: {e}")
            # Return original image as RGB if contouring fails
            pass
        
        return rgb_image

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return {
            "cached_nuclei": len(self._mip_cache),
            "cached_labels": list(self._mip_cache.keys()),
            "cache_enabled": self.config.processing.cache_mips,
        }
