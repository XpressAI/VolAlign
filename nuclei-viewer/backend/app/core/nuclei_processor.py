"""
Nuclei processing utilities for MIP generation and image analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import dask.array as da
from PIL import Image
import io
import base64

from .config import AppConfig
from .data_loader import DataLoader, NucleusInfo, DatasetInfo

logger = logging.getLogger(__name__)


class MIPResult:
    """Result of maximum intensity projection computation."""
    
    def __init__(self, 
                 nucleus_label: int,
                 mip_data: Dict[str, np.ndarray],
                 bbox: Tuple[int, ...],
                 padded_bbox: Tuple[int, ...],
                 metadata: Dict[str, Any]):
        self.nucleus_label = nucleus_label
        self.mip_data = mip_data  # Dict mapping channel name to MIP array
        self.bbox = bbox
        self.padded_bbox = padded_bbox
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'nucleus_label': self.nucleus_label,
            'bbox': self.bbox,
            'padded_bbox': self.padded_bbox,
            'metadata': self.metadata,
            'channels': list(self.mip_data.keys())
        }


class NucleiProcessor:
    """Main processor for nuclei analysis and MIP generation."""
    
    def __init__(self, config: AppConfig, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader
        self._mip_cache: Dict[int, MIPResult] = {}
    
    def compute_nucleus_mip(self, 
                           nucleus_label: int, 
                           channels: Optional[List[str]] = None,
                           force_recompute: bool = False) -> MIPResult:
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
            available_channels['dapi'] = self.data_loader.dapi_channel
        
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
        
        logger.info(f"Computing MIP for nucleus {nucleus_label}, channels: {channels_to_process}")
        
        # Get padded bounding box
        seg_shape = self.data_loader.segmentation.shape
        padded_bbox = nucleus.get_padded_bbox(self.config.processing.pad_xy, seg_shape)
        min_z, min_y, min_x, max_z, max_y, max_x = padded_bbox
        
        # Load segmentation mask for this region
        seg_array = self.data_loader.segmentation.get_dask_array()
        seg_crop = seg_array[min_z:max_z, min_y:max_y, min_x:max_x]
        
        # Create binary mask for this specific nucleus
        nucleus_mask = (seg_crop == nucleus_label)
        
        # Compute MIP for each channel
        mip_data = {}
        channel_stats = {}
        
        for channel_name in channels_to_process:
            try:
                channel_dataset = available_channels[channel_name]
                channel_array = channel_dataset.get_dask_array()
                
                # Crop channel data to same region
                channel_crop = channel_array[min_z:max_z, min_y:max_y, min_x:max_x]
                
                # Apply nucleus mask (set non-nucleus voxels to 0)
                masked_channel = da.where(nucleus_mask, channel_crop, 0)
                
                # Compute maximum intensity projection along Z-axis
                mip = da.max(masked_channel, axis=0)
                mip_np = mip.compute()
                
                # Store MIP data
                mip_data[channel_name] = mip_np
                
                # Compute channel statistics
                masked_data = masked_channel.compute()
                nonzero_mask = masked_data > 0
                if np.any(nonzero_mask):
                    channel_stats[channel_name] = {
                        'min': float(masked_data[nonzero_mask].min()),
                        'max': float(masked_data[nonzero_mask].max()),
                        'mean': float(masked_data[nonzero_mask].mean()),
                        'std': float(masked_data[nonzero_mask].std()),
                        'nonzero_voxels': int(np.sum(nonzero_mask))
                    }
                else:
                    channel_stats[channel_name] = {
                        'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'nonzero_voxels': 0
                    }
                
                logger.debug(f"Computed MIP for {channel_name}: shape={mip_np.shape}, "
                           f"range=[{channel_stats[channel_name]['min']:.1f}, "
                           f"{channel_stats[channel_name]['max']:.1f}]")
                
            except Exception as e:
                logger.error(f"Failed to compute MIP for channel {channel_name}: {e}")
                continue
        
        if not mip_data:
            raise RuntimeError("Failed to compute MIP for any channel")
        
        # Create metadata
        metadata = {
            'nucleus_area': nucleus.area,
            'nucleus_centroid': nucleus.centroid,
            'crop_shape': (max_z - min_z, max_y - min_y, max_x - min_x),
            'mip_shape': next(iter(mip_data.values())).shape,
            'pad_xy': self.config.processing.pad_xy,
            'channel_stats': channel_stats,
            'processing_params': {
                'min_object_size': self.config.processing.min_object_size,
                'auto_contrast': self.config.processing.auto_contrast,
                'percentile_range': self.config.processing.percentile_range
            }
        }
        
        # Create result
        result = MIPResult(
            nucleus_label=nucleus_label,
            mip_data=mip_data,
            bbox=nucleus.bbox,
            padded_bbox=padded_bbox,
            metadata=metadata
        )
        
        # Cache result if enabled
        if self.config.processing.cache_mips:
            self._mip_cache[nucleus_label] = result
        
        return result
    
    def compute_batch_mips(self, 
                          nucleus_labels: List[int],
                          channels: Optional[List[str]] = None) -> Dict[int, MIPResult]:
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
    
    def apply_contrast_enhancement(self, 
                                 image: np.ndarray,
                                 percentile_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
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
        
        if image.max() == image.min():
            # Flat image
            return np.zeros_like(image, dtype=np.uint8)
        
        # Compute percentiles
        p_min, p_max = np.percentile(image, percentile_range)
        
        if p_max <= p_min:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Scale to [0, 1]
        scaled = np.clip((image.astype(np.float32) - p_min) / (p_max - p_min), 0, 1)
        
        # Convert to uint8
        return (scaled * 255).astype(np.uint8)
    
    def create_composite_image(self, 
                             mip_result: MIPResult,
                             channel_settings: Dict[str, Dict[str, Any]]) -> np.ndarray:
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
        
        for channel_name, mip_data in mip_result.mip_data.items():
            if channel_name not in channel_settings:
                continue
            
            settings = channel_settings[channel_name]
            if not settings.get('enabled', True):
                continue
            
            # Apply contrast enhancement
            if settings.get('auto_contrast', self.config.processing.auto_contrast):
                contrast_range = settings.get('contrast_range', self.config.processing.percentile_range)
                enhanced = self.apply_contrast_enhancement(mip_data, contrast_range)
            else:
                # Manual scaling
                min_val = settings.get('min_value', mip_data.min())
                max_val = settings.get('max_value', mip_data.max())
                if max_val > min_val:
                    scaled = np.clip((mip_data.astype(np.float32) - min_val) / (max_val - min_val), 0, 1)
                    enhanced = (scaled * 255).astype(np.uint8)
                else:
                    enhanced = np.zeros_like(mip_data, dtype=np.uint8)
            
            # Get color and opacity
            color = settings.get('color', '#ffffff')
            opacity = settings.get('opacity', self.config.ui.default_opacity)
            
            # Convert hex color to RGB
            if color.startswith('#'):
                color = color[1:]
            rgb = tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            
            # Apply color and opacity
            enhanced_float = enhanced.astype(np.float32) / 255.0
            for i, color_component in enumerate(rgb):
                composite[:, :, i] += enhanced_float * color_component * opacity
        
        # Clip and convert to uint8
        composite = np.clip(composite, 0, 1)
        return (composite * 255).astype(np.uint8)
    
    def mip_to_base64_png(self, 
                         mip_array: np.ndarray,
                         apply_contrast: bool = True) -> str:
        """
        Convert MIP array to base64-encoded PNG for web display.
        
        Args:
            mip_array: MIP image array
            apply_contrast: Whether to apply contrast enhancement
            
        Returns:
            Base64-encoded PNG string
        """
        if apply_contrast:
            processed = self.apply_contrast_enhancement(mip_array)
        else:
            # Convert to uint8 if needed
            if mip_array.dtype != np.uint8:
                processed = np.clip(mip_array, 0, 255).astype(np.uint8)
            else:
                processed = mip_array
        
        # Convert to PIL Image
        if len(processed.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(processed, mode='L')
        elif len(processed.shape) == 3 and processed.shape[2] == 3:
            # RGB
            pil_image = Image.fromarray(processed, mode='RGB')
        else:
            raise ValueError(f"Unsupported image shape: {processed.shape}")
        
        # Convert to PNG bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode as base64
        png_bytes = buffer.getvalue()
        base64_string = base64.b64encode(png_bytes).decode('utf-8')
        
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
            'label': nucleus.label,
            'area': nucleus.area,
            'centroid': nucleus.centroid,
            'bbox': nucleus.bbox,
            'has_cached_mip': has_cached_mip
        }
        
        if has_cached_mip:
            mip_result = self._mip_cache[nucleus_label]
            summary['available_channels'] = list(mip_result.mip_data.keys())
            summary['mip_shape'] = mip_result.metadata['mip_shape']
        
        return summary
    
    def clear_cache(self) -> None:
        """Clear the MIP cache."""
        self._mip_cache.clear()
        logger.info("Cleared MIP cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return {
            'cached_nuclei': len(self._mip_cache),
            'cached_labels': list(self._mip_cache.keys()),
            'cache_enabled': self.config.processing.cache_mips
        }