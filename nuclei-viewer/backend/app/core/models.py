"""
Enhanced data models for nuclei-viewer with pipeline integration support.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class EpitopeAnalysisData:
    """Epitope analysis results for a single nucleus."""
    
    epitope_calls: Dict[str, bool] = field(default_factory=dict)  # "round_channel" -> positive/negative
    confidence_scores: Dict[str, float] = field(default_factory=dict)  # "round_channel" -> confidence
    intensity_values: Dict[str, float] = field(default_factory=dict)  # "round_channel" -> intensity
    cutoff_values: Dict[str, float] = field(default_factory=dict)  # "round_channel" -> cutoff used
    quality_score: float = 0.0  # Overall quality score for this nucleus
    analysis_region: str = "combined"  # Region used for analysis ("nuclei", "shell", "combined")
    rounds_analyzed: List[str] = field(default_factory=list)  # List of rounds analyzed


@dataclass
class ChannelStatistics:
    """Detailed statistics for a single channel in a nucleus."""
    
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    n_voxels: int = 0
    min_val: float = 0.0
    max_val: float = 0.0


@dataclass
class QualityMetrics:
    """Quality metrics for a channel in a nucleus."""
    
    snr_nuclei: float = 0.0  # Signal-to-noise ratio in nuclei region
    coverage_ratio: float = 0.0  # Ratio of nucleus voxels covered
    shell_to_nuclei_ratio: float = 0.0  # Ratio of shell to nuclei intensity


@dataclass
class EnhancedNucleusInfo:
    """Enhanced nucleus information with pipeline integration support."""
    
    # Core nucleus properties (compatible with existing NucleusInfo)
    label: int
    bbox: Tuple[int, int, int, int, int, int]  # (min_z, min_y, min_x, max_z, max_y, max_x)
    area: int
    centroid: Tuple[float, float, float]  # (z, y, x)
    
    # Pipeline-specific enhancements
    epitope_analysis: Optional[EpitopeAnalysisData] = None
    
    # Per-channel statistics: "round_channel" -> {"nuclei": stats, "shell": stats, "combined": stats}
    channel_statistics: Dict[str, Dict[str, ChannelStatistics]] = field(default_factory=dict)
    
    # Quality metrics per channel: "round_channel" -> QualityMetrics
    quality_metrics: Dict[str, QualityMetrics] = field(default_factory=dict)
    
    # Intensity data per region: "round_channel" -> intensity
    nuclei_intensities: Dict[str, float] = field(default_factory=dict)
    shell_intensities: Dict[str, float] = field(default_factory=dict)
    combined_intensities: Dict[str, float] = field(default_factory=dict)
    
    # Processing metadata
    rounds_processed: List[str] = field(default_factory=list)
    bounding_box_padded: Optional[Tuple[int, int, int, int, int, int]] = None
    
    def get_padded_bbox(self, pad_xy: int, volume_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get bounding box with XY padding, clipped to volume bounds."""
        min_z, min_y, min_x, max_z, max_y, max_x = self.bbox

        # Apply padding in XY dimensions only
        min_y_pad = max(0, min_y - pad_xy)
        max_y_pad = min(volume_shape[1], max_y + pad_xy)
        min_x_pad = max(0, min_x - pad_xy)
        max_x_pad = min(volume_shape[2], max_x + pad_xy)

        return (min_z, min_y_pad, min_x_pad, max_z, max_y_pad, max_x_pad)
    
    def has_epitope_analysis(self) -> bool:
        """Check if this nucleus has epitope analysis data."""
        return self.epitope_analysis is not None
    
    def get_positive_epitope_calls(self) -> Dict[str, bool]:
        """Get only the positive epitope calls."""
        if not self.has_epitope_analysis():
            return {}
        return {
            round_channel: call 
            for round_channel, call in self.epitope_analysis.epitope_calls.items() 
            if call
        }
    
    def get_epitope_call_summary(self) -> Dict[str, int]:
        """Get summary of epitope calls by round."""
        if not self.has_epitope_analysis():
            return {}
        
        summary = {}
        for round_channel, call in self.epitope_analysis.epitope_calls.items():
            round_name = round_channel.split('_')[0]  # Extract round name
            if round_name not in summary:
                summary[round_name] = 0
            if call:
                summary[round_name] += 1
        
        return summary


@dataclass
class PipelineMetadata:
    """Metadata from pipeline epitope analysis."""
    
    analysis_type: str = "nucleus_centric_epitope_analysis"
    n_nuclei: int = 0
    n_rounds: int = 0
    analysis_region: str = "combined"
    cutoff_method: str = "otsu"
    shell_parameters: Dict[str, Any] = field(default_factory=dict)
    reference_round: str = ""
    epitope_channels: List[str] = field(default_factory=list)


@dataclass
class PipelineDataset:
    """Information about a dataset loaded from pipeline outputs."""
    
    working_directory: Path
    reference_round: str
    epitope_analysis_file: Optional[Path] = None
    segmentation_file: Optional[Path] = None
    zarr_volumes_dir: Optional[Path] = None
    aligned_dir: Optional[Path] = None
    
    # Loaded data
    metadata: Optional[PipelineMetadata] = None
    cutoffs: Dict[str, float] = field(default_factory=dict)  # "round_channel" -> cutoff
    nuclei: List[EnhancedNucleusInfo] = field(default_factory=list)
    
    # Discovery results
    discovered_zarr_files: Dict[str, List[Path]] = field(default_factory=dict)
    available_rounds: List[str] = field(default_factory=list)
    available_channels: Dict[str, List[str]] = field(default_factory=dict)  # round -> channels
    
    def is_valid(self) -> bool:
        """Check if the pipeline dataset has minimum required components."""
        return (
            self.working_directory.exists() and
            (self.segmentation_file is None or self.segmentation_file.exists()) and
            (self.epitope_analysis_file is None or self.epitope_analysis_file.exists())
        )
    
    def get_channel_path(self, round_name: str, channel: str) -> Optional[Path]:
        """Get the path to a specific channel zarr file."""
        if round_name == self.reference_round:
            # Reference round data
            if self.zarr_volumes_dir:
                return self.zarr_volumes_dir / round_name / f"{round_name}_{channel}.zarr"
        else:
            # Aligned round data
            if self.aligned_dir:
                return self.aligned_dir / round_name / f"{round_name}_{channel}_aligned.zarr"
        return None
    
    def get_available_channels_for_round(self, round_name: str) -> List[str]:
        """Get list of available channels for a specific round."""
        return self.available_channels.get(round_name, [])


# Backward compatibility - alias for existing code
NucleusInfo = EnhancedNucleusInfo


def convert_legacy_nucleus_info(legacy_info) -> EnhancedNucleusInfo:
    """Convert legacy NucleusInfo to EnhancedNucleusInfo."""
    return EnhancedNucleusInfo(
        label=legacy_info.label,
        bbox=legacy_info.bbox,
        area=legacy_info.area,
        centroid=legacy_info.centroid
    )


def create_channel_statistics_from_dict(stats_dict: Dict[str, Any]) -> ChannelStatistics:
    """Create ChannelStatistics from dictionary."""
    return ChannelStatistics(
        mean=stats_dict.get('mean', 0.0),
        std=stats_dict.get('std', 0.0),
        median=stats_dict.get('median', 0.0),
        p25=stats_dict.get('p25', 0.0),
        p75=stats_dict.get('p75', 0.0),
        n_voxels=stats_dict.get('n_voxels', 0),
        min_val=stats_dict.get('min', 0.0),
        max_val=stats_dict.get('max', 0.0)
    )


def create_quality_metrics_from_dict(metrics_dict: Dict[str, Any]) -> QualityMetrics:
    """Create QualityMetrics from dictionary."""
    return QualityMetrics(
        snr_nuclei=metrics_dict.get('snr_nuclei', 0.0),
        coverage_ratio=metrics_dict.get('coverage_ratio', 0.0),
        shell_to_nuclei_ratio=metrics_dict.get('shell_to_nuclei_ratio', 0.0)
    )