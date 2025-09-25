
"""
Epitope Analysis Module for VolAlign Pipeline

This module provides heavy computational functionality for epitope expression analysis
within nuclei masks and shell regions. Designed for large-scale microscopy data processing
with memory-efficient algorithms and distributed computing support.

Key Features:
- Multi-region intensity extraction (nuclei, shell, combined)
- Statistical distribution analysis with multiple cutoff methods
- Memory-efficient chunked processing for large Zarr volumes
- Integration with existing VolAlign distributed processing framework
- Comprehensive validation and quality control metrics
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zarr
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.stats import gaussian_kde, skew, kurtosis
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from skimage.filters import threshold_otsu
from skimage.morphology import ball
from tqdm import tqdm

from .utils import _calculate_safe_chunks


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class EpitopeChannel:
    """Represents an epitope channel with metadata."""
    name: str
    round_name: str
    zarr_path: Path
    wavelength: str
    epitope_tag: Optional[str] = None
    is_reference_round: bool = False


@dataclass
class ShellParameters:
    """Parameters for shell region generation."""
    erosion_iterations: int = 2
    dilation_iterations: int = 6
    footprint_size: int = 3
    use_3d_morphology: bool = True


@dataclass
class NucleusEpitopeData:
    """Complete epitope data for a single nucleus across all regions."""
    nucleus_label: int
    round_name: str
    nuclei_intensities: Dict[str, float]  # channel -> mean intensity within nuclei
    shell_intensities: Dict[str, float]  # channel -> mean intensity in shell region
    combined_intensities: Dict[str, float]  # channel -> combined nuclei + shell intensity
    channel_statistics: Dict[str, Dict[str, float]]  # detailed stats for all regions
    spatial_location: Tuple[float, float, float]  # centroid (z, y, x)
    shell_parameters: Dict[str, Any]  # parameters used for shell generation
    quality_metrics: Dict[str, float]  # SNR, coverage, etc.


@dataclass
class EpitopeAnalysisResult:
    """Results of epitope analysis with cutoff determinations."""
    nucleus_label: int
    epitope_calls: Dict[str, bool]  # epitope -> positive/negative
    confidence_scores: Dict[str, float]
    intensity_values: Dict[str, float]
    cutoff_values: Dict[str, float]
    analysis_region: str  # "nuclei", "shell", or "combined"
    quality_score: float


@dataclass
class DistributionAnalysis:
    """Statistical analysis of intensity distributions."""
    channel: str
    region: str
    n_samples: int
    mean: float
    std: float
    median: float
    skewness: float
    kurtosis_val: float
    percentiles: Dict[str, float]  # p5, p25, p75, p95, etc.
    is_bimodal: bool
    bimodal_confidence: float
    gmm_components: Optional[Dict[str, Any]] = None


# =============================================================================
# CORE EPITOPE INTENSITY EXTRACTOR
# =============================================================================

class EpitopeIntensityExtractor:
    """
    Memory-efficient epitope intensity extraction with multi-region analysis.
    
    Supports processing of large Zarr volumes with chunked processing and
    integration with VolAlign's distributed processing framework.
    """
    
    def __init__(
        self,
        shell_config: ShellParameters,
        aligned_data_dir: Path,
        zarr_volumes_dir: Path,
        reference_round: str,
        chunk_size: int = 50,
        memory_limit_gb: float = 8.0
    ):
        """
        Initialize epitope intensity extractor.
        
        Args:
            shell_config: Parameters for shell region generation
            aligned_data_dir: Directory containing aligned round data
            zarr_volumes_dir: Directory containing reference round data
            reference_round: Name of reference round
            chunk_size: Number of Z-slices to process at once
            memory_limit_gb: Memory limit for processing chunks
        """
        self.shell_config = shell_config
        self.aligned_data_dir = Path(aligned_data_dir)
        self.zarr_volumes_dir = Path(zarr_volumes_dir)
        self.reference_round = reference_round
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        
        # Cache for loaded volumes to avoid repeated I/O
        self._volume_cache: Dict[str, zarr.Array] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}
        
    def get_channel_path(self, round_name: str, channel: str) -> Path:
        """
        Get correct path for channel data based on round type.
        
        Args:
            round_name: Name of the imaging round
            channel: Channel identifier
            
        Returns:
            Path to the channel Zarr file
        """
        if round_name == self.reference_round:
            return self.zarr_volumes_dir / round_name / f"{round_name}_{channel}.zarr"
        else:
            return self.aligned_data_dir / round_name / f"{round_name}_{channel}.zarr"
    
    def load_channel_volume(self, round_name: str, channel: str) -> zarr.Array:
        """
        Load channel volume with caching.
        
        Args:
            round_name: Name of the imaging round
            channel: Channel identifier
            
        Returns:
            Zarr array for the channel
        """
        cache_key = f"{round_name}_{channel}"
        
        if cache_key not in self._volume_cache:
            channel_path = self.get_channel_path(round_name, channel)
            if not channel_path.exists():
                raise FileNotFoundError(f"Channel file not found: {channel_path}")
            
            self._volume_cache[cache_key] = zarr.open(str(channel_path), mode='r')
            print(f"Loaded channel volume: {cache_key} - Shape: {self._volume_cache[cache_key].shape}")
        
        return self._volume_cache[cache_key]
    
    def create_shell_mask(
        self,
        nucleus_mask: np.ndarray,
        shell_params: Optional[ShellParameters] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create three masks: nucleus, shell, and combined regions.
        
        Args:
            nucleus_mask: Binary mask of the nucleus
            shell_params: Shell generation parameters (uses default if None)
            
        Returns:
            Tuple of (nucleus_mask, shell_mask, combined_mask)
        """
        if shell_params is None:
            shell_params = self.shell_config
        
        # Create morphological footprint
        if shell_params.use_3d_morphology and nucleus_mask.ndim == 3:
            footprint = ball(shell_params.footprint_size)
        else:
            # Use 2D footprint for each slice
            footprint = np.ones((shell_params.footprint_size, shell_params.footprint_size))
        
        # Erode nucleus to create inner boundary
        eroded_nucleus = nucleus_mask.copy()
        for _ in range(shell_params.erosion_iterations):
            if shell_params.use_3d_morphology and nucleus_mask.ndim == 3:
                eroded_nucleus = binary_erosion(eroded_nucleus, structure=footprint)
            else:
                # Apply 2D erosion slice by slice
                for z in range(eroded_nucleus.shape[0]):
                    eroded_nucleus[z] = binary_erosion(eroded_nucleus[z], structure=footprint)
        
        # Dilate original nucleus to create outer boundary
        dilated_nucleus = nucleus_mask.copy()
        for _ in range(shell_params.dilation_iterations):
            if shell_params.use_3d_morphology and nucleus_mask.ndim == 3:
                dilated_nucleus = binary_dilation(dilated_nucleus, structure=footprint)
            else:
                # Apply 2D dilation slice by slice
                for z in range(dilated_nucleus.shape[0]):
                    dilated_nucleus[z] = binary_dilation(dilated_nucleus[z], structure=footprint)
        
        # Create shell mask: dilated - original nucleus
        shell_mask = dilated_nucleus & ~nucleus_mask
        
        # Combined mask: original nucleus + shell
        combined_mask = nucleus_mask | shell_mask
        
        return nucleus_mask, shell_mask, combined_mask
    
    def extract_nucleus_intensities(
        self,
        nucleus_label: int,
        epitope_channels: List[EpitopeChannel],
        segmentation_mask: zarr.Array,
        shell_params: Optional[ShellParameters] = None
    ) -> NucleusEpitopeData:
        """
        Extract intensities from three regions for a single nucleus.
        
        Args:
            nucleus_label: Label ID of the nucleus
            epitope_channels: List of epitope channels to analyze
            segmentation_mask: Segmentation mask Zarr array
            shell_params: Shell generation parameters
            
        Returns:
            NucleusEpitopeData with intensities for all regions
        """
        if shell_params is None:
            shell_params = self.shell_config
        
        # Extract nucleus mask
        nucleus_mask = (segmentation_mask[:] == nucleus_label).astype(bool)
        
        if not np.any(nucleus_mask):
            raise ValueError(f"Nucleus label {nucleus_label} not found in segmentation mask")
        
        # Create shell masks
        nucleus_mask_clean, shell_mask, combined_mask = self.create_shell_mask(
            nucleus_mask, shell_params
        )
        
        # Calculate centroid
        coords = np.where(nucleus_mask_clean)
        centroid = (
            float(np.mean(coords[0])),  # z
            float(np.mean(coords[1])),  # y
            float(np.mean(coords[2]))   # x
        )
        
        # Initialize intensity dictionaries
        nuclei_intensities = {}
        shell_intensities = {}
        combined_intensities = {}
        channel_statistics = {}
        quality_metrics = {}
        
        # Process each epitope channel
        for channel in epitope_channels:
            try:
                # Load channel volume
                volume = self.load_channel_volume(channel.round_name, channel.name)
                
                # Extract intensities for each region
                nuclei_values = volume[:][nucleus_mask_clean]
                shell_values = volume[:][shell_mask] if np.any(shell_mask) else np.array([])
                combined_values = volume[:][combined_mask]
                
                # Calculate mean intensities
                nuclei_intensities[channel.name] = float(np.mean(nuclei_values)) if len(nuclei_values) > 0 else 0.0
                shell_intensities[channel.name] = float(np.mean(shell_values)) if len(shell_values) > 0 else 0.0
                combined_intensities[channel.name] = float(np.mean(combined_values)) if len(combined_values) > 0 else 0.0
                
                # Calculate detailed statistics
                channel_statistics[channel.name] = {
                    'nuclei': {
                        'mean': nuclei_intensities[channel.name],
                        'std': float(np.std(nuclei_values)) if len(nuclei_values) > 0 else 0.0,
                        'median': float(np.median(nuclei_values)) if len(nuclei_values) > 0 else 0.0,
                        'p25': float(np.percentile(nuclei_values, 25)) if len(nuclei_values) > 0 else 0.0,
                        'p75': float(np.percentile(nuclei_values, 75)) if len(nuclei_values) > 0 else 0.0,
                        'n_voxels': len(nuclei_values)
                    },
                    'shell': {
                        'mean': shell_intensities[channel.name],
                        'std': float(np.std(shell_values)) if len(shell_values) > 0 else 0.0,
                        'median': float(np.median(shell_values)) if len(shell_values) > 0 else 0.0,
                        'p25': float(np.percentile(shell_values, 25)) if len(shell_values) > 0 else 0.0,
                        'p75': float(np.percentile(shell_values, 75)) if len(shell_values) > 0 else 0.0,
                        'n_voxels': len(shell_values)
                    },
                    'combined': {
                        'mean': combined_intensities[channel.name],
                        'std': float(np.std(combined_values)) if len(combined_values) > 0 else 0.0,
                        'median': float(np.median(combined_values)) if len(combined_values) > 0 else 0.0,
                        'p25': float(np.percentile(combined_values, 25)) if len(combined_values) > 0 else 0.0,
                        'p75': float(np.percentile(combined_values, 75)) if len(combined_values) > 0 else 0.0,
                        'n_voxels': len(combined_values)
                    }
                }
                
                # Calculate quality metrics
                snr_nuclei = (nuclei_intensities[channel.name] / 
                             (channel_statistics[channel.name]['nuclei']['std'] + 1e-6))
                coverage_ratio = len(nuclei_values) / np.sum(nucleus_mask)
                
                quality_metrics[channel.name] = {
                    'snr_nuclei': snr_nuclei,
                    'coverage_ratio': coverage_ratio,
                    'shell_to_nuclei_ratio': (shell_intensities[channel.name] / 
                                            (nuclei_intensities[channel.name] + 1e-6))
                }
                
            except Exception as e:
                print(f"Error processing channel {channel.name}: {e}")
                # Set default values for failed channels
                nuclei_intensities[channel.name] = 0.0
                shell_intensities[channel.name] = 0.0
                combined_intensities[channel.name] = 0.0
                channel_statistics[channel.name] = {}
                quality_metrics[channel.name] = {'snr_nuclei': 0.0, 'coverage_ratio': 0.0}
        
        return NucleusEpitopeData(
            nucleus_label=nucleus_label,
            round_name=epitope_channels[0].round_name if epitope_channels else "unknown",
            nuclei_intensities=nuclei_intensities,
            shell_intensities=shell_intensities,
            combined_intensities=combined_intensities,
            channel_statistics=channel_statistics,
            spatial_location=centroid,
            shell_parameters=asdict(shell_params),
            quality_metrics=quality_metrics
        )
    
    def batch_extract_intensities(
        self,
        nucleus_labels: List[int],
        epitope_channels: List[EpitopeChannel],
        segmentation_mask: zarr.Array,
        shell_params: Optional[ShellParameters] = None
    ) -> Dict[int, NucleusEpitopeData]:
        """
        Batch processing for multiple nuclei with memory-efficient processing.
        
        Args:
            nucleus_labels: List of nucleus label IDs to process
            epitope_channels: List of epitope channels to analyze
            segmentation_mask: Segmentation mask Zarr array
            shell_params: Shell generation parameters
            
        Returns:
            Dictionary mapping nucleus labels to epitope data
        """
        results = {}
        
        print(f"Processing {len(nucleus_labels)} nuclei across {len(epitope_channels)} channels...")
        
        for nucleus_label in tqdm(nucleus_labels, desc="Extracting epitope intensities"):
            try:
                nucleus_data = self.extract_nucleus_intensities(
                    nucleus_label=nucleus_label,
                    epitope_channels=epitope_channels,
                    segmentation_mask=segmentation_mask,
                    shell_params=shell_params
                )
                results[nucleus_label] = nucleus_data
                
            except Exception as e:
                print(f"Failed to process nucleus {nucleus_label}: {e}")
                continue
        
        print(f"Successfully processed {len(results)}/{len(nucleus_labels)} nuclei")
        return results


# =============================================================================
# STATISTICAL ANALYSIS ENGINE
# =============================================================================

class EpitopeStatisticalAnalyzer:
    """
    Statistical analysis engine for epitope intensity distributions.
    
    Provides multiple methods for cutoff determination and distribution analysis
    with support for multi-region analysis (nuclei, shell, combined).
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize statistical analyzer.
        
        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def compute_intensity_distributions(
        self, 
        intensity_data: Dict[int, NucleusEpitopeData],
        channel: str,
        region: str = "combined"
    ) -> DistributionAnalysis:
        """
        Compute comprehensive distribution analysis for a channel and region.
        
        Args:
            intensity_data: Dictionary of nucleus epitope data
            channel: Channel name to analyze
            region: Region to analyze ("nuclei", "shell", "combined")
            
        Returns:
            DistributionAnalysis with comprehensive statistics
        """
        # Extract intensities for the specified channel and region
        if region == "nuclei":
            intensities = [data.nuclei_intensities.get(channel, 0.0) 
                          for data in intensity_data.values()]
        elif region == "shell":
            intensities = [data.shell_intensities.get(channel, 0.0) 
                          for data in intensity_data.values()]
        elif region == "combined":
            intensities = [data.combined_intensities.get(channel, 0.0) 
                          for data in intensity_data.values()]
        else:
            raise ValueError(f"Unknown region: {region}")
        
        intensities = np.array(intensities)
        intensities = intensities[intensities > 0]  # Remove zero values
        
        if len(intensities) == 0:
            raise ValueError(f"No valid intensities found for channel {channel}, region {region}")
        
        # Basic statistics
        mean_val = float(np.mean(intensities))
        std_val = float(np.std(intensities))
        median_val = float(np.median(intensities))
        skewness = float(skew(intensities))
        kurtosis_val = float(kurtosis(intensities))
        
        # Percentiles
        percentiles = {
            'p5': float(np.percentile(intensities, 5)),
            'p10': float(np.percentile(intensities, 10)),
            'p25': float(np.percentile(intensities, 25)),
            'p75': float(np.percentile(intensities, 75)),
            'p90': float(np.percentile(intensities, 90)),
            'p95': float(np.percentile(intensities, 95)),
            'p99': float(np.percentile(intensities, 99))
        }
        
        # Bimodality detection
        is_bimodal, bimodal_confidence, gmm_components = self.detect_bimodal_distribution(intensities)
        
        return DistributionAnalysis(
            channel=channel,
            region=region,
            n_samples=len(intensities),
            mean=mean_val,
            std=std_val,
            median=median_val,
            skewness=skewness,
            kurtosis_val=kurtosis_val,
            percentiles=percentiles,
            is_bimodal=is_bimodal,
            bimodal_confidence=bimodal_confidence,
            gmm_components=gmm_components
        )
    
    def detect_bimodal_distribution(
        self, 
        intensities: np.ndarray
    ) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
        """
        Detect bimodal distribution using Gaussian Mixture Models.
        
        Args:
            intensities: Array of intensity values
            
        Returns:
            Tuple of (is_bimodal, confidence_score, gmm_components)
        """
        if len(intensities) < 10:
            return False, 0.0, None
        
        try:
            # Fit 1-component and 2-component GMMs
            gmm1 = GaussianMixture(n_components=1, random_state=self.random_state)
            gmm2 = GaussianMixture(n_components=2, random_state=self.random_state)
            
            intensities_reshaped = intensities.reshape(-1, 1)
            
            gmm1.fit(intensities_reshaped)
            gmm2.fit(intensities_reshaped)
            
            # Calculate BIC scores
            bic1 = gmm1.bic(intensities_reshaped)
            bic2 = gmm2.bic(intensities_reshaped)
            
            # Lower BIC is better
            bic_improvement = bic1 - bic2
            
            # Consider bimodal if BIC improvement > 10 and components are well separated
            is_bimodal = bic_improvement > 10
            confidence = min(1.0, bic_improvement / 50.0)  # Normalize to 0-1
            
            if is_bimodal:
                # Extract component information
                means = gmm2.means_.flatten()
                stds = np.sqrt(gmm2.covariances_.flatten())
                weights = gmm2.weights_
                
                # Calculate separation between components
                separation = abs(means[1] - means[0]) / (np.mean(stds) + 1e-6)
                
                # Adjust confidence based on separation
                confidence *= min(1.0, separation / 2.0)
                
                gmm_components = {
                    'means': means.tolist(),
                    'stds': stds.tolist(),
                    'weights': weights.tolist(),
                    'separation': float(separation),
                    'bic_improvement': float(bic_improvement)
                }
            else:
                gmm_components = None
            
            return is_bimodal, float(confidence), gmm_components
            
        except Exception as e:
            print(f"Error in bimodal detection: {e}")
            return False, 0.0, None
    
    def suggest_cutoffs(
        self, 
        intensities: np.ndarray,
        method: str = "otsu"
    ) -> Dict[str, float]:
        """
        Suggest cutoff values using multiple methods.
        
        Args:
            intensities: Array of intensity values
            method: Cutoff determination method
            
        Returns:
            Dictionary of cutoff values from different methods
        """
        cutoffs = {}
        
        if len(intensities) == 0:
            return {"error": "No valid intensities"}
        
        try:
            # Otsu's method
            if method in ["otsu", "all"]:
                otsu_cutoff = threshold_otsu(intensities)
                cutoffs["otsu"] = float(otsu_cutoff)
            
            # Percentile-based cutoffs
            if method in ["percentile", "all"]:
                cutoffs["p75"] = float(np.percentile(intensities, 75))
                cutoffs["p90"] = float(np.percentile(intensities, 90))
                cutoffs["p95"] = float(np.percentile(intensities, 95))
            
            # GMM-based cutoff
            if method in ["gmm", "all"]:
                is_bimodal, confidence, gmm_components = self.detect_bimodal_distribution(intensities)
                if is_bimodal and gmm_components:
                    means = gmm_components['means']
                    stds = gmm_components['stds']
                    # Cutoff at midpoint between components
                    gmm_cutoff = (means[0] + means[1]) / 2.0
                    cutoffs["gmm"] = float(gmm_cutoff)
                    cutoffs["gmm_confidence"] = confidence
            
            # Mean + N*std cutoffs
            if method in ["std", "all"]:
                mean_val = np.mean(intensities)
                std_val = np.std(intensities)
                cutoffs["mean_plus_1std"] = float(mean_val + std_val)
                cutoffs["mean_plus_2std"] = float(mean_val + 2 * std_val)
                cutoffs["mean_plus_3std"] = float(mean_val + 3 * std_val)
            
        except Exception as e:
            cutoffs["error"] = str(e)
        
        return cutoffs


# =============================================================================
# CUTOFF DETERMINATION METHODS
# =============================================================================

class CutoffDetermination:
    """
    Advanced cutoff determination methods for epitope analysis.
    
    Provides multiple algorithms for determining optimal cutoffs with
    validation and region-specific analysis capabilities.
    """
    
    @staticmethod
    def otsu_threshold(intensities: np.ndarray, region: str = "combined") -> float:
        """
        Otsu's method for automatic thresholding.
        
        Args:
            intensities: Array of intensity values
            region: Region identifier for logging
            
        Returns:
            Optimal threshold value
        """
        if len(intensities) == 0:
            return 0.0
        
        try:
            threshold = threshold_otsu(intensities)
            return float(threshold)
        except Exception as e:
            print(f"Otsu threshold failed for region {region}: {e}")
            return float(np.percentile(intensities, 75))  # Fallback to 75th percentile
    
    @staticmethod
    def gmm_cutoff(intensities: np.ndarray, region: str = "combined") -> Dict[str, float]:
        """
        Gaussian mixture model-based cutoff determination.
        
        Args:
            intensities: Array of intensity values
            region: Region identifier for logging
            
        Returns:
            Dictionary with GMM-based cutoff and confidence metrics
        """
        if len(intensities) < 10:
            return {"cutoff": float(np.percentile(intensities, 75)), "confidence": 0.0}
        
        try:
            analyzer = EpitopeStatisticalAnalyzer()
            is_bimodal, confidence, gmm_components = analyzer.detect_bimodal_distribution(intensities)
            
            if is_bimodal and gmm_components:
                means = gmm_components['means']
                cutoff = (means[0] + means[1]) / 2.0
                return {
                    "cutoff": float(cutoff),
                    "confidence": confidence,
                    "component_means": means,
                    "separation": gmm_components['separation']
                }
            else:
                # Fallback to percentile-based cutoff
                return {
                    "cutoff": float(np.percentile(intensities, 75)),
                    "confidence": 0.0,
                    "note": "No bimodal distribution detected"
                }
                
        except Exception as e:
            print(f"GMM cutoff failed for region {region}: {e}")
            return {"cutoff": float(np.percentile(intensities, 75)), "confidence": 0.0}
    
    @staticmethod
    def percentile_cutoff(intensities: np.ndarray, percentile: float) -> float:
        """
        Percentile-based cutoff determination.
        
        Args:
            intensities: Array of intensity values
            percentile: Percentile value (0-100)
            
        Returns:
            Cutoff value at specified percentile
        """
        if len(intensities) == 0:
            return 0.0
        
        return float(np.percentile(intensities, percentile))
    
    @staticmethod
    def region_specific_cutoff(
        nuclei_intensities: np.ndarray,
        shell_intensities: np.ndarray,
        method: str = "optimal"
    ) -> Dict[str, float]:
        """
        Determine optimal cutoffs for each region separately.
        
        Args:
            nuclei_intensities: Intensities from nuclei region
            shell_intensities: Intensities from shell region
            method: Method for cutoff determination
            
        Returns:
            Dictionary with region-specific cutoffs
        """
        cutoffs = {}
        
        if method == "optimal":
            # Use Otsu for each region
            cutoffs["nuclei"] = CutoffDetermination.otsu_threshold(nuclei_intensities, "nuclei")
            cutoffs["shell"] = CutoffDetermination.otsu_threshold(shell_intensities, "shell")
        elif method == "percentile":
            # Use 75th percentile for each region
            cutoffs["nuclei"] = CutoffDetermination.percentile_cutoff(nuclei_intensities, 75)
            cutoffs["shell"] = CutoffDetermination.percentile_cutoff(shell_intensities, 75)
        
        return cutoffs
    
    @staticmethod
    def combined_region_cutoff(
        nuclei_intensities: np.ndarray,
        shell_intensities: np.ndarray,
        weights: Tuple[float, float] = (0.6, 0.4)
    ) -> float:
        """
        Weighted combination of nuclei and shell intensities for cutoff determination.
        
        Args:
            nuclei_intensities: Intensities from nuclei region
            shell_intensities: Intensities from shell region
            weights: Weights for (nuclei, shell) combination
            
        Returns:
            Combined cutoff value
        """
        if len(nuclei_intensities) == 0 and len(shell_intensities) == 0:
            return 0.0
        
        # Calculate individual cutoffs
        nuclei_cutoff = CutoffDetermination.otsu_threshold(nuclei_intensities, "nuclei")
        shell_cutoff = CutoffDetermination.otsu_threshold(shell_intensities, "shell")
        
        # Weighted combination
        combined_cutoff = weights[0] * nuclei_cutoff + weights[1] * shell_cutoff
        
        return float(combined_cutoff)
    
    @staticmethod
    def validate_cutoff(
        intensities: np.ndarray,
        cutoff: float,
        region: str = "combined",
        known_positives: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Validate cutoff using various metrics.
        
        Args:
            intensities: Array of intensity values
            cutoff: Cutoff value to validate
            region: Region identifier for logging
            known_positives: Optional list of known positive indices
            
        Returns:
            Dictionary with validation metrics
        """
        if len(intensities) == 0:
            return {"error": "No intensities provided"}
        validation_metrics = {}
        
        try:
            # Basic classification metrics
            positive_count = np.sum(intensities >= cutoff)
            negative_count = np.sum(intensities < cutoff)
            total_count = len(intensities)
            
            validation_metrics.update({
                "cutoff_value": float(cutoff),
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "positive_fraction": float(positive_count / total_count),
                "negative_fraction": float(negative_count / total_count)
            })
            
            # Separation quality metrics
            if positive_count > 0 and negative_count > 0:
                positive_intensities = intensities[intensities >= cutoff]
                negative_intensities = intensities[intensities < cutoff]
                
                pos_mean = np.mean(positive_intensities)
                neg_mean = np.mean(negative_intensities)
                pos_std = np.std(positive_intensities)
                neg_std = np.std(negative_intensities)
                
                # Cohen's d (effect size)
                pooled_std = np.sqrt(((positive_count - 1) * pos_std**2 + 
                                    (negative_count - 1) * neg_std**2) / 
                                   (positive_count + negative_count - 2))
                cohens_d = (pos_mean - neg_mean) / (pooled_std + 1e-6)
                
                validation_metrics.update({
                    "positive_mean": float(pos_mean),
                    "negative_mean": float(neg_mean),
                    "separation_ratio": float(pos_mean / (neg_mean + 1e-6)),
                    "cohens_d": float(cohens_d)
                })
            
            # If known positives are provided, calculate accuracy metrics
            if known_positives is not None:
                known_pos_array = np.zeros(len(intensities), dtype=bool)
                known_pos_array[known_positives] = True
                predicted_pos = intensities >= cutoff
                
                # Confusion matrix elements
                tp = np.sum(known_pos_array & predicted_pos)
                tn = np.sum(~known_pos_array & ~predicted_pos)
                fp = np.sum(~known_pos_array & predicted_pos)
                fn = np.sum(known_pos_array & ~predicted_pos)
                
                # Calculate metrics
                accuracy = (tp + tn) / len(intensities) if len(intensities) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                validation_metrics.update({
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score),
                    "true_positives": int(tp),
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn)
                })
            
        except Exception as e:
            validation_metrics["error"] = str(e)
        
        return validation_metrics


# =============================================================================
# EPITOPE ANALYSIS ENGINE
# =============================================================================

class EpitopeAnalyzer:
    """
    Epitope analysis engine for large-scale microscopy data processing.
    
    Currently implements sequential processing with plans for distributed
    processing in future versions. Integrates with VolAlign's processing
    framework for memory-efficient epitope expression analysis.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        pipeline_output_dir: Optional[Path] = None
    ):
        """
        Initialize distributed epitope analyzer from configuration.
        
        Args:
            config: Complete VolAlign configuration dictionary (from YAML)
            pipeline_output_dir: Optional override for pipeline output directory
        """
        self.config = config
        
        # Extract configuration sections
        self.epitope_config = config.get("epitope_analysis", {})
        if not self.epitope_config.get("enabled", False):
            raise ValueError("Epitope analysis is not enabled in configuration")
        
        # Set up directories
        if pipeline_output_dir:
            self.pipeline_output_dir = Path(pipeline_output_dir)
        else:
            self.pipeline_output_dir = Path(config["working_directory"])
        
        # Extract configuration parameters
        self._setup_from_config()
        
        # Initialize components
        self.extractor = None
        self.analyzer = EpitopeStatisticalAnalyzer(
            random_state=self.statistical_config.get("random_state", 42)
        )
        
        # Set up pipeline structure from config
        self._setup_pipeline_structure()
    
    def _setup_from_config(self):
        """Extract and validate configuration parameters."""
        # Shell parameters
        shell_params = self.epitope_config.get("shell_parameters", {})
        self.shell_config = ShellParameters(
            erosion_iterations=shell_params.get("erosion_iterations", 2),
            dilation_iterations=shell_params.get("dilation_iterations", 6),
            footprint_size=shell_params.get("footprint_size", 3),
            use_3d_morphology=shell_params.get("use_3d_morphology", True)
        )
        
        # Epitope channels configuration
        epitope_channels_config = self.epitope_config.get("epitope_channels", {})
        self.epitope_channel_names = epitope_channels_config.get("channels", [])
        self.channel_tags = epitope_channels_config.get("channel_tags", {})
        
        # Analysis parameters
        analysis_params = self.epitope_config.get("analysis_parameters", {})
        self.max_nuclei_per_round = analysis_params.get("max_nuclei_per_round")
        self.memory_limit_gb = analysis_params.get("memory_limit_gb", 8.0)
        self.chunk_size = analysis_params.get("chunk_size", 50)
        self.default_region = analysis_params.get("default_region", "combined")
        
        # Statistical analysis configuration
        self.statistical_config = self.epitope_config.get("statistical_analysis", {})
        
        # Output configuration
        output_config = self.epitope_config.get("output", {})
        self.output_subdir = output_config.get("output_subdir", "epitope_analysis")
        self.save_formats = output_config.get("save_formats", ["json"])
        self.save_detailed_stats = output_config.get("save_detailed_stats", True)
        self.save_plots = output_config.get("save_plots", False)
        
        # Cluster configuration (use main pipeline cluster config)
        self.cluster_config = self.config.get("cluster_config", {})
        
        # Validate epitope channels
        if not self.epitope_channel_names:
            raise ValueError("No epitope channels specified in configuration")
        
        print(f"Epitope analysis configuration loaded:")
        print(f"  - Epitope channels: {self.epitope_channel_names}")
        print(f"  - Channel tags: {self.channel_tags}")
        print(f"  - Shell parameters: {self.shell_config}")
        print(f"  - Max nuclei per round: {self.max_nuclei_per_round}")
    
    def _setup_pipeline_structure(self):
        """Set up pipeline structure from configuration."""
        # Get reference round from config
        self.reference_round = self.config["data"]["reference_round"]
        
        # Set up directory paths
        self.zarr_volumes_dir = self.pipeline_output_dir / "zarr_volumes"
        self.aligned_dir = self.pipeline_output_dir / "aligned"
        self.segmentation_dir = self.pipeline_output_dir / "segmentation"
        
        # Construct segmentation mask path from configuration
        segmentation_config = self.config.get("segmentation", {})
        upsample_results = segmentation_config.get("upsample_results", True)
        
        if upsample_results:
            self.segmentation_mask_path = (
                self.segmentation_dir / f"{self.reference_round}_nuclei_segmentation_fullres.zarr"
            )
        else:
            self.segmentation_mask_path = (
                self.segmentation_dir / f"{self.reference_round}_nuclei_segmentation.zarr"
            )
        
        print(f"Pipeline structure configured:")
        print(f"  - Reference round: {self.reference_round}")
        print(f"  - Segmentation mask: {self.segmentation_mask_path}")
        print(f"  - Zarr volumes: {self.zarr_volumes_dir}")
        print(f"  - Aligned data: {self.aligned_dir}")
    
    def discover_epitope_channels(self) -> Dict[str, List[EpitopeChannel]]:
        """
        Discover epitope channels based on configuration.
        
        Returns:
            Dictionary mapping round names to lists of epitope channels
        """
        epitope_channels = {}
        
        # Get rounds from configuration
        rounds_data = self.config["data"]["rounds"]
        
        for round_name, round_channels in rounds_data.items():
            channels = []
            
            # Process only configured epitope channels
            for channel_name in self.epitope_channel_names:
                if channel_name in round_channels:
                    # Determine the correct zarr path
                    if round_name == self.reference_round:
                        zarr_path = self.zarr_volumes_dir / round_name / f"{round_name}_{channel_name}.zarr"
                        is_reference = True
                    else:
                        zarr_path = self.aligned_dir / round_name / f"{round_name}_{channel_name}_aligned.zarr"
                        is_reference = False
                    
                    # Check if file exists
                    if zarr_path.exists():
                        epitope_tag = self.channel_tags.get(channel_name, None)
                        
                        channels.append(EpitopeChannel(
                            name=channel_name,
                            round_name=round_name,
                            zarr_path=zarr_path,
                            wavelength=channel_name,
                            epitope_tag=epitope_tag,
                            is_reference_round=is_reference
                        ))
                    else:
                        print(f"Warning: Expected epitope channel file not found: {zarr_path}")
            
            if channels:
                epitope_channels[round_name] = channels
                print(f"Found {len(channels)} epitope channels for round {round_name}")
        
        return epitope_channels
    
    def analyze_round_epitopes(
        self,
        round_name: str,
        epitope_channels: List[EpitopeChannel],
        nucleus_labels: Optional[List[int]] = None,
        max_nuclei: Optional[int] = None
    ) -> Dict[int, NucleusEpitopeData]:
        """
        Analyze epitope expression for a specific round.
        
        Args:
            round_name: Name of the round to analyze
            epitope_channels: List of epitope channels for this round
            nucleus_labels: Specific nucleus labels to analyze (None for all)
            max_nuclei: Maximum number of nuclei to process (overrides config)
            
        Returns:
            Dictionary mapping nucleus labels to epitope data
        """
        if not self.segmentation_mask_path or not self.segmentation_mask_path.exists():
            raise FileNotFoundError(f"Segmentation mask not found: {self.segmentation_mask_path}")
        
        # Load segmentation mask
        segmentation_mask = zarr.open(str(self.segmentation_mask_path), mode='r')
        
        # Get nucleus labels if not provided
        if nucleus_labels is None:
            unique_labels = np.unique(segmentation_mask[:])
            nucleus_labels = [int(label) for label in unique_labels if label > 0]
        
        # Use configured max_nuclei if not overridden
        if max_nuclei is None:
            max_nuclei = self.max_nuclei_per_round
        
        # Limit number of nuclei if specified
        if max_nuclei and len(nucleus_labels) > max_nuclei:
            nucleus_labels = nucleus_labels[:max_nuclei]
            print(f"Limited analysis to {max_nuclei} nuclei (from config: {self.max_nuclei_per_round})")
        
        # Initialize extractor with configuration parameters
        if self.extractor is None:
            self.extractor = EpitopeIntensityExtractor(
                shell_config=self.shell_config,
                aligned_data_dir=self.aligned_dir,
                zarr_volumes_dir=self.zarr_volumes_dir,
                reference_round=self.reference_round,
                chunk_size=self.chunk_size,
                memory_limit_gb=self.memory_limit_gb
            )
        
        # Extract intensities
        print(f"Analyzing {len(nucleus_labels)} nuclei for round {round_name}")
        print(f"Using epitope channels: {[ch.name for ch in epitope_channels]}")
        results = self.extractor.batch_extract_intensities(
            nucleus_labels=nucleus_labels,
            epitope_channels=epitope_channels,
            segmentation_mask=segmentation_mask,
            shell_params=self.shell_config
        )
        
        return results
    
    def save_analysis_results(
        self,
        results: Dict[str, Any],
        output_dir: Optional[Path] = None,
        round_name: str = "unknown"
    ) -> Dict[str, str]:
        """
        Save analysis results in configured formats.
        
        Args:
            results: Analysis results dictionary
            output_dir: Output directory (uses config default if None)
            round_name: Name of the round
            
        Returns:
            Dictionary of saved file paths
        """
        # Use configured output directory if not specified
        if output_dir is None:
            output_dir = self.pipeline_output_dir / self.output_subdir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save in configured formats
        for format_type in self.save_formats:
            if format_type == "json":
                json_path = output_dir / f"{round_name}_epitope_analysis.json"
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                saved_files["json"] = str(json_path)
            
            elif format_type == "csv":
                # Save intensity data as CSV
                csv_path = output_dir / f"{round_name}_epitope_intensities.csv"
                self._save_as_csv(results, csv_path)
                saved_files["csv"] = str(csv_path)
            
            elif format_type == "hdf5":
                # Save as HDF5 (requires h5py)
                try:
                    import h5py
                    hdf5_path = output_dir / f"{round_name}_epitope_analysis.h5"
                    self._save_as_hdf5(results, hdf5_path)
                    saved_files["hdf5"] = str(hdf5_path)
                except ImportError:
                    print("Warning: h5py not available, skipping HDF5 format")
        
        print(f"Analysis results saved to: {output_dir}")
        for file_type, file_path in saved_files.items():
            print(f"  - {file_type}: {file_path}")
        
        return saved_files
    
    def _save_as_csv(self, results: Dict[str, Any], csv_path: Path):
        """Save intensity data as CSV format."""
        try:
            import pandas as pd
            
            # Extract intensity data if available
            intensity_data = results.get("intensity_data", {})
            if not intensity_data:
                print("No intensity data found for CSV export")
                return
            
            # Convert to DataFrame format
            rows = []
            for nucleus_label, nucleus_data in intensity_data.items():
                if hasattr(nucleus_data, 'nuclei_intensities'):
                    row = {
                        'nucleus_label': nucleus_label,
                        'round_name': nucleus_data.round_name,
                        'centroid_z': nucleus_data.spatial_location[0],
                        'centroid_y': nucleus_data.spatial_location[1],
                        'centroid_x': nucleus_data.spatial_location[2],
                    }
                    
                    # Add intensity values for each channel and region
                    for channel, intensity in nucleus_data.nuclei_intensities.items():
                        row[f'{channel}_nuclei'] = intensity
                    for channel, intensity in nucleus_data.shell_intensities.items():
                        row[f'{channel}_shell'] = intensity
                    for channel, intensity in nucleus_data.combined_intensities.items():
                        row[f'{channel}_combined'] = intensity
                    
                    rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False)
                print(f"CSV data saved with {len(rows)} nuclei")
            
        except ImportError:
            print("Warning: pandas not available, skipping CSV format")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    def _save_as_hdf5(self, results: Dict[str, Any], hdf5_path: Path):
        """Save results as HDF5 format."""
        try:
            import h5py
            
            with h5py.File(hdf5_path, 'w') as f:
                # Save metadata
                metadata = results.get("metadata", {})
                if metadata:
                    meta_group = f.create_group("metadata")
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float)):
                            meta_group.attrs[key] = value
                
                # Save intensity data
                intensity_data = results.get("intensity_data", {})
                if intensity_data:
                    intensity_group = f.create_group("intensity_data")
                    
                    for nucleus_label, nucleus_data in intensity_data.items():
                        if hasattr(nucleus_data, 'nuclei_intensities'):
                            nucleus_group = intensity_group.create_group(str(nucleus_label))
                            
                            # Save intensities as datasets
                            for region in ['nuclei', 'shell', 'combined']:
                                region_intensities = getattr(nucleus_data, f'{region}_intensities', {})
                                if region_intensities:
                                    region_group = nucleus_group.create_group(region)
                                    for channel, intensity in region_intensities.items():
                                        region_group.create_dataset(channel, data=intensity)
                            
                            # Save spatial location
                            nucleus_group.create_dataset('centroid', data=nucleus_data.spatial_location)
            
            print(f"HDF5 data saved with {len(intensity_data)} nuclei")
            
        except Exception as e:
            print(f"Error saving HDF5: {e}")
