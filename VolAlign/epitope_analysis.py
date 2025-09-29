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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zarr
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import gaussian_kde, kurtosis, skew
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import ball
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

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
    """Complete epitope data for a single nucleus across all rounds and channels."""

    nucleus_label: int
    # Changed to support all rounds - key format: "round_name_channel_name"
    nuclei_intensities: Dict[
        str, float
    ]  # "round_channel" -> mean intensity within nuclei
    shell_intensities: Dict[
        str, float
    ]  # "round_channel" -> mean intensity in shell region
    combined_intensities: Dict[
        str, float
    ]  # "round_channel" -> combined nuclei + shell intensity
    channel_statistics: Dict[
        str, Dict[str, float]
    ]  # "round_channel" -> detailed stats for all regions
    spatial_location: Tuple[float, float, float]  # centroid (z, y, x)
    bounding_box: Tuple[
        int, int, int, int, int, int
    ]  # bbox (min_z, min_y, min_x, max_z, max_y, max_x)
    shell_parameters: Dict[str, Any]  # parameters used for shell generation
    quality_metrics: Dict[str, float]  # "round_channel" -> SNR, coverage, etc.
    rounds_processed: List[str]  # list of rounds that were processed for this nucleus


@dataclass
class EpitopeAnalysisResult:
    """Results of epitope analysis with cutoff determinations per round per channel."""

    nucleus_label: int
    epitope_calls: Dict[str, bool]  # "round_channel" -> positive/negative
    confidence_scores: Dict[str, float]  # "round_channel" -> confidence score
    intensity_values: Dict[str, float]  # "round_channel" -> intensity value
    cutoff_values: Dict[str, float]  # "round_channel" -> cutoff value used
    analysis_region: str  # "nuclei", "shell", or "combined"
    quality_score: float
    rounds_analyzed: List[str]  # list of rounds analyzed


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

    Supports processing of large Zarr volumes with integration with VolAlign's
    distributed processing framework.
    """

    def __init__(
        self,
        shell_config: ShellParameters,
        aligned_data_dir: Path,
        zarr_volumes_dir: Path,
        reference_round: str,
    ):
        """
        Initialize epitope intensity extractor.

        Args:
            shell_config: Parameters for shell region generation
            aligned_data_dir: Directory containing aligned round data
            zarr_volumes_dir: Directory containing reference round data
            reference_round: Name of reference round
        """
        self.shell_config = shell_config
        self.aligned_data_dir = Path(aligned_data_dir)
        self.zarr_volumes_dir = Path(zarr_volumes_dir)
        self.reference_round = reference_round

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
            return (
                self.aligned_data_dir
                / round_name
                / f"{round_name}_{channel}_aligned.zarr"
            )

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

            self._volume_cache[cache_key] = zarr.open(str(channel_path), mode="r")
            print(
                f"Loaded channel volume: {cache_key} - Shape: {self._volume_cache[cache_key].shape}"
            )

        return self._volume_cache[cache_key]

    def create_shell_mask(
        self, nucleus_mask: np.ndarray, shell_params: Optional[ShellParameters] = None
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
            # 3D morphology: expand in all directions (Z, Y, X)
            footprint = ball(shell_params.footprint_size)
        else:
            # 2D morphology: expand only in X-Y plane, preserve Z-thickness
            footprint = np.ones(
                (1, shell_params.footprint_size, shell_params.footprint_size)
            )

        # Erode nucleus to create inner boundary
        eroded_nucleus = binary_erosion(
            nucleus_mask,
            structure=footprint,
            iterations=shell_params.erosion_iterations,
        )

        # Dilate original nucleus to create outer boundary
        dilated_nucleus = binary_dilation(
            nucleus_mask,
            structure=footprint,
            iterations=shell_params.dilation_iterations,
        )

        # Create shell mask: dilated - original nucleus
        shell_mask = dilated_nucleus & ~nucleus_mask

        # Combined mask: original nucleus + shell
        combined_mask = nucleus_mask | shell_mask

        return nucleus_mask, shell_mask, combined_mask


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

    def compute_intensity_distributions_per_round_channel(
        self,
        intensity_data: Dict[int, NucleusEpitopeData],
        round_name: str,
        channel: str,
        region: str = "combined",
    ) -> DistributionAnalysis:
        """
        Compute comprehensive distribution analysis for a specific round-channel combination.

        Args:
            intensity_data: Dictionary of nucleus epitope data
            round_name: Round name to analyze
            channel: Channel name to analyze
            region: Region to analyze ("nuclei", "shell", "combined")

        Returns:
            DistributionAnalysis with comprehensive statistics
        """
        # Create round_channel key
        round_channel_key = f"{round_name}_{channel}"

        # Extract intensities for the specified round-channel and region
        if region == "nuclei":
            intensities = [
                data.nuclei_intensities.get(round_channel_key, 0.0)
                for data in intensity_data.values()
            ]
        elif region == "shell":
            intensities = [
                data.shell_intensities.get(round_channel_key, 0.0)
                for data in intensity_data.values()
            ]
        elif region == "combined":
            intensities = [
                data.combined_intensities.get(round_channel_key, 0.0)
                for data in intensity_data.values()
            ]
        else:
            raise ValueError(f"Unknown region: {region}")

        intensities = np.array(intensities)
        intensities = intensities[intensities > 0]  # Remove zero values

        if len(intensities) == 0:
            raise ValueError(
                f"No valid intensities found for round {round_name}, channel {channel}, region {region}"
            )

        # Basic statistics
        mean_val = float(np.mean(intensities))
        std_val = float(np.std(intensities))
        median_val = float(np.median(intensities))
        skewness = float(skew(intensities))
        kurtosis_val = float(kurtosis(intensities))

        # Percentiles
        percentiles = {
            "p5": float(np.percentile(intensities, 5)),
            "p10": float(np.percentile(intensities, 10)),
            "p25": float(np.percentile(intensities, 25)),
            "p75": float(np.percentile(intensities, 75)),
            "p90": float(np.percentile(intensities, 90)),
            "p95": float(np.percentile(intensities, 95)),
            "p99": float(np.percentile(intensities, 99)),
        }

        # Bimodality detection
        is_bimodal, bimodal_confidence, gmm_components = (
            self.detect_bimodal_distribution(intensities)
        )

        return DistributionAnalysis(
            channel=f"{round_name}_{channel}",  # Use round_channel format for consistency
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
            gmm_components=gmm_components,
        )

    def compute_all_distributions(
        self,
        intensity_data: Dict[int, NucleusEpitopeData],
        all_epitope_channels: Dict[str, List[EpitopeChannel]],
        region: str = "combined",
    ) -> Dict[str, DistributionAnalysis]:
        """
        Compute distribution analysis for all round-channel combinations.

        Args:
            intensity_data: Dictionary of nucleus epitope data
            all_epitope_channels: Dictionary mapping round names to lists of epitope channels
            region: Region to analyze ("nuclei", "shell", "combined")

        Returns:
            Dictionary mapping round_channel keys to DistributionAnalysis
        """
        distributions = {}

        for round_name, epitope_channels in all_epitope_channels.items():
            for channel in epitope_channels:
                round_channel_key = f"{round_name}_{channel.name}"
                try:
                    distribution = (
                        self.compute_intensity_distributions_per_round_channel(
                            intensity_data=intensity_data,
                            round_name=round_name,
                            channel=channel.name,
                            region=region,
                        )
                    )
                    distributions[round_channel_key] = distribution
                except ValueError as e:
                    print(
                        f"Warning: Could not compute distribution for {round_channel_key}: {e}"
                    )
                    continue

        return distributions

    def detect_bimodal_distribution(
        self, intensities: np.ndarray
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
                    "means": means.tolist(),
                    "stds": stds.tolist(),
                    "weights": weights.tolist(),
                    "separation": float(separation),
                    "bic_improvement": float(bic_improvement),
                }
            else:
                gmm_components = None

            return is_bimodal, float(confidence), gmm_components

        except Exception as e:
            print(f"Error in bimodal detection: {e}")
            return False, 0.0, None


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
    def gmm_cutoff(
        intensities: np.ndarray, region: str = "combined"
    ) -> Dict[str, float]:
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
            is_bimodal, confidence, gmm_components = (
                analyzer.detect_bimodal_distribution(intensities)
            )

            if is_bimodal and gmm_components:
                means = gmm_components["means"]
                cutoff = (means[0] + means[1]) / 2.0
                return {
                    "cutoff": float(cutoff),
                    "confidence": confidence,
                    "component_means": means,
                    "separation": gmm_components["separation"],
                }
            else:
                # Fallback to percentile-based cutoff
                return {
                    "cutoff": float(np.percentile(intensities, 75)),
                    "confidence": 0.0,
                    "note": "No bimodal distribution detected",
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
        self, config: Dict[str, Any], pipeline_output_dir: Optional[Path] = None
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
            use_3d_morphology=shell_params.get("use_3d_morphology", True),
        )

        # Epitope channels configuration
        epitope_channels_config = self.epitope_config.get("epitope_channels", {})
        self.epitope_channel_names = epitope_channels_config.get("channels", [])
        self.channel_tags = epitope_channels_config.get("channel_tags", {})

        # Analysis parameters
        analysis_params = self.epitope_config.get("analysis_parameters", {})
        self.max_nuclei_per_round = analysis_params.get("max_nuclei_per_round")
        self.default_region = analysis_params.get("default_region", "combined")

        # Statistical analysis configuration
        self.statistical_config = self.epitope_config.get("statistical_analysis", {})

        # Output configuration
        output_config = self.epitope_config.get("output", {})
        self.output_subdir = output_config.get("output_subdir", "epitope_analysis")
        self.save_formats = output_config.get("save_formats", ["json"])
        self.save_detailed_stats = output_config.get("save_detailed_stats", True)

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
                self.segmentation_dir
                / f"{self.reference_round}_nuclei_segmentation_fullres.zarr"
            )
        else:
            self.segmentation_mask_path = (
                self.segmentation_dir
                / f"{self.reference_round}_nuclei_segmentation.zarr"
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
                        zarr_path = (
                            self.zarr_volumes_dir
                            / round_name
                            / f"{round_name}_{channel_name}.zarr"
                        )
                        is_reference = True
                    else:
                        zarr_path = (
                            self.aligned_dir
                            / round_name
                            / f"{round_name}_{channel_name}_aligned.zarr"
                        )
                        is_reference = False

                    # Check if file exists
                    if zarr_path.exists():
                        epitope_tag = self.channel_tags.get(channel_name, None)

                        channels.append(
                            EpitopeChannel(
                                name=channel_name,
                                round_name=round_name,
                                zarr_path=zarr_path,
                                wavelength=channel_name,
                                epitope_tag=epitope_tag,
                                is_reference_round=is_reference,
                            )
                        )
                    else:
                        print(
                            f"Warning: Expected epitope channel file not found: {zarr_path}"
                        )

            if channels:
                epitope_channels[round_name] = channels
                print(f"Found {len(channels)} epitope channels for round {round_name}")

        return epitope_channels

    def extract_nucleus_intensities_all_rounds_with_bbox(
        self,
        nucleus_label: int,
        all_epitope_channels: Dict[str, List[EpitopeChannel]],
        nucleus_region: Any,  # regionprops region object
        segmentation_mask: zarr.Array,
        shell_params: Optional[ShellParameters] = None,
    ) -> NucleusEpitopeData:
        """
        Extract intensities from all rounds and channels for a single nucleus using bounding box optimization.

        Args:
            nucleus_label: Label ID of the nucleus
            all_epitope_channels: Dictionary mapping round names to lists of epitope channels
            nucleus_region: regionprops region object for this nucleus
            segmentation_mask: Segmentation mask Zarr array
            shell_params: Shell generation parameters

        Returns:
            NucleusEpitopeData with intensities for all rounds and channels
        """
        if shell_params is None:
            shell_params = self.shell_config

        # Initialize extractor if needed
        if self.extractor is None:
            self.extractor = EpitopeIntensityExtractor(
                shell_config=self.shell_config,
                aligned_data_dir=self.aligned_dir,
                zarr_volumes_dir=self.zarr_volumes_dir,
                reference_round=self.reference_round,
            )

        # Get bounding box from regionprops
        min_z, min_y, min_x, max_z, max_y, max_x = nucleus_region.bbox

        # Calculate padding needed for shell region
        pad_xy = shell_params.dilation_iterations * shell_params.footprint_size
        pad_z = pad_xy if shell_params.use_3d_morphology else 0

        # Expand the bounding box with padding, ensuring we stay within volume bounds
        volume_shape = segmentation_mask.shape
        min_z_context = max(0, min_z - pad_z)
        max_z_context = min(volume_shape[0], max_z + pad_z)
        min_y_context = max(0, min_y - pad_xy)
        max_y_context = min(volume_shape[1], max_y + pad_xy)
        min_x_context = max(0, min_x - pad_xy)
        max_x_context = min(volume_shape[2], max_x + pad_xy)

        # Crop the nucleus mask from segmentation using bounding box
        crop_mask = (
            segmentation_mask[
                min_z_context:max_z_context,
                min_y_context:max_y_context,
                min_x_context:max_x_context,
            ]
            == nucleus_label
        ).astype(bool)

        if not np.any(crop_mask):
            raise ValueError(
                f"Nucleus label {nucleus_label} not found in cropped region"
            )

        # Create shell masks on cropped region
        nucleus_mask_clean, shell_mask, combined_mask = (
            self.extractor.create_shell_mask(crop_mask, shell_params)
        )

        # Get centroid and bounding box from regionprops (already in global coordinates)
        centroid = (
            float(nucleus_region.centroid[0]),  # z
            float(nucleus_region.centroid[1]),  # y
            float(nucleus_region.centroid[2]),  # x
        )

        # Store bounding box information (min_z, min_y, min_x, max_z, max_y, max_x)
        bbox = (int(min_z), int(min_y), int(min_x), int(max_z), int(max_y), int(max_x))

        # Initialize intensity dictionaries with round_channel keys
        nuclei_intensities = {}
        shell_intensities = {}
        combined_intensities = {}
        channel_statistics = {}
        quality_metrics = {}
        rounds_processed = []

        # Process each round and its epitope channels
        for round_name, epitope_channels in all_epitope_channels.items():
            rounds_processed.append(round_name)

            for channel in epitope_channels:
                try:
                    # Load channel volume
                    volume = self.extractor.load_channel_volume(
                        channel.round_name, channel.name
                    )

                    # Create round_channel key
                    round_channel_key = f"{round_name}_{channel.name}"

                    # Crop the channel volume using bounding box
                    crop_volume = volume[
                        min_z_context:max_z_context,
                        min_y_context:max_y_context,
                        min_x_context:max_x_context,
                    ]

                    # Extract intensities for each region from cropped volume
                    nuclei_values = crop_volume[nucleus_mask_clean]
                    shell_values = (
                        crop_volume[shell_mask] if np.any(shell_mask) else np.array([])
                    )
                    combined_values = crop_volume[combined_mask]

                    # Calculate mean intensities
                    nuclei_intensities[round_channel_key] = (
                        float(np.mean(nuclei_values)) if len(nuclei_values) > 0 else 0.0
                    )
                    shell_intensities[round_channel_key] = (
                        float(np.mean(shell_values)) if len(shell_values) > 0 else 0.0
                    )
                    combined_intensities[round_channel_key] = (
                        float(np.mean(combined_values))
                        if len(combined_values) > 0
                        else 0.0
                    )

                    # Calculate detailed statistics
                    channel_statistics[round_channel_key] = {
                        "nuclei": {
                            "mean": nuclei_intensities[round_channel_key],
                            "std": (
                                float(np.std(nuclei_values))
                                if len(nuclei_values) > 0
                                else 0.0
                            ),
                            "median": (
                                float(np.median(nuclei_values))
                                if len(nuclei_values) > 0
                                else 0.0
                            ),
                            "p25": (
                                float(np.percentile(nuclei_values, 25))
                                if len(nuclei_values) > 0
                                else 0.0
                            ),
                            "p75": (
                                float(np.percentile(nuclei_values, 75))
                                if len(nuclei_values) > 0
                                else 0.0
                            ),
                            "n_voxels": len(nuclei_values),
                        },
                        "shell": {
                            "mean": shell_intensities[round_channel_key],
                            "std": (
                                float(np.std(shell_values))
                                if len(shell_values) > 0
                                else 0.0
                            ),
                            "median": (
                                float(np.median(shell_values))
                                if len(shell_values) > 0
                                else 0.0
                            ),
                            "p25": (
                                float(np.percentile(shell_values, 25))
                                if len(shell_values) > 0
                                else 0.0
                            ),
                            "p75": (
                                float(np.percentile(shell_values, 75))
                                if len(shell_values) > 0
                                else 0.0
                            ),
                            "n_voxels": len(shell_values),
                        },
                        "combined": {
                            "mean": combined_intensities[round_channel_key],
                            "std": (
                                float(np.std(combined_values))
                                if len(combined_values) > 0
                                else 0.0
                            ),
                            "median": (
                                float(np.median(combined_values))
                                if len(combined_values) > 0
                                else 0.0
                            ),
                            "p25": (
                                float(np.percentile(combined_values, 25))
                                if len(combined_values) > 0
                                else 0.0
                            ),
                            "p75": (
                                float(np.percentile(combined_values, 75))
                                if len(combined_values) > 0
                                else 0.0
                            ),
                            "n_voxels": len(combined_values),
                        },
                    }

                    # Calculate quality metrics
                    snr_nuclei = nuclei_intensities[round_channel_key] / (
                        channel_statistics[round_channel_key]["nuclei"]["std"] + 1e-6
                    )
                    coverage_ratio = len(nuclei_values) / np.sum(crop_mask)

                    quality_metrics[round_channel_key] = {
                        "snr_nuclei": snr_nuclei,
                        "coverage_ratio": coverage_ratio,
                        "shell_to_nuclei_ratio": (
                            shell_intensities[round_channel_key]
                            / (nuclei_intensities[round_channel_key] + 1e-6)
                        ),
                    }

                except Exception as e:
                    print(
                        f"Error processing channel {channel.name} in round {round_name}: {e}"
                    )
                    # Set default values for failed channels
                    round_channel_key = f"{round_name}_{channel.name}"
                    nuclei_intensities[round_channel_key] = 0.0
                    shell_intensities[round_channel_key] = 0.0
                    combined_intensities[round_channel_key] = 0.0
                    channel_statistics[round_channel_key] = {}
                    quality_metrics[round_channel_key] = {
                        "snr_nuclei": 0.0,
                        "coverage_ratio": 0.0,
                    }

        return NucleusEpitopeData(
            nucleus_label=nucleus_label,
            nuclei_intensities=nuclei_intensities,
            shell_intensities=shell_intensities,
            combined_intensities=combined_intensities,
            channel_statistics=channel_statistics,
            spatial_location=centroid,
            bounding_box=bbox,
            shell_parameters=asdict(shell_params),
            quality_metrics=quality_metrics,
            rounds_processed=rounds_processed,
        )

    def batch_extract_intensities_all_rounds(
        self,
        nucleus_labels: List[int],
        all_epitope_channels: Dict[str, List[EpitopeChannel]],
        segmentation_mask: zarr.Array,
        shell_params: Optional[ShellParameters] = None,
    ) -> Dict[int, NucleusEpitopeData]:
        """
        Batch processing for multiple nuclei across all rounds using bounding box optimization.

        Args:
            nucleus_labels: List of nucleus label IDs to process
            all_epitope_channels: Dictionary mapping round names to lists of epitope channels
            segmentation_mask: Segmentation mask Zarr array
            shell_params: Shell generation parameters

        Returns:
            Dictionary mapping nucleus labels to epitope data
        """
        results = {}

        total_channels = sum(
            len(channels) for channels in all_epitope_channels.values()
        )
        print(
            f"Processing {len(nucleus_labels)} nuclei across {len(all_epitope_channels)} rounds with {total_channels} total channels..."
        )

        # Compute regionprops once for all nuclei - this is the key optimization
        print("Computing regionprops for all nuclei (one-time operation)...")
        labeled_volume = segmentation_mask[:]
        regions = regionprops(labeled_volume)

        # Create a lookup dictionary for fast access
        region_lookup = {region.label: region for region in regions}
        print(f"Found {len(region_lookup)} regions in segmentation mask")

        for nucleus_label in tqdm(
            nucleus_labels, desc="Extracting epitope intensities (all rounds)"
        ):
            try:
                # Get the region for this nucleus
                nucleus_region = region_lookup.get(nucleus_label)
                if nucleus_region is None:
                    print(f"Warning: Nucleus {nucleus_label} not found in regionprops")
                    continue

                nucleus_data = self.extract_nucleus_intensities_all_rounds_with_bbox(
                    nucleus_label=nucleus_label,
                    all_epitope_channels=all_epitope_channels,
                    nucleus_region=nucleus_region,
                    segmentation_mask=segmentation_mask,
                    shell_params=shell_params,
                )
                results[nucleus_label] = nucleus_data

            except Exception as e:
                print(f"Failed to process nucleus {nucleus_label}: {e}")
                continue

        print(f"Successfully processed {len(results)}/{len(nucleus_labels)} nuclei")
        return results

    def compute_per_round_channel_cutoffs(
        self,
        intensity_data: Dict[int, NucleusEpitopeData],
        all_epitope_channels: Dict[str, List[EpitopeChannel]],
        region: str = "combined",
        method: str = "otsu",
    ) -> Dict[str, float]:
        """
        Compute cutoffs for each round-channel combination separately.

        Args:
            intensity_data: Dictionary of nucleus epitope data
            all_epitope_channels: Dictionary mapping round names to lists of epitope channels
            region: Region to analyze ("nuclei", "shell", "combined")
            method: Cutoff determination method

        Returns:
            Dictionary mapping round_channel keys to cutoff values
        """
        cutoffs = {}

        for round_name, epitope_channels in all_epitope_channels.items():
            for channel in epitope_channels:
                round_channel_key = f"{round_name}_{channel.name}"

                try:
                    # Extract intensities for this specific round-channel combination
                    if region == "nuclei":
                        intensities = [
                            data.nuclei_intensities.get(round_channel_key, 0.0)
                            for data in intensity_data.values()
                        ]
                    elif region == "shell":
                        intensities = [
                            data.shell_intensities.get(round_channel_key, 0.0)
                            for data in intensity_data.values()
                        ]
                    elif region == "combined":
                        intensities = [
                            data.combined_intensities.get(round_channel_key, 0.0)
                            for data in intensity_data.values()
                        ]
                    else:
                        raise ValueError(f"Unknown region: {region}")

                    intensities = np.array(intensities)
                    intensities = intensities[intensities > 0]  # Remove zero values

                    if len(intensities) == 0:
                        print(f"Warning: No valid intensities for {round_channel_key}")
                        cutoffs[round_channel_key] = 0.0
                        continue

                    # Calculate cutoff using specified method
                    if method == "otsu":
                        cutoff = CutoffDetermination.otsu_threshold(
                            intensities, round_channel_key
                        )
                    elif method == "percentile":
                        cutoff = CutoffDetermination.percentile_cutoff(intensities, 75)
                    elif method == "gmm":
                        gmm_result = CutoffDetermination.gmm_cutoff(
                            intensities, round_channel_key
                        )
                        cutoff = gmm_result["cutoff"]
                    else:
                        # Default to Otsu
                        cutoff = CutoffDetermination.otsu_threshold(
                            intensities, round_channel_key
                        )

                    cutoffs[round_channel_key] = cutoff
                    print(
                        f"Computed cutoff for {round_channel_key}: {cutoff:.3f} (n={len(intensities)})"
                    )

                except Exception as e:
                    print(f"Error computing cutoff for {round_channel_key}: {e}")
                    cutoffs[round_channel_key] = 0.0

        return cutoffs

    def apply_cutoffs_to_nucleus(
        self,
        nucleus_data: NucleusEpitopeData,
        cutoffs: Dict[str, float],
        region: str = "combined",
    ) -> EpitopeAnalysisResult:
        """
        Apply cutoffs to determine epitope calls for a single nucleus.

        Args:
            nucleus_data: Nucleus epitope data
            cutoffs: Dictionary of cutoffs for each round_channel
            region: Region to use for analysis

        Returns:
            EpitopeAnalysisResult with epitope calls
        """
        # Get intensities for the specified region
        if region == "nuclei":
            intensities = nucleus_data.nuclei_intensities
        elif region == "shell":
            intensities = nucleus_data.shell_intensities
        elif region == "combined":
            intensities = nucleus_data.combined_intensities
        else:
            raise ValueError(f"Unknown region: {region}")

        epitope_calls = {}
        confidence_scores = {}
        intensity_values = {}
        cutoff_values = {}

        for round_channel_key, intensity in intensities.items():
            cutoff = cutoffs.get(round_channel_key, 0.0)

            # Make epitope call
            is_positive = intensity >= cutoff
            epitope_calls[round_channel_key] = is_positive

            # Calculate confidence score (distance from cutoff normalized by cutoff)
            if cutoff > 0:
                confidence = abs(intensity - cutoff) / cutoff
            else:
                confidence = 0.0

            confidence_scores[round_channel_key] = confidence
            intensity_values[round_channel_key] = intensity
            cutoff_values[round_channel_key] = cutoff

        # Calculate overall quality score (mean SNR across all channels)
        quality_scores = []
        for round_channel_key in intensities.keys():
            snr = nucleus_data.quality_metrics.get(round_channel_key, {}).get(
                "snr_nuclei", 0.0
            )
            quality_scores.append(snr)

        overall_quality = float(np.mean(quality_scores)) if quality_scores else 0.0

        return EpitopeAnalysisResult(
            nucleus_label=nucleus_data.nucleus_label,
            epitope_calls=epitope_calls,
            confidence_scores=confidence_scores,
            intensity_values=intensity_values,
            cutoff_values=cutoff_values,
            analysis_region=region,
            quality_score=overall_quality,
            rounds_analyzed=nucleus_data.rounds_processed,
        )

    def analyze_all_epitopes_nucleus_centric(
        self,
        nucleus_labels: Optional[List[int]] = None,
        max_nuclei: Optional[int] = None,
    ) -> Tuple[
        Dict[int, NucleusEpitopeData],
        Dict[str, float],
        Dict[int, EpitopeAnalysisResult],
    ]:
        """
        Perform nucleus-centric epitope analysis across all rounds and channels.

        Args:
            nucleus_labels: Specific nucleus labels to analyze (None for all)
            max_nuclei: Maximum number of nuclei to process

        Returns:
            Tuple of (intensity_data, cutoffs, analysis_results)
        """
        print("Starting nucleus-centric epitope analysis...")

        # Discover all epitope channels
        all_epitope_channels = self.discover_epitope_channels()
        if not all_epitope_channels:
            raise ValueError("No epitope channels found")

        # Load segmentation mask
        if not self.segmentation_mask_path or not self.segmentation_mask_path.exists():
            raise FileNotFoundError(
                f"Segmentation mask not found: {self.segmentation_mask_path}"
            )

        segmentation_mask = zarr.open(str(self.segmentation_mask_path), mode="r")

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
            print(f"Limited analysis to {max_nuclei} nuclei")

        # Step 1: Extract intensities for all nuclei across all rounds
        print("Step 1: Extracting intensities across all rounds...")
        intensity_data = self.batch_extract_intensities_all_rounds(
            nucleus_labels=nucleus_labels,
            all_epitope_channels=all_epitope_channels,
            segmentation_mask=segmentation_mask,
            shell_params=self.shell_config,
        )

        # Step 2: Compute per-round-per-channel cutoffs
        print("Step 2: Computing per-round-per-channel cutoffs...")
        cutoff_method = self.statistical_config.get("cutoff_method", "otsu")
        cutoffs = self.compute_per_round_channel_cutoffs(
            intensity_data=intensity_data,
            all_epitope_channels=all_epitope_channels,
            region=self.default_region,
            method=cutoff_method,
        )

        # Step 3: Apply cutoffs to determine epitope calls
        print("Step 3: Applying cutoffs to determine epitope calls...")
        analysis_results = {}
        for nucleus_label, nucleus_data in intensity_data.items():
            try:
                result = self.apply_cutoffs_to_nucleus(
                    nucleus_data=nucleus_data,
                    cutoffs=cutoffs,
                    region=self.default_region,
                )
                analysis_results[nucleus_label] = result
            except Exception as e:
                print(f"Error analyzing nucleus {nucleus_label}: {e}")
                continue

        print(f"Nucleus-centric analysis completed:")
        print(f"  - Processed {len(intensity_data)} nuclei")
        print(f"  - Computed {len(cutoffs)} round-channel cutoffs")
        print(f"  - Generated {len(analysis_results)} analysis results")

        return intensity_data, cutoffs, analysis_results

    def _json_serializer(self, obj):
        """Custom JSON serializer for dataclass objects and other non-serializable types."""
        if hasattr(obj, "__dataclass_fields__"):
            # Convert dataclass to dictionary
            return asdict(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return str(obj)

    def save_analysis_results(
        self,
        results: Dict[str, Any],
        output_dir: Optional[Path] = None,
        round_name: str = "unknown",
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
                with open(json_path, "w") as f:
                    json.dump(
                        results,
                        f,
                        indent=2,
                        separators=(",", ": "),
                        ensure_ascii=False,
                        default=self._json_serializer,
                    )
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
        """Save intensity data as CSV format with new round_channel structure."""
        try:
            import pandas as pd

            # Extract intensity data if available
            intensity_data = results.get("intensity_data", {})
            analysis_results = results.get("analysis_results", {})

            if not intensity_data:
                print("No intensity data found for CSV export")
                return

            # Convert to DataFrame format
            rows = []
            for nucleus_label, nucleus_data in intensity_data.items():
                if hasattr(nucleus_data, "nuclei_intensities"):
                    row = {
                        "nucleus_label": nucleus_label,
                        "centroid_z": nucleus_data.spatial_location[0],
                        "centroid_y": nucleus_data.spatial_location[1],
                        "centroid_x": nucleus_data.spatial_location[2],
                        "bbox_min_z": nucleus_data.bounding_box[0],
                        "bbox_min_y": nucleus_data.bounding_box[1],
                        "bbox_min_x": nucleus_data.bounding_box[2],
                        "bbox_max_z": nucleus_data.bounding_box[3],
                        "bbox_max_y": nucleus_data.bounding_box[4],
                        "bbox_max_x": nucleus_data.bounding_box[5],
                        "rounds_processed": ",".join(nucleus_data.rounds_processed),
                    }

                    # Add intensity values for each round_channel and region
                    for (
                        round_channel,
                        intensity,
                    ) in nucleus_data.nuclei_intensities.items():
                        row[f"{round_channel}_nuclei"] = intensity
                    for (
                        round_channel,
                        intensity,
                    ) in nucleus_data.shell_intensities.items():
                        row[f"{round_channel}_shell"] = intensity
                    for (
                        round_channel,
                        intensity,
                    ) in nucleus_data.combined_intensities.items():
                        row[f"{round_channel}_combined"] = intensity

                    # Add epitope calls if available
                    if nucleus_label in analysis_results:
                        analysis_result = analysis_results[nucleus_label]
                        for (
                            round_channel,
                            call,
                        ) in analysis_result.epitope_calls.items():
                            row[f"{round_channel}_positive"] = call
                            row[f"{round_channel}_cutoff"] = (
                                analysis_result.cutoff_values.get(round_channel, 0.0)
                            )
                            row[f"{round_channel}_confidence"] = (
                                analysis_result.confidence_scores.get(
                                    round_channel, 0.0
                                )
                            )

                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False)
                print(f"CSV data saved with {len(rows)} nuclei")

        except ImportError:
            print("Warning: pandas not available, skipping CSV format")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def run_complete_nucleus_centric_analysis(
        self,
        nucleus_labels: Optional[List[int]] = None,
        max_nuclei: Optional[int] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete nucleus-centric epitope analysis pipeline.

        Args:
            nucleus_labels: Specific nucleus labels to analyze (None for all)
            max_nuclei: Maximum number of nuclei to process
            save_results: Whether to save results to files

        Returns:
            Dictionary containing all analysis results
        """
        print("Starting nucleus-centric epitope analysis...")

        # Run the main analysis
        intensity_data, cutoffs, analysis_results = (
            self.analyze_all_epitopes_nucleus_centric(
                nucleus_labels=nucleus_labels, max_nuclei=max_nuclei
            )
        )

        # Compute distribution statistics if requested
        distributions = {}
        if self.save_detailed_stats:
            print("Computing distribution statistics...")
            all_epitope_channels = self.discover_epitope_channels()
            distributions = self.analyzer.compute_all_distributions(
                intensity_data=intensity_data,
                all_epitope_channels=all_epitope_channels,
                region=self.default_region,
            )

        # Compile results
        results = {
            "metadata": {
                "analysis_type": "nucleus_centric_epitope_analysis",
                "n_nuclei": len(intensity_data),
                "n_rounds": len(
                    set(
                        round_name
                        for nucleus_data in intensity_data.values()
                        for round_name in nucleus_data.rounds_processed
                    )
                ),
                "analysis_region": self.default_region,
                "cutoff_method": self.statistical_config.get("cutoff_method", "otsu"),
                "shell_parameters": asdict(self.shell_config),
            },
            "intensity_data": intensity_data,
            "cutoffs": cutoffs,
            "analysis_results": analysis_results,
            "distributions": distributions,
        }

        # Save results if requested
        if save_results:
            saved_files = self.save_analysis_results(
                results=results, round_name="nucleus_centric_analysis"
            )
            results["saved_files"] = saved_files

        print(f"Nucleus-centric analysis completed for {len(intensity_data)} nuclei")
        return results

    def _save_as_hdf5(self, results: Dict[str, Any], hdf5_path: Path):
        """Save results as HDF5 format."""
        try:
            import h5py

            with h5py.File(hdf5_path, "w") as f:
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
                        if hasattr(nucleus_data, "nuclei_intensities"):
                            nucleus_group = intensity_group.create_group(
                                str(nucleus_label)
                            )

                            # Save intensities as datasets
                            for region in ["nuclei", "shell", "combined"]:
                                region_intensities = getattr(
                                    nucleus_data, f"{region}_intensities", {}
                                )
                                if region_intensities:
                                    region_group = nucleus_group.create_group(region)
                                    for (
                                        channel,
                                        intensity,
                                    ) in region_intensities.items():
                                        region_group.create_dataset(
                                            channel, data=intensity
                                        )

                            # Save spatial location and bounding box
                            nucleus_group.create_dataset(
                                "centroid", data=nucleus_data.spatial_location
                            )
                            nucleus_group.create_dataset(
                                "bounding_box", data=nucleus_data.bounding_box
                            )

            print(f"HDF5 data saved with {len(intensity_data)} nuclei")

        except Exception as e:
            print(f"Error saving HDF5: {e}")
