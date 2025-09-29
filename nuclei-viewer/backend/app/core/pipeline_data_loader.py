"""
Pipeline data loader for nuclei-viewer with VolAlign pipeline integration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import zarr
from skimage.measure import regionprops

from .config import AppConfig, PipelineDataSourceConfig
from .data_loader import DataLoader, DatasetInfo, load_zarr_as_dask
from .epitope_parser import EpitopeAnalysisParser, load_epitope_analysis
from .models import EnhancedNucleusInfo, PipelineDataset, PipelineMetadata
from .validation import validate_pipeline_config, ValidationError, PipelineValidationResult

logger = logging.getLogger(__name__)


class PipelineDataLoader(DataLoader):
    """
    Data loader for VolAlign pipeline outputs.
    
    Extends the base DataLoader to support pipeline-based data sources
    with pre-computed epitope analysis results.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize pipeline data loader.
        
        Args:
            config: Application configuration with pipeline settings
        """
        super().__init__(config)
        
        if not config.data.is_pipeline_mode():
            raise ValueError("PipelineDataLoader requires pipeline configuration")
        
        self.pipeline_config = config.data.pipeline
        self.pipeline_dataset: Optional[PipelineDataset] = None
        self.epitope_parser: Optional[EpitopeAnalysisParser] = None
        
        # Override nuclei list with enhanced version
        self.nuclei: List[EnhancedNucleusInfo] = []
        self._nuclei_loaded = False
        
        # Initialize pipeline dataset
        self._initialize_pipeline_dataset()
    
    def _initialize_pipeline_dataset(self):
        """Initialize the pipeline dataset structure."""
        working_dir = Path(self.pipeline_config.pipeline_working_directory)
        
        if not working_dir.exists():
            raise FileNotFoundError(f"Pipeline working directory not found: {working_dir}")
        
        # Set up paths
        epitope_analysis_file = None
        if self.pipeline_config.epitope_analysis_file:
            epitope_analysis_file = working_dir / self.pipeline_config.epitope_analysis_file
        
        segmentation_file = None
        if self.pipeline_config.segmentation_file:
            segmentation_file = working_dir / self.pipeline_config.segmentation_file
        else:
            # Auto-detect segmentation file
            segmentation_dir = working_dir / "segmentation"
            if segmentation_dir.exists():
                # Try fullres first, then regular
                fullres_file = segmentation_dir / f"{self.pipeline_config.reference_round}_nuclei_segmentation_fullres.zarr"
                regular_file = segmentation_dir / f"{self.pipeline_config.reference_round}_nuclei_segmentation.zarr"
                
                if fullres_file.exists():
                    segmentation_file = fullres_file
                elif regular_file.exists():
                    segmentation_file = regular_file
        
        zarr_volumes_dir = None
        if self.pipeline_config.zarr_volumes_dir:
            zarr_volumes_dir = working_dir / self.pipeline_config.zarr_volumes_dir
        
        aligned_dir = None
        if self.pipeline_config.aligned_dir:
            aligned_dir = working_dir / self.pipeline_config.aligned_dir
        
        # Create pipeline dataset
        self.pipeline_dataset = PipelineDataset(
            working_directory=working_dir,
            reference_round=self.pipeline_config.reference_round,
            epitope_analysis_file=epitope_analysis_file,
            segmentation_file=segmentation_file,
            zarr_volumes_dir=zarr_volumes_dir,
            aligned_dir=aligned_dir
        )
        
        logger.info(f"Initialized pipeline dataset:")
        logger.info(f"  - Working directory: {working_dir}")
        logger.info(f"  - Reference round: {self.pipeline_config.reference_round}")
        logger.info(f"  - Epitope analysis: {epitope_analysis_file}")
        logger.info(f"  - Segmentation: {segmentation_file}")
    
    def discover_datasets(self) -> Dict[str, List[Path]]:
        """
        Discover available datasets from pipeline structure.
        
        Returns:
            Dict mapping dataset type to list of found files
        """
        if not self.pipeline_dataset:
            raise RuntimeError("Pipeline dataset not initialized")
        
        discovered = {}
        
        # Discover segmentation
        if self.pipeline_dataset.segmentation_file and self.pipeline_dataset.segmentation_file.exists():
            discovered["segmentation"] = [self.pipeline_dataset.segmentation_file]
        else:
            discovered["segmentation"] = []
        
        # Discover zarr volumes and aligned data
        rounds_discovered = {}
        
        # Reference round data
        if self.pipeline_dataset.zarr_volumes_dir:
            ref_round_dir = self.pipeline_dataset.zarr_volumes_dir / self.pipeline_dataset.reference_round
            if ref_round_dir.exists():
                zarr_files = list(ref_round_dir.glob("*.zarr"))
                rounds_discovered[self.pipeline_dataset.reference_round] = zarr_files
                discovered[f"round_{self.pipeline_dataset.reference_round}"] = zarr_files
        
        # Aligned rounds data
        if self.pipeline_dataset.aligned_dir:
            for round_dir in self.pipeline_dataset.aligned_dir.iterdir():
                if round_dir.is_dir():
                    round_name = round_dir.name
                    zarr_files = list(round_dir.glob("*_aligned.zarr"))
                    if zarr_files:
                        rounds_discovered[round_name] = zarr_files
                        discovered[f"round_{round_name}"] = zarr_files
        
        # Update pipeline dataset with discovery results
        self.pipeline_dataset.discovered_zarr_files = discovered
        self.pipeline_dataset.available_rounds = list(rounds_discovered.keys())
        
        # Extract available channels per round
        channels_by_round = {}
        for round_name, zarr_files in rounds_discovered.items():
            channels = []
            for zarr_file in zarr_files:
                # Extract channel from filename
                filename = zarr_file.stem
                if round_name == self.pipeline_dataset.reference_round:
                    # Format: {round}_{channel}.zarr
                    if filename.startswith(f"{round_name}_"):
                        channel = filename[len(f"{round_name}_"):]
                        channels.append(channel)
                else:
                    # Format: {round}_{channel}_aligned.zarr
                    if filename.endswith("_aligned") and filename.startswith(f"{round_name}_"):
                        channel = filename[len(f"{round_name}_"):-len("_aligned")]
                        channels.append(channel)
            
            channels_by_round[round_name] = sorted(channels)
        
        self.pipeline_dataset.available_channels = channels_by_round
        
        logger.info(f"Discovered pipeline datasets:")
        for dataset_type, files in discovered.items():
            logger.info(f"  - {dataset_type}: {len(files)} files")
        
        return discovered
    
    def load_epitope_analysis(self) -> Tuple[PipelineMetadata, Dict[str, float], List[EnhancedNucleusInfo]]:
        """
        Load epitope analysis results from the pipeline.
        
        Returns:
            Tuple of (metadata, cutoffs, nuclei_with_analysis)
        """
        if not self.pipeline_dataset or not self.pipeline_dataset.epitope_analysis_file:
            raise RuntimeError("No epitope analysis file configured")
        
        if not self.pipeline_dataset.epitope_analysis_file.exists():
            raise FileNotFoundError(f"Epitope analysis file not found: {self.pipeline_dataset.epitope_analysis_file}")
        
        # Initialize parser if needed
        if not self.epitope_parser:
            self.epitope_parser = EpitopeAnalysisParser(self.pipeline_dataset.epitope_analysis_file)
        
        # Load and parse analysis data
        metadata, cutoffs, nuclei = load_epitope_analysis(self.pipeline_dataset.epitope_analysis_file)
        
        # Update pipeline dataset
        self.pipeline_dataset.metadata = metadata
        self.pipeline_dataset.cutoffs = cutoffs
        self.pipeline_dataset.nuclei = nuclei
        
        logger.info(f"Loaded epitope analysis:")
        logger.info(f"  - Nuclei: {len(nuclei)}")
        logger.info(f"  - Rounds: {metadata.n_rounds}")
        logger.info(f"  - Cutoffs: {len(cutoffs)}")
        logger.info(f"  - Analysis region: {metadata.analysis_region}")
        
        return metadata, cutoffs, nuclei
    
    def load_segmentation(self, file_path: Optional[Path] = None, array_key: Optional[str] = None) -> DatasetInfo:
        """
        Load segmentation dataset from pipeline.
        
        Args:
            file_path: Override segmentation file path
            array_key: Array key for zarr groups
            
        Returns:
            DatasetInfo for segmentation
        """
        if file_path is None:
            if not self.pipeline_dataset or not self.pipeline_dataset.segmentation_file:
                raise FileNotFoundError("No segmentation file found in pipeline")
            file_path = self.pipeline_dataset.segmentation_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {file_path}")
        
        dask_array = load_zarr_as_dask(file_path, array_key)
        
        self.segmentation = DatasetInfo(
            name="segmentation",
            path=file_path,
            shape=dask_array.shape,
            dtype=str(dask_array.dtype),
            array_key=array_key,
        )
        
        logger.info(f"Loaded pipeline segmentation: {self.segmentation}")
        return self.segmentation
    
    def load_pipeline_channel(self, round_name: str, channel_name: str) -> DatasetInfo:
        """
        Load a specific channel from the pipeline structure.
        
        Args:
            round_name: Name of the imaging round
            channel_name: Name of the channel
            
        Returns:
            DatasetInfo for the channel
        """
        if not self.pipeline_dataset:
            raise RuntimeError("Pipeline dataset not initialized")
        
        channel_path = self.pipeline_dataset.get_channel_path(round_name, channel_name)
        if not channel_path or not channel_path.exists():
            raise FileNotFoundError(f"Channel file not found: {round_name}_{channel_name}")
        
        dask_array = load_zarr_as_dask(channel_path, None)
        
        dataset_info = DatasetInfo(
            name=f"{round_name}_{channel_name}",
            path=channel_path,
            shape=dask_array.shape,
            dtype=str(dask_array.dtype),
            array_key=None,
        )
        
        logger.debug(f"Loaded pipeline channel: {dataset_info}")
        return dataset_info
    
    def load_all_pipeline_channels(self) -> Dict[str, DatasetInfo]:
        """
        Load all available channels from the pipeline.
        
        Returns:
            Dictionary mapping channel names to DatasetInfo
        """
        if not self.pipeline_dataset:
            raise RuntimeError("Pipeline dataset not initialized")
        
        # Discover datasets if not already done
        if not self.pipeline_dataset.available_channels:
            self.discover_datasets()
        
        all_channels = {}
        
        for round_name, channels in self.pipeline_dataset.available_channels.items():
            for channel_name in channels:
                try:
                    dataset_info = self.load_pipeline_channel(round_name, channel_name)
                    all_channels[f"{round_name}_{channel_name}"] = dataset_info
                except Exception as e:
                    logger.error(f"Failed to load channel {round_name}_{channel_name}: {e}")
                    continue
        
        logger.info(f"Loaded {len(all_channels)} pipeline channels")
        return all_channels
    
    def load_all_datasets(self) -> None:
        """
        Load all datasets using pipeline discovery.
        
        Overrides the base class method to use pipeline-specific loading.
        """
        logger.info("Loading all pipeline datasets...")
        
        # Discover datasets first
        discovered = self.discover_datasets()
        
        # Load segmentation
        try:
            self.load_segmentation()
            logger.info("✓ Segmentation loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load segmentation: {e}")
            raise
        
        # Load all available channels from pipeline
        try:
            all_channels = self.load_all_pipeline_channels()
            
            # Identify DAPI/nuclear channel - look for 405nm first (reference channel)
            dapi_channel_name = None
            ref_round = self.pipeline_dataset.reference_round
            
            # First, look for 405nm DAPI channel in reference round
            dapi_candidate = f"{ref_round}_405"
            if dapi_candidate in all_channels:
                dapi_channel_name = dapi_candidate
                self.dapi_channel = all_channels[dapi_candidate]
                logger.info(f"✓ DAPI channel identified: {dapi_candidate} (405nm reference channel)")
            else:
                # 405nm channel not found in discovered channels, try to load it directly
                # This can happen if the config doesn't include 405nm but the files exist
                working_dir = self.pipeline_dataset.working_directory
                dapi_full_path = working_dir / "zarr_volumes" / ref_round / f"{ref_round}_405.zarr"
                
                if dapi_full_path.exists():
                    # Load the full resolution 405nm channel directly
                    try:
                        dask_array = load_zarr_as_dask(dapi_full_path, None)
                        dapi_dataset_info = DatasetInfo(
                            name=f"{ref_round}_405",
                            path=dapi_full_path,
                            shape=dask_array.shape,
                            dtype=str(dask_array.dtype),
                            array_key=None,
                        )
                        dapi_channel_name = f"{ref_round}_405"
                        self.dapi_channel = dapi_dataset_info
                        logger.info(f"✓ DAPI channel identified: {dapi_channel_name} (405nm full resolution - direct load)")
                    except Exception as e:
                        logger.error(f"Failed to load 405nm channel directly: {e}")
                
                # If still no DAPI channel found, look for other nuclear indicators
                if not dapi_channel_name:
                    for channel_name, dataset_info in all_channels.items():
                        if any(dapi_indicator in channel_name.lower() for dapi_indicator in ['dapi', 'hoechst']):
                            dapi_channel_name = channel_name
                            self.dapi_channel = dataset_info
                            logger.info(f"✓ DAPI channel identified: {channel_name} (nuclear indicator)")
                            break
                
                # Final fallback: use first channel from reference round as nuclear channel
                if not dapi_channel_name:
                    ref_channels = [name for name in all_channels.keys() if name.startswith(f"{ref_round}_")]
                    if ref_channels:
                        dapi_channel_name = sorted(ref_channels)[0]  # Use first alphabetically
                        self.dapi_channel = all_channels[dapi_channel_name]
                        logger.warning(f"⚠ No 405nm DAPI channel found, using first available: {dapi_channel_name}")
                    else:
                        logger.error("No channels found for reference round")
                        raise RuntimeError("No nuclear channel could be identified")
            
            # Load epitope channels - exclude only the DAPI channel (reference round 405nm)
            # Allow 405nm channels from other rounds as they may contain epitope signals
            self.epitope_channels = {}
            for channel_name, dataset_info in all_channels.items():
                # Skip only the identified DAPI channel, but allow other 405nm channels
                if channel_name != dapi_channel_name:
                    self.epitope_channels[channel_name] = dataset_info
            
            logger.info(f"✓ Loaded {len(self.epitope_channels)} epitope channels")
            logger.info(f"✓ Pipeline datasets loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline channels: {e}")
            raise
    
    def extract_nuclei_info(self, force_reload: bool = False) -> List[EnhancedNucleusInfo]:
        """
        Extract nuclei information with epitope analysis integration.
        
        Args:
            force_reload: Force re-extraction even if already loaded
            
        Returns:
            List of EnhancedNucleusInfo objects
        """
        if self._nuclei_loaded and not force_reload:
            return self.nuclei
        
        logger.info("Extracting nuclei information from pipeline...")
        
        # Load epitope analysis first
        try:
            metadata, cutoffs, nuclei_with_analysis = self.load_epitope_analysis()
            
            # Use nuclei from epitope analysis as primary source
            self.nuclei = nuclei_with_analysis
            self._nuclei_loaded = True
            
            logger.info(f"Extracted {len(self.nuclei)} nuclei with epitope analysis")
            
        except Exception as e:
            logger.warning(f"Failed to load epitope analysis: {e}")
            logger.info("Falling back to segmentation-based nuclei extraction...")
            
            # Fallback to segmentation-based extraction
            self.nuclei = self._extract_nuclei_from_segmentation()
            self._nuclei_loaded = True
        
        return self.nuclei
    
    def _extract_nuclei_from_segmentation(self) -> List[EnhancedNucleusInfo]:
        """
        Extract nuclei from segmentation mask (fallback method).
        
        Returns:
            List of EnhancedNucleusInfo objects without epitope analysis
        """
        # Load segmentation
        if self.segmentation is None:
            self.load_segmentation()
        
        logger.info("Extracting nuclei from segmentation mask...")
        
        # Load segmentation mask
        seg_array = self.segmentation.get_dask_array()
        seg_np = seg_array.compute()
        
        # Extract region properties
        regions = regionprops(seg_np)
        
        nuclei = []
        min_size = self.config.processing.min_object_size
        
        for region in regions:
            if region.area >= min_size:
                nucleus_info = EnhancedNucleusInfo(
                    label=region.label,
                    bbox=region.bbox,
                    area=region.area,
                    centroid=region.centroid,
                )
                nuclei.append(nucleus_info)
        
        nuclei.sort(key=lambda x: x.label)
        
        logger.info(f"Extracted {len(nuclei)} nuclei from segmentation (min_size={min_size})")
        return nuclei
    
    def get_nucleus_by_label(self, label: int) -> Optional[EnhancedNucleusInfo]:
        """Get nucleus information by label."""
        if not self._nuclei_loaded:
            self.extract_nuclei_info()
        
        for nucleus in self.nuclei:
            if nucleus.label == label:
                return nucleus
        return None
    
    def get_epitope_cutoffs(self) -> Dict[str, float]:
        """
        Get epitope cutoffs from the pipeline analysis.
        
        Returns:
            Dictionary mapping round_channel keys to cutoff values
        """
        if self.pipeline_dataset and self.pipeline_dataset.cutoffs:
            return self.pipeline_dataset.cutoffs
        
        # Try to load from epitope analysis
        try:
            _, cutoffs, _ = self.load_epitope_analysis()
            return cutoffs
        except Exception as e:
            logger.error(f"Failed to load epitope cutoffs: {e}")
            return {}
    
    def get_pipeline_metadata(self) -> Optional[PipelineMetadata]:
        """
        Get pipeline metadata.
        
        Returns:
            PipelineMetadata or None if not available
        """
        if self.pipeline_dataset and self.pipeline_dataset.metadata:
            return self.pipeline_dataset.metadata
        
        # Try to load from epitope analysis
        try:
            metadata, _, _ = self.load_epitope_analysis()
            return metadata
        except Exception as e:
            logger.error(f"Failed to load pipeline metadata: {e}")
            return None
    
    def validate_pipeline_data(self) -> Dict[str, Any]:
        """
        Validate the pipeline data consistency.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'pipeline_info': {}
        }
        
        try:
            # Check pipeline dataset validity
            if not self.pipeline_dataset or not self.pipeline_dataset.is_valid():
                validation_results['is_valid'] = False
                validation_results['issues'].append("Pipeline dataset is not valid")
                return validation_results
            
            # Validate epitope analysis if available
            if self.epitope_parser:
                epitope_validation = self.epitope_parser.validate_data_consistency()
                validation_results['is_valid'] = validation_results['is_valid'] and epitope_validation['is_valid']
                validation_results['issues'].extend(epitope_validation['issues'])
                validation_results['warnings'].extend(epitope_validation['warnings'])
                validation_results['pipeline_info'].update(epitope_validation['statistics'])
            
            # Add pipeline structure info
            validation_results['pipeline_info'].update({
                'working_directory': str(self.pipeline_dataset.working_directory),
                'reference_round': self.pipeline_dataset.reference_round,
                'available_rounds': self.pipeline_dataset.available_rounds,
                'available_channels': self.pipeline_dataset.available_channels,
                'has_epitope_analysis': self.pipeline_dataset.epitope_analysis_file is not None,
                'has_segmentation': self.pipeline_dataset.segmentation_file is not None,
            })
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Validation error: {e}")
        
        return validation_results


def create_pipeline_data_loader(config: AppConfig) -> PipelineDataLoader:
    """
    Factory function to create a pipeline data loader.
    
    Args:
        config: Application configuration
        
    Returns:
        PipelineDataLoader instance
    """
    return PipelineDataLoader(config)