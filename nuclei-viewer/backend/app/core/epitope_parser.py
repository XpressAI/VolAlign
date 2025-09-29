"""
Epitope analysis results parser for nuclei-viewer pipeline integration.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    EpitopeAnalysisData,
    EnhancedNucleusInfo,
    PipelineMetadata,
    ChannelStatistics,
    QualityMetrics,
    create_channel_statistics_from_dict,
    create_quality_metrics_from_dict,
)

logger = logging.getLogger(__name__)


class EpitopeAnalysisParser:
    """Parser for epitope analysis JSON results from VolAlign pipeline."""
    
    def __init__(self, analysis_file: Path):
        """
        Initialize parser with epitope analysis file.
        
        Args:
            analysis_file: Path to the epitope analysis JSON file
        """
        self.analysis_file = Path(analysis_file)
        self._raw_data: Optional[Dict[str, Any]] = None
        self._parsed_metadata: Optional[PipelineMetadata] = None
        self._parsed_nuclei: Optional[List[EnhancedNucleusInfo]] = None
        self._cutoffs: Optional[Dict[str, float]] = None
    
    def load_analysis_file(self) -> Dict[str, Any]:
        """
        Load and parse the epitope analysis JSON file.
        
        Returns:
            Raw analysis data dictionary
            
        Raises:
            FileNotFoundError: If analysis file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
            ValueError: If file structure is invalid
        """
        if not self.analysis_file.exists():
            raise FileNotFoundError(f"Epitope analysis file not found: {self.analysis_file}")
        
        try:
            with open(self.analysis_file, 'r') as f:
                self._raw_data = json.load(f)
            
            # Validate required sections
            required_sections = ['metadata', 'intensity_data', 'cutoffs', 'analysis_results']
            missing_sections = [section for section in required_sections if section not in self._raw_data]
            
            if missing_sections:
                raise ValueError(f"Missing required sections in analysis file: {missing_sections}")
            
            logger.info(f"Loaded epitope analysis file: {self.analysis_file}")
            logger.info(f"  - Nuclei: {len(self._raw_data.get('intensity_data', {}))}")
            logger.info(f"  - Cutoffs: {len(self._raw_data.get('cutoffs', {}))}")
            logger.info(f"  - Analysis results: {len(self._raw_data.get('analysis_results', {}))}")
            
            return self._raw_data
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in epitope analysis file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading epitope analysis file: {e}")
    
    def parse_metadata(self) -> PipelineMetadata:
        """
        Parse metadata from the analysis file.
        
        Returns:
            PipelineMetadata object
        """
        if self._raw_data is None:
            self.load_analysis_file()
        
        if self._parsed_metadata is not None:
            return self._parsed_metadata
        
        metadata_dict = self._raw_data.get('metadata', {})
        
        self._parsed_metadata = PipelineMetadata(
            analysis_type=metadata_dict.get('analysis_type', 'nucleus_centric_epitope_analysis'),
            n_nuclei=metadata_dict.get('n_nuclei', 0),
            n_rounds=metadata_dict.get('n_rounds', 0),
            analysis_region=metadata_dict.get('analysis_region', 'combined'),
            cutoff_method=metadata_dict.get('cutoff_method', 'otsu'),
            shell_parameters=metadata_dict.get('shell_parameters', {}),
            reference_round=metadata_dict.get('reference_round', ''),
            epitope_channels=self._extract_epitope_channels_from_cutoffs()
        )
        
        return self._parsed_metadata
    
    def parse_cutoffs(self) -> Dict[str, float]:
        """
        Parse cutoff values from the analysis file.
        
        Returns:
            Dictionary mapping round_channel keys to cutoff values
        """
        if self._raw_data is None:
            self.load_analysis_file()
        
        if self._cutoffs is not None:
            return self._cutoffs
        
        self._cutoffs = self._raw_data.get('cutoffs', {})
        
        # Convert any non-float values to float
        for round_channel, cutoff in self._cutoffs.items():
            if not isinstance(cutoff, (int, float)):
                logger.warning(f"Invalid cutoff value for {round_channel}: {cutoff}, setting to 0.0")
                self._cutoffs[round_channel] = 0.0
            else:
                self._cutoffs[round_channel] = float(cutoff)
        
        return self._cutoffs
    
    def parse_nuclei(self) -> List[EnhancedNucleusInfo]:
        """
        Parse nuclei data from the analysis file.
        
        Returns:
            List of EnhancedNucleusInfo objects with epitope analysis data
        """
        if self._raw_data is None:
            self.load_analysis_file()
        
        if self._parsed_nuclei is not None:
            return self._parsed_nuclei
        
        intensity_data = self._raw_data.get('intensity_data', {})
        analysis_results = self._raw_data.get('analysis_results', {})
        
        self._parsed_nuclei = []
        
        for nucleus_label_str, nucleus_intensity_data in intensity_data.items():
            try:
                nucleus_label = int(nucleus_label_str)
                
                # Parse nucleus from intensity data
                nucleus_info = self._parse_single_nucleus(
                    nucleus_label, 
                    nucleus_intensity_data,
                    analysis_results.get(nucleus_label_str)
                )
                
                if nucleus_info:
                    self._parsed_nuclei.append(nucleus_info)
                    
            except Exception as e:
                logger.error(f"Error parsing nucleus {nucleus_label_str}: {e}")
                continue
        
        # Sort by label for consistency
        self._parsed_nuclei.sort(key=lambda x: x.label)
        
        logger.info(f"Parsed {len(self._parsed_nuclei)} nuclei from epitope analysis")
        return self._parsed_nuclei
    
    def _parse_single_nucleus(
        self, 
        nucleus_label: int, 
        intensity_data: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None
    ) -> Optional[EnhancedNucleusInfo]:
        """
        Parse a single nucleus from the raw data.
        
        Args:
            nucleus_label: Nucleus label ID
            intensity_data: Raw intensity data for this nucleus
            analysis_result: Raw analysis result for this nucleus
            
        Returns:
            EnhancedNucleusInfo object or None if parsing fails
        """
        try:
            # Extract basic nucleus properties
            spatial_location = intensity_data.get('spatial_location', [0, 0, 0])
            bounding_box = intensity_data.get('bounding_box', [0, 0, 0, 0, 0, 0])
            
            # Calculate area from bounding box if not provided
            if len(bounding_box) == 6:
                min_z, min_y, min_x, max_z, max_y, max_x = bounding_box
                area = (max_z - min_z) * (max_y - min_y) * (max_x - min_x)
            else:
                area = 0
            
            # Extract intensity data
            nuclei_intensities = intensity_data.get('nuclei_intensities', {})
            shell_intensities = intensity_data.get('shell_intensities', {})
            combined_intensities = intensity_data.get('combined_intensities', {})
            
            # Parse channel statistics
            channel_statistics = {}
            raw_channel_stats = intensity_data.get('channel_statistics', {})
            for round_channel, stats_dict in raw_channel_stats.items():
                if isinstance(stats_dict, dict):
                    channel_statistics[round_channel] = {}
                    for region in ['nuclei', 'shell', 'combined']:
                        if region in stats_dict:
                            channel_statistics[round_channel][region] = create_channel_statistics_from_dict(
                                stats_dict[region]
                            )
            
            # Parse quality metrics
            quality_metrics = {}
            raw_quality_metrics = intensity_data.get('quality_metrics', {})
            for round_channel, metrics_dict in raw_quality_metrics.items():
                if isinstance(metrics_dict, dict):
                    quality_metrics[round_channel] = create_quality_metrics_from_dict(metrics_dict)
            
            # Parse epitope analysis results
            epitope_analysis = None
            if analysis_result:
                epitope_analysis = EpitopeAnalysisData(
                    epitope_calls=analysis_result.get('epitope_calls', {}),
                    confidence_scores=analysis_result.get('confidence_scores', {}),
                    intensity_values=analysis_result.get('intensity_values', {}),
                    cutoff_values=analysis_result.get('cutoff_values', {}),
                    quality_score=analysis_result.get('quality_score', 0.0),
                    analysis_region=analysis_result.get('analysis_region', 'combined'),
                    rounds_analyzed=analysis_result.get('rounds_analyzed', [])
                )
            
            # Create enhanced nucleus info
            nucleus_info = EnhancedNucleusInfo(
                label=nucleus_label,
                bbox=tuple(bounding_box),
                area=area,
                centroid=tuple(spatial_location),
                epitope_analysis=epitope_analysis,
                channel_statistics=channel_statistics,
                quality_metrics=quality_metrics,
                nuclei_intensities=nuclei_intensities,
                shell_intensities=shell_intensities,
                combined_intensities=combined_intensities,
                rounds_processed=intensity_data.get('rounds_processed', [])
            )
            
            return nucleus_info
            
        except Exception as e:
            logger.error(f"Error parsing nucleus {nucleus_label}: {e}")
            return None
    
    def _extract_epitope_channels_from_cutoffs(self) -> List[str]:
        """Extract unique epitope channel names from cutoffs."""
        if self._raw_data is None:
            return []
        
        cutoffs = self._raw_data.get('cutoffs', {})
        channels = set()
        
        for round_channel in cutoffs.keys():
            # Extract channel name from "round_channel" format
            parts = round_channel.split('_', 1)
            if len(parts) == 2:
                channels.add(parts[1])  # Channel name
        
        return sorted(list(channels))
    
    def get_rounds_and_channels(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Extract available rounds and their channels from the analysis data.
        
        Returns:
            Tuple of (rounds_list, channels_by_round_dict)
        """
        if self._raw_data is None:
            self.load_analysis_file()
        
        cutoffs = self._raw_data.get('cutoffs', {})
        rounds_channels = {}
        
        for round_channel in cutoffs.keys():
            parts = round_channel.split('_', 1)
            if len(parts) == 2:
                round_name, channel_name = parts
                if round_name not in rounds_channels:
                    rounds_channels[round_name] = []
                if channel_name not in rounds_channels[round_name]:
                    rounds_channels[round_name].append(channel_name)
        
        # Sort for consistency
        rounds = sorted(rounds_channels.keys())
        for round_name in rounds_channels:
            rounds_channels[round_name].sort()
        
        return rounds, rounds_channels
    
    def validate_data_consistency(self) -> Dict[str, Any]:
        """
        Validate the consistency of the parsed data.
        
        Returns:
            Dictionary with validation results and any issues found
        """
        if self._raw_data is None:
            self.load_analysis_file()
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            metadata = self.parse_metadata()
            cutoffs = self.parse_cutoffs()
            nuclei = self.parse_nuclei()
            
            # Check metadata consistency
            if metadata.n_nuclei != len(nuclei):
                validation_results['issues'].append(
                    f"Metadata nuclei count ({metadata.n_nuclei}) doesn't match parsed nuclei ({len(nuclei)})"
                )
                validation_results['is_valid'] = False
            
            # Check cutoffs consistency
            rounds, channels_by_round = self.get_rounds_and_channels()
            expected_cutoffs = sum(len(channels) for channels in channels_by_round.values())
            if len(cutoffs) != expected_cutoffs:
                validation_results['warnings'].append(
                    f"Expected {expected_cutoffs} cutoffs but found {len(cutoffs)}"
                )
            
            # Check nuclei data consistency
            nuclei_with_epitope_analysis = sum(1 for n in nuclei if n.has_epitope_analysis())
            if nuclei_with_epitope_analysis != len(nuclei):
                validation_results['warnings'].append(
                    f"Only {nuclei_with_epitope_analysis}/{len(nuclei)} nuclei have epitope analysis data"
                )
            
            # Collect statistics
            validation_results['statistics'] = {
                'total_nuclei': len(nuclei),
                'nuclei_with_epitope_analysis': nuclei_with_epitope_analysis,
                'total_rounds': len(rounds),
                'total_cutoffs': len(cutoffs),
                'rounds': rounds,
                'channels_by_round': channels_by_round
            }
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Validation error: {e}")
        
        return validation_results


def load_epitope_analysis(analysis_file: Path) -> Tuple[PipelineMetadata, Dict[str, float], List[EnhancedNucleusInfo]]:
    """
    Convenience function to load and parse epitope analysis file.
    
    Args:
        analysis_file: Path to epitope analysis JSON file
        
    Returns:
        Tuple of (metadata, cutoffs, nuclei_list)
    """
    parser = EpitopeAnalysisParser(analysis_file)
    
    metadata = parser.parse_metadata()
    cutoffs = parser.parse_cutoffs()
    nuclei = parser.parse_nuclei()
    
    return metadata, cutoffs, nuclei