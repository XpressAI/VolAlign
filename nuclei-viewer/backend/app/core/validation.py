"""
Data validation and error handling for pipeline integration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import zarr

from .config import AppConfig, PipelineDataSourceConfig

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class PipelineValidationResult:
    """Result of pipeline validation with detailed information."""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.missing_files: List[str] = []
        self.invalid_files: List[str] = []
        self.structure_issues: List[str] = []
        
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
        
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        
    def add_missing_file(self, file_path: str):
        """Add a missing file."""
        self.missing_files.append(file_path)
        self.add_error(f"Missing required file: {file_path}")
        
    def add_invalid_file(self, file_path: str, reason: str):
        """Add an invalid file."""
        self.invalid_files.append(file_path)
        self.add_error(f"Invalid file {file_path}: {reason}")
        
    def add_structure_issue(self, issue: str):
        """Add a structure issue."""
        self.structure_issues.append(issue)
        self.add_error(f"Structure issue: {issue}")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_files": self.missing_files,
            "invalid_files": self.invalid_files,
            "structure_issues": self.structure_issues,
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "missing_files_count": len(self.missing_files),
                "invalid_files_count": len(self.invalid_files),
            }
        }


class PipelineValidator:
    """Comprehensive validator for pipeline outputs and structure."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline_config = config.data.pipeline
        self.working_directory = Path(self.pipeline_config.pipeline_working_directory)
        
    def validate_complete_pipeline(self) -> PipelineValidationResult:
        """
        Perform comprehensive validation of pipeline structure and data.
        
        Returns:
            PipelineValidationResult: Detailed validation results
        """
        result = PipelineValidationResult()
        
        logger.info("Starting comprehensive pipeline validation")
        
        # 1. Validate basic structure
        self._validate_directory_structure(result)
        
        # 2. Validate zarr volumes
        self._validate_zarr_volumes(result)
        
        # 3. Validate segmentation data
        self._validate_segmentation_data(result)
        
        # 4. Validate epitope analysis results
        self._validate_epitope_analysis(result)
        
        # 5. Validate data consistency
        self._validate_data_consistency(result)
        
        # 6. Check optional components
        self._validate_optional_components(result)
        
        logger.info(f"Pipeline validation completed: {len(result.errors)} errors, {len(result.warnings)} warnings")
        
        return result
    
    def _validate_directory_structure(self, result: PipelineValidationResult):
        """Validate expected directory structure."""
        logger.debug("Validating directory structure")
        
        # Check working directory exists
        if not self.working_directory.exists():
            result.add_error(f"Working directory does not exist: {self.working_directory}")
            return
            
        if not self.working_directory.is_dir():
            result.add_error(f"Working directory is not a directory: {self.working_directory}")
            return
            
        # Check required directories
        required_dirs = [
            "zarr_volumes",
            "segmentation", 
            "epitope_analysis"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.working_directory / dir_name
            if not dir_path.exists():
                result.add_structure_issue(f"Missing required directory: {dir_name}")
            elif not dir_path.is_dir():
                result.add_structure_issue(f"Expected directory but found file: {dir_name}")
                
        # Check optional directories
        optional_dirs = ["aligned", "registration"]
        for dir_name in optional_dirs:
            dir_path = self.working_directory / dir_name
            if dir_path.exists() and not dir_path.is_dir():
                result.add_warning(f"Expected directory but found file: {dir_name}")
    
    def _validate_zarr_volumes(self, result: PipelineValidationResult):
        """Validate zarr volume files."""
        logger.debug("Validating zarr volumes")
        
        zarr_dir = self.working_directory / "zarr_volumes"
        if not zarr_dir.exists():
            return  # Already reported in structure validation
            
        # Check reference round directory
        ref_round = self.pipeline_config.reference_round
        ref_round_dir = zarr_dir / ref_round
        
        if not ref_round_dir.exists():
            result.add_missing_file(f"zarr_volumes/{ref_round}")
            return
            
        # Validate zarr files in reference round
        zarr_files = list(ref_round_dir.glob("*.zarr"))
        if not zarr_files:
            result.add_structure_issue(f"No zarr files found in reference round: {ref_round}")
            return
            
        # Validate each zarr file
        for zarr_file in zarr_files:
            self._validate_zarr_file(zarr_file, result)
            
        # Check for DAPI channel (typically 405nm)
        dapi_files = list(ref_round_dir.glob("*405*.zarr"))
        if not dapi_files:
            result.add_warning(f"No DAPI channel (405nm) found in reference round: {ref_round}")
            
        # Check for epitope channels
        epitope_files = [f for f in zarr_files if "405" not in f.name]
        if not epitope_files:
            result.add_warning(f"No epitope channels found in reference round: {ref_round}")
            
        # Validate other rounds
        for round_dir in zarr_dir.iterdir():
            if round_dir.is_dir() and round_dir.name != ref_round:
                self._validate_round_directory(round_dir, result)
    
    def _validate_zarr_file(self, zarr_path: Path, result: PipelineValidationResult):
        """Validate individual zarr file."""
        try:
            # Try to open zarr file
            zarr_array = zarr.open(str(zarr_path), mode='r')
            
            # Check basic properties
            if zarr_array.ndim != 3:
                result.add_invalid_file(str(zarr_path), f"Expected 3D array, got {zarr_array.ndim}D")
                
            if zarr_array.size == 0:
                result.add_invalid_file(str(zarr_path), "Empty array")
                
            # Check data type
            if zarr_array.dtype not in ['uint16', 'uint8', 'float32']:
                result.add_warning(f"Unusual data type in {zarr_path.name}: {zarr_array.dtype}")
                
        except Exception as e:
            result.add_invalid_file(str(zarr_path), f"Cannot open zarr file: {e}")
    
    def _validate_round_directory(self, round_dir: Path, result: PipelineValidationResult):
        """Validate a round directory."""
        zarr_files = list(round_dir.glob("*.zarr"))
        if not zarr_files:
            result.add_warning(f"No zarr files found in round: {round_dir.name}")
            return
            
        for zarr_file in zarr_files:
            self._validate_zarr_file(zarr_file, result)
    
    def _validate_segmentation_data(self, result: PipelineValidationResult):
        """Validate segmentation data."""
        logger.debug("Validating segmentation data")
        
        seg_dir = self.working_directory / "segmentation"
        if not seg_dir.exists():
            return  # Already reported in structure validation
            
        # Look for segmentation files
        seg_files = list(seg_dir.glob("*.zarr"))
        if not seg_files:
            result.add_structure_issue("No segmentation zarr files found")
            return
            
        # Validate reference round segmentation
        ref_round = self.pipeline_config.reference_round
        ref_seg_files = [f for f in seg_files if ref_round in f.name]
        
        if not ref_seg_files:
            result.add_missing_file(f"segmentation/{ref_round}_nuclei_labels.zarr")
            return
            
        # Validate segmentation file
        seg_file = ref_seg_files[0]
        try:
            seg_array = zarr.open(str(seg_file), mode='r')
            
            # Check properties
            if seg_array.ndim != 3:
                result.add_invalid_file(str(seg_file), f"Expected 3D segmentation, got {seg_array.ndim}D")
                
            if seg_array.dtype not in ['uint16', 'uint32', 'int32']:
                result.add_invalid_file(str(seg_file), f"Invalid segmentation dtype: {seg_array.dtype}")
                
            # Check for labels
            unique_labels = len(set(seg_array.flat))
            if unique_labels < 10:
                result.add_warning(f"Very few unique labels in segmentation: {unique_labels}")
                
        except Exception as e:
            result.add_invalid_file(str(seg_file), f"Cannot validate segmentation: {e}")
    
    def _validate_epitope_analysis(self, result: PipelineValidationResult):
        """Validate epitope analysis results."""
        logger.debug("Validating epitope analysis results")
        
        epitope_dir = self.working_directory / "epitope_analysis"
        if not epitope_dir.exists():
            return  # Already reported in structure validation
            
        # Check for main results file
        results_file = epitope_dir / self.pipeline_config.epitope_analysis_file.split('/')[-1]
        if not results_file.exists():
            result.add_missing_file(str(results_file))
            return
            
        # Validate JSON structure
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                
            # Check required sections
            required_sections = ["metadata", "intensity_data", "cutoffs", "analysis_results"]
            for section in required_sections:
                if section not in data:
                    result.add_invalid_file(str(results_file), f"Missing required section: {section}")
                    
            # Validate metadata
            if "metadata" in data:
                metadata = data["metadata"]
                required_metadata = ["analysis_type", "n_nuclei", "n_rounds"]
                for field in required_metadata:
                    if field not in metadata:
                        result.add_invalid_file(str(results_file), f"Missing metadata field: {field}")
                        
            # Validate data consistency
            if "intensity_data" in data and "analysis_results" in data:
                intensity_nuclei = set(data["intensity_data"].keys())
                analysis_nuclei = set(data["analysis_results"].keys())
                
                if intensity_nuclei != analysis_nuclei:
                    result.add_invalid_file(str(results_file), 
                                          "Mismatch between intensity_data and analysis_results nuclei")
                    
        except json.JSONDecodeError as e:
            result.add_invalid_file(str(results_file), f"Invalid JSON: {e}")
        except Exception as e:
            result.add_invalid_file(str(results_file), f"Cannot validate epitope analysis: {e}")
    
    def _validate_data_consistency(self, result: PipelineValidationResult):
        """Validate consistency between different data components."""
        logger.debug("Validating data consistency")
        
        # This would involve cross-checking:
        # - Segmentation labels match epitope analysis nuclei
        # - Zarr file dimensions are consistent
        # - Round names match between zarr volumes and epitope analysis
        
        # For now, add basic checks
        zarr_dir = self.working_directory / "zarr_volumes"
        epitope_file = self.working_directory / "epitope_analysis" / self.pipeline_config.epitope_analysis_file.split('/')[-1]
        
        if zarr_dir.exists() and epitope_file.exists():
            try:
                # Get round names from zarr directory
                zarr_rounds = [d.name for d in zarr_dir.iterdir() if d.is_dir()]
                
                # Get round names from epitope analysis
                with open(epitope_file, 'r') as f:
                    epitope_data = json.load(f)
                    
                if "metadata" in epitope_data:
                    epitope_rounds = epitope_data["metadata"].get("n_rounds", 0)
                    
                    if len(zarr_rounds) != epitope_rounds:
                        result.add_warning(f"Round count mismatch: {len(zarr_rounds)} zarr rounds vs {epitope_rounds} in analysis")
                        
            except Exception as e:
                result.add_warning(f"Could not validate data consistency: {e}")
    
    def _validate_optional_components(self, result: PipelineValidationResult):
        """Validate optional pipeline components."""
        logger.debug("Validating optional components")
        
        # Check aligned data
        aligned_dir = self.working_directory / "aligned"
        if aligned_dir.exists():
            aligned_rounds = [d.name for d in aligned_dir.iterdir() if d.is_dir()]
            if not aligned_rounds:
                result.add_warning("Aligned directory exists but contains no round directories")
            else:
                # Validate aligned zarr files
                for round_dir in aligned_dir.iterdir():
                    if round_dir.is_dir():
                        aligned_files = list(round_dir.glob("*_aligned.zarr"))
                        if not aligned_files:
                            result.add_warning(f"No aligned zarr files in {round_dir.name}")
                            
        # Check registration data
        reg_dir = self.working_directory / "registration"
        if reg_dir.exists():
            reg_subdirs = [d for d in reg_dir.iterdir() if d.is_dir()]
            if not reg_subdirs:
                result.add_warning("Registration directory exists but contains no subdirectories")


def validate_pipeline_config(config: AppConfig) -> PipelineValidationResult:
    """
    Validate pipeline configuration and structure.
    
    Args:
        config: Application configuration
        
    Returns:
        PipelineValidationResult: Validation results
    """
    if not config.data.is_pipeline_mode():
        result = PipelineValidationResult()
        result.add_error("Not in pipeline mode - cannot validate pipeline")
        return result
        
    validator = PipelineValidator(config)
    return validator.validate_complete_pipeline()


def validate_epitope_analysis_file(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate epitope analysis JSON file structure.
    
    Args:
        file_path: Path to epitope analysis JSON file
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not file_path.exists():
        return False, [f"File does not exist: {file_path}"]
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Check required top-level keys
        required_keys = ["metadata", "intensity_data", "cutoffs", "analysis_results"]
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")
                
        # Validate metadata structure
        if "metadata" in data:
            metadata = data["metadata"]
            required_metadata = ["analysis_type", "n_nuclei", "n_rounds", "analysis_region", "cutoff_method"]
            for key in required_metadata:
                if key not in metadata:
                    errors.append(f"Missing metadata key: {key}")
                    
        # Validate data consistency
        if "intensity_data" in data and "analysis_results" in data:
            intensity_keys = set(data["intensity_data"].keys())
            analysis_keys = set(data["analysis_results"].keys())
            
            if intensity_keys != analysis_keys:
                errors.append("Nucleus keys mismatch between intensity_data and analysis_results")
                
        return len(errors) == 0, errors
        
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON format: {e}"]
    except Exception as e:
        return False, [f"Validation error: {e}"]


def create_validation_report(result: PipelineValidationResult, output_path: Optional[Path] = None) -> str:
    """
    Create a detailed validation report.
    
    Args:
        result: Validation results
        output_path: Optional path to save report
        
    Returns:
        Report content as string
    """
    report_lines = [
        "# Pipeline Validation Report",
        f"Generated at: {logger.handlers[0].formatter.formatTime(logger.makeRecord('', 0, '', 0, '', (), None)) if logger.handlers else 'Unknown'}",
        "",
        f"## Summary",
        f"- **Status**: {'✅ VALID' if result.is_valid else '❌ INVALID'}",
        f"- **Total Errors**: {len(result.errors)}",
        f"- **Total Warnings**: {len(result.warnings)}",
        f"- **Missing Files**: {len(result.missing_files)}",
        f"- **Invalid Files**: {len(result.invalid_files)}",
        "",
    ]
    
    if result.errors:
        report_lines.extend([
            "## Errors",
            ""
        ])
        for i, error in enumerate(result.errors, 1):
            report_lines.append(f"{i}. {error}")
        report_lines.append("")
        
    if result.warnings:
        report_lines.extend([
            "## Warnings",
            ""
        ])
        for i, warning in enumerate(result.warnings, 1):
            report_lines.append(f"{i}. {warning}")
        report_lines.append("")
        
    if result.missing_files:
        report_lines.extend([
            "## Missing Files",
            ""
        ])
        for file_path in result.missing_files:
            report_lines.append(f"- {file_path}")
        report_lines.append("")
        
    if result.invalid_files:
        report_lines.extend([
            "## Invalid Files",
            ""
        ])
        for file_path in result.invalid_files:
            report_lines.append(f"- {file_path}")
        report_lines.append("")
        
    report_content = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_content)
            
    return report_content