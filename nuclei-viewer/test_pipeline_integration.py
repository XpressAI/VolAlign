#!/usr/bin/env python3
"""
Test script for nuclei-viewer pipeline integration.

This script validates the integration between the nuclei-viewer and VolAlign pipeline outputs
by testing the complete data flow from pipeline results to frontend display.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import zarr
import yaml

# Add nuclei-viewer backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.core.config import load_config
from app.core.pipeline_data_loader import PipelineDataLoader
from app.core.epitope_parser import EpitopeAnalysisParser
from app.core.validation import PipelineValidator
from app.core.models import PipelineMetadata, EpitopeAnalysisData

logger = logging.getLogger(__name__)


class PipelineIntegrationTester:
    """Comprehensive tester for pipeline integration."""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "warnings": []
        }
        
    def setup_test_environment(self):
        """Set up a complete test environment with mock pipeline data."""
        logger.info("Setting up test environment...")
        
        # Create directory structure
        working_dir = self.test_dir / "pipeline_working_directory"
        working_dir.mkdir(parents=True, exist_ok=True)
        
        # Create zarr_volumes directory
        zarr_dir = working_dir / "zarr_volumes"
        zarr_dir.mkdir(exist_ok=True)
        
        # Create round directories and zarr files
        for round_name in ["round1", "round2"]:
            round_dir = zarr_dir / round_name
            round_dir.mkdir(exist_ok=True)
            
            # Create mock zarr files for each channel
            for channel in ["405", "488", "561"]:
                zarr_file = round_dir / f"{round_name}_{channel}.zarr"
                
                # Create 3D mock data (Z, Y, X)
                mock_data = np.random.randint(0, 4096, size=(10, 100, 100), dtype=np.uint16)
                zarr_array = zarr.open(str(zarr_file), mode='w', shape=mock_data.shape, 
                                     dtype=mock_data.dtype, chunks=(5, 50, 50))
                zarr_array[:] = mock_data
        
        # Create segmentation directory
        seg_dir = working_dir / "segmentation"
        seg_dir.mkdir(exist_ok=True)
        
        # Create mock segmentation
        seg_file = seg_dir / "round1_nuclei_labels.zarr"
        seg_data = np.zeros((10, 100, 100), dtype=np.uint16)
        
        # Add some mock nuclei
        for i in range(1, 6):  # 5 nuclei
            z_start, z_end = 2, 8
            y_start, y_end = i * 15, i * 15 + 10
            x_start, x_end = 20, 30
            seg_data[z_start:z_end, y_start:y_end, x_start:x_end] = i
        
        seg_array = zarr.open(str(seg_file), mode='w', shape=seg_data.shape,
                             dtype=seg_data.dtype, chunks=(5, 50, 50))
        seg_array[:] = seg_data
        
        # Create epitope analysis directory and results
        epitope_dir = working_dir / "epitope_analysis"
        epitope_dir.mkdir(exist_ok=True)
        
        # Create mock epitope analysis results
        epitope_results = self._create_mock_epitope_results()
        epitope_file = epitope_dir / "nucleus_centric_analysis_results.json"
        
        # Also create the file with the expected name from config
        epitope_file_alt = epitope_dir / "nucleus_centric_analysis_epitope_analysis.json"
        
        with open(epitope_file, 'w') as f:
            json.dump(epitope_results, f, indent=2)
        
        with open(epitope_file_alt, 'w') as f:
            json.dump(epitope_results, f, indent=2)
        
        # Create aligned directory
        aligned_dir = working_dir / "aligned"
        aligned_dir.mkdir(exist_ok=True)
        
        # Create test configuration
        config_data = {
            "version": "v1.1",
            "app": {
                "title": "Nuclei Viewer - Pipeline Integration Test",
                "description": "Testing pipeline integration"
            },
            "server": {
                "host": "localhost",
                "port": 8000,
                "debug": True
            },
            "data": {
                "pipeline": {
                    "pipeline_working_directory": str(working_dir),
                    "reference_round": "round1"
                }
            },
            "processing": {
                "min_object_size": 50,
                "pad_xy": 10,
                "max_objects_per_page": 20,
                "cache_mips": True
            },
            "ui": {
                "epitope_analysis": {
                    "show_epitope_calls": True,
                    "show_confidence_scores": True,
                    "show_quality_scores": True
                }
            }
        }
        
        config_file = self.test_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Test environment created at: {working_dir}")
        return config_file, working_dir
    
    def _create_mock_epitope_results(self) -> Dict[str, Any]:
        """Create mock epitope analysis results."""
        return {
            "metadata": {
                "analysis_type": "nucleus_centric",
                "n_nuclei": 5,
                "n_rounds": 2,
                "analysis_region": "combined",
                "cutoff_method": "otsu",
                "shell_parameters": {
                    "inner_radius": 2,
                    "outer_radius": 5
                }
            },
            "cutoffs": {
                "round1_488": 150.5,
                "round1_561": 200.3,
                "round2_488": 175.8,
                "round2_561": 190.2
            },
            "intensity_data": {
                str(i): {
                    "nuclei_intensities": {
                        "round1_488": np.random.uniform(100, 300),
                        "round1_561": np.random.uniform(80, 250),
                        "round2_488": np.random.uniform(120, 280),
                        "round2_561": np.random.uniform(90, 220)
                    },
                    "shell_intensities": {
                        "round1_488": np.random.uniform(80, 200),
                        "round1_561": np.random.uniform(60, 180),
                        "round2_488": np.random.uniform(90, 200),
                        "round2_561": np.random.uniform(70, 160)
                    },
                    "combined_intensities": {
                        "round1_488": np.random.uniform(90, 250),
                        "round1_561": np.random.uniform(70, 200),
                        "round2_488": np.random.uniform(100, 240),
                        "round2_561": np.random.uniform(80, 190)
                    }
                } for i in range(1, 6)
            },
            "analysis_results": {
                str(i): {
                    "epitope_calls": {
                        "round1_488": bool(np.random.choice([True, False])),
                        "round1_561": bool(np.random.choice([True, False])),
                        "round2_488": bool(np.random.choice([True, False])),
                        "round2_561": bool(np.random.choice([True, False]))
                    },
                    "confidence_scores": {
                        "round1_488": float(np.random.uniform(0.3, 0.95)),
                        "round1_561": float(np.random.uniform(0.3, 0.95)),
                        "round2_488": float(np.random.uniform(0.3, 0.95)),
                        "round2_561": float(np.random.uniform(0.3, 0.95))
                    },
                    "quality_score": float(np.random.uniform(0.5, 0.95)),
                    "rounds_analyzed": ["round1", "round2"]
                } for i in range(1, 6)
            }
        }
    
    def test_configuration_loading(self, config_file: Path) -> bool:
        """Test configuration loading and validation."""
        logger.info("Testing configuration loading...")
        
        try:
            # Load configuration
            config = load_config(str(config_file))
            
            # Validate it's in pipeline mode
            if not config.data.is_pipeline_mode():
                self.test_results["errors"].append("Configuration not detected as pipeline mode")
                return False
            
            # Check required fields
            pipeline_config = config.data.pipeline
            if not pipeline_config.pipeline_working_directory:
                self.test_results["errors"].append("Working directory not configured")
                return False
            
            if not pipeline_config.reference_round:
                self.test_results["errors"].append("Reference round not configured")
                return False
            
            logger.info("✓ Configuration loading passed")
            self.test_results["passed"] += 1
            return True
            
        except Exception as e:
            self.test_results["errors"].append(f"Configuration loading failed: {e}")
            self.test_results["failed"] += 1
            return False
    
    def test_pipeline_validation(self, config) -> bool:
        """Test pipeline structure validation."""
        logger.info("Testing pipeline validation...")
        
        try:
            validation_result = validate_pipeline_config(config)
            
            if not validation_result.is_valid:
                self.test_results["errors"].append(f"Pipeline validation failed: {validation_result.errors}")
                return False
            
            if validation_result.warnings:
                self.test_results["warnings"].extend(validation_result.warnings)
            
            logger.info("✓ Pipeline validation passed")
            self.test_results["passed"] += 1
            return True
            
        except Exception as e:
            self.test_results["errors"].append(f"Pipeline validation failed: {e}")
            self.test_results["failed"] += 1
            return False
    
    def test_epitope_parser(self, working_dir: Path) -> bool:
        """Test epitope analysis parser."""
        logger.info("Testing epitope analysis parser...")
        
        try:
            epitope_file = working_dir / "epitope_analysis" / "nucleus_centric_analysis_results.json"
            parser = EpitopeAnalysisParser(epitope_file)
            
            # Test parsing
            epitope_data = parser.parse_analysis_results()
            
            # Validate parsed data
            if not isinstance(epitope_data, EpitopeAnalysisData):
                self.test_results["errors"].append("Parser did not return EpitopeAnalysisData")
                return False
            
            if epitope_data.metadata.n_nuclei != 5:
                self.test_results["errors"].append(f"Expected 5 nuclei, got {epitope_data.metadata.n_nuclei}")
                return False
            
            if len(epitope_data.cutoffs) != 4:
                self.test_results["errors"].append(f"Expected 4 cutoffs, got {len(epitope_data.cutoffs)}")
                return False
            
            logger.info("✓ Epitope parser test passed")
            self.test_results["passed"] += 1
            return True
            
        except Exception as e:
            self.test_results["errors"].append(f"Epitope parser test failed: {e}")
            self.test_results["failed"] += 1
            return False
    
    def test_pipeline_data_loader(self, config) -> bool:
        """Test pipeline data loader."""
        logger.info("Testing pipeline data loader...")
        
        try:
            # Create pipeline data loader
            data_loader = PipelineDataLoader(config)
            
            # Test dataset discovery
            discovered = data_loader.discover_datasets()
            
            if "segmentation" not in discovered:
                self.test_results["errors"].append("Segmentation not discovered")
                return False
            
            # Test that data loader can be created and initialized
            if not hasattr(data_loader, 'epitope_parser'):
                self.test_results["warnings"].append("Data loader does not have epitope parser")
            
            # Test basic functionality
            try:
                # Test if we can access the working directory
                working_dir = Path(config.data.pipeline.working_directory)
                if not working_dir.exists():
                    self.test_results["errors"].append("Working directory does not exist")
                    return False
                
                # Test if epitope analysis file exists
                epitope_file = working_dir / "epitope_analysis" / "nucleus_centric_analysis_results.json"
                if not epitope_file.exists():
                    self.test_results["errors"].append("Epitope analysis file does not exist")
                    return False
                
            except Exception as e:
                self.test_results["errors"].append(f"Data loader basic test failed: {e}")
                return False
            
            logger.info("✓ Pipeline data loader test passed")
            self.test_results["passed"] += 1
            return True
            
        except Exception as e:
            self.test_results["errors"].append(f"Pipeline data loader test failed: {e}")
            self.test_results["failed"] += 1
            return False
    
    def test_data_consistency(self, config) -> bool:
        """Test data consistency between components."""
        logger.info("Testing data consistency...")
        
        try:
            data_loader = PipelineDataLoader(config)
            
            # Load all data
            metadata, cutoffs, nuclei_from_epitope = data_loader.load_epitope_analysis()
            extracted_nuclei = data_loader.extract_nuclei_info()
            
            # Check consistency
            if len(nuclei_from_epitope) != len(extracted_nuclei):
                self.test_results["errors"].append("Inconsistent nuclei count between epitope analysis and extraction")
                return False
            
            # Check that all nuclei have consistent labels
            epitope_labels = set(n.label for n in nuclei_from_epitope)
            extracted_labels = set(n.label for n in extracted_nuclei)
            
            if epitope_labels != extracted_labels:
                self.test_results["errors"].append("Inconsistent nucleus labels between data sources")
                return False
            
            # Check that cutoffs match expected round-channel combinations
            expected_keys = {"round1_488", "round1_561", "round2_488", "round2_561"}
            cutoff_keys = set(cutoffs.keys())
            
            if cutoff_keys != expected_keys:
                self.test_results["errors"].append(f"Cutoff keys mismatch. Expected: {expected_keys}, Got: {cutoff_keys}")
                return False
            
            logger.info("✓ Data consistency test passed")
            self.test_results["passed"] += 1
            return True
            
        except Exception as e:
            self.test_results["errors"].append(f"Data consistency test failed: {e}")
            self.test_results["failed"] += 1
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Starting pipeline integration tests...")
        
        try:
            # Setup test environment
            config_file, working_dir = self.setup_test_environment()
            
            # Load configuration
            config = load_config(str(config_file))
            
            # Run tests
            tests = [
                ("Configuration Loading", lambda: self.test_configuration_loading(config_file)),
                ("Pipeline Validation", lambda: self.test_pipeline_validation(config)),
                ("Epitope Parser", lambda: self.test_epitope_parser(working_dir)),
                ("Pipeline Data Loader", lambda: self.test_pipeline_data_loader(config)),
                ("Data Consistency", lambda: self.test_data_consistency(config)),
            ]
            
            for test_name, test_func in tests:
                logger.info(f"\n--- Running {test_name} Test ---")
                try:
                    success = test_func()
                    if success:
                        logger.info(f"✓ {test_name} test PASSED")
                    else:
                        logger.error(f"✗ {test_name} test FAILED")
                        self.test_results["failed"] += 1
                except Exception as e:
                    logger.error(f"✗ {test_name} test ERROR: {e}")
                    self.test_results["errors"].append(f"{test_name}: {e}")
                    self.test_results["failed"] += 1
            
            # Generate summary
            total_tests = self.test_results["passed"] + self.test_results["failed"]
            success_rate = (self.test_results["passed"] / total_tests * 100) if total_tests > 0 else 0
            
            summary = {
                "total_tests": total_tests,
                "passed": self.test_results["passed"],
                "failed": self.test_results["failed"],
                "success_rate": success_rate,
                "errors": self.test_results["errors"],
                "warnings": self.test_results["warnings"]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return {
                "total_tests": 0,
                "passed": 0,
                "failed": 1,
                "success_rate": 0,
                "errors": [f"Test suite setup failed: {e}"],
                "warnings": []
            }


def main():
    """Main test function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Run tests
        tester = PipelineIntegrationTester(test_dir)
        results = tester.run_all_tests()
        
        # Print results
        print("\n" + "="*60)
        print("PIPELINE INTEGRATION TEST RESULTS")
        print("="*60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']}")
        print(f"Failed: {results['failed']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        
        if results['warnings']:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"  ⚠️  {warning}")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"  ❌ {error}")
        
        print("\n" + "="*60)
        
        # Return appropriate exit code
        return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())