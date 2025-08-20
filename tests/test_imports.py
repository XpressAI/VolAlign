"""
Test module imports and basic functionality.
"""

import pytest
import sys
import importlib


class TestPackageImports:
    """Test that all package modules can be imported successfully."""

    def test_main_package_import(self):
        """Test that the main VolAlign package can be imported."""
        import VolAlign
        assert VolAlign is not None
        assert hasattr(VolAlign, '__version__') or hasattr(VolAlign, '__name__')

    def test_utils_module_import(self):
        """Test that the utils module can be imported."""
        from VolAlign import utils
        assert utils is not None

    def test_alignment_tools_import(self):
        """Test that the alignment_tools module can be imported."""
        from VolAlign import alignment_tools
        assert alignment_tools is not None

    def test_distributed_processing_import(self):
        """Test that the distributed_processing module can be imported."""
        from VolAlign import distributed_processing
        assert distributed_processing is not None

    def test_pipeline_orchestrator_import(self):
        """Test that the pipeline_orchestrator module can be imported."""
        from VolAlign import pipeline_orchestrator
        assert pipeline_orchestrator is not None

    def test_all_modules_importable(self):
        """Test that all expected modules are importable."""
        expected_modules = [
            'VolAlign.utils',
            'VolAlign.alignment_tools',
            'VolAlign.distributed_processing',
            'VolAlign.pipeline_orchestrator'
        ]
        
        for module_name in expected_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Failed to import {module_name}"
            except ImportError as e:
                pytest.fail(f"Could not import {module_name}: {e}")

    def test_package_structure(self):
        """Test that the package has the expected structure."""
        import VolAlign
        
        # Check that the package directory exists and has expected attributes
        assert hasattr(VolAlign, '__path__') or hasattr(VolAlign, '__file__')

    def test_no_import_errors(self):
        """Test that importing modules doesn't raise unexpected errors."""
        modules_to_test = [
            'VolAlign',
            'VolAlign.utils',
            'VolAlign.alignment_tools',
            'VolAlign.distributed_processing',
            'VolAlign.pipeline_orchestrator'
        ]
        
        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                # Only fail if it's not a missing optional dependency
                if "No module named" not in str(e) or "VolAlign" in str(e):
                    pytest.fail(f"Unexpected import error for {module_name}: {e}")


class TestFunctionAvailability:
    """Test that expected functions are available in modules."""

    def test_utils_functions_available(self):
        """Test that expected functions are available in utils module."""
        from VolAlign import utils
        
        # Test for actual functions from the utils module
        expected_functions = [
            'blend_ind',
            'convert_zarr_to_tiff',
            'convert_tiff_to_zarr',
            'downsample_zarr_volume',
            'downsample_tiff',
            'upsample_segmentation_labels',
            'merge_zarr_channels',
            'stack_tiff_images',
            'scale_intensity_to_uint16',
            'reorient_volume_and_save_tiff'
        ]
        
        # Get all callable attributes
        available_functions = [name for name in dir(utils)
                             if callable(getattr(utils, name)) and not name.startswith('_')]
        
        # Check that expected functions are available
        for func_name in expected_functions:
            assert hasattr(utils, func_name), f"Function {func_name} not found in utils module"
            assert callable(getattr(utils, func_name)), f"{func_name} is not callable"
        
        # At least check that the module has some functions
        assert len(available_functions) > 0, "utils module should have callable functions"

    def test_pipeline_orchestrator_classes(self):
        """Test that expected classes are available in pipeline_orchestrator."""
        from VolAlign import pipeline_orchestrator
        
        # Check for the main pipeline class
        assert hasattr(pipeline_orchestrator, 'MicroscopyProcessingPipeline'), \
            "MicroscopyProcessingPipeline class not found"
        
        # Verify it's actually a class
        pipeline_class = getattr(pipeline_orchestrator, 'MicroscopyProcessingPipeline')
        assert isinstance(pipeline_class, type), "MicroscopyProcessingPipeline should be a class"
        
        # Check for classes (adjust based on actual implementation)
        available_classes = [name for name in dir(pipeline_orchestrator)
                           if isinstance(getattr(pipeline_orchestrator, name), type)]
        
        # At least check that the module has some classes
        assert len(available_classes) > 0, "pipeline_orchestrator should have classes"

    def test_alignment_tools_functions_available(self):
        """Test that expected functions are available in alignment_tools module."""
        from VolAlign import alignment_tools
        
        # Test for actual functions from the alignment_tools module
        expected_functions = [
            'create_bdv_xml',
            'stitch_tiles',
            'blend_tiles',
            'voxel_spacing_resample',
            'apply_manual_alignment',
            'linear_alignment_tuning'
        ]
        
        # Check that expected functions are available
        for func_name in expected_functions:
            assert hasattr(alignment_tools, func_name), f"Function {func_name} not found in alignment_tools module"
            assert callable(getattr(alignment_tools, func_name)), f"{func_name} is not callable"

    def test_distributed_processing_functions_available(self):
        """Test that expected functions are available in distributed_processing module."""
        from VolAlign import distributed_processing
        
        # Test for actual functions from the distributed_processing module
        expected_functions = [
            'compute_affine_registration',
            'compute_deformation_field_registration',
            'distributed_nuclei_segmentation',
            'apply_deformation_to_channels',
            'create_registration_summary'
        ]
        
        # Check that expected functions are available
        for func_name in expected_functions:
            assert hasattr(distributed_processing, func_name), f"Function {func_name} not found in distributed_processing module"
            assert callable(getattr(distributed_processing, func_name)), f"{func_name} is not callable"


class TestOptionalDependencies:
    """Test handling of optional dependencies."""

    def test_git_dependencies_handling(self):
        """Test that the package handles missing Git dependencies gracefully."""
        # Test that the package can be imported even if Git dependencies are missing
        import VolAlign
        assert VolAlign is not None

    def test_import_with_missing_optional_deps(self):
        """Test that core functionality works even with missing optional dependencies."""
        # This test ensures that the package doesn't crash if optional dependencies are missing
        try:
            from VolAlign import utils
            from VolAlign import alignment_tools
            # If we get here, the imports succeeded
            assert True
        except ImportError as e:
            # Only fail if it's a core dependency issue
            if any(core_dep in str(e).lower() for core_dep in ['numpy', 'scipy', 'zarr']):
                pytest.fail(f"Core dependency missing: {e}")


if __name__ == "__main__":
    pytest.main([__file__])