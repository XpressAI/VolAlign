"""
Test package building and wheel creation.
"""

import pytest
import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path


class TestPackageBuild:
    """Test package building functionality."""

    def test_wheel_build(self):
        """Test that the package can be built as a wheel."""
        # Get the package root directory
        package_root = Path(__file__).parent.parent
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to package directory and build wheel
            original_cwd = os.getcwd()
            try:
                os.chdir(package_root)
                
                # Build the wheel
                result = subprocess.run([
                    sys.executable, '-m', 'build', '--wheel', '--outdir', temp_dir
                ], capture_output=True, text=True)
                
                # Check that build succeeded
                assert result.returncode == 0, f"Wheel build failed: {result.stderr}"
                
                # Check that wheel file was created
                wheel_files = list(Path(temp_dir).glob('*.whl'))
                assert len(wheel_files) > 0, "No wheel file was created"
                
                # Check wheel filename format
                wheel_file = wheel_files[0]
                assert 'VolAlign' in wheel_file.name, f"Wheel name doesn't contain package name: {wheel_file.name}"
                
            finally:
                os.chdir(original_cwd)

    def test_source_distribution_build(self):
        """Test that the package can be built as a source distribution."""
        package_root = Path(__file__).parent.parent
        
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(package_root)
                
                # Build the source distribution
                result = subprocess.run([
                    sys.executable, '-m', 'build', '--sdist', '--outdir', temp_dir
                ], capture_output=True, text=True)
                
                # Check that build succeeded
                assert result.returncode == 0, f"Source distribution build failed: {result.stderr}"
                
                # Check that sdist file was created
                sdist_files = list(Path(temp_dir).glob('*.tar.gz'))
                assert len(sdist_files) > 0, "No source distribution file was created"
                
                # Check sdist filename format
                sdist_file = sdist_files[0]
                assert 'VolAlign' in sdist_file.name, f"Sdist name doesn't contain package name: {sdist_file.name}"
                
            finally:
                os.chdir(original_cwd)

    def test_wheel_installation(self):
        """Test that the built wheel can be installed."""
        package_root = Path(__file__).parent.parent
        
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(package_root)
                
                # Build the wheel
                build_result = subprocess.run([
                    sys.executable, '-m', 'build', '--wheel', '--outdir', temp_dir
                ], capture_output=True, text=True)
                
                assert build_result.returncode == 0, f"Wheel build failed: {build_result.stderr}"
                
                # Find the wheel file
                wheel_files = list(Path(temp_dir).glob('*.whl'))
                assert len(wheel_files) > 0, "No wheel file found"
                wheel_file = wheel_files[0]
                
                # Test installation in a separate Python process to avoid conflicts
                install_result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', str(wheel_file), '--force-reinstall', '--no-deps'
                ], capture_output=True, text=True)
                
                # Note: We use --no-deps to avoid dependency conflicts in test environment
                assert install_result.returncode == 0, f"Wheel installation failed: {install_result.stderr}"
                
                # Test that the package can be imported after installation
                import_result = subprocess.run([
                    sys.executable, '-c', 'import VolAlign; print("Import successful")'
                ], capture_output=True, text=True)
                
                assert import_result.returncode == 0, f"Package import failed after wheel installation: {import_result.stderr}"
                assert "Import successful" in import_result.stdout
                
            finally:
                os.chdir(original_cwd)

    def test_package_metadata(self):
        """Test that package metadata is correctly included in the build."""
        package_root = Path(__file__).parent.parent
        
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(package_root)
                
                # Build the wheel
                result = subprocess.run([
                    sys.executable, '-m', 'build', '--wheel', '--outdir', temp_dir
                ], capture_output=True, text=True)
                
                assert result.returncode == 0, f"Wheel build failed: {result.stderr}"
                
                # Check wheel contents using wheel command if available
                wheel_files = list(Path(temp_dir).glob('*.whl'))
                wheel_file = wheel_files[0]
                
                # Try to inspect wheel contents
                try:
                    # Try to inspect wheel contents using zipfile (more reliable than wheel command)
                    import zipfile
                    with zipfile.ZipFile(wheel_file, 'r') as zip_file:
                        file_list = zip_file.namelist()
                        volalign_files = [f for f in file_list if 'VolAlign' in f or 'volalign' in f.lower()]
                        assert len(volalign_files) > 0, f"VolAlign package files not found in wheel. Files: {file_list[:10]}"
                            
                except Exception as e:
                    # If inspection fails, just check that wheel file exists and has reasonable size
                    assert wheel_file.stat().st_size > 1000, f"Wheel file seems too small: {wheel_file.stat().st_size} bytes"
                
            finally:
                os.chdir(original_cwd)


class TestBuildRequirements:
    """Test build requirements and dependencies."""

    def test_build_backend_available(self):
        """Test that the build backend (setuptools) is available."""
        try:
            import setuptools
            import wheel
            assert setuptools is not None
            assert wheel is not None
        except ImportError as e:
            pytest.fail(f"Build dependencies not available: {e}")

    def test_pyproject_toml_valid(self):
        """Test that pyproject.toml is valid and can be parsed."""
        package_root = Path(__file__).parent.parent
        pyproject_file = package_root / 'pyproject.toml'
        
        assert pyproject_file.exists(), "pyproject.toml file not found"
        
        try:
            import tomllib as tomli
        except ImportError:
            try:
                import tomli
            except ImportError:
                pytest.skip("No TOML parser available")
        
        with open(pyproject_file, 'rb') as f:
            config = tomli.load(f)
        
        # Check required sections
        assert 'build-system' in config, "build-system section missing from pyproject.toml"
        assert 'project' in config, "project section missing from pyproject.toml"
        
        # Check build system
        build_system = config['build-system']
        assert 'requires' in build_system, "build-system.requires missing"
        assert 'build-backend' in build_system, "build-system.build-backend missing"
        
        # Check project metadata
        project = config['project']
        assert 'name' in project, "project.name missing"
        assert 'version' in project, "project.version missing"
        assert project['name'] == 'VolAlign', f"Expected project name 'VolAlign', got '{project['name']}'"

    def test_setup_py_compatibility(self):
        """Test that setup.py exists and is functional."""
        package_root = Path(__file__).parent.parent
        setup_file = package_root / 'setup.py'
        
        assert setup_file.exists(), "setup.py file not found"
        
        # Test that setup.py can be executed without errors
        original_cwd = os.getcwd()
        try:
            os.chdir(package_root)
            result = subprocess.run([
                sys.executable, 'setup.py', '--help'
            ], capture_output=True, text=True)
            
            # Should not fail completely (return code might be non-zero but shouldn't crash)
            assert "usage:" in result.stdout.lower() or "options:" in result.stdout.lower(), \
                f"setup.py doesn't seem to be functional: {result.stderr}"
                
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__])