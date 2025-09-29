#!/usr/bin/env python3
"""
Configuration checker for nuclei-viewer.

This script helps users understand how their configuration will be loaded
and used by the nuclei-viewer application.

Usage:
    python check_config.py [--config CONFIG_PATH]
"""

import argparse
import json
import sys
from pathlib import Path

# Add the backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.core.config import load_config


def print_config_section(title, data, indent=0):
    """Pretty print a configuration section."""
    prefix = "  " * indent
    print(f"{prefix}{title}:")
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print_config_section(key, value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}  {key}: [{len(value)} items]")
                for i, item in enumerate(value[:3]):  # Show first 3 items
                    print(f"{prefix}    [{i}]: {item}")
                if len(value) > 3:
                    print(f"{prefix}    ... and {len(value) - 3} more")
            else:
                print(f"{prefix}  {key}: {value}")
    else:
        print(f"{prefix}  {data}")


def check_file_paths(config):
    """Check if important file paths exist."""
    print("\nFile Path Validation:")
    
    if config.data.is_pipeline_mode():
        working_dir = Path(config.data.pipeline.pipeline_working_directory)
        print(f"  Working directory: {working_dir}")
        print(f"    Exists: {'✓' if working_dir.exists() else '✗'}")
        
        # Check pipeline-specific paths
        epitope_file = working_dir / config.data.pipeline.epitope_analysis_file
        print(f"  Epitope analysis file: {epitope_file}")
        print(f"    Exists: {'✓' if epitope_file.exists() else '✗'}")
        
        zarr_dir = working_dir / config.data.pipeline.zarr_volumes_dir
        print(f"  Zarr volumes directory: {zarr_dir}")
        print(f"    Exists: {'✓' if zarr_dir.exists() else '✗'}")
        if zarr_dir.exists():
            zarr_files = list(zarr_dir.glob("**/*.zarr"))
            print(f"    Contains {len(zarr_files)} .zarr files")
        
        seg_dir = working_dir / config.data.pipeline.segmentation_dir
        print(f"  Segmentation directory: {seg_dir}")
        print(f"    Exists: {'✓' if seg_dir.exists() else '✗'}")
        if seg_dir.exists():
            seg_files = list(seg_dir.glob("*.zarr"))
            print(f"    Contains {len(seg_files)} segmentation files")
        
        aligned_dir = working_dir / config.data.pipeline.aligned_dir
        print(f"  Aligned directory: {aligned_dir}")
        print(f"    Exists: {'✓' if aligned_dir.exists() else '✗'}")
        if aligned_dir.exists():
            aligned_files = list(aligned_dir.glob("**/*.zarr"))
            print(f"    Contains {len(aligned_files)} aligned files")
    
    else:
        # Check manual mode paths
        if hasattr(config.data, 'segmentation') and config.data.segmentation:
            seg_path = Path(config.data.segmentation.file_path)
            print(f"  Segmentation file: {seg_path}")
            print(f"    Exists: {'✓' if seg_path.exists() else '✗'}")
        
        if hasattr(config.data, 'dapi_channel') and config.data.dapi_channel:
            dapi_path = Path(config.data.dapi_channel.file_path)
            print(f"  DAPI channel file: {dapi_path}")
            print(f"    Exists: {'✓' if dapi_path.exists() else '✗'}")


def analyze_configuration(config):
    """Analyze and explain the configuration."""
    print("\nConfiguration Analysis:")
    
    # Determine mode based on configuration structure
    mode = "pipeline" if config.data.is_pipeline_mode() else "manual"
    print(f"  Mode: {mode}")
    
    if mode == "pipeline":
        print(f"  Pipeline Integration:")
        print(f"    - Working directory: {config.data.pipeline.pipeline_working_directory}")
        print(f"    - Reference round: {config.data.pipeline.reference_round}")
        print(f"    - Auto-discover channels: {config.data.pipeline.auto_discover_channels}")
        print(f"    - Epitope analysis file: {config.data.pipeline.epitope_analysis_file}")
        
        if hasattr(config.data.pipeline, 'epitope_channels') and config.data.pipeline.epitope_channels:
            print(f"    - Configured epitope channels: {len(config.data.pipeline.epitope_channels)}")
            for channel in config.data.pipeline.epitope_channels[:3]:
                print(f"      • {channel}")
            if len(config.data.pipeline.epitope_channels) > 3:
                print(f"      ... and {len(config.data.pipeline.epitope_channels) - 3} more")
    
    else:
        print(f"  Manual Configuration:")
        if hasattr(config.data, 'segmentation') and config.data.segmentation:
            print(f"    - Segmentation file: {config.data.segmentation.file_path}")
        if hasattr(config.data, 'dapi_channel') and config.data.dapi_channel:
            print(f"    - DAPI channel: {config.data.dapi_channel.file_path}")
        if hasattr(config.data, 'epitope_channels'):
            print(f"    - Epitope channels: {len(config.data.epitope_channels)}")
    
    print(f"  Server Configuration:")
    print(f"    - Host: {config.server.host}")
    print(f"    - Port: {config.server.port}")
    print(f"    - Debug: {config.server.debug}")
    print(f"    - CORS origins: {len(config.server.cors_origins)} configured")
    
    print(f"  Processing Configuration:")
    print(f"    - Min object size: {config.processing.min_object_size}")
    print(f"    - XY padding: {config.processing.pad_xy}")
    print(f"    - Max objects per page: {config.processing.max_objects_per_page}")
    print(f"    - Cache MIPs: {config.processing.cache_mips}")


def main():
    parser = argparse.ArgumentParser(
        description="Check nuclei-viewer configuration"
    )
    parser.add_argument(
        "--config",
        default="../config_template.yaml",
        help="Path to configuration file (default: ../config_template.yaml)"
    )
    parser.add_argument(
        "--show-full-config",
        action="store_true",
        help="Show complete configuration structure"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output configuration as JSON"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Nuclei Viewer - Configuration Checker")
    print("=" * 60)
    
    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    try:
        config = load_config(args.config)
        print(f"✓ Configuration loaded successfully")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return 1
    
    if args.json:
        # Output as JSON
        config_dict = config.dict()
        print(json.dumps(config_dict, indent=2, default=str))
        return 0
    
    # Analyze configuration
    analyze_configuration(config)
    
    # Check file paths
    check_file_paths(config)
    
    if args.show_full_config:
        print("\nFull Configuration Structure:")
        config_dict = config.dict()
        print_config_section("nuclei_viewer", config_dict)
    
    print(f"\n{'=' * 60}")
    print("Configuration check complete!")
    print(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())