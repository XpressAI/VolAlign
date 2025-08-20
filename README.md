# VolAlign

VolAlign is a comprehensive Python package for volumetric image alignment, stitching, and processing, specifically designed for large-scale microscopy workflows.

## Features

### Core Functionality
- Create BDV/XML files from FOV TIFF images
- Stitch tiles using Fiji in headless mode
- Blend image tiles into composite volumes
- Resample 3D volumes with different voxel spacing
- Apply manual affine alignments
- Tune alignment with customizable pipelines
- Convert between TIFF and Zarr formats
- Downsample and stack TIFF images
- Reorient volumes with rotation and optional flipping

### Enhanced Microscopy Processing (New)
- **Memory-efficient Zarr processing** for very large volumes
- **Two-stage registration workflow**: Affine registration + deformation field computation
- **Distributed nuclei segmentation** using Cellpose with GPU acceleration
- **Channel merging** for robust registration (405nm + 488nm channels)
- **Label-preserving upsampling** for segmentation masks
- **High-level pipeline orchestrator** for complete workflows

## Installation

### Prerequisites

- Python 3.10 or higher
- Large memory systems (recommended: 64GB+ RAM for large datasets)
- GPU support for segmentation (CUDA-compatible, optional)
- Distributed computing cluster (optional, for very large datasets)

### Quick Installation (Recommended)

#### Option 1: Automated Installation Script
```bash
git clone https://github.com/XpressAI/VolAlign
cd VolAlign

# Development installation (recommended)
./install.sh dev

# Or install in a virtual environment
./install.sh venv my_volalign_env
```

#### Option 2: Direct pip Installation
```bash
git clone https://github.com/XpressAI/VolAlign
cd VolAlign

# Install all dependencies (including Git-based packages)
pip install -e .
```

### Installation Options

The installation script provides several options:

```bash
# Development installation (recommended)
./install.sh dev

# Production installation
./install.sh prod

# Install in virtual environment
./install.sh venv [environment_name]

# Show help
./install.sh help
```

### Install from PyPI (available soon!)

```bash
pip install VolAlign

```

### Verify Installation

```python
import VolAlign
print("VolAlign installed successfully!")

# Verify Git dependencies are available
try:
    import exm
    print("ExSeq-Toolbox available!")
except ImportError:
    print("ExSeq-Toolbox not available - check installation")

try:
    import cellpose
    print("Cellpose fork available!")
except ImportError:
    print("Cellpose not available - check installation")
```

## Quick Start

### Complete Pipeline Example

```python
from VolAlign import MicroscopyProcessingPipeline

# Configure the pipeline
config = {
    'working_directory': './processing_output',
    'voxel_spacing': [0.2, 0.1625, 0.1625],  # z, y, x in microns
    'downsample_factors': (4, 7, 7),
    'block_size': [512, 512, 512]
}

# Initialize pipeline
pipeline = MicroscopyProcessingPipeline(config)

# Prepare data (TIFF to Zarr conversion)
round1_zarr = pipeline.prepare_round_data('round1', {
    '405': '/path/to/round1_405nm.tif',
    '488': '/path/to/round1_488nm.tif',
    'channel3': '/path/to/round1_channel3.tif'
})

round2_zarr = pipeline.prepare_round_data('round2', {
    '405': '/path/to/round2_405nm.tif',
    '488': '/path/to/round2_488nm.tif',
    'channel3': '/path/to/round2_channel3.tif'
})

# Run registration workflow
registration_results = pipeline.run_registration_workflow(
    fixed_round_data=round1_zarr,
    moving_round_data=round2_zarr,
    registration_output_dir='./registration',
    registration_name='round1_to_round2'
)

# Run segmentation workflow
segmentation_results = pipeline.run_segmentation_workflow(
    input_405_channel=round1_zarr['405'],
    segmentation_output_dir='./segmentation',
    segmentation_name='round1_nuclei'
)

# Apply registration to all channels
aligned_channels = pipeline.apply_registration_to_all_channels(
    reference_round_data=round1_zarr,
    target_round_data=round2_zarr,
    deformation_field_path=registration_results['deformation_field'],
    output_directory='./aligned_channels'
)
```

### Individual Function Usage

```python
from VolAlign import (
    downsample_zarr_volume,
    merge_zarr_channels,
    compute_affine_registration,
    compute_deformation_field_registration,
    distributed_nuclei_segmentation
)

# Downsample large volume
downsample_zarr_volume(
    input_zarr_path='large_volume.zarr',
    output_zarr_path='downsampled.zarr',
    downsample_factors=(4, 7, 7)
)

# Merge registration channels
merge_zarr_channels(
    channel_a_path='405nm.zarr',
    channel_b_path='488nm.zarr',
    output_path='merged_registration.zarr',
    merge_strategy='mean'
)

# Compute affine registration
affine_matrix = compute_affine_registration(
    fixed_volume_path='fixed.zarr',
    moving_volume_path='moving.zarr',
    voxel_spacing=[0.2, 0.1625, 0.1625],
    output_matrix_path='affine_matrix.txt'
)

# Compute deformation field
aligned_path = compute_deformation_field_registration(
    fixed_zarr_path='fixed.zarr',
    moving_zarr_path='moving.zarr',
    affine_matrix_path='affine_matrix.txt',
    output_directory='./output',
    output_name='registration_result',
    voxel_spacing=[0.2, 0.1625, 0.1625]
)

# Distributed segmentation
segments, boxes = distributed_nuclei_segmentation(
    input_zarr_path='405nm_channel.zarr',
    output_zarr_path='segmentation_masks.zarr',
    model_type='cpsam'
)
```

## Function Reference

### Volume Processing (`utils.py`)
- `downsample_zarr_volume()` - Memory-efficient volume downsampling
- `merge_zarr_channels()` - Merge imaging channels with different strategies
- `upsample_segmentation_labels()` - Upsample segmentation masks preserving labels
- `convert_zarr_to_tiff()` - Convert Zarr to TIFF (with optional chunking for large volumes)
- `scale_intensity_to_uint16()` - Normalize intensity values
- `blend_ind()` - Blend volume tiles into composite images
- `convert_tiff_to_zarr()` - Convert TIFF to Zarr format
- `downsample_tiff()` - Downsample TIFF images
- `stack_tiff_images()` - Stack TIFF images along channel axis
- `reorient_volume_and_save_tiff()` - Reorient volumes with rotation and flipping

### Distributed Processing (`distributed_processing.py`)
- `compute_affine_registration()` - Initial coarse alignment (replaces "initial")
- `compute_deformation_field_registration()` - Fine alignment (replaces "final")
- `distributed_nuclei_segmentation()` - GPU-accelerated segmentation
- `apply_deformation_to_channels()` - Apply registration to multiple channels
- `create_registration_summary()` - Generate registration metadata

### Pipeline Orchestrator (`pipeline_orchestrator.py`)
- `MicroscopyProcessingPipeline` - High-level workflow management
- Complete pipeline state tracking and reporting
- Automatic file organization and naming

### Legacy Functions (Maintained)
- All original VolAlign functions remain available
- `create_bdv_xml()`, `stitch_tiles()`, `blend_tiles()`
- `voxel_spacing_resample()`, `apply_manual_alignment()`

## Workflow Overview

The enhanced VolAlign package supports the complete microscopy processing workflow:

1. **Data Management**: Efficient TIFF ↔ Zarr conversion and scaling
2. **Registration**: Two-stage process with affine + deformation field
3. **Segmentation**: Distributed processing with down/upsampling
4. **Channel Processing**: Apply transformations to all imaging channels
5. **Quality Control**: Comprehensive reporting and state tracking

## Requirements

- Python 3.10+
- Large memory systems (recommended: 64GB+ RAM)
- GPU support for segmentation (CUDA-compatible)
- Distributed computing cluster (optional, for large datasets)

## Dependencies

See `requirements.txt` for complete dependency list. Key dependencies:
- `zarr` - Efficient array storage
- `bigstream` - Registration algorithms
- `cellpose` - Segmentation models
- `dask` - Distributed computing
- `SimpleITK` - Image processing
- `tifffile` - TIFF I/O

## Examples

Complete examples are available in the `examples/` directory:
- `complete_microscopy_pipeline.py` - Full workflow demonstration
- Individual function usage examples

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see our contributing guidelines for details.


```
VolAlign: A comprehensive Python package for volumetric image alignment and processing
