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

### Configuration Setup

VolAlign now uses YAML configuration files for streamlined pipeline setup. First, create your configuration file:

```bash
# Copy the template configuration
cp config_template.yaml my_config.yaml

# Edit the configuration file with your specific paths and parameters
nano my_config.yaml  # or use your preferred editor
```

### Complete Pipeline Example

```python
from VolAlign import MicroscopyProcessingPipeline

# Initialize pipeline with YAML configuration file path
pipeline = MicroscopyProcessingPipeline('my_config.yaml')

# Run the complete multi-round processing pipeline
results = pipeline.run_complete_pipeline_from_config()

# The pipeline will automatically:
# 1. Convert all TIFF files to Zarr format
# 2. Register all rounds to the reference round
# 3. Perform nuclei segmentation on the reference round
# 4. Apply registration transformations to all channels
# 5. Generate comprehensive reports and summaries

print("Pipeline completed successfully!")
print(f"Results saved to: {pipeline.working_directory}")
```

### Manual Step-by-Step Processing

For more control over individual steps:

```python
from VolAlign import MicroscopyProcessingPipeline

# Initialize pipeline with YAML configuration file
pipeline = MicroscopyProcessingPipeline('my_config.yaml')

# Step 1: Process all rounds from configuration
all_round_data = pipeline.process_all_rounds_from_config()

# Step 2: Run registration for each round against reference
reference_round = pipeline.reference_round
registration_results = {}

for round_name in pipeline.rounds_data.keys():
    if round_name != reference_round:
        registration_results[round_name] = pipeline.run_registration_workflow(
            fixed_round_data=all_round_data[reference_round],
            moving_round_data=all_round_data[round_name],
            registration_output_dir=f"{pipeline.working_directory}/registration",
            registration_name=f"{reference_round}_to_{round_name}"
        )

# Step 3: Run segmentation on reference round
segmentation_results = pipeline.run_segmentation_workflow(
    input_channel=all_round_data[reference_round][pipeline.segmentation_channel],
    segmentation_output_dir=f"{pipeline.working_directory}/segmentation",
    segmentation_name=f"{reference_round}_nuclei"
)

# Step 4: Apply registration to all channels
for round_name, reg_results in registration_results.items():
    aligned_channels = pipeline.apply_registration_to_all_channels(
        reference_round_data=all_round_data[reference_round],
        target_round_data=all_round_data[round_name],
        deformation_field_path=reg_results['deformation_field'],
        output_directory=f"{pipeline.working_directory}/aligned_channels/{round_name}"
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
