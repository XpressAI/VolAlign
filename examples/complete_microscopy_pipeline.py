#!/usr/bin/env python3
"""
Complete Microscopy Processing Pipeline Example

This example demonstrates how to use the enhanced VolAlign package for
processing multi-round microscopy data with better naming conventions.

The pipeline includes:
1. Data preparation (TIFF to Zarr conversion)
2. Two-stage registration (affine + deformation field)
3. Distributed nuclei segmentation
4. Channel alignment using computed transformations
"""

import os
from pathlib import Path
from VolAlign import MicroscopyProcessingPipeline

def main():
    # Configuration for the processing pipeline
    config = {
        'working_directory': './microscopy_processing_output',
        'voxel_spacing': [0.2, 0.1625, 0.1625],  # z, y, x in microns
        'downsample_factors': (4, 7, 7),  # z, y, x downsampling for registration
        'block_size': [512, 512, 512],  # Processing block size
        'cluster_config': {
            "n_workers": 8,
            "threads_per_worker": 1,
            "memory_limit": '150GB',
            'config': {
                'distributed.nanny.pre-spawn-environ': {
                    'MALLOC_TRIM_THRESHOLD_': 65536,
                    'MKL_NUM_THREADS': 10,
                    'OMP_NUM_THREADS': 10,
                    'OPENBLAS_NUM_THREADS': 10,
                },
                'distributed.scheduler.worker-ttl': None
            }
        }
    }
    
    # Initialize the processing pipeline
    pipeline = MicroscopyProcessingPipeline(config)
    
    # Example data paths - replace with your actual data paths
    round1_tiff_files = {
        '405': '/path/to/round1_405nm.tif',
        '488': '/path/to/round1_488nm.tif',
        'channel3': '/path/to/round1_channel3.tif',
        'channel4': '/path/to/round1_channel4.tif'
    }
    
    round2_tiff_files = {
        '405': '/path/to/round2_405nm.tif',
        '488': '/path/to/round2_488nm.tif',
        'channel3': '/path/to/round2_channel3.tif',
        'channel4': '/path/to/round2_channel4.tif'
    }
    
    print("=== Step 1: Data Preparation ===")
    # Convert TIFF files to Zarr format for efficient processing
    round1_zarr = pipeline.prepare_round_data('round1', round1_tiff_files)
    round2_zarr = pipeline.prepare_round_data('round2', round2_tiff_files)
    
    print("Round 1 Zarr files:", round1_zarr)
    print("Round 2 Zarr files:", round2_zarr)
    
    print("\n=== Step 2: Registration Workflow ===")
    # Perform two-stage registration between rounds
    registration_results = pipeline.run_registration_workflow(
        fixed_round_data=round1_zarr,
        moving_round_data=round2_zarr,
        registration_output_dir='./registration_results',
        registration_name='round1_to_round2'
    )
    
    print("Registration results:", registration_results)
    
    print("\n=== Step 3: Nuclei Segmentation ===")
    # Perform distributed nuclei segmentation on 405nm channel
    segmentation_results = pipeline.run_segmentation_workflow(
        input_405_channel=round1_zarr['405'],
        segmentation_output_dir='./segmentation_results',
        segmentation_name='round1_nuclei',
        downsample_for_segmentation=True,
        upsample_results=True
    )
    
    print("Segmentation results:", segmentation_results)
    print(f"Number of nuclei detected: {segmentation_results['num_objects']}")
    
    print("\n=== Step 4: Apply Registration to All Channels ===")
    # Apply computed registration to all imaging channels
    aligned_channels = pipeline.apply_registration_to_all_channels(
        reference_round_data=round1_zarr,
        target_round_data=round2_zarr,
        deformation_field_path=registration_results['deformation_field'],
        output_directory='./aligned_channels'
    )
    
    print("Aligned channels:", aligned_channels)
    
    print("\n=== Step 5: Save Pipeline State and Generate Report ===")
    # Save pipeline state for reproducibility
    pipeline.save_pipeline_state('./pipeline_state.json')
    
    # Generate comprehensive processing report
    report = pipeline.generate_processing_report('./processing_report.json')
    
    print("Processing complete!")
    print(f"Total rounds processed: {report['pipeline_summary']['total_rounds_processed']}")
    print(f"Total registrations: {report['pipeline_summary']['total_registrations']}")
    print(f"Total segmentations: {report['pipeline_summary']['total_segmentations']}")


def example_individual_functions():
    """
    Example of using individual functions with better naming conventions.
    
    This demonstrates how to use the new functions directly without the
    high-level pipeline orchestrator.
    """
    from VolAlign import (
        downsample_zarr_volume,
        merge_zarr_channels,
        compute_affine_registration,
        compute_deformation_field_registration,
        distributed_nuclei_segmentation,
        upsample_segmentation_labels
    )
    
    print("\n=== Individual Function Examples ===")
    
    # Example 1: Downsample a large Zarr volume
    print("1. Downsampling Zarr volume...")
    downsample_zarr_volume(
        input_zarr_path='/path/to/large_volume.zarr',
        output_zarr_path='/path/to/downsampled_volume.zarr',
        downsample_factors=(4, 7, 7),
        chunk_size=50
    )
    
    # Example 2: Merge registration channels
    print("2. Merging registration channels...")
    merge_zarr_channels(
        channel_a_path='/path/to/405nm.zarr',
        channel_b_path='/path/to/488nm.zarr',
        output_path='/path/to/merged_registration.zarr',
        merge_strategy='mean'
    )
    
    # Example 3: Compute affine registration (replaces "initial" alignment)
    print("3. Computing affine registration...")
    affine_matrix = compute_affine_registration(
        fixed_volume_path='/path/to/fixed.zarr',
        moving_volume_path='/path/to/moving.zarr',
        voxel_spacing=[0.2, 0.1625, 0.1625],
        output_matrix_path='/path/to/affine_matrix.txt'
    )
    
    # Example 4: Compute deformation field (replaces "final" alignment)
    print("4. Computing deformation field registration...")
    aligned_path = compute_deformation_field_registration(
        fixed_zarr_path='/path/to/fixed.zarr',
        moving_zarr_path='/path/to/moving.zarr',
        affine_matrix_path='/path/to/affine_matrix.txt',
        output_directory='/path/to/output',
        output_name='registration_result',
        voxel_spacing=[0.2, 0.1625, 0.1625]
    )
    
    # Example 5: Distributed nuclei segmentation
    print("5. Running distributed nuclei segmentation...")
    segments, boxes = distributed_nuclei_segmentation(
        input_zarr_path='/path/to/405nm_channel.zarr',
        output_zarr_path='/path/to/segmentation_masks.zarr',
        model_type='cpsam'
    )
    
    # Example 6: Upsample segmentation labels
    print("6. Upsampling segmentation labels...")
    upsample_segmentation_labels(
        input_zarr_path='/path/to/downsampled_masks.zarr',
        output_zarr_path='/path/to/fullres_masks.zarr',
        upsample_factors=(4, 7, 7)
    )
    
    print("Individual function examples complete!")


if __name__ == '__main__':
    # Run the complete pipeline example
    main()
    
    # Uncomment to run individual function examples
    # example_individual_functions()