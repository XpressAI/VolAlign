#!/usr/bin/env python3
"""
Complete Microscopy Processing Pipeline Example

This example demonstrates how to use the enhanced VolAlign package for
processing multi-round microscopy data with YAML configuration files.

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
    """Main pipeline example using YAML configuration."""
    print("=== VolAlign Pipeline with YAML Configuration ===")
    
    # Load pipeline from YAML configuration file (REQUIRED)
    print("\n--- Loading from YAML configuration file ---")
    try:
        config_file = Path(__file__).parent.parent / "your_config.yaml"
        
        if not config_file.exists():
            print(f"Configuration file not found: {config_file}")
            print("Please ensure you have a valid YAML configuration file.")
            print("You can use 'config_template.yaml' as a starting point.")
            return
        
        pipeline = MicroscopyProcessingPipeline(str(config_file))
        print(f"Pipeline loaded successfully from: {config_file}")
        
    except ValueError as e:
        print(f"Configuration validation error: {e}")
        print("Please check your YAML configuration file for missing required parameters.")
        return
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return
    
    # Check if data is configured in YAML
    if pipeline.rounds_data:
        print("✓ Multi-round data found in configuration")
        print(f"  - Reference round: {pipeline.reference_round}")
        print(f"  - Available rounds: {list(pipeline.rounds_data.keys())}")
        
        # Option 1: Run complete pipeline automatically from config
        print("\n=== Option 1: Complete Pipeline from Configuration ===")
        print("Running complete pipeline using data from YAML configuration...")
        
        # Uncomment the following line to run the complete pipeline:
        # results = pipeline.run_complete_pipeline_from_config()
        # print("✓ Complete pipeline finished!")
        
        print("(Complete pipeline commented out - uncomment to run with real data)")
        
        # Option 2: Manual step-by-step processing
        print("\n=== Option 2: Manual Step-by-Step Processing ===")
        print("You can also run individual steps manually:")
        
        # Example of manual processing (commented out)
        """
        # Step 1: Process all rounds from config
        processed_rounds = pipeline.process_all_rounds_from_config()
        
        # Step 2: Get reference round data
        reference_round_zarr = processed_rounds[pipeline.reference_round]
        
        # Step 3: Run segmentation on reference round
        segmentation_results = pipeline.run_segmentation_workflow(
            input_channel=reference_round_zarr[pipeline.segmentation_channel],
            segmentation_output_dir='./segmentation_results',
            segmentation_name=f'{pipeline.reference_round}_nuclei'
        )
        
        # Step 4: Register and align all other rounds
        for round_name, round_zarr in processed_rounds.items():
            if round_name == pipeline.reference_round:
                continue
                
            # Run registration
            registration_results = pipeline.run_registration_workflow(
                fixed_round_data=reference_round_zarr,
                moving_round_data=round_zarr,
                registration_output_dir=f'./registration_{round_name}',
                registration_name=f'{pipeline.reference_round}_to_{round_name}'
            )
            
            # Apply registration to all channels
            aligned_channels = pipeline.apply_registration_to_all_channels(
                reference_round_data=reference_round_zarr,
                target_round_data=round_zarr,
                deformation_field_path=registration_results['deformation_field'],
                output_directory=f'./aligned_{round_name}'
            )
        """
        
    else:
        print("No multi-round data found in configuration")
        print("Please ensure your YAML configuration includes the 'data' section with 'reference_round' and 'rounds'.")
        print("See 'config_template.yaml' for the required structure.")
        return



def example_individual_functions():
    """
    Example of using individual functions with better naming conventions.
    
    This demonstrates how to use the new functions directly without the
    high-level pipeline orchestrator.
    """
    from VolAlign import (compute_affine_registration,
                          compute_deformation_field_registration,
                          distributed_nuclei_segmentation,
                          downsample_zarr_volume, merge_zarr_channels,
                          upsample_segmentation_labels)
    
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
    # Run the complete pipeline example with YAML configuration
    main()
    
    # Uncomment to run individual function examples
    # example_individual_functions()