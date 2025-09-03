#!/usr/bin/env python3
"""
Complete Microscopy Processing Pipeline Example with Step Tracking

This example demonstrates how to use the enhanced VolAlign package for
processing multi-round microscopy data with YAML configuration files,
including the new step tracking functionality.

The pipeline includes:
1. Data preparation (TIFF to Zarr conversion)
2. Two-stage registration (affine + deformation field)
3. Distributed nuclei segmentation
4. Channel alignment using computed transformations

New features:
- Step tracking and progress monitoring
- Resume capability for interrupted pipelines
- Extended configuration with step definitions
- Manual processing with step tracking support
"""

import os
from pathlib import Path

import yaml

from VolAlign import (
    MicroscopyProcessingPipeline,
    generate_extended_config_from_original,
    load_extended_config_if_exists,
)


def main():
    """Main pipeline example using YAML configuration with step tracking."""
    print("=== VolAlign Pipeline with YAML Configuration and Step Tracking ===")

    # Load pipeline from YAML configuration file (REQUIRED)
    print("\n--- Loading from YAML configuration file ---")
    try:
        config_file = Path(__file__).parent.parent / "config_template.yaml"

        if not config_file.exists():
            print(f"Configuration file not found: {config_file}")
            print("Please ensure you have a valid YAML configuration file.")
            print("You can use 'config_template.yaml' as a starting point.")
            return

        # Initialize pipeline with step tracking enabled (default)
        pipeline = MicroscopyProcessingPipeline(
            str(config_file), enable_step_tracking=True
        )
        print(f"Pipeline loaded successfully from: {config_file}")
        print(f"Step tracking enabled: {pipeline.enable_step_tracking}")
        print(f"Working directory: {pipeline.working_directory}")

        # Check if extended config was generated or loaded
        if hasattr(pipeline, "extended_config"):
            print(
                f"Extended config available with {len(pipeline.extended_config['pipeline_steps']['steps'])} phases"
            )
            print(
                f"Extended config saved to: {pipeline.working_directory}/extended_config.yaml"
            )

    except ValueError as e:
        print(f"Configuration validation error: {e}")
        print(
            "Please check your YAML configuration file for missing required parameters."
        )
        return
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    # Check if data is configured in YAML
    if pipeline.rounds_data:
        print(" Multi-round data found in configuration")
        print(f"  - Reference round: {pipeline.reference_round}")
        print(f"  - Available rounds: {list(pipeline.rounds_data.keys())}")

        # Show current progress if any
        print("\n--- Current Pipeline Progress ---")
        pipeline.print_progress_summary()

        # Option 1: Run complete pipeline automatically from config with step tracking
        print("\n=== Option 1: Complete Pipeline with Step Tracking ===")
        print("Running complete pipeline using data from YAML configuration...")
        print("Features:")
        print("   Automatic step tracking and progress monitoring")
        print("   Resume capability if interrupted")
        print("   Extended config generation and persistence")
        print("   Output validation for each step")

        # Uncomment the following lines to run the complete pipeline:
        """
        try:
            results = pipeline.run_complete_pipeline_from_config()
            print(" Complete pipeline finished!")
            
            # Show final progress
            pipeline.print_progress_summary()
            
            # Get detailed results
            progress = pipeline.get_pipeline_progress()
            print(f"Total execution time: {progress.get('timestamps', {}).get('pipeline_completed', 'N/A')}")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            print("Progress at failure:")
            pipeline.print_progress_summary()
        """

        print("(Complete pipeline commented out - uncomment to run with real data)")

        # Option 2: Manual step-by-step processing with step tracking
        print("\n=== Option 2: Manual Step-by-Step Processing with Step Tracking ===")
        print(
            "You can also run individual steps manually while maintaining step tracking:"
        )

        # Show how to do manual processing with step tracking
        manual_processing_with_step_tracking_example(pipeline)

        # Option 3: Resume interrupted pipeline
        print("\n=== Option 3: Resume Interrupted Pipeline ===")
        resume_pipeline_example(pipeline)

        # Option 4: Monitor progress without running
        print("\n=== Option 4: Monitor Pipeline Progress ===")
        monitor_progress_example(pipeline)

    else:
        print("No multi-round data found in configuration")
        print(
            "Please ensure your YAML configuration includes the 'data' section with 'reference_round' and 'rounds'."
        )
        print("See 'config_template.yaml' for the required structure.")
        return


def manual_processing_with_step_tracking_example(pipeline):
    """Example of manual step-by-step processing with step tracking support."""
    print("Manual processing with step tracking maintains all benefits:")
    print("   Progress tracking for each step")
    print("   Output validation")
    print("   Error handling and recovery")
    print("   State persistence")

    # Example of manual processing with step tracking (commented out)
    """
    try:
        # Step 1: Process all rounds from config
        print("Step 1: Processing all rounds...")
        processed_rounds = pipeline.process_all_rounds_from_config()
        
        # Update step tracking manually if needed
        if pipeline.enable_step_tracking:
            # The individual functions will automatically update step status
            # when called through the step tracking system
            pass
        
        # Step 2: Get reference round data
        reference_round_zarr = processed_rounds[pipeline.reference_round]
        
        # Step 3: Run segmentation on reference round
        print("Step 3: Running segmentation...")
        segmentation_results = pipeline.run_segmentation_workflow(
            input_channel=reference_round_zarr[pipeline.segmentation_channel],
            segmentation_output_dir=str(pipeline.working_directory / 'segmentation'),
            segmentation_name=f'{pipeline.reference_round}_nuclei'
        )
        
        # Step 4: Register and align all other rounds
        for round_name, round_zarr in processed_rounds.items():
            if round_name == pipeline.reference_round:
                continue
                
            print(f"Step 4: Processing {round_name}...")
            
            # Run registration
            registration_results = pipeline.run_registration_workflow(
                fixed_round_data=reference_round_zarr,
                moving_round_data=round_zarr,
                registration_output_dir=str(pipeline.working_directory / 'registration' / f'{pipeline.reference_round}_to_{round_name}'),
                registration_name=f'{pipeline.reference_round}_to_{round_name}'
            )
            
            # Apply registration to all channels
            aligned_channels = pipeline.apply_registration_to_all_channels(
                reference_round_data=reference_round_zarr,
                target_round_data=round_zarr,
                affine_matrix_path=registration_results['affine_matrix'],
                deformation_field_path=registration_results['deformation_field'],
                output_directory=str(pipeline.working_directory / 'aligned' / round_name)
            )
            
            print(f" {round_name} processing completed")
        
        # Show final progress
        if pipeline.enable_step_tracking:
            pipeline.print_progress_summary()
            
    except Exception as e:
        print(f"Manual processing failed: {e}")
        if pipeline.enable_step_tracking:
            print("Progress at failure:")
            pipeline.print_progress_summary()
    """

    print("(Manual processing example commented out - uncomment to run with real data)")


def resume_pipeline_example(pipeline):
    """Example of resuming an interrupted pipeline."""
    print("Resume functionality allows you to continue from where you left off:")
    print("   Automatically detects previous progress")
    print("   Validates completed steps")
    print("   Continues from next pending step")
    print("   Maintains all state information")

    # Check if there's previous progress
    if pipeline.enable_step_tracking:
        progress = pipeline.get_pipeline_progress()
        if progress.get("completed_steps", 0) > 0:
            print(
                f"Previous progress detected: {progress['completed_steps']}/{progress['total_steps']} steps completed"
            )
            print("To resume, simply call:")
            print("  results = pipeline.run_complete_pipeline_from_config()")
            print(
                "The pipeline will automatically resume from the last completed step."
            )
        else:
            print("No previous progress found - pipeline will start from the beginning")
    else:
        print("Step tracking not enabled - resume functionality not available")


def monitor_progress_example(pipeline):
    """Example of monitoring pipeline progress."""
    print("Progress monitoring provides detailed information:")

    if pipeline.enable_step_tracking:
        progress = pipeline.get_pipeline_progress()

        print(f"  • Total steps: {progress.get('total_steps', 0)}")
        print(f"  • Completed steps: {progress.get('completed_steps', 0)}")
        print(f"  • Progress: {progress.get('percentage_complete', 0):.1f}%")
        print(f"  • Current phase: {progress.get('current_phase', 'Not started')}")
        print(f"  • Failed steps: {len(progress.get('failed_steps', []))}")

        # Show remaining steps
        remaining = progress.get("remaining_steps", [])
        if remaining:
            print(f"  • Next {min(3, len(remaining))} steps:")
            for step in remaining[:3]:
                print(f"    - {step['id']}: {step['description']}")

        # Show extended config location
        print(f"  • Extended config: {pipeline.working_directory}/extended_config.yaml")

        # Show timestamps if available
        timestamps = progress.get("timestamps", {})
        if any(timestamps.values()):
            print("  • Timestamps:")
            for key, value in timestamps.items():
                if value:
                    print(f"    - {key}: {value}")
    else:
        print("Step tracking not enabled - detailed progress not available")


def create_extended_config_example():
    """Example of manually creating an extended config from the original template."""
    print("\n=== Example: Creating Extended Config Manually ===")

    config_file = Path(__file__).parent.parent / "config_template.yaml"
    working_dir = "./example_output"

    if not config_file.exists():
        print(f"Configuration file not found: {config_file}")
        return

    try:
        # Load the original config
        with open(config_file, "r") as f:
            original_config = yaml.safe_load(f)

        print(f" Original config loaded from: {config_file}")
        print(f"  - Reference round: {original_config['data']['reference_round']}")
        print(f"  - Available rounds: {list(original_config['data']['rounds'].keys())}")

        # Check if extended config already exists
        existing_config = load_extended_config_if_exists(working_dir)

        if existing_config:
            print(
                f" Extended config already exists in: {working_dir}/extended_config.yaml"
            )
            print("  - Loading existing extended config...")
            extended_config = existing_config
        else:
            print(" Creating new extended config...")
            # Generate extended config manually
            extended_config = generate_extended_config_from_original(
                original_config, working_dir
            )
            print(
                f" Extended config created and saved to: {working_dir}/extended_config.yaml"
            )

        # Show extended config details
        pipeline_steps = extended_config["pipeline_steps"]
        print(f"  - Pipeline phases: {len(pipeline_steps['steps'])}")
        print(f"  - Execution order: {pipeline_steps['execution_order']}")

        # Count total substeps
        total_substeps = 0
        for phase_name, phase_config in pipeline_steps["steps"].items():
            substeps = phase_config.get("substeps", {})
            total_substeps += len(substeps)
            print(
                f"    • {phase_name}: {len(substeps)} substeps - {phase_config['description']}"
            )

        print(f"  - Total substeps: {total_substeps}")

        # Show step execution metadata
        step_execution = extended_config["step_execution"]
        print(
            f"  - Progress tracking initialized: {step_execution['progress']['total_steps']} total steps"
        )

        print("\nYou can now use this extended config with the pipeline:")
        print(
            "  pipeline = MicroscopyProcessingPipeline(config_file, enable_step_tracking=True)"
        )
        print(
            "  # The pipeline will automatically load the extended config from the working directory"
        )

    except Exception as e:
        print(f"✗ Error creating extended config: {e}")
        import traceback

        traceback.print_exc()


def example_disable_step_tracking():
    """Example of running pipeline without step tracking (original behavior)."""
    print("\n=== Example: Pipeline without Step Tracking ===")

    config_file = Path(__file__).parent.parent / "config_template.yaml"

    # Initialize pipeline with step tracking disabled
    pipeline = MicroscopyProcessingPipeline(
        str(config_file), enable_step_tracking=False
    )

    print(f"Step tracking enabled: {pipeline.enable_step_tracking}")
    print("This runs the original pipeline implementation without step tracking.")

    # Try to get progress (should return message about step tracking not enabled)
    progress = pipeline.get_pipeline_progress()
    print(f"Progress info: {progress}")

    # The pipeline will run using the original implementation
    print("Pipeline will use original implementation when step tracking is disabled.")


def example_individual_functions():
    """
    Example of using individual functions with better naming conventions.

    This demonstrates how to use the new functions directly without the
    high-level pipeline orchestrator.
    """
    from VolAlign import (
        compute_affine_registration,
        compute_deformation_field_registration,
        distributed_nuclei_segmentation,
        downsample_zarr_volume,
        merge_zarr_channels,
        upsample_segmentation_labels,
    )

    print("\n=== Individual Function Examples ===")

    # Example 1: Downsample a large Zarr volume
    print("1. Downsampling Zarr volume...")
    downsample_zarr_volume(
        input_zarr_path="/path/to/large_volume.zarr",
        output_zarr_path="/path/to/downsampled_volume.zarr",
        downsample_factors=(4, 7, 7),
        chunk_size=50,
    )

    # Example 2: Merge registration channels
    print("2. Merging registration channels...")
    merge_zarr_channels(
        channel_a_path="/path/to/405nm.zarr",
        channel_b_path="/path/to/488nm.zarr",
        output_path="/path/to/merged_registration.zarr",
        merge_strategy="mean",
    )

    # Example 3: Compute affine registration (replaces "initial" alignment)
    print("3. Computing affine registration...")
    affine_matrix = compute_affine_registration(
        fixed_volume_path="/path/to/fixed.zarr",
        moving_volume_path="/path/to/moving.zarr",
        voxel_spacing=[0.2, 0.1625, 0.1625],
        output_matrix_path="/path/to/affine_matrix.txt",
    )

    # Example 4: Compute deformation field (replaces "final" alignment)
    print("4. Computing deformation field registration...")
    aligned_path = compute_deformation_field_registration(
        fixed_zarr_path="/path/to/fixed.zarr",
        moving_zarr_path="/path/to/moving.zarr",
        affine_matrix_path="/path/to/affine_matrix.txt",
        output_directory="/path/to/output",
        output_name="registration_result",
        voxel_spacing=[0.2, 0.1625, 0.1625],
    )

    # Example 4b: Using the new split registration methods for more granular control
    print("4b. Using split registration methods...")
    # Step 1: Initial registration (create channels + affine registration)
    init_results = pipeline.initial_affine_registration(
        fixed_round_data={"405": "/path/to/fixed_405.zarr", "488": "/path/to/fixed_488.zarr"},
        moving_round_data={"405": "/path/to/moving_405.zarr", "488": "/path/to/moving_488.zarr"},
        registration_output_dir="/path/to/registration_output",
        registration_name="round1_to_round2"
    )
    
    # Step 2: Final deformation registration (deformation field registration)
    final_results = pipeline.final_deformation_registration(
        fixed_registration_channel=init_results["fixed_registration_channel"],
        moving_registration_channel=init_results["moving_registration_channel"],
        affine_matrix_path=init_results["affine_matrix"],
        registration_output_dir="/path/to/registration_output",
        registration_name="round1_to_round2"
    )
    
    # Note: The original run_registration_workflow() method still works and internally
    # calls initial_affine_registration() and final_deformation_registration() for backward compatibility

    # Example 5: Distributed nuclei segmentation
    print("5. Running distributed nuclei segmentation...")
    segments, boxes = distributed_nuclei_segmentation(
        input_zarr_path="/path/to/405nm_channel.zarr",
        output_zarr_path="/path/to/segmentation_masks.zarr",
        model_type="cpsam",
        segmentation_cluster_config={
            "cluster_type": "local_cluster",
            "n_workers": 3,
            "threads_per_worker": 1,
            "memory_limit": "300GB",
            "use_local_cuda": True,
        },
    )

    # Example 6: Upsample segmentation labels
    print("6. Upsampling segmentation labels...")
    upsample_segmentation_labels(
        input_zarr_path="/path/to/downsampled_masks.zarr",
        output_zarr_path="/path/to/fullres_masks.zarr",
        upsample_factors=(4, 7, 7),
    )

    print("Individual function examples complete!")


if __name__ == "__main__":
    # Run the complete pipeline example with YAML configuration and step tracking
    main()

    # Uncomment to create extended config manually
    # create_extended_config_example()

    # Uncomment to run individual function examples
    # example_individual_functions()

    # Uncomment to see example of pipeline without step tracking
    # example_disable_step_tracking()
