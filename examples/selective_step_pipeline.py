#!/usr/bin/env python3
"""
Selective Step Pipeline Example

This example allows you to run specific pipeline steps via command-line arguments,
eliminating the need to manually comment and uncomment code sections.

Usage:
    python selective_step_pipeline.py --step process_rounds
    python selective_step_pipeline.py --step segmentation
    python selective_step_pipeline.py --step registration --round round2
    python selective_step_pipeline.py --step alignment --round round2
    python selective_step_pipeline.py --step all
    python selective_step_pipeline.py --list-steps

Available steps:
    - process_rounds: Process all rounds from config (TIFF to Zarr conversion)
    - segmentation: Run nuclei segmentation on reference round
    - registration: Run registration workflow for a specific round
    - alignment: Apply registration to align channels for a specific round
    - all: Run all steps sequentially
    - progress: Show current pipeline progress
    - resume: Resume interrupted pipeline

Based on the manual_processing_with_step_tracking_example from complete_microscopy_pipeline.py
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

from VolAlign import (
    MicroscopyProcessingPipeline,
    generate_extended_config_from_original,
    load_extended_config_if_exists,
)


class SelectiveStepPipeline:
    """Pipeline wrapper that allows selective step execution."""

    def __init__(self, config_file=None):
        """Initialize the pipeline with configuration."""
        if config_file is None:
            config_file = Path(__file__).parent.parent / "config_template.yaml"

        self.config_file = Path(config_file)
        self.pipeline = None
        self.processed_rounds = None
        self.reference_round_zarr = None

        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the VolAlign pipeline."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        try:
            self.pipeline = MicroscopyProcessingPipeline(
                str(self.config_file), enable_step_tracking=True
            )
            print(f" Pipeline loaded successfully from: {self.config_file}")
            print(f" Step tracking enabled: {self.pipeline.enable_step_tracking}")
            print(f" Working directory: {self.pipeline.working_directory}")

            if not self.pipeline.rounds_data:
                raise ValueError("No multi-round data found in configuration")

        except Exception as e:
            print(f"✗ Error loading pipeline: {e}")
            raise

    def list_available_steps(self):
        """List all available steps."""
        steps = {
            "process_rounds": "Process all rounds from config (TIFF to Zarr conversion)",
            "segmentation": "Run nuclei segmentation on reference round",
            "registration": "Run registration workflow for a specific round (requires --round)",
            "alignment": "Apply registration to align channels for a specific round (requires --round)",
            "all": "Run all steps sequentially",
            "progress": "Show current pipeline progress",
            "resume": "Resume interrupted pipeline",
        }

        print("\n=== Available Pipeline Steps ===")
        for step, description in steps.items():
            print(f"  {step:15} - {description}")

        print(f"\n Reference round: {self.pipeline.reference_round}")
        print(f" Available rounds: {list(self.pipeline.rounds_data.keys())}")

        return steps

    def step_process_rounds(self):
        """Step 1: Process all rounds from config."""
        print("\n=== Step 1: Processing All Rounds ===")
        print("Converting TIFF data to Zarr format for all rounds...")

        try:
            self.processed_rounds = self.pipeline.process_all_rounds_from_config()
            self.reference_round_zarr = self.processed_rounds[
                self.pipeline.reference_round
            ]

            print(" All rounds processed successfully!")
            print(f" Processed rounds: {list(self.processed_rounds.keys())}")
            print(f" Reference round data: {self.reference_round_zarr}")

            return self.processed_rounds

        except Exception as e:
            print(f"✗ Error processing rounds: {e}")
            raise

    def step_segmentation(self):
        """Step 2: Run segmentation on reference round."""
        print("\n=== Step 2: Nuclei Segmentation ===")
        print(
            f"Running segmentation on reference round: {self.pipeline.reference_round}"
        )

        # Ensure rounds are processed first
        if self.processed_rounds is None:
            print(" Rounds not processed yet. Running process_rounds first...")
            self.step_process_rounds()

        try:
            segmentation_results = self.pipeline.run_segmentation_workflow(
                input_channel=self.reference_round_zarr[
                    self.pipeline.segmentation_channel
                ],
                segmentation_output_dir=str(
                    self.pipeline.working_directory / "segmentation"
                ),
                segmentation_name=f"{self.pipeline.reference_round}_nuclei",
            )

            print(" Segmentation completed successfully!")
            print(f" Segmentation results: {segmentation_results}")

            return segmentation_results

        except Exception as e:
            print(f" Error running segmentation: {e}")
            raise

    def step_registration(self, target_round):
        """Step 3: Run registration workflow for a specific round."""
        print(f"\n=== Step 3: Registration for {target_round} ===")
        print(
            f"Registering {target_round} to reference round {self.pipeline.reference_round}"
        )

        # Ensure rounds are processed first
        if self.processed_rounds is None:
            print(" Rounds not processed yet. Running process_rounds first...")
            self.step_process_rounds()

        if target_round == self.pipeline.reference_round:
            print(f" Skipping registration for reference round {target_round}")
            return None

        if target_round not in self.processed_rounds:
            raise ValueError(
                f"Round {target_round} not found in processed rounds: {list(self.processed_rounds.keys())}"
            )

        try:
            target_round_zarr = self.processed_rounds[target_round]

            registration_results = self.pipeline.run_registration_workflow(
                fixed_round_data=self.reference_round_zarr,
                moving_round_data=target_round_zarr,
                registration_output_dir=str(
                    self.pipeline.working_directory
                    / "registration"
                    / f"{self.pipeline.reference_round}_to_{target_round}"
                ),
                registration_name=f"{self.pipeline.reference_round}_to_{target_round}",
            )

            print(f" Registration completed for {target_round}!")
            print(f" Registration results: {registration_results}")

            return registration_results

        except Exception as e:
            print(f"✗ Error running registration for {target_round}: {e}")
            raise

    def step_alignment(self, target_round):
        """Step 4: Apply registration to align channels for a specific round."""
        print(f"\n=== Step 4: Channel Alignment for {target_round} ===")
        print(f"Applying registration to align all channels for {target_round}")

        # Ensure rounds are processed first
        if self.processed_rounds is None:
            print("⚠ Rounds not processed yet. Running process_rounds first...")
            self.step_process_rounds()

        if target_round == self.pipeline.reference_round:
            print(f"⚠ Skipping alignment for reference round {target_round}")
            return None

        if target_round not in self.processed_rounds:
            raise ValueError(
                f"Round {target_round} not found in processed rounds: {list(self.processed_rounds.keys())}"
            )

        try:
            target_round_zarr = self.processed_rounds[target_round]

            # Check if registration results exist
            registration_dir = (
                self.pipeline.working_directory
                / "registration"
                / f"{self.pipeline.reference_round}_to_{target_round}"
            )
            deformation_field_path = registration_dir / "deformation_field.zarr"

            if not deformation_field_path.exists():
                print(
                    f"⚠ Registration not found for {target_round}. Running registration first..."
                )
                registration_results = self.step_registration(target_round)
                deformation_field_path = registration_results["deformation_field"]

            aligned_channels = self.pipeline.apply_registration_to_all_channels(
                reference_round_data=self.reference_round_zarr,
                target_round_data=target_round_zarr,
                deformation_field_path=str(deformation_field_path),
                output_directory=str(
                    self.pipeline.working_directory / "aligned" / target_round
                ),
            )

            print(f"✓ Channel alignment completed for {target_round}!")
            print(f"✓ Aligned channels: {aligned_channels}")

            return aligned_channels

        except Exception as e:
            print(f"✗ Error running alignment for {target_round}: {e}")
            raise

    def step_all(self):
        """Run all steps sequentially."""
        print("\n=== Running All Pipeline Steps ===")

        try:
            # Step 1: Process rounds
            self.step_process_rounds()

            # Step 2: Segmentation
            self.step_segmentation()

            # Step 3 & 4: Registration and alignment for all non-reference rounds
            for round_name in self.processed_rounds.keys():
                if round_name == self.pipeline.reference_round:
                    continue

                print(f"\n--- Processing {round_name} ---")
                self.step_registration(round_name)
                self.step_alignment(round_name)

            print("\n✓ All pipeline steps completed successfully!")
            self.show_progress()

        except Exception as e:
            print(f" Pipeline failed: {e}")
            print("\nProgress at failure:")
            self.show_progress()
            raise

    def show_progress(self):
        """Show current pipeline progress."""
        print("\n=== Pipeline Progress ===")

        if self.pipeline.enable_step_tracking:
            self.pipeline.print_progress_summary()

            progress = self.pipeline.get_pipeline_progress()
            print(f"\n Progress Summary:")
            print(f"  - Total steps: {progress.get('total_steps', 0)}")
            print(f"  - Completed steps: {progress.get('completed_steps', 0)}")
            print(f"  - Progress: {progress.get('percentage_complete', 0):.1f}%")
            print(f"  - Current phase: {progress.get('current_phase', 'Not started')}")
            print(f"  - Failed steps: {len(progress.get('failed_steps', []))}")

        else:
            print("Step tracking not enabled - detailed progress not available")

    def resume_pipeline(self):
        """Resume interrupted pipeline."""
        print("\n=== Resuming Pipeline ===")

        if not self.pipeline.enable_step_tracking:
            print("✗ Step tracking not enabled - resume functionality not available")
            return

        progress = self.pipeline.get_pipeline_progress()
        if progress.get("completed_steps", 0) > 0:
            print(
                f"Previous progress detected: {progress['completed_steps']}/{progress['total_steps']} steps completed"
            )
            print("Resuming pipeline from last completed step...")

            try:
                results = self.pipeline.run_complete_pipeline_from_config()
                print("✓ Pipeline resumed and completed successfully!")
                self.show_progress()
                return results

            except Exception as e:
                print(f"✗ Resume failed: {e}")
                self.show_progress()
                raise
        else:
            print("No previous progress found - pipeline will start from the beginning")
            print("Use --step all to run the complete pipeline")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Selective Step Pipeline - Run specific VolAlign pipeline steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-steps                    # List available steps
  %(prog)s --step process_rounds           # Process all rounds
  %(prog)s --step segmentation             # Run segmentation
  %(prog)s --step registration --round round2  # Register round2
  %(prog)s --step alignment --round round2     # Align round2 channels
  %(prog)s --step all                      # Run complete pipeline
  %(prog)s --step progress                 # Show progress
  %(prog)s --step resume                   # Resume interrupted pipeline
        """,
    )

    parser.add_argument(
        "--step",
        choices=[
            "process_rounds",
            "segmentation",
            "registration",
            "alignment",
            "all",
            "progress",
            "resume",
        ],
        help="Pipeline step to execute",
    )

    parser.add_argument(
        "--round",
        help="Target round name (required for registration and alignment steps)",
    )

    parser.add_argument(
        "--config",
        help="Path to YAML configuration file (default: ../config_template.yaml)",
    )

    parser.add_argument(
        "--list-steps", action="store_true", help="List all available pipeline steps"
    )

    args = parser.parse_args()

    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        return

    try:
        # Initialize pipeline
        pipeline = SelectiveStepPipeline(config_file=args.config)

        # List steps if requested
        if args.list_steps:
            pipeline.list_available_steps()
            return

        # Validate step argument
        if not args.step:
            print("✗ Error: --step argument is required")
            parser.print_help()
            return

        # Validate round argument for steps that need it
        if args.step in ["registration", "alignment"] and not args.round:
            print(f"✗ Error: --round argument is required for {args.step} step")
            return

        # Execute the requested step
        print(f"Executing step: {args.step}")

        if args.step == "process_rounds":
            pipeline.step_process_rounds()

        elif args.step == "segmentation":
            pipeline.step_segmentation()

        elif args.step == "registration":
            pipeline.step_registration(args.round)

        elif args.step == "alignment":
            pipeline.step_alignment(args.round)

        elif args.step == "all":
            pipeline.step_all()

        elif args.step == "progress":
            pipeline.show_progress()

        elif args.step == "resume":
            pipeline.resume_pipeline()

        print(f"\nStep '{args.step}' completed successfully!")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
