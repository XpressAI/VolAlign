import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
import zarr

from .distributed_processing import (
    apply_deformation_to_channels,
    compute_affine_registration,
    compute_chunk_alignment,
    compute_deformation_field_registration,
    create_registration_summary,
    distributed_nuclei_segmentation,
)
from .step_tracker import (
    PipelineStepManager,
    generate_extended_config_from_original,
    load_extended_config_if_exists,
)
from .utils import (
    convert_tiff_to_zarr,
    convert_zarr_to_tiff,
    downsample_zarr_volume,
    merge_zarr_channels,
    scale_intensity_to_uint16,
    upsample_segmentation_labels,
)


class MicroscopyProcessingPipeline:
    """
    High-level orchestrator for multi-round microscopy processing workflows.

    Manages the complete pipeline from raw data to aligned, segmented volumes
    including registration, segmentation, and channel processing.
    """

    def __init__(self, config_file: str, enable_step_tracking: bool = True):
        """
        Initialize the processing pipeline with YAML configuration file.

        Args:
            config_file (str): Path to YAML configuration file (original template)
            enable_step_tracking (bool): Enable step tracking functionality

        Raises:
            ValueError: If required configuration parameters are missing
            FileNotFoundError: If config file doesn't exist
        """
        self.config = self.load_config_from_yaml(config_file)

        # Validate and extract required parameters
        self._validate_and_extract_config()

        # NEW: Step tracking initialization
        self.enable_step_tracking = enable_step_tracking
        if enable_step_tracking:
            self.extended_config = self._initialize_extended_config()
            self.step_manager = PipelineStepManager(
                self.extended_config, str(self.working_directory)
            )

        # Pipeline state tracking (enhanced)
        self.pipeline_state = {
            "rounds_processed": [],
            "registration_pairs": [],
            "segmentation_results": [],
            "step_execution": {},  # NEW: Step execution tracking
        }

    def _validate_and_extract_config(self):
        """
        Validate and extract configuration parameters from YAML.

        Raises:
            ValueError: If required configuration parameters are missing
        """
        # Required parameters - no defaults
        required_params = [
            "working_directory",
            "voxel_spacing",
            "downsample_factors",
            "block_size",
        ]

        for param in required_params:
            if param not in self.config:
                raise ValueError(
                    f"Required parameter '{param}' missing from YAML configuration"
                )

        # Extract basic parameters
        self.working_directory = Path(self.config["working_directory"])
        self.working_directory.mkdir(parents=True, exist_ok=True)

        self.voxel_spacing = np.asarray(self.config["voxel_spacing"], dtype=float)

        # Handle both old and new downsample_factors format
        downsample_config = self.config["downsample_factors"]
        if isinstance(downsample_config, dict):
            # New format with separate factors for registration and segmentation
            self.registration_downsample_factors = tuple(
                downsample_config["registration"]
            )
            self.segmentation_downsample_factors = tuple(
                downsample_config["segmentation"]
            )
            # Keep backward compatibility
            self.downsample_factors = self.registration_downsample_factors
        else:
            # Old format - use same factors for both tasks
            self.downsample_factors = tuple(downsample_config)
            self.registration_downsample_factors = self.downsample_factors
            self.segmentation_downsample_factors = self.downsample_factors

        self.block_size = tuple(int(x) for x in self.config["block_size"])

        # Registration parameters - required
        if "registration" not in self.config:
            raise ValueError(
                "Required section 'registration' missing from YAML configuration"
            )

        self.registration_config = self.config["registration"]

        if "merge_strategy" not in self.registration_config:
            raise ValueError(
                "Required parameter 'registration.merge_strategy' missing from YAML configuration"
            )
        if "channels" not in self.registration_config:
            raise ValueError(
                "Required parameter 'registration.channels' missing from YAML configuration"
            )

        self.merge_strategy = self.registration_config["merge_strategy"]
        self.registration_channels = self.registration_config["channels"]

        # Chunk alignment configuration - optional
        self.chunk_alignment_config = self.registration_config.get(
            "chunk_alignment", {}
        )

        # Segmentation parameters - required
        if "segmentation" not in self.config:
            raise ValueError(
                "Required section 'segmentation' missing from YAML configuration"
            )

        self.segmentation_config = self.config["segmentation"]

        if "channel" not in self.segmentation_config:
            raise ValueError(
                "Required parameter 'segmentation.channel' missing from YAML configuration"
            )
        if "downsample_for_processing" not in self.segmentation_config:
            raise ValueError(
                "Required parameter 'segmentation.downsample_for_processing' missing from YAML configuration"
            )
        if "upsample_results" not in self.segmentation_config:
            raise ValueError(
                "Required parameter 'segmentation.upsample_results' missing from YAML configuration"
            )

        self.segmentation_channel = self.segmentation_config["channel"]
        self.downsample_for_segmentation = self.segmentation_config[
            "downsample_for_processing"
        ]
        self.upsample_results = self.segmentation_config["upsample_results"]

        # Multi-round data configuration - required for multi-round processing
        if "data" not in self.config:
            raise ValueError("Required section 'data' missing from YAML configuration")

        self.data_config = self.config["data"]

        if "reference_round" not in self.data_config:
            raise ValueError(
                "Required parameter 'data.reference_round' missing from YAML configuration"
            )
        if "rounds" not in self.data_config:
            raise ValueError(
                "Required parameter 'data.rounds' missing from YAML configuration"
            )

        self.reference_round = self.data_config["reference_round"]
        self.rounds_data = self.data_config["rounds"]

        # Validate that reference round exists in rounds data
        if self.reference_round not in self.rounds_data:
            raise ValueError(
                f"Reference round '{self.reference_round}' not found in data.rounds configuration"
            )

        # Cluster configuration - required
        if "cluster_config" not in self.config:
            raise ValueError(
                "Required section 'cluster_config' missing from YAML configuration"
            )

        cluster_config = self.config["cluster_config"]
        required_cluster_params = [
            "cluster_type",
            "n_workers",
            "threads_per_worker",
            "memory_limit",
        ]

        for param in required_cluster_params:
            if param not in cluster_config:
                raise ValueError(
                    f"Required parameter 'cluster_config.{param}' missing from YAML configuration"
                )

        self.cluster_config = cluster_config

        # Segmentation cluster configuration - optional, defaults to cluster_config if not provided
        if "segmentation_cluster_config" in self.config:
            segmentation_cluster_config = self.config["segmentation_cluster_config"]
            # Validate required parameters for segmentation cluster config (excluding cluster_type)
            required_segmentation_cluster_params = [
                "n_workers",
                "threads_per_worker",
                "memory_limit",
            ]
            for param in required_segmentation_cluster_params:
                if param not in segmentation_cluster_config:
                    raise ValueError(
                        f"Required parameter 'segmentation_cluster_config.{param}' missing from YAML configuration"
                    )
            self.segmentation_cluster_config = segmentation_cluster_config
        else:
            # Use regular cluster_config as fallback
            self.segmentation_cluster_config = cluster_config.copy()

    def _initialize_extended_config(self) -> Dict[str, Any]:
        """Initialize extended config with step tracking."""
        # Check if extended config already exists in working directory
        existing_config = load_extended_config_if_exists(str(self.working_directory))

        if existing_config:
            print(
                f"Loading existing extended config from: {self.working_directory}/extended_config.yaml"
            )
            return existing_config
        else:
            print("Generating new extended config with step tracking")
            extended_config = generate_extended_config_from_original(
                self.config, str(self.working_directory)
            )
            return extended_config

    @staticmethod
    def load_config_from_yaml(config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_file (str): Path to YAML configuration file

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded from: {config_file}")
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")

    def process_all_rounds_from_config(self) -> Dict[str, Dict[str, str]]:
        """
        Process all rounds defined in the configuration by converting TIFF to Zarr.

        Returns:
            Dict[str, Dict[str, str]]: Dictionary mapping round names to Zarr file paths
        """
        if not self.rounds_data:
            raise ValueError("No rounds data found in configuration")

        processed_rounds = {}

        print(f"Processing {len(self.rounds_data)} rounds from configuration...")

        for round_name, tiff_files in self.rounds_data.items():
            print(f"\nProcessing round: {round_name}")
            zarr_files = self.prepare_round_data(round_name, tiff_files)
            processed_rounds[round_name] = zarr_files

        return processed_rounds

    def run_complete_pipeline_from_config(self) -> Dict[str, Any]:
        """
        Run the complete pipeline using configuration data.
        Now includes step tracking and resume capabilities.

        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        if self.enable_step_tracking:
            return self._run_pipeline_with_step_tracking()
        else:
            # Original implementation (unchanged)
            return self._run_original_pipeline()

    def _run_pipeline_with_step_tracking(self) -> Dict[str, Any]:
        """Run complete pipeline with step-by-step tracking."""
        # 1. Check for existing progress
        if self.step_manager.has_previous_progress():
            print(
                "Previous pipeline execution detected - resuming from last checkpoint"
            )
            return self._resume_pipeline()

        # 2. Start fresh pipeline with step tracking
        print("Starting new pipeline with step tracking")
        print(
            f"Extended config saved to: {self.working_directory}/extended_config.yaml"
        )

        # Mark pipeline as started
        self.extended_config["step_execution"]["timestamps"][
            "pipeline_started"
        ] = self._get_timestamp()
        self.step_manager.save_current_state()

        return self._execute_pipeline_steps()

    def _execute_pipeline_steps(self) -> Dict[str, Any]:
        """Execute pipeline steps in order with validation and tracking."""
        results = {
            "processed_rounds": {},
            "registrations": {},
            "segmentation": {},
            "aligned_channels": {},
            "step_execution_log": [],
        }

        # Get execution order from extended config
        execution_order = self.extended_config["pipeline_steps"]["execution_order"]

        for phase in execution_order:
            print(f"\n=== Executing Phase: {phase} ===")
            phase_result = self._execute_phase(phase)
            results[phase] = phase_result

            # Update phase status and save state
            self._update_phase_status(phase, "completed")
            self.step_manager.save_current_state()

        # Mark pipeline as completed
        self.extended_config["step_execution"]["timestamps"][
            "pipeline_completed"
        ] = self._get_timestamp()
        self.step_manager.save_current_state()

        # Generate final report
        self._finalize_pipeline_execution(results)
        return results

    def _execute_phase(self, phase: str) -> Dict[str, Any]:
        """Execute all substeps in a phase."""
        phase_config = self.extended_config["pipeline_steps"]["steps"][phase]
        phase_results = {}

        # Validate phase dependencies
        if not self._validate_phase_prerequisites(phase):
            raise RuntimeError(f"Prerequisites not met for phase: {phase}")

        # Execute substeps
        for substep_id, substep_config in phase_config["substeps"].items():
            print(f"  Executing substep: {substep_id}")

            # Validate substep prerequisites
            valid, missing_deps = self.step_manager.validate_step_prerequisites(
                substep_id
            )
            if not valid:
                raise RuntimeError(
                    f"Prerequisites not met for step: {substep_id}. Missing: {missing_deps}"
                )

            # Update status to in_progress and save
            self.step_manager.update_step_status(substep_id, "in_progress")

            try:
                # Execute the substep
                substep_result = self._execute_substep(substep_id, substep_config)
                phase_results[substep_id] = substep_result

                # Validate outputs
                if self._validate_substep_outputs(substep_id, substep_config):
                    self.step_manager.update_step_status(
                        substep_id,
                        "completed",
                        outputs=substep_config.get("expected_outputs", []),
                    )
                    print(f"{substep_id} completed successfully")
                else:
                    raise RuntimeError(f"Output validation failed for {substep_id}")

            except Exception as e:
                self.step_manager.update_step_status(substep_id, "failed", error=str(e))
                print(f"{substep_id} failed: {e}")
                raise

        return phase_results

    def _execute_substep(self, substep_id: str, substep_config: Dict[str, Any]) -> Any:
        """Execute a single substep by calling the appropriate function."""
        function_name = substep_config["function_call"]
        function_args = substep_config.get("function_args", {})

        # Resolve any variable references in function_args
        resolved_args = self._resolve_function_args(function_args)

        # Call the appropriate method
        if hasattr(self, function_name):
            method = getattr(self, function_name)
            return method(**resolved_args)
        else:
            raise RuntimeError(f"Unknown function: {function_name}")

    def _resolve_function_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve variable references in function arguments."""
        resolved = {}

        for key, value in args.items():
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                # This is a variable reference - for now, pass through as-is
                # In a full implementation, you'd resolve these references
                resolved[key] = value
            else:
                resolved[key] = value

        return resolved

    def _validate_substep_outputs(
        self, substep_id: str, substep_config: Dict[str, Any]
    ) -> bool:
        """Validate that substep outputs exist and are valid."""
        expected_outputs = substep_config.get("expected_outputs", [])

        for output_path in expected_outputs:
            full_path = self.working_directory / output_path
            if not full_path.exists():
                print(f"    Missing expected output: {full_path}")
                return False

            # Additional validation based on file type
            if not self._validate_file_integrity(full_path):
                print(f"    Invalid file: {full_path}")
                return False

        return True

    def _validate_file_integrity(self, file_path: Path) -> bool:
        """Validate file integrity based on file type."""
        try:
            if file_path.suffix == ".zarr":
                # Validate Zarr file
                zarr.open(str(file_path), mode="r")
                return True
            elif file_path.suffix == ".json":
                # Validate JSON file
                with open(file_path, "r") as f:
                    json.load(f)
                return True
            elif file_path.suffix == ".txt":
                # Basic text file validation
                return file_path.stat().st_size > 0
            else:
                # For other files, just check if they exist and have size > 0
                return file_path.stat().st_size > 0
        except Exception:
            return False

    def _validate_phase_prerequisites(self, phase_name: str) -> bool:
        """Check if phase dependencies are satisfied."""
        phase_config = self.extended_config["pipeline_steps"]["steps"][phase_name]
        dependencies = phase_config.get("dependencies", [])

        for dep_phase in dependencies:
            # Check if all substeps in dependency phase are completed
            dep_phase_config = self.extended_config["pipeline_steps"]["steps"][
                dep_phase
            ]
            dep_substeps = dep_phase_config.get("substeps", {})

            for substep_config in dep_substeps.values():
                if substep_config.get("status") != "completed":
                    return False

        return True

    def _update_phase_status(self, phase_name: str, status: str) -> None:
        """Update the status of a phase."""
        self.extended_config["pipeline_steps"]["steps"][phase_name]["status"] = status

    def _resume_pipeline(self) -> Dict[str, Any]:
        """Resume pipeline from last completed step."""
        print("Resuming pipeline from last checkpoint...")

        # Get next steps to execute
        next_steps = self.step_manager.get_next_steps_to_execute()

        if not next_steps:
            print("Pipeline already completed!")
            return self.step_manager.get_final_results()

        print(f"Resuming from step: {next_steps[0]}")

        # Continue execution from where we left off
        return self._execute_remaining_steps(next_steps)

    def _execute_remaining_steps(self, remaining_steps: List[str]) -> Dict[str, Any]:
        """Execute remaining steps in the pipeline."""
        results = {
            "processed_rounds": {},
            "registrations": {},
            "segmentation": {},
            "aligned_channels": {},
            "step_execution_log": [],
        }

        # Group remaining steps by phase and execute
        phases_to_execute = set()
        for step_id in remaining_steps:
            for phase_name, phase_config in self.extended_config["pipeline_steps"][
                "steps"
            ].items():
                if step_id in phase_config.get("substeps", {}):
                    phases_to_execute.add(phase_name)
                    break

        # Execute phases in order
        execution_order = self.extended_config["pipeline_steps"]["execution_order"]
        for phase in execution_order:
            if phase in phases_to_execute:
                print(f"\n=== Resuming Phase: {phase} ===")
                phase_result = self._execute_phase(phase)
                results[phase] = phase_result

                self._update_phase_status(phase, "completed")
                self.step_manager.save_current_state()

        # Mark pipeline as completed
        self.extended_config["step_execution"]["timestamps"][
            "pipeline_completed"
        ] = self._get_timestamp()
        self.step_manager.save_current_state()

        self._finalize_pipeline_execution(results)
        return results

    def _finalize_pipeline_execution(self, results: Dict[str, Any]) -> None:
        """Finalize pipeline execution with reports."""
        print("\n=== Finalizing Pipeline Execution ===")

        # Save pipeline state
        state_path = self.working_directory / "pipeline_state.json"
        self.save_pipeline_state(str(state_path))

        # Generate processing report
        report_path = self.working_directory / "processing_report.json"
        report = self.generate_processing_report(str(report_path))
        results["report"] = report

        print(f"Pipeline completed successfully!")
        progress = self.step_manager.get_progress_report()
        print(
            f"Completed: {progress['completed_steps']}/{progress['total_steps']} steps"
        )
        print(f"Results saved to: {self.working_directory}")

    def _run_original_pipeline(self) -> Dict[str, Any]:
        """Run original pipeline implementation without step tracking."""
        if not self.rounds_data:
            raise ValueError(
                "No rounds data found in configuration. Please specify data.rounds in your YAML config."
            )

        results = {
            "processed_rounds": {},
            "registrations": {},
            "segmentation": {},
            "aligned_channels": {},
        }

        # Step 1: Process all rounds
        print("=== Step 1: Processing all rounds ===")
        processed_rounds = self.process_all_rounds_from_config()
        results["processed_rounds"] = processed_rounds

        # Get reference round data
        reference_round_zarr = processed_rounds[self.reference_round]

        # Step 2: Run segmentation on reference round
        print(
            f"\n=== Step 2: Running segmentation on reference round ({self.reference_round}) ==="
        )
        segmentation_results = self.run_segmentation_workflow(
            input_channel=reference_round_zarr[self.segmentation_channel],
            segmentation_output_dir=str(self.working_directory / "segmentation"),
            segmentation_name=f"{self.reference_round}_nuclei",
        )
        results["segmentation"] = segmentation_results

        # Step 3: Run registration for all non-reference rounds
        print("\n=== Step 3: Running registration workflows ===")
        for round_name, round_zarr in processed_rounds.items():
            if round_name == self.reference_round:
                continue  # Skip reference round

            print(f"\nRegistering {round_name} to {self.reference_round}...")
            registration_name = f"{self.reference_round}_to_{round_name}"

            registration_results = self.run_registration_workflow(
                fixed_round_data=reference_round_zarr,
                moving_round_data=round_zarr,
                registration_output_dir=str(
                    self.working_directory / "registration" / registration_name
                ),
                registration_name=registration_name,
            )
            results["registrations"][round_name] = registration_results

            # Step 4: Apply registration to all channels
            print(f"Applying registration to all channels for {round_name}...")
            
            # Check if this was chunk alignment by looking for the registration method
            use_chunk_alignment = registration_results.get("registration_method") == "chunk_alignment"
            initial_deformation_field_path = None
            
            if use_chunk_alignment:
                # For chunk alignment, get the initial deformation field and final deformation field
                initial_deformation_field_path = registration_results["initial_deformation_field"]
                # Look for the final deformation field
                final_deformation_field_path = str(self.working_directory / "registration" / f"{self.reference_round}_to_{round_name}" / f"{self.reference_round}_to_{round_name}_deformation_field.zarr")
            else:
                final_deformation_field_path = registration_results["deformation_field"]
            
            aligned_channels = self.apply_registration_to_all_channels(
                reference_round_data=reference_round_zarr,
                target_round_data=round_zarr,
                affine_matrix_path=registration_results["affine_matrix"],
                deformation_field_path=final_deformation_field_path,
                output_directory=str(self.working_directory / "aligned" / round_name),
                use_chunk_alignment=use_chunk_alignment,
                initial_deformation_field_path=initial_deformation_field_path,
            )
            results["aligned_channels"][round_name] = aligned_channels

        # Step 5: Save pipeline state and generate report
        print("\n=== Step 5: Saving results and generating reports ===")
        state_path = self.working_directory / "pipeline_state.json"
        report_path = self.working_directory / "processing_report.json"

        self.save_pipeline_state(str(state_path))
        report = self.generate_processing_report(str(report_path))
        results["report"] = report

        print(f"Complete pipeline finished successfully!")
        print(f"Processed {len(processed_rounds)} rounds")
        print(f"Completed {len(results['registrations'])} registrations")
        print(f"Segmented {segmentation_results['num_objects']} nuclei")
        print(f"Results saved to: {self.working_directory}")

        return results

    def prepare_round_data(
        self,
        round_name: str,
        tiff_files: Dict[str, str],
        output_zarr_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Convert TIFF files to Zarr format and prepare for processing.

        Args:
            round_name (str): Name identifier for the imaging round
            tiff_files (Dict[str, str]): Dictionary mapping channel names to TIFF file paths
            output_zarr_dir (Optional[str]): Directory for Zarr outputs

        Returns:
            Dict[str, str]: Dictionary mapping channel names to Zarr file paths
        """
        if output_zarr_dir is None:
            output_zarr_dir = self.working_directory / "zarr_volumes" / round_name

        output_zarr_dir = Path(output_zarr_dir)
        output_zarr_dir.mkdir(parents=True, exist_ok=True)

        zarr_files = {}

        print(f"Converting TIFF files to Zarr for round: {round_name}")

        for channel_name, tiff_path in tiff_files.items():
            zarr_path = output_zarr_dir / f"{round_name}_{channel_name}.zarr"

            print(f"Converting {channel_name}: {tiff_path} -> {zarr_path}")
            convert_tiff_to_zarr(str(tiff_path), str(zarr_path))

            zarr_files[channel_name] = str(zarr_path)

        # Update pipeline state
        self.pipeline_state["rounds_processed"].append(
            {
                "round_name": round_name,
                "zarr_files": zarr_files,
                "timestamp": self._get_timestamp(),
            }
        )

        return zarr_files

    def create_registration_channels(
        self,
        channel_paths: List[str],
        output_path: str,
        merge_strategy: Optional[str] = None,
    ) -> str:
        """
        Create registration channel from one or more input channels.

        For single channel: copies the channel data
        For multiple channels: merges them using the specified strategy

        Args:
            channel_paths (List[str]): List of channel paths to use for registration
            output_path (str): Path for output registration channel
            merge_strategy (Optional[str]): Merging strategy ("mean", "max", "stack", "single")
                                          If None, uses config value

        Returns:
            str: Path to registration channel
        """
        import shutil
        from pathlib import Path

        if merge_strategy is None:
            merge_strategy = self.merge_strategy

        output_path_obj = Path(output_path)

        # Handle single channel case
        if len(channel_paths) == 1:
            print(f"Using single channel for registration: {channel_paths[0]}")

            # Check if output already exists
            if output_path_obj.exists():
                print(f"Registration channel already exists: {output_path}")
                return output_path

            # Create parent directory if needed
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Copy the channel data
            channel_path_obj = Path(channel_paths[0])
            print(f"Copying registration channel: {channel_paths[0]} -> {output_path}")
            shutil.copytree(channel_path_obj, output_path_obj)
            print(f"Single channel registration data copied to: {output_path}")

            return output_path

        # Handle multiple channel case (original merging behavior)
        elif len(channel_paths) == 2:
            print(f"Merging registration channels using strategy: {merge_strategy}")

            merge_zarr_channels(
                channel_a_path=channel_paths[0],
                channel_b_path=channel_paths[1],
                output_path=output_path,
                merge_strategy=merge_strategy,
            )

            return output_path

        else:
            raise ValueError(
                f"Invalid number of registration channels: {len(channel_paths)}. "
                "Expected 1 (single channel) or 2 (dual channel merging). "
                f"Received channels: {channel_paths}"
            )

    def initial_global_alignment(
        self,
        fixed_round_data: Dict[str, str],
        moving_round_data: Dict[str, str],
        registration_output_dir: str,
        registration_name: str,
    ) -> Dict[str, str]:
        """
        Execute initial global alignment workflow: create registration channels and compute global affine registration.

        This method performs one-shot global alignment on downsampled volumes using traditional
        affine registration methods. Suitable for datasets with primarily linear deformation.

        Args:
            fixed_round_data (Dict[str, str]): Fixed round channel paths
            moving_round_data (Dict[str, str]): Moving round channel paths
            registration_output_dir (str): Directory for registration outputs
            registration_name (str): Base name for registration files

        Returns:
            Dict[str, str]: Dictionary with paths to initial affine registration results
        """
        reg_dir = Path(registration_output_dir)
        reg_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting initial registration workflow: {registration_name}")

        # Step 1: Create registration channels from configured channels
        fixed_reg_channel = reg_dir / f"{registration_name}_fixed_registration.zarr"
        moving_reg_channel = reg_dir / f"{registration_name}_moving_registration.zarr"

        # Get channel paths for registration
        fixed_channel_paths = [
            fixed_round_data[ch] for ch in self.registration_channels
        ]
        moving_channel_paths = [
            moving_round_data[ch] for ch in self.registration_channels
        ]

        self.create_registration_channels(
            channel_paths=fixed_channel_paths,
            output_path=str(fixed_reg_channel),
        )

        self.create_registration_channels(
            channel_paths=moving_channel_paths,
            output_path=str(moving_reg_channel),
        )

        # Step 2: Compute initial affine registration
        affine_matrix_path = reg_dir / f"{registration_name}_affine_matrix.txt"

        compute_affine_registration(
            fixed_volume_path=str(fixed_reg_channel),
            moving_volume_path=str(moving_reg_channel),
            voxel_spacing=self.voxel_spacing,
            output_matrix_path=str(affine_matrix_path),
            downsample_factors=self.registration_downsample_factors,
        )

        init_registration_results = {
            "fixed_registration_channel": str(fixed_reg_channel),
            "moving_registration_channel": str(moving_reg_channel),
            "affine_matrix": str(affine_matrix_path),
        }

        return init_registration_results

    def initial_chunk_alignment(
        self,
        fixed_round_data: Dict[str, str],
        moving_round_data: Dict[str, str],
        registration_output_dir: str,
        registration_name: str,
        downsample_factors: Optional[Tuple[int, int, int]] = None,
        interpolation_method: str = "linear",
        alignment_kwargs: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Execute initial chunk-based alignment workflow for handling non-linear deformation.

        This method is designed for datasets with significant non-linear deformation that cannot
        be handled effectively by traditional global alignment. It performs chunk-by-chunk processing:
        1. Creation of registration channels from configured channels
        2. Volume downsampling for efficient processing
        3. Distributed chunk-based deformation field computation
        4. Deformation field upsampling to original resolution

        Args:
            fixed_round_data (Dict[str, str]): Fixed round channel paths
            moving_round_data (Dict[str, str]): Moving round channel paths
            registration_output_dir (str): Directory for registration outputs
            registration_name (str): Base name for registration files
            downsample_factors (Optional[Tuple[int, int, int]]): Downsampling factors for (z, y, x)
                                                               If None, uses (1, 3, 3)
            interpolation_method (str): Interpolation method for upsampling ("linear", "bspline", "cubic")
            alignment_kwargs (Optional[Dict]): Alignment parameters for chunk-based processing
                                             If None, uses default {"blob_sizes": [8, 200], "use_gpu": True}

        Returns:
            Dict[str, str]: Dictionary with paths to distributed registration results
        """
        # Validate dependencies
        try:
            import SimpleITK as sitk
            from bigstream.piecewise_align import (
                distributed_piecewise_alignment_pipeline,
            )
            from bigstream.piecewise_transform import distributed_apply_transform
        except ImportError as e:
            raise ImportError(
                f"Chunk-based alignment requires additional dependencies: {e}. "
                "Please ensure bigstream and SimpleITK are installed."
            )

        reg_dir = Path(registration_output_dir)
        reg_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting chunk-based alignment workflow: {registration_name}")

        # Validate parameters
        if downsample_factors is not None:
            if len(downsample_factors) != 3 or any(f < 1 for f in downsample_factors):
                raise ValueError(
                    f"downsample_factors must be a tuple of 3 positive integers, got: {downsample_factors}"
                )

        if interpolation_method not in ["linear", "bspline", "cubic"]:
            print(
                f"Warning: Unknown interpolation method '{interpolation_method}', using 'linear'"
            )
            interpolation_method = "linear"

        # Step 1: Create registration channels from configured channels
        fixed_reg_channel = reg_dir / f"{registration_name}_fixed_registration.zarr"
        moving_reg_channel = reg_dir / f"{registration_name}_moving_registration.zarr"

        # Get channel paths for registration
        fixed_channel_paths = [
            fixed_round_data[ch] for ch in self.registration_channels
        ]
        moving_channel_paths = [
            moving_round_data[ch] for ch in self.registration_channels
        ]

        self.create_registration_channels(
            channel_paths=fixed_channel_paths,
            output_path=str(fixed_reg_channel),
        )

        self.create_registration_channels(
            channel_paths=moving_channel_paths,
            output_path=str(moving_reg_channel),
        )

        # Step 2: Set default parameters if not provided
        if downsample_factors is None:
            downsample_factors = (1, 3, 3)

        if alignment_kwargs is None:
            alignment_kwargs = {"blob_sizes": [2 * 4, 100 * 2], "use_gpu": True}

        # Step 3: Compute chunk-based alignment (initial deformation field only)
        print("Computing chunk-based initial deformation field...")
        initial_deformation_field_path = compute_chunk_alignment(
            fixed_zarr_path=str(fixed_reg_channel),
            moving_zarr_path=str(moving_reg_channel),
            output_directory=str(reg_dir),
            output_name=registration_name,
            voxel_spacing=self.voxel_spacing,
            downsample_factors=downsample_factors,
            interpolation_method=interpolation_method,
            block_size=self.block_size,
            alignment_kwargs=alignment_kwargs,
            cluster_config=self.cluster_config,
        )

        # Step 4: Create a dummy affine matrix (identity) since chunk alignment handles the transformation
        affine_matrix_path = reg_dir / f"{registration_name}_chunk_identity_matrix.txt"
        identity_matrix = np.eye(4)[:3, :]  # 3x4 identity matrix
        np.savetxt(affine_matrix_path, identity_matrix)

        distributed_registration_results = {
            "fixed_registration_channel": str(fixed_reg_channel),
            "moving_registration_channel": str(moving_reg_channel),
            "initial_deformation_field": initial_deformation_field_path,
            "identity_matrix": str(affine_matrix_path),  # Identity matrix for chunk alignment
            "registration_method": "chunk_alignment",
        }

        print(f"Chunk-based alignment completed: {registration_name}")
        return distributed_registration_results

    def final_deformation_registration(
        self,
        fixed_registration_channel: str,
        moving_registration_channel: str,
        affine_matrix_path: str,
        registration_output_dir: str,
        registration_name: str,
        use_chunk_alignment: bool = False,
        initial_deformation_field_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Execute final deformation registration workflow: compute deformation field registration.

        Args:
            fixed_registration_channel (str): Path to fixed registration channel
            moving_registration_channel (str): Path to moving registration channel
            affine_matrix_path (str): Path to computed affine transformation matrix
            registration_output_dir (str): Directory for registration outputs
            registration_name (str): Base name for registration files
            use_chunk_alignment (bool): Whether to use chunk-based alignment mode
            initial_deformation_field_path (Optional[str]): Path to initial deformation field
                                                           (required when use_chunk_alignment=True)

        Returns:
            Dict[str, str]: Dictionary with paths to final deformation registration results
        """
        reg_dir = Path(registration_output_dir)
        reg_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting final deformation registration workflow: {registration_name}")

        # Compute deformation field registration
        final_aligned_path = compute_deformation_field_registration(
            fixed_zarr_path=fixed_registration_channel,
            moving_zarr_path=moving_registration_channel,
            affine_matrix_path=affine_matrix_path,
            output_directory=str(reg_dir),
            output_name=registration_name,
            voxel_spacing=self.voxel_spacing,
            block_size=self.block_size,
            cluster_config=self.cluster_config,
            use_chunk_alignment=use_chunk_alignment,
            initial_deformation_field_path=initial_deformation_field_path,
        )

        deformation_field_path = reg_dir / f"{registration_name}_deformation_field.zarr"

        final_registration_results = {
            "deformation_field": str(deformation_field_path),
            "final_aligned": final_aligned_path,
        }

        return final_registration_results

    def run_registration_workflow(
        self,
        fixed_round_data: Dict[str, str],
        moving_round_data: Dict[str, str],
        registration_output_dir: str,
        registration_name: str,
    ) -> Dict[str, str]:
        """
        Execute complete two-stage registration workflow.

        Args:
            fixed_round_data (Dict[str, str]): Fixed round channel paths
            moving_round_data (Dict[str, str]): Moving round channel paths
            registration_output_dir (str): Directory for registration outputs
            registration_name (str): Base name for registration files

        Returns:
            Dict[str, str]: Dictionary with paths to registration results
        """
        reg_dir = Path(registration_output_dir)
        reg_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting registration workflow: {registration_name}")

        # Step 1: Initial registration (create channels + global affine registration)
        init_results = self.initial_global_alignment(
            fixed_round_data=fixed_round_data,
            moving_round_data=moving_round_data,
            registration_output_dir=registration_output_dir,
            registration_name=registration_name,
        )

        # Step 2: Final registration (deformation field registration)
        final_results = self.final_deformation_registration(
            fixed_registration_channel=init_results["fixed_registration_channel"],
            moving_registration_channel=init_results["moving_registration_channel"],
            affine_matrix_path=init_results["affine_matrix"],
            registration_output_dir=registration_output_dir,
            registration_name=registration_name,
        )

        # Step 3: Create registration summary
        summary_path = reg_dir / f"{registration_name}_summary.json"

        create_registration_summary(
            fixed_path=init_results["fixed_registration_channel"],
            moving_path=init_results["moving_registration_channel"],
            affine_matrix_path=init_results["affine_matrix"],
            deformation_field_path=final_results["deformation_field"],
            final_aligned_path=final_results["final_aligned"],
            output_summary_path=str(summary_path),
        )

        # Combine all results
        registration_results = {
            **init_results,
            **final_results,
            "summary": str(summary_path),
        }

        # Update pipeline state
        self.pipeline_state["registration_pairs"].append(
            {
                "registration_name": registration_name,
                "results": registration_results,
                "timestamp": self._get_timestamp(),
            }
        )

        return registration_results

    def run_segmentation_workflow(
        self,
        input_channel: str,
        segmentation_output_dir: str,
        segmentation_name: str,
        downsample_for_segmentation: Optional[bool] = None,
        upsample_results: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Execute nuclei segmentation workflow with optional down/upsampling.

        Args:
            input_channel (str): Path to segmentation channel
            segmentation_output_dir (str): Directory for segmentation outputs
            segmentation_name (str): Base name for segmentation files
            downsample_for_segmentation (Optional[bool]): Whether to downsample for processing
                                                        If None, uses config value
            upsample_results (Optional[bool]): Whether to upsample results back to original resolution
                                             If None, uses config value

        Returns:
            Dict[str, Any]: Dictionary with segmentation results and paths
        """
        if downsample_for_segmentation is None:
            downsample_for_segmentation = self.downsample_for_segmentation
        if upsample_results is None:
            upsample_results = self.upsample_results

        seg_dir = Path(segmentation_output_dir)
        seg_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting segmentation workflow: {segmentation_name}")

        # Step 1: Optionally downsample for efficient processing
        if downsample_for_segmentation:
            channel_name = self.segmentation_channel
            downsampled_path = (
                seg_dir / f"{segmentation_name}_{channel_name}_downsampled.zarr"
            )

            downsample_zarr_volume(
                input_zarr_path=input_channel,
                output_zarr_path=str(downsampled_path),
                downsample_factors=self.segmentation_downsample_factors,
            )

            segmentation_input = str(downsampled_path)
        else:
            segmentation_input = input_channel

        # Step 2: Run distributed segmentation
        segmentation_output = seg_dir / f"{segmentation_name}_segmentation.zarr"

        segments, bounding_boxes = distributed_nuclei_segmentation(
            input_zarr_path=segmentation_input,
            output_zarr_path=str(segmentation_output),
            segmentation_cluster_config=self.segmentation_cluster_config,
            temporary_directory=str(seg_dir),
        )

        # Step 3: Optionally upsample results back to original resolution
        final_segmentation_path = str(segmentation_output)

        if downsample_for_segmentation and upsample_results:
            upsampled_path = seg_dir / f"{segmentation_name}_segmentation_fullres.zarr"

            upsample_segmentation_labels(
                input_zarr_path=str(segmentation_output),
                output_zarr_path=str(upsampled_path),
                upsample_factors=self.segmentation_downsample_factors,
            )

            final_segmentation_path = str(upsampled_path)

        segmentation_results = {
            "input_channel": input_channel,
            "segmentation_masks": final_segmentation_path,
            "num_objects": len(bounding_boxes),
            "downsampled_input": (
                str(downsampled_path) if downsample_for_segmentation else None
            ),
            "downsampled_segmentation": (
                str(segmentation_output) if upsample_results else None
            ),
        }

        # Update pipeline state
        self.pipeline_state["segmentation_results"].append(
            {
                "segmentation_name": segmentation_name,
                "results": segmentation_results,
                "timestamp": self._get_timestamp(),
            }
        )

        return segmentation_results

    def apply_registration_to_all_channels(
        self,
        reference_round_data: Dict[str, str],
        target_round_data: Dict[str, str],
        affine_matrix_path: str,
        deformation_field_path: str,
        output_directory: str,
        use_chunk_alignment: bool = False,
        initial_deformation_field_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Apply computed registration to all imaging channels.

        Args:
            reference_round_data (Dict[str, str]): Reference round channel paths
            target_round_data (Dict[str, str]): Target round channel paths to align
            affine_matrix_path (str): Path to computed affine transformation matrix
            deformation_field_path (str): Path to computed deformation field
            output_directory (str): Directory for aligned channel outputs
            use_chunk_alignment (bool): Whether to use chunk-based alignment mode
            initial_deformation_field_path (Optional[str]): Path to initial deformation field
                                                           (required when use_chunk_alignment=True)

        Returns:
            Dict[str, str]: Dictionary mapping channel names to aligned paths
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        if use_chunk_alignment:
            print("Applying chunk-based registration to all imaging channels...")
        else:
            print("Applying global registration to all imaging channels...")

        # Get all target channels (excluding registration channels if they exist)
        target_channels = [
            path
            for name, path in target_round_data.items()
            if not name.endswith("_registration")
        ]

        # Apply deformation field to all channels
        # Use first registration channel as reference
        reference_channel = self.registration_channels[0]
        aligned_paths = apply_deformation_to_channels(
            reference_zarr_path=reference_round_data[reference_channel],
            channel_zarr_paths=target_channels,
            affine_matrix_path=affine_matrix_path,
            deformation_field_path=deformation_field_path,
            output_directory=str(output_dir),
            voxel_spacing=self.voxel_spacing,
            block_size=self.block_size,
            cluster_config=self.cluster_config,
            use_chunk_alignment=use_chunk_alignment,
            initial_deformation_field_path=initial_deformation_field_path,
        )

        # Create mapping from channel names to aligned paths
        aligned_channels = {}
        for i, (channel_name, _) in enumerate(target_round_data.items()):
            if not channel_name.endswith("_registration"):
                aligned_channels[channel_name] = aligned_paths[i]

        return aligned_channels

    def save_pipeline_state(self, output_path: str) -> None:
        """
        Save current pipeline state to JSON file.

        Args:
            output_path (str): Path to save pipeline state
        """
        state_with_config = {
            "config": self.config,
            "pipeline_state": self.pipeline_state,
            "timestamp": self._get_timestamp(),
        }

        with open(output_path, "w") as f:
            json.dump(state_with_config, f, indent=2, default=str)

        print(f"Pipeline state saved: {output_path}")

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def generate_processing_report(self, output_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive processing report.

        Args:
            output_path (str): Path to save processing report

        Returns:
            Dict[str, Any]: Processing report dictionary
        """
        report = {
            "pipeline_summary": {
                "total_rounds_processed": len(self.pipeline_state["rounds_processed"]),
                "total_registrations": len(self.pipeline_state["registration_pairs"]),
                "total_segmentations": len(self.pipeline_state["segmentation_results"]),
                "processing_config": self.config,
            },
            "rounds_processed": self.pipeline_state["rounds_processed"],
            "registration_results": self.pipeline_state["registration_pairs"],
            "segmentation_results": self.pipeline_state["segmentation_results"],
            "report_timestamp": self._get_timestamp(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Processing report generated: {output_path}")
        return report

    def get_pipeline_progress(self) -> Dict[str, Any]:
        """Get current pipeline progress information."""
        if not self.enable_step_tracking:
            return {"message": "Step tracking not enabled"}

        return self.step_manager.get_progress_report()

    def print_progress_summary(self) -> None:
        """Print human-readable progress summary."""
        if not self.enable_step_tracking:
            print("Step tracking not enabled")
            return

        progress = self.get_pipeline_progress()
        print(f"\n=== Pipeline Progress Summary ===")
        print(f"Overall Progress: {progress['percentage_complete']:.1f}%")
        print(
            f"Completed Steps: {progress['completed_steps']}/{progress['total_steps']}"
        )
        print(f"Current Phase: {progress.get('current_phase', 'Not started')}")
        print(f"Current Step: {progress.get('current_step', 'None')}")

        if progress.get("remaining_steps"):
            print(f"Remaining Steps: {len(progress['remaining_steps'])}")
            print("Next steps to execute:")
            for step in progress["remaining_steps"][:3]:  # Show next 3 steps
                print(f"  - {step['id']}: {step['description']}")
