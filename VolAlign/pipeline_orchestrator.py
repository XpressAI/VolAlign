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
    compute_deformation_field_registration,
    create_registration_summary,
    distributed_nuclei_segmentation,
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

    def __init__(self, config_file: str):
        """
        Initialize the processing pipeline with YAML configuration file.

        Args:
            config_file (str): Path to YAML configuration file

        Raises:
            ValueError: If required configuration parameters are missing
            FileNotFoundError: If config file doesn't exist
        """
        self.config = self.load_config_from_yaml(config_file)

        # Validate and extract required parameters
        self._validate_and_extract_config()

        # Pipeline state tracking
        self.pipeline_state = {
            "rounds_processed": [],
            "registration_pairs": [],
            "segmentation_results": [],
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

        self.voxel_spacing = self.config["voxel_spacing"]
        self.downsample_factors = tuple(self.config["downsample_factors"])
        self.block_size = self.config["block_size"]

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
        required_cluster_params = ["n_workers", "threads_per_worker", "memory_limit"]

        for param in required_cluster_params:
            if param not in cluster_config:
                raise ValueError(
                    f"Required parameter 'cluster_config.{param}' missing from YAML configuration"
                )

        self.cluster_config = cluster_config

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

        This method:
        1. Processes all rounds from config
        2. Runs registration between reference round and all other rounds
        3. Runs segmentation on reference round
        4. Applies registration to all channels
        5. Generates reports

        Returns:
            Dict[str, Any]: Complete pipeline results
        """
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
            aligned_channels = self.apply_registration_to_all_channels(
                reference_round_data=reference_round_zarr,
                target_round_data=round_zarr,
                deformation_field_path=registration_results["deformation_field"],
                output_directory=str(self.working_directory / "aligned" / round_name),
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
        channel_405_path: str,
        channel_488_path: str,
        output_path: str,
        merge_strategy: Optional[str] = None,
    ) -> str:
        """
        Merge 405nm and 488nm channels for robust registration.

        Args:
            channel_405_path (str): Path to 405nm channel Zarr volume
            channel_488_path (str): Path to 488nm channel Zarr volume
            output_path (str): Path for merged registration channel
            merge_strategy (Optional[str]): Merging strategy ("mean", "max", "stack")
                                          If None, uses config value

        Returns:
            str: Path to merged registration channel
        """
        if merge_strategy is None:
            merge_strategy = self.merge_strategy

        print(f"Merging registration channels: {merge_strategy}")

        merge_zarr_channels(
            channel_a_path=channel_405_path,
            channel_b_path=channel_488_path,
            output_path=output_path,
            merge_strategy=merge_strategy,
        )

        return output_path

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

        # Step 1: Create registration channels by merging configured channels
        fixed_reg_channel = reg_dir / f"{registration_name}_fixed_registration.zarr"
        moving_reg_channel = reg_dir / f"{registration_name}_moving_registration.zarr"

        # Use configured registration channels
        channel_a, channel_b = (
            self.registration_channels[0],
            self.registration_channels[1],
        )

        self.create_registration_channels(
            fixed_round_data[channel_a],
            fixed_round_data[channel_b],
            str(fixed_reg_channel),
        )

        self.create_registration_channels(
            moving_round_data[channel_a],
            moving_round_data[channel_b],
            str(moving_reg_channel),
        )

        # Step 2: Compute initial affine registration
        affine_matrix_path = reg_dir / f"{registration_name}_affine_matrix.txt"

        compute_affine_registration(
            fixed_volume_path=str(fixed_reg_channel),
            moving_volume_path=str(moving_reg_channel),
            voxel_spacing=self.voxel_spacing,
            output_matrix_path=str(affine_matrix_path),
            downsample_factors=self.downsample_factors,
        )

        # Step 3: Compute deformation field registration
        final_aligned_path = compute_deformation_field_registration(
            fixed_zarr_path=str(fixed_reg_channel),
            moving_zarr_path=str(moving_reg_channel),
            affine_matrix_path=str(affine_matrix_path),
            output_directory=str(reg_dir),
            output_name=registration_name,
            voxel_spacing=self.voxel_spacing,
            block_size=self.block_size,
            cluster_config=self.cluster_config,
        )

        # Step 4: Create registration summary
        summary_path = reg_dir / f"{registration_name}_summary.json"
        deformation_field_path = reg_dir / f"{registration_name}_deformation_field.zarr"

        create_registration_summary(
            fixed_path=str(fixed_reg_channel),
            moving_path=str(moving_reg_channel),
            affine_matrix_path=str(affine_matrix_path),
            deformation_field_path=str(deformation_field_path),
            final_aligned_path=final_aligned_path,
            output_summary_path=str(summary_path),
        )

        registration_results = {
            "fixed_registration_channel": str(fixed_reg_channel),
            "moving_registration_channel": str(moving_reg_channel),
            "affine_matrix": str(affine_matrix_path),
            "deformation_field": str(deformation_field_path),
            "final_aligned": final_aligned_path,
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
                downsample_factors=self.downsample_factors,
            )

            segmentation_input = str(downsampled_path)
        else:
            segmentation_input = input_channel

        # Step 2: Run distributed segmentation
        segmentation_output = seg_dir / f"{segmentation_name}_segmentation.zarr"

        segments, bounding_boxes = distributed_nuclei_segmentation(
            input_zarr_path=segmentation_input,
            output_zarr_path=str(segmentation_output),
            cluster_config=self.cluster_config,
            temporary_directory=str(seg_dir),
        )

        # Step 3: Optionally upsample results back to original resolution
        final_segmentation_path = str(segmentation_output)

        if downsample_for_segmentation and upsample_results:
            upsampled_path = seg_dir / f"{segmentation_name}_segmentation_fullres.zarr"

            upsample_segmentation_labels(
                input_zarr_path=str(segmentation_output),
                output_zarr_path=str(upsampled_path),
                upsample_factors=self.downsample_factors,
            )

            final_segmentation_path = str(upsampled_path)

        segmentation_results = {
            "input_channel": input_channel,
            "segmentation_masks": final_segmentation_path,
            "bounding_boxes": bounding_boxes,
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
        deformation_field_path: str,
        output_directory: str,
    ) -> Dict[str, str]:
        """
        Apply computed registration to all imaging channels.

        Args:
            reference_round_data (Dict[str, str]): Reference round channel paths
            target_round_data (Dict[str, str]): Target round channel paths to align
            deformation_field_path (str): Path to computed deformation field
            output_directory (str): Directory for aligned channel outputs

        Returns:
            Dict[str, str]: Dictionary mapping channel names to aligned paths
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Applying registration to all imaging channels...")

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
            deformation_field_path=deformation_field_path,
            output_directory=str(output_dir),
            voxel_spacing=self.voxel_spacing,
            block_size=self.block_size,
            cluster_config=self.cluster_config,
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
