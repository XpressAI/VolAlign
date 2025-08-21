import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the processing pipeline with configuration.

        Args:
            config (Dict[str, Any]): Pipeline configuration dictionary
        """
        self.config = config
        self.working_directory = Path(config.get("working_directory", "./"))
        self.working_directory.mkdir(parents=True, exist_ok=True)

        # Default processing parameters
        self.voxel_spacing = config.get("voxel_spacing", [0.2, 0.1625, 0.1625])
        self.downsample_factors = config.get("downsample_factors", (4, 7, 7))
        self.block_size = config.get("block_size", [512, 512, 512])

        # Cluster configuration
        self.cluster_config = config.get(
            "cluster_config",
            {"n_workers": 8, "threads_per_worker": 1, "memory_limit": "150GB"},
        )

        # Pipeline state tracking
        self.pipeline_state = {
            "rounds_processed": [],
            "registration_pairs": [],
            "segmentation_results": [],
        }

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
        merge_strategy: str = "mean",
    ) -> str:
        """
        Merge 405nm and 488nm channels for robust registration.

        Args:
            channel_405_path (str): Path to 405nm channel Zarr volume
            channel_488_path (str): Path to 488nm channel Zarr volume
            output_path (str): Path for merged registration channel
            merge_strategy (str): Merging strategy ("mean", "max", "stack")

        Returns:
            str: Path to merged registration channel
        """
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

        # Step 1: Create registration channels by merging 405nm and 488nm
        fixed_reg_channel = reg_dir / f"{registration_name}_fixed_registration.zarr"
        moving_reg_channel = reg_dir / f"{registration_name}_moving_registration.zarr"

        self.create_registration_channels(
            fixed_round_data["405"], fixed_round_data["488"], str(fixed_reg_channel)
        )

        self.create_registration_channels(
            moving_round_data["405"], moving_round_data["488"], str(moving_reg_channel)
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
        input_405_channel: str,
        segmentation_output_dir: str,
        segmentation_name: str,
        downsample_for_segmentation: bool = True,
        upsample_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute nuclei segmentation workflow with optional down/upsampling.

        Args:
            input_405_channel (str): Path to 405nm channel for segmentation
            segmentation_output_dir (str): Directory for segmentation outputs
            segmentation_name (str): Base name for segmentation files
            downsample_for_segmentation (bool): Whether to downsample for processing
            upsample_results (bool): Whether to upsample results back to original resolution

        Returns:
            Dict[str, Any]: Dictionary with segmentation results and paths
        """
        seg_dir = Path(segmentation_output_dir)
        seg_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting segmentation workflow: {segmentation_name}")

        # Step 1: Optionally downsample for efficient processing
        if downsample_for_segmentation:
            downsampled_path = seg_dir / f"{segmentation_name}_405_downsampled.zarr"

            downsample_zarr_volume(
                input_zarr_path=input_405_channel,
                output_zarr_path=str(downsampled_path),
                downsample_factors=self.downsample_factors,
            )

            segmentation_input = str(downsampled_path)
        else:
            segmentation_input = input_405_channel

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
            "input_channel": input_405_channel,
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
        aligned_paths = apply_deformation_to_channels(
            reference_zarr_path=reference_round_data["405"],  # Use 405 as reference
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

    def load_pipeline_state(self, input_path: str) -> None:
        """
        Load pipeline state from JSON file.

        Args:
            input_path (str): Path to load pipeline state from
        """
        with open(input_path, "r") as f:
            state_data = json.load(f)

        self.config = state_data.get("config", self.config)
        self.pipeline_state = state_data.get("pipeline_state", self.pipeline_state)

        print(f"Pipeline state loaded: {input_path}")

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
