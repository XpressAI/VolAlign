"""
Pipeline Step Tracker Module

This module provides step tracking functionality for the VolAlign pipeline,
including extended config generation, step status management, and progress tracking.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def generate_extended_config_from_original(
    original_config: Dict[str, Any], working_directory: str
) -> Dict[str, Any]:
    """
    Generate extended config with pipeline steps from original config.
    Saves the extended config to working_directory/extended_config.yaml

    Args:
        original_config (Dict[str, Any]): Original configuration dictionary
        working_directory (str): Working directory path

    Returns:
        Dict[str, Any]: Extended configuration dictionary
    """
    # Start with a copy of the original config
    extended_config = original_config.copy()

    # Extract rounds data to generate dynamic steps
    rounds_data = original_config.get("data", {}).get("rounds", {})
    reference_round = original_config.get("data", {}).get("reference_round")

    if not rounds_data or not reference_round:
        raise ValueError(
            "Original config must contain data.rounds and data.reference_round"
        )

    # Generate pipeline steps section
    pipeline_steps = _generate_pipeline_steps(
        rounds_data, reference_round, original_config
    )
    extended_config["pipeline_steps"] = pipeline_steps

    # Add step execution metadata
    extended_config["step_execution"] = {
        "current_step": None,
        "current_substep": None,
        "last_completed_step": None,
        "last_completed_substep": None,
        "failed_steps": [],
        "execution_log": [],
        "timestamps": {
            "pipeline_started": None,
            "last_step_completed": None,
            "pipeline_completed": None,
        },
        "progress": {
            "total_steps": _count_total_substeps(pipeline_steps),
            "completed_steps": 0,
            "percentage_complete": 0.0,
        },
    }

    # Save extended config to working directory
    save_extended_config(extended_config, working_directory)

    return extended_config


def _generate_pipeline_steps(
    rounds_data: Dict[str, Dict[str, str]],
    reference_round: str,
    original_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate pipeline steps based on rounds data."""

    # Define execution order
    execution_order = [
        "data_preparation",
        "registration_workflows",
        "segmentation_workflow",
        "channel_alignment",
        "finalization",
    ]

    steps = {}

    # 1. Data Preparation Steps
    data_prep_substeps = {}
    for round_name in rounds_data.keys():
        substep_id = f"tiff_to_zarr_{round_name}"
        data_prep_substeps[substep_id] = {
            "description": f"Convert {round_name} TIFF files to Zarr",
            "status": "pending",
            "dependencies": [],
            "expected_outputs": [
                f"zarr_volumes/{round_name}/{round_name}_{channel}.zarr"
                for channel in rounds_data[round_name].keys()
            ],
            "function_call": "prepare_round_data",
            "function_args": {
                "round_name": round_name,
                "tiff_files": f"${{data.rounds.{round_name}}}",
            },
        }

    steps["data_preparation"] = {
        "description": "Convert TIFF files to Zarr format for all rounds",
        "status": "pending",
        "phase": "data_preparation",
        "substeps": data_prep_substeps,
    }

    # 2. Registration Workflows
    registration_substeps = {}
    for round_name in rounds_data.keys():
        if round_name == reference_round:
            continue  # Skip reference round

        substep_id = f"registration_{reference_round}_to_{round_name}"
        registration_substeps[substep_id] = {
            "description": f"Register {round_name} to {reference_round} (reference)",
            "status": "pending",
            "dependencies": [
                f"tiff_to_zarr_{reference_round}",
                f"tiff_to_zarr_{round_name}",
            ],
            "expected_outputs": [
                f"registration/{reference_round}_to_{round_name}/{reference_round}_to_{round_name}_affine_matrix.txt",
                f"registration/{reference_round}_to_{round_name}/{reference_round}_to_{round_name}_deformation_field.zarr",
                f"registration/{reference_round}_to_{round_name}/{reference_round}_to_{round_name}_final_aligned.zarr",
                f"registration/{reference_round}_to_{round_name}/{reference_round}_to_{round_name}_summary.json",
            ],
            "function_call": "run_registration_workflow",
            "function_args": {
                "fixed_round_data": f"${{steps.data_preparation.substeps.tiff_to_zarr_{reference_round}.outputs}}",
                "moving_round_data": f"${{steps.data_preparation.substeps.tiff_to_zarr_{round_name}.outputs}}",
                "registration_output_dir": f"registration/{reference_round}_to_{round_name}",
                "registration_name": f"{reference_round}_to_{round_name}",
            },
        }

    steps["registration_workflows"] = {
        "description": "Register all rounds to reference round",
        "status": "pending",
        "phase": "registration",
        "dependencies": ["data_preparation"],
        "substeps": registration_substeps,
    }

    # 3. Segmentation Workflow
    segmentation_substeps = {
        "segment_reference_round": {
            "description": "Perform nuclei segmentation on reference round",
            "status": "pending",
            "dependencies": [f"tiff_to_zarr_{reference_round}"],
            "expected_outputs": [
                f"segmentation/{reference_round}_nuclei_segmentation.zarr",
                f"segmentation/{reference_round}_nuclei_segmentation_fullres.zarr",
            ],
            "function_call": "run_segmentation_workflow",
            "function_args": {
                "input_channel": f"${{steps.data_preparation.substeps.tiff_to_zarr_{reference_round}.outputs.{original_config.get('segmentation', {}).get('channel', '405')}}}",
                "segmentation_output_dir": "segmentation",
                "segmentation_name": f"{reference_round}_nuclei",
            },
        }
    }

    steps["segmentation_workflow"] = {
        "description": "Segment nuclei in reference round",
        "status": "pending",
        "phase": "segmentation",
        "dependencies": ["data_preparation"],
        "substeps": segmentation_substeps,
    }

    # 4. Channel Alignment
    alignment_substeps = {}
    for round_name in rounds_data.keys():
        if round_name == reference_round:
            continue  # Skip reference round

        substep_id = f"align_channels_{round_name}"
        alignment_substeps[substep_id] = {
            "description": f"Apply {reference_round}_to_{round_name} registration to all {round_name} channels",
            "status": "pending",
            "dependencies": [f"registration_{reference_round}_to_{round_name}"],
            "expected_outputs": [
                f"aligned/{round_name}/{round_name}_{channel}_aligned.zarr"
                for channel in rounds_data[round_name].keys()
            ],
            "function_call": "apply_registration_to_all_channels",
            "function_args": {
                "reference_round_data": f"${{steps.data_preparation.substeps.tiff_to_zarr_{reference_round}.outputs}}",
                "target_round_data": f"${{steps.data_preparation.substeps.tiff_to_zarr_{round_name}.outputs}}",
                "deformation_field_path": f"${{steps.registration_workflows.substeps.registration_{reference_round}_to_{round_name}.outputs.deformation_field}}",
                "output_directory": f"aligned/{round_name}",
            },
        }

    steps["channel_alignment"] = {
        "description": "Apply registration to all imaging channels",
        "status": "pending",
        "phase": "alignment",
        "dependencies": ["registration_workflows"],
        "substeps": alignment_substeps,
    }

    # 5. Finalization
    finalization_substeps = {
        "save_pipeline_state": {
            "description": "Save current pipeline state",
            "status": "pending",
            "expected_outputs": ["pipeline_state.json"],
            "function_call": "save_pipeline_state",
            "function_args": {"output_path": "pipeline_state.json"},
        },
        "generate_processing_report": {
            "description": "Generate comprehensive processing report",
            "status": "pending",
            "dependencies": ["save_pipeline_state"],
            "expected_outputs": ["processing_report.json"],
            "function_call": "generate_processing_report",
            "function_args": {"output_path": "processing_report.json"},
        },
    }

    steps["finalization"] = {
        "description": "Save pipeline state and generate reports",
        "status": "pending",
        "phase": "finalization",
        "dependencies": ["segmentation_workflow", "channel_alignment"],
        "substeps": finalization_substeps,
    }

    return {"execution_order": execution_order, "steps": steps}


def _count_total_substeps(pipeline_steps: Dict[str, Any]) -> int:
    """Count total number of substeps in the pipeline."""
    total = 0
    for step_config in pipeline_steps["steps"].values():
        total += len(step_config.get("substeps", {}))
    return total


def save_extended_config(
    extended_config: Dict[str, Any], working_directory: str
) -> str:
    """
    Save extended config to working directory.

    Args:
        extended_config (Dict[str, Any]): Extended configuration
        working_directory (str): Working directory path

    Returns:
        str: Path to saved extended config file
    """
    working_dir = Path(working_directory)
    working_dir.mkdir(parents=True, exist_ok=True)

    config_path = working_dir / "extended_config.yaml"

    with open(config_path, "w") as f:
        yaml.dump(extended_config, f, default_flow_style=False, indent=2)

    print(f"Extended config saved to: {config_path}")
    return str(config_path)


def load_extended_config_if_exists(working_directory: str) -> Optional[Dict[str, Any]]:
    """
    Load existing extended config from working directory if it exists.

    Args:
        working_directory (str): Working directory path

    Returns:
        Optional[Dict[str, Any]]: Extended config if exists, None otherwise
    """
    config_path = Path(working_directory) / "extended_config.yaml"

    if not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Extended config loaded from: {config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error loading extended config: {e}")
        return None


class PipelineStepManager:
    """Manages pipeline step status and execution tracking."""

    def __init__(self, extended_config: Dict[str, Any], working_directory: str):
        """
        Initialize with extended config dictionary and working directory.

        Args:
            extended_config (Dict[str, Any]): Extended configuration
            working_directory (str): Working directory for saving state
        """
        self.extended_config = extended_config
        self.working_directory = Path(working_directory)
        self.config_path = self.working_directory / "extended_config.yaml"

        # Ensure working directory exists
        self.working_directory.mkdir(parents=True, exist_ok=True)

    def check_step_completed(self, step_id: str) -> bool:
        """Check if a step is completed."""
        # Check if it's a main step or substep
        if "." in step_id:
            # It's a substep (e.g., "data_preparation.tiff_to_zarr_round1")
            phase, substep = step_id.split(".", 1)
            substeps = self.extended_config["pipeline_steps"]["steps"][phase].get(
                "substeps", {}
            )
            return substeps.get(substep, {}).get("status") == "completed"
        else:
            # It's a main step
            for phase_config in self.extended_config["pipeline_steps"][
                "steps"
            ].values():
                substeps = phase_config.get("substeps", {})
                if step_id in substeps:
                    return substeps[step_id].get("status") == "completed"
            return False

    def update_step_status(
        self,
        step_id: str,
        status: str,
        outputs: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Update step status and record outputs.
        Automatically saves updated config to working directory.
        """
        # Find the step in the config
        step_found = False

        for phase_name, phase_config in self.extended_config["pipeline_steps"][
            "steps"
        ].items():
            substeps = phase_config.get("substeps", {})
            if step_id in substeps:
                # Update substep status
                substeps[step_id]["status"] = status

                if outputs:
                    substeps[step_id]["outputs"] = outputs

                if error:
                    substeps[step_id]["error"] = error

                # Add to execution log
                log_entry = {
                    "step_id": step_id,
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                    "phase": phase_name,
                }

                if error:
                    log_entry["error"] = error

                self.extended_config["step_execution"]["execution_log"].append(
                    log_entry
                )

                # Update current step tracking
                if status == "in_progress":
                    self.extended_config["step_execution"]["current_substep"] = step_id
                    self.extended_config["step_execution"]["current_step"] = phase_name
                elif status == "completed":
                    self.extended_config["step_execution"][
                        "last_completed_substep"
                    ] = step_id
                    self.extended_config["step_execution"][
                        "last_completed_step"
                    ] = phase_name
                    self.extended_config["step_execution"]["timestamps"][
                        "last_step_completed"
                    ] = datetime.now().isoformat()

                    # Update progress
                    self._update_progress()
                elif status == "failed":
                    if (
                        step_id
                        not in self.extended_config["step_execution"]["failed_steps"]
                    ):
                        self.extended_config["step_execution"]["failed_steps"].append(
                            step_id
                        )

                step_found = True
                break

        if not step_found:
            raise ValueError(f"Step '{step_id}' not found in pipeline configuration")

        # Save updated config
        self.save_current_state()
        
        # Check if phase should be updated to completed
        if status == "completed":
            self._check_and_update_phase_status(step_id)

    def _update_progress(self) -> None:
        """Update progress statistics."""
        total_steps = 0
        completed_steps = 0

        for phase_config in self.extended_config["pipeline_steps"]["steps"].values():
            substeps = phase_config.get("substeps", {})
            total_steps += len(substeps)

            for substep_config in substeps.values():
                if substep_config.get("status") == "completed":
                    completed_steps += 1

        progress = self.extended_config["step_execution"]["progress"]
        progress["total_steps"] = total_steps
        progress["completed_steps"] = completed_steps
        progress["percentage_complete"] = (
            (completed_steps / total_steps * 100) if total_steps > 0 else 0.0
        )

    def get_next_steps_to_execute(self) -> List[str]:
        """Get list of next steps ready for execution."""
        ready_steps = []

        for phase_name in self.extended_config["pipeline_steps"]["execution_order"]:
            phase_config = self.extended_config["pipeline_steps"]["steps"][phase_name]

            # Check if phase dependencies are met
            if not self._check_phase_dependencies(phase_name):
                continue

            # Check substeps in this phase
            substeps = phase_config.get("substeps", {})
            for substep_id, substep_config in substeps.items():
                if substep_config.get("status") == "pending":
                    # Check if substep dependencies are met
                    if self.validate_step_prerequisites(substep_id)[0]:
                        ready_steps.append(substep_id)

        return ready_steps

    def _check_phase_dependencies(self, phase_name: str) -> bool:
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

    def validate_step_prerequisites(self, step_id: str) -> Tuple[bool, List[str]]:
        """Validate that all prerequisites for a step are met."""
        # Find the step
        step_config = None
        for phase_config in self.extended_config["pipeline_steps"]["steps"].values():
            substeps = phase_config.get("substeps", {})
            if step_id in substeps:
                step_config = substeps[step_id]
                break

        if not step_config:
            return False, [f"Step '{step_id}' not found"]

        dependencies = step_config.get("dependencies", [])
        missing_deps = []

        for dep_step in dependencies:
            if not self.check_step_completed(dep_step):
                missing_deps.append(dep_step)

        return len(missing_deps) == 0, missing_deps

    def get_progress_report(self) -> Dict[str, Any]:
        """Generate detailed progress report."""
        progress = self.extended_config["step_execution"]["progress"].copy()

        # Add current status information
        progress["current_phase"] = self.extended_config["step_execution"].get(
            "current_step"
        )
        progress["current_step"] = self.extended_config["step_execution"].get(
            "current_substep"
        )
        progress["last_completed_step"] = self.extended_config["step_execution"].get(
            "last_completed_substep"
        )
        progress["failed_steps"] = self.extended_config["step_execution"].get(
            "failed_steps", []
        )

        # Add remaining steps
        remaining_steps = []
        for phase_name in self.extended_config["pipeline_steps"]["execution_order"]:
            phase_config = self.extended_config["pipeline_steps"]["steps"][phase_name]
            substeps = phase_config.get("substeps", {})

            for substep_id, substep_config in substeps.items():
                if substep_config.get("status") == "pending":
                    remaining_steps.append(
                        {
                            "id": substep_id,
                            "description": substep_config.get("description", ""),
                            "phase": phase_name,
                        }
                    )

        progress["remaining_steps"] = remaining_steps
        progress["timestamps"] = self.extended_config["step_execution"]["timestamps"]

        return progress

    def has_previous_progress(self) -> bool:
        """Check if there's previous pipeline progress."""
        completed_steps = self.extended_config["step_execution"]["progress"][
            "completed_steps"
        ]
        return completed_steps > 0

    def save_current_state(self) -> None:
        """Save current step tracking state to extended config file."""
        save_extended_config(self.extended_config, str(self.working_directory))

    def get_final_results(self) -> Dict[str, Any]:
        """Get final pipeline results if completed."""
        if (
            self.extended_config["step_execution"]["progress"]["percentage_complete"]
            < 100.0
        ):
            return {"status": "incomplete", "progress": self.get_progress_report()}

        return {
            "status": "completed",
            "progress": self.get_progress_report(),
            "execution_log": self.extended_config["step_execution"]["execution_log"],
            "timestamps": self.extended_config["step_execution"]["timestamps"],
        }

    def _check_and_update_phase_status(self, completed_step_id: str) -> None:
        """Check if a phase should be marked as completed when a substep completes."""
        # Find which phase this step belongs to
        for phase_name, phase_config in self.extended_config["pipeline_steps"]["steps"].items():
            substeps = phase_config.get("substeps", {})
            if completed_step_id in substeps:
                # Check if all substeps in this phase are completed
                all_completed = True
                for substep_id, substep_config in substeps.items():
                    if substep_config.get("status") != "completed":
                        all_completed = False
                        break
                
                # Update phase status if all substeps are completed
                if all_completed and phase_config.get("status") != "completed":
                    print(f"🎉 All substeps in phase '{phase_name}' completed - updating phase status")
                    phase_config["status"] = "completed"
                    
                    # Add phase completion to execution log
                    log_entry = {
                        "step_id": phase_name,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "phase": phase_name,
                        "note": "Phase completed - all substeps finished"
                    }
                    self.extended_config["step_execution"]["execution_log"].append(log_entry)
                    
                    # Update last completed step to phase level
                    self.extended_config["step_execution"]["last_completed_step"] = phase_name
                
                break
