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
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from VolAlign import (
    MicroscopyProcessingPipeline,
    generate_extended_config_from_original,
    load_extended_config_if_exists,
)


class SelectiveStepPipeline:
    """Pipeline wrapper that allows selective step execution."""

    def __init__(self, config_file=None, debug=False, use_chunk_alignment=False):
        """Initialize the pipeline with configuration."""
        if config_file is None:
            config_file = Path(__file__).parent.parent / "config_template.yaml"

        self.config_file = Path(config_file)
        self.pipeline = None
        self.processed_rounds = None
        self.reference_round_zarr = None
        self._status_update_lock = threading.Lock()
        self.debug = debug
        self.use_chunk_alignment = use_chunk_alignment

        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the VolAlign pipeline."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        try:
            self.pipeline = MicroscopyProcessingPipeline(
                str(self.config_file), enable_step_tracking=True
            )
            
            # Coerce working directory to Path for consistent / operator usage
            from pathlib import Path
            self.pipeline.working_directory = Path(self.pipeline.working_directory)
            
            # Assert step_manager exists if step tracking is enabled
            if self.pipeline.enable_step_tracking and not hasattr(self.pipeline, "step_manager"):
                raise RuntimeError("Step tracking enabled but no step_manager present.")
            
            print(f" Pipeline loaded successfully from: {self.config_file}")
            print(f" Step tracking enabled: {self.pipeline.enable_step_tracking}")
            print(f" Working directory: {self.pipeline.working_directory}")

            if not self.pipeline.rounds_data:
                raise ValueError("No multi-round data found in configuration")

            # Ensure extended config exists
            self._ensure_extended_config()

        except Exception as e:
            print(f"✗ Error loading pipeline: {e}")
            raise

    def _ensure_extended_config(self):
        """Ensure extended config is available, create if needed."""
        if not hasattr(self.pipeline, 'extended_config') or self.pipeline.extended_config is None:
            # Try to load existing extended config
            extended_config_path = self.pipeline.working_directory / "extended_config.yaml"
            if extended_config_path.exists():
                self.pipeline.extended_config = load_extended_config_if_exists(str(extended_config_path))
            else:
                # Generate new extended config
                self.pipeline.extended_config = generate_extended_config_from_original(
                    str(self.config_file), str(self.pipeline.working_directory)
                )
        
        # Validate extended config schema
        self._validate_extended_config_schema()
        
        # Ensure required structure exists
        if "step_execution" not in self.pipeline.extended_config:
            self.pipeline.extended_config["step_execution"] = {}
        if "execution_log" not in self.pipeline.extended_config["step_execution"]:
            self.pipeline.extended_config["step_execution"]["execution_log"] = []

    def _validate_extended_config_schema(self):
        """Validate that extended config has the required structure."""
        ec = self.pipeline.extended_config
        missing = []
        for path in [
            ("pipeline_steps",),
            ("pipeline_steps", "steps"),
            ("step_execution",),
            ("step_execution", "execution_log"),
        ]:
            d = ec
            for k in path:
                if not isinstance(d, dict) or k not in d:
                    missing.append(".".join(path))
                    break
                d = d[k]
        if missing:
            raise RuntimeError(f"Extended config missing required keys: {missing}")

    def _get_substeps_safely(self, main_step_name: str):
        """Safely get substeps, handling singular/plural naming inconsistencies."""
        try:
            steps = self.pipeline.extended_config["pipeline_steps"]["steps"]
            
            # Try exact match first
            if main_step_name in steps:
                return steps[main_step_name].get("substeps", {})
            
            # Try common variations
            variations = [
                main_step_name + "s",  # singular -> plural
                main_step_name[:-1] if main_step_name.endswith("s") else main_step_name,  # plural -> singular
                main_step_name.replace("_workflow", "_workflows"),  # workflow -> workflows
                main_step_name.replace("_workflows", "_workflow"),  # workflows -> workflow
            ]
            
            for variation in variations:
                if variation in steps:
                    print(f"⚠️  Using '{variation}' instead of '{main_step_name}' for step lookup")
                    return steps[variation].get("substeps", {})
            
            # If nothing found, list available keys for debugging
            available_keys = list(steps.keys())
            raise KeyError(f"Step '{main_step_name}' not found. Available steps: {available_keys}")
            
        except KeyError as e:
            print(f"✗ Error accessing substeps for '{main_step_name}': {e}")
            return {}

    def _extract_round_name_safely(self, substep_id: str):
        """Safely extract round name from substep ID with better error handling."""
        # Try common patterns
        patterns = ["tiff_to_zarr_", "prepare_round_", "process_"]
        
        for pattern in patterns:
            if pattern in substep_id:
                return substep_id.replace(pattern, "")
        
        # Fallback: assume the round name is the last part after underscore
        parts = substep_id.split("_")
        if len(parts) > 1:
            return parts[-1]
        
        # Last resort: return the substep_id itself
        print(f"⚠️  Could not extract round name from '{substep_id}', using as-is")
        return substep_id

    def _resolve_main_step_key(self, main_step_name: str):
        """Resolve the canonical main step key, handling singular/plural variations."""
        steps = self.pipeline.extended_config["pipeline_steps"]["steps"]
        if main_step_name in steps:
            return main_step_name
        
        # Try common variations
        variations = [
            main_step_name + "s",  # singular -> plural
            main_step_name[:-1] if main_step_name.endswith("s") else None,  # plural -> singular
            main_step_name.replace("_workflow", "_workflows"),  # workflow -> workflows
            main_step_name.replace("_workflows", "_workflow"),  # workflows -> workflow
        ]
        
        for alt in variations:
            if alt and alt in steps:
                return alt
        
        raise KeyError(f"Main step '{main_step_name}' not found. Available: {list(steps.keys())}")

    def list_available_steps(self):
        """List all available steps."""
        steps = {
            "process_rounds": "Process all rounds from config (TIFF to Zarr conversion)",
            "segmentation": "Run nuclei segmentation on reference round",
            "registration": "Run registration workflow for a specific round (requires --round)",
            "initial_global_alignment": "Run initial global alignment (create channels + global affine) for a specific round (requires --round)",
            "initial_chunk_alignment": "Run initial chunk-based alignment for non-linear deformation (requires --round)",
            "final_deformation_registration": "Run final deformation registration (deformation field) for a specific round (requires --round)",
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
        print(f"\n Note: 'initial_global_alignment' and 'initial_chunk_alignment' are alternative")
        print(f"       initial registration methods. Use 'initial_global_alignment' for linear deformation")
        print(f"       and 'initial_chunk_alignment' for non-linear deformation.")
        print(f"       'final_deformation_registration' can follow either initial method.")
        print(f"       The original 'registration' step still works and calls initial_global_alignment + final internally.")

        return steps

    def _validate_outputs(self, expected_outputs):
        """Validate that all expected outputs exist, collecting all missing paths."""
        missing_paths = []
        for output_path in expected_outputs:
            full_path = self.pipeline.working_directory / output_path
            if not full_path.exists():
                missing_paths.append(str(full_path))
        
        if missing_paths:
            print(f"    ⚠️  Missing expected outputs:")
            for path in missing_paths:
                print(f"      - {path}")
            return False
        return True

    def _load_completed_outputs(self, substep_config, substep_id):
        """Load outputs for already completed substeps."""
        expected_outputs = substep_config.get("expected_outputs", [])
        if not expected_outputs:
            return {"status": "already_completed", "note": "No outputs to load"}

        def to_abs(p):
            return str(self.pipeline.working_directory / p)

        # For data preparation steps, reconstruct zarr files mapping
        if "tiff_to_zarr" in substep_id:
            round_name = self._extract_round_name_safely(substep_id)
            zarr_files = {}
            for output_path in expected_outputs:
                # Extract channel from path (e.g., "zarr_volumes/round1/round1_405.zarr" -> "405")
                channel = output_path.split('_')[-1].replace('.zarr', '')
                full_path = self.pipeline.working_directory / output_path
                if full_path.exists():
                    zarr_files[channel] = str(full_path)
            return zarr_files

        # Heuristic mapping for common workflow outputs
        named = {}
        for p in expected_outputs:
            ap = to_abs(p)
            lname = p.lower()
            if "deformation_field" in lname:
                named["deformation_field"] = ap
            elif "affine" in lname and lname.endswith((".npy", ".json", ".txt", ".yaml", ".yml")):
                named.setdefault("affine_transforms", []).append(ap)
            elif "mask" in lname or "labels" in lname:
                named.setdefault("masks", []).append(ap)
            elif "segmentation" in lname:
                named.setdefault("segmentation", []).append(ap)

        return named or {"status": "already_completed", "outputs": [to_abs(p) for p in expected_outputs]}

    def _execute_substep_with_tracking(self, substep_id, substep_config, main_step_name, execution_func, load_outputs_func=None):
        """Common pattern for executing substeps with tracking."""
        if substep_config.get("status") == "completed":
            print(f"  ⏭️  Skipping {substep_id} (already completed)")
            # Load existing outputs if function provided
            if load_outputs_func:
                return load_outputs_func()
            else:
                return self._load_completed_outputs(substep_config, substep_id)
        
        print(f"  🔄 Processing {substep_id}...")
        
        # Update status to in_progress
        self.pipeline.step_manager.update_step_status(substep_id, "in_progress")
        
        try:
            result = execution_func()
            
            # Validate outputs exist
            expected_outputs = substep_config.get("expected_outputs", [])
            if self._validate_outputs(expected_outputs):
                # Update status to completed
                self.pipeline.step_manager.update_step_status(
                    substep_id, "completed", outputs=expected_outputs
                )
                print(f"  ✅ {substep_id} completed successfully")
                
                # Check if all substeps in main step are completed and update main step status
                self._check_and_update_main_step_status(main_step_name)
            else:
                raise RuntimeError(f"Output validation failed for {substep_id}")
                
            return result
            
        except Exception as e:
            # Update status to failed
            self.pipeline.step_manager.update_step_status(substep_id, "failed", error=str(e))
            print(f"  ❌ {substep_id} failed: {e}")
            # Propagate failure to main step
            self._check_and_update_main_step_status(main_step_name)
            raise

    def _propagate_main_step_to_substeps(self, main_step_name: str, new_status: str):
        """Propagate main step status changes to all substeps."""
        if new_status == "pending":
            try:
                substeps = self._get_substeps_safely(main_step_name)
                if not substeps:
                    print(f"⚠️  No substeps found for '{main_step_name}'; nothing to reset.")
                    return
                
                for substep_id in substeps:
                    self.pipeline.step_manager.update_step_status(substep_id, "pending")
                print(f"🔄 Propagated 'pending' to {len(substeps)} substeps in '{main_step_name}'")
            except Exception as e:
                print(f"⚠️  Warning: Could not propagate status to substeps in '{main_step_name}': {e}")

    def reset_step_to_pending(self, main_step_name: str):
        """Reset a main step and all its substeps to pending status."""
        print(f"🔄 Resetting '{main_step_name}' to pending...")
        
        # Reset main step
        try:
            # Use safe lookup for main step key
            key = self._resolve_main_step_key(main_step_name)
            main_step_config = self.pipeline.extended_config["pipeline_steps"]["steps"][key]
            main_step_config["status"] = "pending"
            self._add_status_log_entry(main_step_name, "pending", "Main step reset to pending")
            
            # Reset all substeps
            self._propagate_main_step_to_substeps(main_step_name, "pending")
            
            # Save changes
            self.pipeline.step_manager.save_current_state()
            print(f"✅ Successfully reset '{main_step_name}' and all substeps to pending")
            
        except Exception as e:
            print(f"❌ Failed to reset '{main_step_name}': {e}")

    def step_process_rounds(self):
        """Step 1: Process all rounds from config."""
        print("\n=== Step 1: Processing All Rounds ===")
        print("Converting TIFF data to Zarr format for all rounds...")

        # DIAGNOSTIC: Check if step tracking is enabled (only if debug mode)
        if self.debug:
            print(f"🔍 DEBUG: Step tracking enabled: {self.pipeline.enable_step_tracking}")
            if self.pipeline.enable_step_tracking:
                print(f"🔍 DEBUG: Extended config path: {self.pipeline.working_directory}/extended_config.yaml")
                print(f"🔍 DEBUG: Step manager available: {hasattr(self.pipeline, 'step_manager')}")

        try:
            # If step tracking is enabled, process rounds through step tracking system
            if self.pipeline.enable_step_tracking:
                self.processed_rounds = self._process_rounds_with_step_tracking()
            else:
                # Fallback to direct processing
                self.processed_rounds = self.pipeline.process_all_rounds_from_config()
            
            # Guard after processing rounds
            self.reference_round_zarr = self.processed_rounds.get(self.pipeline.reference_round)
            if self.reference_round_zarr is None:
                raise RuntimeError(
                    f"Reference round '{self.pipeline.reference_round}' missing from processed rounds: "
                    f"{list(self.processed_rounds.keys())}"
                )

            print("✓ All rounds processed successfully!")
            print(f"✓ Processed rounds: {list(self.processed_rounds.keys())}")
            print(f"✓ Reference round data: {self._summarize_result(self.reference_round_zarr, 'reference round')}")

            # DIAGNOSTIC: Check if step tracking was updated (only if debug mode)
            if self.debug and self.pipeline.enable_step_tracking:
                progress = self.pipeline.get_pipeline_progress()
                print(f"🔍 DEBUG: Progress after processing: {progress.get('completed_steps', 0)}/{progress.get('total_steps', 0)} steps")
                print(f"🔍 DEBUG: Percentage complete: {progress.get('percentage_complete', 0):.1f}%")

            return self.processed_rounds

        except Exception as e:
            print(f"✗ Error processing rounds: {e}")
            raise

    def _process_rounds_with_step_tracking(self):
        """Process rounds with proper step tracking integration."""
        processed_rounds = {}
        
        # Get all data preparation substeps using safe lookup
        data_prep_steps = self._get_substeps_safely("data_preparation")
        if not data_prep_steps:
            raise RuntimeError(
                "No 'data_preparation' substeps found in extended_config['pipeline_steps']['steps'].\n"
                "Check the extended config generation and step names (singular/plural)."
            )
        
        for substep_id, substep_config in data_prep_steps.items():
            # Extract round name safely
            round_name = self._extract_round_name_safely(substep_id)
            
            # Fix closure binding by using default parameters
            def execute_round_processing(round_name=round_name):
                # Process the round with better error context
                try:
                    tiff_files = self.pipeline.rounds_data[round_name]
                    zarr_files = self.pipeline.prepare_round_data(round_name, tiff_files)
                    return zarr_files
                except KeyError:
                    raise RuntimeError(
                        f"Round '{round_name}' (from substep '{substep_id}') not found in rounds_data. "
                        f"Available rounds: {list(self.pipeline.rounds_data.keys())}"
                    )
            
            def load_completed_outputs(substep_config=substep_config):
                # Load existing zarr files for completed rounds
                expected_outputs = substep_config.get("expected_outputs", [])
                zarr_files = {}
                for output_path in expected_outputs:
                    # Extract channel from path (e.g., "zarr_volumes/round1/round1_405.zarr" -> "405")
                    channel = output_path.split('_')[-1].replace('.zarr', '')
                    full_path = self.pipeline.working_directory / output_path
                    if full_path.exists():
                        zarr_files[channel] = str(full_path)
                return zarr_files
            
            # Use common execution pattern with custom output loader
            result = self._execute_substep_with_tracking(
                substep_id, substep_config, "data_preparation",
                execute_round_processing, load_completed_outputs
            )
            
            processed_rounds[round_name] = result
        
        return processed_rounds

    def step_segmentation(self):
        """Step 2: Run segmentation on reference round."""
        print("\n=== Step 2: Nuclei Segmentation ===")
        print(
            f"Running segmentation on reference round: {self.pipeline.reference_round}"
        )

        # Ensure rounds are processed first
        if self.processed_rounds is None:
            print("⚠️ Rounds not processed yet. Running process_rounds first...")
            self.step_process_rounds()

        try:
            # If step tracking is enabled, run segmentation through step tracking system
            if self.pipeline.enable_step_tracking:
                segmentation_results = self._run_segmentation_with_step_tracking()
            else:
                # Fallback to direct processing
                segmentation_results = self.pipeline.run_segmentation_workflow(
                    input_channel=self.reference_round_zarr[
                        self.pipeline.segmentation_channel
                    ],
                    segmentation_output_dir=str(
                        self.pipeline.working_directory / "segmentation"
                    ),
                    segmentation_name=f"{self.pipeline.reference_round}_nuclei",
                )

            print("✓ Segmentation completed successfully!")
            print(f"✓ Segmentation results: {self._summarize_result(segmentation_results, 'segmentation')}")

            return segmentation_results

        except Exception as e:
            print(f"✗ Error running segmentation: {e}")
            raise

    def _run_segmentation_with_step_tracking(self):
        """Run segmentation with proper step tracking integration."""
        # Get segmentation substep using safe lookup
        seg_steps = self._get_substeps_safely("segmentation_workflow")
        if not seg_steps:
            raise RuntimeError(
                "No 'segmentation_workflow' substeps found in extended_config['pipeline_steps']['steps'].\n"
                "Check the extended config generation and step names (singular/plural)."
            )
        
        substep_id = "segment_reference_round"
        if substep_id not in seg_steps:
            raise KeyError(f"Substep '{substep_id}' not found in segmentation steps. Available: {list(seg_steps.keys())}")
        
        substep_config = seg_steps[substep_id]
        
        def execute_segmentation():
            # Run the segmentation
            return self.pipeline.run_segmentation_workflow(
                input_channel=self.reference_round_zarr[
                    self.pipeline.segmentation_channel
                ],
                segmentation_output_dir=str(
                    self.pipeline.working_directory / "segmentation"
                ),
                segmentation_name=f"{self.pipeline.reference_round}_nuclei",
            )
        
        # Use common execution pattern
        return self._execute_substep_with_tracking(
            substep_id, substep_config, "segmentation_workflow", execute_segmentation
        )

    def step_registration(self, target_round):
        """Step 3: Run registration workflow for a specific round."""
        print(f"\n=== Step 3: Registration for {target_round} ===")
        print(
            f"Registering {target_round} to reference round {self.pipeline.reference_round}"
        )

        # Ensure rounds are processed first
        if self.processed_rounds is None:
            print("⚠️ Rounds not processed yet. Running process_rounds first...")
            self.step_process_rounds()

        if target_round == self.pipeline.reference_round:
            print(f"⏭️ Skipping registration for reference round {target_round}")
            return None

        if target_round not in self.processed_rounds:
            raise ValueError(
                f"Round {target_round} not found in processed rounds: {list(self.processed_rounds.keys())}"
            )

        try:
            # If step tracking is enabled, run registration through step tracking system
            if self.pipeline.enable_step_tracking:
                registration_results = self._run_registration_with_step_tracking(target_round)
            else:
                # Fallback to direct processing
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

            print(f"✓ Registration completed for {target_round}!")
            print(f"✓ Registration results: {self._summarize_result(registration_results, 'registration')}")

            return registration_results

        except Exception as e:
            print(f"✗ Error running registration for {target_round}: {e}")
            raise

    def _run_registration_with_step_tracking(self, target_round):
        """Run registration with proper step tracking integration."""
        # Get registration substep using safe lookup
        reg_steps = self._get_substeps_safely("registration_workflows")
        substep_id = f"registration_{self.pipeline.reference_round}_to_{target_round}"
        
        if substep_id not in reg_steps:
            raise KeyError(f"Substep '{substep_id}' not found in registration steps. Available: {list(reg_steps.keys())}")
        
        substep_config = reg_steps[substep_id]
        
        def execute_registration():
            target_round_zarr = self.processed_rounds[target_round]
            
            # Run the registration
            return self.pipeline.run_registration_workflow(
                fixed_round_data=self.reference_round_zarr,
                moving_round_data=target_round_zarr,
                registration_output_dir=str(
                    self.pipeline.working_directory
                    / "registration"
                    / f"{self.pipeline.reference_round}_to_{target_round}"
                ),
                registration_name=f"{self.pipeline.reference_round}_to_{target_round}",
            )
        
        # Use common execution pattern
        return self._execute_substep_with_tracking(
            substep_id, substep_config, "registration_workflows", execute_registration
        )

    def step_initial_global_alignment(self, target_round):
        """Step 3a: Run initial global alignment (create channels + global affine) for a specific round."""
        print(f"\n=== Step 3a: Initial Global Alignment for {target_round} ===")
        print(
            f"Running initial global alignment (create channels + global affine) for {target_round} to reference round {self.pipeline.reference_round}"
        )

        # Ensure rounds are processed first
        if self.processed_rounds is None:
            print("⚠️ Rounds not processed yet. Running process_rounds first...")
            self.step_process_rounds()

        if target_round == self.pipeline.reference_round:
            print(f"⏭️ Skipping initial registration for reference round {target_round}")
            return None

        if target_round not in self.processed_rounds:
            raise ValueError(
                f"Round {target_round} not found in processed rounds: {list(self.processed_rounds.keys())}"
            )

        try:
            target_round_zarr = self.processed_rounds[target_round]
            init_results = self.pipeline.initial_global_alignment(
                fixed_round_data=self.reference_round_zarr,
                moving_round_data=target_round_zarr,
                registration_output_dir=str(
                    self.pipeline.working_directory
                    / "registration"
                    / f"{self.pipeline.reference_round}_to_{target_round}"
                ),
                registration_name=f"{self.pipeline.reference_round}_to_{target_round}",
            )

            print(f"✓ Initial registration completed for {target_round}!")
            print(f"✓ Initial registration results: {self._summarize_result(init_results, 'initial registration')}")

            return init_results

        except Exception as e:
            print(f"✗ Error running initial registration for {target_round}: {e}")
            raise

    def step_initial_chunk_alignment(self, target_round):
        """Step 3a-alt: Run initial chunk-based alignment for non-linear deformation."""
        print(f"\n=== Step 3a-alt: Initial Chunk-Based Alignment for {target_round} ===")
        print(
            f"Running chunk-based alignment for non-linear deformation: {target_round} to reference round {self.pipeline.reference_round}"
        )

        # Ensure rounds are processed first
        if self.processed_rounds is None:
            print("⚠️ Rounds not processed yet. Running process_rounds first...")
            self.step_process_rounds()

        if target_round == self.pipeline.reference_round:
            print(f"⏭️ Skipping chunk alignment for reference round {target_round}")
            return None

        if target_round not in self.processed_rounds:
            raise ValueError(
                f"Round {target_round} not found in processed rounds: {list(self.processed_rounds.keys())}"
            )

        try:
            target_round_zarr = self.processed_rounds[target_round]
            
            # Get chunk alignment parameters from config if available
            chunk_config = self.pipeline.chunk_alignment_config
            downsample_factors = tuple(chunk_config.get('downsample_factors', [1, 3, 3]))
            interpolation_method = chunk_config.get('interpolation_method', 'linear')
            alignment_kwargs = chunk_config.get('alignment_kwargs', {"blob_sizes": [8, 200], "use_gpu": True})
            
            chunk_results = self.pipeline.initial_chunk_alignment(
                fixed_round_data=self.reference_round_zarr,
                moving_round_data=target_round_zarr,
                registration_output_dir=str(
                    self.pipeline.working_directory
                    / "registration"
                    / f"{self.pipeline.reference_round}_to_{target_round}"
                ),
                registration_name=f"{self.pipeline.reference_round}_to_{target_round}",
                downsample_factors=downsample_factors,
                interpolation_method=interpolation_method,
                alignment_kwargs=alignment_kwargs,
            )

            print(f"✓ Chunk-based alignment completed for {target_round}!")
            print(f"✓ Chunk alignment results: {self._summarize_result(chunk_results, 'chunk alignment')}")

            return chunk_results

        except Exception as e:
            print(f"✗ Error running chunk alignment for {target_round}: {e}")
            raise

    def step_final_deformation_registration(self, target_round, use_chunk_alignment=None):
        """Step 3b: Run final deformation registration (deformation field) for a specific round."""
        if use_chunk_alignment is None:
            use_chunk_alignment = self.use_chunk_alignment
            
        if use_chunk_alignment:
            print(f"\n=== Step 3b: Final Deformation Registration for {target_round} (Chunk Alignment Mode) ===")
            print(f"Note: When using chunk-based alignment, the initial deformation field is already computed.")
            print(f"Checking for chunk alignment results for {target_round}...")
        else:
            print(f"\n=== Step 3b: Final Deformation Registration for {target_round} ===")
            print(
                f"Running final deformation registration (deformation field) for {target_round} to reference round {self.pipeline.reference_round}"
            )

        # Ensure rounds are processed first
        if self.processed_rounds is None:
            print("⚠️ Rounds not processed yet. Running process_rounds first...")
            self.step_process_rounds()

        if target_round == self.pipeline.reference_round:
            print(f"⏭️ Skipping final registration for reference round {target_round}")
            return None

        if target_round not in self.processed_rounds:
            raise ValueError(
                f"Round {target_round} not found in processed rounds: {list(self.processed_rounds.keys())}"
            )

        try:
            registration_dir = (
                self.pipeline.working_directory
                / "registration"
                / f"{self.pipeline.reference_round}_to_{target_round}"
            )
            registration_name = f"{self.pipeline.reference_round}_to_{target_round}"
            
            if use_chunk_alignment:
                # Check for chunk alignment initial deformation field
                initial_deformation_field = registration_dir / f"{registration_name}_initial_deformation_field.zarr"
                
                if not initial_deformation_field.exists():
                    print(f"⚠️ Chunk alignment initial deformation field not found for {target_round}. Running chunk alignment first...")
                    init_results = self.step_initial_chunk_alignment(target_round)
                    if not init_results:
                        raise RuntimeError(f"Failed to generate initial deformation field for {target_round}")
                
                # Get the registration channels and identity matrix from initial chunk alignment
                fixed_reg_channel = registration_dir / f"{registration_name}_fixed_registration.zarr"
                moving_reg_channel = registration_dir / f"{registration_name}_moving_registration.zarr"
                identity_matrix_path = registration_dir / f"{registration_name}_chunk_identity_matrix.txt"
                
                if not all([fixed_reg_channel.exists(), moving_reg_channel.exists(), identity_matrix_path.exists()]):
                    print(f"⚠️ Registration channels not found for {target_round}. Running chunk alignment first...")
                    init_results = self.step_initial_chunk_alignment(target_round)
                    if not init_results:
                        raise RuntimeError(f"Failed to generate registration channels for {target_round}")
                
                # Run final deformation registration with chunk alignment mode
                final_results = self.pipeline.final_deformation_registration(
                    fixed_registration_channel=str(fixed_reg_channel),
                    moving_registration_channel=str(moving_reg_channel),
                    affine_matrix_path=str(identity_matrix_path),
                    registration_output_dir=str(registration_dir),
                    registration_name=registration_name,
                    use_chunk_alignment=True,
                    initial_deformation_field_path=str(initial_deformation_field),
                )
            else:
                # Traditional affine registration path
                fixed_reg_channel = registration_dir / f"{registration_name}_fixed_registration.zarr"
                moving_reg_channel = registration_dir / f"{registration_name}_moving_registration.zarr"
                affine_matrix_path = registration_dir / f"{registration_name}_affine_matrix.txt"

                if not all([fixed_reg_channel.exists(), moving_reg_channel.exists(), affine_matrix_path.exists()]):
                    print(f"⚠️ Initial global alignment not found for {target_round}. Running initial registration first...")
                    init_results = self.step_initial_global_alignment(target_round)
                    if init_results:
                        fixed_reg_channel = init_results["fixed_registration_channel"]
                        moving_reg_channel = init_results["moving_registration_channel"]
                        affine_matrix_path = init_results["affine_matrix"]

                final_results = self.pipeline.final_deformation_registration(
                    fixed_registration_channel=str(fixed_reg_channel),
                    moving_registration_channel=str(moving_reg_channel),
                    affine_matrix_path=str(affine_matrix_path),
                    registration_output_dir=str(registration_dir),
                    registration_name=registration_name,
                )

                print(f"✓ Final registration completed for {target_round}!")
                print(f"✓ Final registration results: {self._summarize_result(final_results, 'final registration')}")

                return final_results

        except Exception as e:
            print(f"✗ Error running final registration for {target_round}: {e}")
            raise

    def _get_deformation_field_path(self, target_round, substep_config):
        """Get deformation field path from substep config or construct fallback."""
        # Try to get from expected outputs first
        expected_outputs = substep_config.get("expected_outputs", [])
        for output_path in expected_outputs:
            if "deformation_field" in output_path:
                return self.pipeline.working_directory / output_path
        
        # Fallback to hardcoded path (for backward compatibility)
        registration_dir = (
            self.pipeline.working_directory
            / "registration"
            / f"{self.pipeline.reference_round}_to_{target_round}"
        )
        return registration_dir / f"{self.pipeline.reference_round}_to_{target_round}_deformation_field.zarr"

    def _get_affine_matrix_path(self, target_round, substep_config):
        """Get affine matrix path from substep config or construct fallback."""
        # Try to get from expected outputs first
        expected_outputs = substep_config.get("expected_outputs", [])
        for output_path in expected_outputs:
            if "affine_matrix" in output_path:
                return self.pipeline.working_directory / output_path
        
        # Fallback to hardcoded path (for backward compatibility)
        registration_dir = (
            self.pipeline.working_directory
            / "registration"
            / f"{self.pipeline.reference_round}_to_{target_round}"
        )
        return registration_dir / f"{self.pipeline.reference_round}_to_{target_round}_affine_matrix.txt"

    def step_alignment(self, target_round):
        """Step 4: Apply registration to align channels for a specific round."""
        print(f"\n=== Step 4: Channel Alignment for {target_round} ===")
        print(f"Applying registration to align all channels for {target_round}")

        # Ensure rounds are processed first
        if self.processed_rounds is None:
            print("⚠️ Rounds not processed yet. Running process_rounds first...")
            self.step_process_rounds()

        if target_round == self.pipeline.reference_round:
            print(f"⏭️ Skipping alignment for reference round {target_round}")
            return None

        if target_round not in self.processed_rounds:
            raise ValueError(
                f"Round {target_round} not found in processed rounds: {list(self.processed_rounds.keys())}"
            )

        try:
            # If step tracking is enabled, run alignment through step tracking system
            if self.pipeline.enable_step_tracking:
                aligned_channels = self._run_alignment_with_step_tracking(target_round)
            else:
                # Fallback to direct processing
                target_round_zarr = self.processed_rounds[target_round]

                # Check if we should use chunk alignment mode
                use_chunk_alignment = self.use_chunk_alignment
                initial_deformation_field_path = None
                
                # Try to locate reg substep config (if present) for path hints
                reg_steps = self._get_substeps_safely("registration_workflows")
                reg_substep_id = f"registration_{self.pipeline.reference_round}_to_{target_round}"
                reg_substep_config = reg_steps.get(reg_substep_id, {})
                deformation_field_path = self._get_deformation_field_path(target_round, reg_substep_config)
                affine_matrix_path = self._get_affine_matrix_path(target_round, reg_substep_config)

                if use_chunk_alignment:
                    # For chunk alignment, look for initial deformation field and final deformation field
                    registration_dir = (
                        self.pipeline.working_directory
                        / "registration"
                        / f"{self.pipeline.reference_round}_to_{target_round}"
                    )
                    initial_deformation_field_path = str(registration_dir / f"{self.pipeline.reference_round}_to_{target_round}_initial_deformation_field.zarr")
                    final_deformation_field_path = str(registration_dir / f"{self.pipeline.reference_round}_to_{target_round}_deformation_field.zarr")
                    identity_matrix_path = str(registration_dir / f"{self.pipeline.reference_round}_to_{target_round}_chunk_identity_matrix.txt")
                    
                    # Check if chunk alignment files exist
                    if not all([Path(initial_deformation_field_path).exists(), Path(final_deformation_field_path).exists(), Path(identity_matrix_path).exists()]):
                        print(f"⚠️ Chunk alignment not found for {target_round}. Running chunk alignment first...")
                        self.step_initial_chunk_alignment(target_round)
                        self.step_final_deformation_registration(target_round, use_chunk_alignment=True)
                    
                    deformation_field_path = Path(final_deformation_field_path)
                    affine_matrix_path = Path(identity_matrix_path)
                else:
                    # Traditional global alignment
                    if not deformation_field_path.exists() or not affine_matrix_path.exists():
                        print(f"⚠️ Registration not found for {target_round}. Running registration first...")
                        registration_results = self.step_registration(target_round)
                        if isinstance(registration_results, dict):
                            deformation_field_path = registration_results.get("deformation_field", deformation_field_path)
                            affine_matrix_path = registration_results.get("affine_matrix", affine_matrix_path)

                aligned_channels = self.pipeline.apply_registration_to_all_channels(
                    reference_round_data=self.reference_round_zarr,
                    target_round_data=target_round_zarr,
                    affine_matrix_path=str(affine_matrix_path),
                    deformation_field_path=str(deformation_field_path),
                    output_directory=str(
                        self.pipeline.working_directory / "aligned" / target_round
                    ),
                    use_chunk_alignment=use_chunk_alignment,
                    initial_deformation_field_path=initial_deformation_field_path,
                )

            self._print_done(f"Channel alignment for {target_round}", aligned_channels)
            return aligned_channels

        except Exception as e:
            print(f"✗ Error running alignment for {target_round}: {e}")
            raise

    def _run_alignment_with_step_tracking(self, target_round):
        """Run alignment with proper step tracking integration."""
        # Get alignment substep using safe lookup
        align_steps = self._get_substeps_safely("channel_alignment")
        if not align_steps:
            raise RuntimeError(
                "No 'channel_alignment' substeps found in extended_config['pipeline_steps']['steps'].\n"
                "Check the extended config generation and step names (singular/plural)."
            )
        
        substep_id = f"align_channels_{target_round}"
        if substep_id not in align_steps:
            raise KeyError(f"Substep '{substep_id}' not found in alignment steps. Available: {list(align_steps.keys())}")
        
        substep_config = align_steps[substep_id]
        
        def execute_alignment():
            target_round_zarr = self.processed_rounds[target_round]

            # Check if we should use chunk alignment mode
            use_chunk_alignment = self.use_chunk_alignment
            initial_deformation_field_path = None

            # Get registration substep config to find deformation field path
            reg_steps = self._get_substeps_safely("registration_workflows")
            reg_substep_id = f"registration_{self.pipeline.reference_round}_to_{target_round}"
            reg_substep_config = reg_steps.get(reg_substep_id, {})
            
            # Get deformation field and affine matrix paths from config or fallback
            deformation_field_path = self._get_deformation_field_path(target_round, reg_substep_config)
            affine_matrix_path = self._get_affine_matrix_path(target_round, reg_substep_config)

            if use_chunk_alignment:
                # For chunk alignment, look for initial deformation field and final deformation field
                registration_dir = (
                    self.pipeline.working_directory
                    / "registration"
                    / f"{self.pipeline.reference_round}_to_{target_round}"
                )
                initial_deformation_field_path = str(registration_dir / f"{self.pipeline.reference_round}_to_{target_round}_initial_deformation_field.zarr")
                final_deformation_field_path = str(registration_dir / f"{self.pipeline.reference_round}_to_{target_round}_deformation_field.zarr")
                identity_matrix_path = str(registration_dir / f"{self.pipeline.reference_round}_to_{target_round}_chunk_identity_matrix.txt")
                
                # Check if chunk alignment files exist
                if not all([Path(initial_deformation_field_path).exists(), Path(final_deformation_field_path).exists(), Path(identity_matrix_path).exists()]):
                    print(f"    ⚠️ Chunk alignment not found for {target_round}. Running chunk alignment first...")
                    self.step_initial_chunk_alignment(target_round)
                    self.step_final_deformation_registration(target_round, use_chunk_alignment=True)
                
                deformation_field_path = Path(final_deformation_field_path)
                affine_matrix_path = Path(identity_matrix_path)
            else:
                # Traditional global alignment
                if not deformation_field_path.exists() or not affine_matrix_path.exists():
                    print(f"    ⚠️ Registration not found for {target_round}. Running registration first...")
                    registration_results = self.step_registration(target_round)
                    # Try to get paths from results, fallback to constructed paths
                    if isinstance(registration_results, dict):
                        if "deformation_field" in registration_results:
                            deformation_field_path = registration_results["deformation_field"]
                        if "affine_matrix" in registration_results:
                            affine_matrix_path = registration_results["affine_matrix"]
                    else:
                        deformation_field_path = self._get_deformation_field_path(target_round, reg_substep_config)
                        affine_matrix_path = self._get_affine_matrix_path(target_round, reg_substep_config)

            # Run the alignment
            return self.pipeline.apply_registration_to_all_channels(
                reference_round_data=self.reference_round_zarr,
                target_round_data=target_round_zarr,
                affine_matrix_path=str(affine_matrix_path),
                deformation_field_path=str(deformation_field_path),
                output_directory=str(
                    self.pipeline.working_directory / "aligned" / target_round
                ),
                use_chunk_alignment=use_chunk_alignment,
                initial_deformation_field_path=initial_deformation_field_path,
            )
        
        # Use common execution pattern
        return self._execute_substep_with_tracking(
            substep_id, substep_config, "channel_alignment", execute_alignment
        )

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
                if self.use_chunk_alignment:
                    print(f"Using chunk-based alignment for {round_name}")
                    self.step_initial_chunk_alignment(round_name)
                    self.step_final_deformation_registration(round_name, use_chunk_alignment=True)
                else:
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

    def _check_and_update_main_step_status(self, main_step_name: str):
        """
        Check if all substeps in a main step are completed and update main step status.
        Thread-safe implementation with proper locking.
        
        Args:
            main_step_name (str): Name of the main step (e.g., "data_preparation", "segmentation_workflow")
        """
        if not self.pipeline.enable_step_tracking:
            return
            
        with self._status_update_lock:
            try:
                # Get the main step configuration using safe lookup
                key = self._resolve_main_step_key(main_step_name)
                main_step_config = self.pipeline.extended_config["pipeline_steps"]["steps"][key]
                substeps = main_step_config.get("substeps", {})
                
                # Check if all substeps are completed
                all_completed = True
                any_failed = False
                for substep_id, substep_config in substeps.items():
                    status = substep_config.get("status")
                    if status == "failed":
                        any_failed = True
                        break
                    elif status != "completed":
                        all_completed = False
                
                # Update main step status based on substep states
                current_status = main_step_config.get("status")
                
                if any_failed and current_status != "failed":
                    print(f"❌ Some substeps in '{main_step_name}' failed - updating main step status to failed")
                    main_step_config["status"] = "failed"
                    self._add_status_log_entry(main_step_name, "failed", "Main step failed - some substeps failed")
                    
                elif all_completed and current_status != "completed":
                    print(f"🎉 All substeps in '{main_step_name}' completed - updating main step status")
                    main_step_config["status"] = "completed"
                    self._add_status_log_entry(main_step_name, "completed", "Main step completed - all substeps finished")
                    
                    # Update last completed step to main step level
                    self.pipeline.extended_config["step_execution"]["last_completed_step"] = main_step_name
                
                # Save the updated configuration if status changed
                if current_status != main_step_config.get("status"):
                    self.pipeline.step_manager.save_current_state()
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not update main step status for '{main_step_name}': {e}")

    def _add_status_log_entry(self, step_id: str, status: str, note: str):
        """Add a status change entry to the execution log."""
        # Ensure execution log structure exists
        self.pipeline.extended_config.setdefault("step_execution", {})
        self.pipeline.extended_config["step_execution"].setdefault("execution_log", [])
        
        log_entry = {
            "step_id": step_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": step_id,
            "note": note
        }
        self.pipeline.extended_config["step_execution"]["execution_log"].append(log_entry)
        
        # Persist logs immediately (optional but safer)
        try:
            if hasattr(self.pipeline, "step_manager"):
                self.pipeline.step_manager.save_current_state()
        except Exception as e:
            if self.debug:
                print(f"⚠️  Could not persist execution log: {e}")

    def _summarize_result(self, result, result_type="result"):
        """Summarize large results to avoid printing huge structures."""
        if isinstance(result, dict):
            if len(result) > 5:
                sample_keys = list(result.keys())[:3]
                return f"{result_type} dict with {len(result)} items (sample keys: {sample_keys}...)"
            else:
                return f"{result_type} dict with keys: {list(result.keys())}"
        elif isinstance(result, list):
            if len(result) > 10:
                return f"{result_type} list with {len(result)} items"
            else:
                return f"{result_type} list: {result}"
        elif isinstance(result, str) and len(result) > 100:
            return f"{result_type} string ({len(result)} chars): {result[:50]}..."
        else:
            return f"{result_type}: {result}"

    def _print_done(self, label, result):
        """Unified printing for completed operations."""
        print(f"✓ {label} completed!")
        print(f"✓ {label} results: {self._summarize_result(result, label)}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Selective Step Pipeline - Run specific VolAlign pipeline steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-steps                                    # List available steps
  %(prog)s --step process_rounds                           # Process all rounds
  %(prog)s --step segmentation                             # Run segmentation
  %(prog)s --step registration --round round2              # Register round2
  %(prog)s --step initial_global_alignment --round round2         # Run initial global alignment for round2
  %(prog)s --step initial_chunk_alignment --round round2          # Run chunk-based alignment for round2
  %(prog)s --step final_deformation_registration --round round2   # Run final deformation registration for round2
  %(prog)s --step final_deformation_registration --round round2 --use-chunk-alignment  # Use chunk alignment results for final step
  %(prog)s --step registration --round all                 # Register all non-reference rounds
  %(prog)s --step alignment --round round2                 # Align round2 channels
  %(prog)s --step alignment --round all                    # Align all non-reference rounds
  %(prog)s --step all                                      # Run complete pipeline
  %(prog)s --step all --use-chunk-alignment                # Run complete pipeline with chunk-based alignment
  %(prog)s --step progress                                 # Show progress
  %(prog)s --step resume                                   # Resume interrupted pipeline
  %(prog)s --reset-step data_preparation                   # Reset step to pending
        """,
    )

    parser.add_argument(
        "--step",
        choices=[
            "process_rounds",
            "segmentation",
            "registration",
            "initial_global_alignment",
            "initial_chunk_alignment",
            "final_deformation_registration",
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

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output"
    )

    parser.add_argument(
        "--reset-step", help="Reset a specific step to pending status"
    )

    parser.add_argument(
        "--use-chunk-alignment", action="store_true",
        help="Use chunk-based alignment instead of traditional global alignment"
    )

    args = parser.parse_args()

    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # Make --list-steps mutually exclusive with --step
    if args.list_steps and args.step:
        print("✗ Error: --list-steps and --step are mutually exclusive")
        return

    try:
        # Initialize pipeline with debug flag
        pipeline = SelectiveStepPipeline(config_file=args.config, debug=args.debug, use_chunk_alignment=args.use_chunk_alignment)

        # Handle reset step if requested
        if args.reset_step:
            pipeline.reset_step_to_pending(args.reset_step)
            return

        # List steps if requested
        if args.list_steps:
            pipeline.list_available_steps()
            return

        # Validate step argument
        if not args.step:
            print("✗ Error: --step argument is required (unless using --list-steps or --reset-step)")
            parser.print_help()
            return

        # Validate round argument for steps that need it
        if args.step in ["registration", "initial_global_alignment", "initial_chunk_alignment", "final_deformation_registration", "alignment"] and not args.round:
            print(f"✗ Error: --round argument is required for {args.step} step")
            return

        # Handle --round all convenience for registration and alignment
        if args.step in ["registration", "initial_global_alignment", "initial_chunk_alignment", "final_deformation_registration", "alignment"] and args.round == "all":
            print(f"Executing {args.step} for all non-reference rounds...")
            
            # Ensure rounds are processed first
            if not hasattr(pipeline, 'processed_rounds') or pipeline.processed_rounds is None:
                pipeline.step_process_rounds()
            
            # Process all non-reference rounds
            processed_any = False
            for round_name in pipeline.processed_rounds.keys():
                if round_name != pipeline.pipeline.reference_round:
                    print(f"\n--- {args.step.title()} for {round_name} ---")
                    if args.step == "registration":
                        pipeline.step_registration(round_name)
                    elif args.step == "initial_global_alignment":
                        pipeline.step_initial_global_alignment(round_name)
                    elif args.step == "initial_chunk_alignment":
                        pipeline.step_initial_chunk_alignment(round_name)
                    elif args.step == "final_deformation_registration":
                        pipeline.step_final_deformation_registration(round_name, use_chunk_alignment=args.use_chunk_alignment)
                    else:  # alignment
                        pipeline.step_alignment(round_name)
                    processed_any = True
            
            if not processed_any:
                print(f"⚠️  No non-reference rounds found to process for {args.step}")
            else:
                print(f"\n✓ {args.step.title()} completed for all non-reference rounds!")
            return

        # Execute the requested step
        print(f"Executing step: {args.step}")

        if args.step == "process_rounds":
            pipeline.step_process_rounds()

        elif args.step == "segmentation":
            pipeline.step_segmentation()

        elif args.step == "registration":
            pipeline.step_registration(args.round)

        elif args.step == "initial_global_alignment":
            pipeline.step_initial_global_alignment(args.round)

        elif args.step == "initial_chunk_alignment":
            pipeline.step_initial_chunk_alignment(args.round)

        elif args.step == "final_deformation_registration":
            pipeline.step_final_deformation_registration(args.round, use_chunk_alignment=args.use_chunk_alignment)

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
