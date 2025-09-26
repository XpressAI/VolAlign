#!/usr/bin/env python3
"""
Production Nucleus-Centric Epitope Analysis for VolAlign Pipeline

This script provides a production-ready implementation of nucleus-centric epitope analysis
with comprehensive error handling, logging, and cluster integration capabilities.

Key Features:
- Nucleus-centric analysis (each nucleus against all rounds/channels)
- Per-round-per-channel cutoff determination
- Robust error handling and recovery
- Comprehensive logging and progress tracking
- Memory-efficient processing with configurable limits
- Integration with cluster computing environments
- Detailed validation and quality control
- Multiple output formats and comprehensive reporting

Usage:
    python production_epitope_analysis.py config.yaml [options]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# Add VolAlign to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from VolAlign.epitope_analysis import (
    EpitopeAnalyzer,
    EpitopeChannel,
    EpitopeStatisticalAnalyzer,
    ShellParameters,
    CutoffDetermination,
    NucleusEpitopeData,
    EpitopeAnalysisResult,
)


def setup_logging(
    log_file: Optional[str] = None, log_level: str = "INFO"
) -> logging.Logger:
    """Set up comprehensive logging for production use."""

    # Create logger
    logger = logging.getLogger("epitope_analysis")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_and_validate_config(config_path: str, logger: logging.Logger) -> Dict:
    """Load and validate configuration with comprehensive error checking."""

    logger.info(f"Loading configuration from: {config_path}")

    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

    # Validate required sections
    required_sections = ["working_directory", "data", "epitope_analysis"]
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate epitope analysis configuration
    epitope_config = config["epitope_analysis"]
    if not epitope_config.get("enabled", False):
        logger.error("Epitope analysis is not enabled in configuration")
        raise ValueError("Epitope analysis is not enabled in configuration")

    epitope_channels = epitope_config.get("epitope_channels", {}).get("channels", [])
    if not epitope_channels:
        logger.error("No epitope channels specified in configuration")
        raise ValueError("No epitope channels specified in configuration")

    # Validate working directory
    working_dir = Path(config["working_directory"])
    if not working_dir.exists():
        logger.error(f"Working directory not found: {working_dir}")
        raise FileNotFoundError(f"Working directory not found: {working_dir}")

    logger.info("Configuration validation passed")
    logger.info(f"  - Working directory: {working_dir}")
    logger.info(f"  - Epitope channels: {epitope_channels}")
    logger.info(f"  - Reference round: {config['data']['reference_round']}")

    return config


def run_nucleus_centric_analysis(
    config: Dict,
    logger: logging.Logger,
    max_nuclei: Optional[int] = None,
    save_results: bool = True,
) -> Dict:
    """Run comprehensive nucleus-centric epitope analysis."""

    logger.info("Starting nucleus-centric epitope analysis")
    start_time = time.time()

    try:
        # Initialize analyzer
        logger.info("Initializing EpitopeAnalyzer")
        analyzer = EpitopeAnalyzer(config=config)

        # Discover epitope channels
        logger.info("Discovering epitope channels across all rounds")
        epitope_channels = analyzer.discover_epitope_channels()

        if not epitope_channels:
            logger.error("No epitope channels found")
            raise ValueError("No epitope channels found")

        total_channels = sum(len(channels) for channels in epitope_channels.values())
        logger.info(
            f"Found {total_channels} epitope channels across {len(epitope_channels)} rounds"
        )

        for round_name, channels in epitope_channels.items():
            channel_names = [ch.name for ch in channels]
            logger.info(f"  - {round_name}: {channel_names}")

        # Run nucleus-centric analysis
        logger.info(
            "Running nucleus-centric analysis (each nucleus against all rounds/channels)"
        )
        logger.info(f"Configuration parameters:")
        logger.info(f"  - Max nuclei: {max_nuclei or analyzer.max_nuclei_per_round}")
        logger.info(f"  - Analysis region: {analyzer.default_region}")
        logger.info(f"  - Shell parameters: {analyzer.shell_config}")

        # This is the NEW nucleus-centric method
        results = analyzer.run_complete_nucleus_centric_analysis(
            max_nuclei=max_nuclei, save_results=save_results
        )

        # Log results summary
        metadata = results["metadata"]
        logger.info("Nucleus-centric analysis completed successfully")
        logger.info(f"Results summary:")
        logger.info(f"  - Nuclei analyzed: {metadata['n_nuclei']}")
        logger.info(f"  - Rounds processed: {metadata['n_rounds']}")
        logger.info(f"  - Analysis region: {metadata['analysis_region']}")
        logger.info(f"  - Cutoff method: {metadata['cutoff_method']}")

        # Log cutoff statistics
        cutoffs = results["cutoffs"]
        logger.info(f"Per-round-per-channel cutoffs calculated: {len(cutoffs)}")

        cutoff_values = [
            cutoff for cutoff in cutoffs.values() if isinstance(cutoff, (int, float))
        ]
        if cutoff_values:
            logger.info(
                f"  - Cutoff range: {min(cutoff_values):.2f} - {max(cutoff_values):.2f}"
            )
            logger.info(f"  - Mean cutoff: {np.mean(cutoff_values):.2f}")

        # Log analysis results statistics
        analysis_results = results["analysis_results"]
        if analysis_results:
            quality_scores = [
                result.quality_score for result in analysis_results.values()
            ]
            logger.info(f"Quality score statistics:")
            logger.info(f"  - Mean quality score: {np.mean(quality_scores):.3f}")
            logger.info(
                f"  - Quality score range: {min(quality_scores):.3f} - {max(quality_scores):.3f}"
            )

            # Count positive calls per round-channel
            positive_counts = {}
            for result in analysis_results.values():
                for round_channel, is_positive in result.epitope_calls.items():
                    if round_channel not in positive_counts:
                        positive_counts[round_channel] = 0
                    if is_positive:
                        positive_counts[round_channel] += 1

            logger.info(f"Positive call statistics (top 5 round-channels):")
            sorted_counts = sorted(
                positive_counts.items(), key=lambda x: x[1], reverse=True
            )
            for round_channel, count in sorted_counts[:5]:
                percentage = (count / len(analysis_results)) * 100
                logger.info(
                    f"  - {round_channel}: {count}/{len(analysis_results)} ({percentage:.1f}%)"
                )

        # Log saved files
        if save_results and "saved_files" in results:
            logger.info("Results saved to:")
            for file_type, file_path in results["saved_files"].items():
                logger.info(f"  - {file_type}: {file_path}")

        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")

        return results

    except Exception as e:
        logger.error(f"Error in nucleus-centric analysis: {e}")
        logger.exception("Full traceback:")
        raise


def validate_results(results: Dict, logger: logging.Logger) -> bool:
    """Validate analysis results for quality control."""

    logger.info("Validating analysis results")

    try:
        # Check required sections
        required_sections = [
            "metadata",
            "intensity_data",
            "cutoffs",
            "analysis_results",
        ]
        for section in required_sections:
            if section not in results:
                logger.error(f"Missing required results section: {section}")
                return False

        metadata = results["metadata"]
        intensity_data = results["intensity_data"]
        cutoffs = results["cutoffs"]
        analysis_results = results["analysis_results"]

        # Validate metadata
        if metadata["n_nuclei"] != len(intensity_data):
            logger.error(
                f"Metadata nuclei count mismatch: {metadata['n_nuclei']} vs {len(intensity_data)}"
            )
            return False

        if metadata["n_nuclei"] != len(analysis_results):
            logger.error(
                f"Analysis results count mismatch: {metadata['n_nuclei']} vs {len(analysis_results)}"
            )
            return False

        # Validate intensity data structure
        for nucleus_label, nucleus_data in intensity_data.items():
            if not hasattr(nucleus_data, "rounds_processed"):
                logger.error(f"Nucleus {nucleus_label} missing rounds_processed")
                return False

            if not hasattr(nucleus_data, "nuclei_intensities"):
                logger.error(f"Nucleus {nucleus_label} missing nuclei_intensities")
                return False

            # Check that all intensities use round_channel format
            for round_channel_key in nucleus_data.nuclei_intensities.keys():
                if "_" not in round_channel_key:
                    logger.warning(
                        f"Intensity key {round_channel_key} doesn't follow round_channel format"
                    )

        # Validate cutoffs
        if len(cutoffs) == 0:
            logger.error("No cutoffs calculated")
            return False

        for round_channel_key, cutoff_value in cutoffs.items():
            if not isinstance(cutoff_value, (int, float)):
                logger.error(
                    f"Invalid cutoff value type for {round_channel_key}: {type(cutoff_value)}"
                )
                return False

            if cutoff_value < 0:
                logger.warning(
                    f"Negative cutoff for {round_channel_key}: {cutoff_value}"
                )

        # Validate analysis results
        for nucleus_label, analysis_result in analysis_results.items():
            if not hasattr(analysis_result, "epitope_calls"):
                logger.error(
                    f"Analysis result for nucleus {nucleus_label} missing epitope_calls"
                )
                return False

            if not hasattr(analysis_result, "rounds_analyzed"):
                logger.error(
                    f"Analysis result for nucleus {nucleus_label} missing rounds_analyzed"
                )
                return False

            # Check consistency between epitope calls and cutoffs
            for round_channel_key in analysis_result.epitope_calls.keys():
                if round_channel_key not in cutoffs:
                    logger.warning(
                        f"Epitope call for {round_channel_key} but no cutoff available"
                    )

        logger.info("Results validation passed")
        return True

    except Exception as e:
        logger.error(f"Error during results validation: {e}")
        logger.exception("Full traceback:")
        return False


def generate_summary_report(results: Dict, output_path: str, logger: logging.Logger):
    """Generate comprehensive summary report."""

    logger.info(f"Generating summary report: {output_path}")

    try:
        metadata = results["metadata"]
        intensity_data = results["intensity_data"]
        cutoffs = results["cutoffs"]
        analysis_results = results["analysis_results"]

        # Compile summary statistics
        summary = {
            "analysis_summary": {
                "analysis_type": metadata["analysis_type"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_nuclei_analyzed": metadata["n_nuclei"],
                "n_rounds_processed": metadata["n_rounds"],
                "analysis_region": metadata["analysis_region"],
                "cutoff_method": metadata["cutoff_method"],
            },
            "cutoff_statistics": {},
            "epitope_call_statistics": {},
            "quality_metrics": {},
            "configuration_used": metadata.get("shell_parameters", {}),
        }

        # Cutoff statistics
        cutoff_values = []
        cutoff_methods = {"otsu": 0}  # Default method used in epitope analysis
        for round_channel_key, cutoff_value in cutoffs.items():
            if isinstance(cutoff_value, (int, float)):
                cutoff_values.append(cutoff_value)
                # Since the epitope analysis uses a single method, we'll track it as 'otsu' (default)
                cutoff_methods["otsu"] += 1

        if cutoff_values:
            summary["cutoff_statistics"] = {
                "n_cutoffs_calculated": len(cutoff_values),
                "cutoff_range": [float(min(cutoff_values)), float(max(cutoff_values))],
                "mean_cutoff": float(np.mean(cutoff_values)),
                "median_cutoff": float(np.median(cutoff_values)),
                "std_cutoff": float(np.std(cutoff_values)),
                "methods_used": cutoff_methods,
            }

        # Epitope call statistics
        positive_counts = {}
        total_calls = 0
        quality_scores = []

        for nucleus_label, analysis_result in analysis_results.items():
            quality_scores.append(analysis_result.quality_score)

            for round_channel_key, is_positive in analysis_result.epitope_calls.items():
                if round_channel_key not in positive_counts:
                    positive_counts[round_channel_key] = {"positive": 0, "total": 0}

                positive_counts[round_channel_key]["total"] += 1
                if is_positive:
                    positive_counts[round_channel_key]["positive"] += 1
                total_calls += 1

        # Calculate positive rates
        positive_rates = {}
        for round_channel_key, counts in positive_counts.items():
            positive_rates[round_channel_key] = {
                "positive_count": counts["positive"],
                "total_count": counts["total"],
                "positive_rate": (
                    counts["positive"] / counts["total"] if counts["total"] > 0 else 0.0
                ),
            }

        summary["epitope_call_statistics"] = {
            "total_calls_made": total_calls,
            "n_round_channel_combinations": len(positive_counts),
            "positive_rates_by_round_channel": positive_rates,
        }

        # Quality metrics
        if quality_scores:
            summary["quality_metrics"] = {
                "mean_quality_score": float(np.mean(quality_scores)),
                "median_quality_score": float(np.median(quality_scores)),
                "std_quality_score": float(np.std(quality_scores)),
                "quality_score_range": [
                    float(min(quality_scores)),
                    float(max(quality_scores)),
                ],
            }

        # Save summary report
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Summary report generated successfully")
        logger.info(
            f"  - Total nuclei: {summary['analysis_summary']['n_nuclei_analyzed']}"
        )
        logger.info(
            f"  - Total calls: {summary['epitope_call_statistics']['total_calls_made']}"
        )
        logger.info(
            f"  - Round-channel combinations: {summary['epitope_call_statistics']['n_round_channel_combinations']}"
        )
        logger.info(
            f"  - Mean quality score: {summary['quality_metrics'].get('mean_quality_score', 'N/A')}"
        )

    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        logger.exception("Full traceback:")


def main():
    """Main production analysis function."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Production Nucleus-Centric Epitope Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--max-nuclei",
        type=int,
        help="Maximum number of nuclei to analyze (overrides config)",
    )

    parser.add_argument(
        "--log-file", help="Path to log file (default: epitope_analysis.log)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to files"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and exit",
    )

    args = parser.parse_args()

    # Set up logging
    log_file = args.log_file or "epitope_analysis.log"
    logger = setup_logging(log_file, args.log_level)

    logger.info("=" * 70)
    logger.info("Production Nucleus-Centric Epitope Analysis Starting")
    logger.info("=" * 70)

    try:
        # Load and validate configuration
        config = load_and_validate_config(args.config, logger)

        if args.validate_only:
            logger.info("Configuration validation completed successfully")
            return 0

        # Run nucleus-centric analysis
        results = run_nucleus_centric_analysis(
            config=config,
            logger=logger,
            max_nuclei=args.max_nuclei,
            save_results=not args.no_save,
        )

        # Validate results
        if not validate_results(results, logger):
            logger.error("Results validation failed")
            return 1

        # Generate summary report
        if not args.no_save:
            output_dir = Path(config["working_directory"]) / "epitope_analysis"
            summary_path = output_dir / "nucleus_centric_analysis_summary.json"
            generate_summary_report(results, str(summary_path), logger)

        logger.info("=" * 70)
        logger.info(
            "Production Nucleus-Centric Epitope Analysis Completed Successfully"
        )
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Production analysis failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
