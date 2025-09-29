import logging
from pathlib import Path
from typing import Optional, Dict, Any


class ValidationError(Exception):
    """Custom validation error compatible with previous imports."""
    pass


class PipelineValidationResult:
    def __init__(self, is_valid: bool, issues: list[str] = None, warnings: list[str] = None, pipeline_info: Dict[str, Any] = None):
        self.is_valid = is_valid
        self.issues = issues or []
        self.warnings = warnings or []
        self.pipeline_info = pipeline_info or {}

logger = logging.getLogger(__name__)


def validate_file_exists(path_str: Optional[str], name: str) -> bool:
    """Validate that a given file path exists."""
    if not path_str:
        logger.warning(f"{name} path is not set")
        return False

    path_obj = Path(path_str)
    if not path_obj.exists():
        logger.warning(f"{name} file does not exist: {path_obj}")
        return False

    logger.info(f"{name} file is valid: {path_obj}")
    return True


def validate_pipeline_config(config) -> PipelineValidationResult:
    """
    Dummy placeholder validation to ensure compatibility with imports.
    A fuller validator could implement actual file/directory checks.
    """
    try:
        pipeline_info = {
            "working_directory": getattr(config.data.pipeline, "pipeline_working_directory", None),
            "reference_round": getattr(config.data.pipeline, "reference_round", None),
            "epitope_analysis_file": getattr(config.data.pipeline, "epitope_analysis_file", None),
        }
        return PipelineValidationResult(True, [], [], pipeline_info)
    except Exception as e:
        return PipelineValidationResult(False, [str(e)], [], {})


def validate_directory_exists(path_str: Optional[str], name: str) -> bool:
    """Validate that a given directory path exists."""
    if not path_str:
        logger.warning(f"{name} directory path is not set")
        return False

    path_obj = Path(path_str)
    if not path_obj.exists() or not path_obj.is_dir():
        logger.warning(f"{name} directory does not exist: {path_obj}")
        return False

    logger.info(f"{name} directory is valid: {path_obj}")
    return True