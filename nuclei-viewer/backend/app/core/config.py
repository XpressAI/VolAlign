"""
Configuration management for the nuclei viewer application.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class EpitopeChannelConfig(BaseModel):
    """Configuration for an epitope channel."""

    name: str
    file_path: str  # Full path to zarr file
    array_key: Optional[str] = None
    default_color: str = "#ffffff"


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""

    file_path: str  # Full path to zarr file
    array_key: Optional[str] = None


class PipelineDataSourceConfig(BaseModel):
    """Configuration for pipeline-based data sources."""
    
    type: str = "pipeline"  # "pipeline" or "manual"
    pipeline_working_directory: str
    reference_round: str
    epitope_analysis_file: Optional[str] = "epitope_analysis/nucleus_centric_analysis_epitope_analysis.json"
    
    # Optional overrides for pipeline structure
    segmentation_file: Optional[str] = None  # Auto-detected if None
    zarr_volumes_dir: Optional[str] = "zarr_volumes"
    aligned_dir: Optional[str] = "aligned"
    segmentation_dir: Optional[str] = "segmentation"
    
    # Channel discovery settings
    auto_discover_channels: bool = True
    epitope_channels: List[str] = []  # Specific channels to load, empty = all


class DataConfig(BaseModel):
    """Data configuration section."""

    # Legacy manual configuration (backward compatibility)
    segmentation: Optional[DataSourceConfig] = None
    dapi_channel: Optional[DataSourceConfig] = None
    epitope_channels: List[EpitopeChannelConfig] = []
    
    # New pipeline configuration
    pipeline: Optional[PipelineDataSourceConfig] = None
    
    @validator('pipeline', 'segmentation')
    def validate_data_source(cls, v, values):
        """Ensure either pipeline or manual configuration is provided."""
        pipeline = values.get('pipeline') if 'pipeline' in values else v
        segmentation = values.get('segmentation') if 'segmentation' in values else None
        
        if pipeline is None and segmentation is None:
            raise ValueError("Either pipeline or manual data configuration must be provided")
        
        return v
    
    def is_pipeline_mode(self) -> bool:
        """Check if using pipeline-based configuration."""
        return self.pipeline is not None


class ProcessingConfig(BaseModel):
    """Processing parameters configuration."""

    min_object_size: int = 100
    pad_xy: int = 25
    max_objects_per_page: int = 10
    mip_chunk_size: int = 50
    cache_mips: bool = True
    auto_contrast: bool = True
    percentile_range: List[float] = [1.0, 99.9]

    @validator("percentile_range")
    def validate_percentile_range(cls, v):
        if len(v) != 2 or v[0] >= v[1] or v[0] < 0 or v[1] > 100:
            raise ValueError(
                "percentile_range must be [min, max] with 0 <= min < max <= 100"
            )
        return v


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True
    cors_origins: List[str] = ["http://localhost:3000"]


class EpitopeAnalysisUIConfig(BaseModel):
    """Epitope analysis UI configuration."""
    
    show_epitope_calls: bool = True
    show_confidence_scores: bool = True
    show_quality_scores: bool = True
    confidence_threshold: float = 0.7
    positive_call_color: str = "#4CAF50"
    negative_call_color: str = "#F44336"
    uncertain_call_color: str = "#FF9800"
    quality_score_thresholds: Dict[str, float] = {
        "high": 0.8,
        "medium": 0.5,
        "low": 0.0
    }


class UIConfig(BaseModel):
    """UI configuration."""

    default_opacity: float = 0.8
    color_palette: List[str] = [
        "#ff0000",
        "#00ff00",
        "#0000ff",
        "#ffff00",
        "#ff00ff",
        "#00ffff",
        "#ff8000",
        "#8000ff",
        "#80ff00",
        "#0080ff",
    ]
    epitope_analysis: Optional[EpitopeAnalysisUIConfig] = None

    @validator("default_opacity")
    def validate_opacity(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("default_opacity must be between 0.0 and 1.0")
        return v


class AppConfig(BaseModel):
    """Main application configuration."""

    data: DataConfig
    processing: ProcessingConfig
    server: ServerConfig
    ui: UIConfig


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    config_file: str = Field(
        default="config/default_config.yaml", env="NUCLEI_VIEWER_CONFIG"
    )

    class Config:
        env_prefix = "NUCLEI_VIEWER_"


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file.
    
    Supports both standalone nuclei-viewer config and VolAlign pipeline config
    with nuclei_viewer section.

    Args:
        config_path: Path to configuration file. If None, uses default from settings.

    Returns:
        AppConfig: Loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If config validation fails
    """
    if config_path is None:
        settings = Settings()
        config_path = settings.config_file

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")

    # Handle VolAlign pipeline config structure
    if "nuclei_viewer" in config_data:
        # Extract nuclei_viewer section and inherit from main config
        nuclei_config = config_data["nuclei_viewer"].copy()
        
        # Inherit working directory and reference round if not specified
        if "data" in nuclei_config and "pipeline" in nuclei_config["data"]:
            pipeline_config = nuclei_config["data"]["pipeline"]
            
            # Inherit working_directory if not specified
            if not pipeline_config.get("pipeline_working_directory"):
                pipeline_config["pipeline_working_directory"] = config_data.get("working_directory", "")
            
            # Inherit reference_round if not specified
            if not pipeline_config.get("reference_round"):
                if "data" in config_data and "reference_round" in config_data["data"]:
                    pipeline_config["reference_round"] = config_data["data"]["reference_round"]
        
        # Use the nuclei_viewer section as the main config
        config_data = nuclei_config

    try:
        return AppConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def save_config(config: AppConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        config_path: Path to save configuration file
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False, indent=2)


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the global configuration instance.

    Returns:
        AppConfig: Current configuration

    Raises:
        RuntimeError: If configuration hasn't been initialized
    """
    global _config
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call init_config() first.")
    return _config


def init_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Initialize the global configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        AppConfig: Loaded configuration
    """
    global _config
    _config = load_config(config_path)
    return _config


def update_config(**kwargs) -> AppConfig:
    """
    Update configuration with new values.

    Args:
        **kwargs: Configuration updates

    Returns:
        AppConfig: Updated configuration
    """
    global _config
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call init_config() first.")

    # Create updated config dict
    config_dict = _config.dict()

    # Apply updates (supports nested updates)
    def update_nested_dict(d: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and key in d and isinstance(d[key], dict):
                update_nested_dict(d[key], value)
            else:
                d[key] = value

    update_nested_dict(config_dict, kwargs)

    # Validate and update global config
    _config = AppConfig(**config_dict)
    return _config
