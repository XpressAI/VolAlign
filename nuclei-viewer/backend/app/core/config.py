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


class DataConfig(BaseModel):
    """Data configuration section."""

    segmentation: DataSourceConfig
    dapi_channel: DataSourceConfig
    epitope_channels: List[EpitopeChannelConfig] = []


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
