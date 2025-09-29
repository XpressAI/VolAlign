"""
Factory for creating the appropriate DataLoader (pipeline or manual).
"""

from .config import AppConfig
from .data_loader import DataLoader
from .pipeline_data_loader import PipelineDataLoader


def create_data_loader(config: AppConfig):
    """
    Factory function to create the appropriate data loader
    based on pipeline vs. manual mode.

    Args:
        config (AppConfig): Application configuration

    Returns:
        DataLoader | PipelineDataLoader
    """
    if config.data.is_pipeline_mode():
        return PipelineDataLoader(config)
    return DataLoader(config)