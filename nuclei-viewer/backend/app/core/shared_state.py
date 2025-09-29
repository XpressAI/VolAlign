"""
Shared application state management.
"""

from typing import Optional, Union

from .config import get_config
from .data_loader import DataLoader
from .pipeline_data_loader import PipelineDataLoader
from .nuclei_processor import NucleiProcessor

# Global shared instances
_data_loader: Optional[Union[DataLoader, PipelineDataLoader]] = None
_nuclei_processor: Optional[NucleiProcessor] = None


def get_shared_data_loader() -> Union[DataLoader, PipelineDataLoader]:
    """
    Get the shared data loader instance.
    
    Returns:
        DataLoader or PipelineDataLoader: Shared data loader instance based on configuration
    """
    global _data_loader
    if _data_loader is None:
        config = get_config()
        
        # Choose data loader based on configuration
        if config.data.is_pipeline_mode():
            _data_loader = PipelineDataLoader(config)
        else:
            _data_loader = DataLoader(config)
    
    return _data_loader


def get_shared_nuclei_processor() -> NucleiProcessor:
    """Get the shared nuclei processor instance."""
    global _nuclei_processor, _data_loader
    if _nuclei_processor is None:
        config = get_config()
        data_loader = get_shared_data_loader()
        _nuclei_processor = NucleiProcessor(config, data_loader)
    return _nuclei_processor


def reset_shared_state():
    """Reset all shared state."""
    global _data_loader, _nuclei_processor
    _data_loader = None
    _nuclei_processor = None


def is_pipeline_mode() -> bool:
    """
    Check if the application is running in pipeline mode.
    
    Returns:
        bool: True if using pipeline data loader
    """
    try:
        config = get_config()
        return config.data.is_pipeline_mode()
    except:
        return False


def get_pipeline_data_loader() -> Optional[PipelineDataLoader]:
    """
    Get the pipeline data loader if in pipeline mode.
    
    Returns:
        PipelineDataLoader or None: Pipeline data loader if available
    """
    data_loader = get_shared_data_loader()
    if isinstance(data_loader, PipelineDataLoader):
        return data_loader
    return None
