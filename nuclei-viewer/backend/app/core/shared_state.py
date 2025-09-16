"""
Shared application state management.
"""

from typing import Optional
from .config import get_config
from .data_loader import DataLoader
from .nuclei_processor import NucleiProcessor

# Global shared instances
_data_loader: Optional[DataLoader] = None
_nuclei_processor: Optional[NucleiProcessor] = None


def get_shared_data_loader() -> DataLoader:
    """Get the shared data loader instance."""
    global _data_loader
    if _data_loader is None:
        config = get_config()
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