import logging
from typing import Optional
from .config import AppConfig
from .nuclei_processor import NucleiProcessor
from .data_loader import AbstractDataLoader

_shared_state = {}

def get_logger(name: str) -> logging.Logger:
    """Get a standardized logger with consistent formatting across modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_shared_data_loader(loader: AbstractDataLoader):
    _shared_state["data_loader"] = loader


def get_shared_data_loader() -> Optional[AbstractDataLoader]:
    return _shared_state.get("data_loader")


def set_shared_nuclei_processor(processor: NucleiProcessor):
    _shared_state["nuclei_processor"] = processor


def get_shared_nuclei_processor() -> Optional[NucleiProcessor]:
    return _shared_state.get("nuclei_processor")


def reset_shared_state():
    """Reset all shared state (data loader, nuclei processor, etc.)."""
    _shared_state.clear()
