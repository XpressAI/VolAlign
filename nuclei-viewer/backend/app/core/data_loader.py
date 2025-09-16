"""
Data loading utilities for Zarr files and nuclei processing.
"""

import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import zarr
from skimage.measure import label, regionprops

from .config import AppConfig, EpitopeChannelConfig

logger = logging.getLogger(__name__)


class DatasetInfo:
    """Information about a loaded dataset."""

    def __init__(
        self,
        name: str,
        path: Path,
        shape: Tuple[int, ...],
        dtype: str,
        array_key: Optional[str] = None,
    ):
        self.name = name
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.array_key = array_key
        self._dask_array: Optional[da.Array] = None

    def get_dask_array(self) -> da.Array:
        """Get the dask array for this dataset, loading if necessary."""
        if self._dask_array is None:
            self._dask_array = load_zarr_as_dask(self.path, self.array_key)
        return self._dask_array

    def __repr__(self) -> str:
        return (
            f"DatasetInfo(name='{self.name}', shape={self.shape}, dtype='{self.dtype}')"
        )


class NucleusInfo:
    """Information about a segmented nucleus."""

    def __init__(
        self, label: int, bbox: Tuple[int, ...], area: int, centroid: Tuple[float, ...]
    ):
        self.label = label
        self.bbox = bbox  # (min_z, min_y, min_x, max_z, max_y, max_x)
        self.area = area
        self.centroid = centroid  # (z, y, x)

    def get_padded_bbox(
        self, pad_xy: int, volume_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Get bounding box with XY padding, clipped to volume bounds."""
        min_z, min_y, min_x, max_z, max_y, max_x = self.bbox

        # Apply padding in XY dimensions only
        min_y_pad = max(0, min_y - pad_xy)
        max_y_pad = min(volume_shape[1], max_y + pad_xy)
        min_x_pad = max(0, min_x - pad_xy)
        max_x_pad = min(volume_shape[2], max_x + pad_xy)

        return (min_z, min_y_pad, min_x_pad, max_z, max_y_pad, max_x_pad)

    def __repr__(self) -> str:
        return f"NucleusInfo(label={self.label}, area={self.area}, bbox={self.bbox})"


class DataLoader:
    """Main data loader for nuclei visualization."""

    def __init__(self, config: AppConfig):
        self.config = config

        # Loaded datasets
        self.segmentation: Optional[DatasetInfo] = None
        self.dapi_channel: Optional[DatasetInfo] = None
        self.epitope_channels: Dict[str, DatasetInfo] = {}

        # Nuclei information
        self.nuclei: List[NucleusInfo] = []
        self._nuclei_loaded = False

    def discover_datasets(self) -> Dict[str, List[Path]]:
        """
        Discover available datasets based on file paths.

        Returns:
            Dict mapping dataset type to list of found files
        """
        discovered = {}

        # Find segmentation files
        seg_files = self._resolve_dataset_files(self.config.data.segmentation)
        discovered["segmentation"] = seg_files

        # Find DAPI channel files
        dapi_files = self._resolve_dataset_files(self.config.data.dapi_channel)
        discovered["dapi"] = dapi_files

        # Find epitope channel files
        for epitope_config in self.config.data.epitope_channels:
            epitope_files = self._resolve_dataset_files(epitope_config)
            discovered[epitope_config.name] = epitope_files

        logger.info(
            f"Discovered datasets: {[(k, len(v)) for k, v in discovered.items()]}"
        )
        return discovered

    def _resolve_dataset_files(self, config) -> List[Path]:
        """
        Resolve dataset files from file_path.

        Args:
            config: DataSourceConfig or EpitopeChannelConfig

        Returns:
            List of resolved file paths
        """
        files = []

        file_path = Path(config.file_path)
        if file_path.exists():
            files.append(file_path)
        else:
            logger.warning(f"File path does not exist: {file_path}")

        return files

    def load_segmentation(
        self, file_path: Optional[Path] = None, array_key: Optional[str] = None
    ) -> DatasetInfo:
        """Load segmentation dataset."""
        if file_path is None:
            # Auto-discover
            discovered = self.discover_datasets()
            seg_files = discovered.get("segmentation", [])
            if not seg_files:
                raise FileNotFoundError("No segmentation files found")
            file_path = seg_files[0]  # Use first found file

        if array_key is None:
            array_key = self.config.data.segmentation.array_key

        dask_array = load_zarr_as_dask(file_path, array_key)

        self.segmentation = DatasetInfo(
            name="segmentation",
            path=file_path,
            shape=dask_array.shape,
            dtype=str(dask_array.dtype),
            array_key=array_key,
        )

        logger.info(f"Loaded segmentation: {self.segmentation}")
        return self.segmentation

    def load_dapi_channel(
        self, file_path: Optional[Path] = None, array_key: Optional[str] = None
    ) -> DatasetInfo:
        """Load DAPI channel dataset."""
        if file_path is None:
            # Auto-discover
            discovered = self.discover_datasets()
            dapi_files = discovered.get("dapi", [])
            if not dapi_files:
                raise FileNotFoundError("No DAPI channel files found")
            file_path = dapi_files[0]  # Use first found file

        if array_key is None:
            array_key = self.config.data.dapi_channel.array_key

        dask_array = load_zarr_as_dask(file_path, array_key)

        self.dapi_channel = DatasetInfo(
            name="dapi",
            path=file_path,
            shape=dask_array.shape,
            dtype=str(dask_array.dtype),
            array_key=array_key,
        )

        logger.info(f"Loaded DAPI channel: {self.dapi_channel}")
        return self.dapi_channel

    def load_epitope_channel(
        self, name: str, file_path: Path, array_key: Optional[str] = None
    ) -> DatasetInfo:
        """Load a single epitope channel dataset."""
        dask_array = load_zarr_as_dask(file_path, array_key)

        dataset_info = DatasetInfo(
            name=name,
            path=file_path,
            shape=dask_array.shape,
            dtype=str(dask_array.dtype),
            array_key=array_key,
        )

        self.epitope_channels[name] = dataset_info
        logger.info(f"Loaded epitope channel: {dataset_info}")
        return dataset_info

    def load_epitope_channels(
        self, file_paths: Optional[Dict[str, Path]] = None
    ) -> Dict[str, DatasetInfo]:
        """Load epitope channel datasets."""
        if file_paths is None:
            # Auto-discover
            discovered = self.discover_datasets()
            file_paths = {}
            for epitope_config in self.config.data.epitope_channels:
                epitope_files = discovered.get(epitope_config.name, [])
                if epitope_files:
                    file_paths[epitope_config.name] = epitope_files[0]

        self.epitope_channels = {}

        for epitope_config in self.config.data.epitope_channels:
            if epitope_config.name not in file_paths:
                logger.warning(
                    f"No file found for epitope channel: {epitope_config.name}"
                )
                continue

            file_path = file_paths[epitope_config.name]
            array_key = epitope_config.array_key

            try:
                dask_array = load_zarr_as_dask(file_path, array_key)

                dataset_info = DatasetInfo(
                    name=epitope_config.name,
                    path=file_path,
                    shape=dask_array.shape,
                    dtype=str(dask_array.dtype),
                    array_key=array_key,
                )

                self.epitope_channels[epitope_config.name] = dataset_info
                logger.info(f"Loaded epitope channel: {dataset_info}")

            except Exception as e:
                logger.error(
                    f"Failed to load epitope channel {epitope_config.name}: {e}"
                )

        return self.epitope_channels

    def load_all_datasets(self) -> None:
        """Load all datasets (segmentation, DAPI, and epitope channels)."""
        self.load_segmentation()
        self.load_dapi_channel()
        self.load_epitope_channels()

    def extract_nuclei_info(self, force_reload: bool = False) -> List[NucleusInfo]:
        """
        Extract nuclei information from segmentation mask.

        Args:
            force_reload: Force re-extraction even if already loaded

        Returns:
            List of NucleusInfo objects
        """
        if self._nuclei_loaded and not force_reload:
            return self.nuclei

        if self.segmentation is None:
            raise RuntimeError("Segmentation dataset not loaded")

        logger.info("Extracting nuclei information from segmentation mask...")

        # Load segmentation mask
        seg_array = self.segmentation.get_dask_array()
        seg_np = seg_array.compute()  # Load into memory for regionprops

        # Ensure we have a labeled mask
        if seg_np.max() <= 1:
            logger.info("Converting binary mask to labeled mask...")
            labeled_mask = label(seg_np)
        else:
            labeled_mask = seg_np

        # Extract region properties
        regions = regionprops(labeled_mask)

        self.nuclei = []
        min_size = self.config.processing.min_object_size

        for region in regions:
            if region.area >= min_size:
                nucleus_info = NucleusInfo(
                    label=region.label,
                    bbox=region.bbox,
                    area=region.area,
                    centroid=region.centroid,
                )
                self.nuclei.append(nucleus_info)

        self.nuclei.sort(key=lambda x: x.label)  # Sort by label for consistency
        self._nuclei_loaded = True

        logger.info(f"Extracted {len(self.nuclei)} nuclei (min_size={min_size})")
        return self.nuclei

    def get_nucleus_by_label(self, label: int) -> Optional[NucleusInfo]:
        """Get nucleus information by label."""
        if not self._nuclei_loaded:
            self.extract_nuclei_info()

        for nucleus in self.nuclei:
            if nucleus.label == label:
                return nucleus
        return None

    def get_nuclei_page(
        self, page: int, page_size: Optional[int] = None
    ) -> List[NucleusInfo]:
        """
        Get a page of nuclei for pagination.

        Args:
            page: Page number (0-based)
            page_size: Number of nuclei per page (uses config default if None)

        Returns:
            List of NucleusInfo objects for the requested page
        """
        if not self._nuclei_loaded:
            self.extract_nuclei_info()

        if page_size is None:
            page_size = self.config.processing.max_objects_per_page

        start_idx = page * page_size
        end_idx = start_idx + page_size

        return self.nuclei[start_idx:end_idx]

    def get_total_pages(self, page_size: Optional[int] = None) -> int:
        """Get total number of pages for pagination."""
        if not self._nuclei_loaded:
            self.extract_nuclei_info()

        if page_size is None:
            page_size = self.config.processing.max_objects_per_page

        return (len(self.nuclei) + page_size - 1) // page_size


def load_zarr_as_dask(
    zarr_path: Union[str, Path], array_key: Optional[str] = None
) -> da.Array:
    """
    Load a zarr array as a dask array for memory-efficient processing.

    Args:
        zarr_path: Path to zarr store
        array_key: Key/name of the array within the zarr store (None for direct zarr arrays)

    Returns:
        Dask array

    Raises:
        FileNotFoundError: If zarr file doesn't exist
        KeyError: If array_key not found in zarr store
    """
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")

    try:
        zarr_store = zarr.open(zarr_path, mode="r")

        # Check if this is a direct zarr array (not a group with keys)
        if hasattr(zarr_store, "shape"):
            # This is a direct zarr array
            if array_key is not None:
                logger.warning(
                    f"array_key '{array_key}' specified but {zarr_path} is a direct zarr array"
                )
            dask_array = da.from_zarr(zarr_store)
        else:
            # This is a zarr group with keys
            if array_key is None:
                available_keys = list(zarr_store.keys())
                raise KeyError(
                    f"array_key required for zarr group. Available keys: {available_keys}"
                )

            try:
                zarr_array = zarr_store[array_key]
                dask_array = da.from_zarr(zarr_array)
            except KeyError:
                available_keys = list(zarr_store.keys())
                raise KeyError(
                    f"Array '{array_key}' not found in zarr store. Available keys: {available_keys}"
                )

        logger.debug(
            f"Loaded zarr array: shape={dask_array.shape}, dtype={dask_array.dtype}, chunks={dask_array.chunks}"
        )
        return dask_array

    except Exception as e:
        logger.error(f"Failed to load zarr array from {zarr_path}: {e}")
        raise


def validate_shape_compatibility(datasets: List[DatasetInfo]) -> Tuple[int, ...]:
    """
    Validate that datasets have compatible shapes and return the common shape.

    Args:
        datasets: List of datasets to validate

    Returns:
        Common shape tuple

    Raises:
        ValueError: If shapes are incompatible
    """
    if not datasets:
        raise ValueError("No datasets provided")

    shapes = [ds.shape for ds in datasets]

    # Find minimum dimensions for cropping
    min_shape = tuple(min(dims) for dims in zip(*shapes))

    # Check if any cropping is needed
    needs_cropping = any(shape != min_shape for shape in shapes)

    if needs_cropping:
        logger.warning(
            f"Shape mismatch detected. Will crop to common shape: {min_shape}"
        )
        for i, (dataset, shape) in enumerate(zip(datasets, shapes)):
            if shape != min_shape:
                logger.warning(f"  {dataset.name}: {shape} -> {min_shape}")

    return min_shape
