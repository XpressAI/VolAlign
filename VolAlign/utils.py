from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import tifffile
import zarr
from exm.stitching.tileset import Tileset
from numcodecs import Blosc
from scipy.ndimage import zoom
from tqdm import tqdm

# =============================================================================
# VOLUME BLENDING AND COMPOSITION
# =============================================================================


def blend_ind(
    offsets: List[np.ndarray],
    pictures: List[np.ndarray],
    indices: Optional[List[int]] = None,
    inverts: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Blends a list of volume tiles into a single composite image based on specified offsets.

    This function computes the overall dimensions required by examining the offsets and the shape of each tile.
    Each tile is then placed into the composite volume at its corresponding offset (adjusted relative to the minimum offset),
    with optional inversion along specified axes. In the case of overlapping regions, later tiles will overwrite earlier ones.

    Args:
        offsets (List[np.ndarray]): A list of offset vectors for each image tile, each specified as [x, y, z].
        pictures (List[np.ndarray]): A list of 3D image tiles (with shape [z, y, x]) to be merged.
        indices (Optional[List[int]]): (Reserved for future use) Optional list specifying an order for blending.
        inverts (Optional[List[int]]): Optional axis or list of axes along which to invert (flip) each corresponding tile before blending.

    Returns:
        np.ndarray: A new 3D NumPy array (dtype uint16) representing the blended composite image.

    Raises:
        ValueError: If the list of pictures is empty or if the number of offsets does not match the number of pictures.
        RuntimeError: If an unexpected error occurs during the blending process.
    """
    if not pictures:
        raise ValueError("The list of pictures must not be empty.")
    if len(offsets) != len(pictures):
        raise ValueError("The number of offsets must match the number of pictures.")

    try:
        # Print basic statistics for debugging.
        for idx, pic in enumerate(pictures):
            print(
                f"Tile {idx} dtype: {pic.dtype}, max value: {pic.max()}, min value: {pic.min()}"
            )

        # Determine the overall Z, Y, and X ranges based on offsets and tile sizes.
        min_z = int(min(offset[2] for offset in offsets))
        max_z = int(
            max(offset[2] + tile.shape[0] for offset, tile in zip(offsets, pictures))
        )
        total_z_range = max_z - min_z

        min_y = int(min(offset[1] for offset in offsets))
        max_y = int(
            max(offset[1] + tile.shape[1] for offset, tile in zip(offsets, pictures))
        )
        total_y_range = max_y - min_y

        min_x = int(min(offset[0] for offset in offsets))
        max_x = int(
            max(offset[0] + tile.shape[2] for offset, tile in zip(offsets, pictures))
        )
        total_x_range = max_x - min_x

        # Initialize the composite image.
        newpic_shape = (total_z_range, total_y_range, total_x_range)
        newpic = np.zeros(newpic_shape, dtype=np.uint16)

        # Process and place each tile.
        for off, tile in zip(offsets, pictures):
            start_z = int(off[2] - min_z)
            start_y = int(off[1] - min_y)
            start_x = int(off[0] - min_x)

            # Apply inversion if specified.
            if inverts:
                tile = np.flip(tile, axis=inverts)

            update_range_z = slice(start_z, start_z + tile.shape[0])
            update_range_y = slice(start_y, start_y + tile.shape[1])
            update_range_x = slice(start_x, start_x + tile.shape[2])

            newpic[update_range_z, update_range_y, update_range_x] = tile

        return newpic
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred during image blending: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _calculate_safe_chunks(
    shape: Tuple[int, ...], dtype: np.dtype, chunk_size: int, max_memory_mb: int = 512
) -> Tuple[int, ...]:
    """
    Helper function to calculate safe chunk sizes to avoid memory issues.

    Args:
        shape: Volume shape
        dtype: Data type
        chunk_size: Desired chunk size along first axis
        max_memory_mb: Maximum memory per chunk in MB

    Returns:
        Tuple of chunk sizes for each dimension
    """
    bytes_per_voxel = np.dtype(dtype).itemsize

    if len(shape) >= 3:
        spatial_size = shape[1] * shape[2] * bytes_per_voxel
        max_z_slices = max(
            1, min(chunk_size, (max_memory_mb * 1024 * 1024) // spatial_size)
        )
        return (min(max_z_slices, shape[0]),) + shape[1:]
    else:
        return (min(chunk_size, shape[0]),) + shape[1:]


def bytes_to_float(b_array):
    """Converts an array of bytes to a float."""
    return float(''.join([b.decode() for b in b_array]))


def extract_positions(file_path):
    """Extracts XPosition and YPosition from a .ims file.

    Args:
    file_path (str): Path to the .ims file.

    Returns:
    tuple: (XPosition, YPosition, ZPosition) as floats or None if not found or on error.
    """
    try:
        with h5py.File(file_path, 'r') as file:
            if 'DataSetInfo/CustomData' in file:
                custom_data = file['DataSetInfo/CustomData']
                positions = {}
                for attr_name in ['XPosition', 'YPosition']:
                    if attr_name in custom_data.attrs:
                        attr_value = custom_data.attrs[attr_name]
                        positions[attr_name] = bytes_to_float(attr_value)
                    else:
                        positions[attr_name] = None  # Attribute not found
                return positions.get('XPosition'), positions.get('YPosition'), 0
            else:
                print("No 'CustomData' found in 'DataSetInfo'")
                return None, None, None
    except Exception as e:
        print(f"Failed to extract positions due to an error: {e}")
        return None, None, None


def extract_dataset_from_ims(file_path, channel):
    """Extracts dataset from .ims file for a specific channel.
    
    Args:
        file_path (str): Path to the .ims file.
        channel (int): Channel index to extract.
        
    Returns:
        np.ndarray: The extracted volume data.
    """
    with h5py.File(file_path, 'r') as file:
        # Directly access the Data which seems to contain the whole volume
        base_path = '/DataSet/ResolutionLevel 0/TimePoint 0/Channel {}/Data'.format(channel)
        
        # Assuming this path directly contains the volume
        if base_path in file:
            volume = file[base_path][:]
            print(f"Directly extracted channel {channel} volume shape: {volume.shape}")
            return volume
        else:
            raise ValueError(f"Channel {channel} not found in {file_path}")


def prepare_offsets_from_ims_files(ims_files, overlap_percentage=0.05):
    """Prepares offset array from .ims files by extracting positions.
    
    Args:
        ims_files (List[str]): List of .ims file paths.
        overlap_percentage (float): Overlap percentage between tiles for calculating increments.
        
    Returns:
        np.ndarray: Array of offsets with shape (n, 3) containing [x, y, z] positions.
    """
    offsets = []
    
    def get_offset(filename):
        x_pos, y_pos, z_pos = extract_positions(filename)
        return [x_pos if x_pos is not None else 0.0,
                y_pos if y_pos is not None else 0.0,
                z_pos if z_pos is not None else 0.0]
    
    for file_path in tqdm(ims_files, desc="Extracting positions from .ims files"):
        offsets.append(get_offset(file_path))
    
    offsets = np.array(offsets)
    # Normalize offsets to start from zero
    offsets = offsets - np.min(offsets, axis=0)
    
    # Calculate increment scale from overlap percentage
    increment_scale = (1 - overlap_percentage) * 2048
    
    # Apply increments for unique x and y values
    def apply_increments(column):
        unique_values, inverse_indices = np.unique(column, return_inverse=True)
        increments = np.zeros_like(unique_values, dtype=np.float64)
        # Compute the increment only for the unique ranks
        increments[1:] = increment_scale * np.arange(1, len(unique_values))
        # Apply the increments using the inverse indices to broadcast the increment back to the original array shape
        return increments[inverse_indices]
    
    # Modify x and y values in the array
    offsets[:, 0] += apply_increments(offsets[:, 0])
    offsets[:, 1] += apply_increments(offsets[:, 1])
    
    return offsets


# =============================================================================
# FORMAT CONVERSION FUNCTIONS
# =============================================================================


def convert_zarr_to_tiff(
    zarr_path: str,
    tiff_path: str,
    chunk_size: Optional[int] = None,
    photometric: str = "minisblack",
) -> None:
    """
    Convert Zarr volumes to TIFF format with optional chunked processing for large volumes.

    For small volumes (chunk_size=None), loads entire volume into memory for fast conversion.
    For large volumes, processes in chunks to avoid memory issues.

    Args:
        zarr_path (str): Path to input Zarr volume
        tiff_path (str): Path for output TIFF file
        chunk_size (Optional[int]): Number of Z-slices to process at once.
                                   If None, loads entire volume (faster for small volumes)
        photometric (str): TIFF photometric interpretation

    Returns:
        None
    """
    zarr_data = zarr.open(zarr_path, mode="r")

    if chunk_size is None:
        # Simple conversion - load entire volume (suitable for smaller volumes)
        print(f"Converting Zarr {zarr_data.shape} to TIFF: {tiff_path}")
        try:
            data_array = np.array(zarr_data)
            tifffile.imwrite(tiff_path, data_array, photometric=photometric)
            print(f"Conversion complete: {tiff_path}")
        except Exception as e:
            print(f"Error during conversion: {e}")
            raise
    else:
        # Chunked conversion for large volumes
        print(f"Converting large Zarr {zarr_data.shape} to TIFF (chunked): {tiff_path}")

        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tiff_writer:
            for z_start in tqdm(
                range(0, zarr_data.shape[0], chunk_size), desc="Converting to TIFF"
            ):
                z_end = min(z_start + chunk_size, zarr_data.shape[0])
                chunk = zarr_data[z_start:z_end]

                # Write each slice in the chunk
                for i, slice_data in enumerate(chunk):
                    tiff_writer.write(slice_data, photometric=photometric)

        print(f"Chunked conversion complete: {tiff_path}")


def convert_tiff_to_zarr(tiff_file: str, zarr_file: str) -> None:
    """
    Converts a TIFF image file to a Zarr-formatted file.

    This function reads a TIFF image file into a NumPy array and saves it in Zarr format.
    It supports multi-dimensional image data (e.g., 3D volumes) for efficient, chunked storage.

    Args:
        tiff_file (str): Path to the input TIFF file.
        zarr_file (str): Path where the output Zarr file will be stored.

    Returns:
        None. The Zarr file is written directly to disk.

    Note:
        Any errors during the conversion process are caught and logged.
    """
    try:
        # Read the TIFF file into a NumPy array.
        image_array = tifffile.imread(tiff_file)
        # Save the NumPy array to a Zarr file.
        zarr.save(zarr_file, image_array)
        print(f"Successfully converted {tiff_file} to {zarr_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


# =============================================================================
# DOWNSAMPLING FUNCTIONS
# =============================================================================


def downsample_zarr_volume(
    input_zarr_path: str,
    output_zarr_path: str,
    downsample_factors: Tuple[float, float, float],
    chunk_size: int = 50,
    compression: str = "zstd",
    compression_level: int = 3,
) -> None:
    """
    Memory-efficient downsampling of Zarr volumes with chunked processing.

    Processes the volume in chunks to avoid loading the entire dataset into memory,
    making it suitable for very large microscopy volumes.

    Args:
        input_zarr_path (str): Path to input Zarr volume
        output_zarr_path (str): Path for output downsampled Zarr volume
        downsample_factors (Tuple[float, float, float]): Downsampling factors for (z, y, x)
        chunk_size (int): Number of Z-slices to process at once
        compression (str): Compression algorithm ('zstd', 'lz4', 'blosc')
        compression_level (int): Compression level (1-9)

    Returns:
        None
    """
    # Open input Zarr volume
    input_data = zarr.open(input_zarr_path, mode="r")

    # Calculate output shape
    original_shape = input_data.shape
    output_shape = tuple(
        int(dim / factor) for dim, factor in zip(original_shape, downsample_factors)
    )

    print(f"Downsampling from {original_shape} to {output_shape}")

    # Create output Zarr with compression
    compressor = Blosc(
        cname=compression, clevel=compression_level, shuffle=Blosc.BITSHUFFLE
    )
    output_chunks = _calculate_safe_chunks(output_shape, input_data.dtype, chunk_size)

    output_data = zarr.open(
        output_zarr_path,
        mode="w",
        shape=output_shape,
        dtype=input_data.dtype,
        chunks=output_chunks,
        compressor=compressor,
    )

    # Process in chunks along Z-axis
    scale_factors = tuple(1.0 / factor for factor in downsample_factors)

    for z_start in tqdm(
        range(0, original_shape[0], chunk_size), desc="Downsampling chunks"
    ):
        z_end = min(z_start + chunk_size, original_shape[0])

        # Load chunk
        chunk = input_data[z_start:z_end]

        # Downsample chunk
        downsampled_chunk = zoom(chunk, scale_factors, order=1)

        # Calculate output indices
        out_z_start = int(z_start / downsample_factors[0])
        out_z_end = min(out_z_start + downsampled_chunk.shape[0], output_shape[0])

        # Store downsampled chunk
        output_data[out_z_start:out_z_end] = downsampled_chunk[
            : out_z_end - out_z_start
        ]

    print(f"Downsampling complete: {output_zarr_path}")


def downsample_tiff(
    input_path: str, output_path: str, factors: tuple, order: int = 1
) -> None:
    """
    Downsamples a 3D TIFF image using specified downsampling factors and interpolation order.

    The function performs the following steps:
      1. Reads the image from the input TIFF file.
      2. Computes the downsampling scale as the reciprocal of each provided factor.
      3. Uses scipy.ndimage.zoom to rescale the image with the specified spline interpolation order.
      4. Writes the downsampled image to the output TIFF file using the 'minisblack' photometric setting.

    Args:
        input_path (str): Path to the input TIFF file containing the original 3D image volume.
        output_path (str): Path where the downsampled TIFF image will be saved.
        factors (tuple or list of float): Downsampling factors for each axis (e.g., (6, 6, 6)).
        order (int, optional): Order of the spline interpolation used in zoom (default is 1 for linear interpolation).

    Returns:
        None. The downsampled image is saved directly to disk.

    Example:
        >>> downsample_tiff("input_volume.tif", "downsampled_volume.tif", (6, 6, 6), order=1)
    """
    try:
        # Read the 3D image from the TIFF file.
        image_array = tifffile.imread(input_path)
        print(f"Input image shape: {image_array.shape}")

        # Compute the scale factors as the reciprocal of each provided factor.
        scale_factors = tuple(1 / factor for factor in factors)

        # Downsample the image.
        downsampled_array = zoom(image_array, scale_factors, order=order)
        print(f"Downsampled image shape: {downsampled_array.shape}")

        # Save the downsampled image.
        tifffile.imwrite(output_path, downsampled_array, photometric="minisblack")
        print(f"Downsampled image saved to {output_path}")

    except Exception as e:
        print(f"Error during downsampling: {e}")
        raise


# =============================================================================
# UPSAMPLING FUNCTIONS
# =============================================================================


def upsample_segmentation_labels(
    input_zarr_path: str,
    output_zarr_path: str,
    upsample_factors: Tuple[float, float, float],
    chunk_size: int = 50,
    compression: str = "zstd",
    target_shape: Optional[Tuple[int, int, int]] = None,
) -> None:
    """
    Upsample segmentation label volumes while preserving integer label values.

    Uses nearest-neighbor interpolation (order=0) to maintain discrete integer
    labels during upsampling, essential for segmentation masks.

    Args:
        input_zarr_path (str): Path to input segmentation Zarr volume
        output_zarr_path (str): Path for upsampled output Zarr volume
        upsample_factors (Tuple[float, float, float]): Upsampling factors for (z, y, x)
        chunk_size (int): Number of Z-slices to process at once
        compression (str): Compression algorithm for output
        target_shape (Optional[Tuple[int, int, int]]): Override output shape (z, y, x).
                                                      If provided, takes precedence over
                                                      calculated shape from upsample_factors.
                                                      Essential for masks that must exactly
                                                      match full resolution volume dimensions.

    Returns:
        None
    """
    # Open input segmentation volume
    input_data = zarr.open(input_zarr_path, mode="r")

    # Calculate output shape
    original_shape = input_data.shape

    if target_shape is not None:
        # Use override target shape
        output_shape = target_shape
        # Recalculate actual upsample factors based on target shape
        actual_factors = tuple(
            target_dim / orig_dim
            for target_dim, orig_dim in zip(target_shape, original_shape)
        )
        print(
            f"Upsampling segmentation from {original_shape} to {output_shape} (target override)"
        )
        print(f"Actual upsample factors: {actual_factors}")
    else:
        # Use calculated shape from provided factors
        output_shape = tuple(
            int(dim * factor) for dim, factor in zip(original_shape, upsample_factors)
        )
        actual_factors = upsample_factors
        print(f"Upsampling segmentation from {original_shape} to {output_shape}")

    # Create output Zarr with compression
    compressor = Blosc(cname=compression, clevel=3, shuffle=Blosc.BITSHUFFLE)
    output_chunks = _calculate_safe_chunks(output_shape, input_data.dtype, chunk_size)

    output_data = zarr.open(
        output_zarr_path,
        mode="w",
        shape=output_shape,
        dtype=input_data.dtype,  # Preserve integer dtype for labels
        chunks=output_chunks,
        compressor=compressor,
    )

    # Process in chunks along Z-axis
    for z_start in tqdm(
        range(0, original_shape[0], chunk_size), desc="Upsampling labels"
    ):
        z_end = min(z_start + chunk_size, original_shape[0])

        # Load chunk
        chunk = input_data[z_start:z_end]

        # Upsample chunk with nearest-neighbor (order=0) to preserve labels
        upsampled_chunk = zoom(chunk, actual_factors, order=0)

        # Calculate output indices
        out_z_start = int(z_start * actual_factors[0])
        out_z_end = min(out_z_start + upsampled_chunk.shape[0], output_shape[0])

        # Store upsampled chunk
        output_data[out_z_start:out_z_end] = upsampled_chunk[: out_z_end - out_z_start]

    print(f"Label upsampling complete: {output_zarr_path}")


# =============================================================================
# CHANNEL OPERATIONS
# =============================================================================


def merge_zarr_channels(
    channel_a_path: str,
    channel_b_path: str,
    output_path: str,
    merge_strategy: str = "mean",
    block_depth: int = 50,
    compression: str = "zstd",
    scale_to_uint16: bool = True,
) -> None:
    """
    Memory-efficient merging of two Zarr volumes representing different imaging channels.

    Supports different merging strategies for combining channels (e.g., 405nm and 488nm)
    commonly used in microscopy registration workflows.

    Optionally scales each input volume to the full uint16 range using that volume's
    *global* (cropped) min and max, computed in a first pass. Scaling is then applied
    per block during the merge pass.

    Args:
        channel_a_path (str): Path to first channel Zarr volume
        channel_b_path (str): Path to second channel Zarr volume
        output_path (str): Path for merged output Zarr volume
        merge_strategy (str): Merging method - "mean", "max", or "stack"
        block_depth (int): Number of Z-slices to process per block
        compression (str): Compression algorithm for output
        scale_to_uint16 (bool): If True, scale each input to [0, 65535] via its
            global min/max before merging (computed once over the cropped region).

    Returns:
        None
    """
    # Open input volumes
    vol_a = zarr.open(channel_a_path, mode="r")
    vol_b = zarr.open(channel_b_path, mode="r")

    print(f"Channel A shape: {vol_a.shape}, dtype: {vol_a.dtype}")
    print(f"Channel B shape: {vol_b.shape}, dtype: {vol_b.dtype}")

    # Ensure compatible shapes (crop to common overlapping region)
    if vol_a.shape != vol_b.shape:
        min_shape = tuple(min(sa, sb) for sa, sb in zip(vol_a.shape, vol_b.shape))
        print(f"Cropping to common shape: {min_shape}")
    else:
        min_shape = vol_a.shape

    mz, my, mx = min_shape

    # Configure output based on merge strategy
    if merge_strategy == "stack":
        out_shape = (2, mz, my, mx)
        out_chunks = (1, min(block_depth, mz), min(512, my), min(512, mx))
    else:
        out_shape = (mz, my, mx)
        out_chunks = (min(block_depth, mz), min(512, my), min(512, mx))

    # Choose output dtype
    out_dtype = (
        np.uint16 if scale_to_uint16 else np.dtype(vol_a.dtype).newbyteorder("<")
    )

    # Create output Zarr
    compressor = Blosc(cname=compression, clevel=3, shuffle=Blosc.SHUFFLE)
    z_out = zarr.open(
        output_path,
        mode="w",
        shape=out_shape,
        chunks=out_chunks,
        dtype=out_dtype,
        compressor=compressor,
    )

    def _compute_min_max(vol, mz, my, mx, block_depth):
        gmin = np.inf
        gmax = -np.inf
        for z0 in tqdm(range(0, mz, block_depth), desc="Scanning min/max"):
            z1 = min(z0 + block_depth, mz)
            blk = vol[z0:z1, :my, :mx]
            # ensure numpy array (zarr returns ndarray-like already)
            bmin = np.min(blk)
            bmax = np.max(blk)
            if bmin < gmin:
                gmin = float(bmin)
            if bmax > gmax:
                gmax = float(bmax)
        # Handle empty/degenerate cases defensively
        if not np.isfinite(gmin) or not np.isfinite(gmax):
            gmin, gmax = 0.0, 0.0
        return gmin, gmax

    def _scale_block_to_uint16(block, vmin, vmax):
        if vmax <= vmin:
            # Flat volume: map everything to 0
            out = np.zeros_like(block, dtype=np.uint16)
            return out
        scale = 65535.0 / (vmax - vmin)
        # Use float32 for memory efficiency; rint to keep integer behavior
        out = np.rint((block.astype(np.float32) - vmin) * scale)
        # Clip and cast
        np.clip(out, 0.0, 65535.0, out=out)
        return out.astype(np.uint16, copy=False)

    # First pass: compute global min/max if scaling is requested
    if scale_to_uint16:
        print("Computing global min/max for scaling...")
        a_min, a_max = _compute_min_max(vol_a, mz, my, mx, block_depth)
        b_min, b_max = _compute_min_max(vol_b, mz, my, mx, block_depth)
        print(f"A: min={a_min}, max={a_max} | B: min={b_min}, max={b_max}")

    # Second pass: process blocks along Z-axis and write output
    for z0 in tqdm(
        range(0, mz, block_depth), desc=f"Merging channels ({merge_strategy})"
    ):
        z1 = min(z0 + block_depth, mz)
        a_block = vol_a[z0:z1, :my, :mx]
        b_block = vol_b[z0:z1, :my, :mx]

        if scale_to_uint16:
            a_block = _scale_block_to_uint16(a_block, a_min, a_max)
            b_block = _scale_block_to_uint16(b_block, b_min, b_max)
        else:
            # Keep original; we will cast as needed on write
            pass

        if merge_strategy == "max":
            merged_block = np.maximum(a_block, b_block)
            z_out[z0:z1, :, :] = merged_block.astype(out_dtype, copy=False)

        elif merge_strategy == "mean":
            # Compute in float32 to avoid overflow, then round and cast
            merged_block = (
                a_block.astype(np.float32) + b_block.astype(np.float32)
            ) * 0.5
            merged_block = np.rint(merged_block).astype(out_dtype, copy=False)
            z_out[z0:z1, :, :] = merged_block

        elif merge_strategy == "stack":
            # Write each as its own slice (scaled if requested)
            z_out[0, z0:z1, :, :] = a_block.astype(out_dtype, copy=False)
            z_out[1, z0:z1, :, :] = b_block.astype(out_dtype, copy=False)

        else:
            raise ValueError(f"Unknown merge_strategy: {merge_strategy}")

    print(
        f"Channel merging complete: {output_path} (shape: {z_out.shape}, dtype: {z_out.dtype})"
    )


def stack_tiff_images(file1: str, file2: str, output_file: str) -> None:
    """
    Stacks two TIFF images along a new channel axis and writes the result to an output file.

    The function performs the following steps:
      1. Reads two TIFF images (expected to have the same shape, e.g., (z, y, x)).
      2. Verifies that the shapes match; otherwise, raises a ValueError.
      3. Stacks the images along a new axis (resulting shape: (z, 2, y, x)) to create a multi-channel volume.
      4. Saves the stacked image to the specified output file.

    Args:
        file1 (str): Path to the first input TIFF file.
        file2 (str): Path to the second input TIFF file.
        output_file (str): Path where the output stacked TIFF image will be saved.

    Returns:
        None. The stacked image is written directly to disk.

    Raises:
        ValueError: If the input images do not have the same shape.
    """
    try:
        # Read the first TIFF image.
        with tifffile.TiffFile(file1) as tif:
            img1 = tif.asarray()

        # Read the second TIFF image.
        with tifffile.TiffFile(file2) as tif:
            img2 = tif.asarray()

        # Check that both images have the same shape.
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same shape (z, y, x)")

        # Stack the images along a new axis to create a two-channel volume.
        stacked_img = np.stack([img1, img2], axis=1)

        # Save the stacked image.
        tifffile.imwrite(output_file, stacked_img)
        print(f"Stacked image saved as {output_file}")

    except Exception as e:
        print(f"Error during stacking: {e}")
        raise


# =============================================================================
# INTENSITY SCALING FUNCTIONS
# =============================================================================


def scale_intensity_to_uint16(
    input_zarr_path: str,
    output_zarr_path: str,
    intensity_range: Optional[Tuple[float, float]] = None,
    chunk_size: int = 50,
) -> None:
    """
    Scale intensity values in a Zarr volume to uint16 range [0, 65535].

    Useful for normalizing microscopy data before processing or visualization.

    Args:
        input_zarr_path (str): Path to input Zarr volume
        output_zarr_path (str): Path for scaled output Zarr volume
        intensity_range (Optional[Tuple[float, float]]): Min/max values for scaling.
                                                        If None, uses global min/max
        chunk_size (int): Number of Z-slices to process at once

    Returns:
        None
    """
    # Open input volume
    input_data = zarr.open(input_zarr_path, mode="r")

    # Determine intensity range
    if intensity_range is None:
        print("Computing global intensity range...")
        min_val = float(np.min(input_data))
        max_val = float(np.max(input_data))
    else:
        min_val, max_val = intensity_range

    print(f"Scaling intensity range [{min_val}, {max_val}] to uint16")

    # Create output Zarr
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    output_chunks = _calculate_safe_chunks(input_data.shape, np.uint16, chunk_size)

    output_data = zarr.open(
        output_zarr_path,
        mode="w",
        shape=input_data.shape,
        dtype=np.uint16,
        chunks=output_chunks,
        compressor=compressor,
    )

    # Process in chunks
    for z_start in tqdm(
        range(0, input_data.shape[0], chunk_size), desc="Scaling intensity"
    ):
        z_end = min(z_start + chunk_size, input_data.shape[0])
        chunk = input_data[z_start:z_end]

        # Scale to [0, 65535] range
        if max_val > min_val:
            scaled_chunk = (chunk.astype(np.float32) - min_val) * (
                65535.0 / (max_val - min_val)
            )
            scaled_chunk = np.clip(scaled_chunk, 0.0, 65535.0)
            scaled_chunk = (scaled_chunk + 0.5).astype(np.uint16)
        else:
            scaled_chunk = np.zeros_like(chunk, dtype=np.uint16)

        output_data[z_start:z_end] = scaled_chunk

    print(f"Intensity scaling complete: {output_zarr_path}")


# =============================================================================
# VOLUME TRANSFORMATION FUNCTIONS
# =============================================================================


def reorient_volume_and_save_tiff(
    input_path: str, output_path: str, rotation: int, flip: bool
) -> np.ndarray:
    """
    Reorients a 3D volume from a TIFF file by applying a specified rotation (in multiples of 90°)
    and an optional flip along the first (z) axis, then saves the result as a TIFF file.

    The rotation is performed on the (y, x) axes using numpy.rot90. Allowed rotation values are 0, 90, 180, or 270 degrees.
    For example:
      - rotation=90 rotates the volume 90° clockwise.
      - rotation=180 rotates the volume 180°.
      - rotation=270 rotates the volume 270° clockwise (or 90° counter-clockwise).
    After rotation, if flip is True, the volume is flipped along the first (z) axis.

    Args:
        input_path (str): Path to the input TIFF file containing the 3D volume.
        output_path (str): Path where the reoriented TIFF file will be saved.
        rotation (int): Rotation angle in degrees. Must be one of [0, 90, 180, 270].
        flip (bool): If True, the volume is flipped along the first (z) axis after rotation.

    Returns:
        np.ndarray: The reoriented volume as a NumPy array.

    Example:
        >>> reoriented = reorient_volume_and_save_tiff("input.tif", "output.tif", rotation=90, flip=True)
    """
    # Read the volume from the TIFF file.
    volume = tifffile.imread(input_path)

    # Validate the rotation value.
    if rotation not in [0, 90, 180, 270]:
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees.")

    # Compute the number of 90° rotations (negative for clockwise rotation).
    k = -(rotation // 90)

    # Rotate the volume along the (y, x) axes.
    reoriented = np.rot90(volume, k=k, axes=(1, 2)) if k != 0 else volume.copy()

    # Optionally flip the volume along the first (z) axis.
    if flip:
        reoriented = reoriented[::-1, :, :]

    # Save the reoriented volume to a TIFF file.
    tifffile.imwrite(output_path, reoriented)
    print(f"Reoriented volume saved as TIFF at: {output_path}")

    return reoriented
