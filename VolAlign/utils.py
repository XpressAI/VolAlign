import zarr
import tifffile
import numpy as np
from scipy.ndimage import zoom
from typing import List, Optional
from exm.stitching.tileset import Tileset

def blend_ind(offsets: List[np.ndarray],
              pictures: List[np.ndarray],
              indices: Optional[List[int]] = None,
              inverts: Optional[List[int]] = None) -> np.ndarray:
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
            print(f"Tile {idx} dtype: {pic.dtype}, max value: {pic.max()}, min value: {pic.min()}")
        
        # Determine the overall Z, Y, and X ranges based on offsets and tile sizes.
        min_z = int(min(offset[2] for offset in offsets))
        max_z = int(max(offset[2] + tile.shape[0] for offset, tile in zip(offsets, pictures)))
        total_z_range = max_z - min_z

        min_y = int(min(offset[1] for offset in offsets))
        max_y = int(max(offset[1] + tile.shape[1] for offset, tile in zip(offsets, pictures)))
        total_y_range = max_y - min_y 

        min_x = int(min(offset[0] for offset in offsets))
        max_x = int(max(offset[0] + tile.shape[2] for offset, tile in zip(offsets, pictures)))
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


def convert_zarr_to_tiff(zarr_file: str, tiff_file: str) -> None:
    """
    Converts a Zarr-formatted file into a TIFF image file.
    
    This function reads a dataset from a Zarr file, converts it to a NumPy array, and writes
    the array to a TIFF file using the 'minisblack' photometric convention. It supports multi-dimensional data.
    
    Args:
        zarr_file (str): Path to the input Zarr file.
        tiff_file (str): Path to the output TIFF file.
    
    Returns:
        None. The TIFF file is written directly to disk.
    
    Note:
        Any errors during reading or writing are caught and logged.
    """
    try:
        # Open the Zarr file in read mode and convert to a NumPy array.
        zarr_data = zarr.open(zarr_file, mode='r')
        data_array = np.array(zarr_data)
        # Write the data array to a TIFF file.
        tifffile.imwrite(tiff_file, data_array, photometric='minisblack')
        print(f"Successfully converted {zarr_file} to {tiff_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")


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


def downsample_tiff(input_path: str, output_path: str, factors: tuple, order: int = 1) -> None:
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
        tifffile.imwrite(output_path, downsampled_array, photometric='minisblack')
        print(f"Downsampled image saved to {output_path}")

    except Exception as e:
        print(f"Error during downsampling: {e}")


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


def reorient_volume_and_save_tiff(input_path: str, 
                                    output_path: str, 
                                    rotation: int, 
                                    flip: bool) -> np.ndarray:
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
