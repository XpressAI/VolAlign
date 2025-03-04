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
    Blends a list of volume tiles together according to specified offsets,
    accounting for different z-stack sizes and starting points.
    
    Args:
        offsets (List[np.ndarray]): An array of offsets for each image tile.
        pictures (List[np.ndarray]): A list of image tiles to blend.
        indices (Optional[List[int]]): Optional list of indices specifying the order 
                                       in which to blend the images.
        inverts (Optional[List[int]]): Optional list of axes along which to invert the corresponding image tiles.
    
    Returns:
        np.ndarray: The blended image.
    
    Raises:
        ValueError: If the list of pictures is empty or if the number of offsets does not match the number of pictures.
        RuntimeError: If an unexpected error occurs during blending.
    """
    if not pictures:
        raise ValueError("The list of pictures must not be empty.")
    if len(offsets) != len(pictures):
        raise ValueError("The number of offsets must match the number of pictures.")
    
    try:
        # Print basic statistics for debugging.
        for idx, pic in enumerate(pictures):
            print(f"Tiile {idx} dtype: {pic.dtype}, max value: {pic.max()}, min value: {pic.min()}")
        
        # Determine the overall Z, Y, and X ranges.
        min_z = int(min(offset[2] for offset in offsets))
        max_z = int(max(offset[2] + tile.shape[0] for offset, tile in zip(offsets, pictures)))
        total_z_range = max_z - min_z

        min_y = int(min(offset[1] for offset in offsets))
        max_y = int(max(offset[1] + tile.shape[1] for offset, tile in zip(offsets, pictures)))
        total_y_range = max_y - min_y 

        min_x = int(min(offset[0] for offset in offsets))
        max_x = int(max(offset[0] + tile.shape[2] for offset, tile in zip(offsets, pictures)))
        total_x_range = max_x - min_x  

        # Initialize the blended image.
        newpic_shape = (total_z_range, total_y_range, total_x_range)
        newpic = np.zeros(newpic_shape, dtype=np.uint16)
        
        # Process each tile.
        for off, tile in zip(offsets, pictures):
            start_z = int(off[2] - min_z)
            start_y = int(off[1] - min_y)
            start_x = int(off[0] - min_x)
            
            # Adjust for inverts if provided.
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
    Convert a Zarr-formatted file to a TIFF image file.
    
    This function reads a dataset from a Zarr file, converts it to a NumPy array,
    and writes the array to a TIFF file using the 'minisblack' photometric convention.
    It is designed to handle multi-dimensional data, including 3D image volumes.
    
    Args:
        zarr_file (str): Path to the input Zarr file.
        tiff_file (str): Path to the output TIFF file.
        
    Raises:
        Exception: If an error occurs during the reading of the Zarr file or the writing
                   of the TIFF file.
        
    """
    try:
        # Open the Zarr file in read mode
        zarr_data = zarr.open(zarr_file, mode='r')
        # Convert the Zarr data to a NumPy array
        data_array = np.array(zarr_data)
        # Write the NumPy array to a TIFF file with the 'minisblack' photometric setting
        tifffile.imwrite(tiff_file, data_array, photometric='minisblack')
        print(f"Successfully converted {zarr_file} to {tiff_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")


def convert_tiff_to_zarr(tiff_file: str, zarr_file: str) -> None:
    """
    Convert a TIFF image file to a Zarr-formatted file.
    
    This function reads a TIFF image file into a NumPy array and saves it in Zarr format.
    It supports multi-dimensional image data, such as 3D volumes, making it useful for 
    converting large image datasets into a format that supports chunked, compressed storage.
    
    Args:
        tiff_file (str): Path to the input TIFF file.
        zarr_file (str): Path where the output Zarr file will be stored.
        
    Raises:
        Exception: If an error occurs during the reading of the TIFF file or the writing
                   of the Zarr file.
    """
    try:
        # Read the TIFF file into a NumPy array
        image_array = tifffile.imread(tiff_file)
        # Save the NumPy array to a Zarr file
        zarr.save(zarr_file, image_array)
        print(f"Successfully converted {tiff_file} to {zarr_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")


def downsample_tiff(input_path: str, output_path: str, factors: tuple, order: int = 1) -> None:
    """
    Reads a 3D TIFF image, downsamples it using specified factors and interpolation order, 
    and writes the downsampled image to a new TIFF file.
    
    This function performs the following steps:
      1. Reads the image from the given input TIFF file path.
      2. Computes the downsampling scale as the reciprocal of each factor.
      3. Uses scipy.ndimage.zoom to downsample the image with the specified spline interpolation order.
      4. Saves the downsampled image to the given output TIFF file path.
    
    Args:
        input_path (str): Path to the input TIFF file containing the original 3D image volume.
        output_path (str): Path where the downsampled TIFF image will be saved.
        factors (tuple or list of float): Downsampling factors for each axis (e.g., (6, 6, 6)).
        order (int, optional): The order of the spline interpolation used in zoom. 
                               Default is 1 (linear interpolation).
    
    Returns:
        None. The function writes the downsampled image directly to the output file.
    
    Example:
        >>> downsample_tiff("input_volume.tif", "downsampled_volume.tif", (6, 6, 6), order=1)
    """
    try:
        # Read the 3D image from the TIFF file
        image_array = tifffile.imread(input_path)
        print(f"Input image shape: {image_array.shape}")

        # Compute downsampling scale for each dimension (reciprocal of each factor)
        scale_factors = tuple(1 / factor for factor in factors)
        
        # Downsample the image using the computed scale factors and specified interpolation order
        downsampled_array = zoom(image_array, scale_factors, order=order)
        print(f"Downsampled image shape: {downsampled_array.shape}")

        # Save the downsampled image to the output TIFF file
        tifffile.imwrite(output_path, downsampled_array, photometric='minisblack')
        print(f"Downsampled image saved to {output_path}")

    except Exception as e:
        print(f"Error during downsampling: {e}")


def stack_tiff_images(file1: str, file2: str, output_file: str) -> None:
    """
    Reads two TIFF files, verifies they have the same shape, stacks them along a new channel axis,
    and saves the resulting stacked image to an output file.

    This function performs the following steps:
      1. Reads two input TIFF images using tifffile.
      2. Checks that both images have the same shape (expected shape: (z, y, x)).
      3. Stacks the two images along the second axis (resulting in a shape of (z, 2, y, x)),
         effectively creating a two-channel volume.
      4. Writes the stacked image to the specified output TIFF file.
      
    Args:
        file1 (str): Path to the first input TIFF file.
        file2 (str): Path to the second input TIFF file.
        output_file (str): Path where the output stacked TIFF image will be saved.
    
    Raises:
        ValueError: If the input images do not have the same shape.
    
    """
    try:
        # Read the first TIFF image
        with tifffile.TiffFile(file1) as tif:
            img1 = tif.asarray()

        # Read the second TIFF image
        with tifffile.TiffFile(file2) as tif:
            img2 = tif.asarray()

        # Ensure both images have the same shape (expected: (z, y, x))
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same shape (z, y, x)")

        # Stack the images along a new axis (axis 1) to create a 2-channel volume (z, 2, y, x)
        stacked_img = np.stack([img1, img2], axis=1)

        # Save the stacked image to the output file
        tifffile.imwrite(output_file, stacked_img)
        print(f"Stacked image saved as {output_file}")

    except Exception as e:
        print(f"Error during stacking: {e}")


def reorient_volume_and_save_tiff(input_path: str, 
                                    output_path: str, 
                                    rotation: int, 
                                    flip: bool) -> np.ndarray:
    """
    Reads a 3D volume from a TIFF file, reorients it by applying a specified rotation 
    (in multiples of 90°) and an optional flip along the first (z) axis, and saves the 
    resulting volume as a TIFF file.
    
    The rotation is applied clockwise on the (y, x) axes using np.rot90. Allowed rotation 
    values are 0, 90, 180, or 270 degrees. For example:
      - rotation=90 rotates the volume 90° clockwise.
      - rotation=180 rotates the volume 180°.
      - rotation=270 rotates the volume 270° clockwise (or 90° counter-clockwise).
    
    After rotation, if flip is True, the volume is flipped along the first axis.
    
    Args:
        input_path (str): Path to the input TIFF file containing the 3D volume.
        output_path (str): Path where the reoriented TIFF file will be saved.
        rotation (int): Rotation angle in degrees. Must be one of [0, 90, 180, 270].
        flip (bool): If True, flip the volume along the first (z) axis after rotation.
    
    Returns:
        np.ndarray: The reoriented volume.
    
    Example:
        >>> reoriented = reorient_volume_from_tiff_and_save_tiff("input.tif", "output.tif", rotation=90, flip=True)
    """
    # Read the volume from the TIFF file.
    volume = tifffile.imread(input_path)
    
    # Validate the rotation value.
    if rotation not in [0, 90, 180, 270]:
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees.")
    
    # Compute the number of 90° clockwise rotations.
    # np.rot90 rotates counter-clockwise when k is positive.
    # To rotate clockwise, use a negative k.
    k = -(rotation // 90)
    
    # Rotate the volume along the (y, x) axes.
    reoriented = np.rot90(volume, k=k, axes=(1, 2)) if k != 0 else volume.copy()
    
    # Optionally flip the volume along the first (z) axis.
    if flip:
        reoriented = reoriented[::-1, :, :]
    
    # Save the reoriented volume as a TIFF file.
    tifffile.imwrite(output_path, reoriented)
    print(f"Reoriented volume saved as TIFF at: {output_path}")
    
    return reoriented


