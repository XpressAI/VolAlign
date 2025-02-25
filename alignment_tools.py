
import os
import tifffile
import numpy as np
import SimpleITK as sitk
from npy2bdv import BdvWriter
from bigstream.align import alignment_pipeline
from tqdm import tqdm
from typing import List, Optional
from exm.stitching.tileset import Tileset
from VolAlign.utils import blend_ind


def create_bdv_xml(tiles_folder: str,
                   fov_list: list,
                   offset_array: np.ndarray,
                   voxel_size: list,
                   nchannels: int = 2,
                   increment_scale: float = 1900.0) -> None:
    """
    Creates a BDV/XML file for BigStitcher from a set of FOV TIFF images and their corresponding offsets.
    
    The provided offset_array must be given in the order [column, row, z]:
      - The first element represents the column position in the final stitched volume.
      - The second element represents the row position.
      - The third element represents the z-offset.
      
    For example:
        [0.0, 0.0, 0.0] corresponds to the first column, first row.
        [2.0, 1.0, 0.0] corresponds to the third column, second row.
    
    The function adjusts the provided column and row offsets by adding increments based on unique values,
    ensuring that each FOV is clearly positioned (i.e. its raw column/row location) in the final stitched volume.
    It then writes the BDV/XML file using the npy2bdv library.
    
    Args:
        tiles_folder (str): Path to the output folder (or H5 file) where the BDV/XML file and associated tile data will be saved.
        fov_list (list of str): List of file paths to the FOV TIFF images.
        offset_array (np.ndarray): Array of shape (n, 3) containing the offsets [column, row, z] for each FOV.
        voxel_size (list or tuple): Voxel size for the volume in [z, y, x] order.
        nchannels (int, optional): Number of channels in each FOV image. Default is 2.
        increment_scale (float, optional): Scale factor used to compute increments for unique offset values. Default is 1900.0.
    
    Returns:
        None. The BDV/XML file is written to the specified output folder.
    
    Raises:
        Exception: If an error occurs during file I/O or BDV/XML file creation.
    
    """
    try:
        # Helper function: Compute increments based on unique values in a coordinate array.
        def apply_increments(coordinate):
            unique_vals, inverse_indices = np.unique(coordinate, return_inverse=True)
            increments = np.zeros_like(unique_vals, dtype=np.float64)
            # For each unique value (except the first), add an increment proportional to its rank.
            increments[1:] = increment_scale * np.arange(1, len(unique_vals))
            return increments[inverse_indices]
        
        # Adjust the offsets for column and row positions.
        # Note: The offset_array is assumed to be [column, row, z].
        adjusted_offsets = offset_array.copy()
        adjusted_offsets[:, 0] += apply_increments(adjusted_offsets[:, 0])  # Adjust column positions.
        adjusted_offsets[:, 1] += apply_increments(adjusted_offsets[:, 1])  # Adjust row positions.
        
        # Initialize the BDV writer with the specified number of FOVs, channels, and voxel size.
        bdv_writer = BdvWriter(tiles_folder, ntiles=len(fov_list), nchannels=nchannels, overwrite=True, voxel_size=voxel_size)
        
        # Process each FOV file.
        for i, fov_file in tqdm(enumerate(fov_list), total=len(fov_list), desc="Processing FOVs"):
            # Read the multi-channel TIFF image (assumed shape: [z, channels, y, x]).
            fov_image = tifffile.imread(fov_file)
            
            # Append each channel view for the current FOV.
            for channel_index in range(nchannels):
                # Extract the tile for the given channel.
                tile = fov_image[:, channel_index, :, :]
                
                # Create a 3x4 affine transformation matrix from an identity matrix and set its translation.
                unit_matrix = np.eye(4)[:3, :].copy()
                unit_matrix[:, 3] = adjusted_offsets[i]
                
                bdv_writer.append_view(
                    tile.astype(np.uint16),
                    time=0,
                    tile=i,
                    m_affine=unit_matrix,
                    channel=channel_index,
                    name_affine=f"tile {i} translation"
                )
        
        # Write the BDV/XML file and close the writer.
        bdv_writer.write_xml()
        bdv_writer.close()
        print(f"BDV/XML file successfully created in {tiles_folder}")
    
    except Exception as e:
        print(f"Error during BDV/XML creation: {e}")
        raise


def blend_tiles(xml_file: str,
                output_folder: str,
                voxel_size: List[float],
                channels: List[str]) -> None:
    """
    Blends stitched tile images for each channel and saves a TIFF file for each channel.
    
    The function performs the following steps:
      1. Loads a tileset from a BDV/XML file (generated by BigStitcher) using the provided voxel size.
      2. Groups tiles into chunks of size 'group_size'. Each group is assumed to correspond to one channel.
      3. For each group, the function computes offsets (converted to pixel coordinates by dividing by the voxel size)
         and collects the corresponding image tiles.
      4. The tiles are blended using `blend_ind2` to generate a single volume per channel.
      5. The blended volume is then saved as a TIFF file named "stitched_{channel}.tif" in the specified output folder.
    
    Args:
        xml_file (str): Path to the BDV/XML file generated by BigStitcher.
        output_folder (str): Folder where the output TIFF files will be saved.
        voxel_size (List[float]): Voxel size as a list in [z, y, x] order.
        channels (List[str]): List of channel names (e.g., ['CH1', 'CH2']).
    
    Returns:
        None. Blended TIFF files are written to disk.
        
    """
    # Initialize the tileset from the BDV/XML file.
    ts = Tileset(voxel_size)
    ts.init_from_bdv(xml_file)
    total_tiles = len(ts)
    group_size = int(total_tiles/2)
    
    # Process tiles in groups.
    for i in range(0, total_tiles, group_size):
        group_index = i // group_size
        # Determine channel name: use provided channel list if available, else default naming.
        if group_index < len(channels):
            channel_name = channels[group_index]
        else:
            channel_name = f"channel_{group_index}"
        
        offsets_group = []
        images_group = []
        
        # Collect tiles for the current group.
        for j in range(i, min(i + group_size, total_tiles)):
            # Convert offset to pixel units by dividing by the voxel size.
            offsets_group.append(ts[j].offset / ts.voxel_size)
            images_group.append(ts[j].img)
        
        # Blend the collected tiles.
        stitched_volume = blend_ind(offsets_group, images_group)
        
        # Build the output file path.
        output_path = os.path.join(output_folder, f"stitched_{channel_name}.tif")
        tifffile.imwrite(output_path, stitched_volume)
        print(f"Saved blended volume for {channel_name} with shape {stitched_volume.shape} to {output_path}")


def voxel_spacing_resample(input_file: str,
                          output_path: str,
                          original_spacing: list,
                          target_spacing: list,
                          interpolator=sitk.sitkLinear) -> sitk.Image:
    """
    Loads a 3D .tif image, sets its original spacing, and resamples it to a target spacing.
    
    This function performs the following steps:
      1. Reads the image from the provided .tif file path using tifffile.
      2. Converts the NumPy array to a SimpleITK image and casts it to uint16.
      3. Sets the image spacing based on the provided original spacing.
      4. Computes the new image size required for the target spacing.
      5. Uses a SimpleITK resampler to adjust the image to the desired target spacing.
    
    Args:
        input_file (str): Path to the 3D .tif image file.
        input_file (str): Path to save the 3D .tif resampled image file.
        original_spacing (list or tuple): Original spacing in [z, y, x] order.
        target_spacing (list or tuple): Desired target spacing in [z, y, x] order.
        interpolator (SimpleITK interpolator, optional): Interpolation method (default: sitk.sitkLinear).
    
    Returns:
        SimpleITK.Image: The resampled image.
    """
    # Load the 3D image from the .tif file
    image_array = tifffile.imread(input_file)
    
    # Convert the NumPy array to a SimpleITK image and cast to uint16
    image = sitk.GetImageFromArray(image_array)
    image = sitk.Cast(image, sitk.sitkUInt16)
    
    # Set the original spacing
    image.SetSpacing(original_spacing)
    
    # Calculate the new image size based on target spacing
    original_size = np.array(image.GetSize(), dtype=np.int16)
    target_size = np.round(original_size * np.array(original_spacing) / np.array(target_spacing)).astype(int)
    
    # Configure the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize([int(sz) for sz in target_size])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputPixelType(sitk.sitkUInt16)
    
    # Resample the image and return the result
    resampled_image = resampler.Execute(image)

    sitk.WriteImage(resampled_image, output_path)



def apply_manual_alignment(matrix_file_path: str,
                  fixed_volume_path: str,
                  moving_volume_path: str,
                  resample_output_fixed_path: str,
                  resample_output_moving_path: str) -> None:
    """
    Aligns a moving volume to a fixed volume using an affine transformation read from a file.
    
    This function performs the following steps:
      1. Loads a 3×3 transformation matrix from a text file.
      2. Constructs a 4×4 homogeneous transformation matrix by embedding the 3×3 matrix into
         the top-left corner of a 4×4 identity matrix. The translation component (last column)
         is left as zeros.
      3. Reads the fixed and moving image volumes from TIFF files and converts them into SimpleITK images.
      4. Sets up an affine transform using the rotation/scale part from the homogeneous matrix.
      5. Resamples the moving image to align it with the fixed image.
      6. Writes the resampled moving image and the fixed image (as adjusted) to the specified output paths.
    
    Parameters:
    -----------
    matrix_file_path : str
        Path to the text file containing a 3×3 transformation matrix.
    fixed_volume_path : str
        Path to the fixed (reference) image volume (e.g., in vivo slice) in TIFF format.
    moving_volume_path : str
        Path to the moving image volume (e.g., ex vivo slice) in TIFF format.
    resample_output_fixed_path : str
        Output path for writing the fixed image (unchanged) in TIFF format.
    resample_output_moving_path : str
        Output path for writing the resampled (aligned) moving image in TIFF format.
    
    Returns:
    --------
    None
    """
    # --- Step 1: Load and construct the transformation matrix ---
    # Load the 3×3 transformation matrix from file
    matrix = np.loadtxt(matrix_file_path)
    
    # Create a 4×4 identity matrix
    matrix_4x4 = np.eye(4)
    # Replace the top-left 3×3 section with the loaded matrix
    matrix_4x4[:3, :3] = matrix
    print("Constructed 4×4 transformation matrix:")
    print(matrix_4x4)
    
    # --- Step 2: Load images and convert to SimpleITK format ---
    # Load the fixed image volume and convert to a SimpleITK image
    fixed_image_data = tifffile.imread(fixed_volume_path)
    fixed_image = sitk.GetImageFromArray(fixed_image_data)
    
    # Load the moving image volume and convert to a SimpleITK image
    moving_image_data = tifffile.imread(moving_volume_path)
    moving_image = sitk.GetImageFromArray(moving_image_data)
    
    # --- Step 3: Set up the affine transform ---
    # Create a 3D affine transform
    transform = sitk.AffineTransform(3)
    # Use the top-left 3×3 part of the homogeneous matrix for rotation/scale
    transform.SetMatrix(matrix_4x4[:3, :3].flatten())
    # Use the last column (excluding the bottom element) as the translation vector
    transform.SetTranslation(matrix_4x4[:3, 3])
    
    # --- Step 4: Resample the moving image ---
    # Configure the resampler with the fixed image as reference
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    # Here, output spacing is taken from the moving image but can be adjusted if needed
    resampler.SetOutputSpacing(moving_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetSize(fixed_image.GetSize())
    
    # Execute the resampling to align the moving image
    resampled_moving_image = resampler.Execute(moving_image)
    
    # --- Step 5: Write output images ---
    # Convert the resampled moving image back to a NumPy array and write to file
    resampled_moving_array = sitk.GetArrayFromImage(resampled_moving_image)
    tifffile.imwrite(resample_output_moving_path, resampled_moving_array)
    
    # Convert the fixed image to a NumPy array and write to file
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    tifffile.imwrite(resample_output_fixed_path, fixed_array)


def linear_alignment_tuning(fixed_path: str,
                           moving_path: str,
                           fixed_spacing: list,
                           moving_spacing: list,
                           output_matrix_file: str,
                           steps: list = None) -> np.ndarray:
    """
    Executes an alignment pipeline between fixed and moving 3D volumes and saves the resulting
    affine transformation matrix.

    The function performs the following steps:
      1. Reads two 3D TIFF volumes (fixed and moving) from the provided file paths.
      2. Prints the shapes of the input volumes for verification.
      3. Uses a customizable set of alignment steps. If no steps are provided, a default configuration
         is used.
      4. Calls the alignment_pipeline to compute the affine transformation matrix.
      5. Saves the computed affine transformation matrix to a text file.

    Args:
        fixed_path (str): File path to the fixed TIFF volume.
        moving_path (str): File path to the moving TIFF volume.
        fixed_spacing (list or tuple): Original spacing for the fixed volume in [z, y, x] order.
        moving_spacing (list or tuple): Original spacing for the moving volume in [z, y, x] order.
        output_matrix_file (str): File path where the computed affine transformation matrix will be saved.
        steps (list, optional): List of alignment steps. Each step is a tuple with the transformation type
                                (e.g., 'affine') and a dictionary of parameters. If None, a default set of
                                steps is used.

    Returns:
        np.ndarray: The computed affine transformation matrix.

    Raises:
        Exception: If an error occurs during file reading, alignment processing, or file saving.

    """
    try:
        # Read the fixed and moving volumes from TIFF files
        fixed_volume = tifffile.imread(fixed_path)
        moving_volume = tifffile.imread(moving_path)
        
        print(f"Fixed volume shape: {fixed_volume.shape}")
        print(f"Moving volume shape: {moving_volume.shape}")
        
        # If no alignment steps are provided, use default settings
        if steps is None:
            common_kwargs = {
                'alignment_spacing': 1,
                'shrink_factors': (4, 2, 1),
                'smooth_sigmas': (4., 2., 1.),
                'optimizer_args': {
                    'learningRate': 0.25,
                    'minStep': 0.,
                    'numberOfIterations': 2000,
                },
            }
            affine_kwargs = {
                'alignment_spacing': 1,
                'shrink_factors': (2, 1),
                'smooth_sigmas': (0., 0.),
            }
            steps = [('affine', {**common_kwargs, **affine_kwargs})]
                
        # Run the alignment pipeline to compute the affine transformation matrix
        affine_3d = alignment_pipeline(fixed_volume,
                                       moving_volume,
                                       fixed_spacing,
                                       moving_spacing,
                                       steps)
        
        print(f"Computed affine matrix shape: {affine_3d.shape}")
        
        # Save the computed affine transformation matrix to a text file
        np.savetxt(output_matrix_file, affine_3d)
        print(f"Affine transformation matrix saved to {output_matrix_file}")
        
        return affine_3d
    
    except Exception as e:
        print(f"Error during alignment pipeline execution: {e}")
        raise