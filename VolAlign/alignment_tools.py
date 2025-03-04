import os
import subprocess
import tempfile
import numpy as np
import tifffile
import SimpleITK as sitk
from npy2bdv import BdvWriter
from bigstream.align import alignment_pipeline
from tqdm import tqdm
from typing import List, Optional
from exm.stitching.tileset import Tileset
from VolAlign.utils import blend_ind


def create_bdv_xml(output_h5_file: str,
                   fov_list: list,
                   offset_array: np.ndarray,
                   nchannels: int = 2,
                   increment_scale: float = 1900.0) -> str:
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
    ensuring that each FOV is clearly positioned according to its raw column/row location in the final volume.
    It then writes the BDV/XML file using the npy2bdv library.
    
    Args:
        output_h5_file (str): Full path to the output H5 file where the BDV/XML file and associated tile data will be saved.
        fov_list (list of str): List of file paths to the FOV TIFF images.
        offset_array (np.ndarray): Array of shape (n, 3) containing the offsets [column, row, z] for each FOV.
        nchannels (int, optional): Number of channels in each FOV image. Default is 2.
        increment_scale (float, optional): Scale factor used to compute increments for unique offset values. Default is 1900.0.
    
    Returns:
        str: The path to the created BDV/XML file.
    
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
        print(adjusted_offsets, flush=True)
        
        # Initialize the BDV writer with the specified number of FOVs, channels, and voxel size.
        bdv_writer = BdvWriter(output_h5_file, ntiles=len(fov_list), nchannels=nchannels, overwrite=True)
        
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
        print(f"BDV/XML file successfully created in {output_h5_file}")
        xml_path, _ = os.path.splitext(output_h5_file)
        return xml_path + ".xml"
    
    except Exception as e:
        print(f"Error during BDV/XML creation: {e}")
        raise


def stitch_tiles(xml_file_path: str, fiji_path: str) -> None:
    """
    Executes a tile stitching pipeline by generating an ImageJ macro and running Fiji in headless mode.
    
    This function constructs an ImageJ macro that performs three sequential operations on the 
    specified XML file:
      1. Calculate pairwise shifts using phase correlation.
      2. Filter pairwise shifts based on specified criteria.
      3. Optimize the global alignment and apply the calculated shifts.
      
    The macro is written to a temporary file which is then executed by Fiji using a headless
    mode command-line call. The temporary macro file is automatically cleaned up.
    
    Args:
        xml_file_path (str): The path to the BDV/XML file that contains the tile configuration.
        fiji_path (str): The full path to the Fiji executable.
    
    Returns:
        None
    
    Example:
        >>> stitch_tiles("/path/to/tiles.xml", "/path/to/Fiji.app/ImageJ-linux64")
    """
    # Construct the ImageJ macro content.
    macro_content = f'''
    run("Calculate pairwise shifts ...", "select=[{xml_file_path}] process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] method=[Phase Correlation] show_expert_grouping_options show_expert_algorithm_parameters how_to_treat_timepoints=[treat individually] how_to_treat_channels=group how_to_treat_illuminations=group how_to_treat_angles=[treat individually] how_to_treat_tiles=compare channels=[Average Channels] downsample_in_x=1 downsample_in_y=1 downsample_in_z=1 number_of_peaks_to_check=75 minimal_overlap=5 subpixel_accuracy");
    run("Filter pairwise shifts ...", "select=[{xml_file_path}] min_r=0 max_r=1 max_shift_in_x=0 max_shift_in_y=0 max_shift_in_z=0 max_displacement=0");
    run("Optimize globally and apply shifts ...", "select=[{xml_file_path}] process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.500 absolute=3.500 global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] fix_group_0-0,");
    eval("script", "System.exit(0);");
    '''.strip()
    
    # Write the macro to a temporary file.
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.ijm', delete=True) as tmp:
        tmp.write(macro_content)
        tmp.flush()  # Ensure the macro content is written to disk.
        
        # Build the command to run Fiji in headless mode with the macro.
        cmd = [
            fiji_path,
            '--headless',
            '--console',
            '-macro',
            tmp.name
        ]
        
        print("Executing command:", " ".join(cmd))
        # Execute the command.
        os.system(" ".join(cmd))


def blend_tiles(xml_file: str,
                output_folder: str,
                voxel_size: List[float],
                channels: List[str]) -> List[str]:
    """
    Blends stitched tile images for each channel and saves a TIFF file for each channel.
    
    The function performs the following steps:
      1. Loads a tileset from a BDV/XML file (generated by BigStitcher) using the provided voxel size.
      2. Groups tiles into chunks of equal size. (Currently, the group size is computed as total number of tiles divided by 2.)
      3. For each group, computes offsets (converted to pixel coordinates by dividing by the voxel size)
         and collects the corresponding image tiles.
      4. Blends the collected tiles using `blend_ind` to generate a single volume per channel.
      5. Saves the blended volume as a TIFF file named "stitched_{channel}.tif" in the specified output folder.
    
    Args:
        xml_file (str): Path to the BDV/XML file generated by BigStitcher.
        output_folder (str): Folder where the output TIFF files will be saved.
        voxel_size (List[float]): Voxel size as a list in [z, y, x] order.
        channels (List[str]): List of channel names (e.g., ['CH1', 'CH2']).
    
    Returns:
        List[str]: List of file paths for the blended TIFF files for each channel.
    """
    # Initialize the tileset from the BDV/XML file.
    ts = Tileset(voxel_size)
    ts.init_from_bdv(xml_file)
    total_tiles = len(ts[:])
    group_size = int(total_tiles / 2)
    output_paths = []
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
        output_paths.append(output_path)

        print(f"Saved blended volume for {channel_name} with shape {stitched_volume.shape} to {output_path}")

    return output_paths


def voxel_spacing_resample(input_file: str,
                           output_path: str,
                           original_spacing: list,
                           target_spacing: list,
                           interpolator=sitk.sitkLinear) -> sitk.Image:
    """
    Loads a 3D .tif image, sets its original spacing, and resamples it to a target spacing.
    
    This function performs the following steps:
      1. Reads the image from the provided .tif file using tifffile.
      2. Converts the NumPy array to a SimpleITK image and casts it to uint16.
      3. Sets the image spacing based on the provided original spacing.
      4. Computes the new image size required for the target spacing.
      5. Uses a SimpleITK resampler to adjust the image to the desired target spacing.
      6. Writes the resampled image to the specified output path.
    
    Args:
        input_file (str): Path to the 3D .tif image file.
        output_path (str): Path where the resampled 3D .tif image will be saved.
        original_spacing (list or tuple): Original spacing in [x, y, z] order.
        target_spacing (list or tuple): Desired target spacing in [x, y, z] order.
        interpolator (SimpleITK interpolator, optional): Interpolation method (default: sitk.sitkLinear).
    
    Returns:
        sitk.Image: The resampled SimpleITK image.
    """
    # Convert the NumPy array to a SimpleITK image and cast to uint16.
    image = sitk.GetImageFromArray(tifffile.imread(input_file))
    image = sitk.Cast(image, sitk.sitkUInt16)
    
    # Set the original spacing.
    image.SetSpacing(original_spacing)
    
    # Calculate the new image size based on target spacing.
    original_size = np.array(image.GetSize(), dtype=np.int16)
    target_size = np.round(original_size * np.array(original_spacing) / np.array(target_spacing)).astype(int)
    
    # Configure the resampler.
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize([int(sz) for sz in target_size])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputPixelType(sitk.sitkUInt16)
    
    # Resample the image.
    resampled_image = resampler.Execute(image)

    # Write the resampled image to disk.
    sitk.WriteImage(resampled_image, output_path)
    print("Original size (XYZ):", image.GetSize())
    print("Original spacing (XYZ):", image.GetSpacing())
    print("Resampled size (XYZ):", resampled_image.GetSize())
    print("Resampled spacing (XYZ):", resampled_image.GetSpacing())

    return resampled_image


def apply_manual_alignment(matrix_file_path: str,
                           fixed_volume_path: str,
                           moving_volume_path: str,
                           resample_output_fixed_path: str,
                           resample_output_moving_path: str) -> None:
    """
    Aligns a moving volume to a fixed volume using an affine transformation read from a file.
    
    This function performs the following steps:
      1. Loads a 3x4 transformation matrix from a text file.
      2. Constructs a 4x4 homogeneous transformation matrix by embedding the 3x4 matrix into the top three rows
         of a 4x4 identity matrix, thereby preserving both the rotation/scale and translation components.
      3. Reads the fixed and moving image volumes from TIFF files and converts them into SimpleITK images.
      4. Sets up an affine transform using the rotation/scale (from the top-left 3x3 block) and translation (last column).
      5. Resamples the moving image to align it with the fixed image.
      6. Writes the resampled moving image and the fixed image (unchanged) to the specified output paths.
    
    Args:
        matrix_file_path (str): Path to the text file containing a 3x4 transformation matrix.
        fixed_volume_path (str): Path to the fixed (reference) image volume in TIFF format.
        moving_volume_path (str): Path to the moving image volume in TIFF format.
        resample_output_fixed_path (str): Output path for writing the fixed image in TIFF format.
        resample_output_moving_path (str): Output path for writing the resampled (aligned) moving image in TIFF format.
    
    Returns:
        None
    """
    # --- Step 1: Load and construct the transformation matrix ---
    matrix = np.loadtxt(matrix_file_path)
    matrix_4x4 = np.eye(4)
    matrix_4x4[:3, :] = matrix

    print("Constructed 4x4 transformation matrix:")
    print(matrix_4x4)
    
    # --- Step 2: Load images and convert to SimpleITK format ---
    fixed_image_data = tifffile.imread(fixed_volume_path)
    fixed_image = sitk.GetImageFromArray(fixed_image_data)
    
    moving_image_data = tifffile.imread(moving_volume_path)
    moving_image = sitk.GetImageFromArray(moving_image_data)
    
    # --- Step 3: Set up the affine transform ---
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix_4x4[:3, :3].flatten())
    transform.SetTranslation(matrix_4x4[:3, 3])
    
    # --- Step 4: Resample the moving image ---
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    resampler.SetOutputSpacing(moving_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetSize(fixed_image.GetSize())
    
    resampled_moving_image = resampler.Execute(moving_image)
    
    # --- Step 5: Write output images ---
    resampled_moving_array = sitk.GetArrayFromImage(resampled_moving_image)
    tifffile.imwrite(resample_output_moving_path, resampled_moving_array)
    
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
      3. Uses a customizable set of alignment steps. If no steps are provided, a default configuration is used.
      4. Calls the alignment_pipeline to compute the affine transformation matrix.
      5. Saves the computed affine transformation matrix to a text file.
    
    Args:
        fixed_path (str): File path to the fixed TIFF volume.
        moving_path (str): File path to the moving TIFF volume.
        fixed_spacing (list or tuple): Original spacing for the fixed volume in [z, y, x] order.
        moving_spacing (list or tuple): Original spacing for the moving volume in [z, y, x] order.
        output_matrix_file (str): File path where the computed affine transformation matrix will be saved.
        steps (list, optional): List of alignment steps. Each step is a tuple with the transformation type
                                (e.g., 'affine') and a dictionary of parameters. If None, a default set of steps is used.
    
    Returns:
        np.ndarray: The computed affine transformation matrix.
    
    Raises:
        Exception: If an error occurs during file reading, alignment processing, or file saving.
    """
    try:
        fixed_volume = tifffile.imread(fixed_path)
        moving_volume = tifffile.imread(moving_path)
        
        print(f"Fixed volume shape: {fixed_volume.shape}")
        print(f"Moving volume shape: {moving_volume.shape}")
        
        # Use default alignment steps if none are provided.
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
        
        affine_3d = alignment_pipeline(fixed_volume,
                                       moving_volume,
                                       fixed_spacing,
                                       moving_spacing,
                                       steps)
        
        
        np.savetxt(output_matrix_file, affine_3d)
        print(f"Affine transformation matrix saved to {output_matrix_file}")
        
        return affine_3d
    
    except Exception as e:
        print(f"Error during alignment pipeline execution: {e}")
        raise
