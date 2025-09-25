import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import zarr
from bigstream.align import alignment_pipeline
from bigstream.piecewise_align import distributed_piecewise_alignment_pipeline
from bigstream.piecewise_transform import distributed_apply_transform
from cellpose.contrib.distributed_segmentation import distributed_eval
from scipy.ndimage import gaussian_filter, zoom


def compute_affine_registration(
    fixed_volume_path: str,
    moving_volume_path: str,
    voxel_spacing: List[float],
    output_matrix_path: str,
    downsample_factors: Tuple[int, int, int] = (4, 7, 7),
    alignment_steps: Optional[List] = None,
) -> np.ndarray:
    """
    Compute initial affine registration between two microscopy volumes.

    Performs coarse alignment on downsampled volumes using feature-based registration
    followed by gradient descent optimization. This replaces the "initial" alignment step.

    Args:
        fixed_volume_path (str): Path to reference volume (TIFF or Zarr)
        moving_volume_path (str): Path to volume to be aligned (TIFF or Zarr)
        voxel_spacing (List[float]): Voxel spacing in [z, y, x] order (microns)
        output_matrix_path (str): Path to save the computed affine transformation matrix
        downsample_factors (Tuple[int, int, int]): Downsampling factors for (z, y, x)
        alignment_steps (Optional[List]): Custom alignment pipeline steps

    Returns:
        np.ndarray: 4x4 affine transformation matrix
    """
    import tifffile
    from scipy.ndimage import zoom

    def downsample_volume(array, factors):
        return zoom(array, (1 / factor for factor in factors), order=1)

    # Load and downsample volumes
    if fixed_volume_path.endswith(".zarr"):
        fixed_volume = zarr.open(fixed_volume_path, mode="r")[:]
    else:
        fixed_volume = tifffile.imread(fixed_volume_path)

    if moving_volume_path.endswith(".zarr"):
        moving_volume = zarr.open(moving_volume_path, mode="r")[:]
    else:
        moving_volume = tifffile.imread(moving_volume_path)

    print(
        f"Original volumes - Fixed: {fixed_volume.shape}, Moving: {moving_volume.shape}"
    )

    # Downsample for efficient processing
    fixed_downsampled = downsample_volume(fixed_volume, downsample_factors)
    moving_downsampled = downsample_volume(moving_volume, downsample_factors)

    print(
        f"Downsampled volumes - Fixed: {fixed_downsampled.shape}, Moving: {moving_downsampled.shape}"
    )

    # Calculate adjusted spacing
    adjusted_spacing = np.array(voxel_spacing) * np.array(downsample_factors)

    # Define default alignment steps if none provided
    if alignment_steps is None:
        # Feature-based initialization
        ransac_kwargs = {"blob_sizes": [2, 100], "use_gpu": True}

        # Gradient descent refinement
        affine_kwargs = {
            "metric": "MMI",
            "optimizer": "OPOE",
            "alignment_spacing": 1,
            "shrink_factors": (4, 2, 1),
            "smooth_sigmas": (1.0, 1.0, 1.0),
            "optimizer_args": {
                "numberOfIterations": 900,
                "epsilon": 1.0e-6,
                "initialRadius": 1.1,
            },
        }

        alignment_steps = [("ransac", ransac_kwargs), ("affine", affine_kwargs)]

    # Execute alignment pipeline
    print("Computing affine registration...")
    affine_matrix = alignment_pipeline(
        fixed_downsampled,
        moving_downsampled,
        adjusted_spacing,
        adjusted_spacing,
        alignment_steps,
    )

    # Save transformation matrix
    np.savetxt(output_matrix_path, affine_matrix)
    print(f"Affine transformation matrix saved to: {output_matrix_path}")

    return affine_matrix


def compute_deformation_field_registration(
    fixed_zarr_path: str,
    moving_zarr_path: str,
    affine_matrix_path: str,
    output_directory: str,
    output_name: str,
    voxel_spacing: List[float],
    block_size: List[int] = [512, 512, 512],
    cluster_config: Optional[Dict] = None,
    use_chunk_alignment: bool = False,
    initial_deformation_field_path: Optional[str] = None,
) -> str:
    """
    Compute dense deformation field registration for precise alignment.

    When use_chunk_alignment=False:
        Applies initial affine transformation then computes a dense deformation field
        for fine-grained alignment. This replaces the "final" alignment step.

    When use_chunk_alignment=True:
        Uses the provided initial deformation field instead of affine matrix to create
        the initial alignment, then continues with deformation field computation.

    Args:
        fixed_zarr_path (str): Path to reference Zarr volume
        moving_zarr_path (str): Path to moving Zarr volume
        affine_matrix_path (str): Path to initial affine transformation matrix
        output_directory (str): Directory to save alignment results
        output_name (str): Base name for output files
        voxel_spacing (List[float]): Voxel spacing in [z, y, x] order (microns)
        block_size (List[int]): Block size for distributed processing
        cluster_config (Optional[Dict]): Dask cluster configuration
        use_chunk_alignment (bool): Whether to use chunk-based alignment mode
        initial_deformation_field_path (Optional[str]): Path to initial deformation field
                                                       (required when use_chunk_alignment=True)

    Returns:
        str: Path to the final aligned volume
    """
    # Load volumes
    fixed_volume = zarr.open(fixed_zarr_path, mode="r")
    moving_volume = zarr.open(moving_zarr_path, mode="r")

    # Default cluster configuration
    if cluster_config is None:
        cluster_config = {
            "cluster_type": "local_cluster",
            "n_workers": 8,
            "threads_per_worker": 1,
            "memory_limit": "150GB",
            "config": {
                "distributed.nanny.pre-spawn-environ": {
                    "MALLOC_TRIM_THRESHOLD_": 65536,
                    "MKL_NUM_THREADS": 10,
                    "OMP_NUM_THREADS": 10,
                    "OPENBLAS_NUM_THREADS": 10,
                },
                "distributed.scheduler.worker-ttl": None,
            },
        }

    # Apply initial transformation (affine matrix or initial deformation field)
    affine_aligned_path = os.path.join(
        output_directory, f"{output_name}_affine_aligned.zarr"
    )

    if use_chunk_alignment:
        # Use initial deformation field instead of affine matrix
        if initial_deformation_field_path is None:
            raise ValueError(
                "initial_deformation_field_path is required when use_chunk_alignment=True"
            )

        if not os.path.exists(initial_deformation_field_path):
            raise FileNotFoundError(
                f"Initial deformation field not found: {initial_deformation_field_path}"
            )

        print("Applying initial deformation field (chunk-based alignment)...")
        initial_deformation_field = zarr.open(initial_deformation_field_path, mode="r")

        distributed_apply_transform(
            fixed_volume,
            moving_volume,
            voxel_spacing,
            voxel_spacing,
            transform_list=[initial_deformation_field],
            blocksize=block_size,
            write_path=affine_aligned_path,
            cluster_kwargs=cluster_config.copy(),
        )

        print("Initial deformation field alignment complete")
    else:
        # Traditional affine transformation
        affine_matrix = np.loadtxt(affine_matrix_path)
        print("Applying initial affine transformation...")

        distributed_apply_transform(
            fixed_volume,
            moving_volume,
            voxel_spacing,
            voxel_spacing,
            transform_list=[affine_matrix],
            blocksize=block_size,
            write_path=affine_aligned_path,
            cluster_kwargs=cluster_config.copy(),
        )

        print("Initial affine alignment complete")

    # Load affine-aligned volume for deformation field computation
    affine_aligned_volume = zarr.open(affine_aligned_path, mode="r")

    # Configure deformation field parameters
    deformation_kwargs = {
        "alignment_spacing": 1,
        "smooth_sigmas": (1,),
        "optimizer": "LBFGSB",
        "control_point_spacing": 1200.0,
        "control_point_levels": (1,),
    }

    deformation_steps = [("deform", deformation_kwargs)]

    # Compute deformation field
    print("Computing deformation field...")
    deformation_field_path = os.path.join(
        output_directory, f"{output_name}_deformation_field.zarr"
    )

    distributed_piecewise_alignment_pipeline(
        fixed_volume,
        affine_aligned_volume,
        voxel_spacing,
        voxel_spacing,
        deformation_steps,
        blocksize=block_size,
        overlap=0.3,
        rebalance_for_missing_neighbors=True,
        write_path=deformation_field_path,
        cluster_kwargs=cluster_config.copy(),
    )

    print("Deformation field computation complete")

    # Apply deformation field to get final alignment
    print("Applying deformation field for final alignment...")
    deformation_field = zarr.open(deformation_field_path, mode="r")
    final_aligned_path = os.path.join(
        output_directory, f"{output_name}_final_aligned.zarr"
    )

    # Increase memory for final transformation
    final_cluster_config = cluster_config.copy()

    # Double the original memory limit
    original_memory = cluster_config["memory_limit"]
    memory_value = int(original_memory.replace("GB", ""))
    final_cluster_config["memory_limit"] = f"{memory_value * 2}GB"

    # Set n_workers to half the original value
    original_workers = cluster_config["n_workers"]
    final_cluster_config["n_workers"] = original_workers // 2

    distributed_apply_transform(
        fixed_volume,
        affine_aligned_volume,
        voxel_spacing,
        voxel_spacing,
        transform_list=[deformation_field],
        blocksize=block_size,
        write_path=final_aligned_path,
        cluster_kwargs=final_cluster_config,
    )

    print(f"Final alignment complete: {final_aligned_path}")
    return final_aligned_path


def distributed_nuclei_segmentation(
    input_zarr_path: str,
    output_zarr_path: str,
    model_type: str = "cpsam",
    block_size: Tuple[int, int, int] = (500, 1024, 1024),
    preprocessing_sigma: float = 2.0,
    segmentation_cluster_config: Optional[Dict] = None,
    temporary_directory: Optional[str] = None,
) -> Tuple[Any, List]:
    """
    Perform distributed nuclei segmentation on large microscopy volumes.

    Uses Cellpose with distributed processing to segment nuclei in 3D volumes,
    typically applied to the 405nm channel for nuclear segmentation.

    Args:
        input_zarr_path (str): Path to input Zarr volume (typically 405nm channel)
        output_zarr_path (str): Path for output segmentation masks
        model_type (str): Cellpose model type ('cpsam', 'nuclei', etc.)
        block_size (Tuple[int, int, int]): Processing block size (z, y, x)
        preprocessing_sigma (float): Gaussian smoothing sigma for preprocessing
        segmentation_cluster_config (Optional[Dict]): Dask cluster configuration for segmentation
        temporary_directory (Optional[str]): Directory for temporary files

    Returns:
        Tuple[Any, List]: Segmentation results and bounding boxes
    """
    # Load input volume
    input_volume = zarr.open(input_zarr_path, mode="r")
    print(f"Input volume shape: {input_volume.shape}")

    # Configure Cellpose model
    model_kwargs = {"gpu": True, "model_type": model_type}

    # Configure evaluation parameters
    eval_kwargs = {
        "z_axis": 0,
        "do_3D": True,
    }

    # Default segmentation cluster configuration for GPU processing
    if segmentation_cluster_config is None:
        segmentation_cluster_config = {
            "n_workers": 3,
            "threads_per_worker": 1,
            "memory_limit": "300GB",
            "use_local_cuda": True,
        }

    # Preprocessing function
    def gaussian_preprocessing(image, sigma, crop):
        return gaussian_filter(image, sigma)

    preprocessing_steps = [(gaussian_preprocessing, {"sigma": preprocessing_sigma})]

    # Set temporary directory
    if temporary_directory is None:
        temporary_directory = os.path.dirname(output_zarr_path)

    print("Starting distributed nuclei segmentation...")

    # Run distributed segmentation
    segments, bounding_boxes = distributed_eval(
        input_zarr=input_volume,
        blocksize=block_size,
        write_path=output_zarr_path,
        preprocessing_steps=preprocessing_steps,
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        cluster_kwargs=segmentation_cluster_config,
        temporary_directory=temporary_directory,
    )

    print(f"Segmentation complete: {output_zarr_path}")
    print(f"Number of detected objects: {len(bounding_boxes)}")

    return segments, bounding_boxes


def apply_deformation_to_channels(
    reference_zarr_path: str,
    channel_zarr_paths: List[str],
    affine_matrix_path: str,
    deformation_field_path: str,
    output_directory: str,
    voxel_spacing: List[float],
    block_size: List[int] = [512, 512, 512],
    cluster_config: Optional[Dict] = None,
    use_chunk_alignment: bool = False,
    initial_deformation_field_path: Optional[str] = None,
) -> List[str]:
    """
    Apply computed transformations to multiple imaging channels.

    When use_chunk_alignment=False (Global Alignment):
        Uses both the affine transformation and deformation field computed from registration
        channels (405nm, 488nm) to align all other imaging channels (epitope markers).
        Transform list: [affine_matrix, deformation_field]

    When use_chunk_alignment=True (Chunk Alignment):
        Uses both the initial deformation field (from chunk alignment) and final deformation
        field to align all other imaging channels. This provides the complete transformation
        pipeline for chunk-based alignment.
        Transform list: [initial_deformation_field, deformation_field]

    Args:
        reference_zarr_path (str): Path to reference volume (fixed)
        channel_zarr_paths (List[str]): Paths to channel volumes to be transformed
        affine_matrix_path (str): Path to computed affine transformation matrix
        deformation_field_path (str): Path to computed deformation field
        output_directory (str): Directory for aligned channel outputs
        voxel_spacing (List[float]): Voxel spacing in [z, y, x] order
        block_size (List[int]): Block size for distributed processing
        cluster_config (Optional[Dict]): Dask cluster configuration
        use_chunk_alignment (bool): Whether to use chunk-based alignment mode
        initial_deformation_field_path (Optional[str]): Path to initial deformation field
                                                       (required when use_chunk_alignment=True)

    Returns:
        List[str]: Paths to aligned channel volumes
    """
    # Load reference volume and deformation field
    reference_volume = zarr.open(reference_zarr_path, mode="r")
    deformation_field = zarr.open(deformation_field_path, mode="r")

    # Load transformation matrices based on alignment mode
    if use_chunk_alignment:
        # Chunk alignment mode: use initial deformation field + final deformation field
        if initial_deformation_field_path is None:
            raise ValueError(
                "initial_deformation_field_path is required when use_chunk_alignment=True"
            )
        
        if not os.path.exists(initial_deformation_field_path):
            raise FileNotFoundError(
                f"Initial deformation field not found: {initial_deformation_field_path}"
            )
        
        print("Using chunk alignment mode: initial_deformation_field + deformation_field")
        initial_deformation_field = zarr.open(initial_deformation_field_path, mode="r")
        transform_list = [initial_deformation_field, deformation_field]
    else:
        # Global alignment mode: use affine matrix + deformation field
        print("Using global alignment mode: affine_matrix + deformation_field")
        affine_matrix = np.loadtxt(affine_matrix_path)
        transform_list = [affine_matrix, deformation_field]

    # Default cluster configuration
    if cluster_config is None:
        cluster_config = {
            "cluster_type": "local_cluster",
            "n_workers": 8,
            "threads_per_worker": 1,
            "memory_limit": "150GB",
            "config": {
                "distributed.nanny.pre-spawn-environ": {
                    "MALLOC_TRIM_THRESHOLD_": 65536,
                    "MKL_NUM_THREADS": 10,
                    "OMP_NUM_THREADS": 10,
                    "OPENBLAS_NUM_THREADS": 10,
                },
                "distributed.scheduler.worker-ttl": None,
            },
        }

    alignment_cluster_config = cluster_config.copy()

    # Double the original memory limit
    original_memory = cluster_config["memory_limit"]
    memory_value = int(original_memory.replace("GB", ""))
    alignment_cluster_config["memory_limit"] = f"{memory_value * 2}GB"

    # Set n_workers to half the original value
    original_workers = cluster_config["n_workers"]
    alignment_cluster_config["n_workers"] = original_workers // 2

    aligned_channel_paths = []

    # Process each channel
    for i, channel_path in enumerate(channel_zarr_paths):
        print(f"Aligning channel {i+1}/{len(channel_zarr_paths)}: {channel_path}")

        # Load channel volume
        channel_volume = zarr.open(channel_path, mode="r")

        # Generate output path
        channel_name = os.path.basename(channel_path).replace(".zarr", "")
        output_path = os.path.join(output_directory, f"{channel_name}_aligned.zarr")

        # Apply transformations based on alignment mode
        distributed_apply_transform(
            reference_volume,
            channel_volume,
            voxel_spacing,
            voxel_spacing,
            transform_list=transform_list,
            blocksize=block_size,
            write_path=output_path,
            cluster_kwargs=alignment_cluster_config.copy(),
        )

        aligned_channel_paths.append(output_path)
        print(f"Channel alignment complete: {output_path}")

    return aligned_channel_paths


def create_registration_summary(
    fixed_path: str,
    moving_path: str,
    affine_matrix_path: str,
    deformation_field_path: str,
    final_aligned_path: str,
    output_summary_path: str,
) -> Dict[str, Any]:
    """
    Create a comprehensive summary of the registration process.

    Generates metadata and quality metrics for the registration workflow,
    useful for tracking and validating alignment results.

    Args:
        fixed_path (str): Path to reference volume
        moving_path (str): Path to moving volume
        affine_matrix_path (str): Path to affine transformation matrix
        deformation_field_path (str): Path to deformation field
        final_aligned_path (str): Path to final aligned volume
        output_summary_path (str): Path to save summary JSON

    Returns:
        Dict[str, Any]: Registration summary dictionary
    """
    import json
    from datetime import datetime

    # Collect volume information
    fixed_vol = zarr.open(fixed_path, mode="r")
    moving_vol = zarr.open(moving_path, mode="r")
    aligned_vol = zarr.open(final_aligned_path, mode="r")

    # Load transformation matrix
    affine_matrix = np.loadtxt(affine_matrix_path)

    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_volumes": {
            "fixed": {
                "path": fixed_path,
                "shape": fixed_vol.shape,
                "dtype": str(fixed_vol.dtype),
            },
            "moving": {
                "path": moving_path,
                "shape": moving_vol.shape,
                "dtype": str(moving_vol.dtype),
            },
        },
        "transformations": {
            "affine_matrix_path": affine_matrix_path,
            "affine_matrix": affine_matrix.tolist(),
            "deformation_field_path": deformation_field_path,
        },
        "output": {
            "aligned_volume_path": final_aligned_path,
            "aligned_shape": aligned_vol.shape,
            "aligned_dtype": str(aligned_vol.dtype),
        },
        "registration_type": "two_stage_affine_deformation",
    }

    # Save summary
    with open(output_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Registration summary saved: {output_summary_path}")
    return summary


def downsample_volume(array: np.ndarray, factors: Tuple[int, int, int]) -> np.ndarray:
    """
    Downsample a volume using zoom with linear interpolation.

    Args:
        array (np.ndarray): Input volume array
        factors (Tuple[int, int, int]): Downsampling factors for (z, y, x)

    Returns:
        np.ndarray: Downsampled volume
    """
    return zoom(array, (1 / factor for factor in factors), order=1)


def upsample_deformation_field_sitk(
    deformation_field_path: str,
    target_shape: Tuple[int, int, int],
    downsample_factors: Tuple[int, int, int],
    voxel_spacing: List[float],
    output_path: str,
    interpolation_method: str = "bspline",
) -> str:
    """
    Upsample a SimpleITK deformation field to match full resolution volumes.

    Parameters:
    -----------
    deformation_field_path : str
        Path to the zarr deformation field computed on downsampled data
    target_shape : tuple
        Target shape for the upsampled deformation field (z, y, x)
    downsample_factors : tuple
        Factors used for downsampling [z, y, x]
    voxel_spacing : list
        Full resolution voxel spacing in microns [z, y, x]
    output_path : str
        Path to save the upsampled deformation field
    interpolation_method : str, optional
        Interpolation method: "linear", "bspline", or "cubic" (default: "bspline")

    Returns:
    --------
    str : Path to the upsampled deformation field
    """
    print(f"Upsampling deformation field from {deformation_field_path}")
    print(f"Target shape: {target_shape}")
    print(f"Downsample factors: {downsample_factors}")
    print(f"Full resolution voxel spacing: {voxel_spacing}")

    # Load the deformation field
    deformation_field = zarr.open(deformation_field_path, mode="r")
    deformation_array = deformation_field[:]

    print(f"Original deformation field shape: {deformation_array.shape}")

    # Convert numpy array to SimpleITK displacement field
    # SimpleITK expects displacement field as a vector image
    sitk_displacement_field = sitk.GetImageFromArray(deformation_array, isVector=True)

    # Set the spacing for the downsampled deformation field
    # SimpleITK uses x,y,z order, so reverse the z,y,x spacing
    downsampled_spacing = np.array(voxel_spacing) * np.array(downsample_factors)
    sitk_spacing = [
        downsampled_spacing[2],
        downsampled_spacing[1],
        downsampled_spacing[0],
    ]
    sitk_displacement_field.SetSpacing(sitk_spacing)

    # Set target spacing (full resolution)
    target_sitk_spacing = [voxel_spacing[2], voxel_spacing[1], voxel_spacing[0]]

    # Calculate target size (x,y,z order for SimpleITK)
    target_size = [target_shape[2], target_shape[1], target_shape[0]]

    print(f"Resampling displacement field:")
    print(f"  From spacing: {sitk_spacing}")
    print(f"  To spacing: {target_sitk_spacing}")
    print(f"  Target size: {target_size}")

    # Choose interpolation method
    interpolation_map = {
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
        "cubic": sitk.sitkBSpline3,  # 3rd order B-spline
    }

    if interpolation_method not in interpolation_map:
        print(
            f"Warning: Unknown interpolation method '{interpolation_method}', using B-spline"
        )
        interpolation_method = "bspline"

    interpolator = interpolation_map[interpolation_method]
    print(f"Using {interpolation_method} interpolation for upsampling")

    # Create resampler for displacement field
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_sitk_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(sitk_displacement_field.GetDirection())
    resampler.SetOutputOrigin(sitk_displacement_field.GetOrigin())
    resampler.SetTransform(sitk.Transform())  # Identity transform
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)

    # Resample the displacement field
    upsampled_displacement_field = resampler.Execute(sitk_displacement_field)

    # Convert back to numpy array
    upsampled_array = sitk.GetArrayFromImage(upsampled_displacement_field)

    print(f"Upsampled deformation field shape: {upsampled_array.shape}")
    print(
        "SimpleITK resampler automatically handles displacement vector scaling based on spacing"
    )

    # Save the upsampled deformation field
    print(f"Saving upsampled deformation field to {output_path}")
    zarr.save(output_path, upsampled_array)

    print("Upsampling complete!")
    return output_path


def compute_chunk_alignment(
    fixed_zarr_path: str,
    moving_zarr_path: str,
    output_directory: str,
    output_name: str,
    voxel_spacing: List[float],
    downsample_factors: Tuple[int, int, int] = (1, 3, 3),
    interpolation_method: str = "linear",
    block_size: List[int] = [512, 512, 512],
    overlap: float = 0.3,
    alignment_kwargs: Optional[Dict] = None,
    cluster_config: Optional[Dict] = None,
) -> str:
    """
    Compute chunk-based alignment for handling non-linear deformation.

    This method performs:
    1. Volume downsampling for efficient processing
    2. Distributed chunk-based deformation field computation
    3. Deformation field upsampling to original resolution

    Note: This function only computes the initial deformation field. The application
    of this deformation field should be done in compute_deformation_field_registration
    when --use-chunk-alignment is active.

    Args:
        fixed_zarr_path (str): Path to reference Zarr volume
        moving_zarr_path (str): Path to moving Zarr volume
        output_directory (str): Directory to save alignment results
        output_name (str): Base name for output files
        voxel_spacing (List[float]): Voxel spacing in [z, y, x] order (microns)
        downsample_factors (Tuple[int, int, int]): Downsampling factors for (z, y, x)
        interpolation_method (str): Interpolation method for upsampling
        block_size (List[int]): Block size for distributed processing
        overlap (float): Overlap fraction for distributed processing
        alignment_kwargs (Optional[Dict]): Alignment parameters for chunk-based processing
        cluster_config (Optional[Dict]): Dask cluster configuration

    Returns:
        str: Path to the initial deformation field
    """
    print("Starting chunk-based alignment...")

    # Validate dependencies
    try:
        import SimpleITK as sitk
        from bigstream.piecewise_align import distributed_piecewise_alignment_pipeline
        from bigstream.piecewise_transform import distributed_apply_transform
    except ImportError as e:
        raise ImportError(
            f"Chunk-based alignment requires additional dependencies: {e}. "
            "Please ensure bigstream and SimpleITK are installed."
        )

    # Validate parameters
    if len(downsample_factors) != 3 or any(f < 1 for f in downsample_factors):
        raise ValueError(
            f"downsample_factors must be a tuple of 3 positive integers, got: {downsample_factors}"
        )

    if interpolation_method not in ["linear", "bspline", "cubic"]:
        print(
            f"Warning: Unknown interpolation method '{interpolation_method}', using 'linear'"
        )
        interpolation_method = "linear"

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Load volumes
    print(f"Loading fixed volume: {fixed_zarr_path}")
    fixed_volume = zarr.open(fixed_zarr_path, mode="r")
    print(f"Fixed volume shape: {fixed_volume.shape}")

    print(f"Loading moving volume: {moving_zarr_path}")
    moving_volume = zarr.open(moving_zarr_path, mode="r")
    print(f"Moving volume shape: {moving_volume.shape}")

    # Store original shapes for upsampling
    original_fixed_shape = fixed_volume.shape
    original_moving_shape = moving_volume.shape

    # Downsample volumes for efficient processing
    print(f"Downsampling volumes with factors: {downsample_factors}")
    fixed_downsampled_path = os.path.join(
        output_directory, f"{output_name}_fixed_downsampled.zarr"
    )
    moving_downsampled_path = os.path.join(
        output_directory, f"{output_name}_moving_downsampled.zarr"
    )

    fixed_downsampled_array = downsample_volume(fixed_volume[:], downsample_factors)
    moving_downsampled_array = downsample_volume(moving_volume[:], downsample_factors)

    zarr.save(fixed_downsampled_path, fixed_downsampled_array)
    zarr.save(moving_downsampled_path, moving_downsampled_array)

    fixed_volume_downsampled = zarr.open(fixed_downsampled_path, mode="r")
    moving_volume_downsampled = zarr.open(moving_downsampled_path, mode="r")

    print(
        f"Downsampled shapes - Fixed: {fixed_volume_downsampled.shape}, Moving: {moving_volume_downsampled.shape}"
    )

    # Default alignment parameters
    if alignment_kwargs is None:
        alignment_kwargs = {"blob_sizes": [2 * 4, 100 * 2], "use_gpu": True}

    # Default cluster configuration
    if cluster_config is None:
        cluster_config = {
            "cluster_type": "local_cluster",
            "n_workers": 1,
            "threads_per_worker": 1,
            "memory_limit": "300GB",
            "config": {
                "distributed.nanny.pre-spawn-environ": {
                    "MALLOC_TRIM_THRESHOLD_": 65536,
                    "MKL_NUM_THREADS": 10,
                    "OMP_NUM_THREADS": 10,
                    "OPENBLAS_NUM_THREADS": 10,
                },
                "distributed.scheduler.worker-ttl": None,
            },
        }

    # Configure alignment steps
    alignment_steps = [("ransac", alignment_kwargs)]

    # Compute deformation field on downsampled data
    print("Computing deformation field on downsampled data...")
    deformation_field_path = os.path.join(
        output_directory, f"{output_name}_deformation_field_downsampled.zarr"
    )

    # Calculate adjusted voxel spacing for downsampled data
    adjusted_voxel_spacing = np.array(voxel_spacing) * np.array(downsample_factors)

    distributed_piecewise_alignment_pipeline(
        fixed_volume_downsampled,
        moving_volume_downsampled,
        adjusted_voxel_spacing,
        adjusted_voxel_spacing,
        alignment_steps,
        blocksize=block_size,
        overlap=overlap,
        rebalance_for_missing_neighbors=True,
        write_path=deformation_field_path,
        cluster_kwargs=cluster_config.copy(),
    )

    print("Deformation field computation complete")

    # Upsample deformation field to match full resolution
    print("Upsampling deformation field to full resolution...")
    initial_deformation_field_path = os.path.join(
        output_directory, f"{output_name}_initial_deformation_field.zarr"
    )

    upsample_deformation_field_sitk(
        deformation_field_path,
        original_fixed_shape,
        downsample_factors,
        voxel_spacing,
        initial_deformation_field_path,
        interpolation_method,
    )

    print(
        f"Chunk-based initial deformation field computation complete: {initial_deformation_field_path}"
    )
    return initial_deformation_field_path
