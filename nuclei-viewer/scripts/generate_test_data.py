#!/usr/bin/env python3
"""
Generate synthetic test data for the Nuclei Viewer application.

This script creates fake Zarr datasets that mimic real microscopy data:
- Segmentation masks with labeled nuclei
- DAPI channel data
- Multiple epitope tag channels
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from skimage.morphology import binary_dilation, disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_nuclei(shape, num_nuclei=50, nucleus_size_range=(5, 15)):
    """
    Create synthetic nuclei segmentation mask.

    Args:
        shape: (Z, Y, X) shape of the volume
        num_nuclei: Number of nuclei to generate
        nucleus_size_range: (min, max) radius range for nuclei

    Returns:
        Labeled segmentation mask
    """
    z_size, y_size, x_size = shape
    mask = np.zeros(shape, dtype=np.uint16)

    logger.info(f"Generating {num_nuclei} synthetic nuclei...")

    for i in range(num_nuclei):
        # Random position (avoid edges)
        z_center = np.random.randint(5, z_size - 5)
        y_center = np.random.randint(20, y_size - 20)
        x_center = np.random.randint(20, x_size - 20)

        # Random size
        radius = np.random.randint(*nucleus_size_range)

        # Create ellipsoid nucleus
        z_radius = max(2, radius // 2)  # Flatter in Z

        # Create coordinate grids
        z_coords, y_coords, x_coords = np.ogrid[:z_size, :y_size, :x_size]

        # Ellipsoid equation
        ellipsoid = (
            ((z_coords - z_center) / z_radius) ** 2
            + ((y_coords - y_center) / radius) ** 2
            + ((x_coords - x_center) / radius) ** 2
        ) <= 1

        # Add some randomness to shape
        noise = np.random.random(shape) > 0.7
        ellipsoid = ellipsoid & ~noise

        # Assign unique label (i+1 since 0 is background)
        mask[ellipsoid] = i + 1

    logger.info(f"Created segmentation mask with {np.max(mask)} nuclei")
    return mask


def create_dapi_channel(segmentation_mask, noise_level=0.1):
    """
    Create synthetic DAPI channel data based on segmentation mask.

    Args:
        segmentation_mask: Labeled segmentation mask
        noise_level: Amount of background noise

    Returns:
        DAPI channel data
    """
    logger.info("Generating DAPI channel data...")

    # Start with segmentation mask
    dapi = (segmentation_mask > 0).astype(np.float32)

    # Add intensity variation within nuclei
    for label_id in range(1, np.max(segmentation_mask) + 1):
        nucleus_mask = segmentation_mask == label_id
        if np.any(nucleus_mask):
            # Random intensity for this nucleus
            base_intensity = np.random.uniform(8000, 16000)
            intensity_variation = np.random.uniform(0.7, 1.3, nucleus_mask.shape)
            dapi[nucleus_mask] = base_intensity * intensity_variation[nucleus_mask]

    # Add background noise
    background_noise = np.random.uniform(0, 2000, dapi.shape)
    dapi += background_noise * noise_level

    # Smooth the data
    dapi = gaussian_filter(dapi, sigma=0.5)

    # Convert to uint16
    dapi = np.clip(dapi, 0, 65535).astype(np.uint16)

    logger.info(
        f"DAPI channel: min={dapi.min()}, max={dapi.max()}, mean={dapi.mean():.1f}"
    )
    return dapi


def create_epitope_channel(
    segmentation_mask, positive_probability=0.3, signal_strength=0.8
):
    """
    Create synthetic epitope tag channel.

    Args:
        segmentation_mask: Labeled segmentation mask
        positive_probability: Probability that a nucleus is positive for this epitope
        signal_strength: Relative strength of the signal

    Returns:
        Epitope channel data
    """
    epitope = np.zeros_like(segmentation_mask, dtype=np.float32)

    # Determine which nuclei are positive
    max_label = np.max(segmentation_mask)
    positive_nuclei = np.random.random(max_label) < positive_probability

    positive_count = 0
    for label_id in range(1, max_label + 1):
        if positive_nuclei[label_id - 1]:  # -1 because array is 0-indexed
            nucleus_mask = segmentation_mask == label_id
            if np.any(nucleus_mask):
                # Create signal around nucleus (not just inside)
                # Create 3D structuring element for dilation
                struct_elem = np.ones((1, 7, 7))  # Only dilate in XY plane
                dilated_mask = binary_dilation(nucleus_mask, struct_elem)

                # Signal intensity varies
                base_intensity = np.random.uniform(6000, 14000) * signal_strength
                intensity_variation = np.random.uniform(0.5, 1.5, dilated_mask.shape)
                epitope[dilated_mask] = (
                    base_intensity * intensity_variation[dilated_mask]
                )
                positive_count += 1

    # Add background noise
    background_noise = np.random.uniform(0, 1500, epitope.shape)
    epitope += background_noise * 0.2

    # Smooth the data
    epitope = gaussian_filter(epitope, sigma=0.7)

    # Convert to uint16
    epitope = np.clip(epitope, 0, 65535).astype(np.uint16)

    logger.info(
        f"Epitope channel: {positive_count}/{max_label} positive nuclei, "
        f"min={epitope.min()}, max={epitope.max()}, mean={epitope.mean():.1f}"
    )
    return epitope


def save_zarr_dataset(data, output_path, array_key=None, compression="zstd"):
    """
    Save data as Zarr format.

    Args:
        data: NumPy array to save
        output_path: Path to save Zarr file
        array_key: Key name for the array (None for direct array)
        compression: Compression algorithm
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if array_key:
        # Save as group with key
        store = zarr.open(output_path, mode="w")
        store.create_dataset(
            array_key,
            data=data,
            chunks=(
                min(10, data.shape[0]),
                min(64, data.shape[1]),
                min(64, data.shape[2]),
            ),
            compression=compression,
        )
        logger.info(f"Saved {array_key} to {output_path}")
    else:
        # Save as direct array
        zarr.save(output_path, data)
        logger.info(f"Saved direct array to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for Nuclei Viewer"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data",
        help="Output directory for test data",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=[20, 256, 256],
        help="Volume shape (Z Y X)",
    )
    parser.add_argument(
        "--num-nuclei", type=int, default=50, help="Number of nuclei to generate"
    )
    parser.add_argument(
        "--epitope-channels",
        type=str,
        nargs="+",
        default=["488nm", "561nm", "640nm"],
        help="Names of epitope channels to generate",
    )
    parser.add_argument(
        "--use-groups",
        action="store_true",
        help="Save as Zarr groups with keys (instead of direct arrays)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating test data in {output_dir}")
    logger.info(f"Volume shape: {args.shape}")
    logger.info(f"Number of nuclei: {args.num_nuclei}")
    logger.info(f"Epitope channels: {args.epitope_channels}")

    # Generate segmentation mask
    segmentation = create_synthetic_nuclei(
        shape=tuple(args.shape), num_nuclei=args.num_nuclei
    )

    # Generate DAPI channel
    dapi = create_dapi_channel(segmentation)

    # Generate epitope channels
    epitope_data = {}
    for channel_name in args.epitope_channels:
        # Vary the probability and strength for different channels
        prob = np.random.uniform(0.2, 0.5)
        strength = np.random.uniform(0.6, 1.0)
        epitope_data[channel_name] = create_epitope_channel(
            segmentation, positive_probability=prob, signal_strength=strength
        )

    # Save datasets
    if args.use_groups:
        # Save as groups with keys
        save_zarr_dataset(
            segmentation, output_dir / "segmentation_labels.zarr", "labels"
        )
        save_zarr_dataset(dapi, output_dir / "dapi_channel.zarr", "raw")

        for channel_name, data in epitope_data.items():
            save_zarr_dataset(data, output_dir / f"{channel_name}_channel.zarr", "raw")
    else:
        # Save as direct arrays
        save_zarr_dataset(segmentation, output_dir / "segmentation_labels.zarr")
        save_zarr_dataset(dapi, output_dir / "dapi_channel.zarr")

        for channel_name, data in epitope_data.items():
            save_zarr_dataset(data, output_dir / f"{channel_name}_channel.zarr")

    # Create a summary file
    summary = {
        "shape": args.shape,
        "num_nuclei": args.num_nuclei,
        "actual_nuclei": int(np.max(segmentation)),
        "epitope_channels": args.epitope_channels,
        "use_groups": args.use_groups,
        "files": {
            "segmentation": "segmentation_labels.zarr",
            "dapi": "dapi_channel.zarr",
            "epitope_channels": {
                name: f"{name}_channel.zarr" for name in args.epitope_channels
            },
        },
    }

    import json

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Test data generation complete!")
    logger.info(f"Generated {np.max(segmentation)} nuclei")
    logger.info(f"Files saved in: {output_dir}")

    # Print configuration for the app
    print("\n" + "=" * 50)
    print("CONFIGURATION FOR NUCLEI VIEWER:")
    print("=" * 50)
    print(f"Base path: {output_dir.absolute()}")
    print("\nUpdate your config/default_config.yaml:")
    print(f"data:")
    print(f'  base_path: "{output_dir.absolute()}"')
    print(f"  segmentation:")
    print(f'    file_pattern: "*_labels.zarr"')
    array_key_seg = '"labels"' if args.use_groups else "null"
    print(f"    array_key: {array_key_seg}")
    print(f"  dapi_channel:")
    print(f'    file_pattern: "dapi_*.zarr"')
    array_key_raw = '"raw"' if args.use_groups else "null"
    print(f"    array_key: {array_key_raw}")
    print(f"  epitope_channels:")
    for i, channel in enumerate(args.epitope_channels):
        colors = ["#00ff00", "#ff0000", "#0000ff", "#ffff00", "#ff00ff"]
        color = colors[i % len(colors)]
        print(f'    - name: "{channel}"')
        print(f'      file_pattern: "{channel}_*.zarr"')
        print(f"      array_key: {array_key_raw}")
        print(f'      default_color: "{color}"')


if __name__ == "__main__":
    main()
