# VolAlign

VolAlign is a work-in-progress Python package for volumetric image alignment, stitching, and processing.

## Features

- Create BDV/XML files from FOV TIFF images.
- Stitch tiles using Fiji in headless mode.
- Blend image tiles into composite volumes.
- Resample 3D volumes.
- Apply manual affine alignments.
- Tune alignment with customizable pipelines.
- Convert between TIFF and Zarr formats.
- Downsample and stack TIFF images.
- Reorient volumes with rotation and optional flipping.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/XpressAI/VolAlign
cd volalign
pip install -r requirements.txt
