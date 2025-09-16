# Nuclei Viewer

A web application for visualizing segmented nuclei and their epitope tags from 3D microscopy data. Built with FastAPI backend and React frontend, designed to handle large Zarr datasets with real-time maximum intensity projection (MIP) generation.

## Features

- **Interactive Nuclei Browser**: Paginated view of segmented nuclei with thumbnail previews
- **Multi-Channel Visualization**: Support for DAPI and multiple epitope tag channels
- **Real-Time MIP Generation**: On-the-fly maximum intensity projections from 3D data
- **Dynamic Channel Controls**: Adjust colors, opacity, and contrast for each channel
- **Detailed Nucleus View**: Zoom into individual nuclei with comprehensive metadata
- **3D Bounding Box Support**: Configurable XY padding around nuclei regions
- **Efficient Data Handling**: Memory-optimized processing of large Zarr volumes

## Architecture

```
nuclei-viewer/
├── backend/           # FastAPI backend server
│   ├── app/
│   │   ├── api/       # REST API endpoints
│   │   ├── core/      # Core processing modules
│   │   └── main.py    # FastAPI application
│   ├── requirements.txt
│   └── run.py         # Server startup script
├── frontend/          # React frontend application
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── services/    # API communication
│   │   └── App.js       # Main application
│   └── package.json
├── config/            # Configuration files
├── scripts/           # Setup and testing scripts
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Node.js 16 or higher
- npm or yarn

### Option 1: Automated Setup (Recommended)

```bash
# Clone or navigate to the nuclei-viewer directory
cd nuclei-viewer

# Run the automated setup script
python scripts/setup_and_test.py --start-server
```

This will:
1. Generate synthetic test data
2. Set up backend environment
3. Install frontend dependencies
4. Start the backend server
5. Provide instructions for starting the frontend

### Option 2: Manual Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

#### Generate Test Data

```bash
cd scripts

# Generate synthetic test data
python generate_test_data.py --output-dir ../test_data
```

## Configuration

The application uses YAML configuration files located in the `config/` directory.

### Basic Configuration

```yaml
data:
  base_path: "/path/to/your/zarr/files"
  
  segmentation:
    file_pattern: "*_labels.zarr"
    array_key: null  # or "labels" for zarr groups
  
  dapi_channel:
    file_pattern: "*_dapi.zarr"
    array_key: null  # or "raw" for zarr groups
  
  epitope_channels:
    - name: "488nm"
      file_pattern: "*_488.zarr"
      array_key: null
      default_color: "#00ff00"
    - name: "561nm"
      file_pattern: "*_561.zarr"
      array_key: null
      default_color: "#ff0000"

processing:
  min_object_size: 100    # Minimum nucleus size in voxels
  pad_xy: 25             # XY padding around bounding boxes
  max_objects_per_page: 10  # Nuclei per page
  cache_mips: true       # Enable MIP caching
```

## Data Format

The application expects Zarr files containing:

1. **Segmentation Masks**: Labeled nuclei (uint16, each nucleus has unique label)
2. **DAPI Channel**: Nuclear staining data (uint16)
3. **Epitope Channels**: Signal channels for different epitope tags (uint16)

### Supported Zarr Formats

- **Direct Arrays**: Single zarr array per file
- **Zarr Groups**: Multiple arrays within a single zarr file with keys

## Usage

### 1. Data Loading

1. Open the application at `http://localhost:3000`
2. Enter the path to your Zarr files
3. Click "Discover Files" to scan for compatible data
4. Click "Load Datasets" to load the data into memory

### 2. Browsing Nuclei

- Use pagination controls to navigate through nuclei
- Adjust the number of nuclei per page (5, 10, 20, 50)
- Click on any nucleus thumbnail to view details

### 3. Channel Controls

- Toggle channels on/off using the switches
- Adjust opacity with sliders
- Change colors using the color picker
- Access advanced settings for contrast adjustment

### 4. Detailed View

- Click on a nucleus to open the detailed view
- Switch between composite and individual channel views
- View metadata and statistics
- Download individual or composite images

## API Documentation

When the backend server is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

- `GET /api/data/discover` - Discover available datasets
- `POST /api/data/load-all` - Load all datasets
- `GET /api/nuclei/list` - Get paginated nuclei list
- `POST /api/nuclei/mip` - Compute MIP for a nucleus
- `GET /api/config/` - Get/update configuration

## Development

### Backend Development

```bash
cd backend

# Install in development mode
pip install -e .

# Run with auto-reload
python run.py --debug --reload

# Run tests
python -m pytest tests/
```

### Frontend Development

```bash
cd frontend

# Start development server with hot reload
npm start

# Build for production
npm run build

# Run tests
npm test
```

### Adding New Features

1. **Backend**: Add new endpoints in `backend/app/api/`
2. **Frontend**: Add new components in `frontend/src/components/`
3. **Configuration**: Update config schema in `backend/app/core/config.py`

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check Python version (3.10+ required)
   - Verify all dependencies are installed
   - Check port 8000 is not in use

2. **Frontend can't connect to backend**
   - Ensure backend server is running on port 8000
   - Check CORS settings in configuration
   - Verify proxy setting in `frontend/package.json`

3. **Data loading fails**
   - Verify Zarr file paths and patterns
   - Check file permissions
   - Ensure Zarr files are valid and readable

4. **Memory issues with large datasets**
   - Reduce `mip_chunk_size` in configuration
   - Disable MIP caching for very large datasets
   - Consider processing smaller regions

### Performance Optimization

- **Large Datasets**: Adjust chunk sizes and enable caching
- **Many Channels**: Process channels selectively
- **Slow Rendering**: Reduce image resolution or use progressive loading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of the VolAlign microscopy processing pipeline
- Uses Zarr for efficient array storage
- Leverages Dask for distributed computing
- UI components from Material-UI

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Check the browser console for frontend errors
4. Review backend logs for server errors

---

**Note**: This application is designed for research use and handles large microscopy datasets. Ensure adequate system resources (RAM, storage) for your data size.