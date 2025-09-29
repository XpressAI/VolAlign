# Nuclei Viewer - User Guide

Nuclei Viewer is a web application for visualizing segmented nuclei and their epitope tags from 3D microscopy data. Built with a **FastAPI backend** and **React frontend**, it supports large Zarr datasets with real-time maximum intensity projections (MIPs) and integrates seamlessly with the **VolAlign pipeline**.

---

## ✨ Features

### Core Features
- **Interactive Nuclei Browser**: Paginated exploration of segmented nuclei with thumbnails.
- **Multi-Channel Visualization**: Supports DAPI and multiple epitope tag channels.
- **Real-Time MIP Generation**: On-demand intensity projections from 3D data.
- **Channel Controls**: Adjust colors, opacity, and contrast independently.
- **Detailed Nucleus View**: Zoom into single nuclei with epitope details and metadata.
- **Segmentation Bounding Box**: Customizable XY padding.
- **Optimized Data Handling**: Memory-efficient processing of large Zarr volumes.

### Pipeline Integration Features
- **Direct VolAlign Integration**: Load pipeline outputs and analysis directly.
- **Pre-computed Epitope Analysis**: Fast loading of JSON results with cutoffs.
- **Epitope Calls & Confidence Scores**: Visualize positive/negative calls with thresholds.
- **Quality Metrics**: Display per-nucleus quality and cutoff statistics.
- **Performance Optimizations**: Eliminates on-the-fly computation for large datasets.

---

## 📂 Architecture

```
nuclei-viewer/
├── backend/           # FastAPI server logic
│   ├── app/           # API, core modules
│   └── run.py         # Backend entrypoint
├── frontend/          # React application
│   └── src/           # Components, services
├── config/            # Example configs
└── scripts/           # Setup and validation scripts
```

---

## 🚀 Quick Start

### Prerequisites
- Python **3.10+**
- Node.js **16+** (npm or yarn)
- VolAlign pipeline outputs (optional, for pipeline mode)

### Option 1 (Recommended): **Pipeline Integration Mode**
```bash
# Run VolAlign pipeline with epitope analysis
cd /path/to/VolAlign
python examples/selective_step_pipeline.py --step all --config config_template.yaml
python examples/production_epitope_analysis.py --config config_template.yaml

# Start nuclei-viewer with pipeline config
cd nuclei-viewer
python start_with_pipeline.py --config ../config_template.yaml
```
This will:
- Detect pipeline outputs (`zarr_volumes`, `segmentation`, `epitope_analysis`)
- Load precomputed statistics and results
- Activate full epitope visualization mode

### Option 2: **Automated Setup Script**
```bash
cd nuclei-viewer
python scripts/setup_and_test.py --start-server
```

### Option 3: **Manual Setup**
#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py --config ../config/pipeline_config_example.yaml
```

#### Frontend
```bash
cd frontend
npm install
npm start
```

---

## ⚙️ Configuration

### Loading Order
1. **Main VolAlign Config** (`../config_template.yaml`) – Recommended.
2. **Local Config** (`nuclei-viewer/config/pipeline_config_example.yaml`).
3. **Defaults** (synthetic test data).

### Example Pipeline Config Section
```yaml
nuclei_viewer:
  enabled: true
  server:
    host: "127.0.0.1"
    port: 8000
    debug: true
    cors_origins: ["http://localhost:3000"]

  data:
    pipeline:
      pipeline_working_directory: null # inherits from main config
      reference_round: null             # inherits from data.reference_round
      epitope_analysis_file: "epitope_analysis/nucleus_centric_analysis_results.json"
      auto_discover_channels: true
```

### Manual Config Example
```yaml
nuclei_viewer:
  enabled: true
  data:
    mode: "manual"
    base_path: "/path/to/zarr/files"
    segmentation:
      file_pattern: "*_labels.zarr"
    dapi_channel:
      file_pattern: "*_dapi.zarr"
    epitope_channels:
      - name: "488nm"
        file_pattern: "*_488.zarr"
        default_color: "#00ff00"
```

---

## 📊 Data Structures

### Pipeline Mode
```
working_directory/
├── zarr_volumes/      
├── aligned/
├── segmentation/
└── epitope_analysis/
    ├── nucleus_centric_analysis_results.json
    └── nucleus_centric_analysis_summary.json
```

### Manual Mode
- Segmentation masks (*.zarr)
- DAPI reference channel (*.zarr)
- Epitope channels (*.zarr)

---

## 🖥️ Usage

### Pipeline Mode
1. Start backend + frontend servers.
2. Visit `http://localhost:3000`.
3. Browse nuclei with epitope calls, confidence scores, metadata.

### Manual Mode
1. Provide file paths in config.
2. Use viewer to explore segmentation and intensity channels only.

### Common Features
- Paginate nuclei browser.
- Change channel visibility/colors.
- Export images of nuclei/overlays.

---

## 🔌 API Endpoints
Accessible at `http://localhost:8000/docs`.

- `GET /api/data/discover` → Discover datasets  
- `POST /api/data/load-all` → Load datasets  
- `GET /api/nuclei/list` → Paginate nuclei  
- `POST /api/nuclei/mip` → MIP for a nucleus  
- `GET /api/config/` → Get/update config  

---

## 🧑‍💻 Development

### Backend
```bash
pip install -e .
python run.py --debug --reload
pytest
```

### Frontend
```bash
npm start
npm run build
npm test
```

---

## 🛠️ Troubleshooting

- **Server won’t start** → Check Python version, dependencies, free port `8000`.  
- **Frontend not connecting** → Check CORS origins and backend availability.  
- **Data not loading** → Verify config paths and file permissions.  
- **Performance issues** → Reduce `max_objects_per_page`, adjust `mip_chunk_size`.  

---

## 📦 Utility Scripts

- `start_with_pipeline.py` – starts with pipeline config and validation.  
- `check_config.py` – tests config consistency.  
- `test_pipeline_integration.py` – validates epitope integration pipeline.  

---

## 📜 License
MIT License (see `LICENSE` file).

---

## 🙌 Acknowledgments
- Built on **VolAlign** microscopy pipeline.  
- Uses **Zarr** for scalable array storage.  
- Uses **Dask** for distributed computation.  
- UI powered by **React + Material-UI**.  

---

## 🆘 Support
1. Use API docs `/docs`.  
2. Check troubleshooting section.  
3. See backend logs + browser console for errors.  

---

**NOTE:** Designed for research use with large microscopy datasets. Ensure sufficient hardware (RAM, storage) for optimal performance.