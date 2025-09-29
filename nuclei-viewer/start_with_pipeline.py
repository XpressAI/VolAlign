#!/usr/bin/env python3
"""
Simple startup script for nuclei-viewer with pipeline integration.

This script demonstrates how the nuclei-viewer loads and uses configuration
from the main VolAlign config file.

Usage:
    python start_with_pipeline.py [--config CONFIG_PATH] [--port PORT]
"""

import argparse
import sys
from pathlib import Path

# Add the backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.core.config import load_config
# Import validation if available, otherwise skip validation
try:
    from app.core.validation import PipelineValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    print("Warning: Pipeline validation not available")


def main():
    parser = argparse.ArgumentParser(
        description="Start nuclei-viewer with pipeline integration"
    )
    parser.add_argument(
        "--config",
        default="../config_template.yaml",
        help="Path to VolAlign config file (default: ../config_template.yaml)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run server on (default: 8000)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and pipeline, don't start server"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Nuclei Viewer - Pipeline Integration Startup")
    print("=" * 60)
    
    # Step 1: Load configuration
    print(f"\n1. Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
        print(f"   ✓ Configuration loaded successfully")
        
        # Determine mode based on configuration structure
        mode = "pipeline" if config.data.is_pipeline_mode() else "manual"
        print(f"   ✓ Mode: {mode}")
        
        if config.data.is_pipeline_mode():
            print(f"   ✓ Working directory: {config.data.pipeline.pipeline_working_directory}")
            print(f"   ✓ Reference round: {config.data.pipeline.reference_round}")
            print(f"   ✓ Epitope analysis file: {config.data.pipeline.epitope_analysis_file}")
        else:
            print(f"   ✓ Manual mode configuration detected")
        
    except ValueError as e:
        print(f"   ✗ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"   ✗ Failed to load configuration: {e}")
        return 1
    
    # Step 2: Validate pipeline (if in pipeline mode)
    if config.data.is_pipeline_mode():
        print(f"\n2. Validating pipeline structure...")
        if VALIDATION_AVAILABLE:
            try:
                validator = PipelineValidator(config)
                validation_result = validator.validate_complete_pipeline()
                
                if validation_result.is_valid:
                    print(f"   ✓ Pipeline validation passed")
                    print(f"   ✓ Found {len(validation_result.zarr_files)} zarr files")
                    print(f"   ✓ Found {len(validation_result.segmentation_files)} segmentation files")
                    if validation_result.epitope_analysis_exists:
                        print(f"   ✓ Epitope analysis results found")
                    else:
                        print(f"   ⚠ Epitope analysis results not found - will use basic mode")
                else:
                    print(f"   ⚠ Pipeline validation warnings:")
                    for error in validation_result.errors:
                        print(f"     - {error}")
                    print(f"   → Continuing with available data...")
                    
            except Exception as e:
                print(f"   ✗ Pipeline validation failed: {e}")
                print(f"   → Will attempt to start in manual mode")
        else:
            print(f"   ⚠ Pipeline validation skipped (validator not available)")
            # Basic file existence check
            from pathlib import Path
            working_dir = Path(config.data.pipeline.pipeline_working_directory)
            epitope_file = working_dir / config.data.pipeline.epitope_analysis_file
            print(f"   ✓ Working directory exists: {working_dir.exists()}")
            print(f"   ✓ Epitope analysis file exists: {epitope_file.exists()}")
    else:
        print(f"\n2. Manual mode detected - skipping pipeline validation")
    
    if args.validate_only:
        print(f"\n✓ Validation complete - exiting as requested")
        return 0
    
    # Step 3: Show configuration summary
    print(f"\n3. Configuration Summary:")
    print(f"   Server: {config.server.host}:{args.port or config.server.port}")
    print(f"   Debug mode: {config.server.debug}")
    print(f"   CORS origins: {len(config.server.cors_origins)} configured")
    print(f"   Max objects per page: {config.processing.max_objects_per_page}")
    print(f"   Cache MIPs: {config.processing.cache_mips}")
    
    # Step 4: Start the server
    print(f"\n4. Starting server...")
    print(f"   Backend will be available at: http://{config.server.host}:{args.port or config.server.port}")
    print(f"   API documentation at: http://{config.server.host}:{args.port or config.server.port}/docs")
    print(f"   Frontend should be started separately at: http://localhost:3000")
    print(f"\n   To start frontend:")
    print(f"   cd frontend && npm install && npm start")
    
    try:
        # Import and start the FastAPI server
        import uvicorn
        import os
        from app.main import app
        
        # Set environment variable so FastAPI app uses the correct config
        os.environ["NUCLEI_VIEWER_CONFIG"] = args.config
        
        # Override port if specified
        port = args.port or config.server.port

        # Reset shared state before starting server
        from app.core.shared_state import set_shared_data_loader, set_shared_nuclei_processor, _shared_state
        _shared_state.clear()
        set_shared_data_loader(None)
        set_shared_nuclei_processor(None)
        
        print(f"\n   Starting FastAPI server...")
        uvicorn.run(
            app,
            host=config.server.host,
            port=port,
            log_level="info" if config.server.debug else "warning"
        )
        
    except KeyboardInterrupt:
        print(f"\n\n✓ Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n✗ Failed to start server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())