/**
 * DataLoader component - handles initial data loading and configuration
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Alert,
  CircularProgress,
  Chip,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Storage as StorageIcon,
  PlayArrow as PlayArrowIcon,
} from '@mui/icons-material';

import { dataAPI, configAPI, utils } from '../services/api';

const DataLoader = ({ onDataLoaded }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [loadedDatasets, setLoadedDatasets] = useState({});

  // Load current configuration on mount and auto-start loading
  useEffect(() => {
    loadConfigAndData();
  }, []);

  const loadConfigAndData = async () => {
    try {
      const configData = await configAPI.getConfig();
      setConfig(configData);
      
      // Check if we're in pipeline mode - comprehensive detection
      const isPipelineMode = configData.data?.mode === 'pipeline' ||
                            !!configData.data?.pipeline ||
                            (configData.data?.base_path && configData.data?.pipeline?.reference_round);
      
      console.log('Initial pipeline mode detection:', {
        mode: configData.data?.mode,
        hasPipelineSection: !!configData.data?.pipeline,
        hasBasePath: !!configData.data?.base_path,
        hasReferenceRound: !!configData.data?.pipeline?.reference_round,
        isPipelineMode,
        fullConfigData: configData.data
      });
      
      if (isPipelineMode) {
        // In pipeline mode, automatically load data and skip to viewer
        setActiveStep(1); // Go to loading step
        // Auto-load data in pipeline mode
        setTimeout(() => {
          handleLoadDatasets(configData);
        }, 500);
      } else {
        // Manual mode - start at step 0 (config editor) to allow user to verify/modify paths
        setActiveStep(0);
      }
    } catch (err) {
      console.error('Failed to load config:', err);
      setError('Failed to load configuration. Please check your config file.');
    }
  };

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setLoadedDatasets({});
    setError(null);
    loadConfigAndData();
  };

  const handleEditConfig = () => {
    setActiveStep(0); // Go to config editing step
  };

  const handleLoadDatasets = async (configData = null) => {
    setLoading(true);
    setError(null);

    try {
      const currentConfig = configData || config;
      if (!currentConfig) {
        throw new Error('No configuration available');
      }

      // Check if we're in pipeline mode - comprehensive detection
      const isPipelineMode = currentConfig.data?.mode === 'pipeline' ||
                            !!currentConfig.data?.pipeline ||
                            (currentConfig.data?.base_path && currentConfig.data?.pipeline?.reference_round);
      
      console.log('Pipeline mode detection:', {
        mode: currentConfig.data?.mode,
        hasPipelineSection: !!currentConfig.data?.pipeline,
        hasBasePath: !!currentConfig.data?.base_path,
        hasReferenceRound: !!currentConfig.data?.pipeline?.reference_round,
        isPipelineMode,
        fullConfigData: currentConfig.data
      });
      
      if (isPipelineMode) {
        // In pipeline mode, use the load-all endpoint which automatically discovers data
        console.log('Using pipeline mode - calling loadAllDatasets()');
        const result = await dataAPI.loadAllDatasets();
        setLoadedDatasets(result.loaded_datasets || {});
        
        // In pipeline mode, we expect different data structure
        if (result.success) {
          // Automatically advance to completion for pipeline mode
          setActiveStep(2);
          // Notify parent that data is loaded
          if (onDataLoaded) {
            onDataLoaded();
          }
        } else {
          setError(result.message || 'Failed to load pipeline data');
        }
      } else {
        // Manual mode - load datasets using the configured paths
        const loadRequest = {
          segmentation: {
            file_path: currentConfig.data.segmentation?.file_path,
            array_key: currentConfig.data.segmentation?.array_key
          },
          dapi_channel: {
            file_path: currentConfig.data.dapi_channel?.file_path,
            array_key: currentConfig.data.dapi_channel?.array_key
          },
          epitope_channels: (currentConfig.data.epitope_channels || []).map(ch => ({
            name: ch.name,
            file_path: ch.file_path,
            array_key: ch.array_key,
            default_color: ch.default_color
          }))
        };

        const result = await dataAPI.loadDatasets(loadRequest);
        setLoadedDatasets(result.loaded_datasets || {});
        
        // Check if essential datasets were loaded
        const hasSegmentation = result.loaded_datasets?.segmentation && !result.loaded_datasets.segmentation.error;
        const hasDapi = result.loaded_datasets?.dapi && !result.loaded_datasets.dapi.error;
        
        if (!hasSegmentation || !hasDapi) {
          setError('Failed to load essential datasets (segmentation and DAPI channel)');
        }
      }
      // Don't automatically advance to next step - let user review the results
    } catch (err) {
      setError(utils.formatErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleConfigUpdate = async (updatedConfig) => {
    // Skip config updates in pipeline mode - comprehensive detection
    const isPipelineMode = config.data?.mode === 'pipeline' ||
                          !!config.data?.pipeline ||
                          (config.data?.base_path && config.data?.pipeline?.reference_round);
    
    if (isPipelineMode) {
      setError(null);
      return;
    }
    
    try {
      await configAPI.updateConfig(updatedConfig);
      await configAPI.saveConfig();
      
      // Reload config
      const newConfig = await configAPI.getConfig();
      setConfig(newConfig);
      
      setError(null);
    } catch (err) {
      setError(utils.formatErrorMessage(err));
    }
  };

  const renderConfigEditor = () => {
    if (!config) return null;

    // Check if we're in pipeline mode - comprehensive detection
    const isPipelineMode = config.data?.mode === 'pipeline' ||
                          !!config.data?.pipeline ||
                          (config.data?.base_path && config.data?.pipeline?.reference_round);
    
    if (isPipelineMode) {
      return (
        <Box>
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>Pipeline Mode Detected</Typography>
            <Typography variant="body2">
              The application is running in pipeline integration mode. Data will be automatically loaded from your VolAlign pipeline outputs.
            </Typography>
          </Alert>
          
          <Typography variant="body2" paragraph>
            <strong>Working Directory:</strong> {config.working_directory}
          </Typography>
          
          <Typography variant="body2" paragraph>
            <strong>Reference Round:</strong> {config.data.pipeline?.reference_round || 'Not specified'}
          </Typography>
          
          <Typography variant="body2" paragraph>
            <strong>Epitope Analysis File:</strong> {config.data.pipeline?.epitope_analysis_file || 'Not specified'}
          </Typography>
          
          {config.data.pipeline?.epitope_channels && config.data.pipeline.epitope_channels.length > 0 && (
            <Box>
              <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                Configured Epitope Channels ({config.data.pipeline.epitope_channels.length})
              </Typography>
              {config.data.pipeline.epitope_channels.map((channel, index) => (
                <Chip
                  key={index}
                  label={`${channel.name} (${channel.wavelength}nm)`}
                  sx={{ mr: 1, mb: 1 }}
                />
              ))}
            </Box>
          )}
        </Box>
      );
    }

    // Manual mode configuration editor
    return (
      <Box>
        <Typography variant="body2" paragraph>
          Review and edit the file paths for your data channels. These paths will be saved to your configuration.
        </Typography>
        
        <Alert severity="info" sx={{ mb: 2 }}>
          Current paths are loaded from your configuration. You can modify them below or use them as-is.
        </Alert>
        
        <TextField
          fullWidth
          label="Segmentation File Path"
          value={config.data.segmentation?.file_path || ''}
          onChange={(e) => setConfig({
            ...config,
            data: {
              ...config.data,
              segmentation: {
                ...config.data.segmentation,
                file_path: e.target.value
              }
            }
          })}
          sx={{ mb: 2 }}
          placeholder="/full/path/to/segmentation_labels.zarr"
        />
        
        <TextField
          fullWidth
          label="DAPI Channel File Path"
          value={config.data.dapi_channel?.file_path || ''}
          onChange={(e) => setConfig({
            ...config,
            data: {
              ...config.data,
              dapi_channel: {
                ...config.data.dapi_channel,
                file_path: e.target.value
              }
            }
          })}
          sx={{ mb: 2 }}
          placeholder="/full/path/to/dapi_channel.zarr"
        />
        
        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
          Epitope Channels ({config.data.epitope_channels?.length || 0})
        </Typography>
        
        {config.data.epitope_channels?.map((channel, index) => (
          <Box key={index} sx={{ mb: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
            <Typography variant="body2" fontWeight={500} gutterBottom>
              {channel.name}
            </Typography>
            <TextField
              fullWidth
              label="File Path"
              value={channel.file_path || ''}
              onChange={(e) => {
                const updatedChannels = [...config.data.epitope_channels];
                updatedChannels[index] = { ...updatedChannels[index], file_path: e.target.value };
                setConfig({
                  ...config,
                  data: {
                    ...config.data,
                    epitope_channels: updatedChannels
                  }
                });
              }}
              size="small"
              placeholder={`/full/path/to/${channel.name}_channel.zarr`}
            />
          </Box>
        )) || (
          <Typography variant="body2" color="text.secondary">
            No epitope channels configured
          </Typography>
        )}
      </Box>
    );
  };

  const renderDatasetStatus = (dataset, name) => {
    if (!dataset) {
      return (
        <Alert severity="warning" sx={{ mb: 1 }}>
          {name} dataset not found
        </Alert>
      );
    }

    if (dataset.error) {
      return (
        <Alert severity="error" sx={{ mb: 1 }}>
          <Typography variant="subtitle2">{name} - Error</Typography>
          <Typography variant="body2">{dataset.error}</Typography>
        </Alert>
      );
    }

    return (
      <Card variant="outlined" sx={{ mb: 1 }}>
        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            <CheckCircleIcon color="success" fontSize="small" />
            <Typography variant="subtitle2">{name}</Typography>
          </Box>
          
          <Box display="flex" gap={1} mb={1}>
            <Chip label={`Shape: ${dataset.shape?.join(' × ')}`} size="small" />
            <Chip label={`Type: ${dataset.dtype}`} size="small" />
          </Box>
          
          <Typography variant="caption" color="text.secondary">
            {dataset.path}
          </Typography>
        </CardContent>
      </Card>
    );
  };

  if (!config) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box maxWidth="md" mx="auto">
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Data Setup
          </Typography>
          
          <Typography variant="body2" color="text.secondary" paragraph>
            Configure your data paths and load your datasets for visualization.
          </Typography>

          <Stepper activeStep={activeStep} orientation="vertical">
            {/* Step 1: Configuration Review/Edit */}
            <Step>
              <StepLabel>
                {(config.data?.mode === 'pipeline' || !!config.data?.pipeline) ? 'Pipeline Configuration' : 'Edit Data Paths'}
              </StepLabel>
              <StepContent>
                {renderConfigEditor()}

                <Box display="flex" gap={1} mt={2}>
                  {!(config.data?.mode === 'pipeline' || !!config.data?.pipeline) ? (
                    <>
                      <Button
                        variant="contained"
                        onClick={async () => {
                          await handleConfigUpdate({
                            data: config.data
                          });
                          handleNext();
                        }}
                        disabled={loading}
                      >
                        Save & Continue
                      </Button>
                      
                      <Button
                        variant="outlined"
                        onClick={() => handleNext()}
                        disabled={loading}
                        startIcon={<PlayArrowIcon />}
                      >
                        Use Current Paths
                      </Button>
                    </>
                  ) : (
                    <Button
                      variant="contained"
                      onClick={() => handleNext()}
                      disabled={loading}
                      startIcon={<PlayArrowIcon />}
                    >
                      Continue to Data Loading
                    </Button>
                  )}
                </Box>
              </StepContent>
            </Step>

            {/* Step 2: Load Datasets */}
            <Step>
              <StepLabel>
                {(config.data?.mode === 'pipeline' || !!config.data?.pipeline) ? 'Loading Pipeline Data' : 'Load Datasets'}
              </StepLabel>
              <StepContent>
                <Typography variant="body2" paragraph>
                  {(config.data?.mode === 'pipeline' || !!config.data?.pipeline)
                    ? 'Loading data from your VolAlign pipeline outputs. This may take several minutes for large datasets...'
                    : 'Load your configured datasets into memory for processing.'
                  }
                </Typography>

                {loading && (config.data?.mode === 'pipeline' || !!config.data?.pipeline) && (
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>Pipeline data loading in progress...</strong>
                    </Typography>
                    <Typography variant="body2">
                      This process may take 2-5 minutes depending on your dataset size.
                      The system is loading zarr files, segmentation masks, and epitope analysis results.
                    </Typography>
                  </Alert>
                )}

                {Object.keys(loadedDatasets).length > 0 && (
                  <Box mb={2}>
                    {renderDatasetStatus(loadedDatasets.segmentation, 'Segmentation')}
                    {renderDatasetStatus(loadedDatasets.dapi, 'DAPI Channel')}
                    
                    {loadedDatasets.epitope_channels && Object.entries(loadedDatasets.epitope_channels).map(([channelName, dataset]) =>
                      <div key={channelName}>
                        {renderDatasetStatus(dataset, `${channelName} Channel`)}
                      </div>
                    )}
                  </Box>
                )}

                <Box display="flex" gap={1}>
                  <Button
                    variant="contained"
                    onClick={() => handleLoadDatasets()}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={16} /> : <StorageIcon />}
                  >
                    {loading ? 'Loading...' : ((config.data?.mode === 'pipeline' || !!config.data?.pipeline) ? 'Load Pipeline Data' : 'Load Datasets')}
                  </Button>
                  
                  {Object.keys(loadedDatasets).length > 0 && !loading && (
                    <Button
                      variant="contained"
                      onClick={handleNext}
                      color="success"
                    >
                      Continue
                    </Button>
                  )}
                  
                  <Button onClick={handleBack}>
                    Back
                  </Button>
                </Box>
              </StepContent>
            </Step>

            {/* Step 3: Complete */}
            <Step>
              <StepLabel>Complete</StepLabel>
              <StepContent>
                <Alert severity="success" sx={{ mb: 2 }}>
                  <Typography variant="h6">Data Loading Complete!</Typography>
                  <Typography>
                    Your datasets have been successfully loaded. The application will now switch to the main interface.
                  </Typography>
                </Alert>

                <Box display="flex" gap={1}>
                  <Button
                    variant="contained"
                    onClick={() => onDataLoaded()}
                    color="primary"
                  >
                    Open Viewer
                  </Button>
                  
                  <Button onClick={handleReset}>
                    Start Over
                  </Button>
                  
                  <Button onClick={handleEditConfig}>
                    Edit Config
                  </Button>
                </Box>
              </StepContent>
            </Step>
          </Stepper>

          {/* Error Display */}
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              <Typography variant="h6">Error</Typography>
              <Typography>{error}</Typography>
            </Alert>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default DataLoader;