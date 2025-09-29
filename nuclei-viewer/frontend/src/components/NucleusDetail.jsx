/**
 * NucleusDetail component - detailed view of a selected nucleus
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  IconButton,
  Chip,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Tooltip,
  Button,
} from '@mui/material';
import {
  Close as CloseIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material';

import { nucleiAPI, utils } from '../services/api';
import ImageViewer from './ImageViewer';

const NucleusDetail = ({ nucleus, channelSettings, onClose, onEpitopeDataChange }) => {
  const [mipData, setMipData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [compositeImage, setCompositeImage] = useState(null);
  const [compositeLoading] = useState(false);
  
  // Refs for managing timeouts and request cancellation
  const loadTimeoutRef = useRef(null);
  const compositeTimeoutRef = useRef(null);
  const currentNucleusRef = useRef(null);

  const loadNucleusMIP = useCallback(async (nucleusToLoad, channels) => {
    if (!nucleusToLoad) return;
    if (!channels || channels.length === 0) {
      return; // prevent infinite loop when no channels
    }

    // Prevent duplicate load requests for the same nucleus with identical channels
    if (mipData && mipData.nucleus_label === nucleusToLoad.label) {
      const loadedChannels = new Set(Object.keys(mipData.individual_mips || {}));
      const requestedChannels = new Set(channels);
      let allPresent = true;
      for (const c of requestedChannels) {
        if (!loadedChannels.has(c)) {
          allPresent = false;
          break;
        }
      }
      if (allPresent) {
        return; // already loaded required channels
      }
    }
    
    // Store current nucleus for cancellation check
    const nucleusLabel = nucleusToLoad.label;
    currentNucleusRef.current = nucleusLabel;
    
    setLoading(true);
    setError(null);
    setMipData(null); // Clear previous data immediately
    setCompositeImage(null); // Clear previous composite
    
    try {
      if (channels.length === 0) {
        setError('No channels enabled');
        return;
      }

      const mipRequest = {
        nucleus_label: nucleusLabel,
        channels: channels,
        force_recompute: false,
        return_individual: true,
        return_composite: false, // We'll generate composite separately
      };

      const result = await nucleiAPI.computeMIP(mipRequest);
      
      // Check if this request is still relevant (user hasn't switched to another nucleus)
      if (currentNucleusRef.current === nucleusLabel) {
        setMipData(result);
        
        // Pass epitope data to parent component if available
        if (onEpitopeDataChange && result.epitope_calls) {
          onEpitopeDataChange({
            epitope_calls: result.epitope_calls,
            confidence_scores: result.confidence_scores || {}
          });
        }
      }

    } catch (err) {
      // Only show error if this request is still relevant
      if (currentNucleusRef.current === nucleusLabel) {
        setError(utils.formatErrorMessage(err));
      }
    } finally {
      // Only clear loading if this request is still relevant
      if (currentNucleusRef.current === nucleusLabel) {
        setLoading(false);
      }
    }
  }, [onEpitopeDataChange]);

  const generateComposite = useCallback(async () => {
    if (!mipData || !mipData.individual_mips || !nucleus) return;

    // Always use frontend rendering for color changes - no API calls
    // This ensures immediate visual feedback and eliminates API spam
    setCompositeImage(null);
    
    // Note: We're intentionally not making any API calls here.
    // The ImageViewer component will handle all color compositing on the frontend.
    // This provides immediate visual feedback and eliminates the 404 API call issues.
    
  }, [mipData, nucleus]);

  // Load MIP data when nucleus changes (with debouncing for rapid switching)
  useEffect(() => {
    if (!nucleus) {
      setMipData(null);
      setCompositeImage(null);
      return;
    }
    
    // Clear any existing timeout
    if (loadTimeoutRef.current) {
      clearTimeout(loadTimeoutRef.current);
    }
    
    // Get enabled channels
    const enabledChannels = Object.entries(channelSettings)
      .filter(([_, settings]) => settings && settings.enabled)
      .map(([channelName, _]) => channelName);
    
    // Debounce nucleus loading to handle rapid switching
    loadTimeoutRef.current = setTimeout(() => {
      loadNucleusMIP(nucleus, enabledChannels);
    }, 150); // 150ms debounce for nucleus switching
    
    return () => {
      if (loadTimeoutRef.current) {
        clearTimeout(loadTimeoutRef.current);
      }
    };
  }, [nucleus, channelSettings, loadNucleusMIP]);

  // Update composite when MIP data loads (no debouncing needed since no API calls)
  useEffect(() => {
    if (!mipData || !nucleus) return;
    
    // Immediately clear composite to force frontend rendering
    generateComposite();
    
  }, [mipData, generateComposite]);
  
  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (loadTimeoutRef.current) {
        clearTimeout(loadTimeoutRef.current);
      }
      const timeoutId = compositeTimeoutRef.current;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      compositeTimeoutRef.current = null;
    };
  }, []);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleRefresh = () => {
    loadNucleusMIP();
  };

  const handleDownload = (imageData, filename) => {
    // Convert base64 to blob and download
    const link = document.createElement('a');
    link.href = imageData;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const renderMetadataTable = () => {
    if (!mipData) return null;

    const metadata = mipData.metadata;
    const rows = [
      { label: 'Nucleus Label', value: nucleus.label },
      { label: 'Area (voxels)', value: nucleus.area.toLocaleString() },
      { label: 'Centroid (Z, Y, X)', value: nucleus.centroid.map(c => Math.round(c)).join(', ') },
      { label: 'Original Bbox', value: `${mipData.bbox.join(', ')}` },
      { label: 'Padded Bbox', value: `${mipData.padded_bbox.join(', ')}` },
      { label: 'Crop Shape', value: metadata.crop_shape?.join(' × ') },
      { label: 'MIP Shape', value: metadata.mip_shape?.join(' × ') },
      { label: 'XY Padding', value: `${metadata.pad_xy} pixels` },
    ];

    return (
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableBody>
            {rows.map((row, index) => (
              <TableRow key={index}>
                <TableCell component="th" scope="row" sx={{ fontWeight: 500 }}>
                  {row.label}
                </TableCell>
                <TableCell>{row.value}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  const renderChannelStats = () => {
    if (!mipData || !mipData.metadata.channel_stats) return null;

    const channelStats = mipData.metadata.channel_stats;

    return (
      <Box>
        {Object.entries(channelStats).map(([channelName, stats]) => {
          // Check if this channel has epitope call data
          const hasEpitopeCall = mipData.epitope_calls && channelName in mipData.epitope_calls;
          const epitopeCall = hasEpitopeCall ? mipData.epitope_calls[channelName] : null;
          const confidenceScore = mipData.confidence_scores && channelName in mipData.confidence_scores
            ? mipData.confidence_scores[channelName]
            : null;
          const is405Channel = channelName.endsWith('_405');

          return (
            <Card key={channelName} variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Box
                    width={12}
                    height={12}
                    borderRadius="50%"
                    bgcolor={channelSettings[channelName]?.color || '#ffffff'}
                    border="1px solid #ccc"
                  />
                  <Typography variant="h6">{channelName}</Typography>
                  
                  {/* Epitope Classification Tag - only for non-405nm channels */}
                  {!is405Channel && hasEpitopeCall && (
                    <Chip
                      icon={epitopeCall ? <CheckCircleIcon /> : <CancelIcon />}
                      label={epitopeCall ? 'Positive' : 'Negative'}
                      size="small"
                      color={epitopeCall ? 'success' : 'default'}
                      variant="outlined"
                      sx={{
                        fontSize: '0.7rem',
                        height: 24,
                        '& .MuiChip-icon': { fontSize: '0.8rem' }
                      }}
                    />
                  )}
                  
                  {/* DAPI Channel Tag - for 405nm channels */}
                  {is405Channel && (
                    <Chip
                      label="DAPI"
                      size="small"
                      color="primary"
                      variant="outlined"
                      sx={{
                        fontSize: '0.7rem',
                        height: 24
                      }}
                    />
                  )}
                  
                  <Chip
                    label={channelSettings[channelName]?.enabled ? 'Enabled' : 'Disabled'}
                    size="small"
                    color={channelSettings[channelName]?.enabled ? 'success' : 'default'}
                  />
                </Box>
                
                {/* Confidence Score - only for epitope channels with calls */}
                {!is405Channel && hasEpitopeCall && confidenceScore !== null && (
                  <Box mb={1}>
                    <Typography variant="caption" color="text.secondary">
                      Confidence: {(confidenceScore * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                )}
              
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Min: {stats.min.toFixed(1)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Max: {stats.max.toFixed(1)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Mean: {stats.mean.toFixed(1)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Std: {stats.std.toFixed(1)}
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="body2" color="text.secondary">
                      Non-zero voxels: {stats.nonzero_voxels.toLocaleString()}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          );
        })}
      </Box>
    );
  };

  const renderIndividualChannels = () => {
    if (!mipData || !mipData.individual_mips) return null;

    return (
      <Grid container spacing={2}>
        {Object.entries(mipData.individual_mips).map(([channelName, imageData]) => {
          // Check if this channel has epitope call data (exclude 405nm DAPI channels)
          const hasEpitopeCall = mipData.epitope_calls && channelName in mipData.epitope_calls;
          const epitopeCall = hasEpitopeCall ? mipData.epitope_calls[channelName] : null;
          const confidenceScore = mipData.confidence_scores && channelName in mipData.confidence_scores
            ? mipData.confidence_scores[channelName]
            : null;
          const is405Channel = channelName.endsWith('_405');

          // Debug logging for each channel
          console.log(`Individual Channel ${channelName}:`, {
            hasEpitopeCall,
            epitopeCall,
            confidenceScore,
            is405Channel
          });

          return (
            <Grid item xs={12} sm={6} key={channelName}>
              <Card variant="outlined">
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="h6">{channelName}</Typography>
                      {/* Epitope Classification Tag - only for non-405nm channels */}
                      {!is405Channel && hasEpitopeCall && (
                        <Chip
                          icon={epitopeCall ? <CheckCircleIcon /> : <CancelIcon />}
                          label={epitopeCall ? 'Positive' : 'Negative'}
                          size="small"
                          color={epitopeCall ? 'success' : 'default'}
                          variant="outlined"
                          sx={{
                            fontSize: '0.7rem',
                            height: 24,
                            '& .MuiChip-icon': { fontSize: '0.8rem' }
                          }}
                        />
                      )}
                      {/* DAPI Channel Tag - for 405nm channels */}
                      {is405Channel && (
                        <Chip
                          label="DAPI"
                          size="small"
                          color="primary"
                          variant="outlined"
                          sx={{
                            fontSize: '0.7rem',
                            height: 24
                          }}
                        />
                      )}
                    </Box>
                    <IconButton
                      size="small"
                      onClick={() => handleDownload(imageData, `nucleus_${nucleus.label}_${channelName}.png`)}
                    >
                      <DownloadIcon fontSize="small" />
                    </IconButton>
                  </Box>
                  
                  {/* Confidence Score - only for epitope channels with calls */}
                  {!is405Channel && hasEpitopeCall && confidenceScore !== null && (
                    <Box mb={1}>
                      <Typography variant="caption" color="text.secondary">
                        Confidence: {(confidenceScore * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  )}
                  
                  <ImageViewer
                    images={{ [channelName]: imageData }}
                    channelSettings={{ [channelName]: channelSettings[channelName] }}
                    width={250}
                    height={250}
                    showControls={true}
                    allowZoom={true}
                  />
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    );
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Loading Nucleus {nucleus.label}...</Typography>
            <IconButton onClick={onClose}>
              <CloseIcon />
            </IconButton>
          </Box>
          
          <Box display="flex" justifyContent="center" py={4}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Nucleus {nucleus.label}</Typography>
            <Box>
              <IconButton onClick={handleRefresh}>
                <RefreshIcon />
              </IconButton>
              <IconButton onClick={onClose}>
                <CloseIcon />
              </IconButton>
            </Box>
          </Box>
          
          <Alert severity="error">
            <Typography variant="h6">Error Loading Nucleus</Typography>
            <Typography>{error}</Typography>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Nucleus {nucleus.label}
          </Typography>
          
          <Box>
            <Tooltip title="Refresh">
              <IconButton onClick={handleRefresh}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            <IconButton onClick={onClose}>
              <CloseIcon />
            </IconButton>
          </Box>
        </Box>

        {/* Tabs */}
        <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 2 }}>
          <Tab label="Composite" />
          <Tab label="Individual" />
          <Tab label="Metadata" />
          <Tab label="Statistics" />
        </Tabs>

        {/* Tab Content */}
        {activeTab === 0 && (
          // Composite View
          <Box>
            {compositeLoading ? (
              <Box display="flex" flexDirection="column" alignItems="center" py={4}>
                <CircularProgress />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Generating composite image...
                </Typography>
              </Box>
            ) : compositeImage ? (
              <Box display="flex" flexDirection="column" alignItems="center">
                <ImageViewer
                  images={{ composite: compositeImage }}
                  channelSettings={{ composite: { enabled: true, color: '#ffffff' } }}
                  width={400}
                  height={400}
                  showControls={true}
                  allowZoom={true}
                />
                
                <Button
                  startIcon={<DownloadIcon />}
                  onClick={() => handleDownload(compositeImage, `nucleus_${nucleus.label}_composite.png`)}
                  sx={{ mt: 2 }}
                >
                  Download Composite
                </Button>
              </Box>
            ) : mipData && mipData.individual_mips ? (
              <Box display="flex" flexDirection="column" alignItems="center">
                <ImageViewer
                  images={mipData.individual_mips}
                  channelSettings={channelSettings}
                  width={400}
                  height={400}
                  showControls={true}
                  allowZoom={true}
                />
                
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Frontend composite rendering
                </Typography>
              </Box>
            ) : (
              <Alert severity="info">
                No composite image available
              </Alert>
            )}
          </Box>
        )}

        {activeTab === 1 && (
          // Individual Channels
          <Box>
            {renderIndividualChannels()}
          </Box>
        )}

        {activeTab === 2 && (
          // Metadata
          <Box>
            {renderMetadataTable()}
          </Box>
        )}

        {activeTab === 3 && (
          // Statistics
          <Box>
            {renderChannelStats()}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default NucleusDetail;