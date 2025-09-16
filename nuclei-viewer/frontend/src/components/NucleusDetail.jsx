/**
 * NucleusDetail component - detailed view of a selected nucleus
 */

import React, { useState, useEffect, useCallback } from 'react';
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
} from '@mui/icons-material';

import { nucleiAPI, utils } from '../services/api';
import ImageViewer from './ImageViewer';

const NucleusDetail = ({ nucleus, channelSettings, onClose }) => {
  const [mipData, setMipData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [compositeImage, setCompositeImage] = useState(null);

  const loadNucleusMIP = useCallback(async () => {
    if (!nucleus) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Get enabled channels
      const enabledChannels = Object.entries(channelSettings)
        .filter(([_, settings]) => settings.enabled)
        .map(([channelName, _]) => channelName);

      if (enabledChannels.length === 0) {
        setError('No channels enabled');
        return;
      }

      const mipRequest = {
        nucleus_label: nucleus.label,
        channels: enabledChannels,
        force_recompute: false,
        return_individual: true,
        return_composite: false, // We'll generate composite on frontend
      };

      const result = await nucleiAPI.computeMIP(mipRequest);
      setMipData(result);

    } catch (err) {
      setError(utils.formatErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [nucleus, channelSettings]);

  const generateComposite = useCallback(async () => {
    if (!mipData || !mipData.individual_mips || !nucleus) return;

    try {
      // Request composite from backend with current channel settings
      const mipRequest = {
        nucleus_label: nucleus.label,
        channels: Object.keys(mipData.individual_mips),
        force_recompute: false,
        channel_settings: channelSettings,
        return_individual: false,
        return_composite: true,
      };

      const result = await nucleiAPI.computeMIP(mipRequest);
      setCompositeImage(result.composite_mip);

    } catch (err) {
      console.error('Failed to generate composite:', err);
    }
  }, [mipData, nucleus, channelSettings]);

  // Load MIP data when nucleus changes
  useEffect(() => {
    if (nucleus) {
      loadNucleusMIP();
    }
  }, [nucleus, loadNucleusMIP]);

  // Update composite when channel settings change
  useEffect(() => {
    if (mipData && channelSettings) {
      generateComposite();
    }
  }, [mipData, channelSettings, generateComposite]);

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
        {Object.entries(channelStats).map(([channelName, stats]) => (
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
                <Chip 
                  label={channelSettings[channelName]?.enabled ? 'Enabled' : 'Disabled'}
                  size="small"
                  color={channelSettings[channelName]?.enabled ? 'success' : 'default'}
                />
              </Box>
              
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
        ))}
      </Box>
    );
  };

  const renderIndividualChannels = () => {
    if (!mipData || !mipData.individual_mips) return null;

    return (
      <Grid container spacing={2}>
        {Object.entries(mipData.individual_mips).map(([channelName, imageData]) => (
          <Grid item xs={12} sm={6} key={channelName}>
            <Card variant="outlined">
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="h6">{channelName}</Typography>
                  <IconButton
                    size="small"
                    onClick={() => handleDownload(imageData, `nucleus_${nucleus.label}_${channelName}.png`)}
                  >
                    <DownloadIcon fontSize="small" />
                  </IconButton>
                </Box>
                
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
        ))}
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
            {compositeImage ? (
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