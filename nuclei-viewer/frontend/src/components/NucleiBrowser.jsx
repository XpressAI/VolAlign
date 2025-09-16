/**
 * NucleiBrowser component - displays paginated grid of nuclei
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardActionArea,
  Typography,
  Pagination,
  Grid,
  Chip,
  CircularProgress,
  Alert,
  Skeleton,
  Tooltip,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

import { nucleiAPI, utils } from '../services/api';
import ImageViewer from './ImageViewer';

const NucleiBrowser = ({ 
  onNucleusSelected, 
  selectedNucleus, 
  channelSettings 
}) => {
  // State management
  const [nucleiData, setNucleiData] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [nucleiMIPs, setNucleiMIPs] = useState({});
  const [loadingMIPs, setLoadingMIPs] = useState(new Set());
  const [nucleiStats, setNucleiStats] = useState(null);

  const loadMIPsForNuclei = useCallback(async (nuclei) => {
    if (!nuclei || nuclei.length === 0) return;
    
    const nucleusLabels = nuclei.map(n => n.label);
    
    // Get enabled channels
    const enabledChannels = Object.entries(channelSettings || {})
      .filter(([_, settings]) => settings && settings.enabled)
      .map(([channelName, _]) => channelName);
    
    if (enabledChannels.length === 0) {
      // No channels enabled, clear any existing MIPs for these nuclei
      setNucleiMIPs(prev => {
        const updated = { ...prev };
        nucleusLabels.forEach(label => delete updated[label]);
        return updated;
      });
      return;
    }
    
    // Check which nuclei actually need MIP loading (avoid redundant requests)
    const nucleiNeedingMIPs = nucleusLabels.filter(label => {
      const existingMIP = nucleiMIPs[label];
      if (!existingMIP) return true;
      
      // Check if existing MIP has all required channels
      const existingChannels = Object.keys(existingMIP.individual_mips || {});
      return !enabledChannels.every(channel => existingChannels.includes(channel));
    });
    
    if (nucleiNeedingMIPs.length === 0) {
      return; // All nuclei already have required MIPs
    }
    
    // Mark only nuclei that need loading
    setLoadingMIPs(prev => {
      const newSet = new Set(prev);
      nucleiNeedingMIPs.forEach(label => newSet.add(label));
      return newSet;
    });
    
    try {
      // Batch load MIPs only for nuclei that need them
      const batchResult = await nucleiAPI.computeBatchMIPs(
        nucleiNeedingMIPs,
        enabledChannels,
        true
      );
      
      // Update MIPs state
      setNucleiMIPs(prev => ({
        ...prev,
        ...batchResult.results
      }));
      
    } catch (err) {
      console.error('Failed to load MIPs:', err);
    } finally {
      // Clear loading state
      setLoadingMIPs(prev => {
        const newSet = new Set(prev);
        nucleiNeedingMIPs.forEach(label => newSet.delete(label));
        return newSet;
      });
    }
  }, [channelSettings, nucleiMIPs]);

  const loadNucleiPage = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // First ensure nuclei are extracted
      await nucleiAPI.extractNuclei();
      
      // Then load the page
      const data = await nucleiAPI.listNuclei(currentPage, pageSize);
      setNucleiData(data);
      
    } catch (err) {
      setError(utils.formatErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [currentPage, pageSize]);

  // Separate effect to load MIPs when nuclei data or channel settings change (with debouncing)
  useEffect(() => {
    if (!nucleiData?.nuclei || Object.keys(channelSettings).length === 0) {
      return;
    }
    
    // Debounce channel settings changes to avoid excessive API calls
    const timeoutId = setTimeout(() => {
      loadMIPsForNuclei(nucleiData.nuclei);
    }, 300); // 300ms debounce
    
    return () => clearTimeout(timeoutId);
  }, [nucleiData, channelSettings, loadMIPsForNuclei]);

  const loadNucleiStats = async () => {
    try {
      const stats = await nucleiAPI.getNucleiStats();
      setNucleiStats(stats);
    } catch (err) {
      console.error('Failed to load nuclei stats:', err);
    }
  };

  // Load nuclei data on mount and when page changes
  useEffect(() => {
    loadNucleiPage();
  }, [loadNucleiPage]);

  // Load nuclei statistics on mount
  useEffect(() => {
    loadNucleiStats();
  }, []);


  const handlePageChange = (event, newPage) => {
    setCurrentPage(newPage - 1); // Material-UI uses 1-based indexing
  };

  const handlePageSizeChange = (event) => {
    setPageSize(event.target.value);
    setCurrentPage(0); // Reset to first page
  };

  const handleNucleusClick = (nucleus) => {
    onNucleusSelected(nucleus);
  };

  const handleRefresh = () => {
    loadNucleiPage();
    loadNucleiStats();
  };

  const renderNucleusCard = (nucleus) => {
    const isSelected = selectedNucleus && selectedNucleus.label === nucleus.label;
    const isLoadingMIP = loadingMIPs.has(nucleus.label);
    const mipData = nucleiMIPs[nucleus.label];
    
    return (
      <Grid item xs={12} sm={6} md={4} lg={3} key={nucleus.label}>
        <Card 
          variant={isSelected ? "outlined" : "elevation"}
          sx={{ 
            height: '100%',
            border: isSelected ? 2 : 0,
            borderColor: isSelected ? 'primary.main' : 'transparent',
          }}
        >
          <CardActionArea 
            onClick={() => handleNucleusClick(nucleus)}
            sx={{ height: '100%' }}
          >
            <CardContent>
              {/* Nucleus Image */}
              <Box 
                sx={{ 
                  height: 150, 
                  mb: 1, 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  backgroundColor: 'grey.100',
                  borderRadius: 1,
                }}
              >
                {isLoadingMIP ? (
                  <CircularProgress size={40} />
                ) : mipData && mipData.individual_mips ? (
                  <ImageViewer
                    images={mipData.individual_mips}
                    channelSettings={channelSettings}
                    width={140}
                    height={140}
                    showControls={false}
                    allowZoom={false}
                  />
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No image
                  </Typography>
                )}
              </Box>
              
              {/* Nucleus Info */}
              <Typography variant="h6" component="div" gutterBottom>
                Nucleus {nucleus.label}
              </Typography>
              
              <Box display="flex" flexWrap="wrap" gap={0.5} mb={1}>
                <Chip 
                  label={`Area: ${nucleus.area}`} 
                  size="small" 
                  variant="outlined" 
                />
                <Chip 
                  label={`Z: ${Math.round(nucleus.centroid[0])}`} 
                  size="small" 
                  variant="outlined" 
                />
              </Box>
              
              {nucleus.available_channels && (
                <Typography variant="body2" color="text.secondary">
                  Channels: {nucleus.available_channels.join(', ')}
                </Typography>
              )}
            </CardContent>
          </CardActionArea>
        </Card>
      </Grid>
    );
  };

  if (loading && !nucleiData) {
    return (
      <Box>
        <Typography variant="h5" gutterBottom>
          Loading Nuclei...
        </Typography>
        <Grid container spacing={2}>
          {[...Array(pageSize)].map((_, index) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
              <Card>
                <CardContent>
                  <Skeleton variant="rectangular" height={150} sx={{ mb: 1 }} />
                  <Skeleton variant="text" height={32} />
                  <Skeleton variant="text" height={20} />
                  <Skeleton variant="text" height={20} />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert 
        severity="error" 
        action={
          <IconButton color="inherit" size="small" onClick={handleRefresh}>
            <RefreshIcon />
          </IconButton>
        }
      >
        <Typography variant="h6">Error Loading Nuclei</Typography>
        <Typography>{error}</Typography>
      </Alert>
    );
  }

  if (!nucleiData) {
    return (
      <Alert severity="info">
        <Typography>No nuclei data available. Please load data first.</Typography>
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5">
          Nuclei Browser
        </Typography>
        
        <Box display="flex" alignItems="center" gap={2}>
          {/* Statistics */}
          {nucleiStats && (
            <Tooltip title="Dataset Statistics">
              <Chip
                icon={<InfoIcon />}
                label={`${nucleiStats.total_nuclei} nuclei`}
                variant="outlined"
              />
            </Tooltip>
          )}
          
          {/* Page Size Selector */}
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Per Page</InputLabel>
            <Select
              value={pageSize}
              label="Per Page"
              onChange={handlePageSizeChange}
            >
              <MenuItem value={5}>5</MenuItem>
              <MenuItem value={10}>10</MenuItem>
              <MenuItem value={20}>20</MenuItem>
              <MenuItem value={50}>50</MenuItem>
            </Select>
          </FormControl>
          
          {/* Refresh Button */}
          <IconButton onClick={handleRefresh} disabled={loading}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Nuclei Grid */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {nucleiData.nuclei.map(renderNucleusCard)}
      </Grid>

      {/* Pagination */}
      {nucleiData.total_pages > 1 && (
        <Box display="flex" justifyContent="center">
          <Pagination
            count={nucleiData.total_pages}
            page={currentPage + 1} // Material-UI uses 1-based indexing
            onChange={handlePageChange}
            color="primary"
            size="large"
            showFirstButton
            showLastButton
          />
        </Box>
      )}

      {/* Loading Overlay */}
      {loading && (
        <Box 
          position="absolute" 
          top={0} 
          left={0} 
          right={0} 
          bottom={0} 
          display="flex" 
          alignItems="center" 
          justifyContent="center"
          bgcolor="rgba(255, 255, 255, 0.8)"
          zIndex={1}
        >
          <CircularProgress />
        </Box>
      )}
    </Box>
  );
};

export default NucleiBrowser;