/**
 * ChannelControls component - manages channel visualization settings
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Switch,
  Slider,
  Button,
  Chip,
  Collapse,
  IconButton,
  FormControlLabel,
  Divider,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Settings as SettingsIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material';
import { ChromePicker } from 'react-color';

const ChannelControls = ({
  channels,
  channelSettings,
  onChannelSettingsChange,
  epitopeData = null
}) => {
  const [expanded, setExpanded] = useState(true);
  const [colorPickerOpen, setColorPickerOpen] = useState(null);
  const [advancedSettingsOpen, setAdvancedSettingsOpen] = useState(null);

  // Cleanup timeouts on unmount
  useEffect(() => {
    const timeouts = { ...colorChangeTimeouts.current };
    return () => {
      Object.values(timeouts).forEach(timeoutId => {
        clearTimeout(timeoutId);
      });
    };
  }, []);

  const handleToggleExpanded = () => {
    setExpanded(!expanded);
  };

  const handleChannelToggle = (channelName) => {
    const currentSettings = channelSettings[channelName] || {};
    onChannelSettingsChange(channelName, {
      ...currentSettings,
      enabled: !currentSettings.enabled,
    });
  };

  const handleOpacityChange = (channelName, opacity) => {
    const currentSettings = channelSettings[channelName] || {};
    onChannelSettingsChange(channelName, {
      ...currentSettings,
      opacity: opacity / 100, // Convert to 0-1 range
    });
  };

  // Debounced color change handler to reduce API calls
  const colorChangeTimeouts = useRef({});
  
  const handleColorChange = useCallback((channelName, color) => {
    // Clear existing timeout for this channel
    if (colorChangeTimeouts.current[channelName]) {
      clearTimeout(colorChangeTimeouts.current[channelName]);
    }
    
    // Set new timeout for debounced update
    colorChangeTimeouts.current[channelName] = setTimeout(() => {
      const currentSettings = channelSettings[channelName] || {};
      onChannelSettingsChange(channelName, {
        ...currentSettings,
        color: color.hex,
      });
      delete colorChangeTimeouts.current[channelName];
    }, 200); // 200ms debounce for color changes
  }, [channelSettings, onChannelSettingsChange]);

  const handleContrastToggle = (channelName) => {
    const currentSettings = channelSettings[channelName] || {};
    onChannelSettingsChange(channelName, {
      ...currentSettings,
      auto_contrast: !currentSettings.auto_contrast,
    });
  };

  const handleContrastRangeChange = (channelName, range) => {
    const currentSettings = channelSettings[channelName] || {};
    onChannelSettingsChange(channelName, {
      ...currentSettings,
      contrast_range: range,
    });
  };

  const toggleAllChannels = (enabled) => {
    Object.keys(channels).forEach(channelName => {
      const currentSettings = channelSettings[channelName] || {};
      onChannelSettingsChange(channelName, {
        ...currentSettings,
        enabled,
      });
    });
  };

  const resetChannelSettings = () => {
    Object.entries(channels).forEach(([channelName, channelInfo]) => {
      onChannelSettingsChange(channelName, {
        enabled: true,
        color: channelInfo.default_color,
        opacity: 0.8,
        auto_contrast: true,
        contrast_range: [1.0, 99.9],
      });
    });
  };

  const getEnabledChannelCount = () => {
    return Object.values(channelSettings).filter(settings => settings.enabled).length;
  };

  const renderChannelControl = (channelName, channelInfo) => {
    const settings = channelSettings[channelName] || {};
    const isEnabled = settings.enabled !== false;
    const opacity = Math.round((settings.opacity || 0.8) * 100);
    const color = settings.color || channelInfo.default_color;

    // Check for epitope call data (exclude 405nm DAPI channels)
    const is405Channel = channelName.endsWith('_405');
    const hasEpitopeCall = epitopeData && epitopeData.epitope_calls && channelName in epitopeData.epitope_calls;
    const epitopeCall = hasEpitopeCall ? epitopeData.epitope_calls[channelName] : null;
    const confidenceScore = epitopeData && epitopeData.confidence_scores && channelName in epitopeData.confidence_scores
      ? epitopeData.confidence_scores[channelName]
      : null;

    return (
      <Card key={channelName} variant="outlined" sx={{ mb: 1 }}>
        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            {/* Channel Name and Type */}
            <Box display="flex" alignItems="center" gap={1}>
              <FormControlLabel
                control={
                  <Switch
                    checked={isEnabled}
                    onChange={() => handleChannelToggle(channelName)}
                    size="small"
                  />
                }
                label={
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography variant="body2" fontWeight={500}>
                      {channelInfo.name || channelName}
                    </Typography>
                    <Chip
                      label={channelInfo.type}
                      size="small"
                      variant="outlined"
                      sx={{ height: 20, fontSize: '0.7rem' }}
                    />
                    {/* Epitope Classification Tag - only for non-405nm channels */}
                    {!is405Channel && hasEpitopeCall && (
                      <Chip
                        icon={epitopeCall ? <CheckCircleIcon /> : <CancelIcon />}
                        label={epitopeCall ? 'Positive' : 'Negative'}
                        size="small"
                        color={epitopeCall ? 'success' : 'default'}
                        variant="outlined"
                        sx={{
                          fontSize: '0.65rem',
                          height: 20,
                          '& .MuiChip-icon': { fontSize: '0.7rem' }
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
                          fontSize: '0.65rem',
                          height: 20
                        }}
                      />
                    )}
                  </Box>
                }
              />
            </Box>

            {/* Controls */}
            <Box display="flex" alignItems="center" gap={1}>
              {/* Color Picker */}
              <Tooltip title="Change color">
                <IconButton
                  size="small"
                  onClick={() => setColorPickerOpen(channelName)}
                  disabled={!isEnabled}
                >
                  <Box
                    width={20}
                    height={20}
                    borderRadius="50%"
                    bgcolor={color}
                    border="1px solid #ccc"
                  />
                </IconButton>
              </Tooltip>

              {/* Advanced Settings */}
              <Tooltip title="Advanced settings">
                <IconButton
                  size="small"
                  onClick={() => setAdvancedSettingsOpen(channelName)}
                  disabled={!isEnabled}
                >
                  <SettingsIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {/* Confidence Score - only for epitope channels with calls */}
          {!is405Channel && hasEpitopeCall && confidenceScore !== null && (
            <Box mt={1}>
              <Typography variant="caption" color="text.secondary">
                Confidence: {(confidenceScore * 100).toFixed(1)}%
              </Typography>
            </Box>
          )}

          {/* Opacity Slider */}
          {isEnabled && (
            <Box mt={1}>
              <Typography variant="caption" color="text.secondary">
                Opacity: {opacity}%
              </Typography>
              <Slider
                value={opacity}
                onChange={(_, value) => handleOpacityChange(channelName, value)}
                min={0}
                max={100}
                size="small"
                sx={{ mt: 0.5 }}
              />
            </Box>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderColorPicker = () => {
    if (!colorPickerOpen) return null;

    const channelInfo = channels[colorPickerOpen];
    const settings = channelSettings[colorPickerOpen] || {};
    const currentColor = settings.color || channelInfo.default_color;

    return (
      <Dialog
        open={true}
        onClose={() => setColorPickerOpen(null)}
        maxWidth="xs"
      >
        <DialogTitle>
          Choose Color for {channelInfo.name || colorPickerOpen}
        </DialogTitle>
        <DialogContent>
          <ChromePicker
            color={currentColor}
            onChange={(color) => handleColorChange(colorPickerOpen, color)}
            disableAlpha={true}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setColorPickerOpen(null)}>
            Done
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  const renderAdvancedSettings = () => {
    if (!advancedSettingsOpen) return null;

    const channelInfo = channels[advancedSettingsOpen];
    const settings = channelSettings[advancedSettingsOpen] || {};
    const contrastRange = settings.contrast_range || [1.0, 99.9];

    return (
      <Dialog
        open={true}
        onClose={() => setAdvancedSettingsOpen(null)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Advanced Settings - {channelInfo.name || advancedSettingsOpen}
        </DialogTitle>
        <DialogContent>
          <Box py={2}>
            {/* Auto Contrast */}
            <FormControlLabel
              control={
                <Switch
                  checked={settings.auto_contrast !== false}
                  onChange={() => handleContrastToggle(advancedSettingsOpen)}
                />
              }
              label="Auto Contrast"
              sx={{ mb: 2 }}
            />

            {/* Manual Contrast Range */}
            {settings.auto_contrast !== false && (
              <Box>
                <Typography variant="body2" gutterBottom>
                  Contrast Percentile Range: {contrastRange[0]}% - {contrastRange[1]}%
                </Typography>
                <Slider
                  value={contrastRange}
                  onChange={(_, value) => handleContrastRangeChange(advancedSettingsOpen, value)}
                  valueLabelDisplay="auto"
                  min={0}
                  max={100}
                  step={0.1}
                  marks={[
                    { value: 0, label: '0%' },
                    { value: 50, label: '50%' },
                    { value: 100, label: '100%' },
                  ]}
                />
              </Box>
            )}

            {/* Channel Info */}
            <Divider sx={{ my: 2 }} />
            <Typography variant="body2" color="text.secondary">
              <strong>Type:</strong> {channelInfo.type}<br />
              <strong>Shape:</strong> {channelInfo.shape?.join(' × ')}<br />
              <strong>Data Type:</strong> {channelInfo.dtype}
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAdvancedSettingsOpen(null)}>
            Done
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  if (!channels || Object.keys(channels).length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography color="text.secondary">
            No channels available. Please load data first.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="h6">
              Channel Controls
            </Typography>
            <Chip 
              label={`${getEnabledChannelCount()}/${Object.keys(channels).length} enabled`}
              size="small"
              color={getEnabledChannelCount() > 0 ? "primary" : "default"}
            />
          </Box>

          <Box display="flex" alignItems="center" gap={1}>
            {/* Global Controls */}
            <Tooltip title="Show all channels">
              <IconButton size="small" onClick={() => toggleAllChannels(true)}>
                <VisibilityIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Hide all channels">
              <IconButton size="small" onClick={() => toggleAllChannels(false)}>
                <VisibilityOffIcon fontSize="small" />
              </IconButton>
            </Tooltip>

            <Button size="small" onClick={resetChannelSettings}>
              Reset
            </Button>

            {/* Expand/Collapse */}
            <IconButton size="small" onClick={handleToggleExpanded}>
              {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>
        </Box>

        {/* Channel Controls */}
        <Collapse in={expanded}>
          <Box>
            {Object.entries(channels).map(([channelName, channelInfo]) =>
              renderChannelControl(channelName, channelInfo)
            )}
          </Box>
        </Collapse>

        {/* Color Picker Dialog */}
        {renderColorPicker()}

        {/* Advanced Settings Dialog */}
        {renderAdvancedSettings()}
      </CardContent>
    </Card>
  );
};

export default ChannelControls;