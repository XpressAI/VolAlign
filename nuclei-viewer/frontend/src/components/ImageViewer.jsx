/**
 * ImageViewer component - displays MIP images with channel overlays
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Tooltip,
  Menu,
  MenuItem,
  FormControlLabel,
  Switch,
} from '@mui/material';
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon,
  Layers as LayersIcon,
} from '@mui/icons-material';

const ImageViewer = ({ 
  images, 
  channelSettings, 
  width = 300, 
  height = 300,
  showControls = true,
  allowZoom = true 
}) => {
  const canvasRef = useRef(null);
  const renderIdRef = useRef(0);
  const imageCache = useRef(new Map());
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const [layerMenuAnchor, setLayerMenuAnchor] = useState(null);
  const [visibleLayers, setVisibleLayers] = useState({});

  // Initialize visible layers based on channel settings
  useEffect(() => {
    if (images && channelSettings) {
      const initialVisible = {};
      Object.keys(images).forEach(channelName => {
        const settings = channelSettings[channelName];
        initialVisible[channelName] = settings?.enabled !== false;
      });
      setVisibleLayers(initialVisible);
    }
  }, [images, channelSettings]);

  const renderImages = useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas || !images) return;

    // Increment render ID to track this render call
    const currentRenderId = ++renderIdRef.current;

    const ctx = canvas.getContext('2d');
    
    // Save current state before any modifications
    ctx.save();
    
    // Reset transformation matrix and clear canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Reset all canvas properties to defaults
    ctx.globalAlpha = 1.0;
    ctx.globalCompositeOperation = 'source-over';
    
    // Restore the saved state (this restores the default transform)
    ctx.restore();

    // Now set up our transformation
    ctx.save();
    ctx.translate(canvas.width / 2 + pan.x, canvas.height / 2 + pan.y);
    ctx.scale(zoom, zoom);

    try {
      // Load and composite images with caching
      const imagePromises = Object.entries(images)
        .filter(([channelName]) => visibleLayers[channelName])
        .map(async ([channelName, base64Image]) => {
          // Check cache first
          const cacheKey = `${channelName}_${base64Image}`;
          if (imageCache.current.has(cacheKey)) {
            return { channelName, img: imageCache.current.get(cacheKey) };
          }
          
          return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
              // Cache the loaded image
              imageCache.current.set(cacheKey, img);
              resolve({ channelName, img });
            };
            img.onerror = reject;
            img.src = base64Image;
          });
        });

      const loadedImages = await Promise.all(imagePromises);

      // Check if this render is still the latest one
      if (currentRenderId !== renderIdRef.current) {
        // A newer render has started, abandon this one
        ctx.restore();
        return;
      }

      if (loadedImages.length === 0) {
        // No images to display - restore context first, then draw message
        ctx.restore();
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset to identity for text drawing
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#666';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No images to display', canvas.width / 2, canvas.height / 2);
        ctx.restore();
        return;
      }

      // Get dimensions from first image
      const firstImg = loadedImages[0].img;
      const imgWidth = firstImg.width;
      const imgHeight = firstImg.height;

      // Calculate scaling to fit canvas
      const scaleX = (canvas.width * 0.8) / imgWidth;
      const scaleY = (canvas.height * 0.8) / imgHeight;
      const scale = Math.min(scaleX, scaleY);

      const scaledWidth = imgWidth * scale;
      const scaledHeight = imgHeight * scale;
      const x = -scaledWidth / 2;
      const y = -scaledHeight / 2;

      // Check again before rendering (in case of rapid successive calls)
      if (currentRenderId !== renderIdRef.current) {
        ctx.restore();
        return;
      }

      // For composite generation, show only DAPI channel to avoid bias from empty channels
      // Find DAPI channel (typically contains 'dapi' in name or is the 405nm channel)
      const dapiChannel = loadedImages.find(({ channelName }) =>
        channelName.toLowerCase().includes('dapi') ||
        channelName.includes('405') ||
        channelName.toLowerCase().includes('round1_405')
      );
      
      if (dapiChannel) {
        // Render only DAPI channel
        const { channelName, img } = dapiChannel;
        const settings = channelSettings[channelName] || {};
        const opacity = settings.opacity || 0.8;
        
        // Set blend mode and opacity
        ctx.globalAlpha = opacity;
        ctx.globalCompositeOperation = 'source-over';
        
        // For colored channels, we need to apply color tinting
        if (settings.color && settings.color !== '#ffffff') {
          // Create a temporary canvas for color manipulation
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = img.width;
          tempCanvas.height = img.height;
          const tempCtx = tempCanvas.getContext('2d');
          
          // Clear temp canvas
          tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
          
          // Draw original image
          tempCtx.drawImage(img, 0, 0);
          
          // Apply color tint
          tempCtx.globalCompositeOperation = 'multiply';
          tempCtx.fillStyle = settings.color;
          tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
          
          // Draw tinted image
          ctx.drawImage(tempCanvas, x, y, scaledWidth, scaledHeight);
        } else {
          // Draw image directly
          ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
        }
      } else {
        // Fallback: render all channels if no DAPI found
        for (const { channelName, img } of loadedImages) {
          const settings = channelSettings[channelName] || {};
          const opacity = settings.opacity || 0.8;
          
          // Set blend mode and opacity
          ctx.globalAlpha = opacity;
          ctx.globalCompositeOperation = 'source-over';
          
          // For colored channels, we need to apply color tinting
          if (settings.color && settings.color !== '#ffffff') {
            // Create a temporary canvas for color manipulation
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = img.width;
            tempCanvas.height = img.height;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Clear temp canvas
            tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
            
            // Draw original image
            tempCtx.drawImage(img, 0, 0);
            
            // Apply color tint
            tempCtx.globalCompositeOperation = 'multiply';
            tempCtx.fillStyle = settings.color;
            tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
            
            // Draw tinted image
            ctx.drawImage(tempCanvas, x, y, scaledWidth, scaledHeight);
          } else {
            // Draw image directly
            ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
          }
        }
      }

    } catch (error) {
      console.error('Error rendering images:', error);
      
      // Show error message - restore context first, then draw error
      ctx.restore();
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset to identity for error drawing
      ctx.fillStyle = '#ffebee';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#c62828';
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Error loading images', canvas.width / 2, canvas.height / 2);
      ctx.restore();
      return;
    }

    ctx.restore();
  }, [images, channelSettings, visibleLayers, zoom, pan]);

  // Render images on canvas when data changes (optimized for smooth interactions)
  useEffect(() => {
    // Use requestAnimationFrame for smooth rendering
    const animationId = requestAnimationFrame(() => {
      renderImages();
    });
    
    return () => cancelAnimationFrame(animationId);
  }, [renderImages]);

  // Add wheel event listener with proper passive handling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !allowZoom) return;

    const handleWheelEvent = (event) => {
      event.preventDefault();
      const delta = event.deltaY > 0 ? 0.9 : 1.1;
      setZoom(prev => Math.max(0.1, Math.min(5, prev * delta)));
    };

    // Add event listener with passive: false to allow preventDefault
    canvas.addEventListener('wheel', handleWheelEvent, { passive: false });

    return () => {
      canvas.removeEventListener('wheel', handleWheelEvent);
    };
  }, [allowZoom]);

  const handleMouseDown = (event) => {
    if (!allowZoom) return;
    
    setIsDragging(true);
    setLastMousePos({ x: event.clientX, y: event.clientY });
  };

  const handleMouseMove = (event) => {
    if (!isDragging || !allowZoom) return;

    const deltaX = event.clientX - lastMousePos.x;
    const deltaY = event.clientY - lastMousePos.y;

    setPan(prev => ({
      x: prev.x + deltaX,
      y: prev.y + deltaY
    }));

    setLastMousePos({ x: event.clientX, y: event.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(5, prev * 1.2));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(0.1, prev / 1.2));
  };

  const handleCenter = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleLayerToggle = (channelName) => {
    setVisibleLayers(prev => ({
      ...prev,
      [channelName]: !prev[channelName]
    }));
  };

  const handleLayerMenuOpen = (event) => {
    setLayerMenuAnchor(event.currentTarget);
  };

  const handleLayerMenuClose = () => {
    setLayerMenuAnchor(null);
  };

  if (!images || Object.keys(images).length === 0) {
    return (
      <Box
        width={width}
        height={height}
        display="flex"
        alignItems="center"
        justifyContent="center"
        bgcolor="grey.100"
        borderRadius={1}
      >
        <Typography variant="body2" color="text.secondary">
          No image data
        </Typography>
      </Box>
    );
  }

  return (
    <Box position="relative" width={width} height={height}>
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          border: '1px solid #ddd',
          borderRadius: 4,
          cursor: allowZoom ? (isDragging ? 'grabbing' : 'grab') : 'default',
          display: 'block',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      {/* Controls */}
      {showControls && (
        <Box
          position="absolute"
          top={8}
          right={8}
          display="flex"
          flexDirection="column"
          gap={0.5}
        >
          {allowZoom && (
            <>
              <Tooltip title="Zoom in">
                <IconButton
                  size="small"
                  onClick={handleZoomIn}
                  sx={{ bgcolor: 'rgba(255,255,255,0.8)' }}
                >
                  <ZoomInIcon fontSize="small" />
                </IconButton>
              </Tooltip>

              <Tooltip title="Zoom out">
                <IconButton
                  size="small"
                  onClick={handleZoomOut}
                  sx={{ bgcolor: 'rgba(255,255,255,0.8)' }}
                >
                  <ZoomOutIcon fontSize="small" />
                </IconButton>
              </Tooltip>

              <Tooltip title="Reset view">
                <IconButton
                  size="small"
                  onClick={handleCenter}
                  sx={{ bgcolor: 'rgba(255,255,255,0.8)' }}
                >
                  <CenterIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </>
          )}

          <Tooltip title="Layer controls">
            <IconButton
              size="small"
              onClick={handleLayerMenuOpen}
              sx={{ bgcolor: 'rgba(255,255,255,0.8)' }}
            >
              <LayersIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      )}

      {/* Layer Menu */}
      <Menu
        anchorEl={layerMenuAnchor}
        open={Boolean(layerMenuAnchor)}
        onClose={handleLayerMenuClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        {Object.keys(images).map(channelName => {
          const settings = channelSettings[channelName] || {};
          return (
            <MenuItem key={channelName} dense>
              <FormControlLabel
                control={
                  <Switch
                    checked={visibleLayers[channelName] || false}
                    onChange={() => handleLayerToggle(channelName)}
                    size="small"
                  />
                }
                label={
                  <Box display="flex" alignItems="center" gap={1}>
                    <Box
                      width={12}
                      height={12}
                      borderRadius="50%"
                      bgcolor={settings.color || '#ffffff'}
                      border="1px solid #ccc"
                    />
                    <Typography variant="body2">
                      {channelName}
                    </Typography>
                  </Box>
                }
              />
            </MenuItem>
          );
        })}
      </Menu>

      {/* Zoom indicator */}
      {allowZoom && showControls && zoom !== 1 && (
        <Box
          position="absolute"
          bottom={8}
          left={8}
          bgcolor="rgba(0,0,0,0.7)"
          color="white"
          px={1}
          py={0.5}
          borderRadius={1}
        >
          <Typography variant="caption">
            {Math.round(zoom * 100)}%
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default ImageViewer;