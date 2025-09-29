/**
 * Main App component for the Nuclei Viewer
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Alert,
  Snackbar,
  CircularProgress,
  Backdrop,
} from '@mui/material';

import NucleiBrowser from './components/NucleiBrowser';
import ChannelControls from './components/ChannelControls';
import NucleusDetail from './components/NucleusDetail';
import DataLoader from './components/DataLoader';
import { serverAPI, dataAPI, utils } from './services/api';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
});

function App() {
  // Application state
  const [serverConnected, setServerConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [availableChannels, setAvailableChannels] = useState({});
  const [selectedNucleus, setSelectedNucleus] = useState(null);
  const [channelSettings, setChannelSettings] = useState({});
  const [serverInfo, setServerInfo] = useState(null);
  const [currentEpitopeData, setCurrentEpitopeData] = useState(null);

  // Snackbar state
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info',
  });

  const checkServerConnection = useCallback(async () => {
    setLoading(true);
    try {
      const connected = await utils.checkServerConnection();
      setServerConnected(connected);
      
      if (connected) {
        // Get server info and data status in parallel to reduce API calls
        const [info, status] = await Promise.all([
          serverAPI.getServerInfo(),
          dataAPI.getDataStatus()
        ]);
        
        setServerInfo(info);
        
        if (status.datasets_loaded.segmentation && status.datasets_loaded.dapi) {
          setDataLoaded(true);
          showSnackbar('Connected to server - Data already loaded', 'success');
        } else {
          showSnackbar('Connected to server successfully', 'success');
        }
      } else {
        setError('Unable to connect to server. Please ensure the backend is running.');
      }
    } catch (err) {
      setError(utils.formatErrorMessage(err));
      setServerConnected(false);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadAvailableChannels = useCallback(async () => {
    try {
      const channelsData = await dataAPI.getChannels();
      setAvailableChannels(channelsData.channels);
      
      // Initialize channel settings with defaults only if not already set
      setChannelSettings(prev => {
        if (Object.keys(prev).length > 0) {
          return prev; // Don't reinitialize if already set
        }
        
        const initialSettings = {};
        Object.entries(channelsData.channels).forEach(([channelName, channelInfo]) => {
          initialSettings[channelName] = {
            enabled: true,
            color: channelInfo.default_color,
            opacity: 0.8,
            auto_contrast: true,
          };
        });
        return initialSettings;
      });
      
    } catch (err) {
      showSnackbar(`Failed to load channels: ${utils.formatErrorMessage(err)}`, 'error');
    }
  }, []);

  // Check server connection on mount
  useEffect(() => {
    checkServerConnection();
  }, [checkServerConnection]);

  // Load channels when data is loaded
  useEffect(() => {
    if (dataLoaded) {
      loadAvailableChannels();
    }
  }, [dataLoaded, loadAvailableChannels]);

  const handleDataLoaded = () => {
    setDataLoaded(true);
    showSnackbar('Data loaded successfully', 'success');
  };

  const handleNucleusSelected = (nucleus) => {
    setSelectedNucleus(nucleus);
  };

  const handleChannelSettingsChange = (channelName, settings) => {
    setChannelSettings(prev => ({
      ...prev,
      [channelName]: {
        ...prev[channelName],
        ...settings,
      },
    }));
  };

  const handleEpitopeDataChange = (epitopeData) => {
    setCurrentEpitopeData(epitopeData);
  };

  const showSnackbar = (message, severity = 'info') => {
    setSnackbar({
      open: true,
      message,
      severity,
    });
  };

  const handleSnackbarClose = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  const handleRetryConnection = () => {
    setError(null);
    checkServerConnection();
  };

  // Render loading backdrop
  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Backdrop open={true} style={{ zIndex: 9999 }}>
          <Box display="flex" flexDirection="column" alignItems="center">
            <CircularProgress color="primary" size={60} />
            <Typography variant="h6" style={{ marginTop: 16, color: 'white' }}>
              Connecting to server...
            </Typography>
          </Box>
        </Backdrop>
      </ThemeProvider>
    );
  }

  // Render error state
  if (error && !serverConnected) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Container maxWidth="md" style={{ marginTop: 64 }}>
          <Alert 
            severity="error" 
            action={
              <button onClick={handleRetryConnection}>
                Retry
              </button>
            }
          >
            <Typography variant="h6">Connection Error</Typography>
            <Typography>{error}</Typography>
          </Alert>
        </Container>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      
      {/* App Bar */}
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Nuclei Viewer
          </Typography>
          {serverInfo && (
            <Typography variant="body2">
              v{serverInfo.version} | {Object.keys(availableChannels).length} channels
            </Typography>
          )}
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="xl" style={{ marginTop: 16, marginBottom: 16 }}>
        {!dataLoaded ? (
          // Data loading interface
          <DataLoader onDataLoaded={handleDataLoaded} />
        ) : (
          // Main application interface
          <Box>
            {/* Channel Controls */}
            <Box mb={2}>
              <ChannelControls
                channels={availableChannels}
                channelSettings={channelSettings}
                onChannelSettingsChange={handleChannelSettingsChange}
                epitopeData={currentEpitopeData}
              />
            </Box>

            {/* Main Content Area */}
            <Box display="flex" gap={2}>
              {/* Nuclei Browser */}
              <Box flex={selectedNucleus ? 1 : 2}>
                <NucleiBrowser
                  onNucleusSelected={handleNucleusSelected}
                  selectedNucleus={selectedNucleus}
                  channelSettings={channelSettings}
                />
              </Box>

              {/* Nucleus Detail View */}
              {selectedNucleus && (
                <Box flex={1}>
                  <NucleusDetail
                    nucleus={selectedNucleus}
                    channelSettings={channelSettings}
                    onClose={() => setSelectedNucleus(null)}
                    onEpitopeDataChange={handleEpitopeDataChange}
                  />
                </Box>
              )}
            </Box>
          </Box>
        )}
      </Container>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbar.severity}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App;
