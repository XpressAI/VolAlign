/**
 * API service for communicating with the Nuclei Viewer backend
 */

import axios from 'axios';

// Create axios instance with default configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000, // 30 second timeout for large data operations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging (only in development)
api.interceptors.request.use(
  (config) => {
    if (process.env.NODE_ENV === 'development') {
      // Only log non-health check requests to reduce noise
      if (!config.url?.includes('/health') && !config.url?.includes('/api/config/')) {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
      }
    }
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      console.error(`HTTP ${status}:`, data);
      
      // Handle specific error cases
      if (status === 404) {
        throw new Error(data.detail || 'Resource not found');
      } else if (status === 500) {
        throw new Error(data.detail || 'Internal server error');
      } else {
        throw new Error(data.detail || `HTTP ${status} error`);
      }
    } else if (error.request) {
      // Network error
      throw new Error('Network error - unable to connect to server');
    } else {
      // Other error
      throw new Error(error.message || 'Unknown error occurred');
    }
  }
);

// Data API endpoints
export const dataAPI = {
  // Discover available datasets
  discoverDatasets: async (basePath = null) => {
    const params = basePath ? { base_path: basePath } : {};
    const response = await api.get('/api/data/discover', { params });
    return response.data;
  },

  // Load all datasets automatically
  loadAllDatasets: async (basePath = null) => {
    const params = basePath ? { base_path: basePath } : {};
    const response = await api.post('/api/data/load-all', null, { params });
    return response.data;
  },

  // Load specific datasets
  loadDatasets: async (loadRequest) => {
    const response = await api.post('/api/data/load', loadRequest);
    return response.data;
  },

  // Get data loading status
  getDataStatus: async () => {
    const response = await api.get('/api/data/status');
    return response.data;
  },

  // Get available channels
  getChannels: async () => {
    const response = await api.get('/api/data/channels');
    return response.data;
  },

  // Clear all data
  clearData: async () => {
    const response = await api.delete('/api/data/clear');
    return response.data;
  },
};

// Nuclei API endpoints
export const nucleiAPI = {
  // Extract nuclei information
  extractNuclei: async (forceReload = false) => {
    const params = { force_reload: forceReload };
    const response = await api.get('/api/nuclei/extract', { params });
    return response.data;
  },

  // Get paginated list of nuclei
  listNuclei: async (page = 0, pageSize = null) => {
    const params = { page };
    if (pageSize !== null) params.page_size = pageSize;
    const response = await api.get('/api/nuclei/list', { params });
    return response.data;
  },

  // Get nucleus summary
  getNucleusSummary: async (nucleusLabel) => {
    const response = await api.get(`/api/nuclei/${nucleusLabel}/summary`);
    return response.data;
  },

  // Compute MIP for a nucleus
  computeMIP: async (mipRequest) => {
    const response = await api.post('/api/nuclei/mip', mipRequest);
    return response.data;
  },

  // Compute batch MIPs
  computeBatchMIPs: async (nucleusLabels, channels = null, returnIndividual = true) => {
    const params = { return_individual: returnIndividual };
    const requestBody = {
      nucleus_labels: nucleusLabels,
      channels: channels
    };
    const response = await api.post('/api/nuclei/batch-mip', requestBody, { params });
    return response.data;
  },

  // Get nuclei statistics
  getNucleiStats: async () => {
    const response = await api.get('/api/nuclei/stats');
    return response.data;
  },

  // Get cache information
  getCacheInfo: async () => {
    const response = await api.get('/api/nuclei/cache/info');
    return response.data;
  },

  // Clear cache
  clearCache: async () => {
    const response = await api.delete('/api/nuclei/cache/clear');
    return response.data;
  },
};

// Configuration API endpoints
export const configAPI = {
  // Get current configuration
  getConfig: async () => {
    const response = await api.get('/api/config/');
    return response.data;
  },

  // Update configuration
  updateConfig: async (configUpdate) => {
    const response = await api.put('/api/config/', configUpdate);
    return response.data;
  },

  // Save configuration
  saveConfig: async (configPath = null) => {
    const params = configPath ? { config_path: configPath } : {};
    const response = await api.post('/api/config/save', null, { params });
    return response.data;
  },

  // Get data paths configuration
  getDataPaths: async () => {
    const response = await api.get('/api/config/data-paths');
    return response.data;
  },

  // Update base path
  updateBasePath: async (basePath) => {
    const response = await api.put('/api/config/data-paths/base-path', null, {
      params: { base_path: basePath }
    });
    return response.data;
  },

  // Add epitope channel
  addEpitopeChannel: async (channelConfig) => {
    const response = await api.post('/api/config/epitope-channels', channelConfig);
    return response.data;
  },

  // Update epitope channel
  updateEpitopeChannel: async (channelName, channelConfig) => {
    const response = await api.put(`/api/config/epitope-channels/${channelName}`, channelConfig);
    return response.data;
  },

  // Remove epitope channel
  removeEpitopeChannel: async (channelName) => {
    const response = await api.delete(`/api/config/epitope-channels/${channelName}`);
    return response.data;
  },

  // Get processing configuration
  getProcessingConfig: async () => {
    const response = await api.get('/api/config/processing');
    return response.data;
  },

  // Update processing configuration
  updateProcessingConfig: async (processingParams) => {
    const response = await api.put('/api/config/processing', null, { params: processingParams });
    return response.data;
  },

  // Get UI configuration
  getUIConfig: async () => {
    const response = await api.get('/api/config/ui');
    return response.data;
  },

  // Update UI configuration
  updateUIConfig: async (uiParams) => {
    const response = await api.put('/api/config/ui', null, { params: uiParams });
    return response.data;
  },
};

// Health check and server info
export const serverAPI = {
  // Get server info
  getServerInfo: async () => {
    const response = await api.get('/');
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

// Utility functions
export const utils = {
  // Check if server is reachable
  checkServerConnection: async () => {
    try {
      await serverAPI.healthCheck();
      return true;
    } catch (error) {
      console.error('Server connection check failed:', error);
      return false;
    }
  },

  // Format error message for display
  formatErrorMessage: (error) => {
    if (typeof error === 'string') {
      return error;
    } else if (error.message) {
      return error.message;
    } else {
      return 'An unknown error occurred';
    }
  },

  // Convert base64 image to blob URL
  base64ToBlob: (base64String, mimeType = 'image/png') => {
    const byteCharacters = atob(base64String.split(',')[1]);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: mimeType });
    return URL.createObjectURL(blob);
  },

  // Cleanup blob URL
  cleanupBlobUrl: (url) => {
    if (url && url.startsWith('blob:')) {
      URL.revokeObjectURL(url);
    }
  },
};

export default api;