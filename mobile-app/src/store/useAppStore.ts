import { create } from 'zustand';
import axios from 'axios';

interface Settings {
  use_dnn: boolean;
  use_harris: boolean;
  use_cylindrical: boolean;
  focal_length: number;
  resize: number;
  [key: string]: string | number | boolean;
}

interface Panorama {
  id: number;
  filename: string;
  path: string;
  shape: number[];
}

interface AppState {
  apiBaseUrl: string;
  settings: Settings;
  
  isConnected: boolean;
  isCheckingConnection: boolean;
  
  currentSessionId: string | null;
  jobStatus: 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
  panoramas: Panorama[];
  errorMessage: string | null;

  setApiBaseUrl: (url: string) => void;
  updateSettings: (newSettings: Partial<Settings>) => void;
  checkConnection: () => Promise<boolean>;
  startStitching: (imageUris: string[]) => Promise<void>;
  resetSession: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  apiBaseUrl: 'http://localhost:5000/api', 
  settings: {
    use_dnn: false,
    use_harris: false,
    use_cylindrical: true,
    focal_length: 1200.0,
    resize: 800,
  },
  
  // Session State
  isConnected: false,
  isCheckingConnection: false,
  
  // Job Status
  currentSessionId: null,
  jobStatus: 'idle',
  panoramas: [],
  errorMessage: null,

  // Actions
  setApiBaseUrl: (url) => set({ apiBaseUrl: url }),
  updateSettings: (newSettings) => set((state) => ({
    settings: { ...state.settings, ...newSettings }
  })),

  checkConnection: async () => {
    const { apiBaseUrl } = get();
    set({ isCheckingConnection: true });
    try {
      const response = await axios.get(`${apiBaseUrl}/health`, { timeout: 2000 });
      if (response.status === 200) {
        set({ isConnected: true, isCheckingConnection: false });
        return true;
      }
    } catch (error: any) {
      console.log('Connection check failed:', error.message);
    }
    set({ isConnected: false, isCheckingConnection: false });
    return false;
  },

  startStitching: async (imageUris) => {
    const { apiBaseUrl, settings } = get();
    set({ jobStatus: 'uploading', errorMessage: null, currentSessionId: null, panoramas: [] });

    try {
      const formData = new FormData();
      
      // Add images
      imageUris.forEach((uri) => {
        const filename = uri.split('/').pop() || 'image.jpg';
        const match = /\.(\w+)$/.exec(filename);
        const type = match ? `image/${match[1]}` : 'image/jpeg';
        
        // React Native FormData expects an object with uri, name, type
        formData.append('files', {
          uri,
          name: filename,
          type,
        } as any);
      });

      // Add settings
      Object.keys(settings).forEach(key => {
        formData.append(key, String(settings[key]));
      });

      set({ jobStatus: 'processing' });
      
      const response = await axios.post(`${apiBaseUrl}/stitch`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 600000, // 10 minutes
      });

      if (response.data.status === 'success') {
        set({
          jobStatus: 'completed',
          currentSessionId: response.data.session_id,
          panoramas: response.data.panoramas,
        });
      } else {
        set({ jobStatus: 'error', errorMessage: 'Stitching failed without error detail' });
      }

    } catch (error: any) {
      const msg = error.response?.data?.message || error.message || 'Unknown error occurred';
      set({ jobStatus: 'error', errorMessage: msg });
    }
  },

  resetSession: () => set({ jobStatus: 'idle', currentSessionId: null, panoramas: [], errorMessage: null }),
}));
