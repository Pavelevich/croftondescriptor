import { create } from 'zustand';

export interface ProcessingParams {
  // HSV range
  hueMin: number;
  hueMax: number;
  satMin: number;
  valMin: number;
  
  // Top-Hat
  topHatKernelSize: number;
  topHatKernelShape: 'ellipse' | 'rect';
  
  // Threshold
  useOtsu: boolean;
  manualThreshold: number;
  
  // Morphology
  openSize: number;
  closeSize: number;
  
  // Optional features
  enableCanny: boolean;
  cannyLow: number;
  cannyHigh: number;
  
  // Iterative refinement
  enableRefinement: boolean;
  refinementPasses: number;
  qualityThreshold: number;
  parameterScheduling: 'linear' | 'exponential' | 'adaptive';
}

export interface ProcessingResult {
  original: string;
  hsvMask: string;
  topHat: string;
  topHatBinary: string;
  combinedMask: string;
  opened: string;
  closed: string;
  contourOverlay: string;
}

export interface ProcessingMetrics {
  largestContourArea: number;
  estimatedDiameter: number;
  resampledPointCount: number;
  averageHSV: { h: number; s: number; v: number };
  processingLogs: string[];
  refinementPasses?: number;
  finalQualityScore?: number;
  passScores?: number[];
}

interface AppState {
  // File state
  selectedFile: File | null;
  setSelectedFile: (file: File | null) => void;
  
  // Processing state
  isProcessing: boolean;
  setIsProcessing: (processing: boolean) => void;
  
  processingStatus: string;
  setProcessingStatus: (status: string) => void;
  
  // Parameters
  params: ProcessingParams;
  setParams: (params: Partial<ProcessingParams>) => void;
  resetParams: () => void;
  
  // Results
  results: ProcessingResult | null;
  setResults: (results: ProcessingResult | null) => void;
  
  metrics: ProcessingMetrics | null;
  setMetrics: (metrics: ProcessingMetrics | null) => void;
  
  // OpenCV state
  isOpenCVReady: boolean;
  setIsOpenCVReady: (ready: boolean) => void;
}

const defaultParams: ProcessingParams = {
  hueMin: 100,
  hueMax: 180,
  satMin: 20,
  valMin: 20,
  topHatKernelSize: 15,
  topHatKernelShape: 'ellipse',
  useOtsu: true,
  manualThreshold: 127,
  openSize: 3,
  closeSize: 5,
  enableCanny: false,
  cannyLow: 50,
  cannyHigh: 150,
  enableRefinement: true,  // HABILITADO POR DEFECTO
  refinementPasses: 3,
  qualityThreshold: 0.75,  // Más alcanzable
  parameterScheduling: 'adaptive',
};

export const useAppStore = create<AppState>((set) => ({
  // File state
  selectedFile: null,
  setSelectedFile: (file) => set({ selectedFile: file }),
  
  // Processing state
  isProcessing: false,
  setIsProcessing: (processing) => set({ isProcessing: processing }),
  
  processingStatus: '',
  setProcessingStatus: (status) => set({ processingStatus: status }),
  
  // Parameters
  params: defaultParams,
  setParams: (newParams) => set((state) => ({ 
    params: { ...state.params, ...newParams } 
  })),
  resetParams: () => set({ params: defaultParams }),
  
  // Results
  results: null,
  setResults: (results) => set({ results }),
  
  metrics: null,
  setMetrics: (metrics) => set({ metrics }),
  
  // OpenCV state
  isOpenCVReady: false,
  setIsOpenCVReady: (ready) => set({ isOpenCVReady: ready }),
}));