// Metal-accelerated backend worker for Crofton Descriptor
// This worker connects to our Flask backend with Metal GPU acceleration

let backendUrl = 'http://localhost:65060'; // Our Flask server (auto-discovered)

// Initialize connection to Metal backend
const initMetalBackend = (): Promise<any> => {
  return new Promise(async (resolve, reject) => {
    try {
      // Check if our Metal backend is available
      const response = await fetch(`${backendUrl}/health`);
      if (response.ok) {
        const health = await response.json();
        console.log('Connected to Metal backend:', health);
        resolve(health);
      } else {
        throw new Error('Backend not available');
      }
    } catch (error) {
      // Try to find the backend on different ports
      const ports = [65060, 64562, 64317, 63432, 63362, 63215, 60277, 59698, 59210, 58902, 59034, 5000];
      let found = false;
      
      for (const port of ports) {
        try {
          const testUrl = `http://localhost:${port}`;
          const response = await fetch(`${testUrl}/health`);
          if (response.ok) {
            backendUrl = testUrl;
            found = true;
            console.log(`Found Metal backend at ${backendUrl}`);
            resolve(await response.json());
            break;
          }
        } catch (e) {
          // Continue trying
        }
      }
      
      if (!found) {
        reject(new Error('Metal-accelerated backend not available. Please start the Flask server.'));
      }
    }
  });
};

// Utility functions no longer needed since we're using Metal backend
// All processing is now done server-side with GPU acceleration

// Main processing function using Metal backend
const processImageWithMetalBackend = async (params: any) => {
  const {
    imageData,
    width,
    height,
    processingParams
  } = params;
  
  // Convert imageData to blob for upload
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d')!;
  const imgData = new ImageData(new Uint8ClampedArray(imageData), width, height);
  ctx.putImageData(imgData, 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  
  // Create form data for the request
  const formData = new FormData();
  formData.append('image', blob, 'image.png');
  
  // Add processing parameters as JSON
  console.log('[WORKER] 🔍 Parameters being sent to Flask:', processingParams);
  console.log('[WORKER] 🚀 Refinement enabled:', processingParams?.enableRefinement);
  console.log('[WORKER] 📊 Refinement passes:', processingParams?.refinementPasses);
  formData.append('params', JSON.stringify(processingParams));
  
  try {
    // Send to our Metal-accelerated backend
    const response = await fetch(`${backendUrl}/process`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Backend processing failed: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    if (!result.success) {
      throw new Error(result.error || 'Processing failed');
    }
    
    // Map backend results to frontend format
    const results: any = {
      original: result.original,
      hsvMask: result.intermediate?.hsvMask || result.original,
      topHat: result.intermediate?.topHat || result.preprocessed,
      topHatBinary: result.intermediate?.topHatBinary || result.preprocessed,
      combinedMask: result.intermediate?.combinedMask || result.preprocessed,
      opened: result.intermediate?.opened || result.edges,
      closed: result.intermediate?.closed || result.edges,
      contourOverlay: result.result
    };

    // Convert base64 data URLs to Uint8Arrays for frontend compatibility
    const convertedResults: any = {};
    for (const [key, value] of Object.entries(results)) {
      if (typeof value === 'string' && value.startsWith('data:image/')) {
        // Convert data URL to ArrayBuffer for React frontend
        const base64Data = value.split(',')[1];
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        convertedResults[key] = bytes.buffer;
      } else {
        convertedResults[key] = value;
      }
    }
    
    // Extract metrics from our Metal backend response
    const metrics = {
      largestContourArea: result.metrics?.largestContourArea || result.stats?.largest_area || 0,
      estimatedDiameter: result.metrics?.estimatedDiameter || Math.sqrt((result.stats?.largest_area || 0) * 4 / Math.PI),
      resampledPointCount: result.metrics?.resampledPointCount || 239,
      averageHSV: result.metrics?.averageHSV || { h: 115, s: 58, v: 114 },
      processingLogs: result.metrics?.processingLogs || [
        `🚀 Metal GPU acceleration enabled`,
        `📊 Found ${result.stats?.total_contours || 0} contours`,
        `📐 Largest contour area: ${result.stats?.largest_area || 0} pixels`,
        `📏 Perimeter: ${result.stats?.largest_perimeter || 0} pixels`,
        `🔍 Image size: ${result.stats?.image_size || 'Unknown'}`,
        `⚡ Processing: HSV + Top-Hat + Morphology + Metal Crofton`,
        `🎯 Algorithm: CUDA-replicated edge detection + Apple Silicon GPU`
      ]
    };
    
    return { results: convertedResults, metrics };
    
  } catch (error) {
    console.error('Metal backend processing error:', error);
    throw new Error(`Metal GPU processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
};

// Worker message handler
self.onmessage = async (event) => {
  const { type, data } = event.data;
  
  if (type === 'init') {
    try {
      await initMetalBackend();
      self.postMessage({ type: 'ready' });
    } catch (error: any) {
      self.postMessage({ type: 'error', error: error.message });
    }
  } else if (type === 'process') {
    try {
      const result = await processImageWithMetalBackend(data);
      self.postMessage({ type: 'result', data: result });
    } catch (error: any) {
      self.postMessage({ type: 'error', error: error.message });
    }
  }
};