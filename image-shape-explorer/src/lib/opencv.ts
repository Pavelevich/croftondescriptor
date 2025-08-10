// OpenCV.js loader and utilities
let cvPromise: Promise<any> | null = null;

export const loadOpenCV = (): Promise<any> => {
  if (cvPromise) {
    return cvPromise;
  }

  cvPromise = new Promise((resolve, reject) => {
    // Check if OpenCV is already loaded
    if (typeof window !== 'undefined' && (window as any).cv) {
      resolve((window as any).cv);
      return;
    }

    // Create script element
    const script = document.createElement('script');
    script.src = 'https://docs.opencv.org/4.8.0/opencv.js';
    script.async = true;
    
    // Set up global callback
    (window as any).onOpenCVReady = () => {
      resolve((window as any).cv);
    };

    script.onerror = () => {
      reject(new Error('Failed to load OpenCV.js'));
    };

    document.body.appendChild(script);
  });

  return cvPromise;
};

export const isOpenCVReady = (): boolean => {
  return typeof window !== 'undefined' && 
         (window as any).cv && 
         (window as any).cv.Mat;
};

// Utility functions for OpenCV operations
export const createKernel = (cv: any, size: number, shape: 'ellipse' | 'rect') => {
  if (shape === 'ellipse') {
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size));
  } else {
    return cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(size, size));
  }
};

export const matToImageData = (cv: any, mat: any): ImageData => {
  const canvas = document.createElement('canvas');
  canvas.width = mat.cols;
  canvas.height = mat.rows;
  const ctx = canvas.getContext('2d')!;
  
  cv.imshow(canvas, mat);
  
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
};

export const imageDataToBlob = (imageData: ImageData): Promise<Blob> => {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext('2d')!;
    
    ctx.putImageData(imageData, 0, 0);
    
    canvas.toBlob((blob) => {
      resolve(blob!);
    }, 'image/png');
  });
};

export const blobToDataURL = (blob: Blob): Promise<string> => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.readAsDataURL(blob);
  });
};