import React, { useEffect, useRef } from 'react';
import { Play, Image as ImageIcon, Beaker } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Card } from '@/components/ui/card.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Dropzone } from '@/components/Dropzone.tsx';
import { PanelParams } from '@/components/PanelParams.tsx';
import { ResultCard } from '@/components/ResultCard.tsx';
import { MetricsPanel } from '@/components/MetricsPanel.tsx';
import { useAppStore } from '@/lib/store.ts';
import { useToast } from '@/hooks/use-toast.ts';

const Index = () => {
  const workerRef = useRef<Worker | null>(null);
  const { 
    selectedFile, 
    isProcessing, 
    setIsProcessing,
    processingStatus,
    setProcessingStatus,
    params,
    results,
    setResults,
    setMetrics,
    isOpenCVReady,
    setIsOpenCVReady
  } = useAppStore();
  const { toast } = useToast();

  // Initialize Web Worker and OpenCV
  useEffect(() => {
    workerRef.current = new Worker(new URL('../workers/opencvWorker.ts', import.meta.url), {
      type: 'module'
    });

    workerRef.current.onmessage = (event) => {
      const { type, data, error } = event.data;
      
      if (type === 'ready') {
        setIsOpenCVReady(true);
        setProcessingStatus('Metal GPU ready');
        toast({ title: "🚀 Metal GPU backend connected successfully!" });
      } else if (type === 'result') {
        // Convert results to blob URLs
        const resultUrls: any = {};
        Object.entries(data.results).forEach(([key, value]) => {
          const uint8Array = new Uint8Array(value as ArrayBuffer);
          const blob = new Blob([uint8Array], { type: 'image/png' });
          resultUrls[key] = URL.createObjectURL(blob);
        });
        
        setResults(resultUrls);
        setMetrics(data.metrics);
        setIsProcessing(false);
        setProcessingStatus('Processing complete');
        toast({ title: "Image processed successfully" });
      } else if (type === 'error') {
        setIsProcessing(false);
        setProcessingStatus('Error occurred');
        toast({
          title: "Processing failed",
          description: error,
          variant: "destructive"
        });
      }
    };

    // Initialize Metal backend
    setProcessingStatus('Connecting to Metal GPU...');
    workerRef.current.postMessage({ type: 'init' });

    return () => {
      workerRef.current?.terminate();
    };
  }, []);

  const handleProcess = async () => {
    if (!selectedFile || !isOpenCVReady) return;

    setIsProcessing(true);
    setProcessingStatus('Processing image...');
    setResults(null);
    setMetrics(null);

    try {
      const arrayBuffer = await selectedFile.arrayBuffer();
      const img = new Image();
      
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        
        console.log('[REACT] 🔍 Current parameters from store:', params);
        console.log('[REACT] 🚀 Refinement enabled:', params.enableRefinement);
        console.log('[REACT] 📊 Refinement passes:', params.refinementPasses);
        
        workerRef.current?.postMessage({
          type: 'process',
          data: {
            imageData: imageData.data.buffer,
            width: img.width,
            height: img.height,
            processingParams: params
          }
        });
      };
      
      img.src = URL.createObjectURL(selectedFile);
    } catch (error) {
      setIsProcessing(false);
      setProcessingStatus('Error reading file');
      toast({
        title: "Failed to read image",
        description: "Please try a different image file.",
        variant: "destructive"
      });
    }
  };

  const resultStages = [
    { key: 'original', title: 'Original' },
    { key: 'hsvMask', title: 'HSV Mask' },
    { key: 'topHat', title: 'Top-Hat' },
    { key: 'topHatBinary', title: 'Top-Hat Binary' },
    { key: 'combinedMask', title: 'Combined Mask' },
    { key: 'opened', title: 'Opened' },
    { key: 'closed', title: 'Closed' },
    { key: 'contourOverlay', title: 'Contour Overlay' }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary rounded-lg">
                <Beaker className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-semibold">Metal-Accelerated Edge Detection</h1>
                <p className="text-sm text-muted-foreground">
                  Apple Silicon GPU + CUDA-replicated algorithm
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Badge variant={isOpenCVReady ? "default" : "secondary"}>
                {processingStatus || 'Connecting to Metal GPU...'}
              </Badge>
              {isOpenCVReady && (
                <Badge variant="outline" className="text-green-600">
                  ⚡ GPU Ready
                </Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Left column - Upload and process */}
          <div className="lg:col-span-1 space-y-6">
            <Dropzone />
            
            <Button
              onClick={handleProcess}
              disabled={!selectedFile || !isOpenCVReady || isProcessing}
              className="w-full"
              size="lg"
            >
              <Play className="h-4 w-4 mr-2" />
              {isProcessing ? 'Processing...' : 'Process Image'}
            </Button>
            
            <PanelParams />
          </div>

          {/* Middle column - Results */}
          <div className="lg:col-span-2">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <ImageIcon className="h-5 w-5" />
              Processing Results
            </h2>
            
            {results ? (
              <div className="grid grid-cols-2 gap-4">
                {resultStages.map((stage) => (
                  <ResultCard
                    key={stage.key}
                    title={stage.title}
                    imageUrl={results[stage.key as keyof typeof results]}
                  />
                ))}
              </div>
            ) : (
              <Card className="aspect-video flex items-center justify-center shadow-soft">
                <div className="text-center text-muted-foreground">
                  <ImageIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Upload and process an image to see results</p>
                </div>
              </Card>
            )}
          </div>

          {/* Right column - Metrics */}
          <div className="lg:col-span-1">
            <MetricsPanel />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;