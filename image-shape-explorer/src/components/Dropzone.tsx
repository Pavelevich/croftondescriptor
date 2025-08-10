import React, { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, X } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Card } from '@/components/ui/card.tsx';
import { useAppStore } from '@/lib/store.ts';
import { useToast } from '@/hooks/use-toast.ts';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

export const Dropzone: React.FC = () => {
  const [isDragOver, setIsDragOver] = useState(false);
  const { selectedFile, setSelectedFile } = useAppStore();
  const { toast } = useToast();

  const validateFile = (file: File): boolean => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      toast({
        title: "Invalid file type",
        description: "Please select a JPEG, PNG, or WebP image.",
        variant: "destructive",
      });
      return false;
    }

    if (file.size > MAX_FILE_SIZE) {
      toast({
        title: "File too large",
        description: "Please select an image smaller than 10MB.",
        variant: "destructive",
      });
      return false;
    }

    return true;
  };

  const handleFileSelect = useCallback((file: File) => {
    if (validateFile(file)) {
      setSelectedFile(file);
      toast({
        title: "Image loaded",
        description: `Selected ${file.name}`,
      });
    }
  }, [setSelectedFile, toast]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleRemoveFile = useCallback(() => {
    setSelectedFile(null);
  }, [setSelectedFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  if (selectedFile) {
    return (
      <Card className="relative overflow-hidden shadow-soft">
        <div className="aspect-video w-full bg-muted relative">
          <img
            src={URL.createObjectURL(selectedFile)}
            alt="Selected image"
            className="w-full h-full object-contain"
          />
          <Button
            variant="destructive"
            size="sm"
            className="absolute top-2 right-2 h-8 w-8 p-0"
            onClick={handleRemoveFile}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="p-4">
          <p className="text-sm font-medium truncate">{selectedFile.name}</p>
          <p className="text-xs text-muted-foreground mt-1">
            {(selectedFile.size / 1024 / 1024).toFixed(1)} MB
          </p>
        </div>
      </Card>
    );
  }

  return (
    <Card
      className={`
        relative overflow-hidden shadow-soft border-2 border-dashed transition-all duration-200
        ${isDragOver 
          ? 'border-primary bg-primary-muted' 
          : 'border-border hover:border-primary/50'
        }
      `}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <div className="aspect-video w-full flex flex-col items-center justify-center p-8 text-center">
        <div className={`
          rounded-full p-4 mb-4 transition-colors duration-200
          ${isDragOver ? 'bg-primary text-primary-foreground' : 'bg-muted'}
        `}>
          {isDragOver ? (
            <Upload className="h-8 w-8" />
          ) : (
            <ImageIcon className="h-8 w-8" />
          )}
        </div>
        
        <h3 className="text-lg font-semibold mb-2">
          {isDragOver ? 'Drop your image here' : 'Select an image'}
        </h3>
        
        <p className="text-sm text-muted-foreground mb-4 max-w-xs">
          Drag and drop an image file, or click to browse
        </p>
        
        <div className="space-y-2">
          <Button asChild>
            <label htmlFor="file-input" className="cursor-pointer">
              Choose Image
            </label>
          </Button>
          
          <input
            id="file-input"
            type="file"
            accept={ACCEPTED_TYPES.join(',')}
            onChange={handleFileInput}
            className="hidden"
          />
          
          <p className="text-xs text-muted-foreground">
            JPEG, PNG, WebP • Max 10MB
          </p>
        </div>
      </div>
    </Card>
  );
};