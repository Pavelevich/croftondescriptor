import React, { useState } from 'react';
import { Download, Maximize2, X } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';

interface ResultCardProps {
  title: string;
  imageUrl: string;
  resolution?: string;
  onDownload?: () => void;
}

export const ResultCard: React.FC<ResultCardProps> = ({
  title,
  imageUrl,
  resolution,
  onDownload
}) => {
  const [isFullscreenOpen, setIsFullscreenOpen] = useState(false);

  const handleDownload = () => {
    if (onDownload) {
      onDownload();
      return;
    }

    // Default download behavior
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = `${title.toLowerCase().replace(/\s+/g, '-')}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <>
      <Card className="overflow-hidden shadow-soft hover:shadow-medium transition-shadow duration-200">
        <div className="relative group">
          <div className="aspect-square bg-muted">
            <img
              src={imageUrl}
              alt={title}
              className="w-full h-full object-contain"
            />
          </div>
          
          {/* Overlay controls */}
          <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center">
            <div className="flex gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setIsFullscreenOpen(true)}
                className="bg-white/20 hover:bg-white/30 backdrop-blur text-white border-white/20"
              >
                <Maximize2 className="h-4 w-4" />
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleDownload}
                className="bg-white/20 hover:bg-white/30 backdrop-blur text-white border-white/20"
              >
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
        
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-sm">{title}</h3>
              {resolution && (
                <Badge variant="secondary" className="mt-1 text-xs">
                  {resolution}
                </Badge>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Fullscreen dialog */}
      <Dialog open={isFullscreenOpen} onOpenChange={setIsFullscreenOpen}>
        <DialogContent className="max-w-7xl max-h-screen h-[90vh] p-0">
          <DialogHeader className="p-6 pb-0">
            <div className="flex items-center justify-between">
              <DialogTitle>{title}</DialogTitle>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDownload}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download PNG
                </Button>
              </div>
            </div>
          </DialogHeader>
          
          <div className="flex-1 p-6 pt-0">
            <div className="w-full h-full bg-muted rounded-lg overflow-hidden">
              <img
                src={imageUrl}
                alt={title}
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};