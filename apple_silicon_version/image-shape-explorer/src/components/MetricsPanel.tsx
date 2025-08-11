import React, { useState } from 'react';
import { Download, ChevronDown, ChevronRight, Activity, Target, Hash, Palette, RefreshCw, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible.tsx';
import { Separator } from '@/components/ui/separator.tsx';
import { useAppStore } from '@/lib/store.ts';

export const MetricsPanel: React.FC = () => {
  const { metrics, params } = useAppStore();
  const [isLogsOpen, setIsLogsOpen] = useState(false);

  if (!metrics) {
    return (
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground text-center py-8">
            Process an image to see metrics
          </p>
        </CardContent>
      </Card>
    );
  }

  const handleDownloadJson = () => {
    const data = {
      metrics,
      parameters: params,
      timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `image-analysis-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const MetricItem: React.FC<{
    icon: React.ReactNode;
    label: string;
    value: string | number;
    unit?: string;
    description?: string;
  }> = ({ icon, label, value, unit, description }) => (
    <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
      <div className="flex items-center gap-3">
        <div className="text-primary">
          {icon}
        </div>
        <div>
          <p className="text-sm font-medium">{label}</p>
          {description && (
            <p className="text-xs text-muted-foreground">{description}</p>
          )}
        </div>
      </div>
      <div className="text-right">
        <p className="text-sm font-mono">
          {typeof value === 'number' ? value.toFixed(1) : value}
          {unit && <span className="text-muted-foreground ml-1">{unit}</span>}
        </p>
      </div>
    </div>
  );

  return (
    <Card className="shadow-soft">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Analysis Results
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={handleDownloadJson}
            className="h-8 px-3"
          >
            <Download className="h-3 w-3 mr-1" />
            JSON
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Main metrics */}
        <div className="space-y-3">
          <MetricItem
            icon={<Target className="h-4 w-4" />}
            label="Largest Contour Area"
            value={metrics.largestContourArea}
            unit="px²"
            description="Area of the detected object"
          />
          
          <MetricItem
            icon={<Hash className="h-4 w-4" />}
            label="Estimated Diameter"
            value={metrics.estimatedDiameter}
            unit="px"
            description="Maximum distance across contour points"
          />
          
          <MetricItem
            icon={<Target className="h-4 w-4" />}
            label="Resampled Points"
            value={metrics.resampledPointCount}
            description="Contour points after uniform resampling"
          />
        </div>

        <Separator />

        {/* HSV averages */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium flex items-center gap-2">
            <Palette className="h-4 w-4" />
            Average HSV Along Contour
          </h4>
          
          <div className="grid grid-cols-3 gap-2">
            <div className="text-center p-2 bg-muted/50 rounded">
              <p className="text-xs text-muted-foreground">Hue</p>
              <p className="text-sm font-mono">
                {metrics.averageHSV.h.toFixed(0)}°
              </p>
            </div>
            <div className="text-center p-2 bg-muted/50 rounded">
              <p className="text-xs text-muted-foreground">Saturation</p>
              <p className="text-sm font-mono">
                {metrics.averageHSV.s.toFixed(0)}%
              </p>
            </div>
            <div className="text-center p-2 bg-muted/50 rounded">
              <p className="text-xs text-muted-foreground">Value</p>
              <p className="text-sm font-mono">
                {metrics.averageHSV.v.toFixed(0)}%
              </p>
            </div>
          </div>
        </div>

        <Separator />

        {/* Refinement metrics if available */}
        {metrics.refinementPasses && metrics.refinementPasses > 1 && (
          <>
            <div className="space-y-3">
              <h4 className="text-sm font-medium flex items-center gap-2">
                <RefreshCw className="h-4 w-4" />
                Refinement Analysis
              </h4>
              
              <MetricItem
                icon={<RefreshCw className="h-4 w-4" />}
                label="Refinement Passes"
                value={metrics.refinementPasses}
                description="Total processing iterations"
              />
              
              {metrics.finalQualityScore !== undefined && (
                <MetricItem
                  icon={<TrendingUp className="h-4 w-4" />}
                  label="Final Quality Score"
                  value={metrics.finalQualityScore}
                  description="Edge quality assessment (0-1)"
                />
              )}
              
              {metrics.passScores && metrics.passScores.length > 1 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium">Quality Score by Pass:</p>
                  <div className="flex flex-wrap gap-2">
                    {metrics.passScores.map((score, index) => (
                      <Badge
                        key={index}
                        variant={index === metrics.passScores!.indexOf(Math.max(...metrics.passScores!)) ? "default" : "secondary"}
                        className="text-xs"
                      >
                        {index + 1}: {score.toFixed(3)}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Best pass highlighted
                  </p>
                </div>
              )}
            </div>

            <Separator />
          </>
        )}

        {/* Processing logs */}
        <Collapsible open={isLogsOpen} onOpenChange={setIsLogsOpen}>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="w-full justify-between p-0 h-auto">
              <span className="text-sm font-medium">Processing Logs</span>
              {isLogsOpen ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </Button>
          </CollapsibleTrigger>
          
          <CollapsibleContent className="mt-3">
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {metrics.processingLogs.map((log, index) => (
                <div key={index} className="text-xs font-mono bg-muted p-2 rounded">
                  {log}
                </div>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
};