import React from 'react';
import { RotateCcw, HelpCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Slider } from '@/components/ui/slider.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip.tsx';
import { Separator } from '@/components/ui/separator.tsx';
import { useAppStore } from '@/lib/store.ts';

export const PanelParams: React.FC = () => {
  const { params, setParams, resetParams } = useAppStore();

  const ParamSection: React.FC<{ 
    title: string; 
    children: React.ReactNode;
    tooltip?: string;
  }> = ({ title, children, tooltip }) => (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <h4 className="text-sm font-medium">{title}</h4>
        {tooltip && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <HelpCircle className="h-3 w-3 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs text-sm">{tooltip}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
      {children}
    </div>
  );

  const SliderParam: React.FC<{
    label: string;
    value: number;
    min: number;
    max: number;
    step?: number;
    onChange: (value: number) => void;
    tooltip?: string;
  }> = ({ label, value, min, max, step = 1, onChange, tooltip }) => (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Label className="text-xs">{label}</Label>
          {tooltip && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3 w-3 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">{tooltip}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <span className="text-xs font-mono bg-muted px-2 py-1 rounded">
          {value}
        </span>
      </div>
      <Slider
        value={[value]}
        onValueChange={(values) => onChange(values[0])}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
    </div>
  );

  return (
    <Card className="h-fit shadow-soft bg-muted/50">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg text-muted-foreground">Parameters</CardTitle>
          <Button
            variant="outline"
            size="sm"
            disabled
            className="h-8 px-3 opacity-50"
          >
            <RotateCcw className="h-3 w-3 mr-1" />
            Reset
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6">
        <div className="flex items-center justify-center py-12">
          <div className="text-center space-y-2">
            <div className="text-2xl">🚧</div>
            <div className="text-sm text-muted-foreground font-medium">Coming Soon</div>
            <div className="text-xs text-muted-foreground">Parameter adjustments will be available soon</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};