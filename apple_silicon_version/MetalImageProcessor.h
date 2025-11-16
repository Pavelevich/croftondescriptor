//
//  MetalImageProcessor.h
//  Crofton Descriptor - Metal Optimized Image Processing
//
//  High-performance GPU-accelerated image processing for cell boundary detection
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#endif

NS_ASSUME_NONNULL_BEGIN

/// Processing parameters for image preprocessing pipeline
@interface ProcessingParams : NSObject

// HSV range parameters
@property (nonatomic, assign) int hueMin;
@property (nonatomic, assign) int hueMax;
@property (nonatomic, assign) int satMin;
@property (nonatomic, assign) int valMin;

// Morphology kernel sizes
@property (nonatomic, assign) int topHatKernelSize;
@property (nonatomic, assign) int openSize;
@property (nonatomic, assign) int closeSize;

// Threshold parameters
@property (nonatomic, assign) BOOL useOtsu;
@property (nonatomic, assign) float manualThreshold;

// Edge detection parameters
@property (nonatomic, assign) float sobelScale3x3;
@property (nonatomic, assign) float sobelScale5x5;
@property (nonatomic, assign) float edgeThreshold;

/// Create default parameters matching CUDA algorithm
+ (instancetype)defaultParams;

@end

/// Result of image processing pipeline
@interface ProcessingResult : NSObject

#ifdef __cplusplus
@property (nonatomic, assign) cv::Mat finalMask;
@property (nonatomic, assign) cv::Mat hsvMask;
@property (nonatomic, assign) cv::Mat topHat;
@property (nonatomic, assign) cv::Mat topHatBinary;
@property (nonatomic, assign) cv::Mat combinedMask;
@property (nonatomic, assign) cv::Mat opened;
@property (nonatomic, assign) cv::Mat closed;
#endif

@property (nonatomic, assign) double gpuTimeMs;
@property (nonatomic, assign) BOOL success;
@property (nonatomic, strong, nullable) NSString *errorMessage;

@end

/// Main Metal-accelerated image processor
@interface MetalImageProcessor : NSObject

@property (nonatomic, strong, readonly) id<MTLDevice> device;
@property (nonatomic, strong, readonly) id<MTLCommandQueue> commandQueue;

// Metal Performance Shaders
@property (nonatomic, strong, readonly) MPSImageSobel *sobelFilter;
@property (nonatomic, strong, readonly) MPSImageGaussianBlur *blurFilter;

/// Initialize with Metal device (uses system default if nil)
- (nullable instancetype)initWithDevice:(nullable id<MTLDevice>)device;

#ifdef __cplusplus
/// Process image with given parameters (main entry point)
- (ProcessingResult*)processImage:(const cv::Mat&)inputImage
                       withParams:(ProcessingParams*)params;

/// Compute Otsu threshold for grayscale image
- (float)computeOtsuThreshold:(const cv::Mat&)grayImage;
#endif

/// Log processing pipeline statistics
- (void)logStatistics;

@end

NS_ASSUME_NONNULL_END
