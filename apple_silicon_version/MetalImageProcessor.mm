//
//  MetalImageProcessor.mm
//  Crofton Descriptor - Metal Optimized Image Processing
//

#import "MetalImageProcessor.h"
#import "MetalCapabilities.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

// ============================================================================
// MARK: - ProcessingParams Implementation
// ============================================================================

@implementation ProcessingParams

+ (instancetype)defaultParams {
    ProcessingParams *params = [[ProcessingParams alloc] init];

    // HSV range (matching CUDA algorithm)
    params.hueMin = 100;
    params.hueMax = 180;
    params.satMin = 20;
    params.valMin = 20;

    // Morphology kernels
    params.topHatKernelSize = 15;
    params.openSize = 3;
    params.closeSize = 5;

    // Threshold
    params.useOtsu = YES;
    params.manualThreshold = 0.5;

    // Edge detection
    params.sobelScale3x3 = 0.7;
    params.sobelScale5x5 = 0.3;
    params.edgeThreshold = 0.1;

    return params;
}

@end

// ============================================================================
// MARK: - ProcessingResult Implementation
// ============================================================================

@implementation ProcessingResult

- (instancetype)init {
    self = [super init];
    if (self) {
        _success = NO;
        _gpuTimeMs = 0.0;
    }
    return self;
}

@end

// ============================================================================
// MARK: - MetalImageProcessor Implementation
// ============================================================================

@interface MetalImageProcessor ()

// Custom compute pipeline states
@property (nonatomic, strong) id<MTLComputePipelineState> bgrToGrayPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> bgrToHSVPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> hsvMaskPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> subtractPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> bitwiseOrPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> thresholdPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> histogramPipeline;

// MPS morphology filters (cached for different sizes)
@property (nonatomic, strong) NSMutableDictionary<NSNumber*, MPSImageDilate*> *dilateFilters;
@property (nonatomic, strong) NSMutableDictionary<NSNumber*, MPSImageErode*> *erodeFilters;

// Statistics
@property (nonatomic, assign) NSUInteger totalProcessedImages;
@property (nonatomic, assign) double totalGPUTime;

@end

@implementation MetalImageProcessor

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device ?: [MetalCapabilities sharedDevice];

        if (!_device) {
            NSLog(@"❌ Metal device not available");
            return nil;
        }

        _commandQueue = [_device newCommandQueue];
        _dilateFilters = [NSMutableDictionary dictionary];
        _erodeFilters = [NSMutableDictionary dictionary];

        // Initialize MPS filters
        _sobelFilter = [[MPSImageSobel alloc] initWithDevice:_device];
        _blurFilter = [[MPSImageGaussianBlur alloc] initWithDevice:_device sigma:1.0];

        // Compile custom shaders
        if (![self compileShaders]) {
            NSLog(@"❌ Failed to compile Metal shaders");
            return nil;
        }

        NSLog(@"✅ MetalImageProcessor initialized successfully");
        [MetalCapabilities logCapabilities];
    }
    return self;
}

- (BOOL)compileShaders {
    NSError *error = nil;

    // Try to load from compiled metallib first (production)
    NSString *libraryPath = [[NSBundle mainBundle] pathForResource:@"image_processing" ofType:@"metallib"];
    id<MTLLibrary> library = nil;

    if (libraryPath && [[NSFileManager defaultManager] fileExistsAtPath:libraryPath]) {
        library = [_device newLibraryWithFile:libraryPath error:&error];
        if (library) {
            NSLog(@"✅ Loaded pre-compiled Metal library");
        }
    }

    // Fall back to runtime compilation (development)
    if (!library) {
        NSLog(@"📝 Runtime compiling Metal shaders...");
        NSString *shaderSource = [self loadShaderSource];
        if (!shaderSource) {
            NSLog(@"❌ Could not load shader source");
            return NO;
        }

        library = [_device newLibraryWithSource:shaderSource options:nil error:&error];
        if (error) {
            NSLog(@"❌ Shader compilation error: %@", error);
            return NO;
        }
    }

    // Create pipeline states
    _bgrToGrayPipeline = [self createPipeline:@"bgrToGray" library:library];
    _bgrToHSVPipeline = [self createPipeline:@"bgrToHSV" library:library];
    _hsvMaskPipeline = [self createPipeline:@"hsvRangeMask" library:library];
    _subtractPipeline = [self createPipeline:@"subtractImages" library:library];
    _bitwiseOrPipeline = [self createPipeline:@"bitwiseOr" library:library];
    _thresholdPipeline = [self createPipeline:@"applyThreshold" library:library];
    _histogramPipeline = [self createPipeline:@"computeHistogram" library:library];

    BOOL success = _bgrToGrayPipeline && _bgrToHSVPipeline && _hsvMaskPipeline &&
                   _subtractPipeline && _bitwiseOrPipeline && _thresholdPipeline &&
                   _histogramPipeline;

    if (success) {
        NSLog(@"✅ All Metal pipelines compiled successfully");
    }

    return success;
}

- (id<MTLComputePipelineState>)createPipeline:(NSString*)functionName library:(id<MTLLibrary>)library {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:functionName];

    if (!function) {
        NSLog(@"❌ Function '%@' not found in library", functionName);
        return nil;
    }

    id<MTLComputePipelineState> pipeline =
        [_device newComputePipelineStateWithFunction:function error:&error];

    if (error) {
        NSLog(@"❌ Pipeline creation failed for %@: %@", functionName, error);
    }

    return pipeline;
}

- (NSString*)loadShaderSource {
    // Try to load from file first
    NSString *shaderPath = [[NSBundle mainBundle] pathForResource:@"image_processing" ofType:@"metal"];

    if (shaderPath) {
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:shaderPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (source) {
            return source;
        }
    }

    // If not in bundle, try relative path (development)
    NSString *relativePath = @"./image_processing.metal";
    if ([[NSFileManager defaultManager] fileExistsAtPath:relativePath]) {
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:relativePath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (source) {
            return source;
        }
    }

    NSLog(@"⚠️ Shader source file not found, using embedded source");
    return nil; // In production, shaders should be pre-compiled
}

// ============================================================================
// MARK: - Main Processing Pipeline
// ============================================================================

- (ProcessingResult*)processImage:(const cv::Mat&)inputImage withParams:(ProcessingParams*)params {
    ProcessingResult *result = [[ProcessingResult alloc] init];

    if (inputImage.empty()) {
        result.errorMessage = @"Input image is empty";
        return result;
    }

    NSLog(@"🚀 Starting Metal-accelerated image processing...");
    NSLog(@"📐 Image size: %dx%d", inputImage.cols, inputImage.rows);

    CFTimeInterval startTime = CACurrentMediaTime();

    @autoreleasepool {
        // Create Metal textures from OpenCV Mat
        id<MTLTexture> bgrTexture = [self createTextureFromMat:inputImage format:MTLPixelFormatRGBA32Float];

        if (!bgrTexture) {
            result.errorMessage = @"Failed to create Metal texture from image";
            return result;
        }

        // Allocate intermediate textures
        id<MTLTexture> hsvTexture = [self createEmptyTexture:inputImage.cols
                                                      height:inputImage.rows
                                                      format:MTLPixelFormatRGBA32Float];
        id<MTLTexture> grayTexture = [self createEmptyTexture:inputImage.cols
                                                       height:inputImage.rows
                                                       format:MTLPixelFormatR32Float];
        id<MTLTexture> maskHSV = [self createEmptyTexture:inputImage.cols
                                                   height:inputImage.rows
                                                   format:MTLPixelFormatR32Float];
        id<MTLTexture> topHatTexture = [self createEmptyTexture:inputImage.cols
                                                         height:inputImage.rows
                                                         format:MTLPixelFormatR32Float];
        id<MTLTexture> topHatBinary = [self createEmptyTexture:inputImage.cols
                                                        height:inputImage.rows
                                                        format:MTLPixelFormatR32Float];
        id<MTLTexture> combined = [self createEmptyTexture:inputImage.cols
                                                    height:inputImage.rows
                                                    format:MTLPixelFormatR32Float];
        id<MTLTexture> opened = [self createEmptyTexture:inputImage.cols
                                                  height:inputImage.rows
                                                  format:MTLPixelFormatR32Float];
        id<MTLTexture> closed = [self createEmptyTexture:inputImage.cols
                                                  height:inputImage.rows
                                                  format:MTLPixelFormatR32Float];

        // Create command buffer
        id<MTLCommandBuffer> cmdBuf = [_commandQueue commandBuffer];
        cmdBuf.label = @"Image Processing Pipeline";

        // STEP 1: BGR to HSV
        [self encodeBGRToHSV:bgrTexture output:hsvTexture commandBuffer:cmdBuf];

        // STEP 2: HSV Range Mask
        simd_float3 lowerBound = simd_make_float3(
            params.hueMin / 255.0f,
            params.satMin / 255.0f,
            params.valMin / 255.0f
        );
        simd_float3 upperBound = simd_make_float3(
            params.hueMax / 255.0f,
            1.0f,
            1.0f
        );
        [self encodeHSVMask:hsvTexture
                     output:maskHSV
                 lowerBound:lowerBound
                 upperBound:upperBound
              commandBuffer:cmdBuf];

        // STEP 3: BGR to Grayscale
        [self encodeBGRToGray:bgrTexture output:grayTexture commandBuffer:cmdBuf];

        // STEP 4: Top-Hat Transform
        [self encodeTopHat:grayTexture
                    output:topHatTexture
                kernelSize:params.topHatKernelSize
             commandBuffer:cmdBuf];

        // STEP 5: Threshold top-hat
        float threshold;
        if (params.useOtsu) {
            // For Otsu, we need to execute the command buffer first to get the histogram
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            // Compute Otsu threshold on CPU (could be optimized to GPU)
            Mat topHatMat = [self matFromTexture:topHatTexture];
            threshold = [self computeOtsuThresholdMat:topHatMat] / 255.0f;

            NSLog(@"📊 Computed Otsu threshold: %.3f", threshold);

            // Start new command buffer for remaining operations
            cmdBuf = [_commandQueue commandBuffer];
            cmdBuf.label = @"Image Processing Pipeline (Part 2)";
        } else {
            threshold = params.manualThreshold;
        }

        [self encodeThreshold:topHatTexture
                       output:topHatBinary
                    threshold:threshold
                commandBuffer:cmdBuf];

        // STEP 6: Combine masks (bitwise OR)
        [self encodeBitwiseOr:maskHSV
                       imageB:topHatBinary
                       output:combined
                commandBuffer:cmdBuf];

        // STEP 7: Morphological Opening (Erode + Dilate)
        [self encodeMorphOpen:combined
                       output:opened
                   kernelSize:params.openSize
                commandBuffer:cmdBuf];

        // STEP 8: Morphological Closing (Dilate + Erode)
        [self encodeMorphClose:opened
                        output:closed
                    kernelSize:params.closeSize
                 commandBuffer:cmdBuf];

        // Execute pipeline
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        CFTimeInterval endTime = CACurrentMediaTime();
        double gpuTime = (endTime - startTime) * 1000.0; // Convert to ms

        // Convert results back to OpenCV Mats
        result.finalMask = [self matFromTexture:closed];
        result.hsvMask = [self matFromTexture:maskHSV];
        result.topHat = [self matFromTexture:topHatTexture];
        result.topHatBinary = [self matFromTexture:topHatBinary];
        result.combinedMask = [self matFromTexture:combined];
        result.opened = [self matFromTexture:opened];
        result.closed = [self matFromTexture:closed];

        result.gpuTimeMs = gpuTime;
        result.success = YES;

        // Update statistics
        _totalProcessedImages++;
        _totalGPUTime += gpuTime;

        NSLog(@"✅ Metal pipeline completed in %.2f ms", gpuTime);
        NSLog(@"📊 White pixels - HSV: %d, TopHat: %d, Final: %d",
              countNonZero(result.hsvMask),
              countNonZero(result.topHatBinary),
              countNonZero(result.finalMask));
    }

    return result;
}

// ============================================================================
// MARK: - Encoding Helper Methods
// ============================================================================

- (void)encodeBGRToGray:(id<MTLTexture>)input
                 output:(id<MTLTexture>)output
          commandBuffer:(id<MTLCommandBuffer>)cmdBuf {

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encoder.label = @"BGR to Gray";

    [encoder setComputePipelineState:_bgrToGrayPipeline];
    [encoder setTexture:input atIndex:0];
    [encoder setTexture:output atIndex:1];

    [self dispatchTexture:output encoder:encoder];
    [encoder endEncoding];
}

- (void)encodeBGRToHSV:(id<MTLTexture>)input
                output:(id<MTLTexture>)output
         commandBuffer:(id<MTLCommandBuffer>)cmdBuf {

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encoder.label = @"BGR to HSV";

    [encoder setComputePipelineState:_bgrToHSVPipeline];
    [encoder setTexture:input atIndex:0];
    [encoder setTexture:output atIndex:1];

    [self dispatchTexture:output encoder:encoder];
    [encoder endEncoding];
}

- (void)encodeHSVMask:(id<MTLTexture>)input
               output:(id<MTLTexture>)output
           lowerBound:(simd_float3)lower
           upperBound:(simd_float3)upper
        commandBuffer:(id<MTLCommandBuffer>)cmdBuf {

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encoder.label = @"HSV Range Mask";

    [encoder setComputePipelineState:_hsvMaskPipeline];
    [encoder setTexture:input atIndex:0];
    [encoder setTexture:output atIndex:1];
    [encoder setBytes:&lower length:sizeof(simd_float3) atIndex:0];
    [encoder setBytes:&upper length:sizeof(simd_float3) atIndex:1];

    [self dispatchTexture:output encoder:encoder];
    [encoder endEncoding];
}

- (void)encodeTopHat:(id<MTLTexture>)input
              output:(id<MTLTexture>)output
          kernelSize:(int)size
       commandBuffer:(id<MTLCommandBuffer>)cmdBuf {

    // Create temporary textures
    id<MTLTexture> temp1 = [self createEmptyTexture:input.width
                                             height:input.height
                                             format:input.pixelFormat];
    id<MTLTexture> temp2 = [self createEmptyTexture:input.width
                                             height:input.height
                                             format:input.pixelFormat];

    // Get or create morphology filters for this size
    MPSImageErode *erode = [self getErodeFilter:size];
    MPSImageDilate *dilate = [self getDilateFilter:size];

    // Opening = Erode + Dilate
    [erode encodeToCommandBuffer:cmdBuf sourceTexture:input destinationTexture:temp1];
    [dilate encodeToCommandBuffer:cmdBuf sourceTexture:temp1 destinationTexture:temp2];

    // Top-Hat = Original - Opening
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encoder.label = @"Top-Hat Transform";

    [encoder setComputePipelineState:_subtractPipeline];
    [encoder setTexture:input atIndex:0];
    [encoder setTexture:temp2 atIndex:1];
    [encoder setTexture:output atIndex:2];

    [self dispatchTexture:output encoder:encoder];
    [encoder endEncoding];
}

- (void)encodeThreshold:(id<MTLTexture>)input
                 output:(id<MTLTexture>)output
              threshold:(float)threshold
          commandBuffer:(id<MTLCommandBuffer>)cmdBuf {

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encoder.label = @"Apply Threshold";

    [encoder setComputePipelineState:_thresholdPipeline];
    [encoder setTexture:input atIndex:0];
    [encoder setTexture:output atIndex:1];
    [encoder setBytes:&threshold length:sizeof(float) atIndex:0];

    [self dispatchTexture:output encoder:encoder];
    [encoder endEncoding];
}

- (void)encodeBitwiseOr:(id<MTLTexture>)imageA
                 imageB:(id<MTLTexture>)imageB
                 output:(id<MTLTexture>)output
          commandBuffer:(id<MTLCommandBuffer>)cmdBuf {

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encoder.label = @"Bitwise OR";

    [encoder setComputePipelineState:_bitwiseOrPipeline];
    [encoder setTexture:imageA atIndex:0];
    [encoder setTexture:imageB atIndex:1];
    [encoder setTexture:output atIndex:2];

    [self dispatchTexture:output encoder:encoder];
    [encoder endEncoding];
}

- (void)encodeMorphOpen:(id<MTLTexture>)input
                 output:(id<MTLTexture>)output
             kernelSize:(int)size
          commandBuffer:(id<MTLCommandBuffer>)cmdBuf {

    id<MTLTexture> temp = [self createEmptyTexture:input.width
                                            height:input.height
                                            format:input.pixelFormat];

    MPSImageErode *erode = [self getErodeFilter:size];
    MPSImageDilate *dilate = [self getDilateFilter:size];

    // Open = Erode + Dilate
    [erode encodeToCommandBuffer:cmdBuf sourceTexture:input destinationTexture:temp];
    [dilate encodeToCommandBuffer:cmdBuf sourceTexture:temp destinationTexture:output];
}

- (void)encodeMorphClose:(id<MTLTexture>)input
                  output:(id<MTLTexture>)output
              kernelSize:(int)size
           commandBuffer:(id<MTLCommandBuffer>)cmdBuf {

    id<MTLTexture> temp = [self createEmptyTexture:input.width
                                            height:input.height
                                            format:input.pixelFormat];

    MPSImageDilate *dilate = [self getDilateFilter:size];
    MPSImageErode *erode = [self getErodeFilter:size];

    // Close = Dilate + Erode
    [dilate encodeToCommandBuffer:cmdBuf sourceTexture:input destinationTexture:temp];
    [erode encodeToCommandBuffer:cmdBuf sourceTexture:temp destinationTexture:output];
}

// ============================================================================
// MARK: - Helper Methods
// ============================================================================

- (void)dispatchTexture:(id<MTLTexture>)texture encoder:(id<MTLComputeCommandEncoder>)encoder {
    MTLSize threadgroupSize = [MetalCapabilities recommendedThreadgroupSize2D];
    MTLSize threadgroups = MTLSizeMake(
        (texture.width + threadgroupSize.width - 1) / threadgroupSize.width,
        (texture.height + threadgroupSize.height - 1) / threadgroupSize.height,
        1
    );

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
}

- (MPSImageDilate*)getDilateFilter:(int)size {
    NSNumber *key = @(size);
    MPSImageDilate *filter = _dilateFilters[key];

    if (!filter) {
        filter = [[MPSImageDilate alloc] initWithDevice:_device
                                            kernelWidth:size
                                           kernelHeight:size];
        _dilateFilters[key] = filter;
    }

    return filter;
}

- (MPSImageErode*)getErodeFilter:(int)size {
    NSNumber *key = @(size);
    MPSImageErode *filter = _erodeFilters[key];

    if (!filter) {
        filter = [[MPSImageErode alloc] initWithDevice:_device
                                           kernelWidth:size
                                          kernelHeight:size];
        _erodeFilters[key] = filter;
    }

    return filter;
}

- (id<MTLTexture>)createTextureFromMat:(const cv::Mat&)mat format:(MTLPixelFormat)format {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:format
        width:mat.cols
        height:mat.rows
        mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

    id<MTLTexture> texture = [_device newTextureWithDescriptor:desc];

    // Convert Mat to float and upload
    Mat floatMat;
    mat.convertTo(floatMat, CV_32FC3, 1.0/255.0);

    // Ensure RGBA format
    Mat rgba;
    if (floatMat.channels() == 3) {
        cvtColor(floatMat, rgba, COLOR_BGR2RGBA);
    } else if (floatMat.channels() == 1) {
        cvtColor(floatMat, rgba, COLOR_GRAY2RGBA);
    } else {
        rgba = floatMat;
    }

    [texture replaceRegion:MTLRegionMake2D(0, 0, mat.cols, mat.rows)
               mipmapLevel:0
                 withBytes:rgba.data
               bytesPerRow:rgba.cols * sizeof(float) * 4];

    return texture;
}

- (id<MTLTexture>)createEmptyTexture:(NSUInteger)width
                              height:(NSUInteger)height
                              format:(MTLPixelFormat)format {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:format
        width:width
        height:height
        mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = [MetalCapabilities recommendedBufferStorageMode] == MTLResourceStorageModeShared
                       ? MTLStorageModeShared : MTLStorageModePrivate;

    return [_device newTextureWithDescriptor:desc];
}

- (cv::Mat)matFromTexture:(id<MTLTexture>)texture {
    NSUInteger width = texture.width;
    NSUInteger height = texture.height;
    NSUInteger bytesPerRow = width * sizeof(float);

    Mat result(height, width, CV_32F);

    [texture getBytes:result.data
          bytesPerRow:bytesPerRow
           fromRegion:MTLRegionMake2D(0, 0, width, height)
          mipmapLevel:0];

    // Convert to 8-bit
    Mat result8u;
    result.convertTo(result8u, CV_8U, 255.0);

    return result8u;
}

- (float)computeOtsuThreshold:(const cv::Mat&)grayImage {
    Mat gray8u;
    if (grayImage.type() != CV_8U) {
        grayImage.convertTo(gray8u, CV_8U, 255.0);
    } else {
        gray8u = grayImage;
    }

    return [self computeOtsuThresholdMat:gray8u];
}

- (float)computeOtsuThresholdMat:(const cv::Mat&)gray8u {
    // Compute histogram
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    Mat hist;
    calcHist(&gray8u, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    // Normalize histogram
    hist /= (gray8u.rows * gray8u.cols);

    // Compute cumulative sums
    float sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += i * hist.at<float>(i);
    }

    float sumB = 0;
    float wB = 0;
    float wF = 0;
    float maxVariance = 0;
    int threshold = 0;

    for (int i = 0; i < 256; i++) {
        wB += hist.at<float>(i);
        if (wB == 0) continue;

        wF = 1.0 - wB;
        if (wF == 0) break;

        sumB += i * hist.at<float>(i);

        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;

        float variance = wB * wF * (mB - mF) * (mB - mF);

        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = i;
        }
    }

    return threshold;
}

- (void)logStatistics {
    NSLog(@"");
    NSLog(@"╔════════════════════════════════════════════════════════════╗");
    NSLog(@"║        MetalImageProcessor Statistics                     ║");
    NSLog(@"╠════════════════════════════════════════════════════════════╣");
    NSLog(@"║ Total Images Processed: %-35lu║", (unsigned long)_totalProcessedImages);

    if (_totalProcessedImages > 0) {
        double avgTime = _totalGPUTime / _totalProcessedImages;
        NSLog(@"║ Average GPU Time:       %-30.2f ms ║", avgTime);
        NSLog(@"║ Total GPU Time:         %-30.2f ms ║", _totalGPUTime);
        NSLog(@"║ Throughput:             %-30.1f fps║", 1000.0 / avgTime);
    }

    NSLog(@"╚════════════════════════════════════════════════════════════╝");
    NSLog(@"");
}

@end
