#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

// Configuration constants
static const int CROFTON_MAX_POINTS = 239;
static const float PI = 3.1415927f;
static const float ENHANCED_BANDA = 20.0f; // Increased from 10.0f for better detection

@interface MetalEdgeDetector : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLComputePipelineState> sobelPipelineState;
@property (nonatomic, strong) id<MTLComputePipelineState> croftonPipelineState;
@end

@implementation MetalEdgeDetector

- (instancetype)init {
    self = [super init];
    if (self) {
        self.device = MTLCreateSystemDefaultDevice();
        if (!self.device) {
            NSLog(@"Metal is not supported on this device");
            return nil;
        }
        
        self.commandQueue = [self.device newCommandQueue];
        [self setupComputePipelines];
    }
    return self;
}

- (void)setupComputePipelines {
    NSError *error = nil;
    
    // Enhanced Sobel kernel with multi-scale edge detection
    NSString *sobelSource = @"
    #include <metal_stdlib>
    using namespace metal;
    
    // Enhanced multi-scale Sobel with adaptive thresholding
    kernel void enhanced_sobel_kernel(
        texture2d<float, access::read> inputTexture [[texture(0)]],
        texture2d<float, access::write> outputTexture [[texture(1)]],
        constant float& threshold [[buffer(0)]],
        constant float& scale1 [[buffer(1)]],
        constant float& scale2 [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= inputTexture.get_width() || gid.y >= inputTexture.get_height()) {
            return;
        }
        
        // Multi-scale Sobel operators (3x3 and 5x5)
        float3x3 sobelX_3x3 = float3x3(-1, 0, 1, -2, 0, 2, -1, 0, 1);
        float3x3 sobelY_3x3 = float3x3(-1, -2, -1, 0, 0, 0, 1, 2, 1);
        
        // 5x5 Sobel for better weak edge detection
        float gx_5x5 = 0.0, gy_5x5 = 0.0;
        float gx_3x3 = 0.0, gy_3x3 = 0.0;
        
        // 3x3 Sobel
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                uint2 coord = uint2(int(gid.x) + j, int(gid.y) + i);
                coord = clamp(coord, uint2(0), uint2(inputTexture.get_width()-1, inputTexture.get_height()-1));
                
                float pixel = inputTexture.read(coord).r;
                gx_3x3 += pixel * sobelX_3x3[i+1][j+1];
                gy_3x3 += pixel * sobelY_3x3[i+1][j+1];
            }
        }
        
        // 5x5 Sobel for weak edges
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                uint2 coord = uint2(int(gid.x) + j, int(gid.y) + i);
                coord = clamp(coord, uint2(0), uint2(inputTexture.get_width()-1, inputTexture.get_height()-1));
                
                float pixel = inputTexture.read(coord).r;
                // Simplified 5x5 Sobel weights
                float wx = (j == -2 || j == 2) ? 0.5 : (j == -1 || j == 1) ? 2.0 : 0.0;
                float wy = (i == -2 || i == 2) ? 0.5 : (i == -1 || i == 1) ? 2.0 : 0.0;
                
                gx_5x5 += pixel * wx * ((j > 0) ? 1 : -1);
                gy_5x5 += pixel * wy * ((i > 0) ? 1 : -1);
            }
        }
        
        // Combine multi-scale gradients
        float magnitude = sqrt(scale1 * (gx_3x3*gx_3x3 + gy_3x3*gy_3x3) + 
                              scale2 * (gx_5x5*gx_5x5 + gy_5x5*gy_5x5));
        
        // Adaptive threshold with local contrast enhancement
        float result = magnitude > threshold ? 1.0 : 0.0;
        
        outputTexture.write(float4(result, result, result, 1.0), gid);
    }";
    
    // Crofton descriptor kernel optimized for Apple Silicon
    NSString *croftonSource = @"
    #include <metal_stdlib>
    using namespace metal;
    
    constant float PI_CONSTANT = 3.1415927;
    
    kernel void crofton_kernel(
        constant float* boundaries [[buffer(0)]],
        device float* results [[buffer(1)]],
        constant int& numPoints [[buffer(2)]],
        constant int& numAngles [[buffer(3)]],
        uint gid [[thread_position_in_grid]]
    ) {
        if (gid >= numAngles) return;
        
        float angle = (gid * PI_CONSTANT) / 180.0;
        float cosA = cos(angle);
        float sinA = sin(angle);
        
        // Project all points onto the line perpendicular to angle
        float minProj = INFINITY;
        float maxProj = -INFINITY;
        
        for (int i = 0; i < numPoints; i++) {
            float x = boundaries[i];
            float y = boundaries[numPoints + i];
            float proj = x * cosA + y * sinA;
            
            minProj = min(minProj, proj);
            maxProj = max(maxProj, proj);
        }
        
        // Store the width for this angle
        results[gid] = maxProj - minProj;
    }";
    
    // Compile shaders
    id<MTLLibrary> sobelLibrary = [self.device newLibraryWithSource:sobelSource options:nil error:&error];
    if (error) {
        NSLog(@"Error compiling Sobel shader: %@", error.localizedDescription);
        return;
    }
    
    id<MTLLibrary> croftonLibrary = [self.device newLibraryWithSource:croftonSource options:nil error:&error];
    if (error) {
        NSLog(@"Error compiling Crofton shader: %@", error.localizedDescription);
        return;
    }
    
    id<MTLFunction> sobelFunction = [sobelLibrary newFunctionWithName:@"enhanced_sobel_kernel"];
    id<MTLFunction> croftonFunction = [croftonLibrary newFunctionWithName:@"crofton_kernel"];
    
    self.sobelPipelineState = [self.device newComputePipelineStateWithFunction:sobelFunction error:&error];
    self.croftonPipelineState = [self.device newComputePipelineStateWithFunction:croftonFunction error:&error];
    
    if (error) {
        NSLog(@"Error creating pipeline states: %@", error.localizedDescription);
    }
}

- (Mat)enhancedEdgeDetection:(const Mat&)inputImage {
    // Convert to grayscale if needed
    Mat grayImage;
    if (inputImage.channels() == 3) {
        cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
    } else {
        grayImage = inputImage.clone();
    }
    
    // Normalize to float
    Mat floatImage;
    grayImage.convertTo(floatImage, CV_32F, 1.0/255.0);
    
    // Create Metal textures
    MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                                  width:floatImage.cols
                                                                                                 height:floatImage.rows
                                                                                              mipmapped:NO];
    textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    
    id<MTLTexture> inputTexture = [self.device newTextureWithDescriptor:textureDescriptor];
    id<MTLTexture> outputTexture = [self.device newTextureWithDescriptor:textureDescriptor];
    
    // Upload image data
    [inputTexture replaceRegion:MTLRegionMake2D(0, 0, floatImage.cols, floatImage.rows)
                    mipmapLevel:0
                      withBytes:floatImage.data
                    bytesPerRow:floatImage.cols * sizeof(float)];
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set pipeline and parameters
    [encoder setComputePipelineState:self.sobelPipelineState];
    [encoder setTexture:inputTexture atIndex:0];
    [encoder setTexture:outputTexture atIndex:1];
    
    // Enhanced parameters for better weak edge detection
    float threshold = 0.1f;  // Lower threshold for weak edges
    float scale1 = 0.7f;     // Weight for 3x3 Sobel
    float scale2 = 0.3f;     // Weight for 5x5 Sobel
    
    [encoder setBytes:&threshold length:sizeof(float) atIndex:0];
    [encoder setBytes:&scale1 length:sizeof(float) atIndex:1];
    [encoder setBytes:&scale2 length:sizeof(float) atIndex:2];
    
    // Dispatch threads
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroupCount = MTLSizeMake(
        (floatImage.cols + threadgroupSize.width - 1) / threadgroupSize.width,
        (floatImage.rows + threadgroupSize.height - 1) / threadgroupSize.height,
        1);
    
    [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Download result
    Mat result(floatImage.rows, floatImage.cols, CV_32F);
    [outputTexture getBytes:result.data
                bytesPerRow:floatImage.cols * sizeof(float)
                 fromRegion:MTLRegionMake2D(0, 0, floatImage.cols, floatImage.rows)
                mipmapLevel:0];
    
    // Convert back to 8-bit
    Mat binaryResult;
    result.convertTo(binaryResult, CV_8U, 255.0);
    
    return binaryResult;
}

- (vector<float>)computeCroftonDescriptor:(const vector<float>&)contourData {
    int numPoints = contourData.size() / 2;
    int numAngles = 361;
    
    // Create Metal buffers
    id<MTLBuffer> boundariesBuffer = [self.device newBufferWithBytes:contourData.data()
                                                              length:contourData.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> resultsBuffer = [self.device newBufferWithLength:numAngles * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:self.croftonPipelineState];
    [encoder setBuffer:boundariesBuffer offset:0 atIndex:0];
    [encoder setBuffer:resultsBuffer offset:0 atIndex:1];
    [encoder setBytes:&numPoints length:sizeof(int) atIndex:2];
    [encoder setBytes:&numAngles length:sizeof(int) atIndex:3];
    
    MTLSize threadgroupSize = MTLSizeMake(64, 1, 1);
    MTLSize threadgroupCount = MTLSizeMake((numAngles + 63) / 64, 1, 1);
    
    [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Get results
    float* results = (float*)[resultsBuffer contents];
    vector<float> descriptor(results, results + numAngles);
    
    return descriptor;
}

@end

// C++ wrapper functions
vector<Point> resampleContour(const vector<Point>& contour, int nPoints) {
    vector<Point> resampled;
    if (contour.empty() || nPoints <= 0) return resampled;

    // Compute perimeter
    vector<double> cumDist;
    cumDist.push_back(0.0);
    double totalLen = 0.0;
    for (size_t i = 0; i < contour.size(); ++i) {
        Point p1 = contour[i];
        Point p2 = contour[(i + 1) % contour.size()];
        double dx = double(p2.x - p1.x);
        double dy = double(p2.y - p1.y);
        double segLen = sqrt(dx*dx + dy*dy);
        totalLen += segLen;
        cumDist.push_back(totalLen);
    }

    if (totalLen < 1e-9) {
        resampled.resize(nPoints, contour[0]);
        return resampled;
    }

    double step = totalLen / nPoints;
    resampled.reserve(nPoints);

    for (int i = 0; i < nPoints; ++i) {
        double currentDist = i * step;
        while (currentDist >= totalLen) currentDist -= totalLen;

        auto it = std::lower_bound(cumDist.begin(), cumDist.end(), currentDist);
        int idx = int(it - cumDist.begin());
        if (idx == 0) {
            resampled.push_back(contour[0]);
        } else {
            int prevIdx = idx - 1;
            double prevDist = cumDist[prevIdx];
            double segLen = (cumDist[idx] - prevDist);
            double frac = (currentDist - prevDist) / segLen;

            Point p1 = contour[prevIdx];
            Point p2 = contour[idx % contour.size()];

            float x = float(p1.x + frac * (p2.x - p1.x));
            float y = float(p1.y + frac * (p2.y - p1.y));
            resampled.push_back(Point(cvRound(x), cvRound(y)));
        }
    }
    return resampled;
}

Mat enhancedPreprocessing(const Mat& inputImage) {
    Mat hsv;
    cvtColor(inputImage, hsv, COLOR_BGR2HSV);
    
    // Expanded HSV range for better color detection
    Scalar lowerPurple(90, 15, 15);   // Expanded range
    Scalar upperPurple(180, 255, 255);
    Mat maskHSV;
    inRange(hsv, lowerPurple, upperPurple, maskHSV);
    
    // Multi-scale top-hat transform
    Mat imgGray;
    cvtColor(inputImage, imgGray, COLOR_BGR2GRAY);
    
    // Apply different kernel sizes for multi-scale edge detection
    vector<Mat> topHats;
    vector<int> kernelSizes = {7, 15, 25}; // Multi-scale approach
    
    for (int kernelSize : kernelSizes) {
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernelSize, kernelSize));
        Mat opened, topHat;
        morphologyEx(imgGray, opened, MORPH_OPEN, kernel);
        topHat = imgGray - opened;
        topHats.push_back(topHat);
    }
    
    // Combine multi-scale top-hats
    Mat combinedTopHat = Mat::zeros(imgGray.size(), CV_8U);
    for (const Mat& th : topHats) {
        Mat binTH;
        threshold(th, binTH, 0, 255, THRESH_BINARY | THRESH_OTSU);
        combinedTopHat = max(combinedTopHat, binTH);
    }
    
    // Combine with HSV mask
    Mat combined;
    bitwise_or(maskHSV, combinedTopHat, combined);
    
    // Enhanced morphological operations
    Mat kernelOpen = getStructuringElement(MORPH_ELLIPSE, Size(2,2));  // Smaller kernel
    Mat kernelClose = getStructuringElement(MORPH_ELLIPSE, Size(7,7)); // Larger closing
    Mat opened, closed;
    
    morphologyEx(combined, opened, MORPH_OPEN, kernelOpen);
    morphologyEx(opened, closed, MORPH_CLOSE, kernelClose);
    
    return closed;
}

int main() {
    // Initialize Metal detector
    MetalEdgeDetector* detector = [[MetalEdgeDetector alloc] init];
    if (!detector) {
        cerr << "Failed to initialize Metal edge detector" << endl;
        return -1;
    }
    
    // Load image
    Mat imgColor = imread("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/test_cell.jpg");
    if (imgColor.empty()) {
        cerr << "Error: Could not load image" << endl;
        return -1;
    }
    
    cout << "Processing with enhanced Apple Silicon edge detection..." << endl;
    
    // Enhanced preprocessing
    Mat preprocessed = enhancedPreprocessing(imgColor);
    imshow("Enhanced Preprocessing", preprocessed);
    
    // Apply Metal-accelerated edge detection
    Mat metalEdges = [detector enhancedEdgeDetection:preprocessed];
    imshow("Metal Edge Detection", metalEdges);
    
    // Find contours
    vector<vector<Point>> contours;
    findContours(metalEdges, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    if (contours.empty()) {
        cerr << "No contours found" << endl;
        waitKey(0);
        return -1;
    }
    
    // Find largest contour
    int largestIdx = 0;
    double largestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area > largestArea) {
            largestArea = area;
            largestIdx = (int)i;
        }
    }
    
    cout << "Largest contour area: " << largestArea << " pixels" << endl;
    
    // Visualize result
    Mat result = imgColor.clone();
    drawContours(result, contours, largestIdx, Scalar(0, 255, 0), 3);
    imshow("Enhanced Edge Detection Result", result);
    
    // Resample and center contour
    vector<Point> resampled = resampleContour(contours[largestIdx], CROFTON_MAX_POINTS);
    
    Rect box = boundingRect(resampled);
    float cx = box.x + box.width * 0.5f;
    float cy = box.y + box.height * 0.5f;
    for (auto& pt : resampled) {
        pt.x = cvRound(pt.x - cx);
        pt.y = cvRound(pt.y - cy);
    }
    
    // Convert to contour data format
    vector<float> contourData;
    contourData.reserve(resampled.size() * 2);
    for (const Point& p : resampled) {
        contourData.push_back((float)p.x);
    }
    for (const Point& p : resampled) {
        contourData.push_back((float)p.y);
    }
    
    // Compute Crofton descriptor using Metal
    cout << "Computing Crofton descriptor on Apple Silicon..." << endl;
    vector<float> descriptor = [detector computeCroftonDescriptor:contourData];
    
    // Output results
    cout << "Crofton descriptor computed with " << descriptor.size() << " angles" << endl;
    cout << "Sample descriptor values: ";
    for (int i = 0; i < min(10, (int)descriptor.size()); ++i) {
        cout << descriptor[i] << " ";
    }
    cout << endl;
    
    // Save descriptor to file
    ofstream outFile("apple_silicon_crofton_result.txt");
    if (outFile.is_open()) {
        outFile << "Apple Silicon Crofton Descriptor Results\n";
        outFile << "=========================================\n";
        outFile << "Number of boundary points: " << CROFTON_MAX_POINTS << "\n";
        outFile << "Number of projection angles: " << descriptor.size() << "\n";
        outFile << "Contour area: " << largestArea << " pixels\n\n";
        outFile << "Descriptor values:\n";
        for (size_t i = 0; i < descriptor.size(); ++i) {
            outFile << "Angle " << i << ": " << descriptor[i] << "\n";
        }
        outFile.close();
        cout << "Results saved to apple_silicon_crofton_result.txt" << endl;
    }
    
    cout << "\nEnhanced edge detection completed!" << endl;
    cout << "Improvements made:" << endl;
    cout << "- Multi-scale Sobel edge detection (3x3 + 5x5)" << endl;
    cout << "- Expanded HSV color range" << endl;
    cout << "- Multi-scale top-hat transforms" << endl;
    cout << "- Metal Performance Shaders acceleration" << endl;
    cout << "- Enhanced morphological operations" << endl;
    
    waitKey(0);
    return 0;
}