// crofton_metal.mm - Metal compute shader wrapper for Apple Silicon
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdexcept>

static const int CROFTON_MAX_POINTS = 239;

static NSString* CroftonKernelSource() {
    return @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "\n"
           "kernel void croftonKernel(\n"
           "    device const float* dBorde   [[buffer(0)]],\n"
           "    device const float* dSProyX  [[buffer(1)]],\n"
           "    device float*       dSMap    [[buffer(2)]],\n"
           "    constant uint&      N        [[buffer(3)]],\n"
           "    constant uint&      cant_phi [[buffer(4)]],\n"
           "    constant uint&      cant_p   [[buffer(5)]],\n"
           "    constant float&     banda    [[buffer(6)]],\n"
           "    constant float&     diameter [[buffer(7)]],\n"
           "    uint phi [[thread_position_in_grid]])\n"
           "{\n"
           "    if (phi >= cant_phi) return;\n"
           "\n"
           "    for (uint p = 0; p < cant_p; ++p) {\n"
           "        dSMap[p * cant_phi + phi] = 0.0f;\n"
           "    }\n"
           "\n"
           "    for (uint j = 0; j < N; ++j) {\n"
           "        float s = dSProyX[phi * N + j];\n"
           "\n"
           "        float centered_s = s + (diameter / 2.0f);\n"
           "        int p = int(floor(centered_s));\n"
           "\n"
           "        if (p >= 0 && p < int(cant_p)) {\n"
           "            float contribution = 1.0f;\n"
           "            if (abs(s) < banda) {\n"
           "                contribution = abs(s) / banda;\n"
           "            }\n"
           "            dSMap[p * cant_phi + phi] += contribution;\n"
           "        }\n"
           "    }\n"
           "}\n"
           "\n"
           "kernel void proyectionKernel(\n"
           "    device const float* borde     [[buffer(0)]],\n"
           "    device float*       sproyx    [[buffer(1)]],\n"
           "    constant uint&      N         [[buffer(2)]],\n"
           "    constant uint&      cant_phi  [[buffer(3)]],\n"
           "    uint2 gid [[thread_position_in_grid]])\n"
           "{\n"
           "    uint phi_idx = gid.x;\n"
           "    uint point_idx = gid.y;\n"
           "\n"
           "    if (phi_idx >= cant_phi || point_idx >= N) return;\n"
           "\n"
           "    float PI = 3.1415927f;\n"
           "    float ang = (phi_idx * PI) / 180.0f;\n"
           "    float cos_ang = cos(ang);\n"
           "    float sin_ang = sin(ang);\n"
           "\n"
           "    float x = borde[point_idx];\n"
           "    float y = borde[N + point_idx];\n"
           "\n"
           "    float projection = x * cos_ang + y * sin_ang;\n"
           "    sproyx[phi_idx * N + point_idx] = projection;\n"
           "}\n";
}

// Helper function for CPU projection (fallback or comparison)
void SproyectX_CPU(const float* borde, int NN, int phi, float* SProyX) {
    const float PI = 3.1415927f;
    for (int i = 0; i < phi; ++i) {
        float ang = (i * PI) / 180.0f;
        float cos_ang = cosf(ang);
        float sin_ang = sinf(ang);
        for (int j = 0; j < NN; ++j) {
            float x = borde[j];
            float y = borde[NN + j];
            SProyX[i * NN + j] = x * cos_ang + y * sin_ang;
        }
    }
}

// Helper function to compute diameter (same as CUDA version)
float Sdiametro(const std::vector<float>& contourData) {
    const int realNumPoints = std::min<int>(contourData.size() / 2, CROFTON_MAX_POINTS);
    if (realNumPoints == 0) return 0.0f;
    
    float minX = contourData[0], maxX = contourData[0];
    float minY = contourData[realNumPoints], maxY = contourData[realNumPoints];
    
    for (int i = 0; i < realNumPoints; ++i) {
        float x = contourData[i];
        float y = contourData[realNumPoints + i];
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }
    
    float width = maxX - minX;
    float height = maxY - minY;
    return std::sqrt(width * width + height * height);
}

// Main Metal-accelerated Crofton descriptor function
// Same signature/behavior as CUDA wrapper, but using Metal
void croftonDescriptorGPU_Metal(const std::vector<float>& contourData,
                                int cant_phi, int cant_p,
                                float banda,
                                std::vector<float>& hostSMap /* out */)
{
    std::cout << "🚀 Starting Metal-accelerated Crofton descriptor computation..." << std::endl;
    
    // Initialize Metal device and command queue
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Metal device not available on this system");
    }
    std::cout << "✓ Metal device: " << [[device name] UTF8String] << std::endl;
    
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        throw std::runtime_error("Failed to create Metal command queue");
    }

    // Compile Metal shaders at runtime
    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [device newLibraryWithSource:CroftonKernelSource()
                                              options:opts
                                                error:&err];
    if (!lib) {
        std::string msg = "Metal library compile failed: " + std::string([[err localizedDescription] UTF8String]);
        throw std::runtime_error(msg);
    }
    
    // Get compute functions
    id<MTLFunction> croftonFn = [lib newFunctionWithName:@"croftonKernel"];
    id<MTLFunction> projectionFn = [lib newFunctionWithName:@"proyectionKernel"];
    if (!croftonFn || !projectionFn) {
        throw std::runtime_error("Metal functions not found in library");
    }

    // Create compute pipeline states
    err = nil;
    id<MTLComputePipelineState> croftonPSO = [device newComputePipelineStateWithFunction:croftonFn error:&err];
    if (!croftonPSO) {
        std::string msg = "Crofton pipeline create failed: " + std::string([[err localizedDescription] UTF8String]);
        throw std::runtime_error(msg);
    }
    
    err = nil;
    id<MTLComputePipelineState> projectionPSO = [device newComputePipelineStateWithFunction:projectionFn error:&err];
    if (!projectionPSO) {
        std::string msg = "Projection pipeline create failed: " + std::string([[err localizedDescription] UTF8String]);
        throw std::runtime_error(msg);
    }

    const uint32_t N = CROFTON_MAX_POINTS;
    
    // Prepare contour data buffer (dBorde: 2*N floats)
    std::vector<float> hostBorde(2 * N, 0.0f);
    const int realNumPoints = std::min<int>(contourData.size() / 2, N);
    std::cout << "✓ Processing " << realNumPoints << " boundary points" << std::endl;
    
    for (int i = 0; i < realNumPoints; ++i) {
        hostBorde[i]       = contourData[i];           // x coordinates
        hostBorde[N + i]   = contourData[realNumPoints + i]; // y coordinates
    }

    const size_t bytesBorde = sizeof(float) * hostBorde.size();
    id<MTLBuffer> dBorde = [device newBufferWithBytes:hostBorde.data()
                                               length:bytesBorde
                                              options:MTLResourceStorageModeShared];

    // Prepare projection buffer (dSProyX: cant_phi * N)
    std::vector<float> hostSProyX(cant_phi * N, 0.0f);
    
    // Option 1: Use GPU for projection (fully GPU accelerated)
    id<MTLBuffer> dSProyX = [device newBufferWithLength:sizeof(float) * hostSProyX.size()
                                                 options:MTLResourceStorageModeShared];
    
    // Launch projection kernel
    id<MTLCommandBuffer> projCmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> projEnc = [projCmd computeCommandEncoder];
    [projEnc setComputePipelineState:projectionPSO];
    [projEnc setBuffer:dBorde   offset:0 atIndex:0];
    [projEnc setBuffer:dSProyX  offset:0 atIndex:1];
    [projEnc setBytes:&N        length:sizeof(uint32_t) atIndex:2];
    [projEnc setBytes:&cant_phi length:sizeof(uint32_t) atIndex:3];
    
    // 2D dispatch: [cant_phi, N]
    MTLSize projThreadsPerTG = MTLSizeMake(16, 16, 1);
    MTLSize projGridSize = MTLSizeMake(
        ((uint32_t)cant_phi + 15) / 16 * 16,
        (N + 15) / 16 * 16,
        1
    );
    [projEnc dispatchThreads:projGridSize threadsPerThreadgroup:projThreadsPerTG];
    [projEnc endEncoding];
    [projCmd commit];
    [projCmd waitUntilCompleted];
    
    std::cout << "✓ GPU projection computation completed" << std::endl;

    // Prepare Crofton map buffer (dSMap: cant_p * cant_phi)
    hostSMap.assign(cant_p * cant_phi, 0.0f);
    id<MTLBuffer> dSMap = [device newBufferWithBytes:hostSMap.data()
                                              length:sizeof(float) * hostSMap.size()
                                             options:MTLResourceStorageModeShared];

    // Compute diameter for binning
    float diameter = Sdiametro(contourData);
    std::cout << "✓ Computed diameter: " << diameter << std::endl;
    std::cout << "✓ Using BANDA: " << banda << std::endl;
    std::cout << "✓ Bins (cant_p): " << cant_p << ", Angles (cant_phi): " << cant_phi << std::endl;

    // Launch main Crofton descriptor kernel
    id<MTLCommandBuffer> croftonCmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> croftonEnc = [croftonCmd computeCommandEncoder];
    [croftonEnc setComputePipelineState:croftonPSO];
    [croftonEnc setBuffer:dBorde   offset:0 atIndex:0];
    [croftonEnc setBuffer:dSProyX  offset:0 atIndex:1];
    [croftonEnc setBuffer:dSMap    offset:0 atIndex:2];
    [croftonEnc setBytes:&N        length:sizeof(uint32_t) atIndex:3];
    [croftonEnc setBytes:&cant_phi length:sizeof(uint32_t) atIndex:4];
    [croftonEnc setBytes:&cant_p   length:sizeof(uint32_t) atIndex:5];
    [croftonEnc setBytes:&banda    length:sizeof(float)    atIndex:6];
    [croftonEnc setBytes:&diameter length:sizeof(float)    atIndex:7];

    // Launch: one thread per phi angle, pad up to multiple of threadgroup size
    const NSUInteger tg = std::min<NSUInteger>(croftonPSO.maxTotalThreadsPerThreadgroup, 64);
    MTLSize threadsPerTG = MTLSizeMake(tg, 1, 1);
    // Round up grid size to multiple of threadgroup size
    NSUInteger gridX = ((NSUInteger)cant_phi + tg - 1) / tg * tg;
    MTLSize gridSize = MTLSizeMake(gridX, 1, 1);

    [croftonEnc dispatchThreads:gridSize threadsPerThreadgroup:threadsPerTG];
    [croftonEnc endEncoding];
    [croftonCmd commit];
    [croftonCmd waitUntilCompleted];

    // Read back results (shared buffer allows direct access)
    std::memcpy(hostSMap.data(), [dSMap contents], sizeof(float) * hostSMap.size());
    
    std::cout << "🎉 Metal-accelerated Crofton descriptor computation completed!" << std::endl;
    std::cout << "✓ Generated " << hostSMap.size() << " descriptor values" << std::endl;
    
    // Log some sample values for verification
    std::cout << "Sample SMap values: ";
    for (int i = 0; i < std::min(10, (int)hostSMap.size()); ++i) {
        std::cout << hostSMap[i] << " ";
    }
    std::cout << std::endl;
}

// C++ wrapper function to match existing interface
extern "C" {
    void computeMetalCroftonDescriptor(const std::vector<float>& contourData,
                                      std::vector<float>& descriptor) {
        try {
            int cant_phi = 361;  // Same as CUDA: 0-360 degrees
            float diameter = Sdiametro(contourData);
            int cant_p = int(std::ceil(diameter / 2.0f));
            float banda = 20.0f; // Enhanced BANDA from our improved algorithm
            
            std::vector<float> SMap;
            croftonDescriptorGPU_Metal(contourData, cant_phi, cant_p, banda, SMap);
            
            // Convert SMap to final descriptor format (same as CUDA)
            descriptor.assign(cant_phi, 0.0f);
            for (int phi = 0; phi < cant_phi; ++phi) {
                float width = 0.0f;
                for (int p = 0; p < cant_p; ++p) {
                    if (SMap[p * cant_phi + phi] > 0.0f) {
                        width = p; // Find maximum p with non-zero value
                    }
                }
                descriptor[phi] = width;
            }
            
            std::cout << "✓ Metal Crofton descriptor computed successfully!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Metal Crofton descriptor failed: " << e.what() << std::endl;
            // Fallback to CPU version could be implemented here
            throw;
        }
    }
}