//
//  image_processing_optimized.metal
//  Enhanced kernels with simdgroup operations for better performance
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// MARK: - Optimized Color Space Conversions with Simdgroups
// ============================================================================

/// Optimized BGR to HSV with simdgroup operations
kernel void bgrToHSV_optimized(
    texture2d<float, access::read> bgr [[texture(0)]],
    texture2d<float, access::write> hsv [[texture(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (gid.x >= bgr.get_width() || gid.y >= bgr.get_height())
        return;

    float3 pixel = bgr.read(gid).rgb;
    float b = pixel.b;
    float g = pixel.g;
    float r = pixel.r;

    // Use simd operations for min/max
    float maxVal = max(max(r, g), b);
    float minVal = min(min(r, g), b);
    float delta = maxVal - minVal;

    // Hue calculation (same as before but potentially auto-vectorized)
    float h = 0.0;
    if (delta > 0.0001) {
        if (maxVal == r) {
            h = 60.0 * fmod((g - b) / delta, 6.0);
        } else if (maxVal == g) {
            h = 60.0 * ((b - r) / delta + 2.0);
        } else {
            h = 60.0 * ((r - g) / delta + 4.0);
        }
    }
    if (h < 0.0) h += 360.0;

    float s = (maxVal > 0.0001) ? (delta / maxVal) : 0.0;
    float v = maxVal;

    // Normalize
    h = h / 360.0 * (179.0 / 255.0);

    hsv.write(float4(h, s, v, 1.0), gid);
}

// ============================================================================
// MARK: - Optimized Histogram with Atomic Operations
// ============================================================================

/// Optimized histogram computation with threadgroup memory
kernel void computeHistogram_optimized(
    texture2d<float, access::read> image [[texture(0)]],
    device atomic_uint* globalHistogram [[buffer(0)]],
    threadgroup atomic_uint* localHistogram [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]])
{
    // Initialize local histogram
    uint flatTid = tid.y * tgSize.x + tid.x;
    if (flatTid < 256) {
        atomic_store_explicit(&localHistogram[flatTid], 0, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate to local histogram
    if (gid.x < image.get_width() && gid.y < image.get_height()) {
        float pixel = image.read(gid).r;
        uint bin = uint(clamp(pixel, 0.0f, 0.9999f) * 256.0);
        atomic_fetch_add_explicit(&localHistogram[bin], 1, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Merge local histogram to global
    if (flatTid < 256) {
        uint localValue = atomic_load_explicit(&localHistogram[flatTid], memory_order_relaxed);
        if (localValue > 0) {
            atomic_fetch_add_explicit(&globalHistogram[flatTid], localValue, memory_order_relaxed);
        }
    }
}

// ============================================================================
// MARK: - Optimized Morphology Operations
// ============================================================================

/// Fast dilate using shared memory
kernel void fastDilate(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant int& kernelSize [[buffer(0)]],
    threadgroup float* sharedMem [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]])
{
    if (gid.x >= input.get_width() || gid.y >= input.get_height())
        return;

    int radius = kernelSize / 2;

    // Load to shared memory with padding
    int sharedWidth = tgSize.x + 2 * radius;
    int sharedIdx = (lid.y + radius) * sharedWidth + (lid.x + radius);

    // Load center pixel
    sharedMem[sharedIdx] = input.read(gid).r;

    // Load border pixels
    if (lid.x < radius) {
        int2 leftCoord = int2(max(int(gid.x) - radius, 0), gid.y);
        sharedMem[sharedIdx - radius] = input.read(uint2(leftCoord)).r;
    }
    if (lid.x >= tgSize.x - radius) {
        int2 rightCoord = int2(min(int(gid.x) + radius, int(input.get_width()) - 1), gid.y);
        sharedMem[sharedIdx + radius] = input.read(uint2(rightCoord)).r;
    }
    if (lid.y < radius) {
        int2 topCoord = int2(gid.x, max(int(gid.y) - radius, 0));
        sharedMem[sharedIdx - radius * sharedWidth] = input.read(uint2(topCoord)).r;
    }
    if (lid.y >= tgSize.y - radius) {
        int2 bottomCoord = int2(gid.x, min(int(gid.y) + radius, int(input.get_height()) - 1));
        sharedMem[sharedIdx + radius * sharedWidth] = input.read(uint2(bottomCoord)).r;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute max over kernel (dilate)
    float maxVal = 0.0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            // Check if within ellipse kernel
            float dist = sqrt(float(dx*dx + dy*dy));
            if (dist <= radius) {
                int idx = (lid.y + radius + dy) * sharedWidth + (lid.x + radius + dx);
                maxVal = max(maxVal, sharedMem[idx]);
            }
        }
    }

    output.write(float4(maxVal), gid);
}

/// Fast erode using shared memory
kernel void fastErode(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant int& kernelSize [[buffer(0)]],
    threadgroup float* sharedMem [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]])
{
    if (gid.x >= input.get_width() || gid.y >= input.get_height())
        return;

    int radius = kernelSize / 2;

    // Load to shared memory with padding (same as dilate)
    int sharedWidth = tgSize.x + 2 * radius;
    int sharedIdx = (lid.y + radius) * sharedWidth + (lid.x + radius);

    sharedMem[sharedIdx] = input.read(gid).r;

    // Load border pixels
    if (lid.x < radius) {
        int2 leftCoord = int2(max(int(gid.x) - radius, 0), gid.y);
        sharedMem[sharedIdx - radius] = input.read(uint2(leftCoord)).r;
    }
    if (lid.x >= tgSize.x - radius) {
        int2 rightCoord = int2(min(int(gid.x) + radius, int(input.get_width()) - 1), gid.y);
        sharedMem[sharedIdx + radius] = input.read(uint2(rightCoord)).r;
    }
    if (lid.y < radius) {
        int2 topCoord = int2(gid.x, max(int(gid.y) - radius, 0));
        sharedMem[sharedIdx - radius * sharedWidth] = input.read(uint2(topCoord)).r;
    }
    if (lid.y >= tgSize.y - radius) {
        int2 bottomCoord = int2(gid.x, min(int(gid.y) + radius, int(input.get_height()) - 1));
        sharedMem[sharedIdx + radius * sharedWidth] = input.read(uint2(bottomCoord)).r;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute min over kernel (erode)
    float minVal = 1.0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            float dist = sqrt(float(dx*dx + dy*dy));
            if (dist <= radius) {
                int idx = (lid.y + radius + dy) * sharedWidth + (lid.x + radius + dx);
                minVal = min(minVal, sharedMem[idx]);
            }
        }
    }

    output.write(float4(minVal), gid);
}

// ============================================================================
// MARK: - Simdgroup Reduction Operations
// ============================================================================

/// Simdgroup sum reduction
float simdgroup_sum_float(float value, uint simd_lane_id) {
    // Parallel reduction using simd shuffle
    for (uint offset = 16; offset > 0; offset /= 2) {
        value += simd_shuffle_down(value, offset);
    }
    return value;
}

/// Simdgroup max reduction
float simdgroup_max_float(float value, uint simd_lane_id) {
    for (uint offset = 16; offset > 0; offset /= 2) {
        value = max(value, simd_shuffle_down(value, offset));
    }
    return value;
}

// ============================================================================
// MARK: - Optimized Edge Detection with Simdgroup
// ============================================================================

/// Enhanced Sobel with simdgroup reductions
kernel void sobelEdges_optimized(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float& threshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    if (gid.x >= input.get_width() || gid.y >= input.get_height())
        return;

    int2 coord = int2(gid);
    int width = input.get_width();
    int height = input.get_height();

    // Sobel X and Y gradients
    float gx = 0.0, gy = 0.0;

    // 3x3 Sobel kernel
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 sampleCoord = clamp(coord + int2(dx, dy), int2(0), int2(width-1, height-1));
            float pixel = input.read(uint2(sampleCoord)).r;

            // Sobel weights
            float wx = (dx == -1) ? -1.0 : (dx == 1) ? 1.0 : 0.0;
            float wy = (dy == -1) ? -1.0 : (dy == 1) ? 1.0 : 0.0;

            if (dy == 0) wx *= 2.0;
            if (dx == 0) wy *= 2.0;

            gx += pixel * wx;
            gy += pixel * wy;
        }
    }

    // Compute magnitude
    float magnitude = sqrt(gx * gx + gy * gy);

    // Use simdgroup to find local maximum (optional, for adaptive thresholding)
    float localMax = simdgroup_max_float(magnitude, simd_lane_id);

    // Adaptive threshold based on local context
    float adaptiveThreshold = threshold * (localMax > 0.0 ? (magnitude / localMax) : 1.0);

    float result = magnitude > adaptiveThreshold ? 1.0 : 0.0;

    output.write(float4(result), gid);
}

// ============================================================================
// MARK: - Batch Processing Kernels
// ============================================================================

/// Process multiple regions in parallel
kernel void batchBGRToGray(
    texture2d_array<float, access::read> bgrBatch [[texture(0)]],
    texture2d_array<float, access::write> grayBatch [[texture(1)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint slice = gid.z;  // Batch index
    uint2 pos = gid.xy;

    if (pos.x >= bgrBatch.get_width() || pos.y >= bgrBatch.get_height() || slice >= bgrBatch.get_array_size())
        return;

    float3 pixel = bgrBatch.read(pos, slice).rgb;
    float gray = 0.114 * pixel.b + 0.587 * pixel.g + 0.299 * pixel.r;

    grayBatch.write(float4(gray), pos, slice);
}

// ============================================================================
// MARK: - Advanced Feature Extraction
// ============================================================================

/// Compute image statistics using simdgroup reductions
kernel void computeImageStats(
    texture2d<float, access::read> image [[texture(0)]],
    device float4* stats [[buffer(0)]],  // [mean, std, min, max]
    threadgroup float* localMean [[threadgroup(0)]],
    threadgroup float* localMin [[threadgroup(1)]],
    threadgroup float* localMax [[threadgroup(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    float pixelValue = 0.0;

    if (gid.x < image.get_width() && gid.y < image.get_height()) {
        pixelValue = image.read(gid).r;
    }

    // Simdgroup reductions
    float simdSum = simdgroup_sum_float(pixelValue, simd_lane_id);
    float simdMin = simd_min(pixelValue);
    float simdMax = simd_max(pixelValue);

    // Store simdgroup results to threadgroup memory
    if (simd_lane_id == 0) {
        localMean[simd_group_id] = simdSum;
        localMin[simd_group_id] = simdMin;
        localMax[simd_group_id] = simdMax;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction (first simdgroup only)
    if (simd_group_id == 0 && tid < 32) {
        float threadSum = (tid < 32) ? localMean[tid] : 0.0;
        float threadMin = (tid < 32) ? localMin[tid] : 1.0;
        float threadMax = (tid < 32) ? localMax[tid] : 0.0;

        threadSum = simdgroup_sum_float(threadSum, simd_lane_id);
        threadMin = simd_min(threadMin);
        threadMax = simd_max(threadMax);

        if (simd_lane_id == 0) {
            uint totalPixels = image.get_width() * image.get_height();
            float mean = threadSum / totalPixels;

            stats[0] = float4(mean, 0.0, threadMin, threadMax);
        }
    }
}
