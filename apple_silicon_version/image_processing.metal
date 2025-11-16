//
//  image_processing.metal
//  Crofton Descriptor - Metal Optimized Image Processing Kernels
//
//  GPU-accelerated image processing operations for cell boundary detection
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// MARK: - Color Space Conversions
// ============================================================================

/// Convert BGR to Grayscale using OpenCV formula
kernel void bgrToGray(
    texture2d<float, access::read> bgr [[texture(0)]],
    texture2d<float, access::write> gray [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= bgr.get_width() || gid.y >= bgr.get_height())
        return;

    float3 pixel = bgr.read(gid).rgb;
    // OpenCV formula: 0.299*R + 0.587*G + 0.114*B
    // But texture is BGR, so: B=pixel.b, G=pixel.g, R=pixel.r
    float grayValue = 0.114 * pixel.b + 0.587 * pixel.g + 0.299 * pixel.r;

    gray.write(float4(grayValue, grayValue, grayValue, 1.0), gid);
}

/// Convert BGR to HSV color space
kernel void bgrToHSV(
    texture2d<float, access::read> bgr [[texture(0)]],
    texture2d<float, access::write> hsv [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= bgr.get_width() || gid.y >= bgr.get_height())
        return;

    float3 pixel = bgr.read(gid).rgb;  // B, G, R in texture
    float b = pixel.b;
    float g = pixel.g;
    float r = pixel.r;

    float maxVal = max(max(r, g), b);
    float minVal = min(min(r, g), b);
    float delta = maxVal - minVal;

    // Hue calculation
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

    // Saturation
    float s = (maxVal > 0.0001) ? (delta / maxVal) : 0.0;

    // Value
    float v = maxVal;

    // Normalize: H to [0, 179/255], S and V to [0, 1] to match OpenCV
    h = h / 360.0 * (179.0 / 255.0);

    hsv.write(float4(h, s, v, 1.0), gid);
}

// ============================================================================
// MARK: - HSV Range Masking
// ============================================================================

/// Create binary mask for pixels within HSV range
kernel void hsvRangeMask(
    texture2d<float, access::read> hsvTexture [[texture(0)]],
    texture2d<float, access::write> maskTexture [[texture(1)]],
    constant float3& lowerBound [[buffer(0)]],
    constant float3& upperBound [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= hsvTexture.get_width() || gid.y >= hsvTexture.get_height())
        return;

    float3 pixel = hsvTexture.read(gid).rgb;

    // Check if pixel is within HSV range
    bool hInRange = pixel.r >= lowerBound.r && pixel.r <= upperBound.r;
    bool sInRange = pixel.g >= lowerBound.g && pixel.g <= upperBound.g;
    bool vInRange = pixel.b >= lowerBound.b && pixel.b <= upperBound.b;

    float result = (hInRange && sInRange && vInRange) ? 1.0 : 0.0;
    maskTexture.write(float4(result, result, result, 1.0), gid);
}

// ============================================================================
// MARK: - Image Arithmetic Operations
// ============================================================================

/// Subtract imageB from imageA (for Top-Hat transform)
kernel void subtractImages(
    texture2d<float, access::read> imageA [[texture(0)]],
    texture2d<float, access::read> imageB [[texture(1)]],
    texture2d<float, access::write> result [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= imageA.get_width() || gid.y >= imageA.get_height())
        return;

    float a = imageA.read(gid).r;
    float b = imageB.read(gid).r;
    float diff = max(a - b, 0.0);  // Clamp to positive values

    result.write(float4(diff, diff, diff, 1.0), gid);
}

/// Bitwise OR of two binary images
kernel void bitwiseOr(
    texture2d<float, access::read> imageA [[texture(0)]],
    texture2d<float, access::read> imageB [[texture(1)]],
    texture2d<float, access::write> result [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= imageA.get_width() || gid.y >= imageA.get_height())
        return;

    float a = imageA.read(gid).r;
    float b = imageB.read(gid).r;
    float orResult = max(a, b);  // Binary OR for normalized [0,1] values

    result.write(float4(orResult, orResult, orResult, 1.0), gid);
}

// ============================================================================
// MARK: - Thresholding Operations
// ============================================================================

/// Apply binary threshold to image
kernel void applyThreshold(
    texture2d<float, access::read> image [[texture(0)]],
    texture2d<float, access::write> binary [[texture(1)]],
    constant float& threshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= image.get_width() || gid.y >= image.get_height())
        return;

    float pixel = image.read(gid).r;
    float result = pixel >= threshold ? 1.0 : 0.0;
    binary.write(float4(result, result, result, 1.0), gid);
}

/// Compute histogram for Otsu thresholding (parallel reduction)
kernel void computeHistogram(
    texture2d<float, access::read> image [[texture(0)]],
    device atomic_uint* histogram [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= image.get_width() || gid.y >= image.get_height())
        return;

    float pixel = image.read(gid).r;
    uint bin = uint(clamp(pixel, 0.0f, 0.9999f) * 256.0);
    atomic_fetch_add_explicit(&histogram[bin], 1, memory_order_relaxed);
}

// ============================================================================
// MARK: - Normalization and Conversion
// ============================================================================

/// Convert float texture to 8-bit normalized
kernel void normalizeToUInt8(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= input.get_width() || gid.y >= input.get_height())
        return;

    float value = input.read(gid).r;
    value = clamp(value, 0.0f, 1.0f);
    output.write(float4(value, value, value, 1.0), gid);
}

/// Copy single channel to RGB
kernel void grayToRGB(
    texture2d<float, access::read> gray [[texture(0)]],
    texture2d<float, access::write> rgb [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= gray.get_width() || gid.y >= gray.get_height())
        return;

    float value = gray.read(gid).r;
    rgb.write(float4(value, value, value, 1.0), gid);
}

// ============================================================================
// MARK: - Enhanced Edge Detection
// ============================================================================

/// Multi-scale Sobel edge detection (3x3 + 5x5 combined)
kernel void multiScaleSobel(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> edges [[texture(1)]],
    constant float& scale3x3 [[buffer(0)]],
    constant float& scale5x5 [[buffer(1)]],
    constant float& threshold [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= input.get_width() || gid.y >= input.get_height())
        return;

    int2 coord = int2(gid);
    int width = input.get_width();
    int height = input.get_height();

    // 3x3 Sobel kernels
    float gx_3x3 = 0.0;
    float gy_3x3 = 0.0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 sampleCoord = clamp(coord + int2(dx, dy), int2(0), int2(width-1, height-1));
            float pixel = input.read(uint2(sampleCoord)).r;

            // Sobel X weights
            float wx = (dx == -1) ? -1.0 : (dx == 1) ? 1.0 : 0.0;
            if (dy == 0) wx *= 2.0;

            // Sobel Y weights
            float wy = (dy == -1) ? -1.0 : (dy == 1) ? 1.0 : 0.0;
            if (dx == 0) wy *= 2.0;

            gx_3x3 += pixel * wx;
            gy_3x3 += pixel * wy;
        }
    }

    // 5x5 Sobel for weak edges
    float gx_5x5 = 0.0;
    float gy_5x5 = 0.0;

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int2 sampleCoord = clamp(coord + int2(dx, dy), int2(0), int2(width-1, height-1));
            float pixel = input.read(uint2(sampleCoord)).r;

            // Simplified 5x5 Sobel weights
            float wx = (abs(dx) == 2) ? 0.5 * sign(float(dx)) :
                       (abs(dx) == 1) ? 2.0 * sign(float(dx)) : 0.0;
            float wy = (abs(dy) == 2) ? 0.5 * sign(float(dy)) :
                       (abs(dy) == 1) ? 2.0 * sign(float(dy)) : 0.0;

            gx_5x5 += pixel * wx;
            gy_5x5 += pixel * wy;
        }
    }

    // Combine multi-scale gradients
    float magnitude = sqrt(scale3x3 * (gx_3x3*gx_3x3 + gy_3x3*gy_3x3) +
                          scale5x5 * (gx_5x5*gx_5x5 + gy_5x5*gy_5x5));

    // Apply threshold
    float result = magnitude > threshold ? 1.0 : 0.0;

    edges.write(float4(result, result, result, 1.0), gid);
}

// ============================================================================
// MARK: - Debug and Utility Kernels
// ============================================================================

/// Copy texture (for debugging)
kernel void copyTexture(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= input.get_width() || gid.y >= input.get_height())
        return;

    float4 pixel = input.read(gid);
    output.write(pixel, gid);
}

/// Fill texture with constant value
kernel void fillTexture(
    texture2d<float, access::write> output [[texture(0)]],
    constant float& value [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= output.get_width() || gid.y >= output.get_height())
        return;

    output.write(float4(value, value, value, 1.0), gid);
}
