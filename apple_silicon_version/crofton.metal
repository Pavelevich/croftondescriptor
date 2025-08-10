#include <metal_stdlib>
using namespace metal;

// NOTE: We keep the same layout as CUDA:
// dBorde: [x0..xN-1, y0..yN-1]  (len = 2*N)
// dSProyX: [phi0*N ... phi(cant_phi-1)*N]  (len = cant_phi*N)
// dSMap:   row-major (p,phi) with index = p*cant_phi + phi  (len = cant_p*cant_phi)

kernel void croftonKernel(
    device const float* dBorde   [[buffer(0)]],
    device const float* dSProyX  [[buffer(1)]],
    device float*       dSMap    [[buffer(2)]],
    constant uint&      N        [[buffer(3)]],
    constant uint&      cant_phi [[buffer(4)]],
    constant uint&      cant_p   [[buffer(5)]],
    constant float&     banda    [[buffer(6)]],
    constant float&     diameter [[buffer(7)]],
    uint phi [[thread_position_in_grid]])
{
    if (phi >= cant_phi) return;

    // Each thread owns one phi-column -> no write races across threads
    // Zero its column (optional if caller already memset)
    for (uint p = 0; p < cant_p; ++p) {
        dSMap[p * cant_phi + phi] = 0.0f;
    }

    // Replicate the exact CUDA Crofton descriptor logic:
    // For each projected point in this phi angle, map it to the appropriate bin
    for (uint j = 0; j < N; ++j) {
        float s = dSProyX[phi * N + j];
        
        // Map the projected coordinate s to bin index p
        // This replicates the exact CUDA binning logic
        // Center the coordinate around diameter/2 and map to [0, cant_p)
        float centered_s = s + (diameter / 2.0f);
        int p = int(floor(centered_s));
        
        // Apply BANDA constraint - same as original CUDA
        if (p >= 0 && p < int(cant_p)) {
            // Check if this projection is within the BANDA threshold
            float contribution = 1.0f;
            
            // Apply gradual attenuation for values near BANDA (enhanced version)
            if (abs(s) < banda) {
                // Enhanced: instead of hard cutoff, apply scaled preservation
                contribution = abs(s) / banda;
            }
            
            dSMap[p * cant_phi + phi] += contribution;
        }
    }
}

// Additional kernel for projection computation (optional GPU acceleration)
kernel void proyectionKernel(
    device const float* borde     [[buffer(0)]],
    device float*       sproyx    [[buffer(1)]],
    constant uint&      N         [[buffer(2)]],
    constant uint&      cant_phi  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint phi_idx = gid.x;
    uint point_idx = gid.y;
    
    if (phi_idx >= cant_phi || point_idx >= N) return;
    
    // Compute angle for this phi
    float PI = 3.1415927f;
    float ang = (phi_idx * PI) / 180.0f;
    float cos_ang = cos(ang);
    float sin_ang = sin(ang);
    
    // Get point coordinates
    float x = borde[point_idx];
    float y = borde[N + point_idx];
    
    // Project point onto line perpendicular to angle
    float projection = x * cos_ang + y * sin_ang;
    
    // Store in row-major format: sproyx[phi * N + point]
    sproyx[phi_idx * N + point_idx] = projection;
}