// Crofton descriptor — CUDA device code (single source of truth).
//
// This file is compiled two ways:
//   1) #include'd by main.cu and built with nvcc          (native CLI)
//   2) loaded at runtime and compiled with NVRTC by CuPy   (webapp backend)
// Keep it free of host code and #includes so NVRTC can build it as-is.
//
// Buffer layouts (same convention as the original code and the Metal port):
//   borde  : [x0..xN-1 | y0..yN-1]        len = 2*N   (SoA, centered contour)
//   sproyx : [phi*N + j]                  len = cant_phi*N
//   smap   : [p*cant_phi + phi]           len = cant_p*cant_phi
//
// The vote map follows the classical Crofton construction: for every angle
// phi the family of parallel lines with normal direction phi is sampled at
// cant_p signed offsets covering [-window/2, +window/2], and SMap[p][phi]
// is the number of times that line crosses the closed polygon.
// (The previous implementations either left the kernel empty — CUDA — or
// binned point projections with an off-by-half range that silently dropped
// every positive projection — Metal.)
//
// IMPORTANT: the host must pass window >= 2*max_j ||p_j|| of the centered
// contour (e.g. window = 2*max radial distance). A pairwise diameter is NOT
// a valid window: for non-centrally-symmetric shapes the projection interval
// at oblique angles is not centered at 0 and crossings would be clipped.

#ifndef CROFTON_KERNELS_CU
#define CROFTON_KERNELS_CU

extern "C" {

// One thread per (phi, point): project every contour point onto the
// direction phi (in degrees). Padded slots j >= realN are written as 0 but
// never read by the kernels below.
__global__ void proyectionKernel(const float* __restrict__ borde,
                                 float* __restrict__ sproyx,
                                 int N, int realN, int cant_phi)
{
    int phi = blockIdx.x * blockDim.x + threadIdx.x;
    int j   = blockIdx.y * blockDim.y + threadIdx.y;
    if (phi >= cant_phi || j >= N) return;

    const float PI = 3.14159265358979f;
    float ang = (phi * PI) / 180.0f;
    float x = (j < realN) ? borde[j]     : 0.0f;
    float y = (j < realN) ? borde[N + j] : 0.0f;
    sproyx[phi * N + j] = x * cosf(ang) + y * sinf(ang);
}

// One thread per phi column (no atomics: each thread owns its column).
// A polygon edge (j -> j+1) crosses the line at offset o exactly when o lies
// between the projections of its endpoints; the half-open interval [lo, hi)
// keeps crossing counts consistent when a vertex falls on a sampled line.
// Bin p samples the offset o_p = -window/2 + (p + 0.5) * window/cant_p.
__global__ void kernelCrofton(const float* __restrict__ sproyx,
                              float* __restrict__ smap,
                              int N, int realN, int cant_phi, int cant_p,
                              float window)
{
    int phi = blockIdx.x * blockDim.x + threadIdx.x;
    if (phi >= cant_phi) return;

    for (int p = 0; p < cant_p; ++p)
        smap[p * cant_phi + phi] = 0.0f;

    if (realN < 3 || window <= 0.0f || cant_p <= 0) return;

    const float* s = &sproyx[phi * N];
    float binW   = window / cant_p;
    float origin = -0.5f * window;

    for (int j = 0; j < realN; ++j) {
        float a = s[j];
        float b = s[(j + 1) % realN];
        float lo = fminf(a, b);
        float hi = fmaxf(a, b);
        if (hi <= lo) continue;

        // bins whose sampled offset lies in [lo, hi):
        //   lo <= origin + (p + 0.5)*binW < hi
        int p0 = (int)ceilf((lo - origin) / binW - 0.5f);
        int p1 = (int)ceilf((hi - origin) / binW - 0.5f) - 1;
        if (p0 < 0) p0 = 0;
        if (p1 > cant_p - 1) p1 = cant_p - 1;
        for (int p = p0; p <= p1; ++p)
            smap[p * cant_phi + phi] += 1.0f;
    }
}

// One thread per phi: reduce the vote map to the per-angle signature.
//   curve[phi] = (binW/2) * sum_p SMap[p][phi]
//     For a convex shape every line crosses 0 or 2 times, so curve equals
//     the support (Feret) width; concavities push it above the width.
//   feret[phi] = max_j s_j - min_j s_j  (plain support width, for reference)
__global__ void reduceKernel(const float* __restrict__ sproyx,
                             const float* __restrict__ smap,
                             float* __restrict__ curve,
                             float* __restrict__ feret,
                             int N, int realN, int cant_phi, int cant_p,
                             float window)
{
    int phi = blockIdx.x * blockDim.x + threadIdx.x;
    if (phi >= cant_phi) return;

    float binW = (cant_p > 0) ? window / cant_p : 0.0f;
    float sum = 0.0f;
    for (int p = 0; p < cant_p; ++p)
        sum += smap[p * cant_phi + phi];
    curve[phi] = 0.5f * binW * sum;

    const float* s = &sproyx[phi * N];
    float mn =  3.402823e38f;
    float mx = -3.402823e38f;
    for (int j = 0; j < realN; ++j) {
        mn = fminf(mn, s[j]);
        mx = fmaxf(mx, s[j]);
    }
    feret[phi] = (realN > 0) ? (mx - mn) : 0.0f;
}

} // extern "C"

#endif // CROFTON_KERNELS_CU
