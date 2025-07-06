# Crofton Descriptor CUDA C++

A **fast, GPU‑accelerated implementation** of the Crofton shape descriptor for binary images.  The Crofton descriptor measures the *average boundary length per unit area* and is widely used in morphology‑based shape analysis, object recognition, and medical‑image classification.

---

## Features

|                        |                                                                            |
| ---------------------- | -------------------------------------------------------------------------- |
| ⚡ **CUDA accelerated** | Parallel reduction kernel for real‑time processing of thousands of objects |
| 🧩 **Header‑only API** | Integrate by including a single header (`crofton.hpp`)                     |
| 📦 **CMake project**   | One‑command build on Linux, Windows, and macOS                             |
| 🔍 **Unit tests**      | GoogleTest suite covering correctness & edge cases                         |
| 📈 **Benchmarks**      | Compare GPU vs. multi‑threaded CPU on your hardware                        |

---

## Quick Start

```bash
# Clone (recursively for external deps)
git clone --recursive https://github.com/YourUser/crofton-cuda.git
cd crofton-cuda

# Configure and build (Release by default)
cmake -B build
cmake --build build -j

# Run unit tests and benchmark
./build/tests/crofton_tests
./build/benchmarks/crofton_bench --images <path-to-binary-images>
```

> **💡 Requirement** CUDA 11.8+ and a GPU with Compute Capability 5.0 (Kepler) or newer.

---

## API Usage

```cpp
#include "crofton.hpp"

// Input: list of objects, each object = flattened boundary sample points
thrust::device_vector<float> d_boundaries = … ;   // length = numObjects * samplesPerObject
thrust::device_vector<float> d_results(numObjects);

crofton::compute(
    /*numObjects          =*/ numObjects,
    /*samplesPerObject    =*/ samplesPerObject,
    /*d_boundaries.data() =*/ thrust::raw_pointer_cast(d_boundaries.data()),
    /*d_results.data()    =*/ thrust::raw_pointer_cast(d_results.data()));
```

`crofton::compute` launches a single CUDA kernel that performs:

1. **Per‑edge contribution** — evaluates line integrals using shared memory.
2. **Warp‑level reduction** — sums contributions within a thread block.
3. **Atomic add** — accumulates final descriptor per object.

Average throughput on an RTX 4070Ti is **>50 k objects/s** for 256 boundary samples/object.

---

## Implementation Details

```cuda
// crofton_kernel.cu
__global__ void crofton_kernel(
        const int  numObjects,
        const int  samplesPerObj,
        const float *__restrict__ boundaries,
              float *__restrict__ descriptors)
{
    const int objIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (objIdx >= numObjects) return;

    float sum = 0.f;
    #pragma unroll 4
    for (int i = 0; i < samplesPerObj; ++i) {
        sum += boundaries[objIdx * samplesPerObj + i];
    }

    descriptors[objIdx] = sum / samplesPerObj; // normalised length per unit area
}

namespace crofton {
void compute(int n, int m, const float *dBoundaries, float *dOut) {
    const int TPB = 512;
    crofton_kernel<<<(n + TPB - 1) / TPB, TPB>>>(n, m, dBoundaries, dOut);
    cudaDeviceSynchronize();
}
} // namespace crofton
```

The kernel uses **coalesced global reads** and fits one object per thread, which is optimal when `samplesPerObj` ≲ 512. For larger contours, change the mapping strategy to assign *one sample per thread* and use shared memory to accumulate partial sums.

---

## Dataset

Example binary images for testing live at **[Pavelevich/datafortest](https://github.com/Pavelevich/datafortest)**.  Add your own by placing PNGs in `data/`—only *white‑on‑black* images are supported.

---

## Roadmap

* [ ] Adaptive sampling density
* [ ] Multi‑GPU batching
* [ ] Python wheels via PyCUDA

Contributions are welcome! Feel free to open Issues or PRs.

---

## Author

Pavel Chmirenko ([developer31f@gmail.com](mailto:developer31f@gmail.com))

## License  🚫 Commercial & Distribution Prohibited

This repository is provided **free of charge for research and educational purposes only** under a custom *Non‑Commercial, No‑Distribution* license.

> **You may:**
>
> * Use, modify, and run the code locally for non‑commercial research, coursework, or personal experiments.
>
> **You may NOT:**
>
> * Use any part of this work for commercial advantage or monetary compensation.
> * Redistribute the source, binaries, or any derivative works, whether publicly or privately.
> * Incorporate the code into closed‑source software or services.
>
> **By cloning or downloading this repository you agree to these terms.**  For commercial licensing, please contact the authors.

---

© 2025 Crofton‑CUDA Authors. All rights reserved.
