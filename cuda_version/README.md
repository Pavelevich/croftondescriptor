# CUDA Version (Experimental)

This directory contains the NVIDIA/CUDA branch of the Crofton boundary-detection prototype.

## Current status

- OpenCV-based preprocessing and contour extraction are implemented in `main.cu`.
- CUDA Crofton kernel wiring exists, but core device helpers are still marked as TODO placeholders.
- This branch should be treated as experimental/in-progress.

## Build

From the repository root:

```bash
cd cuda_version
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
./build/crofton_cuda test_cell.jpg
```

If no image path is provided, the binary defaults to `test_cell.jpg` in the current working directory.

## Notes

- Requires CUDA toolkit and OpenCV.
- Logs are written to `crofton_log.txt`.
