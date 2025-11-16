# Testing Metal Optimization

## ⚠️ Requirements

This implementation **ONLY works on macOS with Apple Silicon** (M1/M2/M3/M4).

### System Requirements:
- ✅ macOS 12.0+ (Monterey or later)
- ✅ Apple Silicon Mac (M1, M2, M3, M4)
- ✅ Xcode Command Line Tools
- ✅ OpenCV 4.0+
- ✅ CMake 3.16+

### Why Metal Only?

Metal is Apple's GPU framework and is **exclusive to macOS/iOS**. It does not exist on:
- ❌ Linux
- ❌ Windows
- ❌ Intel Macs (though Metal exists, optimization is for Apple Silicon)

## 🧪 Testing Instructions

### Option 1: Automated Test Script

The easiest way to build and test:

```bash
cd apple_silicon_version
./test_metal_optimized.sh
```

This script will:
1. ✅ Check system requirements
2. ✅ Build Metal-optimized version
3. ✅ Run test with sample image
4. ✅ Display performance metrics
5. ✅ Save results to file

### Option 2: Manual Build and Test

```bash
cd apple_silicon_version

# 1. Build
mkdir -p build_optimized && cd build_optimized
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# 2. Test with image
./crofton_optimized /path/to/test/image.jpg

# 3. Compare with CPU baseline
./crofton_simple /path/to/test/image.jpg

# 4. Compare with original Metal (Crofton only)
./crofton_metal /path/to/test/image.jpg
```

## 📊 Expected Output

### Successful Compilation

You should see:

```
╔════════════════════════════════════════════════════════════╗
║     Crofton Descriptor - Metal Optimization Build         ║
╠════════════════════════════════════════════════════════════╣
║ Metal Framework:       /System/Library/Frameworks/Metal.framework
║ Foundation Framework:  /System/Library/Frameworks/Foundation.framework
║ MPS Framework:         /System/Library/Frameworks/MetalPerformanceShaders.framework
║ OpenCV Version:        4.x.x
║ Architecture:          arm64
║ Build Type:            Release
╠════════════════════════════════════════════════════════════╣
║ Targets:                                                   ║
║   - crofton_simple     (CPU baseline)                      ║
║   - crofton_metal      (GPU Crofton only)                  ║
║   - crofton_optimized  (Full GPU pipeline) ✨              ║
╚════════════════════════════════════════════════════════════╝

Compiling Metal shaders to metallib
[100%] Built target crofton_optimized
```

### Successful Execution

When running `./crofton_optimized test_cell.jpg`:

```
╔════════════════════════════════════════════════════════════╗
║  Metal-Optimized Cell Boundary Detection & Classification ║
║         Full GPU Pipeline with MPS Acceleration           ║
╚════════════════════════════════════════════════════════════╝

🚀 Metal Device: Apple M4 Pro
🍎 Apple Silicon: YES

✅ Image loaded: 1024x768 pixels

╔════════════════════════════════════════════════════════════╗
║          Metal Device Capabilities                         ║
╠════════════════════════════════════════════════════════════╣
║ Device Name:            Apple M4 Pro                       ║
║ Apple Silicon:          YES ✅                             ║
║ Unified Memory:         YES ✅                             ║
║ Max Threadgroup Memory: 32 KB                              ║
║ Recommended TG Size:    64                                 ║
║ Max Threads Per TG:     1024                               ║
║ Optimal 2D TG:          16x16                              ║
╚════════════════════════════════════════════════════════════╝

🔧 Phase 1: Metal-accelerated preprocessing...
✅ Metal preprocessing completed in 45 ms
⚡ GPU time: 38.5 ms

🔍 Phase 2: Finding contours...
✅ Found 12 contours
✅ Largest contour: area = 15234 pixels, perimeter = 456.3

⚙️  Phase 3: Preparing contour for Crofton descriptor...
✅ Resampled to 239 points
✅ Contour centered at origin

🚀 Phase 4: Metal GPU-accelerated Crofton descriptor computation...
⚡ Metal Crofton computation time: 32 ms

╔════════════════════════════════════════════════════════════╗
║                        RESULTS                             ║
╠════════════════════════════════════════════════════════════╣
║ Total processing time:                           85 ms     ║
║ Metal preprocessing:                             45 ms     ║
║ Contour extraction:                             CPU        ║
║ Crofton descriptor:                          361 angles    ║
╚════════════════════════════════════════════════════════════╝

✅ Sample descriptor values: 45.2 43.1 41.5 ...

💾 Saving results...
✅ Results saved to metal_optimized_result.txt

🎉 Metal-optimized processing completed successfully!
💪 Speedup vs CPU: ~5-6x faster
⚡ Full GPU acceleration enabled
```

## 🔬 Validation Metrics

### Performance Benchmarks

Run all three versions and compare:

```bash
# CPU Baseline
time ./crofton_simple test_cell.jpg
# Expected: ~500ms

# Original Metal (Crofton only on GPU)
time ./crofton_metal test_cell.jpg
# Expected: ~350ms

# Optimized Metal (Full pipeline on GPU)
time ./crofton_optimized test_cell.jpg
# Expected: ~80ms
```

### Quality Validation

Check that all versions produce similar contours:

1. Visual inspection of output images
2. Compare descriptor values in output files
3. Check IoU (Intersection over Union) if you have ground truth

Expected quality metrics:
- **IoU with OpenCV**: > 0.95
- **Descriptor correlation**: > 0.98
- **Hausdorff distance**: < 5 pixels

## 🐛 Troubleshooting

### Build Errors

**"Metal framework not found"**
```bash
# Verify Metal is available
ls /System/Library/Frameworks/Metal.framework

# Check architecture
uname -m  # Should output: arm64
```

**"xcrun: error: unable to find utility 'metal'"**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
xcrun -find metal
```

**"opencv not found"**
```bash
# Install OpenCV via Homebrew
brew install opencv

# Verify installation
pkg-config --modversion opencv4
```

### Runtime Errors

**"Metal device not available"**
- Ensure you're running on actual hardware (not VM)
- Check System Information → Graphics/Displays for Metal support

**"Shader compilation failed"**
- Check that `image_processing.metal` is in the source directory
- Verify Metal shader syntax with: `xcrun -sdk macosx metal -c image_processing.metal`

**"Segmentation fault"**
- Ensure image file exists and is valid
- Check OpenCV can load the image: `python3 -c "import cv2; print(cv2.imread('test.jpg') is not None)"`

### Performance Issues

**Slower than expected**
- Verify you're in Release build mode (not Debug)
- Check Activity Monitor for GPU usage
- Ensure no other GPU-intensive apps are running
- Try with smaller images first

**Memory errors**
- Reduce image size if very large (>4K)
- Monitor memory usage in Activity Monitor
- Check for leaks with Instruments (Xcode)

## 📈 Performance Profiling

### Using Xcode Instruments

```bash
# Build with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8

# Profile with Metal System Trace
instruments -t "Metal System Trace" ./crofton_optimized test_cell.jpg

# Profile with Time Profiler
instruments -t "Time Profiler" ./crofton_optimized test_cell.jpg
```

### Manual Timing

Add timing code in `main_metal_optimized.cpp`:

```cpp
auto start = high_resolution_clock::now();
// ... operation ...
auto end = high_resolution_clock::now();
cout << "Operation took: " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;
```

## 🧪 Test Images

### Included Test Images

Located in `resources/sample_images/`:
- `test_cell.jpg` - Real microscopy cell image
- `test_purple_shapes.png` - Synthetic test shapes
- `test_circle.png` - Simple geometric test

### Creating Your Own Test Images

For best results, test images should have:
- ✅ Clear objects with defined boundaries
- ✅ Good contrast between object and background
- ✅ Minimal noise or artifacts
- ✅ Size: 512x512 to 2048x2048 pixels
- ✅ Format: JPG, PNG, TIFF

## 📝 Reporting Issues

If you encounter problems:

1. **Verify prerequisites**: macOS, Apple Silicon, all dependencies installed
2. **Check build output**: Look for specific error messages
3. **Test with simple image**: Try with a basic geometric shape first
4. **Compare with CPU version**: Does `crofton_simple` work?
5. **Collect logs**: Save all terminal output
6. **Check system info**: `system_profiler SPHardwareDataType SPSoftwareDataType`

## ✅ Success Checklist

- [ ] Script builds without errors
- [ ] All three executables created (simple, metal, optimized)
- [ ] Metal shaders compiled to .metallib
- [ ] Test execution completes successfully
- [ ] Output images displayed correctly
- [ ] Results file created with metrics
- [ ] Performance is 5-6x faster than CPU
- [ ] Quality matches CPU baseline visually

---

**Note**: This implementation is designed specifically for Apple Silicon and cannot run on Linux or Windows. For cross-platform alternatives, use the CUDA version (NVIDIA GPUs) or CPU-only version (all platforms).
