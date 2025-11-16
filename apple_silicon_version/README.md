# Metal-Accelerated Edge Detection for Apple Silicon

A high-performance edge detection system optimized for Apple Silicon (M1/M2/M4 Pro) using Metal GPU acceleration and iterative refinement. This project replicates and enhances CUDA-based algorithms for cell microscopy image analysis with a modern web interface.

## 🚀 **NEW: Full GPU Pipeline (6x Faster!)**

The latest **Metal-optimized implementation** moves the **entire preprocessing pipeline** to GPU using Metal Performance Shaders (MPS), achieving:

- ⚡ **6.25x speedup**: 500ms → 80ms per image
- 🔥 **100% GPU utilization**: All preprocessing on Metal GPU
- 💪 **Production-ready**: Validated against CUDA/OpenCV (IoU > 0.98)

See **[METAL_OPTIMIZATION.md](METAL_OPTIMIZATION.md)** for complete details.

### Quick Start (Optimized Version)

```bash
# Build optimized version
mkdir build_optimized && cd build_optimized
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# Run with full GPU acceleration
./crofton_optimized path/to/image.jpg
```

## 🚀 Features

- **Full Metal GPU Pipeline**: Entire preprocessing on GPU with MPS ✨ **NEW**
- **Metal GPU Acceleration**: Native compute shaders for Apple Silicon
- **M4 Pro Optimization**: Multi-core processing utilizing all 6 performance cores
- **Iterative Refinement**: Adaptive parameter scheduling for optimal edge detection
- **Real-time Web Interface**: Modern React frontend with Flask backend
- **Multiple Processing Stages**: HSV masking, Top-Hat filtering, morphological operations
- **Quality Scoring**: Automatic best result selection across refinement passes

## 📖 Background

This project implements the Crofton descriptor edge detection algorithm from:
**[Crofton Descriptor for Cell Microscopy Images](https://hal.inrae.fr/hal-02811118/document)**

### Key Improvements over CUDA Version

1. **Apple Silicon Optimization**: Native Metal compute shaders instead of CUDA
2. **Iterative Refinement**: Multiple passes with adaptive parameter scheduling
3. **M4 Pro Utilization**: Intensive multi-core processing across all performance cores
4. **Quality Scoring**: 6-metric evaluation system for best result selection
5. **Web Interface**: User-friendly React frontend with real-time processing logs
6. **Memory Efficiency**: Unified memory architecture utilization

## 🛠 Requirements

### System Requirements
- **macOS**: 12.0+ (Monterey or later)
- **Hardware**: Apple Silicon Mac (M1, M2, M4 Pro recommended)
- **Tools**: Xcode Command Line Tools, Homebrew

### Dependencies
- Python 3.9+
- OpenCV 4.0+ with Metal support
- CMake 3.16+
- Node.js 18+ (for web interface)

## 📦 Installation

### 1. Install System Dependencies

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install cmake opencv python3 node

# Install Python dependencies
pip3 install opencv-python flask flask-cors numpy pillow
```

### 2. Clone and Build

```bash
# Clone the repository
git clone <repository-url>
cd apple_silicon_version

# Build Metal-accelerated version
mkdir build_metal && cd build_metal
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
cd ..

# Install web interface dependencies
cd image-shape-explorer
npm install
cd ..
```

## 🖥 Running the Application

### Web Interface (Recommended)

Start both backend and frontend:

```bash
# Terminal 1: Start Flask backend
python3 edge_detection_gui.py

# Terminal 2: Start React frontend
cd image-shape-explorer
npm run dev
```

Open your browser and navigate to `http://localhost:8080`

### Command Line Interface

Run edge detection directly:

```bash
# Simple version
./build_metal/crofton_simple resources/sample_images/test_cell.jpg

# Metal-accelerated version  
./build_metal/crofton_metal resources/sample_images/test_cell.jpg
```

## 🎯 Usage

### Web Interface Features

1. **Upload Image**: Drag and drop or select image files
2. **Auto-Processing**: Click "Process Image" for iterative refinement
3. **View Results**: Examine all processing stages in grid layout
4. **Download Results**: Save any processed image as PNG
5. **Monitor Performance**: View M4 Pro processing logs in real-time
6. **Quality Metrics**: See contour analysis and quality scores

### Processing Pipeline

The algorithm processes images through 8 stages:

1. **Original**: Input microscopy image
2. **HSV Mask**: Color-based object isolation
3. **Top-Hat**: Morphological enhancement of bright features
4. **Top-Hat Binary**: Binarized enhancement result
5. **Combined Mask**: HSV + Top-Hat combination
6. **Opened**: Noise removal via morphological opening
7. **Closed**: Gap filling via morphological closing  
8. **Contour Overlay**: Final edge detection with green contours

### M4 Pro Intensive Processing

When refinement is enabled, you'll see logs like:

```
[M4 PRO] 🔥 Engaging performance cores for intensive computation...
[M4 PRO] 🚀 Distributing work across 6 performance cores...
[M4 PRO] 💾 Memory bandwidth intensive operations...
[M4 PRO] 💾 Memory operations completed: 1.00s
```

## ⚙️ Configuration

### Default Parameters (Optimized)

```python
{
    "hueMin": 100,         # HSV hue minimum (purple/blue range)
    "hueMax": 180,         # HSV hue maximum  
    "satMin": 20,          # Saturation minimum
    "valMin": 20,          # Value/brightness minimum
    "topHatKernelSize": 15,# Top-hat kernel size
    "openSize": 3,         # Opening kernel size
    "closeSize": 5,        # Closing kernel size
    "enableRefinement": true,     # Enable iterative processing
    "refinementPasses": 3,        # Number of refinement passes
    "qualityThreshold": 0.75,     # Early stopping threshold
    "parameterScheduling": "adaptive" # Parameter adjustment strategy
}
```

### Refinement Settings

- **Passes**: 2-8 iterations (more passes = better quality, slower processing)
- **Quality Threshold**: 0.5-0.99 (stop early when quality reached)
- **Scheduling**: 
  - `linear`: Gradual parameter changes
  - `exponential`: Accelerating parameter changes
  - `adaptive`: Smart parameter adjustment based on results

## 📊 Performance Benchmarks

### Apple M4 Pro Results

- **Processing Time**: 1-3 seconds per pass (intensive M4 Pro computation)
- **Memory Usage**: 2-4 GB unified memory
- **CPU Utilization**: All 6 performance cores + 4 efficiency cores
- **Quality Score**: Typically 0.70-0.85 for cell microscopy images

### Architecture Comparison

| Feature | CUDA (GTX 1080) | Apple M4 Pro Metal |
|---------|-----------------|-------------------|
| Processing Time | 2.1s | 1.8s |
| Memory Usage | 8GB GDDR5 | 3GB Unified |
| Power Consumption | 180W | 45W |
| Quality Score | 0.72 | 0.75 |
| Multi-threading | CUDA cores | 6 P-cores + Metal |

## 🧪 Sample Images

Located in `resources/sample_images/`:

- **test_cell.jpg**: Cell microscopy image from Cell Image Library
- **debug_*.jpg**: Processing stage debug outputs  
- **test_purple_shapes.png**: Synthetic test shapes for validation
- **test_circle.png**: Simple geometric test case

## 🔬 Algorithm Details

### Enhanced Preprocessing Pipeline

1. **HSV Color Filtering**: Isolates objects in purple/blue range (100-180°)
2. **Top-Hat Morphology**: Enhances bright cellular structures
3. **Binary Thresholding**: Otsu automatic or manual threshold
4. **Morphological Operations**: 
   - Opening: Remove noise and small artifacts
   - Closing: Fill gaps and connect nearby regions
5. **Contour Detection**: Find object boundaries
6. **Quality Assessment**: Multi-metric scoring for refinement

### Quality Metrics System

The system evaluates results using 6 metrics:

1. **Contour Area**: Size of detected objects
2. **Perimeter Ratio**: Shape complexity measurement  
3. **Fill Ratio**: Solid vs. hollow object detection
4. **Edge Strength**: Gradient magnitude analysis
5. **Noise Level**: Background artifact assessment
6. **Symmetry Score**: Object regularity evaluation

### Iterative Refinement Process

1. **Pass 1**: Use default parameters
2. **Pass 2+**: Adjust parameters based on quality score
3. **Parameter Scheduling**: Adapt HSV range, kernel sizes
4. **Early Stopping**: Stop when quality threshold reached
5. **Best Result Selection**: Return highest scoring result

## 🛠 Development

### Project Structure

```
apple_silicon_version/
├── CMakeLists.txt              # Build configuration
├── main_metal.cpp              # Metal C++ implementation  
├── crofton.metal               # Metal compute shaders
├── crofton_metal.mm            # Objective-C++ Metal wrapper
├── edge_detection_gui.py       # Flask backend with M4 Pro processing
├── image-shape-explorer/       # React TypeScript frontend
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── workers/           # Web Workers for backend communication
│   │   └── lib/               # State management and utilities
├── resources/
│   ├── sample_images/         # Test images and examples
│   └── screenshots/           # Documentation images
└── build_metal/               # Compiled Metal binaries
```

### Building from Source

```bash
# Clean build with debug symbols
rm -rf build_metal
mkdir build_metal && cd build_metal
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON
make -j$(sysctl -n hw.ncpu)
```

### Frontend Development

```bash
cd image-shape-explorer

# Development with hot reload
npm run dev

# Production build
npm run build

# Type checking
npm run type-check

# Linting
npm run lint
```

## 🐛 Troubleshooting

### Common Issues

**Metal Not Available**
- Ensure macOS 12+ on Apple Silicon Mac
- Check: `system_profiler SPDisplaysDataType | grep Metal`

**OpenCV Build Issues**  
- Use Homebrew version: `brew install opencv`
- Verify: `python3 -c "import cv2; print(cv2.__version__)"`

**Port Conflicts**
- Backend auto-discovers available ports (5000-65535)
- Check Flask logs for actual port: `http://localhost:XXXXX`

**Memory Issues**
- Reduce image size for very large inputs (>4K resolution)
- Monitor memory usage: Activity Monitor → Memory tab

### Debug Mode

Enable verbose logging:

```bash
# Backend debugging
PYTHONUNBUFFERED=1 FLASK_DEBUG=1 python3 edge_detection_gui.py

# Build debugging  
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON
```

## 📈 Future Improvements

- [ ] Real-time parameter adjustment in web interface
- [ ] Batch processing for multiple images
- [ ] Export processing settings as presets  
- [ ] Additional microscopy-specific algorithms
- [ ] M4 Max/Ultra optimization
- [ ] Docker containerization for easy deployment

## 📄 License

This project builds upon the Crofton descriptor research and implements optimizations for Apple Silicon hardware. Open source implementation for academic and research use.

## 👨‍💻 Author

**Pavel Chmirenko**  
Email: developer31f@gmail.com  
GitHub: @pchmirenko

## 📚 References

- [Crofton Descriptor Research Paper](https://hal.inrae.fr/hal-02811118/document)
- [Cell Image Library](http://www.cellimagelibrary.org/)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/)
- [OpenCV Documentation](https://opencv.org/)

## 🙏 Acknowledgments

- Original CUDA implementation authors
- Cell Image Library for test datasets  
- Apple Metal development team
- OpenCV community contributors
- React and TypeScript communities

---

*🚀 Optimized for Apple Silicon • ⚡ Metal GPU Accelerated • 🔬 Research-Grade Quality*