#!/bin/bash
#
# Script de compilación y testing para Metal Optimized
# Ejecutar en macOS con Apple Silicon
#

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Metal Optimization - Build and Test Script               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script must be run on macOS"
    echo "   Current OS: $OSTYPE"
    exit 1
fi

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "⚠️  Warning: Not running on Apple Silicon (arm64)"
    echo "   Current architecture: $ARCH"
    echo "   Metal optimization is designed for Apple Silicon"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Running on macOS $ARCH"
echo ""

# Check for required tools
echo "🔍 Checking prerequisites..."

command -v cmake >/dev/null 2>&1 || { echo "❌ cmake not found. Install with: brew install cmake"; exit 1; }
command -v xcrun >/dev/null 2>&1 || { echo "❌ xcrun not found. Install Xcode Command Line Tools"; exit 1; }
command -v pkg-config >/dev/null 2>&1 || { echo "⚠️  pkg-config not found. Install with: brew install pkg-config"; }

# Check for OpenCV
if ! pkg-config --exists opencv4 2>/dev/null && ! pkg-config --exists opencv 2>/dev/null; then
    echo "❌ OpenCV not found. Install with: brew install opencv"
    exit 1
fi

OPENCV_VERSION=$(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null)
echo "✅ OpenCV $OPENCV_VERSION found"
echo ""

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf build_optimized

# Create build directory
echo "📁 Creating build directory..."
mkdir -p build_optimized
cd build_optimized

# Configure with CMake
echo "⚙️  Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_VERBOSE_MAKEFILE=ON

# Build
echo ""
echo "🔨 Building Metal-optimized version..."
make -j$(sysctl -n hw.ncpu) crofton_optimized

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✅ Build successful!"
echo ""

# Check if test image exists
TEST_IMAGE="../test_cell.jpg"
if [ ! -f "$TEST_IMAGE" ]; then
    echo "⚠️  Test image not found: $TEST_IMAGE"
    echo "   Skipping execution test"
    echo ""
    echo "To test manually:"
    echo "  ./crofton_optimized /path/to/your/image.jpg"
    exit 0
fi

# Run test
echo "🧪 Running test with $TEST_IMAGE..."
echo ""

./crofton_optimized "$TEST_IMAGE" metal_test_output.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                  ✅ TEST SUCCESSFUL                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Results:"

    if [ -f "metal_test_output.txt" ]; then
        echo ""
        grep -E "Total time:|Preprocessing time:|GPU time:" metal_test_output.txt || true
        echo ""
        echo "📄 Full results saved to: metal_test_output.txt"
    fi

    echo ""
    echo "🎉 Metal optimization is working correctly!"
    echo ""
    echo "Next steps:"
    echo "  1. Check the output images (OpenCV windows)"
    echo "  2. Review metal_test_output.txt for detailed metrics"
    echo "  3. Compare performance vs CPU baseline: ./crofton_simple"

else
    echo ""
    echo "❌ Test execution failed"
    exit 1
fi
