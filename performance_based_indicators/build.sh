#!/bin/bash

# GPU-Accelerated Stock Trading System Build Script

echo "=== Building GPU-Accelerated Stock Trading System ==="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Warning: CUDA not found. GPU acceleration will be disabled."
    echo "Install CUDA Toolkit 11.0+ for GPU support."
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring build..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Usage examples:"
    echo "  ./stock_indicators AAPL          # Full analysis"
    echo "  ./stock_indicators AAPL demo     # GPU performance demo"
    echo "  ./stock_indicators AAPL optimize # Parameter optimization"
    echo ""
    echo "Make sure your data files are in the ../stock_indicators/data/ directory"
else
    echo "Build failed. Check error messages above."
    exit 1
fi 