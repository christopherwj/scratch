cmake_minimum_required(VERSION 3.18)
project(StockIndicators LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Source files
set(SOURCES
    src/main.cpp
    src/data/data_loader.cpp
    src/strategy/strategy.cpp
    src/backtester/backtester.cpp
    src/utils/utils.cpp
    src/visualization/plotter.cpp
)

# CUDA source files
set(CUDA_SOURCES
    src/indicators/indicators_gpu.cu
)

# Create executable
add_executable(stock_indicators ${SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(stock_indicators Threads::Threads)

# Compiler flags
target_compile_options(stock_indicators PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Set CUDA properties
set_property(TARGET stock_indicators PROPERTY CUDA_SEPARABLE_COMPILATION ON) 