#include "indicators/indicators_gpu.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// GPU Memory Manager implementation
GPUMemoryManager* GPUMemoryManager::instance = nullptr;

GPUMemoryManager::GPUMemoryManager() {
    checkMemory();
}

GPUMemoryManager* GPUMemoryManager::getInstance() {
    if (!instance) {
        instance = new GPUMemoryManager();
    }
    return instance;
}

void GPUMemoryManager::checkMemory() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    free_memory = free_mem;
    total_memory = total_mem;
    
    std::cout << "GPU Memory: " << total_memory / (1024 * 1024) << " MB total, " 
              << free_memory / (1024 * 1024) << " MB free" << std::endl;
}

bool GPUMemoryManager::canAllocate(size_t bytes) {
    checkMemory();
    return bytes < (free_memory * 0.8); // Leave 20% buffer
}

void GPUMemoryManager::cleanup() {
    CUDA_CHECK(cudaDeviceReset());
}

// CUDA Kernels

__global__ void rsi_kernel(double* prices, double* rsi_values, int size, int period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size || idx < period) {
        if (idx < size) rsi_values[idx] = 50.0; // Default RSI value
        return;
    }
    
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    
    // Calculate initial average gain and loss
    for (int i = 1; i <= period; i++) {
        double price_change = prices[idx - period + i] - prices[idx - period + i - 1];
        if (price_change > 0) {
            avg_gain += price_change;
        } else {
            avg_loss += -price_change;
        }
    }
    
    avg_gain /= period;
    avg_loss /= period;
    
    // Calculate RSI
    if (avg_loss == 0.0) {
        rsi_values[idx] = 100.0;
    } else {
        double rs = avg_gain / avg_loss;
        rsi_values[idx] = 100.0 - (100.0 / (1.0 + rs));
    }
}

__global__ void ema_kernel(double* prices, double* ema_values, int size, double alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    if (idx == 0) {
        ema_values[idx] = prices[idx];
    } else {
        ema_values[idx] = alpha * prices[idx] + (1.0 - alpha) * ema_values[idx - 1];
    }
}

__global__ void macd_kernel(double* prices, double* macd_line, double* signal_line, double* histogram,
                           int size, int fast_period, int slow_period, int signal_period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    // Calculate MACD line (fast EMA - slow EMA)
    // This is a simplified version - in practice, you'd want separate EMA kernels
    double fast_alpha = 2.0 / (fast_period + 1.0);
    double slow_alpha = 2.0 / (slow_period + 1.0);
    double signal_alpha = 2.0 / (signal_period + 1.0);
    
    // Initialize EMAs
    if (idx == 0) {
        macd_line[idx] = 0.0;
        signal_line[idx] = 0.0;
        histogram[idx] = 0.0;
        return;
    }
    
    // Calculate fast and slow EMAs iteratively
    double fast_ema = prices[0];
    double slow_ema = prices[0];
    
    for (int i = 1; i <= idx; i++) {
        fast_ema = fast_alpha * prices[i] + (1.0 - fast_alpha) * fast_ema;
        slow_ema = slow_alpha * prices[i] + (1.0 - slow_alpha) * slow_ema;
    }
    
    macd_line[idx] = fast_ema - slow_ema;
    
    // Calculate signal line
    if (idx == 0) {
        signal_line[idx] = macd_line[idx];
    } else {
        signal_line[idx] = signal_alpha * macd_line[idx] + (1.0 - signal_alpha) * signal_line[idx - 1];
    }
    
    // Calculate histogram
    histogram[idx] = macd_line[idx] - signal_line[idx];
}

__global__ void batch_rsi_kernel(double* prices, double* rsi_batch, int size, int* periods, int num_periods) {
    int period_idx = blockIdx.x;
    int price_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (period_idx >= num_periods || price_idx >= size) return;
    
    int period = periods[period_idx];
    int output_idx = period_idx * size + price_idx;
    
    if (price_idx < period) {
        rsi_batch[output_idx] = 50.0;
        return;
    }
    
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    
    // Calculate average gain and loss
    for (int i = 1; i <= period; i++) {
        double price_change = prices[price_idx - period + i] - prices[price_idx - period + i - 1];
        if (price_change > 0) {
            avg_gain += price_change;
        } else {
            avg_loss += -price_change;
        }
    }
    
    avg_gain /= period;
    avg_loss /= period;
    
    // Calculate RSI
    if (avg_loss == 0.0) {
        rsi_batch[output_idx] = 100.0;
    } else {
        double rs = avg_gain / avg_loss;
        rsi_batch[output_idx] = 100.0 - (100.0 / (1.0 + rs));
    }
}

// Kernel launch functions
extern "C" {
    void launch_rsi_kernel(double* prices, double* rsi_values, int size, int period) {
        int threadsPerBlock = CUDA_BLOCK_SIZE;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        
        rsi_kernel<<<blocksPerGrid, threadsPerBlock>>>(prices, rsi_values, size, period);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void launch_ema_kernel(double* prices, double* ema_values, int size, double alpha) {
        int threadsPerBlock = CUDA_BLOCK_SIZE;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        
        ema_kernel<<<blocksPerGrid, threadsPerBlock>>>(prices, ema_values, size, alpha);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void launch_macd_kernel(double* prices, double* macd_line, double* signal_line, double* histogram,
                           int size, int fast_period, int slow_period, int signal_period) {
        int threadsPerBlock = CUDA_BLOCK_SIZE;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        
        macd_kernel<<<blocksPerGrid, threadsPerBlock>>>(prices, macd_line, signal_line, histogram,
                                                       size, fast_period, slow_period, signal_period);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void launch_batch_rsi_kernel(double* prices, double* rsi_batch, int size, int* periods, int num_periods) {
        dim3 threadsPerBlock(CUDA_BLOCK_SIZE);
        dim3 blocksPerGrid(num_periods, (size + threadsPerBlock.x - 1) / threadsPerBlock.x);
        
        batch_rsi_kernel<<<blocksPerGrid, threadsPerBlock>>>(prices, rsi_batch, size, periods, num_periods);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// GPUIndicators implementation
GPUIndicators::GPUIndicators() : allocated_size(0), gpu_initialized(false) {
    d_prices = nullptr;
    d_rsi_values = nullptr;
    d_macd_line = nullptr;
    d_macd_signal = nullptr;
    d_macd_histogram = nullptr;
    d_ema_fast = nullptr;
    d_ema_slow = nullptr;
    d_temp_buffer = nullptr;
    
    // Initialize GPU
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount > 0) {
        CUDA_CHECK(cudaSetDevice(0));
        gpu_initialized = true;
        std::cout << "GPU initialized successfully" << std::endl;
    } else {
        std::cout << "No GPU devices found, falling back to CPU" << std::endl;
    }
}

GPUIndicators::~GPUIndicators() {
    deallocateGPUMemory();
}

void GPUIndicators::allocateGPUMemory(size_t size) {
    if (allocated_size >= size) return;
    
    deallocateGPUMemory();
    
    size_t bytes_needed = size * sizeof(double);
    GPUMemoryManager* mem_manager = GPUMemoryManager::getInstance();
    
    if (!mem_manager->canAllocate(bytes_needed * 8)) { // 8 arrays needed
        throw std::runtime_error("Insufficient GPU memory");
    }
    
    CUDA_CHECK(cudaMalloc(&d_prices, bytes_needed));
    CUDA_CHECK(cudaMalloc(&d_rsi_values, bytes_needed));
    CUDA_CHECK(cudaMalloc(&d_macd_line, bytes_needed));
    CUDA_CHECK(cudaMalloc(&d_macd_signal, bytes_needed));
    CUDA_CHECK(cudaMalloc(&d_macd_histogram, bytes_needed));
    CUDA_CHECK(cudaMalloc(&d_ema_fast, bytes_needed));
    CUDA_CHECK(cudaMalloc(&d_ema_slow, bytes_needed));
    CUDA_CHECK(cudaMalloc(&d_temp_buffer, bytes_needed));
    
    allocated_size = size;
}

void GPUIndicators::deallocateGPUMemory() {
    if (d_prices) { cudaFree(d_prices); d_prices = nullptr; }
    if (d_rsi_values) { cudaFree(d_rsi_values); d_rsi_values = nullptr; }
    if (d_macd_line) { cudaFree(d_macd_line); d_macd_line = nullptr; }
    if (d_macd_signal) { cudaFree(d_macd_signal); d_macd_signal = nullptr; }
    if (d_macd_histogram) { cudaFree(d_macd_histogram); d_macd_histogram = nullptr; }
    if (d_ema_fast) { cudaFree(d_ema_fast); d_ema_fast = nullptr; }
    if (d_ema_slow) { cudaFree(d_ema_slow); d_ema_slow = nullptr; }
    if (d_temp_buffer) { cudaFree(d_temp_buffer); d_temp_buffer = nullptr; }
    
    allocated_size = 0;
}

void GPUIndicators::copyToGPU(const std::vector<Price>& prices) {
    std::vector<double> double_prices(prices.begin(), prices.end());
    CUDA_CHECK(cudaMemcpy(d_prices, double_prices.data(), 
                         double_prices.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void GPUIndicators::copyFromGPU(std::vector<double>& output, double* d_output, size_t size) {
    output.resize(size);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, 
                         size * sizeof(double), cudaMemcpyDeviceToHost));
}

std::vector<double> GPUIndicators::calculateRSI(const std::vector<Price>& prices, int period) {
    if (!gpu_initialized) {
        return calculateRSI_CPU(prices, period);
    }
    
    try {
        allocateGPUMemory(prices.size());
        copyToGPU(prices);
        
        launch_rsi_kernel(d_prices, d_rsi_values, prices.size(), period);
        
        std::vector<double> rsi_values;
        copyFromGPU(rsi_values, d_rsi_values, prices.size());
        
        return rsi_values;
    } catch (const std::exception& e) {
        std::cerr << "GPU RSI calculation failed: " << e.what() << ", falling back to CPU" << std::endl;
        return calculateRSI_CPU(prices, period);
    }
}

GPUIndicators::MACDResult GPUIndicators::calculateMACD(const std::vector<Price>& prices, 
                                                     int fast_period, int slow_period, int signal_period) {
    if (!gpu_initialized) {
        return calculateMACD_CPU(prices, fast_period, slow_period, signal_period);
    }
    
    try {
        allocateGPUMemory(prices.size());
        copyToGPU(prices);
        
        launch_macd_kernel(d_prices, d_macd_line, d_macd_signal, d_macd_histogram,
                          prices.size(), fast_period, slow_period, signal_period);
        
        MACDResult result;
        copyFromGPU(result.macd_line, d_macd_line, prices.size());
        copyFromGPU(result.signal_line, d_macd_signal, prices.size());
        copyFromGPU(result.histogram, d_macd_histogram, prices.size());
        
        return result;
    } catch (const std::exception& e) {
        std::cerr << "GPU MACD calculation failed: " << e.what() << ", falling back to CPU" << std::endl;
        return calculateMACD_CPU(prices, fast_period, slow_period, signal_period);
    }
}

bool GPUIndicators::isGPUAvailable() {
    return gpu_initialized;
}

void GPUIndicators::warmupGPU() {
    if (!gpu_initialized) return;
    
    // Warm up with a small calculation
    std::vector<Price> dummy_prices(100);
    std::iota(dummy_prices.begin(), dummy_prices.end(), 100.0);
    
    calculateRSI(dummy_prices, 14);
    calculateMACD(dummy_prices, 12, 26, 9);
}

// CPU fallback implementations
std::vector<double> GPUIndicators::calculateRSI_CPU(const std::vector<Price>& prices, int period) {
    std::vector<double> rsi_values(prices.size(), 50.0);
    
    if (prices.size() < period + 1) return rsi_values;
    
    for (size_t i = period; i < prices.size(); i++) {
        double avg_gain = 0.0;
        double avg_loss = 0.0;
        
        for (int j = 1; j <= period; j++) {
            double price_change = prices[i - period + j] - prices[i - period + j - 1];
            if (price_change > 0) {
                avg_gain += price_change;
            } else {
                avg_loss += -price_change;
            }
        }
        
        avg_gain /= period;
        avg_loss /= period;
        
        if (avg_loss == 0.0) {
            rsi_values[i] = 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    
    return rsi_values;
}

GPUIndicators::MACDResult GPUIndicators::calculateMACD_CPU(const std::vector<Price>& prices, 
                                                         int fast_period, int slow_period, int signal_period) {
    MACDResult result;
    
    if (prices.empty()) return result;
    
    // Calculate EMAs
    std::vector<double> fast_ema(prices.size());
    std::vector<double> slow_ema(prices.size());
    
    double fast_alpha = 2.0 / (fast_period + 1.0);
    double slow_alpha = 2.0 / (slow_period + 1.0);
    double signal_alpha = 2.0 / (signal_period + 1.0);
    
    fast_ema[0] = slow_ema[0] = prices[0];
    
    for (size_t i = 1; i < prices.size(); i++) {
        fast_ema[i] = fast_alpha * prices[i] + (1.0 - fast_alpha) * fast_ema[i - 1];
        slow_ema[i] = slow_alpha * prices[i] + (1.0 - slow_alpha) * slow_ema[i - 1];
    }
    
    // Calculate MACD line
    result.macd_line.resize(prices.size());
    for (size_t i = 0; i < prices.size(); i++) {
        result.macd_line[i] = fast_ema[i] - slow_ema[i];
    }
    
    // Calculate signal line
    result.signal_line.resize(prices.size());
    result.signal_line[0] = result.macd_line[0];
    
    for (size_t i = 1; i < prices.size(); i++) {
        result.signal_line[i] = signal_alpha * result.macd_line[i] + (1.0 - signal_alpha) * result.signal_line[i - 1];
    }
    
    // Calculate histogram
    result.histogram.resize(prices.size());
    for (size_t i = 0; i < prices.size(); i++) {
        result.histogram[i] = result.macd_line[i] - result.signal_line[i];
    }
    
    return result;
} 