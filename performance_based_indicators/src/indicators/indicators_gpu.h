#pragma once

#include "common.h"

// GPU memory management
class GPUMemoryManager {
private:
    static GPUMemoryManager* instance;
    size_t total_memory;
    size_t free_memory;
    
    GPUMemoryManager();
    
public:
    static GPUMemoryManager* getInstance();
    void checkMemory();
    bool canAllocate(size_t bytes);
    void cleanup();
};

// GPU-accelerated indicator calculations
class GPUIndicators {
private:
    // GPU memory pointers
    double* d_prices;
    double* d_rsi_values;
    double* d_macd_line;
    double* d_macd_signal;
    double* d_macd_histogram;
    double* d_ema_fast;
    double* d_ema_slow;
    double* d_temp_buffer;
    
    size_t allocated_size;
    bool gpu_initialized;
    
    // Memory management
    void allocateGPUMemory(size_t size);
    void deallocateGPUMemory();
    void copyToGPU(const std::vector<Price>& prices);
    void copyFromGPU(std::vector<double>& output, double* d_output, size_t size);
    
public:
    GPUIndicators();
    ~GPUIndicators();
    
    // Main indicator functions
    std::vector<double> calculateRSI(const std::vector<Price>& prices, int period = DEFAULT_RSI_PERIOD);
    
    struct MACDResult {
        std::vector<double> macd_line;
        std::vector<double> signal_line;
        std::vector<double> histogram;
    };
    
    MACDResult calculateMACD(const std::vector<Price>& prices, 
                           int fast_period = DEFAULT_MACD_FAST, 
                           int slow_period = DEFAULT_MACD_SLOW, 
                           int signal_period = DEFAULT_MACD_SIGNAL);
    
    // Batch processing for optimization
    struct BatchRSIResult {
        std::vector<std::vector<double>> rsi_values;
        std::vector<TradingParameters> parameters;
    };
    
    BatchRSIResult calculateBatchRSI(const std::vector<Price>& prices, 
                                   const std::vector<int>& periods);
    
    struct BatchMACDResult {
        std::vector<MACDResult> macd_results;
        std::vector<TradingParameters> parameters;
    };
    
    BatchMACDResult calculateBatchMACD(const std::vector<Price>& prices,
                                     const std::vector<int>& fast_periods,
                                     const std::vector<int>& slow_periods,
                                     const std::vector<int>& signal_periods);
    
    // Utility functions
    bool isGPUAvailable();
    void warmupGPU();
    void benchmarkGPU(const std::vector<Price>& prices);
    
    // Fallback CPU implementations
    static std::vector<double> calculateRSI_CPU(const std::vector<Price>& prices, int period);
    static MACDResult calculateMACD_CPU(const std::vector<Price>& prices, int fast_period, int slow_period, int signal_period);
};

// CUDA kernel declarations
extern "C" {
    void launch_rsi_kernel(double* prices, double* rsi_values, int size, int period);
    void launch_ema_kernel(double* prices, double* ema_values, int size, double alpha);
    void launch_macd_kernel(double* prices, double* macd_line, double* signal_line, double* histogram, 
                           int size, int fast_period, int slow_period, int signal_period);
    void launch_batch_rsi_kernel(double* prices, double* rsi_batch, int size, int* periods, int num_periods);
} 