#pragma once

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <future>
#include <mutex>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

// Common types
using Price = double;
using Volume = long long;
using Signal = int;
using Timestamp = std::chrono::system_clock::time_point;

// Market data structure
struct MarketData {
    std::string date;
    Price open, high, low, close;
    Volume volume;
    Price adjusted_close;
    
    MarketData() = default;
    MarketData(const std::string& d, Price o, Price h, Price l, Price c, Volume v, Price adj_c = 0)
        : date(d), open(o), high(h), low(l), close(c), volume(v), adjusted_close(adj_c == 0 ? c : adj_c) {}
};

// Performance metrics
struct PerformanceMetrics {
    double total_return_pct;
    double sharpe_ratio;
    double max_drawdown;
    double win_rate;
    int total_trades;
    double avg_trade_return;
    double volatility;
    
    PerformanceMetrics() : total_return_pct(0), sharpe_ratio(0), max_drawdown(0), 
                          win_rate(0), total_trades(0), avg_trade_return(0), volatility(0) {}
};

// Trading parameters
struct TradingParameters {
    int rsi_period;
    int macd_fast_period;
    int macd_slow_period;
    int macd_signal_period;
    double buy_threshold;
    double sell_threshold;
    
    TradingParameters() : rsi_period(14), macd_fast_period(12), macd_slow_period(26), 
                         macd_signal_period(9), buy_threshold(0.6), sell_threshold(-0.6) {}
    
    TradingParameters(int rsi, int macd_fast, int macd_slow, int macd_signal, double buy_th, double sell_th)
        : rsi_period(rsi), macd_fast_period(macd_fast), macd_slow_period(macd_slow), 
          macd_signal_period(macd_signal), buy_threshold(buy_th), sell_threshold(sell_th) {}
};

// Trade record
struct Trade {
    std::string date;
    std::string signal_type; // "BUY" or "SELL"
    Price price;
    int shares;
    double portfolio_value;
    
    Trade() = default;
    Trade(const std::string& d, const std::string& type, Price p, int s, double pv)
        : date(d), signal_type(type), price(p), shares(s), portfolio_value(pv) {}
};

// Portfolio state
struct PortfolioState {
    double cash;
    int shares;
    double total_value;
    std::string date;
    
    PortfolioState() : cash(10000.0), shares(0), total_value(10000.0) {}
    PortfolioState(double c, int s, double tv, const std::string& d) 
        : cash(c), shares(s), total_value(tv), date(d) {}
};

// Constants
constexpr double DEFAULT_INITIAL_CASH = 10000.0;
constexpr double DEFAULT_TRANSACTION_COST = 0.001; // 0.1%
constexpr int DEFAULT_RSI_PERIOD = 14;
constexpr int DEFAULT_MACD_FAST = 12;
constexpr int DEFAULT_MACD_SLOW = 26;
constexpr int DEFAULT_MACD_SIGNAL = 9;
constexpr int TRADING_DAYS_PER_YEAR = 252;

// GPU constants
constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int MAX_GPU_MEMORY_MB = 8192;

// Utility macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0) 