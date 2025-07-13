#pragma once

#include "common.h"
#include "indicators/indicators_gpu.h"

// Signal generation results
struct SignalResult {
    std::vector<std::string> dates;
    std::vector<Price> prices;
    std::vector<double> rsi_values;
    std::vector<double> macd_line;
    std::vector<double> macd_signal;
    std::vector<double> macd_histogram;
    std::vector<Signal> signals; // 1 = BUY, -1 = SELL, 0 = HOLD
    
    SignalResult() = default;
    SignalResult(size_t size) {
        dates.reserve(size);
        prices.reserve(size);
        rsi_values.reserve(size);
        macd_line.reserve(size);
        macd_signal.reserve(size);
        macd_histogram.reserve(size);
        signals.reserve(size);
    }
};

// Strategy configuration
struct StrategyConfig {
    TradingParameters params;
    double rsi_overbought_threshold;
    double rsi_oversold_threshold;
    bool use_gpu;
    bool enable_validation;
    
    StrategyConfig() : rsi_overbought_threshold(70.0), rsi_oversold_threshold(30.0), 
                      use_gpu(true), enable_validation(true) {}
};

// Trading strategy implementation
class TradingStrategy {
private:
    StrategyConfig config;
    std::unique_ptr<GPUIndicators> gpu_indicators;
    
    // Signal generation helpers
    Signal generateCrossoverSignal(size_t idx, const std::vector<double>& macd_line, 
                                  const std::vector<double>& macd_signal, 
                                  const std::vector<double>& rsi_values);
    
    bool isValidBuySignal(double rsi_value, double macd_current, double macd_prev, 
                         double signal_current, double signal_prev);
    
    bool isValidSellSignal(double rsi_value, double macd_current, double macd_prev, 
                          double signal_current, double signal_prev);
    
    // Validation and filtering
    void validateSignals(SignalResult& result);
    void filterConsecutiveSignals(SignalResult& result);
    
    // Performance optimization
    void optimizeForGPU();
    
public:
    TradingStrategy(const StrategyConfig& config = StrategyConfig());
    ~TradingStrategy();
    
    // Main signal generation
    SignalResult generateSignals(const std::vector<MarketData>& market_data);
    SignalResult generateSignals(const std::vector<Price>& prices, const std::vector<std::string>& dates);
    
    // Batch processing for optimization
    std::vector<SignalResult> generateBatchSignals(const std::vector<MarketData>& market_data,
                                                  const std::vector<TradingParameters>& param_sets);
    
    // Configuration
    void updateConfig(const StrategyConfig& new_config);
    void updateParameters(const TradingParameters& new_params);
    StrategyConfig getConfig() const { return config; }
    
    // Strategy analysis
    void analyzeSignalQuality(const SignalResult& result);
    double calculateSignalStrength(const SignalResult& result, size_t index);
    
    // Performance metrics
    struct StrategyMetrics {
        int total_signals;
        int buy_signals;
        int sell_signals;
        double signal_frequency;
        double avg_signal_strength;
        std::vector<double> signal_distribution;
    };
    
    StrategyMetrics calculateMetrics(const SignalResult& result);
    
    // Utility functions
    void printSignalSummary(const SignalResult& result);
    void saveSignalsToCSV(const SignalResult& result, const std::string& filename);
    
    // GPU management
    bool isGPUEnabled() const;
    void enableGPU(bool enable);
    void warmupGPU();
}; 