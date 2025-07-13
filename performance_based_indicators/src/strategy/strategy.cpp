#include "strategy/strategy.h"
#include <algorithm>
#include <numeric>
#include <iomanip>

TradingStrategy::TradingStrategy(const StrategyConfig& config) : config(config) {
    gpu_indicators = std::make_unique<GPUIndicators>();
    if (config.use_gpu) {
        gpu_indicators->warmupGPU();
    }
}

TradingStrategy::~TradingStrategy() = default;

SignalResult TradingStrategy::generateSignals(const std::vector<MarketData>& market_data) {
    if (market_data.empty()) return SignalResult();
    
    std::vector<Price> prices;
    std::vector<std::string> dates;
    
    prices.reserve(market_data.size());
    dates.reserve(market_data.size());
    
    for (const auto& data : market_data) {
        prices.push_back(data.close);
        dates.push_back(data.date);
    }
    
    return generateSignals(prices, dates);
}

SignalResult TradingStrategy::generateSignals(const std::vector<Price>& prices, const std::vector<std::string>& dates) {
    if (prices.empty()) return SignalResult();
    
    SignalResult result(prices.size());
    
    // Copy input data
    result.prices = prices;
    result.dates = dates;
    
    try {
        // Calculate indicators using GPU
        if (config.use_gpu && gpu_indicators->isGPUAvailable()) {
            result.rsi_values = gpu_indicators->calculateRSI(prices, config.params.rsi_period);
            
            auto macd_result = gpu_indicators->calculateMACD(prices, 
                                                           config.params.macd_fast_period,
                                                           config.params.macd_slow_period,
                                                           config.params.macd_signal_period);
            
            result.macd_line = macd_result.macd_line;
            result.macd_signal = macd_result.signal_line;
            result.macd_histogram = macd_result.histogram;
        } else {
            // Fallback to CPU
            result.rsi_values = GPUIndicators::calculateRSI_CPU(prices, config.params.rsi_period);
            
            auto macd_result = GPUIndicators::calculateMACD_CPU(prices, 
                                                              config.params.macd_fast_period,
                                                              config.params.macd_slow_period,
                                                              config.params.macd_signal_period);
            
            result.macd_line = macd_result.macd_line;
            result.macd_signal = macd_result.signal_line;
            result.macd_histogram = macd_result.histogram;
        }
        
        // Generate crossover signals
        result.signals.resize(prices.size(), 0);
        
        for (size_t i = 1; i < prices.size(); i++) {
            result.signals[i] = generateCrossoverSignal(i, result.macd_line, result.macd_signal, result.rsi_values);
        }
        
        // Apply validation and filtering
        if (config.enable_validation) {
            validateSignals(result);
            filterConsecutiveSignals(result);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating signals: " << e.what() << std::endl;
        result.signals.assign(prices.size(), 0);
    }
    
    return result;
}

Signal TradingStrategy::generateCrossoverSignal(size_t idx, const std::vector<double>& macd_line, 
                                               const std::vector<double>& macd_signal, 
                                               const std::vector<double>& rsi_values) {
    
    if (idx == 0 || idx >= macd_line.size()) return 0;
    
    double macd_current = macd_line[idx];
    double macd_prev = macd_line[idx - 1];
    double signal_current = macd_signal[idx];
    double signal_prev = macd_signal[idx - 1];
    double rsi_value = rsi_values[idx];
    
    // Buy signal: MACD crosses above signal line and RSI is not overbought
    if (isValidBuySignal(rsi_value, macd_current, macd_prev, signal_current, signal_prev)) {
        return 1;
    }
    
    // Sell signal: MACD crosses below signal line and RSI is not oversold
    if (isValidSellSignal(rsi_value, macd_current, macd_prev, signal_current, signal_prev)) {
        return -1;
    }
    
    return 0;
}

bool TradingStrategy::isValidBuySignal(double rsi_value, double macd_current, double macd_prev, 
                                      double signal_current, double signal_prev) {
    // MACD crossover: current MACD > signal AND previous MACD <= signal
    bool macd_crossover = (macd_current > signal_current) && (macd_prev <= signal_prev);
    
    // RSI not overbought
    bool rsi_ok = rsi_value < config.rsi_overbought_threshold;
    
    return macd_crossover && rsi_ok;
}

bool TradingStrategy::isValidSellSignal(double rsi_value, double macd_current, double macd_prev, 
                                       double signal_current, double signal_prev) {
    // MACD crossover: current MACD < signal AND previous MACD >= signal
    bool macd_crossover = (macd_current < signal_current) && (macd_prev >= signal_prev);
    
    // RSI not oversold
    bool rsi_ok = rsi_value > config.rsi_oversold_threshold;
    
    return macd_crossover && rsi_ok;
}

void TradingStrategy::validateSignals(SignalResult& result) {
    for (size_t i = 0; i < result.signals.size(); i++) {
        if (result.signals[i] != 0) {
            // Check for NaN or infinite values
            if (std::isnan(result.rsi_values[i]) || std::isinf(result.rsi_values[i]) ||
                std::isnan(result.macd_line[i]) || std::isinf(result.macd_line[i]) ||
                std::isnan(result.macd_signal[i]) || std::isinf(result.macd_signal[i])) {
                
                result.signals[i] = 0; // Invalidate signal
            }
        }
    }
}

void TradingStrategy::filterConsecutiveSignals(SignalResult& result) {
    if (result.signals.empty()) return;
    
    Signal last_signal = 0;
    
    for (size_t i = 0; i < result.signals.size(); i++) {
        if (result.signals[i] != 0) {
            if (result.signals[i] == last_signal) {
                result.signals[i] = 0; // Remove consecutive signals of the same type
            } else {
                last_signal = result.signals[i];
            }
        }
    }
}

std::vector<SignalResult> TradingStrategy::generateBatchSignals(const std::vector<MarketData>& market_data,
                                                               const std::vector<TradingParameters>& param_sets) {
    std::vector<SignalResult> results;
    results.reserve(param_sets.size());
    
    for (const auto& params : param_sets) {
        // Temporarily update parameters
        TradingParameters original_params = config.params;
        config.params = params;
        
        // Generate signals
        SignalResult result = generateSignals(market_data);
        results.push_back(result);
        
        // Restore original parameters
        config.params = original_params;
    }
    
    return results;
}

void TradingStrategy::updateConfig(const StrategyConfig& new_config) {
    config = new_config;
    if (config.use_gpu) {
        gpu_indicators->warmupGPU();
    }
}

void TradingStrategy::updateParameters(const TradingParameters& new_params) {
    config.params = new_params;
}

void TradingStrategy::analyzeSignalQuality(const SignalResult& result) {
    if (result.signals.empty()) return;
    
    int buy_count = 0, sell_count = 0;
    double avg_rsi_at_buy = 0.0, avg_rsi_at_sell = 0.0;
    
    for (size_t i = 0; i < result.signals.size(); i++) {
        if (result.signals[i] == 1) {
            buy_count++;
            avg_rsi_at_buy += result.rsi_values[i];
        } else if (result.signals[i] == -1) {
            sell_count++;
            avg_rsi_at_sell += result.rsi_values[i];
        }
    }
    
    if (buy_count > 0) avg_rsi_at_buy /= buy_count;
    if (sell_count > 0) avg_rsi_at_sell /= sell_count;
    
    std::cout << "\n=== Signal Quality Analysis ===" << std::endl;
    std::cout << "Buy signals: " << buy_count << " (avg RSI: " << std::fixed << std::setprecision(2) << avg_rsi_at_buy << ")" << std::endl;
    std::cout << "Sell signals: " << sell_count << " (avg RSI: " << std::fixed << std::setprecision(2) << avg_rsi_at_sell << ")" << std::endl;
    std::cout << "Signal frequency: " << std::fixed << std::setprecision(2) << (100.0 * (buy_count + sell_count) / result.signals.size()) << "%" << std::endl;
}

double TradingStrategy::calculateSignalStrength(const SignalResult& result, size_t index) {
    if (index >= result.signals.size() || result.signals[index] == 0) {
        return 0.0;
    }
    
    double rsi_strength = 0.0;
    double macd_strength = 0.0;
    
    // RSI strength based on distance from neutral (50)
    if (result.signals[index] == 1) { // Buy signal
        rsi_strength = (50.0 - result.rsi_values[index]) / 50.0; // Stronger when RSI is lower
    } else { // Sell signal
        rsi_strength = (result.rsi_values[index] - 50.0) / 50.0; // Stronger when RSI is higher
    }
    
    // MACD strength based on histogram magnitude
    macd_strength = std::abs(result.macd_histogram[index]) / 10.0; // Normalize
    
    // Combined strength
    return std::min(1.0, (rsi_strength + macd_strength) / 2.0);
}

TradingStrategy::StrategyMetrics TradingStrategy::calculateMetrics(const SignalResult& result) {
    StrategyMetrics metrics;
    
    metrics.total_signals = std::count_if(result.signals.begin(), result.signals.end(), [](Signal s) { return s != 0; });
    metrics.buy_signals = std::count(result.signals.begin(), result.signals.end(), 1);
    metrics.sell_signals = std::count(result.signals.begin(), result.signals.end(), -1);
    metrics.signal_frequency = result.signals.empty() ? 0.0 : (double)metrics.total_signals / result.signals.size();
    
    // Calculate average signal strength
    double total_strength = 0.0;
    int strength_count = 0;
    
    for (size_t i = 0; i < result.signals.size(); i++) {
        if (result.signals[i] != 0) {
            total_strength += calculateSignalStrength(result, i);
            strength_count++;
        }
    }
    
    metrics.avg_signal_strength = strength_count > 0 ? total_strength / strength_count : 0.0;
    
    // Signal distribution over time
    metrics.signal_distribution.resize(10, 0.0);
    size_t chunk_size = result.signals.size() / 10;
    
    for (size_t i = 0; i < 10; i++) {
        size_t start = i * chunk_size;
        size_t end = (i == 9) ? result.signals.size() : (i + 1) * chunk_size;
        
        int chunk_signals = 0;
        for (size_t j = start; j < end; j++) {
            if (result.signals[j] != 0) chunk_signals++;
        }
        
        metrics.signal_distribution[i] = (double)chunk_signals / (end - start);
    }
    
    return metrics;
}

void TradingStrategy::printSignalSummary(const SignalResult& result) {
    auto metrics = calculateMetrics(result);
    
    std::cout << "\n=== Signal Summary ===" << std::endl;
    std::cout << "Total signals: " << metrics.total_signals << std::endl;
    std::cout << "Buy signals: " << metrics.buy_signals << std::endl;
    std::cout << "Sell signals: " << metrics.sell_signals << std::endl;
    std::cout << "Signal frequency: " << std::fixed << std::setprecision(2) << (metrics.signal_frequency * 100) << "%" << std::endl;
    std::cout << "Average signal strength: " << std::fixed << std::setprecision(3) << metrics.avg_signal_strength << std::endl;
}

void TradingStrategy::saveSignalsToCSV(const SignalResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "Date,Price,RSI,MACD,Signal,Histogram,Signal_Type" << std::endl;
    
    for (size_t i = 0; i < result.signals.size(); i++) {
        file << result.dates[i] << ","
             << std::fixed << std::setprecision(2) << result.prices[i] << ","
             << std::fixed << std::setprecision(2) << result.rsi_values[i] << ","
             << std::fixed << std::setprecision(4) << result.macd_line[i] << ","
             << std::fixed << std::setprecision(4) << result.macd_signal[i] << ","
             << std::fixed << std::setprecision(4) << result.macd_histogram[i] << ",";
        
        if (result.signals[i] == 1) {
            file << "BUY";
        } else if (result.signals[i] == -1) {
            file << "SELL";
        } else {
            file << "HOLD";
        }
        
        file << std::endl;
    }
    
    file.close();
    std::cout << "Signals saved to " << filename << std::endl;
}

bool TradingStrategy::isGPUEnabled() const {
    return config.use_gpu && gpu_indicators->isGPUAvailable();
}

void TradingStrategy::enableGPU(bool enable) {
    config.use_gpu = enable;
    if (enable) {
        gpu_indicators->warmupGPU();
    }
}

void TradingStrategy::warmupGPU() {
    if (config.use_gpu) {
        gpu_indicators->warmupGPU();
    }
} 