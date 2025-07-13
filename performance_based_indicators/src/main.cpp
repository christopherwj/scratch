#include "common.h"
#include "data/data_loader.h"
#include "indicators/indicators_gpu.h"
#include "strategy/strategy.h"
#include "backtester/backtester.h"
#include <chrono>
#include <iomanip>

void printSystemInfo() {
    std::cout << "=== GPU-Accelerated Stock Trading System ===" << std::endl;
    std::cout << "Language: C++ with CUDA" << std::endl;
    std::cout << "Build: " << __DATE__ << " " << __TIME__ << std::endl;
    
    // Check GPU availability
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
        std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    } else {
        std::cout << "GPU: Not available (using CPU fallback)" << std::endl;
    }
    
    std::cout << "============================================" << std::endl;
}

void runQuickDemo() {
    std::cout << "\n=== Quick Demo: GPU vs CPU Performance ===" << std::endl;
    
    // Create test data
    std::vector<Price> test_prices;
    for (int i = 0; i < 1000; i++) {
        test_prices.push_back(100.0 + std::sin(i * 0.1) * 10.0 + (rand() % 100) / 100.0);
    }
    
    GPUIndicators gpu_indicators;
    
    // Benchmark RSI calculation
    auto start = std::chrono::high_resolution_clock::now();
    auto gpu_rsi = gpu_indicators.calculateRSI(test_prices, 14);
    auto end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    auto cpu_rsi = GPUIndicators::calculateRSI_CPU(test_prices, 14);
    end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "RSI Calculation (1000 data points):" << std::endl;
    std::cout << "GPU Time: " << gpu_time.count() << " μs" << std::endl;
    std::cout << "CPU Time: " << cpu_time.count() << " μs" << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << (double)cpu_time.count() / gpu_time.count() << "x" << std::endl;
    
    // Benchmark MACD calculation
    start = std::chrono::high_resolution_clock::now();
    auto gpu_macd = gpu_indicators.calculateMACD(test_prices, 12, 26, 9);
    end = std::chrono::high_resolution_clock::now();
    gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    auto cpu_macd = GPUIndicators::calculateMACD_CPU(test_prices, 12, 26, 9);
    end = std::chrono::high_resolution_clock::now();
    cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\nMACD Calculation (1000 data points):" << std::endl;
    std::cout << "GPU Time: " << gpu_time.count() << " μs" << std::endl;
    std::cout << "CPU Time: " << cpu_time.count() << " μs" << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << (double)cpu_time.count() / gpu_time.count() << "x" << std::endl;
}

void runFullAnalysis(const std::string& ticker) {
    std::cout << "\n=== Full Analysis for " << ticker << " ===" << std::endl;
    
    // Load data
    DataLoader data_loader;
    auto market_data = data_loader.load_data(ticker);
    
    if (market_data.empty()) {
        std::cerr << "No data available for " << ticker << std::endl;
        return;
    }
    
    data_loader.print_data_summary(market_data, ticker);
    
    // Configure strategy
    StrategyConfig strategy_config;
    strategy_config.params.rsi_period = 14;
    strategy_config.params.macd_fast_period = 12;
    strategy_config.params.macd_slow_period = 26;
    strategy_config.params.macd_signal_period = 9;
    strategy_config.use_gpu = true;
    
    // Generate signals
    std::cout << "\nGenerating trading signals..." << std::endl;
    TradingStrategy strategy(strategy_config);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto signals = strategy.generateSignals(market_data);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto signal_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Signal generation completed in " << signal_time.count() << " ms" << std::endl;
    
    // Analyze signals
    strategy.printSignalSummary(signals);
    strategy.analyzeSignalQuality(signals);
    
    // Save signals
    strategy.saveSignalsToCSV(signals, ticker + "_signals.csv");
    
    // Run backtest
    std::cout << "\nRunning backtest..." << std::endl;
    BacktestConfig backtest_config;
    backtest_config.initial_cash = 10000.0;
    backtest_config.transaction_cost_pct = 0.001;
    
    Backtester backtester(backtest_config);
    
    start_time = std::chrono::high_resolution_clock::now();
    auto backtest_result = backtester.runBacktest(signals, ticker);
    end_time = std::chrono::high_resolution_clock::now();
    auto backtest_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Backtesting completed in " << backtest_time.count() << " ms" << std::endl;
    
    // Print results
    backtester.printPerformanceReport(backtest_result);
    
    // Save results
    backtester.saveResultsToCSV(backtest_result, ticker + "_portfolio.csv");
    backtester.saveTradestoCSV(backtest_result.trades, ticker + "_trades.csv");
    
    // Run buy and hold benchmark
    std::cout << "\nRunning buy-and-hold benchmark..." << std::endl;
    auto prices = data_loader.extract_closing_prices(market_data);
    auto dates = data_loader.extract_dates(market_data);
    
    auto benchmark_result = backtester.runBuyAndHoldBenchmark(prices, dates);
    
    std::cout << "\n=== Strategy vs Buy-and-Hold Comparison ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Strategy Return: " << backtest_result.total_return_pct << "%" << std::endl;
    std::cout << "Buy-and-Hold Return: " << benchmark_result.total_return_pct << "%" << std::endl;
    std::cout << "Strategy Sharpe: " << std::setprecision(3) << backtest_result.sharpe_ratio << std::endl;
    std::cout << "Buy-and-Hold Sharpe: " << std::setprecision(3) << benchmark_result.sharpe_ratio << std::endl;
    std::cout << "Strategy Max Drawdown: " << std::setprecision(2) << backtest_result.max_drawdown_pct << "%" << std::endl;
    std::cout << "Buy-and-Hold Max Drawdown: " << std::setprecision(2) << benchmark_result.max_drawdown_pct << "%" << std::endl;
}

struct OptimizationResult {
    TradingParameters params;
    double win_rate;
    double total_return;
    double sharpe_ratio;
    double max_drawdown;
    int total_trades;
    double avg_win_rate; // Average across all windows
    
    OptimizationResult() : win_rate(0), total_return(0), sharpe_ratio(0), max_drawdown(0), total_trades(0), avg_win_rate(0) {}
};

void runParameterOptimization(const std::string& ticker) {
    std::cout << "\n=== BRUTE FORCE SLIDING WINDOW OPTIMIZATION for " << ticker << " ===" << std::endl;
    std::cout << "Target: MAXIMUM WIN RATE" << std::endl;
    
    // Load data
    DataLoader data_loader;
    auto market_data = data_loader.load_data(ticker);
    
    if (market_data.empty()) {
        std::cerr << "No data available for " << ticker << std::endl;
        return;
    }
    
    std::cout << "Total data points: " << market_data.size() << std::endl;
    
    // Define COMPREHENSIVE parameter grid for brute force search
    std::vector<int> rsi_periods = {7, 10, 14, 17, 21, 25, 28, 32, 35}; // 9 options
    std::vector<int> macd_fast_periods = {8, 10, 12, 14, 16, 18, 20, 22, 24}; // 9 options  
    std::vector<int> macd_slow_periods = {20, 24, 26, 28, 30, 32, 34, 36, 40, 44, 48}; // 11 options
    std::vector<int> macd_signal_periods = {6, 8, 9, 10, 12, 14, 16, 18, 20}; // 9 options
    
    // Build parameter combinations
    std::vector<TradingParameters> param_grid;
    for (int rsi : rsi_periods) {
        for (int fast : macd_fast_periods) {
            for (int slow : macd_slow_periods) {
                for (int signal : macd_signal_periods) {
                    if (slow > fast + 4) { // Ensure meaningful gap between fast and slow
                        param_grid.emplace_back(rsi, fast, slow, signal, 0.6, -0.6);
                    }
                }
            }
        }
    }
    
    std::cout << "BRUTE FORCE GRID: " << param_grid.size() << " parameter combinations!" << std::endl;
    std::cout << "RSI periods: " << rsi_periods.size() << " options" << std::endl;
    std::cout << "MACD combinations: " << macd_fast_periods.size() << "×" << macd_slow_periods.size() << "×" << macd_signal_periods.size() << std::endl;
    
    // Sliding window configuration
    const int window_size = 252 * 2; // 2 years per window
    const int step_size = 63; // Quarter-year steps
    const int min_data_size = 252; // Minimum 1 year of data
    
    if (market_data.size() < window_size + min_data_size) {
        std::cerr << "Not enough data for sliding window optimization" << std::endl;
        return;
    }
    
    std::cout << "SLIDING WINDOW CONFIG:" << std::endl;
    std::cout << "- Window size: " << window_size << " days (2 years)" << std::endl;
    std::cout << "- Step size: " << step_size << " days (quarter)" << std::endl;
    
    // Calculate number of windows
    int num_windows = 0;
    for (size_t start = 0; start + window_size + min_data_size <= market_data.size(); start += step_size) {
        num_windows++;
    }
    std::cout << "- Number of windows: " << num_windows << std::endl;
    std::cout << "- Total tests: " << param_grid.size() << " × " << num_windows << " = " << (param_grid.size() * num_windows) << std::endl;
    
    // Storage for results
    std::vector<OptimizationResult> results(param_grid.size());
    
    TradingStrategy strategy;
    Backtester backtester;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nStarting brute force optimization..." << std::endl;
    
    // Test each parameter combination
    for (size_t param_idx = 0; param_idx < param_grid.size(); param_idx++) {
        const auto& params = param_grid[param_idx];
        
        // Progress reporting
        if (param_idx % 100 == 0) {
            double progress = (double)param_idx / param_grid.size() * 100.0;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress << "% (" 
                      << param_idx << "/" << param_grid.size() << " combinations)" << std::endl;
        }
        
        // Configure strategy
        StrategyConfig config;
        config.params = params;
        config.use_gpu = true;
        strategy.updateConfig(config);
        
        std::vector<double> window_win_rates;
        double total_return_sum = 0.0;
        double sharpe_sum = 0.0;
        double max_drawdown_max = 0.0;
        int total_trades_sum = 0;
        
        // Test across all sliding windows
        int valid_windows = 0;
        for (size_t start = 0; start + window_size + min_data_size <= market_data.size(); start += step_size) {
            // Training window
            std::vector<MarketData> train_data(market_data.begin() + start, 
                                             market_data.begin() + start + window_size);
            
            // Test window  
            std::vector<MarketData> test_data(market_data.begin() + start + window_size,
                                            market_data.begin() + start + window_size + min_data_size);
            
            try {
                // Generate signals on training data (for parameter fitting)
                auto train_signals = strategy.generateSignals(train_data);
                
                // Test on out-of-sample data
                auto test_signals = strategy.generateSignals(test_data);
                auto test_result = backtester.runBacktest(test_signals, ticker);
                
                // Only count windows with actual trades
                if (test_result.trades.size() >= 4) { // At least 2 complete trades (buy-sell pairs)
                    window_win_rates.push_back(test_result.win_rate);
                    total_return_sum += test_result.total_return_pct;
                    sharpe_sum += test_result.sharpe_ratio;
                    max_drawdown_max = std::max(max_drawdown_max, test_result.max_drawdown_pct);
                    total_trades_sum += test_result.trades.size();
                    valid_windows++;
                }
                
            } catch (const std::exception& e) {
                // Skip problematic parameter combinations
                continue;
            }
        }
        
        // Calculate aggregate metrics
        if (valid_windows > 0 && !window_win_rates.empty()) {
            results[param_idx].params = params;
            results[param_idx].avg_win_rate = std::accumulate(window_win_rates.begin(), window_win_rates.end(), 0.0) / window_win_rates.size();
            results[param_idx].win_rate = results[param_idx].avg_win_rate; // Use average for ranking
            results[param_idx].total_return = total_return_sum / valid_windows;
            results[param_idx].sharpe_ratio = sharpe_sum / valid_windows;
            results[param_idx].max_drawdown = max_drawdown_max;
            results[param_idx].total_trades = total_trades_sum / valid_windows;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto optimization_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nBRUTE FORCE OPTIMIZATION COMPLETED!" << std::endl;
    std::cout << "Time taken: " << optimization_time.count() << " seconds" << std::endl;
    
    // Filter out invalid results and sort by win rate
    std::vector<OptimizationResult> valid_results;
    for (const auto& result : results) {
        if (result.total_trades > 0) {
            valid_results.push_back(result);
        }
    }
    
    if (valid_results.empty()) {
        std::cout << "No valid results found!" << std::endl;
        return;
    }
    
    // Sort by win rate (descending)
    std::sort(valid_results.begin(), valid_results.end(),
              [](const auto& a, const auto& b) { return a.win_rate > b.win_rate; });
    
    std::cout << "\n=== TOP 10 HIGHEST WIN RATE PARAMETERS ===" << std::endl;
    std::cout << "Rank | RSI | Fast | Slow | Sig | Win Rate | Return | Sharpe | Drawdown | Trades" << std::endl;
    std::cout << "-----|-----|------|------|-----|----------|--------|--------|----------|-------" << std::endl;
    
    for (size_t i = 0; i < std::min(size_t(10), valid_results.size()); i++) {
        const auto& result = valid_results[i];
        std::cout << std::setw(4) << (i+1) << " | "
                  << std::setw(3) << result.params.rsi_period << " | "
                  << std::setw(4) << result.params.macd_fast_period << " | "
                  << std::setw(4) << result.params.macd_slow_period << " | "
                  << std::setw(3) << result.params.macd_signal_period << " | "
                  << std::setw(7) << std::fixed << std::setprecision(1) << result.win_rate << "% | "
                  << std::setw(6) << std::fixed << std::setprecision(1) << result.total_return << "% | "
                  << std::setw(6) << std::fixed << std::setprecision(2) << result.sharpe_ratio << " | "
                  << std::setw(7) << std::fixed << std::setprecision(1) << result.max_drawdown << "% | "
                  << std::setw(5) << static_cast<int>(result.total_trades) << std::endl;
    }
    
    // Test best parameters on most recent data
    const auto& best_params = valid_results[0].params;
    std::cout << "\n=== TESTING BEST PARAMETERS ON RECENT DATA ===" << std::endl;
    std::cout << "Best Win Rate: " << std::fixed << std::setprecision(1) << valid_results[0].win_rate << "%" << std::endl;
    std::cout << "Parameters: RSI=" << best_params.rsi_period 
              << ", MACD=" << best_params.macd_fast_period 
              << "/" << best_params.macd_slow_period 
              << "/" << best_params.macd_signal_period << std::endl;
    
    // Test on most recent 30% of data
    size_t recent_start = market_data.size() * 0.7;
    std::vector<MarketData> recent_data(market_data.begin() + recent_start, market_data.end());
    
    StrategyConfig best_config;
    best_config.params = best_params;
    best_config.use_gpu = true;
    strategy.updateConfig(best_config);
    
    auto recent_signals = strategy.generateSignals(recent_data);
    auto recent_backtest = backtester.runBacktest(recent_signals, ticker);
    
    std::cout << "\n=== RECENT DATA PERFORMANCE ===" << std::endl;
    backtester.printPerformanceReport(recent_backtest);
}

int main(int argc, char* argv[]) {
    printSystemInfo();
    
    // Parse command line arguments
    std::string ticker = "AAPL";
    std::string mode = "full";
    
    if (argc > 1) {
        ticker = argv[1];
    }
    if (argc > 2) {
        mode = argv[2];
    }
    
    try {
        if (mode == "demo") {
            runQuickDemo();
        } else if (mode == "optimize") {
            runParameterOptimization(ticker);
        } else {
            runFullAnalysis(ticker);
        }
        
        std::cout << "\n=== Analysis Complete ===" << std::endl;
        std::cout << "GPU memory cleanup..." << std::endl;
        GPUMemoryManager::getInstance()->cleanup();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 