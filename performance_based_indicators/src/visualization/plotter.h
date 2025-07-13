#pragma once

#include "common.h"
#include "strategy/strategy.h"
#include "backtester/backtester.h"

// Plotting and visualization utilities
class Plotter {
public:
    // Plotting functions (placeholders for future graphics library integration)
    static void plotSignals(const SignalResult& signals, const std::string& filename);
    static void plotPortfolio(const BacktestResult& result, const std::string& filename);
    static void plotIndicators(const SignalResult& signals, const std::string& filename);
    static void plotPerformanceComparison(const BacktestResult& strategy_result, 
                                        const BacktestResult& benchmark_result,
                                        const std::string& filename);
    static void plotCorrelationMatrix(const std::vector<std::vector<double>>& data,
                                    const std::vector<std::string>& labels,
                                    const std::string& filename);
    
    // Export functions
    static void exportToCSV(const SignalResult& signals, const std::string& filename);
    static void exportToJSON(const BacktestResult& result, const std::string& filename);
    
    // Console-based visualization
    static void printASCIIChart(const std::vector<double>& values, const std::string& title);
    static void printPerformanceTable(const BacktestResult& result);
    
    // Helper functions
    static std::string generateASCIIBar(double value, double max_value, int width);
    static void normalizeValues(std::vector<double>& values);
}; 