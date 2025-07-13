#include "visualization/plotter.h"
#include <algorithm>
#include <iomanip>
#include <fstream>

// Basic plotting functions - placeholder implementations
void Plotter::plotSignals(const SignalResult& signals, const std::string& filename) {
    std::cout << "Plotting signals to " << filename << " (not implemented - use CSV export)" << std::endl;
    // TODO: Implement with graphics library like matplotlib-cpp
}

void Plotter::plotPortfolio(const BacktestResult& result, const std::string& filename) {
    std::cout << "Plotting portfolio to " << filename << " (not implemented - use CSV export)" << std::endl;
    // TODO: Implement with graphics library
}

void Plotter::plotIndicators(const SignalResult& signals, const std::string& filename) {
    std::cout << "Plotting indicators to " << filename << " (not implemented - use CSV export)" << std::endl;
    // TODO: Implement with graphics library
}

void Plotter::plotPerformanceComparison(const BacktestResult& strategy_result, 
                                      const BacktestResult& benchmark_result,
                                      const std::string& filename) {
    std::cout << "Plotting comparison to " << filename << " (not implemented - use CSV export)" << std::endl;
    // TODO: Implement with graphics library
}

void Plotter::plotCorrelationMatrix(const std::vector<std::vector<double>>& data,
                                  const std::vector<std::string>& labels,
                                  const std::string& filename) {
    std::cout << "Plotting correlation matrix to " << filename << " (not implemented)" << std::endl;
    // TODO: Implement with graphics library
}

// Export functions
void Plotter::exportToCSV(const SignalResult& signals, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "Date,Price,RSI,MACD,Signal,Histogram,Signal_Type" << std::endl;
    
    for (size_t i = 0; i < signals.signals.size(); i++) {
        file << signals.dates[i] << ","
             << std::fixed << std::setprecision(2) << signals.prices[i] << ","
             << std::fixed << std::setprecision(2) << signals.rsi_values[i] << ","
             << std::fixed << std::setprecision(4) << signals.macd_line[i] << ","
             << std::fixed << std::setprecision(4) << signals.macd_signal[i] << ","
             << std::fixed << std::setprecision(4) << signals.macd_histogram[i] << ",";
        
        if (signals.signals[i] == 1) {
            file << "BUY";
        } else if (signals.signals[i] == -1) {
            file << "SELL";
        } else {
            file << "HOLD";
        }
        
        file << std::endl;
    }
    
    file.close();
    std::cout << "Signals exported to " << filename << std::endl;
}

void Plotter::exportToJSON(const BacktestResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"performance\": {\n";
    file << "    \"final_portfolio_value\": " << result.final_portfolio_value << ",\n";
    file << "    \"total_return_pct\": " << result.total_return_pct << ",\n";
    file << "    \"sharpe_ratio\": " << result.sharpe_ratio << ",\n";
    file << "    \"max_drawdown_pct\": " << result.max_drawdown_pct << ",\n";
    file << "    \"win_rate\": " << result.win_rate << ",\n";
    file << "    \"total_trades\": " << result.trades.size() << "\n";
    file << "  },\n";
    
    file << "  \"trades\": [\n";
    for (size_t i = 0; i < result.trades.size(); i++) {
        const auto& trade = result.trades[i];
        file << "    {\n";
        file << "      \"date\": \"" << trade.date << "\",\n";
        file << "      \"type\": \"" << trade.signal_type << "\",\n";
        file << "      \"price\": " << trade.price << ",\n";
        file << "      \"shares\": " << trade.shares << ",\n";
        file << "      \"portfolio_value\": " << trade.portfolio_value << "\n";
        file << "    }";
        if (i < result.trades.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ]\n";
    file << "}\n";
    
    file.close();
    std::cout << "Results exported to " << filename << std::endl;
}

// Console-based visualization
void Plotter::printASCIIChart(const std::vector<double>& values, const std::string& title) {
    if (values.empty()) return;
    
    std::cout << "\n=== " << title << " ===" << std::endl;
    
    double min_val = *std::min_element(values.begin(), values.end());
    double max_val = *std::max_element(values.begin(), values.end());
    
    if (max_val == min_val) {
        std::cout << "All values are the same: " << min_val << std::endl;
        return;
    }
    
    int chart_height = 20;
    int chart_width = std::min(80, (int)values.size());
    
    // Sample values if too many
    std::vector<double> display_values;
    if (values.size() > chart_width) {
        int step = values.size() / chart_width;
        for (int i = 0; i < chart_width; i++) {
            display_values.push_back(values[i * step]);
        }
    } else {
        display_values = values;
    }
    
    // Print chart
    for (int row = chart_height - 1; row >= 0; row--) {
        double threshold = min_val + (max_val - min_val) * row / (chart_height - 1);
        
        std::cout << std::setw(8) << std::fixed << std::setprecision(1) << threshold << " |";
        
        for (double val : display_values) {
            if (val >= threshold) {
                std::cout << "*";
            } else {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // Print x-axis
    std::cout << std::setw(8) << " " << " +";
    for (size_t i = 0; i < display_values.size(); i++) {
        std::cout << "-";
    }
    std::cout << std::endl;
    
    std::cout << "Min: " << min_val << ", Max: " << max_val << ", Count: " << values.size() << std::endl;
}

void Plotter::printPerformanceTable(const BacktestResult& result) {
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << std::setw(25) << std::left << "Metric" << std::setw(15) << std::right << "Value" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    std::cout << std::setw(25) << std::left << "Final Portfolio Value" 
              << std::setw(15) << std::right << std::fixed << std::setprecision(2) << result.final_portfolio_value << std::endl;
    
    std::cout << std::setw(25) << std::left << "Total Return (%)" 
              << std::setw(15) << std::right << std::fixed << std::setprecision(2) << result.total_return_pct << std::endl;
    
    std::cout << std::setw(25) << std::left << "Annualized Return (%)" 
              << std::setw(15) << std::right << std::fixed << std::setprecision(2) << result.annualized_return << std::endl;
    
    std::cout << std::setw(25) << std::left << "Sharpe Ratio" 
              << std::setw(15) << std::right << std::fixed << std::setprecision(3) << result.sharpe_ratio << std::endl;
    
    std::cout << std::setw(25) << std::left << "Max Drawdown (%)" 
              << std::setw(15) << std::right << std::fixed << std::setprecision(2) << result.max_drawdown_pct << std::endl;
    
    std::cout << std::setw(25) << std::left << "Win Rate (%)" 
              << std::setw(15) << std::right << std::fixed << std::setprecision(2) << result.win_rate << std::endl;
    
    std::cout << std::setw(25) << std::left << "Total Trades" 
              << std::setw(15) << std::right << result.trades.size() << std::endl;
    
    // Plot portfolio value over time
    if (!result.portfolio_history.empty()) {
        std::vector<double> portfolio_values;
        for (const auto& state : result.portfolio_history) {
            portfolio_values.push_back(state.total_value);
        }
        printASCIIChart(portfolio_values, "Portfolio Value Over Time");
    }
}

// Helper functions
std::string Plotter::generateASCIIBar(double value, double max_value, int width) {
    if (max_value <= 0) return std::string(width, ' ');
    
    int bar_length = (int)((value / max_value) * width);
    bar_length = std::max(0, std::min(width, bar_length));
    
    return std::string(bar_length, '=') + std::string(width - bar_length, ' ');
}

void Plotter::normalizeValues(std::vector<double>& values) {
    if (values.empty()) return;
    
    double min_val = *std::min_element(values.begin(), values.end());
    double max_val = *std::max_element(values.begin(), values.end());
    
    if (max_val == min_val) return;
    
    for (double& val : values) {
        val = (val - min_val) / (max_val - min_val);
    }
} 