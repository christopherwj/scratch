#pragma once

#include "common.h"
#include "strategy/strategy.h"

// Backtesting configuration
struct BacktestConfig {
    double initial_cash;
    double transaction_cost_pct;
    double slippage_pct;
    bool enable_shorting;
    double margin_requirement;
    double max_position_size;
    
    BacktestConfig() : initial_cash(DEFAULT_INITIAL_CASH), transaction_cost_pct(DEFAULT_TRANSACTION_COST), 
                      slippage_pct(0.0005), enable_shorting(false), margin_requirement(0.5), 
                      max_position_size(1.0) {}
};

// Backtesting results
struct BacktestResult {
    std::vector<PortfolioState> portfolio_history;
    std::vector<Trade> trades;
    PerformanceMetrics performance;
    
    // Additional metrics
    double final_portfolio_value;
    double total_return_pct;
    double annualized_return;
    double max_drawdown_pct;
    double sharpe_ratio;
    double sortino_ratio;
    double calmar_ratio;
    double win_rate;
    double profit_factor;
    double avg_trade_duration;
    
    BacktestResult() : final_portfolio_value(0), total_return_pct(0), annualized_return(0), 
                      max_drawdown_pct(0), sharpe_ratio(0), sortino_ratio(0), calmar_ratio(0), 
                      win_rate(0), profit_factor(0), avg_trade_duration(0) {}
};

// Position tracking
struct Position {
    std::string symbol;
    int shares;
    double avg_price;
    double current_price;
    double unrealized_pnl;
    std::string open_date;
    
    Position() : shares(0), avg_price(0), current_price(0), unrealized_pnl(0) {}
    Position(const std::string& sym, int sh, double price, const std::string& date) 
        : symbol(sym), shares(sh), avg_price(price), current_price(price), unrealized_pnl(0), open_date(date) {}
};

// Portfolio management
class Portfolio {
private:
    double cash;
    std::map<std::string, Position> positions;
    std::vector<PortfolioState> history;
    BacktestConfig config;
    
public:
    Portfolio(const BacktestConfig& cfg) : config(cfg), cash(cfg.initial_cash) {}
    
    // Position management
    bool canBuy(const std::string& symbol, int shares, double price);
    bool canSell(const std::string& symbol, int shares, double price);
    
    void buyStock(const std::string& symbol, int shares, double price, const std::string& date);
    void sellStock(const std::string& symbol, int shares, double price, const std::string& date);
    
    // Portfolio queries
    double getTotalValue(const std::map<std::string, double>& current_prices);
    double getCash() const { return cash; }
    int getPosition(const std::string& symbol) const;
    
    // State management
    void updatePrices(const std::map<std::string, double>& current_prices);
    void recordState(const std::string& date, const std::map<std::string, double>& current_prices);
    
    // History access
    const std::vector<PortfolioState>& getHistory() const { return history; }
    
    // Reset for new backtest
    void reset();
};

// Main backtesting engine
class Backtester {
private:
    BacktestConfig config;
    std::unique_ptr<Portfolio> portfolio;
    std::vector<Trade> trade_history;
    
    // Performance calculation helpers
    double calculateSharpeRatio(const std::vector<double>& returns);
    double calculateSortinoRatio(const std::vector<double>& returns);
    double calculateMaxDrawdown(const std::vector<double>& portfolio_values);
    double calculateCalmarRatio(double annual_return, double max_drawdown);
    double calculateWinRate(const std::vector<Trade>& trades);
    double calculateProfitFactor(const std::vector<Trade>& trades);
    
    // Signal processing
    void processSignal(const SignalResult& signals, size_t index, const std::string& symbol);
    int calculatePositionSize(double available_cash, double price, Signal signal);
    
    // Trade execution
    void executeBuyOrder(const std::string& symbol, double price, const std::string& date);
    void executeSellOrder(const std::string& symbol, double price, const std::string& date);
    
    // Risk management
    bool checkRiskLimits(const std::string& symbol, int shares, double price);
    
public:
    Backtester(const BacktestConfig& config = BacktestConfig());
    ~Backtester();
    
    // Main backtesting functions
    BacktestResult runBacktest(const SignalResult& signals, const std::string& symbol = "STOCK");
    std::vector<BacktestResult> runBatchBacktest(const std::vector<SignalResult>& signal_sets, 
                                               const std::vector<std::string>& symbols);
    
    // Performance analysis
    PerformanceMetrics calculatePerformanceMetrics(const std::vector<PortfolioState>& portfolio_history,
                                                   const std::vector<Trade>& trades);
    
    // Benchmarking
    BacktestResult runBuyAndHoldBenchmark(const std::vector<Price>& prices, 
                                         const std::vector<std::string>& dates);
    
    // Configuration
    void updateConfig(const BacktestConfig& new_config);
    BacktestConfig getConfig() const { return config; }
    
    // Reporting
    void printPerformanceReport(const BacktestResult& result);
    void saveResultsToCSV(const BacktestResult& result, const std::string& filename);
    void saveTradestoCSV(const std::vector<Trade>& trades, const std::string& filename);
    
    // Utility
    void reset();
    
    // Analysis tools
    std::vector<double> calculateDailyReturns(const std::vector<PortfolioState>& portfolio_history);
    std::vector<double> calculateRollingPerformance(const std::vector<PortfolioState>& portfolio_history, 
                                                   int window_days);
}; 