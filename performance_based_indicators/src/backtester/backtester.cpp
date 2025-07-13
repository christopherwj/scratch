#include "backtester/backtester.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>

// Portfolio implementation
bool Portfolio::canBuy(const std::string& symbol, int shares, double price) {
    double total_cost = shares * price * (1.0 + config.transaction_cost_pct);
    return cash >= total_cost;
}

bool Portfolio::canSell(const std::string& symbol, int shares, double price) {
    auto it = positions.find(symbol);
    if (it == positions.end()) return false;
    return it->second.shares >= shares;
}

void Portfolio::buyStock(const std::string& symbol, int shares, double price, const std::string& date) {
    if (!canBuy(symbol, shares, price)) return;
    
    double total_cost = shares * price * (1.0 + config.transaction_cost_pct);
    cash -= total_cost;
    
    auto it = positions.find(symbol);
    if (it != positions.end()) {
        // Update existing position
        int old_shares = it->second.shares;
        double old_avg_price = it->second.avg_price;
        
        int new_shares = old_shares + shares;
        double new_avg_price = ((old_shares * old_avg_price) + (shares * price)) / new_shares;
        
        it->second.shares = new_shares;
        it->second.avg_price = new_avg_price;
        it->second.current_price = price;
    } else {
        // Create new position
        positions[symbol] = Position(symbol, shares, price, date);
    }
}

void Portfolio::sellStock(const std::string& symbol, int shares, double price, const std::string& date) {
    if (!canSell(symbol, shares, price)) return;
    
    double total_revenue = shares * price * (1.0 - config.transaction_cost_pct);
    cash += total_revenue;
    
    auto it = positions.find(symbol);
    if (it != positions.end()) {
        it->second.shares -= shares;
        it->second.current_price = price;
        
        if (it->second.shares == 0) {
            positions.erase(it);
        }
    }
}

double Portfolio::getTotalValue(const std::map<std::string, double>& current_prices) {
    double total = cash;
    
    for (auto& [symbol, position] : positions) {
        auto price_it = current_prices.find(symbol);
        if (price_it != current_prices.end()) {
            total += position.shares * price_it->second;
        }
    }
    
    return total;
}

int Portfolio::getPosition(const std::string& symbol) const {
    auto it = positions.find(symbol);
    return (it != positions.end()) ? it->second.shares : 0;
}

void Portfolio::updatePrices(const std::map<std::string, double>& current_prices) {
    for (auto& [symbol, position] : positions) {
        auto price_it = current_prices.find(symbol);
        if (price_it != current_prices.end()) {
            position.current_price = price_it->second;
            position.unrealized_pnl = position.shares * (position.current_price - position.avg_price);
        }
    } 
}

void Portfolio::recordState(const std::string& date, const std::map<std::string, double>& current_prices) {
    updatePrices(current_prices);
    
    int total_shares = 0;
    for (const auto& [symbol, position] : positions) {
        total_shares += position.shares;
    }
    
    double total_value = getTotalValue(current_prices);
    history.emplace_back(cash, total_shares, total_value, date);
}

void Portfolio::reset() {
    cash = config.initial_cash;
    positions.clear();
    history.clear();
}

// Backtester implementation
Backtester::Backtester(const BacktestConfig& config) : config(config) {
    portfolio = std::make_unique<Portfolio>(config);
}

Backtester::~Backtester() = default;

BacktestResult Backtester::runBacktest(const SignalResult& signals, const std::string& symbol) {
    BacktestResult result;
    
    if (signals.signals.empty()) return result;
    
    // Reset portfolio
    portfolio->reset();
    trade_history.clear();
    
    // Process each signal
    for (size_t i = 0; i < signals.signals.size(); i++) {
        if (signals.signals[i] != 0) {
            processSignal(signals, i, symbol);
        }
        
        // Update portfolio state
        std::map<std::string, double> current_prices;
        current_prices[symbol] = signals.prices[i];
        portfolio->recordState(signals.dates[i], current_prices);
    }
    
    // Calculate performance metrics
    result.portfolio_history = portfolio->getHistory();
    result.trades = trade_history;
    result.performance = calculatePerformanceMetrics(result.portfolio_history, result.trades);
    
    // Calculate additional metrics
    if (!result.portfolio_history.empty()) {
        result.final_portfolio_value = result.portfolio_history.back().total_value;
        result.total_return_pct = ((result.final_portfolio_value / config.initial_cash) - 1.0) * 100.0;
        
        // Calculate annualized return
        double days = result.portfolio_history.size();
        double years = days / 252.0; // Trading days per year
        result.annualized_return = (std::pow(result.final_portfolio_value / config.initial_cash, 1.0 / years) - 1.0) * 100.0;
        
        // Calculate other metrics
        std::vector<double> portfolio_values;
        for (const auto& state : result.portfolio_history) {
            portfolio_values.push_back(state.total_value);
        }
        
        std::vector<double> daily_returns = calculateDailyReturns(result.portfolio_history);
        result.max_drawdown_pct = calculateMaxDrawdown(portfolio_values);
        result.sharpe_ratio = calculateSharpeRatio(daily_returns);
        result.sortino_ratio = calculateSortinoRatio(daily_returns);
        result.calmar_ratio = calculateCalmarRatio(result.annualized_return, result.max_drawdown_pct);
        result.win_rate = calculateWinRate(result.trades);
        result.profit_factor = calculateProfitFactor(result.trades);
    }
    
    return result;
}

void Backtester::processSignal(const SignalResult& signals, size_t index, const std::string& symbol) {
    Signal signal = signals.signals[index];
    double price = signals.prices[index];
    std::string date = signals.dates[index];
    
    if (signal == 1) { // Buy signal
        executeBuyOrder(symbol, price, date);
    } else if (signal == -1) { // Sell signal
        executeSellOrder(symbol, price, date);
    }
}

int Backtester::calculatePositionSize(double available_cash, double price, Signal signal) {
    // Simple position sizing: use max position size percentage
    double max_investment = available_cash * config.max_position_size;
    int max_shares = static_cast<int>(max_investment / price);
    
    return std::max(1, max_shares); // At least 1 share
}

void Backtester::executeBuyOrder(const std::string& symbol, double price, const std::string& date) {
    double available_cash = portfolio->getCash();
    
    if (available_cash < price * (1.0 + config.transaction_cost_pct)) {
        return; // Not enough cash
    }
    
    int shares = calculatePositionSize(available_cash, price, 1);
    
    if (portfolio->canBuy(symbol, shares, price) && checkRiskLimits(symbol, shares, price)) {
        portfolio->buyStock(symbol, shares, price, date);
        
        // Record trade
        double portfolio_value = portfolio->getTotalValue({{symbol, price}});
        trade_history.emplace_back(date, "BUY", price, shares, portfolio_value);
    }
}

void Backtester::executeSellOrder(const std::string& symbol, double price, const std::string& date) {
    int current_position = portfolio->getPosition(symbol);
    
    if (current_position > 0) {
        int shares_to_sell = current_position; // Sell all shares
        
        if (portfolio->canSell(symbol, shares_to_sell, price)) {
            portfolio->sellStock(symbol, shares_to_sell, price, date);
            
            // Record trade
            double portfolio_value = portfolio->getTotalValue({{symbol, price}});
            trade_history.emplace_back(date, "SELL", price, shares_to_sell, portfolio_value);
        }
    }
}

bool Backtester::checkRiskLimits(const std::string& symbol, int shares, double price) {
    // Simple risk check: ensure position size doesn't exceed max position size
    double position_value = shares * price;
    double total_portfolio_value = portfolio->getTotalValue({{symbol, price}});
    
    return (position_value / total_portfolio_value) <= config.max_position_size;
}

PerformanceMetrics Backtester::calculatePerformanceMetrics(const std::vector<PortfolioState>& portfolio_history,
                                                          const std::vector<Trade>& trades) {
    PerformanceMetrics metrics;
    
    if (portfolio_history.empty()) return metrics;
    
    // Calculate returns
    double final_value = portfolio_history.back().total_value;
    double initial_value = config.initial_cash;
    metrics.total_return_pct = ((final_value / initial_value) - 1.0) * 100.0;
    
    // Calculate daily returns for Sharpe ratio
    std::vector<double> daily_returns = calculateDailyReturns(portfolio_history);
    metrics.sharpe_ratio = calculateSharpeRatio(daily_returns);
    
    // Calculate max drawdown
    std::vector<double> portfolio_values;
    for (const auto& state : portfolio_history) {
        portfolio_values.push_back(state.total_value);
    }
    metrics.max_drawdown = calculateMaxDrawdown(portfolio_values);
    
    // Calculate trade metrics
    if (!trades.empty()) {
        metrics.total_trades = trades.size();
        metrics.win_rate = calculateWinRate(trades);
        
        // Calculate average trade return
        double total_trade_return = 0.0;
        for (size_t i = 1; i < trades.size(); i += 2) { // Assuming buy-sell pairs
            if (i < trades.size() && trades[i-1].signal_type == "BUY" && trades[i].signal_type == "SELL") {
                double trade_return = (trades[i].price - trades[i-1].price) / trades[i-1].price;
                total_trade_return += trade_return;
            }
        }
        metrics.avg_trade_return = total_trade_return / (trades.size() / 2);
    }
    
    // Calculate volatility
    if (daily_returns.size() > 1) {
        double mean_return = std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0) / daily_returns.size();
        double variance = 0.0;
        for (double ret : daily_returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        variance /= (daily_returns.size() - 1);
        metrics.volatility = std::sqrt(variance * 252); // Annualized volatility
    }
    
    return metrics;
}

double Backtester::calculateSharpeRatio(const std::vector<double>& returns) {
    if (returns.empty()) return 0.0;
    
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= returns.size();
    
    double std_dev = std::sqrt(variance);
    
    if (std_dev == 0.0) return 0.0;
    
    // Annualized Sharpe ratio (assuming risk-free rate of 0)
    return (mean_return / std_dev) * std::sqrt(252);
}

double Backtester::calculateSortinoRatio(const std::vector<double>& returns) {
    if (returns.empty()) return 0.0;
    
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    
    double downside_variance = 0.0;
    int negative_count = 0;
    
    for (double ret : returns) {
        if (ret < 0) {
            downside_variance += ret * ret;
            negative_count++;
        }
    }
    
    if (negative_count == 0) return 0.0;
    
    downside_variance /= negative_count;
    double downside_std = std::sqrt(downside_variance);
    
    if (downside_std == 0.0) return 0.0;
    
    return (mean_return / downside_std) * std::sqrt(252);
}

double Backtester::calculateMaxDrawdown(const std::vector<double>& portfolio_values) {
    if (portfolio_values.empty()) return 0.0;
    
    double max_drawdown = 0.0;
    double peak = portfolio_values[0];
    
    for (double value : portfolio_values) {
        if (value > peak) {
            peak = value;
        }
        
        double drawdown = (peak - value) / peak;
        max_drawdown = std::max(max_drawdown, drawdown);
    }
    
    return max_drawdown * 100.0; // Return as percentage
}

double Backtester::calculateCalmarRatio(double annual_return, double max_drawdown) {
    if (max_drawdown == 0.0) return 0.0;
    return annual_return / max_drawdown;
}

double Backtester::calculateWinRate(const std::vector<Trade>& trades) {
    if (trades.size() < 2) return 0.0;
    
    int winning_trades = 0;
    int total_trades = 0;
    
    for (size_t i = 1; i < trades.size(); i += 2) {
        if (i < trades.size() && trades[i-1].signal_type == "BUY" && trades[i].signal_type == "SELL") {
            total_trades++;
            if (trades[i].price > trades[i-1].price) {
                winning_trades++;
            }
        }
    }
    
    return total_trades > 0 ? (double)winning_trades / total_trades * 100.0 : 0.0;
}

double Backtester::calculateProfitFactor(const std::vector<Trade>& trades) {
    if (trades.size() < 2) return 0.0;
    
    double gross_profit = 0.0;
    double gross_loss = 0.0;
    
    for (size_t i = 1; i < trades.size(); i += 2) {
        if (i < trades.size() && trades[i-1].signal_type == "BUY" && trades[i].signal_type == "SELL") {
            double profit = (trades[i].price - trades[i-1].price) * trades[i].shares;
            if (profit > 0) {
                gross_profit += profit;
            } else {
                gross_loss += -profit;
            }
        }
    }
    
    return gross_loss > 0 ? gross_profit / gross_loss : 0.0;
}

BacktestResult Backtester::runBuyAndHoldBenchmark(const std::vector<Price>& prices, 
                                                 const std::vector<std::string>& dates) {
    BacktestResult result;
    
    if (prices.empty()) return result;
    
    // Reset portfolio
    portfolio->reset();
    trade_history.clear();
    
    // Buy at the beginning
    std::string symbol = "BENCHMARK";
    double initial_price = prices[0];
    int shares = static_cast<int>(config.initial_cash / initial_price);
    
    if (shares > 0) {
        portfolio->buyStock(symbol, shares, initial_price, dates[0]);
        trade_history.emplace_back(dates[0], "BUY", initial_price, shares, config.initial_cash);
    }
    
    // Record portfolio state for each day
    for (size_t i = 0; i < prices.size(); i++) {
        std::map<std::string, double> current_prices;
        current_prices[symbol] = prices[i];
        portfolio->recordState(dates[i], current_prices);
    }
    
    // Calculate performance metrics
    result.portfolio_history = portfolio->getHistory();
    result.trades = trade_history;
    result.performance = calculatePerformanceMetrics(result.portfolio_history, result.trades);
    
    // Calculate additional metrics (matching the main backtest function)
    if (!result.portfolio_history.empty()) {
        result.final_portfolio_value = result.portfolio_history.back().total_value;
        result.total_return_pct = ((result.final_portfolio_value / config.initial_cash) - 1.0) * 100.0;
        
        // Calculate annualized return
        double days = result.portfolio_history.size();
        double years = days / 252.0; // Trading days per year
        result.annualized_return = (std::pow(result.final_portfolio_value / config.initial_cash, 1.0 / years) - 1.0) * 100.0;
        
        // Calculate other metrics
        std::vector<double> portfolio_values;
        for (const auto& state : result.portfolio_history) {
            portfolio_values.push_back(state.total_value);
        }
        
        std::vector<double> daily_returns = calculateDailyReturns(result.portfolio_history);
        result.max_drawdown_pct = calculateMaxDrawdown(portfolio_values);
        result.sharpe_ratio = calculateSharpeRatio(daily_returns);
        result.sortino_ratio = calculateSortinoRatio(daily_returns);
        result.calmar_ratio = calculateCalmarRatio(result.annualized_return, result.max_drawdown_pct);
        result.win_rate = calculateWinRate(result.trades);
        result.profit_factor = calculateProfitFactor(result.trades);
    }
    
    return result;
}

std::vector<double> Backtester::calculateDailyReturns(const std::vector<PortfolioState>& portfolio_history) {
    std::vector<double> returns;
    
    if (portfolio_history.size() < 2) return returns;
    
    for (size_t i = 1; i < portfolio_history.size(); i++) {
        double prev_value = portfolio_history[i-1].total_value;
        double curr_value = portfolio_history[i].total_value;
        
        if (prev_value > 0) {
            double daily_return = (curr_value - prev_value) / prev_value;
            returns.push_back(daily_return);
        }
    }
    
    return returns;
}

void Backtester::printPerformanceReport(const BacktestResult& result) {
    std::cout << "\n=== BACKTESTING PERFORMANCE REPORT ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Final Portfolio Value: $" << result.final_portfolio_value << std::endl;
    std::cout << "Total Return: " << result.total_return_pct << "%" << std::endl;
    std::cout << "Annualized Return: " << result.annualized_return << "%" << std::endl;
    std::cout << "Max Drawdown: " << result.max_drawdown_pct << "%" << std::endl;
    std::cout << "Sharpe Ratio: " << std::setprecision(3) << result.sharpe_ratio << std::endl;
    std::cout << "Sortino Ratio: " << std::setprecision(3) << result.sortino_ratio << std::endl;
    std::cout << "Win Rate: " << std::setprecision(2) << result.win_rate << "%" << std::endl;
    std::cout << "Profit Factor: " << std::setprecision(3) << result.profit_factor << std::endl;
    std::cout << "Total Trades: " << result.trades.size() << std::endl;
    std::cout << "=========================================" << std::endl;
}

void Backtester::saveResultsToCSV(const BacktestResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "Date,Cash,Shares,TotalValue" << std::endl;
    
    for (const auto& state : result.portfolio_history) {
        file << state.date << ","
             << std::fixed << std::setprecision(2) << state.cash << ","
             << state.shares << ","
             << std::fixed << std::setprecision(2) << state.total_value << std::endl;
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

void Backtester::saveTradestoCSV(const std::vector<Trade>& trades, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "Date,Signal,Price,Shares,PortfolioValue" << std::endl;
    
    for (const auto& trade : trades) {
        file << trade.date << ","
             << trade.signal_type << ","
             << std::fixed << std::setprecision(2) << trade.price << ","
             << trade.shares << ","
             << std::fixed << std::setprecision(2) << trade.portfolio_value << std::endl;
    }
    
    file.close();
    std::cout << "Trades saved to " << filename << std::endl;
}

void Backtester::updateConfig(const BacktestConfig& new_config) {
    config = new_config;
    portfolio = std::make_unique<Portfolio>(config);
}

void Backtester::reset() {
    portfolio->reset();
    trade_history.clear();
} 