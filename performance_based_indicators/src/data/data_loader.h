#pragma once

#include "common.h"

class DataLoader {
private:
    std::string data_directory;
    std::map<std::string, std::vector<MarketData>> cache;
    mutable std::mutex cache_mutex;
    
    // Helper functions
    std::string trim(const std::string& str);
    std::vector<std::string> split(const std::string& str, char delimiter);
    std::string date_to_string(const std::chrono::system_clock::time_point& time_point);
    std::chrono::system_clock::time_point string_to_date(const std::string& date_str);
    
    // File operations
    bool file_exists(const std::string& filepath);
    std::string get_cache_filename(const std::string& ticker, const std::string& start_date, const std::string& end_date);
    
public:
    DataLoader(const std::string& data_dir = "../../src/data/");
    
    // Main data loading functions
    std::vector<MarketData> load_data(const std::string& ticker, const std::string& start_date = "", const std::string& end_date = "");
    std::vector<MarketData> load_cached_data(const std::string& ticker);
    
    // Data manipulation
    std::vector<MarketData> filter_by_date_range(const std::vector<MarketData>& data, const std::string& start_date, const std::string& end_date);
    std::vector<Price> extract_closing_prices(const std::vector<MarketData>& data);
    std::vector<std::string> extract_dates(const std::vector<MarketData>& data);
    
    // Cache management
    void clear_cache();
    void preload_data(const std::vector<std::string>& tickers);
    
    // Utility functions
    size_t get_data_size(const std::string& ticker);
    std::vector<std::string> get_available_tickers();
    
    // Data validation
    bool validate_data(const std::vector<MarketData>& data);
    void print_data_summary(const std::vector<MarketData>& data, const std::string& ticker);
}; 