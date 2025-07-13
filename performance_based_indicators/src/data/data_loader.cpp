#include "data/data_loader.h"
#include <filesystem>
#include <regex>
#include <iomanip>

DataLoader::DataLoader(const std::string& data_dir) : data_directory(data_dir) {
    // Ensure data directory exists
    if (!std::filesystem::exists(data_directory)) {
        std::filesystem::create_directories(data_directory);
    }
}

std::string DataLoader::trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

std::vector<std::string> DataLoader::split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim(token));
    }
    
    return tokens;
}

bool DataLoader::file_exists(const std::string& filepath) {
    return std::filesystem::exists(filepath);
}

std::string DataLoader::get_cache_filename(const std::string& ticker, const std::string& start_date, const std::string& end_date) {
    // Look for existing files with ticker pattern
    std::string pattern = ticker + "_";
    for (const auto& entry : std::filesystem::directory_iterator(data_directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find(pattern) == 0 && filename.find(".csv") != std::string::npos) {
                return entry.path().string();
            }
        }
    }
    return "";
}

std::vector<MarketData> DataLoader::load_data(const std::string& ticker, const std::string& start_date, const std::string& end_date) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Check if data is already cached
    std::string cache_key = ticker + "_" + start_date + "_" + end_date;
    if (cache.find(cache_key) != cache.end()) {
        std::cout << "Loading cached data for " << ticker << std::endl;
        return cache[cache_key];
    }
    
    // Find the data file
    std::string filepath = get_cache_filename(ticker, start_date, end_date);
    if (filepath.empty()) {
        std::cerr << "No data file found for ticker: " << ticker << std::endl;
        return {};
    }
    
    std::cout << "Loading data for " << ticker << " from " << filepath << std::endl;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filepath << std::endl;
        return {};
    }
    
    std::vector<MarketData> data;
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue; // Skip header
        }
        
        std::vector<std::string> tokens = split(line, ',');
        if (tokens.size() < 6) continue;
        
        try {
            MarketData md;
            md.date = tokens[0];
            md.open = std::stod(tokens[1]);
            md.high = std::stod(tokens[2]);
            md.low = std::stod(tokens[3]);
            md.close = std::stod(tokens[4]);
            md.volume = std::stoll(tokens[5]);
            md.adjusted_close = (tokens.size() > 6) ? std::stod(tokens[6]) : md.close;
            
            data.push_back(md);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << " - " << e.what() << std::endl;
        }
    }
    
    file.close();
    
    // Filter by date range if specified
    if (!start_date.empty() || !end_date.empty()) {
        data = filter_by_date_range(data, start_date, end_date);
    }
    
    // Cache the data
    cache[cache_key] = data;
    
    std::cout << "Loaded " << data.size() << " data points for " << ticker << std::endl;
    return data;
}

std::vector<MarketData> DataLoader::load_cached_data(const std::string& ticker) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Find any cached data for this ticker
    for (const auto& pair : cache) {
        if (pair.first.find(ticker) == 0) {
            return pair.second;
        }
    }
    
    // If not cached, try to load from file
    return load_data(ticker);
}

std::vector<MarketData> DataLoader::filter_by_date_range(const std::vector<MarketData>& data, const std::string& start_date, const std::string& end_date) {
    std::vector<MarketData> filtered_data;
    
    for (const auto& md : data) {
        bool include = true;
        
        if (!start_date.empty() && md.date < start_date) {
            include = false;
        }
        if (!end_date.empty() && md.date > end_date) {
            include = false;
        }
        
        if (include) {
            filtered_data.push_back(md);
        }
    }
    
    return filtered_data;
}

std::vector<Price> DataLoader::extract_closing_prices(const std::vector<MarketData>& data) {
    std::vector<Price> prices;
    prices.reserve(data.size());
    
    for (const auto& md : data) {
        prices.push_back(md.close);
    }
    
    return prices;
}

std::vector<std::string> DataLoader::extract_dates(const std::vector<MarketData>& data) {
    std::vector<std::string> dates;
    dates.reserve(data.size());
    
    for (const auto& md : data) {
        dates.push_back(md.date);
    }
    
    return dates;
}

void DataLoader::clear_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache.clear();
}

void DataLoader::preload_data(const std::vector<std::string>& tickers) {
    for (const auto& ticker : tickers) {
        load_data(ticker);
    }
}

size_t DataLoader::get_data_size(const std::string& ticker) {
    auto data = load_cached_data(ticker);
    return data.size();
}

std::vector<std::string> DataLoader::get_available_tickers() {
    std::vector<std::string> tickers;
    
    for (const auto& entry : std::filesystem::directory_iterator(data_directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find(".csv") != std::string::npos) {
                size_t underscore_pos = filename.find('_');
                if (underscore_pos != std::string::npos) {
                    std::string ticker = filename.substr(0, underscore_pos);
                    if (std::find(tickers.begin(), tickers.end(), ticker) == tickers.end()) {
                        tickers.push_back(ticker);
                    }
                }
            }
        }
    }
    
    return tickers;
}

bool DataLoader::validate_data(const std::vector<MarketData>& data) {
    if (data.empty()) return false;
    
    for (const auto& md : data) {
        if (md.close <= 0 || md.high < md.low || md.open <= 0) {
            return false;
        }
    }
    
    return true;
}

void DataLoader::print_data_summary(const std::vector<MarketData>& data, const std::string& ticker) {
    if (data.empty()) {
        std::cout << "No data available for " << ticker << std::endl;
        return;
    }
    
    std::cout << "\n=== Data Summary for " << ticker << " ===" << std::endl;
    std::cout << "Data points: " << data.size() << std::endl;
    std::cout << "Date range: " << data.front().date << " to " << data.back().date << std::endl;
    auto min_it = std::min_element(data.begin(), data.end(), [](const MarketData& a, const MarketData& b) { return a.close < b.close; });
    auto max_it = std::max_element(data.begin(), data.end(), [](const MarketData& a, const MarketData& b) { return a.close < b.close; });
    std::cout << "Price range: $" << std::fixed << std::setprecision(2) 
              << min_it->close << " - $" << max_it->close << std::endl;
} 