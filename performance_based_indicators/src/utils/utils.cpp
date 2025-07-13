#include "utils/utils.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <random>
#include <ctime>
#include <regex>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace Utils {

// String utilities
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim(token));
    }
    
    return tokens;
}

std::string toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string toUpper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

// Date utilities
std::string getCurrentDate() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d");
    return ss.str();
}

std::string formatDate(const std::string& date) {
    // Basic date formatting - assumes YYYY-MM-DD format
    if (date.length() == 10 && date[4] == '-' && date[7] == '-') {
        return date;
    }
    return date; // Return as-is if not standard format
}

bool isValidDate(const std::string& date) {
    std::regex date_regex(R"(\d{4}-\d{2}-\d{2})");
    return std::regex_match(date, date_regex);
}

// Math utilities
double calculateMean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

double calculateStandardDeviation(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    
    double mean = calculateMean(values);
    double variance = 0.0;
    
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    
    variance /= (values.size() - 1);
    return std::sqrt(variance);
}

double calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;
    
    double mean_x = calculateMean(x);
    double mean_y = calculateMean(y);
    
    double numerator = 0.0;
    double sum_sq_x = 0.0;
    double sum_sq_y = 0.0;
    
    for (size_t i = 0; i < x.size(); i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    
    double denominator = std::sqrt(sum_sq_x * sum_sq_y);
    return (denominator == 0.0) ? 0.0 : numerator / denominator;
}

// Timer implementation
Timer::Timer(const std::string& timer_name) : name(timer_name) {
    start_time = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Timer [" << name << "]: " << duration.count() << " ms" << std::endl;
}

double Timer::elapsed() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    return duration.count();
}

void Timer::reset() {
    start_time = std::chrono::high_resolution_clock::now();
}

// File utilities
bool fileExists(const std::string& filename) {
    return std::filesystem::exists(filename);
}

std::string getFileExtension(const std::string& filename) {
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) return "";
    return filename.substr(dot_pos + 1);
}

std::vector<std::string> listFiles(const std::string& directory, const std::string& extension) {
    std::vector<std::string> files;
    
    if (!std::filesystem::exists(directory)) return files;
    
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (extension.empty() || getFileExtension(filename) == extension) {
                files.push_back(filename);
            }
        }
    }
    
    std::sort(files.begin(), files.end());
    return files;
}

// Validation utilities
bool isValidPrice(double price) {
    return price > 0.0 && std::isfinite(price);
}

bool isValidVolume(long long volume) {
    return volume >= 0;
}

bool isValidTicker(const std::string& ticker) {
    if (ticker.empty() || ticker.length() > 10) return false;
    
    for (char c : ticker) {
        if (!std::isalnum(c) && c != '.') return false;
    }
    
    return true;
}

// Format utilities
std::string formatCurrency(double amount) {
    std::stringstream ss;
    ss << "$" << std::fixed << std::setprecision(2) << amount;
    return ss.str();
}

std::string formatPercentage(double percentage) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << percentage << "%";
    return ss.str();
}

std::string formatNumber(double number, int precision) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << number;
    return ss.str();
}

// Memory utilities
size_t getMemoryUsage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    return pmc.WorkingSetSize;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss * 1024; // Convert from KB to bytes
#endif
}

void printMemoryUsage() {
    size_t memory_kb = getMemoryUsage() / 1024;
    std::cout << "Memory usage: " << memory_kb << " KB" << std::endl;
}

// Random utilities
static std::random_device rd;
static std::mt19937 gen(rd());

void setSeed(unsigned int seed) {
    gen.seed(seed);
}

double randomDouble(double min, double max) {
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

int randomInt(int min, int max) {
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

} // namespace Utils 