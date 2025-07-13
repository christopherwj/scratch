#pragma once

#include "common.h"
#include <string>
#include <chrono>

namespace Utils {
    // String utilities
    std::string trim(const std::string& str);
    std::vector<std::string> split(const std::string& str, char delimiter);
    std::string toLower(const std::string& str);
    std::string toUpper(const std::string& str);
    
    // Date utilities
    std::string getCurrentDate();
    std::string formatDate(const std::string& date);
    bool isValidDate(const std::string& date);
    
    // Math utilities
    double calculateMean(const std::vector<double>& values);
    double calculateStandardDeviation(const std::vector<double>& values);
    double calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y);
    
    // Performance utilities
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_time;
        std::string name;
        
    public:
        Timer(const std::string& timer_name);
        ~Timer();
        double elapsed() const;
        void reset();
    };
    
    // File utilities
    bool fileExists(const std::string& filename);
    std::string getFileExtension(const std::string& filename);
    std::vector<std::string> listFiles(const std::string& directory, const std::string& extension = "");
    
    // Validation utilities
    bool isValidPrice(double price);
    bool isValidVolume(long long volume);
    bool isValidTicker(const std::string& ticker);
    
    // Format utilities
    std::string formatCurrency(double amount);
    std::string formatPercentage(double percentage);
    std::string formatNumber(double number, int precision = 2);
    
    // Memory utilities
    size_t getMemoryUsage();
    void printMemoryUsage();
    
    // Random utilities
    void setSeed(unsigned int seed);
    double randomDouble(double min = 0.0, double max = 1.0);
    int randomInt(int min, int max);
} 