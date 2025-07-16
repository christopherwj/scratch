#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <thread>

double simulate_path(double S0, double mu, double sigma, double T, int steps, std::mt19937& gen, std::normal_distribution<>& dist) {
    double dt = T / steps;
    double S = S0;
    for (int i = 0; i < steps; ++i) {
        double Z = dist(gen);
        S *= std::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
    }
    return S;
}

void simulate_chunk(int start, int end, std::vector<double>& results, double S0, double mu, double sigma, double T, int steps) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    for (int i = start; i < end; ++i) {
        results[i] = simulate_path(S0, mu, sigma, T, steps, gen, dist);
    }
}

int main() {
    const double S0 = 100.0;
    const double mu = 0.07;
    const double sigma = 0.2;
    const double T = 1.0;
    const int steps = 252;
    const int simulations = 100000;

    int num_threads = std::thread::hardware_concurrency();
    std::vector<double> results(simulations);
    std::vector<std::thread> threads;

    int chunk_size = simulations / num_threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? simulations : start + chunk_size;
        threads.emplace_back(simulate_chunk, start, end, std::ref(results), S0, mu, sigma, T, steps);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Average result
    double sum = std::accumulate(results.begin(), results.end(), 0.0);
    double mean = sum / simulations;

    std::cout << "Expected price after 1 year: $" << mean << std::endl;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}