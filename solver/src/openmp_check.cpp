#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    std::printf("OpenMP max threads: %d\n", omp_get_max_threads());

    const int n = 50'000'000;
    std::vector<double> a(n);
    for (int threads : {1, omp_get_max_threads()}) {
        auto t0 = std::chrono::steady_clock::now();
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < n; ++i) a[i] = std::sin(i * 1e-6) * std::cos(i * 1e-6);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%2d thread(s): %.3f s\n", threads, s);
    }
    return 0;
}