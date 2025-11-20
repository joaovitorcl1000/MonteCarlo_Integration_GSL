// ============================================================
// Author: João Vitor Costa Lovato
// email: joaovitorcl1000@outlook.com

// 3D Monte Carlo integration with OpenMP (parallelization) + GSL VEGAS (Monte Carlo)
// ============================================================

#include <iostream>     // std::cout, std::endl
#include <cmath>        // math functions (std::sqrt, std::pow, etc.)
#include <chrono>       // timing utilities
#include <vector>       // std::vector
#include <array>        // std::array (fixed-size limits)
#include <numeric>      // std::accumulate

#include <gsl/gsl_monte.h>        // GSL MC base types
#include <gsl/gsl_monte_vegas.h>  // GSL VEGAS algorithm

#include <omp.h>  // OpenMP parallelization

using namespace std;

// ------------------------------------------------------------------------
// Problem constants
// ------------------------------------------------------------------------

constexpr size_t Dim = 3; // number of integration dimensions
constexpr int N_MC = 10000000;   // total Monte Carlo samples

// Integration bounds [0,1] in each dimension
array<double, Dim> Lower_bounds = {0.0, 0.0, 0.0};
array<double, Dim> Upper_bounds = {1.0, 1.0, 1.0};

struct parameters {
    double p;
    double q;
};

// ------------------------------------------------------------------------
// Function to be integrated
// ------------------------------------------------------------------------

double f(double x, double y, double z, const parameters* par)
{
    return (x + y + z)*par->p + (x*x + y*y + z*z)*par->q;
}

// ------------------------------------------------------------------------
// GSL integrand wrapper
// ------------------------------------------------------------------------

double integrand(double* k, size_t dim, void* params)
{
    (void)dim; // dimension is known (Dim) and not needed here

    auto* par = static_cast<parameters*>(params);

    double x = k[0];
    double y = k[1];
    double z = k[2];

    return f(x, y, z, par);
}

// ------------------------------------------------------------------------
// Parallel Monte Carlo integration using VEGAS
// ------------------------------------------------------------------------

void parallel_monte_carlo_integration(size_t calls,
                                      double xl[], double xu[],
                                      size_t dim,
                                      double& result, double& error,
                                      parameters& par)
{
    //omp_set_num_threads(N); //Force the CPU use N threads
    size_t num_threads = omp_get_max_threads();
    vector<double> results(num_threads, 0.0);
    vector<double> errors (num_threads, 0.0);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);

        // Unique seed per thread (time + thread_id)
        unsigned long seed =
                chrono::high_resolution_clock::now().time_since_epoch().count()
                ^ static_cast<unsigned long>(thread_id + 0x9e3779b97f4a7c15ULL);

        gsl_rng_set(rng, seed);

        gsl_monte_vegas_state* state = gsl_monte_vegas_alloc(dim);
        gsl_monte_function G = { &integrand, dim, &par };

        // Each thread uses calls/num_threads samples
        gsl_monte_vegas_integrate(&G, xl, xu, dim,
                                  calls / num_threads,
                                  rng, state,
                                  &results[thread_id],
                                  &errors[thread_id]);

        gsl_monte_vegas_free(state);
        gsl_rng_free(rng);
    }

    // Average of the estimates (simple combining)
    result = accumulate(results.begin(), results.end(), 0.0) / num_threads;

    // Combine errors: sqrt(average of variances)
    double sum_sq_errors = 0.0;
    for (double err_i : errors) {
        sum_sq_errors += err_i * err_i;
    }
    error = sqrt(sum_sq_errors) / num_threads;
}

// ------------------------------------------------------------------------
// Main
// ------------------------------------------------------------------------

int main()
{
    // GSL expects double*, we can use the internal storage of array
    double* xl = Lower_bounds.data();
    double* xu = Upper_bounds.data();

    auto start = chrono::high_resolution_clock::now();

    double res = 0.0;
    double err = 0.0;

    parameters par = {0.1, 0.1};

    // More samples => better precision
    parallel_monte_carlo_integration(N_MC, xl, xu, Dim, res, err, par);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Expected Result: 1/4" << '\n';
    cout << "Result: " << res << '\n';
    cout << "Error:  " << err << '\n';
    cout << "Time taken: " << elapsed.count() << " s\n";

    return 0;
}
