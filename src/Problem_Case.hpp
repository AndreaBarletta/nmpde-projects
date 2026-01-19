#ifndef PROBLEM_CASE_HPP
#define PROBLEM_CASE_HPP

#include "Functions.hpp"
#include <memory>

template <unsigned int dim>
struct Problem_Case
{
    // Gravitational acceleration
    static constexpr double g = 2.5e-4;

    // Chézy’s friction coefficient
    static constexpr const double cf = 1e-2;

    // Initial-exact solutions
    std::unique_ptr<Value_Function<dim>> exact_init_h;
    std::unique_ptr<Vector_Function<dim>> exact_init_u;

    // ----- Test Settings --- [DEFAULT TEST DISABLED]
    // Output computation error
    static constexpr bool ENABLE_OUT_ERR_H = false;
    static constexpr bool ENABLE_OUT_ERR_U = false;
    
    // Exact solution instead of computation
    static constexpr bool ENABLE_COMPUTE_EXACT_H = false;
    static constexpr bool ENABLE_COMPUTE_EXACT_U = false;

    // Forcing term
    static constexpr bool ENABLE_FORCING_H = false;
    static constexpr bool ENABLE_FORCING_U = false;
    std::unique_ptr<Value_Function<dim>> forcing_term_h;
    std::unique_ptr<Vector_Function<dim>> forcing_term_u;

    Problem_Case()
        : exact_init_h(std::make_unique<Value_Function<dim>>()),
          exact_init_u(std::make_unique<Vector_Function<dim>>()),
          forcing_term_h(std::make_unique<Value_Function<dim>>()),
          forcing_term_u(std::make_unique<Vector_Function<dim>>())
    {}
};

#endif // PROBLEM_CASE_HPP