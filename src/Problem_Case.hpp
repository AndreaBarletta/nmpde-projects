#ifndef PROBLEM_CASE_HPP
#define PROBLEM_CASE_HPP

#include "Functions.hpp"

template <unsigned int dim>
struct Problem_Case
{
    // Gravitational acceleration
    static constexpr double g = 2.5e-4;

    // Chézy’s friction coefficient
    static constexpr const double cf = 1e-2;

    // Initial-exact solutions
    Value_Function<dim> exact_init_h;
    Vector_Function<dim> exact_init_u;

    // ----- Test Settings
    // Output computation error
    static constexpr bool ENABLE_OUT_ERR_H = false;
    static constexpr bool ENABLE_OUT_ERR_U = false;
    
    // Exact solution instead of computation
    static constexpr bool ENABLE_COMPUTE_EXACT_H = false;
    static constexpr bool ENABLE_COMPUTE_EXACT_U = false;

    // Forcing term
    static constexpr bool ENABLE_FORCING_H = false;
    static constexpr bool ENABLE_FORCING_U = false;
    Value_Function<dim> forcing_term_h;
    Vector_Function<dim> forcing_term_u;
};



#endif // PROBLEM_CASE_HPP