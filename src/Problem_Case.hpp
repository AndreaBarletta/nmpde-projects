#ifndef PROBLEM_CASE_HPP
#define PROBLEM_CASE_HPP

#include "Functions.hpp"
#include "Test_Settings.hpp"

template <unsigned int dim, template<unsigned int> typename Test_Settings>
struct Problem_Case
{
    // Gravitational acceleration
    static constexpr double g = 2.5e-4;

    // Chézy’s friction coefficient
    static constexpr const double cf = 1e-2;

    // Initial conditions
    Value_Function<dim> initial_conditions_h;
    Vector_Function<dim> initial_conditions_u;

    // Test Settings
    using T_S = Test_Settings<dim>;
    T_S t_settings;
};

#endif // PROBLEM_SPECS_HPP