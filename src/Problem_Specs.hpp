#ifndef PROBLEM_SPECS_HPP
#define PROBLEM_SPECS_HPP

#include "Functions.hpp"

template <unsigned int dim>
struct Problem_Specs
{
    // Initial conditions
    Value_Function<dim> initial_conditions_h;
    Vector_Function<dim> initial_conditions_u;

    // Exact solutions
    Value_Function<dim> exact_solution_h;
    Vector_Function<dim> exact_solution_u;

    // Forcing term
    Value_Function<dim> forcing_term_h;
    Vector_Function<dim> forcing_term_u;
};


#endif // PROBLEM_SPECS_HPP