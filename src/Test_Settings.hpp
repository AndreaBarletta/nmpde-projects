#ifndef TEST_SETTINGS_HPP
#define TEST_SETTINGS_HPP

#include "Functions.hpp"

template<unsigned int dim>
struct Test_Settings{

    // Output computation error
    static constexpr bool ENABLE_OUT_ERR_H = false;
    static constexpr bool ENABLE_OUT_ERR_U = false;
    
    // Exact solutions
    static constexpr bool ENABLE_EXACT_INIT_H = false;
    static constexpr bool ENABLE_EXACT_INIT_U = false;
    static constexpr bool ENABLE_EXACT_H = false;
    static constexpr bool ENABLE_EXACT_U = false;
    Value_Function<dim> exact_solution_h;
    Vector_Function<dim> exact_solution_u;

    // Forcing term
    static constexpr bool ENABLE_FORCING_H = false;
    static constexpr bool ENABLE_FORCING_U = false;
    Value_Function<dim> forcing_term_h;
    Vector_Function<dim> forcing_term_u;
};


#endif // TEST_SETTINGS_HPP