#ifndef TEST_SWITCHES_HPP
#define TEST_SWITCHES_HPP

// Class containing compile-time switches for enabling/disabling features
struct Test_Switches_Default { 
    static constexpr bool ENABLE_EXACT_INIT_H = false;
    static constexpr bool ENABLE_EXACT_INIT_U = false;
    static constexpr bool ENABLE_EXACT_H = false;
    static constexpr bool ENABLE_EXACT_U = false;
    static constexpr bool ENABLE_FORCING_H = false;
    static constexpr bool ENABLE_FORCING_U = false;
    static constexpr bool ENABLE_OUT_ERR_H = false;
    static constexpr bool ENABLE_OUT_ERR_U = false;
};

// Don't compute anything
struct Test_Switches_No_Compute : public Test_Switches_Default {
    static constexpr bool ENABLE_EXACT_INIT_H = true;
    static constexpr bool ENABLE_EXACT_INIT_U = true;
    static constexpr bool ENABLE_EXACT_H = true;
    static constexpr bool ENABLE_EXACT_U = true;
    static constexpr bool ENABLE_FORCING_H = false;
    static constexpr bool ENABLE_FORCING_U = false;
    static constexpr bool ENABLE_OUT_ERR_H = true;
    static constexpr bool ENABLE_OUT_ERR_U = true;
};

// Convergence test for height only
struct Test_Convergence_H : public Test_Switches_Default {
    static constexpr bool ENABLE_EXACT_INIT_H = true;
    static constexpr bool ENABLE_EXACT_INIT_U = true;
    static constexpr bool ENABLE_EXACT_H = false;
    static constexpr bool ENABLE_EXACT_U = true;
    static constexpr bool ENABLE_FORCING_H = true;
    static constexpr bool ENABLE_FORCING_U = false;
    static constexpr bool ENABLE_OUT_ERR_H = true;
    static constexpr bool ENABLE_OUT_ERR_U = false;
};

// Convergence test for velocity only
struct Test_Convergence_U : public Test_Switches_Default {
    static constexpr bool ENABLE_EXACT_INIT_H = true;
    static constexpr bool ENABLE_EXACT_INIT_U = true;
    static constexpr bool ENABLE_EXACT_H = true;
    static constexpr bool ENABLE_EXACT_U = false;
    static constexpr bool ENABLE_FORCING_H = false;
    static constexpr bool ENABLE_FORCING_U = true;
    static constexpr bool ENABLE_OUT_ERR_H = false;
    static constexpr bool ENABLE_OUT_ERR_U = true;
};

// Convergence test for both height and velocity
struct Test_Convergence_HU : public Test_Switches_Default {
    static constexpr bool ENABLE_EXACT_INIT_H = true;
    static constexpr bool ENABLE_EXACT_INIT_U = true;
    static constexpr bool ENABLE_EXACT_H = false;
    static constexpr bool ENABLE_EXACT_U = false;
    static constexpr bool ENABLE_FORCING_H = true;
    static constexpr bool ENABLE_FORCING_U = true;
    static constexpr bool ENABLE_OUT_ERR_H = true;
    static constexpr bool ENABLE_OUT_ERR_U = true;
};

// Test with SUPG stabilization (this does not turn off supg by itself)
struct Test_SUPG : public Test_Switches_Default {
    static constexpr bool ENABLE_EXACT_INIT_H = false;
    static constexpr bool ENABLE_EXACT_INIT_U = true;
    static constexpr bool ENABLE_EXACT_H = false;
    static constexpr bool ENABLE_EXACT_U = true;
    static constexpr bool ENABLE_FORCING_H = false;
    static constexpr bool ENABLE_FORCING_U = false;
    static constexpr bool ENABLE_OUT_ERR_H = false;
    static constexpr bool ENABLE_OUT_ERR_U = false;
};


#endif