#ifndef CASE_TEST_HPP
#define CASE_TEST_HPP

#include "Problem_Case.hpp"

// === MANUFACTURED SOLUTION TEST CASES ===

// This class defines a manufactured solution test case, but does not activate it.
template<unsigned int dim>
struct Manufactured_Test_Case : public Problem_Case<dim>{

    // Exact solutions.
    class Exact_h : public Value_Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];

            return x * y * std::cos(M_PI * t) + 1.0;
        }
    };

    class Exact_u : public Vector_Function<dim>
    {
    public:
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];

            values[0] = y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y);
            values[1] = std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t);
        }

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];

            if (component == 0)
                return y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y);
            else
                return std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t);

        }
    };

    // Forcing terms (used mainly for manufactured-solution testing)
    class ForcingTerm_h : public Value_Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];

            return -M_PI * x * y * std::sin(M_PI * t) + x * std::sin(M_PI * x) * std::sin(M_PI * y) * std::pow(std::cos(M_PI * t), 2) + std::pow(y, 2) * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos(M_PI * t) * std::cos((1.0/2.0) * M_PI * y) + M_PI * (x * y * std::cos(M_PI * t) + 1) * (y * std::sin(M_PI * t) * std::cos(M_PI * x) * std::cos((1.0/2.0) * M_PI * y) + std::sin(M_PI * x) * std::cos(M_PI * t) * std::cos(M_PI * y));
        }
    };

    class ForcingTerm_u : public Vector_Function<dim>
    {
    public:
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];
            const double g = Problem_Case<dim>::g;
            const double cf = Problem_Case<dim>::cf;

            values[0] = cf * y * std::sqrt(std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + std::pow(std::sin(M_PI * x), 2) * std::pow(std::sin(M_PI * y), 2) * std::pow(std::cos(M_PI * t), 2)) * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y)/(x * y * std::cos(M_PI * t) + 1) + g * y * std::cos(M_PI * t) + M_PI * std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::sin(M_PI * x) * std::cos(M_PI * x) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + M_PI * y * std::sin(M_PI * x) * std::cos(M_PI * t) * std::cos((1.0/2.0) * M_PI * y) + (-1.0/2.0 * M_PI * y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin((1.0/2.0) * M_PI * y) + std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y)) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t);
            values[1] = cf * std::sqrt(std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + std::pow(std::sin(M_PI * x), 2) * std::pow(std::sin(M_PI * y), 2) * std::pow(std::cos(M_PI * t), 2)) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t)/(x * y * std::cos(M_PI * t) + 1) + g * x * std::cos(M_PI * t) + M_PI * y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t) * std::cos(M_PI * x) * std::cos((1.0/2.0) * M_PI * y) - M_PI * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin(M_PI * y) + M_PI * std::pow(std::sin(M_PI * x), 2) * std::sin(M_PI * y) * std::pow(std::cos(M_PI * t), 2) * std::cos(M_PI * y);
        }

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];
            const double g = Problem_Case<dim>::g;
            const double cf = Problem_Case<dim>::cf;

            if (component == 0)
                return cf * y * std::sqrt(std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + std::pow(std::sin(M_PI * x), 2) * std::pow(std::sin(M_PI * y), 2) * std::pow(std::cos(M_PI * t), 2)) * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y)/(x * y * std::cos(M_PI * t) + 1) + g * y * std::cos(M_PI * t) + M_PI * std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::sin(M_PI * x) * std::cos(M_PI * x) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + M_PI * y * std::sin(M_PI * x) * std::cos(M_PI * t) * std::cos((1.0/2.0) * M_PI * y) + (-1.0/2.0 * M_PI * y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin((1.0/2.0) * M_PI * y) + std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y)) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t);
            else
                return cf * std::sqrt(std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + std::pow(std::sin(M_PI * x), 2) * std::pow(std::sin(M_PI * y), 2) * std::pow(std::cos(M_PI * t), 2)) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t)/(x * y * std::cos(M_PI * t) + 1) + g * x * std::cos(M_PI * t) + M_PI * y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t) * std::cos(M_PI * x) * std::cos((1.0/2.0) * M_PI * y) - M_PI * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin(M_PI * y) + M_PI * std::pow(std::sin(M_PI * x), 2) * std::sin(M_PI * y) * std::pow(std::cos(M_PI * t), 2) * std::cos(M_PI * y);   
        }
    };

    Manufactured_Test_Case()
    {
        this->exact_init_h = std::make_unique<Exact_h>();
        this->exact_init_u = std::make_unique<Exact_u>();
        this->forcing_term_h = std::make_unique<ForcingTerm_h>();
        this->forcing_term_u = std::make_unique<ForcingTerm_u>();
    }
};

// Don't compute anything
template<unsigned int dim>
struct Manufactured_Test_No_Compute : public Manufactured_Test_Case<dim> {
    static constexpr bool ENABLE_COMPUTE_EXACT_U = true;
    static constexpr bool ENABLE_COMPUTE_EXACT_H = true;
    static constexpr bool ENABLE_FORCING_H = false;
    static constexpr bool ENABLE_FORCING_U = false;
    static constexpr bool ENABLE_OUT_ERR_H = true;
    static constexpr bool ENABLE_OUT_ERR_U = true;
};

// Convergence test for height only
template<unsigned int dim>
struct Manufactured_Test_Convergence_H : public Manufactured_Test_Case<dim> {
    static constexpr bool ENABLE_COMPUTE_EXACT_H = false;
    static constexpr bool ENABLE_COMPUTE_EXACT_U = true;
    static constexpr bool ENABLE_FORCING_H = true;
    static constexpr bool ENABLE_FORCING_U = false;
    static constexpr bool ENABLE_OUT_ERR_H = true;
    static constexpr bool ENABLE_OUT_ERR_U = false;
};

// Convergence test for velocity only
template<unsigned int dim>
struct Manufactured_Test_Convergence_U : public Manufactured_Test_Case<dim> {
    static constexpr bool ENABLE_COMPUTE_EXACT_H = true;
    static constexpr bool ENABLE_COMPUTE_EXACT_U = false;
    static constexpr bool ENABLE_FORCING_H = false;
    static constexpr bool ENABLE_FORCING_U = true;
    static constexpr bool ENABLE_OUT_ERR_H = false;
    static constexpr bool ENABLE_OUT_ERR_U = true;
};

// Convergence test for both height and velocity
template<unsigned int dim>
struct Manufactured_Test_Convergence_HU : public Manufactured_Test_Case<dim> {
    static constexpr bool ENABLE_COMPUTE_EXACT_H = false;
    static constexpr bool ENABLE_COMPUTE_EXACT_U = false;
    static constexpr bool ENABLE_FORCING_H = true;
    static constexpr bool ENABLE_FORCING_U = true;
    static constexpr bool ENABLE_OUT_ERR_H = true;
    static constexpr bool ENABLE_OUT_ERR_U = true;
};


// === SUPG TEST CASE ===

// Test with SUPG stabilization (this does not turn off supg by itself)
template<unsigned int dim>
struct SUPG_Test : public Problem_Case<dim> {

    class Exact_H_F : public Value_Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            const double x = p[0];
            return 1 - 0.3 * x;
        }
    };

    class Exact_U_F : public Vector_Function<dim>
    {
    public:
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            const double x = p[0];
            const double y = p[1];

            values[0] = 10 * std::sin(x) * std::sin(y);
            values[1] = 0;
        }

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];

            if (component == 0)
                return 10 * std::sin(x) * std::sin(y);
            else
                return 0;

        }
    };


    static constexpr bool ENABLE_COMPUTE_EXACT_H = false;
    static constexpr bool ENABLE_COMPUTE_EXACT_U = true;
    static constexpr bool ENABLE_FORCING_H = false;
    static constexpr bool ENABLE_FORCING_U = false;
    static constexpr bool ENABLE_OUT_ERR_H = false;
    static constexpr bool ENABLE_OUT_ERR_U = false;

    SUPG_Test()
    {
        this->exact_init_h = std::make_unique<Exact_H_F>();
        this->exact_init_u = std::make_unique<Exact_U_F>();
    }
};


#endif // CASE_TEST_HPP