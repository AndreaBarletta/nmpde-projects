#ifndef CASE_EXAMPLES_HPP
#define CASE_EXAMPLES_HPP

#include "Problem_Case.hpp"

template<unsigned int dim>
struct Centered_Gaussian_Bump_Case : public Problem_Case<dim>
{
    class Centered_Gaussian_Bump_F : public Value_Function<dim>
    {
        public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            return 0.5 + 0.2 * std::exp(-((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.5) * (p[1] - 0.5)) * 500.0);
        }
    };

    Centered_Gaussian_Bump_Case()
    {
        this->exact_init_h = std::make_unique<Centered_Gaussian_Bump_F>();
    }
};

template<unsigned int dim>
struct Offset_Gaussian_Bump_Case : public Problem_Case<dim>
{
    class Offset_Gaussian_Bump_F : public Value_Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            return 0.5 + 0.2 * std::exp(-((p[0] - 0.25) * (p[0] - 0.25) + (p[1] - 0.25) * (p[1] - 0.25)) * 500.0);
        }
    };

    Offset_Gaussian_Bump_Case()
    {
        this->exact_init_h = std::make_unique<Offset_Gaussian_Bump_F>();
    }
};

template<unsigned int dim>
struct Side_Wave_Case : public Problem_Case<dim>
{
    class Side_Wave_F : public Value_Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            // make the wave a bit wider 
            return 0.5 + 0.05 * std::exp(-((p[0] - 0.1) * (p[0] - 0.1)) * 500.0);
        }
    };

    Side_Wave_Case()
    {
        this->exact_init_h = std::make_unique<Side_Wave_F>();
    }
};

template<unsigned int dim>
struct Angular_Side_Wave_Case : public Problem_Case<dim>
{
    class Angular_Side_Wave_F : public Value_Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            // the wave has to form a L-shape in the corner
            return 0.5 + 0.05 * std::max(std::exp(-((p[0] - 0.1) * (p[0] - 0.1)) * 500.0), std::exp(-((p[1] - 0.1) * (p[1] - 0.1)) * 500.0));
        }
    };

    Angular_Side_Wave_Case()
    {
        this->exact_init_h = std::make_unique<Angular_Side_Wave_F>();
    }
};

template<unsigned int dim>
struct Sloping_Plane_Case : public Problem_Case<dim>
{
    class Sloping_Plane_F : public Value_Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            return 1.0 - 0.3 * p[0];
        }   
    };
  
    Sloping_Plane_Case()
    {
        this->exact_init_h = std::make_unique<Sloping_Plane_F>();
    }
};

template<unsigned int dim>
struct Corner_Gaussian_Drop_Case : public Problem_Case<dim>
{
    class Corner_Gaussian_Drop_F : public Value_Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            return 0.5 + 0.2 * std::exp(-((p[0]) * (p[0]) + (p[1]) * (p[1])) * 500.0);
        }
    };
  
    Corner_Gaussian_Drop_Case()
    {
        this->exact_init_h = std::make_unique<Corner_Gaussian_Drop_F>();
    }
};




template<unsigned int dim>
struct Disappearing_Dam_Break_Case : public Problem_Case<dim>
{
    class Disappearing_Dam_Break_F : public Value_Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            return p[0] < 0.5 ? 1.0 : 0.5;
        }
    };
  
    Disappearing_Dam_Break_Case()
    {
        this->exact_init_h = std::make_unique<Disappearing_Dam_Break_F>();
    }
};

template<unsigned int dim>
struct Still_Water_Case : public Problem_Case<dim>
{
    class Still_Water_F : public Value_Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            return 1.0;
        }
    };

    Still_Water_Case()
    {
        this->exact_init_h = std::make_unique<Still_Water_F>();
    }
};

#endif // CASE_EXAMPLES_HPP