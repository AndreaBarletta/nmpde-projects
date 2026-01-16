#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <deal.II/base/function.h>

using namespace dealii;

// Scalar function base class.
template <unsigned int dim>
class Value_Function : public Function<dim>
{
public:
    virtual double value(const Point<dim> &/*p*/, const unsigned int /*component*/ = 0) const override
    {
        return 0.0;
    }
};

// Vector function base class.
template <unsigned int dim>
class Vector_Function : public Function<dim>
{
public:
    virtual void vector_value(const Point<dim> &/*p*/, Vector<double> &values) const override
    {
        for (unsigned int i = 0; i < this->n_components; ++i)
            values[i] = 0.0;
    }

    virtual double value(const Point<dim> &/*p*/, const unsigned int /*component*/ = 0) const override
    {
        return 0.0;
    }
};


#endif // FUNCTIONS_HPP