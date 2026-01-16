#ifndef SHALLOW_HPP
#define SHALLOW_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/identity_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <chrono>

#include "Problem_Case.hpp"

using namespace dealii;

// Class representing the shallow waters problem.
template<template<unsigned int> class PCase>
class Shallow_waters
{
public:

    // Physical dimension (2D)
    static constexpr unsigned int dim = 2;

    // // Intial conditions solutions.
    // class InitialConditions_h : public Function<dim>
    // {
    // public:
    //     virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    //     {
    //         // Centered Gaussian bump
    //         // return 0.5 + 0.2 * std::exp(-((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.5) * (p[1] - 0.5)) * 500.0);

    //         // Offset Gaussian bump
    //         // return 0.5 + 0.2 * std::exp(-((p[0] - 0.25) * (p[0] - 0.25) + (p[1] - 0.25) * (p[1] - 0.25)) * 500.0);

    //         // Sloping plane
    //         return 1.0 - 0.3 * p[0];

    //         // Disappearing dam break
    //         // return p[0] < 0.5 ? 1.0 : 0.5;

    //         // Still water
    //         // return 1.0;
    //     }
    // };

    // class InitialConditions_u : public Function<dim>
    // {
    // public:
    //     virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
    //     {
    //         values[0] = 0.0;
    //         values[1] = 0.0;
    //     }

    //     virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    //     {
    //         if (component == 0)
    //             return 0.0;
    //         else
    //             return 0.0;
    //     }
    // };

    // // Exact solutions.
    // class ExactSolution_h : public Function<dim>
    // {
    // public:
    //     virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    //     {
    //         const double t = this->get_time();
    //         const double x = p[0];
    //         const double y = p[1];

    //         return x * y * std::cos(M_PI * t) + 1.0;
    //     }
    // };

    // class ExactSolution_u : public Function<dim>
    // {
    // public:
    //     virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
    //     {
    //         const double t = this->get_time();
    //         const double x = p[0];
    //         const double y = p[1];

    //         values[0] = y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y);
    //         values[1] = std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t);
    //     }

    //     virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    //     {
    //         const double t = this->get_time();
    //         const double x = p[0];
    //         const double y = p[1];

    //         if (component == 0)
    //             return y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y);
    //         else
    //             return std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t);

    //     }
    // };

    // // Forcing terms (used mainly for manufactured-solution testing)
    // class ForcingTerm_h : public Function<dim>
    // {
    // public:
    //     virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    //     {
    //         const double t = this->get_time();
    //         const double x = p[0];
    //         const double y = p[1];

    //         return -M_PI * x * y * std::sin(M_PI * t) + x * std::sin(M_PI * x) * std::sin(M_PI * y) * std::pow(std::cos(M_PI * t), 2) + std::pow(y, 2) * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos(M_PI * t) * std::cos((1.0/2.0) * M_PI * y) + M_PI * (x * y * std::cos(M_PI * t) + 1) * (y * std::sin(M_PI * t) * std::cos(M_PI * x) * std::cos((1.0/2.0) * M_PI * y) + std::sin(M_PI * x) * std::cos(M_PI * t) * std::cos(M_PI * y));
    //     }
    // };

    // class ForcingTerm_u : public Function<dim>
    // {
    // public:
    //     virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
    //     {
    //         const double t = this->get_time();
    //         const double x = p[0];
    //         const double y = p[1];
    //         const double g = Shallow_waters::g;
    //         const double cf = Shallow_waters::cf;

    //         values[0] = cf * y * std::sqrt(std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + std::pow(std::sin(M_PI * x), 2) * std::pow(std::sin(M_PI * y), 2) * std::pow(std::cos(M_PI * t), 2)) * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y)/(x * y * std::cos(M_PI * t) + 1) + g * y * std::cos(M_PI * t) + M_PI * std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::sin(M_PI * x) * std::cos(M_PI * x) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + M_PI * y * std::sin(M_PI * x) * std::cos(M_PI * t) * std::cos((1.0/2.0) * M_PI * y) + (-1.0/2.0 * M_PI * y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin((1.0/2.0) * M_PI * y) + std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y)) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t);
    //         values[1] = cf * std::sqrt(std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + std::pow(std::sin(M_PI * x), 2) * std::pow(std::sin(M_PI * y), 2) * std::pow(std::cos(M_PI * t), 2)) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t)/(x * y * std::cos(M_PI * t) + 1) + g * x * std::cos(M_PI * t) + M_PI * y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t) * std::cos(M_PI * x) * std::cos((1.0/2.0) * M_PI * y) - M_PI * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin(M_PI * y) + M_PI * std::pow(std::sin(M_PI * x), 2) * std::sin(M_PI * y) * std::pow(std::cos(M_PI * t), 2) * std::cos(M_PI * y);
    //     }

    //     virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    //     {
    //         const double t = this->get_time();
    //         const double x = p[0];
    //         const double y = p[1];
    //         const double g = Shallow_waters::g;
    //         const double cf = Shallow_waters::cf;

    //         if (component == 0)
    //             return cf * y * std::sqrt(std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + std::pow(std::sin(M_PI * x), 2) * std::pow(std::sin(M_PI * y), 2) * std::pow(std::cos(M_PI * t), 2)) * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y)/(x * y * std::cos(M_PI * t) + 1) + g * y * std::cos(M_PI * t) + M_PI * std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::sin(M_PI * x) * std::cos(M_PI * x) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + M_PI * y * std::sin(M_PI * x) * std::cos(M_PI * t) * std::cos((1.0/2.0) * M_PI * y) + (-1.0/2.0 * M_PI * y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin((1.0/2.0) * M_PI * y) + std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos((1.0/2.0) * M_PI * y)) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t);
    //         else
    //             return cf * std::sqrt(std::pow(y, 2) * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) * std::pow(std::cos((1.0/2.0) * M_PI * y), 2) + std::pow(std::sin(M_PI * x), 2) * std::pow(std::sin(M_PI * y), 2) * std::pow(std::cos(M_PI * t), 2)) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t)/(x * y * std::cos(M_PI * t) + 1) + g * x * std::cos(M_PI * t) + M_PI * y * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * t) * std::cos(M_PI * x) * std::cos((1.0/2.0) * M_PI * y) - M_PI * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin(M_PI * y) + M_PI * std::pow(std::sin(M_PI * x), 2) * std::sin(M_PI * y) * std::pow(std::cos(M_PI * t), 2) * std::cos(M_PI * y);   
    //     }
    // };

    // Constructor. We provide the final time and time step Delta t
    // parameter as constructor arguments.
    Shallow_waters(
        const std::string &mesh_file_name_,
        const unsigned int &degree_velocity_,
        const unsigned int &degree_height_,
        const double &T_,
        const double &deltat_)
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
          mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
          pcout(std::cout, mpi_rank == 0),
          T(T_),
          mesh_file_name(mesh_file_name_),
          degree_velocity(degree_velocity_),
          degree_height(degree_height_),   
          deltat(deltat_),
          mesh(MPI_COMM_WORLD)
    {}

    // Initialization.
    void
    setup();

    // Solve the problem.
    void
    solve();

    // Compute the error.
    double
    compute_h_error(const VectorTools::NormType &norm_type);
    double
    compute_u_error(const VectorTools::NormType &norm_type);

protected:
    // Assemble the mass matrix, stiffness matrix and rhs for the height equation.
    void assemble_lhs_rhs_h(const double &time);

    // Assemble the mass matrix, stiffness matrix and rhs for the velocity equation.
    void assemble_lhs_rhs_u(const double &time);

    // Solve the problem for one time step.
    void solve_time_step(TrilinosWrappers::SparseMatrix &,
                         TrilinosWrappers::MPI::Vector &,
                         TrilinosWrappers::MPI::Vector &,
                         TrilinosWrappers::MPI::Vector &);

    // Output.
    void output(const unsigned int &time_step) const;

    // MPI parallel. /////////////////////////////////////////////////////////////

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    // Problem definition. ///////////////////////////////////////////////////////

    // Current time.
    double time;

    // Final time.
    const double T;

    // Problem specifications.
    using P_C = PCase<dim>;
    P_C p_case;

    // Discretization. ///////////////////////////////////////////////////////////

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree.
    const unsigned int degree_velocity;
    const unsigned int degree_height;

    // Time step.
    const double deltat;

    // Theta parameter of the theta method.
    // NOTE: this should not be changed, unless you know what you are doing
    static constexpr double theta = 0.5;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element spaces.
    std::unique_ptr<FiniteElement<dim>> fe_h;
    std::unique_ptr<FiniteElement<dim>> fe_u;

    // Quadrature formulas.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // DoF handlers.
    DoFHandler<dim> dof_handler_h;
    DoFHandler<dim> dof_handler_u;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs_h;
    IndexSet locally_owned_dofs_u;

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs_h;
    IndexSet locally_relevant_dofs_u;

    // Mass matrix M / deltat.
    TrilinosWrappers::SparseMatrix mass_matrix_h;
    TrilinosWrappers::SparseMatrix mass_matrix_u;

    // Stiffness matrix A.
    TrilinosWrappers::SparseMatrix stiffness_matrix_h;
    TrilinosWrappers::SparseMatrix stiffness_matrix_u;

    // Matrix on the left-hand side (M / deltat + theta A).
    TrilinosWrappers::SparseMatrix lhs_matrix_h;
    TrilinosWrappers::SparseMatrix lhs_matrix_u;

    // Matrix on the right-hand side (M / deltat - (1 - theta) A).
    TrilinosWrappers::SparseMatrix rhs_matrix_h;
    TrilinosWrappers::SparseMatrix rhs_matrix_u;

    // Right-hand side vector in the linear system.
    TrilinosWrappers::MPI::Vector system_rhs_h;
    TrilinosWrappers::MPI::Vector system_rhs_u;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::Vector solution_owned_h;
    TrilinosWrappers::MPI::Vector solution_owned_u;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::Vector solution_h;
    TrilinosWrappers::MPI::Vector solution_u;

    // Previous time step solutions (used to compute u* or forcing term for velocity system).
    TrilinosWrappers::MPI::Vector previous_solution_h;
    TrilinosWrappers::MPI::Vector previous_solution_u;
};

#include "Shallow_waters_impl.hpp"

#endif