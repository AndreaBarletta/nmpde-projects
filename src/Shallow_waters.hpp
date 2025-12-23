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

using namespace dealii;

// Class representing the non-linear diffusion problem.
class Shallow_waters
{
public:
    // Physical dimension (2D)
    static constexpr unsigned int dim = 2;

    // Intial conditions solutions.
    class InitialConditions_h : public Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            // Centered Gaussian bump
            // return 0.5 + 0.2 * std::exp(-((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.5) * (p[1] - 0.5)) * 500.0);

            // Offset Gaussian bump
            // return 0.5 + 0.2 * std::exp(-((p[0] - 0.25) * (p[0] - 0.25) + (p[1] - 0.25) * (p[1] - 0.25)) * 500.0);

            // Sloping plane
            return 1.0 - 0.5 * p[0];

            // Disappearing dam break
            // return p[0] < 0.5 ? 1.0 : 0.5;
        }
    };

    class InitialConditions_u : public Function<dim>
    {
    public:
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            values[0] = 0.0;
            values[1] = 0.0;
        }

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 0.0;
            else
                return 0.0;
        }
    };

    // Exact solutions.
    class ExactSolution_h : public Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            #ifdef TEST_MANUFACTURED_H
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];
            const double pi = M_PI;
            #define sin std::sin
            #define cos std::cos
            #define pow std::pow

            return 1 + 0.5 * cos(pi*t) * sin(pi*x) * sin(pi*y);

            #undef sin
            #undef cos
            #undef pow
            #else
            return 0.0;
            #endif
        }
    };

    class ExactSolution_u : public Function<dim>
    {
    public:
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            #ifnef TEST_MANUFACTURED_U
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];
            const double pi = M_PI;
            #define sin std::sin
            #define cos std::cos
            #define pow std::pow
            #define sqrt std::sqrt

            values[0] = y * sin(pi*t) * cos(pi*x) * cos(pi*y/2);
            values[1] = pow(sin(pi*x),2) * cos(pi*t);

            #undef sin
            #undef cos
            #undef pow
            #undef sqrt
            #else
            values[0] = 0.0;
            values[1] = 0.0;
            #endif
        }

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            #ifdef TEST_MANUFACTURED_U
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];
            const double pi = M_PI;
            #define sin std::sin
            #define cos std::cos
            #define pow std::pow

            if (component == 0)
                return y * sin(pi*t) * cos(pi*x) * cos(pi*y/2);
            else
                return pow(sin(pi*x),2) * cos(pi*t);
            #undef sin
            #undef cos
            #undef pow
            
            #else
            if (component == 0)
                return 0.0;
            else
                return 0.0;
            #endif
        }
    };

    // Forcing terms (used mainly for manufactured-solution testing)
    class ForcingTerm_h : public Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            #ifdef TEST_MANUFACTURED
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];
            const double pi = M_PI;
            #define sin std::sin
            #define cos std::cos
            #define pow std::pow

            return 0.5 * pi * (
                    y * sin(pi*t) * sin(pi*y) * cos(pi*t) * pow(cos(pi*x),2) * cos(pi*y/2)
                  - sin(pi*t) * sin(pi*x) * sin(pi*y)
                  + pow(sin(pi*x),3) * pow(cos(pi*t),2) * cos(pi*y)
            );

            #undef sin
            #undef cos
            #undef pow
            #else
            return 0.0;
            #endif
        }
    };

    class ForcingTerm_u : public Function<dim>
    {
    public:
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            #ifdef TEST_MANUFACTURED
            const double t = this->get_time();
            const double x = p[0];
            const double y = p[1];
            const double pi = M_PI;
            const double g = Shallow_waters::g;
            const double cf = Shallow_waters::cf;
            #define sin std::sin
            #define cos std::cos
            #define pow std::pow
            #define sqrt std::sqrt

            values[0] = (
                2 * cf * y * sqrt(
                        pow(y,2) * pow(sin(pi*t), 2) * pow(cos(pi*x),2) * pow(cos(pi*y/2),2)
                        + pow(sin(pi*x),4) * pow(cos(pi*t),2)
                    ) * sin(pi*t) * cos(pi*y/2)
                + (1.0 + 0.5 * sin(pi*x) * sin(pi*y) * cos(pi*t)) 
                    * (1.0 * pi * g * sin(pi*y) * cos(pi*t)
                        - 2 * pi * pow(y,2) * pow(sin(pi*t),2) * sin(pi*x) * pow(cos(pi*y/2),2)
                        + 2 * pi * y * cos(pi*t) * cos(pi*y/2)
                        - (pi * y * sin(pi*y/2) - 2 * cos(pi*y/2)) * sin(pi*t) * pow(sin(pi*x),2) * cos(pi*t)
                    )
                ) * cos(pi*x)
                / (2 * (0.5 * sin(pi*x) * sin(pi*y) * cos(pi*t) + 1.0));


            values[1] = (
                cf * sqrt(
                        pow(y,2) * pow(sin(pi*t),2) * pow(cos(pi*x),2) * pow(cos(pi*y/2),2)
                        + pow(sin(pi*x),4) * pow(cos(pi*t),2)
                    ) * sin(pi*x) * cos(pi*t)
                + pi * (
                        0.5 * sin(pi*x) * sin(pi*y) * cos(pi*t) 
                        + 1.0
                    ) * (
                        0.5 * g * cos(pi*t) * cos(pi*y)
                        + 2 * y * sin(pi*t) * cos(pi*t) * pow(cos(pi*x),2) * cos(pi*y/2)
                        - sin(pi*t) * sin(pi*x)
                    )
                ) * sin(pi*x)
                / (0.5 * sin(pi*x) * sin(pi*y) * cos(pi*t) + 1.0);
            
            #undef sin
            #undef cos
            #undef pow
            #undef sqrt

            #else
            values[0] = 0.0;
            values[1] = 0.0;
            #endif
        }

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 0.0;
            else
                return 0.0;
        }
    };

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
    {
    }

    // Initialization.
    void
    setup();

    // Solve the problem.
    void
    solve();

    // Compute the error.
    double
    compute_error(const VectorTools::NormType &norm_type);

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

    // Gravitational acceleration
    static constexpr double g = 9.81e-1;

    // Chézy’s friction coefficient
    static constexpr const double cf = 3.0e0;

    // Initial conditions
    InitialConditions_h initial_conditions_h;
    InitialConditions_u initial_conditions_u;

    // Exact solutions
    ExactSolution_h exact_solution_h;
    ExactSolution_u exact_solution_u;

    // Forcing term
    ForcingTerm_h forcing_term_h;
    ForcingTerm_u forcing_term_u;

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

#endif