#include "Shallow_waters.hpp"

void Shallow_waters::setup()
{
    pcout << "===============================================" << std::endl;

    // Create the mesh.
    {
        pcout << "Initializing the mesh" << std::endl;

        Triangulation<dim> mesh_serial;

        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh_serial);

        std::ifstream grid_in_file(mesh_file_name);
        grid_in.read_msh(grid_in_file);

        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const auto construction_data = TriangulationDescription::Utilities::
            create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);

        pcout << "  Number of elements = " << mesh.n_global_active_cells()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element spaces.
    {
        pcout << "Initializing the finite element spaces" << std::endl;

        fe_h = std::make_unique<FE_SimplexP<dim>>(degree_height);
        FE_SimplexP<dim> fe_scalar(degree_velocity);
        fe_u = std::make_unique<FESystem<dim>>(fe_scalar, dim);

        pcout << "  Degree (height)           = " << fe_h->degree << std::endl;
        pcout << "  DoFs per cell             = " << fe_h->dofs_per_cell << std::endl;
        pcout << "  Degree (velocity)         = " << fe_u->degree << std::endl;
        pcout << "  DoFs per cell             = " << fe_u->dofs_per_cell << std::endl;

        // This is needed since we have to use height data when constructing the stiffness
        // matrix for u, and viceversa
        quadrature = std::make_unique<QGaussSimplex<dim>>(std::max(degree_height, degree_velocity) + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handlers.
    {
        pcout << "Initializing the DoF handlers" << std::endl;

        dof_handler_h.reinit(mesh);
        dof_handler_h.distribute_dofs(*fe_h);
        locally_owned_dofs_h = dof_handler_h.locally_owned_dofs();
        locally_relevant_dofs_h = DoFTools::extract_locally_relevant_dofs(dof_handler_h);

        dof_handler_u.reinit(mesh);
        dof_handler_u.distribute_dofs(*fe_u);
        locally_owned_dofs_u = dof_handler_u.locally_owned_dofs();
        locally_relevant_dofs_u = DoFTools::extract_locally_relevant_dofs(dof_handler_u);

        pcout << "  Number of DoFs (height) = " << dof_handler_h.n_dofs() << std::endl;
        pcout << "  Number of DoFs (velocity) = " << dof_handler_u.n_dofs() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        pcout << "Initializing the linear systems" << std::endl;

        pcout << "  Initializing the sparsity patterns" << std::endl;

        TrilinosWrappers::SparsityPattern sparsity_h(locally_owned_dofs_h, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler_h, sparsity_h);
        sparsity_h.compress();

        TrilinosWrappers::SparsityPattern sparsity_u(locally_owned_dofs_u, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler_u, sparsity_u);
        sparsity_u.compress();

        pcout << "  Initializing the matrices" << std::endl;
        mass_matrix_h.reinit(sparsity_h);
        stiffness_matrix_h.reinit(sparsity_h);
        lhs_matrix_h.reinit(sparsity_h);
        rhs_matrix_h.reinit(sparsity_h);

        // mass_matrix_u.reinit(sparsity_u);
        // stiffness_matrix_u.reinit(sparsity_u);
        // lhs_matrix_u.reinit(sparsity_u);
        // rhs_matrix_u.reinit(sparsity_u);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs_h.reinit(locally_owned_dofs_h, MPI_COMM_WORLD);

        // system_rhs_u.reinit(locally_owned_dofs_u, MPI_COMM_WORLD);

        pcout << "  Initializing the solution vectors" << std::endl;
        solution_owned_h.reinit(locally_owned_dofs_h, MPI_COMM_WORLD);
        solution_h.reinit(locally_owned_dofs_h, locally_relevant_dofs_h, MPI_COMM_WORLD);
        // previous_solution_h.reinit(locally_owned_dofs_h, locally_relevant_dofs_h, MPI_COMM_WORLD);

        solution_owned_u.reinit(locally_owned_dofs_u, MPI_COMM_WORLD);
        solution_u.reinit(locally_owned_dofs_u, locally_relevant_dofs_u, MPI_COMM_WORLD);
        // previous_solution_u.reinit(locally_owned_dofs_u, locally_relevant_dofs_u, MPI_COMM_WORLD);
    }
}

void Shallow_waters::assemble_mass_matrix_h()
{
    const unsigned int dofs_per_cell = fe_h->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values_h(
        *fe_h,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    mass_matrix_h = 0.0;
    for (const auto &cell : dof_handler_h.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values_h.reinit(cell);
        cell_mass_matrix = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Mass term
                    cell_mass_matrix(i, j) += fe_values_h.shape_value(i, q) * fe_values_h.shape_value(j, q) * fe_values_h.JxW(q);
                }
            }
        }

        cell->get_dof_indices(dof_indices);
        mass_matrix_h.add(dof_indices, cell_mass_matrix);
    }

    mass_matrix_h.compress(VectorOperation::add);
}

void Shallow_waters::assemble_rhs_h(const double &time)
{
    const unsigned int dofs_per_cell = fe_h->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values_h(*fe_h,
                              *quadrature,
                              update_values | update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_u(*fe_u,
                              *quadrature,
                              update_values | update_quadrature_points | update_JxW_values);

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_rhs_h = 0.0;

    auto cell_h = dof_handler_h.begin_active();
    auto cell_u = dof_handler_u.begin_active();

    const auto endc = dof_handler_h.end();

    for (; cell_h != endc; ++cell_h, ++cell_u)
    {
        if (!cell_h->is_locally_owned())
            continue;

        fe_values_h.reinit(cell_h);
        fe_values_u.reinit(cell_u);

        cell_rhs = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            // Compute departure point
            exact_solution_u.set_time(time-deltat);
            Vector<double> u_exact(dim);
            exact_solution_u.vector_value(fe_values_u.quadrature_point(q), u_exact);

            Point<dim> p_departure;
            Point<dim> p_arrival = fe_values_h.quadrature_point(q);
            for (unsigned int d = 0; d < dim; ++d)
                p_departure[d] = p_arrival[d] + deltat * u_exact[d];

            exact_solution_h.set_time(time-deltat);
            const double h_loc = exact_solution_h.value(p_departure);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                cell_rhs(i) += h_loc * fe_values_h.shape_value(i, q) * fe_values_h.JxW(q);
            }
        }

        cell_h->get_dof_indices(dof_indices);
        system_rhs_h.add(dof_indices, cell_rhs);
    }

    system_rhs_h.compress(VectorOperation::add);
}

void Shallow_waters::solve_time_step(/*TrilinosWrappers::SparseMatrix &lhs_matrix,
                                     TrilinosWrappers::MPI::Vector &system_rhs,
                                     TrilinosWrappers::MPI::Vector &solution_owned,
                                     TrilinosWrappers::MPI::Vector &solution*/
)
{
    pcout << "===============================================" << std::endl;

    SolverControl solver_control(2000, 1e-6 * system_rhs_h.l2_norm());

    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    pcout << "Solving the linear system" << std::endl;
    solver.solve(mass_matrix_h, solution_owned_h, system_rhs_h, PreconditionIdentity());
    pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

    solution_h = solution_owned_h;
}

void Shallow_waters::output(const unsigned int &time_step) const
{
    DataOut<dim> data_out;

    data_out.add_data_vector(dof_handler_h, solution_h, "h");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_names(dim, "u");
    data_out.add_data_vector(dof_handler_u, solution_u, solution_names, data_component_interpretation);

    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
        "./vtk/", "output", time_step, MPI_COMM_WORLD, 3);
}

void Shallow_waters::solve()
{
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the mass matrix for the height equation" << std::endl;
    assemble_mass_matrix_h();

    pcout << "===============================================" << std::endl;

    time = 0.0;

    // Apply the initial condition.
    {
        pcout << "Applying the initial condition" << std::endl;

        exact_solution_h.set_time(time);
        VectorTools::interpolate(dof_handler_h, exact_solution_h, solution_owned_h);
        solution_h = solution_owned_h;

        exact_solution_u.set_time(time);
        VectorTools::interpolate(dof_handler_u, exact_solution_u, solution_owned_u);
        solution_u = solution_owned_u;

        // Output the initial solution.
        output(0);

        // previous_solution_h = solution_h;
        // previous_solution_u = solution_u;

        // time += deltat;

        // exact_solution_h.set_time(time);
        // VectorTools::interpolate(dof_handler_h, exact_solution_h, solution_owned_h);
        // solution_h = solution_owned_h;

        // exact_solution_u.set_time(time);
        // VectorTools::interpolate(dof_handler_u, exact_solution_u, solution_owned_u);
        // solution_u = solution_owned_u;

        // output(1);
    }

    unsigned int time_step = 0;

    while (time < T - 0.5 * deltat)
    {
        time += deltat;
        time_step++;

        pcout << "===============================================" << std::endl;
        pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
              << time << std::endl;
        pcout << "-----------------------------------------------" << std::endl;

        // Solve for h.
        assemble_rhs_h(time);
        solve_time_step(/*lhs_matrix_h,
                        system_rhs_h,
                        solution_owned_h,
                        solution_h*/
        );

        // For now, use the exact solution at each time step
        exact_solution_u.set_time(time);
        VectorTools::interpolate(dof_handler_u, exact_solution_u, solution_owned_u);
        solution_u = solution_owned_u;

        output(time_step);
    }
}

double Shallow_waters::compute_error(const VectorTools::NormType &norm_type)
{
    return 1.0;
}