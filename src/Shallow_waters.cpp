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

        mass_matrix_u.reinit(sparsity_u);
        stiffness_matrix_u.reinit(sparsity_u);
        lhs_matrix_u.reinit(sparsity_u);
        rhs_matrix_u.reinit(sparsity_u);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs_h.reinit(locally_owned_dofs_h, MPI_COMM_WORLD);
        system_rhs_u.reinit(locally_owned_dofs_u, MPI_COMM_WORLD);

        pcout << "  Initializing the solution vectors" << std::endl;
        solution_owned_h.reinit(locally_owned_dofs_h, MPI_COMM_WORLD);
        solution_h.reinit(locally_owned_dofs_h, locally_relevant_dofs_h, MPI_COMM_WORLD);
        previous_solution_h.reinit(locally_owned_dofs_h, locally_relevant_dofs_h, MPI_COMM_WORLD);

        solution_owned_u.reinit(locally_owned_dofs_u, MPI_COMM_WORLD);
        solution_u.reinit(locally_owned_dofs_u, locally_relevant_dofs_u, MPI_COMM_WORLD);
        previous_solution_u.reinit(locally_owned_dofs_u, locally_relevant_dofs_u, MPI_COMM_WORLD);
    }
}

void Shallow_waters::assemble_lhs_rhs_h(const double &time)
{
    pcout << "-----------------------------------------------" << std::endl;
    pcout << "  Assembling the lhs and rhs for the height system" << std::endl;

    const unsigned int dofs_per_cell = fe_h->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values_h(
        *fe_h,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_u(
        *fe_u,
        *quadrature,
        update_values | update_gradients | update_quadrature_points);

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    mass_matrix_h = 0.0;
    stiffness_matrix_h = 0.0;
    system_rhs_h = 0.0;

    auto cell_h = dof_handler_h.begin_active();
    auto cell_u = dof_handler_u.begin_active();

    const auto endc = dof_handler_h.end();

    for (; cell_h != endc; ++cell_h, ++cell_u)
    {
        if (!cell_h->is_locally_owned() || !cell_u->is_locally_owned())
            continue;

        fe_values_h.reinit(cell_h);
        fe_values_u.reinit(cell_u);

        cell_mass_matrix = 0.0;
        cell_stiffness_matrix = 0.0;
        cell_rhs = 0.0;

        // u at time n on the quadrature points of this cell
        std::vector<Vector<double>> u_prev(n_q, Vector<double>(dim));
        fe_values_u.get_function_values(solution_u, u_prev);

        // u at time n-1 on the quadrature points of this cell
        std::vector<Vector<double>> u_prevprev(n_q, Vector<double>(dim));
        fe_values_u.get_function_values(previous_solution_u, u_prevprev);

        // div u at time n on the quadrature points of this cell
        std::vector<double> div_u_prev(n_q);
        fe_values_u[FEValuesExtractors::Vector(0)].get_function_divergences(solution_u, div_u_prev);

        // div u at time n-1 on the quadrature points of this cell
        std::vector<double> div_u_prevprev(n_q);
        fe_values_u[FEValuesExtractors::Vector(0)].get_function_divergences(previous_solution_u, div_u_prevprev);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            // We interpolate u at time n+1/2 = 3/2 u^n - 1/2 u^{n-1}
            Tensor<1, dim> u_interp;
            for (unsigned int d = 0; d < dim; ++d)
                u_interp[d] = 1.5 * u_prev[q][d] - 0.5 * u_prevprev[q][d];

            // Same for div u
            const double div_u_interp = 1.5 * div_u_prev[q] - 0.5 * div_u_prevprev[q];

            forcing_term_h.set_time(time);
            const double f_new_loc =
                forcing_term_h.value(fe_values_h.quadrature_point(q));

            forcing_term_h.set_time(time - deltat);
            const double f_old_loc =
                forcing_term_h.value(fe_values_h.quadrature_point(q));


            // Compute SUPG parameter
            double tau1 = std::pow(2.0/deltat,2.0);
            double tau2 = std::pow(2.0*u_interp.norm()/cell_h->diameter(),2.0);

            double SUPGtau = 1.0/std::sqrt(tau1 + tau2);

            // SUPGtau = 0; // DISABLE SUPG

            //  i is the test function, j is the solution function
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                // Compute the SS operator Lss(phi_i)
                double Lss = u_interp * fe_values_h.shape_grad(i,q) + 1.0/2.0 * div_u_interp * fe_values_h.shape_value(i,q);

                // phi_i terms
                double phi_i = fe_values_h.shape_value(i,q);
                Tensor<1, dim> grad_phi_i = fe_values_h.shape_grad(i,q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // phi_j terms
                    double phi_j = fe_values_h.shape_value(j,q);
                    Tensor<1, dim> grad_phi_j = fe_values_h.shape_grad(j,q);

                    // Mass term
                    cell_mass_matrix(i, j) += phi_i * phi_j / deltat * fe_values_h.JxW(q);

                    // Stiffness term
                    cell_stiffness_matrix(i, j) += -u_interp * grad_phi_i * phi_j * fe_values_h.JxW(q);

                    // SUPG "mass-like" term (come from dh/dt)
                    cell_mass_matrix(i,j) += SUPGtau * Lss * phi_j * fe_values_h.JxW(q);

                    // SUPG "stiffness-like" term
                    cell_stiffness_matrix(i,j) += SUPGtau * (phi_j * div_u_interp + grad_phi_j * u_interp) * Lss * fe_values_h.JxW(q);
                }

                // Forcing term
                cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) * phi_i * fe_values_h.JxW(q);

                // SUPG RHS term
                cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) * SUPGtau * Lss * fe_values_h.JxW(q);
            }
        }

        cell_h->get_dof_indices(dof_indices);

        mass_matrix_h.add(dof_indices, cell_mass_matrix);
        stiffness_matrix_h.add(dof_indices, cell_stiffness_matrix);
        system_rhs_h.add(dof_indices, cell_rhs);
    }

    mass_matrix_h.compress(VectorOperation::add);
    stiffness_matrix_h.compress(VectorOperation::add);
    system_rhs_h.compress(VectorOperation::add);

    // Assemble lhs matrix
    lhs_matrix_h.copy_from(mass_matrix_h);
    lhs_matrix_h.add(theta, stiffness_matrix_h);

    // Assemble rhs matrix
    rhs_matrix_h.copy_from(mass_matrix_h);
    rhs_matrix_h.add(-(1.0 - theta), stiffness_matrix_h);

    // Assemble rhs vector
    rhs_matrix_h.vmult_add(system_rhs_h, solution_owned_h);
}

void Shallow_waters::assemble_lhs_rhs_u(const double &time)
{
    pcout << "-----------------------------------------------" << std::endl;
    pcout << "  Assembling the lhs and rhs for the velocity system" << std::endl;

    const unsigned int dofs_per_cell = fe_u->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values_u(
        *fe_u,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_h(
        *fe_h,
        *quadrature,
        update_values | update_gradients | update_quadrature_points);

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    mass_matrix_u = 0.0;
    stiffness_matrix_u = 0.0;
    system_rhs_u = 0.0;

    FEValuesExtractors::Vector displacement(0);

    auto cell_h = dof_handler_h.begin_active();
    auto cell_u = dof_handler_u.begin_active();

    const auto endc = dof_handler_h.end();

    for (; cell_u != endc; ++cell_u, ++cell_h)
    {
        if (!cell_u->is_locally_owned() || !cell_h->is_locally_owned())
            continue;

        fe_values_u.reinit(cell_u);
        fe_values_h.reinit(cell_h);

        cell_mass_matrix = 0.0;
        cell_stiffness_matrix = 0.0;
        cell_rhs = 0.0;

        // u at time n on the quadrature points of this cell
        std::vector<Vector<double>> u_prev(n_q, Vector<double>(dim));
        fe_values_u.get_function_values(solution_u, u_prev);

        // u at time n-1 on the quadrature points of this cell
        std::vector<Vector<double>> u_prevprev(n_q, Vector<double>(dim));
        fe_values_u.get_function_values(previous_solution_u, u_prevprev);

        // h at time n+1 on the quadrature points of this cell
        std::vector<double> h_curr(n_q);
        fe_values_h.get_function_values(solution_h, h_curr);

        // h at time n on the quadrature points of this cell
        std::vector<double> h_prev(n_q);
        fe_values_h.get_function_values(previous_solution_h, h_prev);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            // we interpolate u at time n+1/2 = 3/2 u^n - 1/2 u^{n-1}
            Tensor<1, dim> u_interp;
            for (unsigned int d = 0; d < dim; ++d)
                u_interp[d] = 1.5 * u_prev[q][d] - 0.5 * u_prevprev[q][d];

            // Compute forcing term
            forcing_term_u.set_time(time);
            Vector<double> f_new_loc(dim);
                forcing_term_u.vector_value(fe_values_u.quadrature_point(q), f_new_loc);

            forcing_term_u.set_time(time - deltat);
            Vector<double> f_old_loc(dim);
            forcing_term_u.vector_value(fe_values_u.quadrature_point(q), f_old_loc);

            // Convert to Tensor for easier computation
            Tensor<1, dim> f_new_loc_tensor;
            Tensor<1, dim> f_old_loc_tensor;
            for (unsigned int d = 0; d < dim; ++d)
            {
                f_new_loc_tensor[d] = f_new_loc[d];
                f_old_loc_tensor[d] = f_old_loc[d];
            }
                
            //  i is the test function, j is the solution function
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                // phi_i terms
                Tensor<1, dim> phi_i = fe_values_u[displacement].value(i, q);
                Tensor<2, dim> grad_phi_i = fe_values_u[displacement].gradient(i,q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                     // phi_j terms
                    Tensor<1, dim> phi_j = fe_values_u[displacement].value(j, q);
                    Tensor<2, dim> grad_phi_j = fe_values_u[displacement].gradient(j,q);

                    // Mass term
                    cell_mass_matrix(i, j) += phi_i * phi_j / deltat * fe_values_u.JxW(q);

                    // Linearized Convective term (u* grad)u
                    cell_stiffness_matrix(i, j) += u_interp * grad_phi_i * phi_j * fe_values_u.JxW(q);
                    
                    // Friction term cf/a ||u|| u
                    cell_stiffness_matrix(i, j) += cf / (h_curr[q]) * u_interp.norm() * phi_i * phi_j * fe_values_u.JxW(q);
                }

                // g grad H term
                cell_rhs(i) += g * (theta * h_curr[q] + (1.0 - theta) * h_prev[q]) * fe_values_u[displacement].divergence(i,q) * fe_values_u.JxW(q);

                 // Forcing term
                cell_rhs(i) += (theta * f_new_loc_tensor + (1.0 - theta) * f_old_loc_tensor) * phi_i * fe_values_u.JxW(q);
            }
        }

        cell_u->get_dof_indices(dof_indices);

        mass_matrix_u.add(dof_indices, cell_mass_matrix);
        stiffness_matrix_u.add(dof_indices, cell_stiffness_matrix);
        system_rhs_u.add(dof_indices, cell_rhs);
    }

    mass_matrix_u.compress(VectorOperation::add);
    stiffness_matrix_u.compress(VectorOperation::add);
    system_rhs_u.compress(VectorOperation::add);

    // Assemble lhs matrix
    lhs_matrix_u.copy_from(mass_matrix_u);
    lhs_matrix_u.add(theta, stiffness_matrix_u);

    // Assemble rhs matrix
    rhs_matrix_u.copy_from(mass_matrix_u);
    rhs_matrix_u.add(-(1.0 - theta), stiffness_matrix_u);

    // Assemble rhs vector
    rhs_matrix_u.vmult_add(system_rhs_u, solution_owned_u);

    // Boundary conditions.
    {
        std::map<types::global_dof_index, double> boundary_values;
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;

        Functions::ZeroFunction<dim> zero_function(dim);
        for (unsigned int i = 0; i < 4; ++i)
            boundary_functions[i] = &zero_function;

        VectorTools::interpolate_boundary_values(dof_handler_u,
                                                 boundary_functions,
                                                 boundary_values);

        MatrixTools::apply_boundary_values(
            boundary_values, lhs_matrix_u, solution_owned_u, system_rhs_u, false);
    }
}

void Shallow_waters::solve_time_step(TrilinosWrappers::SparseMatrix &lhs_matrix,
                                     TrilinosWrappers::MPI::Vector &system_rhs,
                                     TrilinosWrappers::MPI::Vector &solution_owned,
                                     TrilinosWrappers::MPI::Vector &solution
)
{
    pcout << "-----------------------------------------------" << std::endl;

    SolverControl solver_control(2000, 1e-6 * system_rhs.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

    TrilinosWrappers::PreconditionILU preconditioner;
    preconditioner.initialize(lhs_matrix);

    pcout << "Solving the linear system" << std::endl;
    solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
    pcout << "  " << solver_control.last_step() << " GMRES iterations"
          << std::endl;

    solution = solution_owned;
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

    time = 0.0;

    // Apply the initial condition.
    {
        pcout << "Applying the initial condition" << std::endl;

        initial_conditions_h.set_time(time);
        VectorTools::interpolate(dof_handler_h, initial_conditions_h, solution_owned_h);
        solution_h = solution_owned_h;

        initial_conditions_u.set_time(time);
        VectorTools::interpolate(dof_handler_u, initial_conditions_u, solution_owned_u);
        solution_u = solution_owned_u;

        // Output the initial solution.
        output(0);

        previous_solution_h = solution_h;
        previous_solution_u = solution_u;
    }

    unsigned int time_step = 0.0;

    while (time < T - 0.5 * deltat)
    {
        time += deltat;
        ++time_step;

        pcout << "===============================================" << std::endl;
        pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
              << time << std::endl;

        // start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Solve for h.
        assemble_lhs_rhs_h(time);
        previous_solution_h = solution_h;
        solve_time_step(lhs_matrix_h, system_rhs_h, solution_owned_h, solution_h);
        // Now solution_h contains h at timestep n+1, previous_solution_h contains h at timestep n
        // The next iteration (n=n+1) will therefore have solution_h at timestep n and previous_solution_h at timestep n-1

        // // Solve for u.
        assemble_lhs_rhs_u(time);
        previous_solution_u = solution_u;
        solve_time_step(lhs_matrix_u, system_rhs_u, solution_owned_u, solution_u);
        // // Now solution_u contains u at timestep n+1, previous_solution_u contains u at timestep n
        // // The next iteration (n=n+1) will therefore have solution_u at timestep n and previous_solution_u at timestep n-1

        // end timing
        auto end = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // Output the solution
        output(time_step);
        pcout << "-----------------------------------------------" << std::endl;
        pcout << "Time step computation time: " << ns << " ns" << std::endl;
    }
}

double
Shallow_waters::compute_error(const VectorTools::NormType &norm_type)
{
  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);

  const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(degree_height + 2);

  exact_solution_h.set_time(time);

  Vector<double> error_per_cell;
  VectorTools::integrate_difference(mapping,
                                    dof_handler_h,
                                    solution_h,
                                    exact_solution_h,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}