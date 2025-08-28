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
        fe_u = std::make_unique<FE_SimplexP<dim>>(degree_velocity);
        fe_v = std::make_unique<FE_SimplexP<dim>>(degree_velocity);

        pcout << "  Degree (height)           = " << fe_h->degree << std::endl;
        pcout << "  DoFs per cell             = " << fe_h->dofs_per_cell << std::endl;
        pcout << "  Degree (velocity)         = " << fe_u->degree << std::endl;
        pcout << "  DoFs per cell             = " << fe_u->dofs_per_cell << std::endl;

        // quadrature_h = std::make_unique<QGaussSimplex<dim>>(degree_height + 1);
        // quadrature_u = std::make_unique<QGaussSimplex<dim>>(degree_velocity + 1);
        // quadrature_v = std::make_unique<QGaussSimplex<dim>>(degree_velocity + 1);
        quadrature = std::make_unique<QGaussSimplex<dim>>(std::max(degree_height, degree_velocity) + 1);

        // pcout << "  Quadrature points per cell (height) = " << quadrature_h->size() << std::endl;
        // pcout << "  Quadrature points per cell (velocity) = " << quadrature_u->size() << std::endl;
        pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handlers.
    {
        pcout << "Initializing the DoF handlers" << std::endl;

        dof_handler_h.reinit(mesh);
        dof_handler_h.distribute_dofs(*fe_h);
        locally_owned_dofs_h = dof_handler_h.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler_h, locally_relevant_dofs_h);

        dof_handler_u.reinit(mesh);
        dof_handler_u.distribute_dofs(*fe_u);
        locally_owned_dofs_u = dof_handler_u.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler_u, locally_relevant_dofs_u);

        dof_handler_v.reinit(mesh);
        dof_handler_v.distribute_dofs(*fe_v);
        locally_owned_dofs_v = dof_handler_v.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler_v, locally_relevant_dofs_v);

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

        TrilinosWrappers::SparsityPattern sparsity_v(locally_owned_dofs_v, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler_v, sparsity_v);
        sparsity_v.compress();

        pcout << "  Initializing the matrices" << std::endl;
        mass_matrix_h.reinit(sparsity_h);
        stiffness_matrix_h.reinit(sparsity_h);
        lhs_matrix_h.reinit(sparsity_h);
        rhs_matrix_h.reinit(sparsity_h);

        mass_matrix_u.reinit(sparsity_u);
        stiffness_matrix_u.reinit(sparsity_u);
        lhs_matrix_u.reinit(sparsity_u);
        rhs_matrix_u.reinit(sparsity_u);

        mass_matrix_v.reinit(sparsity_v);
        stiffness_matrix_v.reinit(sparsity_v);
        lhs_matrix_v.reinit(sparsity_v);
        rhs_matrix_v.reinit(sparsity_v);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs_h.reinit(locally_owned_dofs_h, MPI_COMM_WORLD);
        system_rhs_u.reinit(locally_owned_dofs_u, MPI_COMM_WORLD);
        system_rhs_v.reinit(locally_owned_dofs_v, MPI_COMM_WORLD);

        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned_h.reinit(locally_owned_dofs_h, MPI_COMM_WORLD);
        solution_owned_u.reinit(locally_owned_dofs_u, MPI_COMM_WORLD);
        solution_owned_v.reinit(locally_owned_dofs_v, MPI_COMM_WORLD);
        solution_h.reinit(locally_owned_dofs_h, locally_relevant_dofs_h, MPI_COMM_WORLD);
        solution_u.reinit(locally_owned_dofs_u, locally_relevant_dofs_u, MPI_COMM_WORLD);
        solution_v.reinit(locally_owned_dofs_v, locally_relevant_dofs_v, MPI_COMM_WORLD);
    }
}

void Shallow_waters::assemble_mass_matrix(FiniteElement<dim> *fe, DoFHandler<dim> &dof_handler, Quadrature<dim> *quadrature, TrilinosWrappers::SparseMatrix &mass_matrix)
{

    pcout << "===============================================" << std::endl;
    pcout << "Assembling the mass matrix" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    mass_matrix = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_mass_matrix = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                              fe_values.shape_value(j, q) /
                                              deltat * fe_values.JxW(q);
                }
            }
        }

        cell->get_dof_indices(dof_indices);

        mass_matrix.add(dof_indices, cell_mass_matrix);
    }

    mass_matrix.compress(VectorOperation::add);
}

void Shallow_waters::assemble_stiffness_and_rhs_h(const double &time)
{
    pcout << "===============================================" << std::endl;
    pcout << "  Assembling the stiffness and rhs for the height system" << std::endl;

    const unsigned int dofs_per_cell = fe_h->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values_h(
        *fe_h,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_u(
        *fe_u,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_v(
        *fe_v,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    std::vector<double> u_old(fe_values_u.n_quadrature_points);
    std::vector<double> v_old(fe_values_v.n_quadrature_points);
    std::vector<double> u_curr(fe_values_u.n_quadrature_points);
    std::vector<double> v_curr(fe_values_v.n_quadrature_points);

    stiffness_matrix_h = 0.0;
    system_rhs_h = 0.0;

    auto cell_h = dof_handler_h.begin_active();
    auto cell_u = dof_handler_u.begin_active();
    auto cell_v = dof_handler_v.begin_active();

    const auto endc = dof_handler_h.end();

    for (; cell_h != endc; ++cell_h, ++cell_u, ++cell_v)
    {
        if (!cell_h->is_locally_owned())
            continue;

        fe_values_h.reinit(cell_h);
        fe_values_u.reinit(cell_u);
        fe_values_v.reinit(cell_v);

        fe_values_u.get_function_values(previous_solution_u, u_old);
        fe_values_v.get_function_values(previous_solution_v, v_old);
        fe_values_u.get_function_values(solution_u, u_curr);
        fe_values_v.get_function_values(solution_v, v_curr);

        cell_matrix = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // -K1
                    cell_matrix(i, j) += -fe_values_h.shape_value(j, q) * (3.0 / 2.0 * u_curr[q] - 1.0 / 2.0 * u_old[q]) * fe_values_h.shape_grad(i, q)[0] * fe_values_h.JxW(q);
                    cell_matrix(i, j) += -fe_values_h.shape_value(j, q) * (3.0 / 2.0 * v_curr[q] - 1.0 / 2.0 * v_old[q]) * fe_values_h.shape_grad(i, q)[1] * fe_values_h.JxW(q);
                }
            }
        }

        cell_h->get_dof_indices(dof_indices);

        stiffness_matrix_h.add(dof_indices, cell_matrix);
    }

    lhs_matrix_h.copy_from(mass_matrix_h);
    lhs_matrix_h.add(theta, stiffness_matrix_h);

    rhs_matrix_h.copy_from(mass_matrix_h);
    rhs_matrix_h.add(-(1.0 - theta), stiffness_matrix_h);

    rhs_matrix_h.vmult_add(system_rhs_h, solution_h);

    // Boundary conditions.
    {
        std::map<types::global_dof_index, double> boundary_values;
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;
        // Functions::ConstantFunction<dim> function_zero(0.0);

        boundary_functions[0] = &exact_solution_h;
        boundary_functions[1] = &exact_solution_h;
        boundary_functions[2] = &exact_solution_h;
        boundary_functions[3] = &exact_solution_h;

        VectorTools::interpolate_boundary_values(dof_handler_h,
                                                 boundary_functions,
                                                 boundary_values);

        MatrixTools::apply_boundary_values(
            boundary_values, lhs_matrix_h, solution_h, system_rhs_h, true);
    }
}

void Shallow_waters::assemble_stiffness_and_rhs_u(const double &time)
{
    pcout << "===============================================" << std::endl;
    pcout << "  Assembling the stiffness and rhs for the u system" << std::endl;

    const unsigned int dofs_per_cell = fe_u->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values_h(
        *fe_h,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_u(
        *fe_u,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_v(
        *fe_v,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> h_old_grad(fe_values_h.n_quadrature_points);
    std::vector<double> u_old(fe_values_u.n_quadrature_points);
    std::vector<double> v_old(fe_values_v.n_quadrature_points);
    std::vector<Tensor<1, dim>> h_curr_grad(fe_values_h.n_quadrature_points);
    std::vector<double> u_curr(fe_values_u.n_quadrature_points);
    std::vector<double> v_curr(fe_values_v.n_quadrature_points);

    stiffness_matrix_u = 0.0;
    system_rhs_u = 0.0;

    auto cell_h = dof_handler_h.begin_active();
    auto cell_u = dof_handler_u.begin_active();
    auto cell_v = dof_handler_v.begin_active();

    const auto endc = dof_handler_u.end();

    for (; cell_u != endc; ++cell_h, ++cell_u, ++cell_v)
    {
        if (!cell_u->is_locally_owned())
            continue;

        fe_values_h.reinit(cell_h);
        fe_values_u.reinit(cell_u);
        fe_values_v.reinit(cell_v);

        fe_values_h.get_function_gradients(previous_solution_h, h_old_grad);
        fe_values_u.get_function_values(previous_solution_u, u_old);
        fe_values_v.get_function_values(previous_solution_v, v_old);
        fe_values_h.get_function_gradients(solution_h, h_curr_grad);
        fe_values_u.get_function_values(solution_u, u_curr);
        fe_values_v.get_function_values(solution_v, v_curr);

        cell_matrix = 0.0;
        cell_rhs = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // K2
                    cell_matrix(i, j) += (3.0 / 2.0 * u_curr[q] - 1.0 / 2.0 * u_old[q]) * fe_values_u.shape_value(i, q) * fe_values_u.shape_grad(j, q)[0] * fe_values_u.JxW(q);
                    cell_matrix(i, j) += (3.0 / 2.0 * v_curr[q] - 1.0 / 2.0 * v_old[q]) * fe_values_u.shape_value(i, q) * fe_values_u.shape_grad(j, q)[1] * fe_values_u.JxW(q);
                }

                //-1/2*K_{21}^{n+1}
                cell_rhs(i) += -1.0 / 2.0 * h_curr_grad[q][0] * fe_values_u.shape_value(i, q) * fe_values_u.JxW(q);
                //-1/2*K_{21}^{n}
                cell_rhs(i) += -1.0 / 2.0 * h_old_grad[q][0] * fe_values_u.shape_value(i, q) * fe_values_u.JxW(q);
                // -P2
                cell_rhs(i) += f * (3.0 / 2.0 * v_curr[q] - 1.0 / 2.0 * v_old[q]) * fe_values_u.shape_value(i, q) * fe_values_u.JxW(q);
            }
        }

        cell_u->get_dof_indices(dof_indices);

        stiffness_matrix_u.add(dof_indices, cell_matrix);
        system_rhs_u.add(dof_indices, cell_rhs);
    }

    lhs_matrix_u.copy_from(mass_matrix_u);
    lhs_matrix_u.add(theta, stiffness_matrix_u);

    rhs_matrix_u.copy_from(mass_matrix_u);
    rhs_matrix_u.add(-(1.0 - theta), stiffness_matrix_u);

    rhs_matrix_u.vmult_add(system_rhs_u, solution_u);

    // Boundary conditions.
    {
        std::map<types::global_dof_index, double> boundary_values;
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;
        // Functions::ConstantFunction<dim> function_zero(0.0);

        boundary_functions[0] = &exact_solution_u;
        boundary_functions[1] = &exact_solution_u;
        boundary_functions[2] = &exact_solution_u;
        boundary_functions[3] = &exact_solution_u;

        VectorTools::interpolate_boundary_values(dof_handler_u,
                                                 boundary_functions,
                                                 boundary_values);

        MatrixTools::apply_boundary_values(
            boundary_values, lhs_matrix_u, solution_u, system_rhs_u, true);
    }
}

void Shallow_waters::assemble_stiffness_and_rhs_v(const double &time)
{
    pcout << "===============================================" << std::endl;
    pcout << "  Assembling the stiffness and rhs for the v system" << std::endl;

    const unsigned int dofs_per_cell = fe_v->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values_h(
        *fe_h,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_u(
        *fe_u,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_v(
        *fe_v,
        *quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> h_old_grad(fe_values_h.n_quadrature_points);
    std::vector<double> v_old(fe_values_v.n_quadrature_points);
    std::vector<Tensor<1, dim>> h_curr_grad(fe_values_h.n_quadrature_points);
    std::vector<double> u_curr(fe_values_u.n_quadrature_points);
    std::vector<double> v_curr(fe_values_v.n_quadrature_points);

    stiffness_matrix_v = 0.0;
    system_rhs_v = 0.0;

    auto cell_h = dof_handler_h.begin_active();
    auto cell_u = dof_handler_u.begin_active();
    auto cell_v = dof_handler_v.begin_active();

    const auto endc = dof_handler_v.end();

    for (; cell_v != endc; ++cell_h, ++cell_u, ++cell_v)
    {
        if (!cell_v->is_locally_owned())
            continue;

        fe_values_h.reinit(cell_h);
        fe_values_u.reinit(cell_u);
        fe_values_v.reinit(cell_v);

        fe_values_h.get_function_gradients(previous_solution_h, h_old_grad);
        fe_values_v.get_function_values(previous_solution_v, v_old);
        fe_values_h.get_function_gradients(solution_h, h_curr_grad);
        fe_values_u.get_function_values(solution_u, u_curr);
        fe_values_v.get_function_values(solution_v, v_curr);

        cell_matrix = 0.0;
        cell_rhs = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // K3
                    cell_matrix(i, j) += u_curr[q] * fe_values_u.shape_value(i, q) * fe_values_u.shape_grad(j, q)[0] * fe_values_u.JxW(q);
                    cell_matrix(i, j) += (3.0 / 2.0 * v_curr[q] - 1.0 / 2.0 * v_old[q]) * fe_values_u.shape_value(i, q) * fe_values_u.shape_grad(j, q)[1] * fe_values_u.JxW(q);
                }

                //-1/2*K_{31}^{n+1}
                cell_rhs(i) += -1.0 / 2.0 * h_curr_grad[q][1] * fe_values_u.shape_value(i, q) * fe_values_u.JxW(q);
                //-1/2*K_{31}^{n}
                cell_rhs(i) += -1.0 / 2.0 * h_old_grad[q][1] * fe_values_u.shape_value(i, q) * fe_values_u.JxW(q);
                // -P3
                cell_rhs(i) += -f * u_curr[q] * fe_values_u.shape_value(i, q) * fe_values_u.JxW(q);
            }
        }

        cell_v->get_dof_indices(dof_indices);

        stiffness_matrix_v.add(dof_indices, cell_matrix);
        system_rhs_v.add(dof_indices, cell_rhs);
    }

    lhs_matrix_v.copy_from(mass_matrix_v);
    lhs_matrix_v.add(theta, stiffness_matrix_v);

    rhs_matrix_v.copy_from(mass_matrix_v);
    rhs_matrix_v.add(-(1.0 - theta), stiffness_matrix_v);

    rhs_matrix_v.vmult_add(system_rhs_v, solution_v);

    // Boundary conditions.
    {
        std::map<types::global_dof_index, double> boundary_values;
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;
        // Functions::ConstantFunction<dim> function_zero(0.0);

        boundary_functions[0] = &exact_solution_v;
        boundary_functions[1] = &exact_solution_v;
        boundary_functions[2] = &exact_solution_v;
        boundary_functions[3] = &exact_solution_v;

        VectorTools::interpolate_boundary_values(dof_handler_v,
                                                 boundary_functions,
                                                 boundary_values);

        MatrixTools::apply_boundary_values(
            boundary_values, lhs_matrix_v, solution_v, system_rhs_v, true);
    }
}

void Shallow_waters::solve_time_step(TrilinosWrappers::SparseMatrix &lhs_matrix,
                                     TrilinosWrappers::MPI::Vector &system_rhs,
                                     TrilinosWrappers::MPI::Vector &solution_owned,
                                     TrilinosWrappers::MPI::Vector &solution)
{
    SolverControl solver_control(3000, 1e-6 * system_rhs.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

    TrilinosWrappers::PreconditionSOR preconditioner;
    preconditioner.initialize(lhs_matrix, TrilinosWrappers::PreconditionSOR::AdditionalData(1.0));

    std::cout << "  Solving the linear system" << std::endl;
    solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
    std::cout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;

    solution = solution_owned;
}

void Shallow_waters::output(const unsigned int &time_step) const
{
    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler_h, solution_h, "h");
    data_out.add_data_vector(dof_handler_u, solution_u, "u");
    data_out.add_data_vector(dof_handler_v, solution_v, "v");

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
    assemble_mass_matrix(fe_h.get(), dof_handler_h, quadrature.get(), mass_matrix_h);
    assemble_mass_matrix(fe_u.get(), dof_handler_u, quadrature.get(), mass_matrix_u);
    assemble_mass_matrix(fe_v.get(), dof_handler_v, quadrature.get(), mass_matrix_v);

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

        exact_solution_v.set_time(time);
        VectorTools::interpolate(dof_handler_v, exact_solution_v, solution_owned_v);
        solution_v = solution_owned_v;

        // Output the initial solution.
        output(0);
        
        previous_solution_h = solution_h;
        previous_solution_u = solution_u;
        previous_solution_v = solution_v;

        time += deltat;

        exact_solution_h.set_time(time);
        VectorTools::interpolate(dof_handler_h, exact_solution_h, solution_owned_h);
        solution_h = solution_owned_h;

        exact_solution_u.set_time(time);
        VectorTools::interpolate(dof_handler_u, exact_solution_u, solution_owned_u);
        solution_u = solution_owned_u;

        exact_solution_v.set_time(time);
        VectorTools::interpolate(dof_handler_v, exact_solution_v, solution_owned_v);
        solution_v = solution_owned_v;

        output(1);
    }

    // Initialize previous solutions.
    // previous_solution_u = solution_u;
    // previous_solution_v = solution_v;

    pcout << theta << std::endl;

    unsigned int time_step = 1;

    while (time < T - 0.5 * deltat)
    {
        time += deltat;
        ++time_step;

        pcout << "===============================================" << std::endl;
        pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
              << time << std::endl;
        pcout << "-----------------------------------------------" << std::endl;

        exact_solution_h.set_time(time);
        // Solve for h.
        assemble_stiffness_and_rhs_h(time);
        previous_solution_h = solution_h;
        solve_time_step(lhs_matrix_h, system_rhs_h, solution_owned_h, solution_h);
        // Now solution_h contains h at timestep n+1, previous_solution_h contains h at timestep n

        exact_solution_u.set_time(time);
        // Solve for u.
        assemble_stiffness_and_rhs_u(time);
        previous_solution_u = solution_u;
        solve_time_step(lhs_matrix_u, system_rhs_u, solution_owned_u, solution_u);
        // Now solution_u contains u at timestep n+1, previous_solution_u contains u at timestep n
        // The next iteration (n=n+1) will therefore have solution_u at timestep n and previous_solution_u at timestep n-1

        exact_solution_v.set_time(time);
        // Solve for v.
        assemble_stiffness_and_rhs_v(time);
        previous_solution_v = solution_v;
        solve_time_step(lhs_matrix_v, system_rhs_v, solution_owned_v, solution_v);
        // Now solution_v contains v at timestep n+1, previous_solution_v contains v at timestep n
        // The next iteration (n=n+1) will therefore have solution_v at timestep n and previous_solution_v at timestep n-1

        output(time_step);
    }
}

double
Shallow_waters::compute_error(const VectorTools::NormType &norm_type)
{
    return 1.0;
}