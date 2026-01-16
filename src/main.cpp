#include "Shallow_waters.hpp"
#include "Problem_Specs.hpp"

// Main function.
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    // Read command line arguments (T, deltat, mesh file name)
    AssertThrow(argc == 4, ExcMessage("Usage: ./shallow_waters T deltat mesh_file_name"));

    const unsigned int degree_velocity = 2;
    const unsigned int degree_height = 1;
    const double T = std::stod(argv[1]);
    const double deltat = std::stod(argv[2]);
    const std::string mesh_file_name = argv[3];

    Shallow_waters<Problem_Specs> problem(mesh_file_name, degree_velocity, degree_height, T, deltat);

    problem.setup();
    problem.solve();

    return 0;
}