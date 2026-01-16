
#include <filesystem>

#include "Shallow_waters.hpp"
#include "Case_Test.hpp"
#include "Case_Examples.hpp"

// Main function.
int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: ./shallow_waters T deltat mesh_file_name" << std::endl;
        return 1;
    }

    const std::string output_directory = "./vtk/";
    std::filesystem::create_directories(output_directory);

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    // Read command line arguments (T, deltat, mesh file name)
    const unsigned int degree_velocity = 2;
    const unsigned int degree_height = 1;
    const double T = std::stod(argv[1]);
    const double deltat = std::stod(argv[2]);
    const std::string mesh_file_name = argv[3];

    Shallow_waters<Problem_Case> problem(mesh_file_name, degree_velocity, degree_height, T, deltat, output_directory);

    problem.setup();
    problem.solve();

    return 0;
}