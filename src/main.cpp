#include "Shallow_waters.hpp"

// Main function.
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const unsigned int degree_velocity = 1;
    const unsigned int degree_height = 1;
    const double T = 3.0;
    const double deltat = 3.0e-1;

    Shallow_waters problem("../mesh/mesh-square-10.msh", degree_velocity, degree_height, T, deltat);

    problem.setup();
    problem.solve();

    return 0;
}