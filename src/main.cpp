#include "Shallow_waters.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // const unsigned int degree = 1;

  const unsigned int degree_velocity = 1;
  const unsigned int degree_height = 1;
  const double T      = 1.0;
  const double deltat = 0.05;
  const double theta  = 0.5;

  Shallow_waters problem(degree_velocity, degree_height, T, deltat, theta);

  problem.setup();
  // problem.solve();

  return 0;
}