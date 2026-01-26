# Finite Element Solver for the Shallow Water Equations (deal.II)

This repository contains a finite element solver for
the **Shallow Water Equations (SWE)** implemented using the **deal.II**
finite element library.\
The software is designed with a dual focus on **numerical robustness**
and **parallel scalability**.

This is part of a project-work for **Numerical Methods For Partial Differential Equations** course [@Polimi](https://www.polimi.it/)

------------------------------------------------------------------------

## Authors

Project developed by:
- [Andrea Barletta](https://github.com/AndreaBarletta)
- [Enrico Tirri](https://github.com/EnricoTirri)

------------------------------------------------------------------------

## Overview

The shallow water equations model fluid flows where the horizontal
length scales dominate the vertical depth. They are widely used in
oceanography, hydraulics, geophysics, and atmospheric sciences.

------------------------------------------------------------------------

## Repository Structure

    .
    ├── docs/               # Project documentation
    ├── gifs/               # Showcase gifs
    ├── mesh/               # Mesh files
    ├── scripts/            # Scripts for:
    │                           + Plot data
    │                           + Generate manufactured solution
    │                           + Run tests
    ├── src/                # C++ source code of the solver
    ├── CMakeLists.txt      # Build configuration
    └── README.md

------------------------------------------------------------------------

## Parallelization

The solver is implemented using deal.II's distributed parallel
infrastructure:

-   `parallel::distributed::Triangulation`
-   MPI communication
-   Trilinos linear algebra backends

------------------------------------------------------------------------

## Dependencies

-   deal.II (compiled with MPI support)
-   CMake ≥ 3.16
-   MPI (OpenMPI or MPICH)
-   C++17 compatible compiler
-   Python 3 with `numpy` and `matplotlib` (optional)

------------------------------------------------------------------------

## Building the Code

From the project root:

``` bash
mkdir build
cd build
cmake ..
make -j
```

------------------------------------------------------------------------

## Running the Solver

Run in serial:

``` bash
./shallow_waters <T> <deltat> <mesh_file_name>
```

Run in parallel:

``` bash
mpirun -np 4 ./shallow_waters <T> <deltat> <mesh_file_name>
```

------------------------------------------------------------------------

## Manufactured Solutions

The script `manufacture.py` implements a **method of manufactured
solutions (MMS)** workflow:

-   An analytic solution is defined
-   Corresponding forcing terms are computed

Run with:

``` bash
python manufacture.py
```

------------------------------------------------------------------------

## Mesh Generation with Gmsh
The repository includes a geometry definition file mesh/mesh_script.geo that can be used to generate computational meshes using [GMSH](https://gmsh.info/)

------------------------------------------------------------------------

## Example Outputs

Below is an example of simulation result.

### Gaussian offset bump

![Free surface](gifs/gauss_bump.gif)
