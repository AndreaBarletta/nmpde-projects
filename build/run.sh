#!/bin/bash
rm -rf vtk/*
mpirun -np 1 shallow_waters > output_1.txt
mpirun -np 2 shallow_waters > output_2.txt
mpirun -np 4 shallow_waters > output_4.txt

