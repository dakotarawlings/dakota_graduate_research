# Molecular Simulations

## Project Overview
* Wrote a moleclar dynamics simulation program to understand the effects of charge density, polymer chain stiffness, and polymer length on ion transport in ionic liquid functionalized conjugated polymers
* Wrote Fortran code for completing computationally expesive steps (namely computing total energies and forces on each atom) and compiled code into Python modules using f2py
* See [this document](https://sites.engineering.ucsb.edu/~shell/che210d/gallery2019/DHanemannRawlings_report.pdf) for more information on my simulation work
* This work inspired the simulation work in my recent [first author publication](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.1c01811)

## References
**Python Version:** 3.7

**Packages:** numpy, scipy, f2py, matplotlib

**Reference for MD simulations in python:** https://sites.engineering.ucsb.edu/~shell/che210d/assignments.html

**This work ultamately inspired the work in my recent publication:** https://pubs.acs.org/doi/full/10.1021/acs.chemmater.1c01811

## Molecular Dynamics Model
1. Molecular dynamics simulations model the behavior of a system of atoms using a classical approach, in which the position and dynamics of atoms are approximated using numerical approximations to newtons equations of motion. 
2. Simulations are initiated with initial positions and velocities for each atom in the system. The forces on each atom are calculated using an approximate potential function. 
3. The system is then evolved using a descrete-time numerical approximation to Newton's equations of motion. Here, we use the VelocityVerlet algorithm which uses the current velocity and position to estimate the  position at the next discrete time interval. The updated positions can then be used to calculate updated forces using our approximate potential function. These forces are then used to calculate updated velocities.
4. The calculation of forces, velocities, and positions for each atom in the system is performed in Fortan for efficiency. The Fortran code was writen and compiled as a python module using f2py. 
5. The total potential energy of the system was approximated with the sum of a non-bonding Lennard-Jones potential, a non-bonding screened coulombic potential, a bonding harmonic potenital to model the behavior of harmonic bonds between adjacent atoms, and a bond angle potential to model the polymer chain stiffness. 
6. The diffusion coefficient of charges in the system was calculated using the slope of the mean squared displacement of all atoms of interest in the system from their initial position with time
7. The radius of gyration of the polymer (a measure of the polymer chain stiffness) was calculated by averaging the instantanious radius of gyration for all polymers in the system and then averaging this value over the production period


