# Overview
GASSPY (GPU Accelerated Spectra Synthesis in PYthon) is a package to post-process hydrodynamical simulations through the photoionization code Cloudy (https://gitlab.nublado.org/cloudy/cloudy/-/wikis/home) to get emission spectra from a cell by cell level. The code takes two steps:
- Database bulding: Takes the physical quantities such as density, temperature and radiation fluxes for each cell and runs a set of Cloudy models to cover them. We attempt to not run one model for each cell, but rather group the cells in phase space and run a single model per group. 
- Radiative transfer: Takes the database generated in the previous step and performs a raytracing and radiative transfer step to generate a set of rays with position and spectra information that can then be used as input data to emulate an observation.

NOTE: GASSPY is not complete and therefore not ready for usage. Its only shown here as a demonstration and should not be used for scientific studies as of writing.



# List of current and past contributors
    Loke Lönnblad Ohlin - Current - Primary developer
    Eric Pellegrini - Past 
    Thomas Peters - Past
    Daniel Rahner - Past

# Installation 
In directory above the repository install via pip using:

pip install gasspy
## Required packages
    "pybind11"
    "numpy"
    "matplotlib"
    "cupy"
    "torch"
    "pandas"
    "astropy"
    "mpi4py"
    "psutil"
    "h5py"

CuPy, pytorch, mpi4py and pybind11 might require special installs. Make sure that you follow their
own install documentation and don't rely on pip's automatic installer.

- Cupy: https://docs.cupy.dev/en/stable/install.html#faq
- pytorch : https://pytorch.org/
- mpi4py : https://mpi4py.readthedocs.io/en/stable/install.html
- pybind11 : https://pybind11.readthedocs.io/en/stable/installing.html


# Usage 
Example scripts of usage can be found in gasspy/scripts/ or in the Wiki or docs. All configuration is done using yaml files read at runtime. The main configuration (called gasspy_config in the code) must be provided. See GASSPY/gasspy/templates/gasspy_config_all.yaml for all the options with descriptions  
