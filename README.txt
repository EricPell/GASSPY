README.txt - Readme file containing information on directory contents
as well as running the code(s)

contributors.txt - List of current and past contributors: 
    Eric Pellegrini - current - primary author
    Thomas Peters - current
    Daniel Rahner - current

defaults.py - sets default geometric and temperature mask. This is
overridden if specified in the local config file saved with the flash
simulation.

Step 1) myconfig-example.py - Copy myconfig-example.py to runtime direcotry and edit there.
Configure parameters.  Local
configuration file which overwrites defaults values in
"defaults.py". Also sets the name of the flash-plot-file to use as an
input by "get_flux.py"

Step 2) get_den_temp_flux.py - Extract density, temperature and 

Step 3) make-cloudy-input.py - create cloudy input files.

Step 4) runall-cloudy.pl - My homebrewee script to farm out CLOUDY
models to multiple CPUs/cores.

Step 5) combine-ems.pl - Combine all of the cloudy emssivity files into a
single table. This output is used in get_flux.py

FINAL NOTE: Files produced to reconstruct local emissivity of simulation:
      1. 'silcc-combined-ems.tbl'
      2. 'silcc.unique_parameters'
      3. 'SILCC_hdf5_plt_cnt_0403' # flash file

Example 1) get_em.py - Match Flash cell{x,y,z} with it's emssivities.

Example 2) get_flux.py - Do a simple projection along a cardinal axis to
create maps of line fluxes. This routine works by calculating the
total luminosity in a column within an evenly spaced grid of dx-dy,
and dividing by dx*dy. Output: unique physical parameters needed to
generate CLOUDY grid in "make-cloudy-input.py".

The following files are support files and are indirectly called by the
programs above. A user should usually not modify these data files.

silcc_flash_postprocess.ini - CLOUDY init file used to set global 
commands for post processing a SILCC simulation. This no longer sets nend=1

cont_shape.py - Replaces individual flxx_shape.py files:
    flge_shape.py - Shape of non-ionizing, photoelectric heating light. (units)
    fluv_shape.py - Shape of non-ionizing, UV light responsible for H2 destruction. (units)
    flih_shape.py - Shape of H-ionizing radiation from 13.6 to 15.2eV (units)
    fli2_shape.py - Shape of ionizing spectrum above 15.2eV (units)