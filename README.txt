README.txt - Readme file containing information on directory contents
as well as running the code(s)

contributors.txt - List of current and past contributors

defaults.py - sets default geometric and temperature mask. This is
overridden if specified in the local config file saved with the flash
simulation.

Step 1) myconfig-example.py - Configure parameters.  Local
configuration file which overwrites defaults values in
"defaults.py". Also sets the name of the flash-plot-file to use as an
input by "get_flux.py"

Step 2) get_den_temp_flux.py - Extract density, temperature and 

Step 3) make-cloudy-input.py - create cloudy input files.

Step 4) runall-cloudy.pl - My homebrewee script to farm out CLOUDY
models to multiple CPUs/cores.

Step 5) get_em.py - Match Flash cell{x,y,z} with it's emssivities.

Step 6) get_flux.py - Do a simple projection along a cardinal axis to
create maps of line fluxes. This routine works by calculating the
total luminosity in a column within an evenly spaced grid of dx-dy,
and dividing by dx*dy. Output: unique physical parameters needed to
generate CLOUDY grid in "make-cloudy-input.py".

The following files are support files and are indirectly called by the
programs above.

silcc_flash_postprocess_singlezone.ini

flge_shape.py

fli2_shape.py

flih_shape.py

fluv_shape.py
