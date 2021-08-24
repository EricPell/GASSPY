# GASSPY Definitions

## Simulation Input Data
### Density fields:
    - dens: log number density of H in cm-3
    - rho : log mass density in g/cm-3
    - temp: log of T in K
    - dx  : depth of cell in log cm

### Ionization Fronts:
    - ForceFullDepth: Force the code to calculate the full cell, no matter what.
    - TODO: define how to select
### Photon Flux definitions:
    Dictionary containing the name of the radiation field as a key, with a subdictionary containing the following information is required:
    - Emin: minimum bin energy in eV
    - Emax: maximum bin energy in eV
    - shape: currently supported is a Cloudy compatible sed file, with energy or wave units specified, and Fnu or nuFnu
    Shape files may also be very simple flat bins, but still need to be defined in the file format.
    Shape is the name of the sed for reading.
    - 'FUV':{'Emax': 13.59844, 'Emin': 0.1, 'shape': 'specFUV.sed'}
    - 'HII':{'Emax': 24.58741, 'Emin': 13.59844, 'shape': 'specHII.sed'}
    - 'HeII':{'Emax': 54.41778, 'Emin': 24.58741, 'shape': 'specHeII.sed'}
    - 'HeIII':{'Emax': 100.0, 'Emin': 54.41778, 'shape': 'specHeIII.sed'}

## GASSPY_subphysics object defition
    - "NULL"
    -- 0-th entry in subphysics model
    -- Coresponds to a zero-int numerical value for easy zeroing of all sim-data and derived fields
    -- Emisivity and opacity entries will automatically be generated with the correct number of spectal bins set to zero.