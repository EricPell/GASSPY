import numpy as np
import cupy

from gasspy.shared_utils.spectra_functions.continuum import continuum_linear_interp 

def broadband(Energies, deltaEnergies, fluxes, numlib = np):
    return numlib.sum(fluxes*deltaEnergies, axis = 1).reshape((len(fluxes),1))

def continuum_removed_broadband(Energies, deltaEnergies, fluxes, numlib = np, continuum_method = continuum_linear_interp):
    """
        Currently identical to integrated_line
    """
    continuum = continuum_method(Energies, fluxes, numlib=numlib)
    return numlib.sum((fluxes - continuum)*deltaEnergies, axis  = 1).reshape((len(fluxes),1))
