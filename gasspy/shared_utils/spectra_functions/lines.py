import numpy as np
import cupy

from gasspy.shared_utils.spectra_functions.continuum import continuum_linear_interp


def integrated_line(Energies, deltaEnergies, fluxes, numlib = np, continuum_method = continuum_linear_interp):
    continuum = continuum_method(Energies, fluxes, numlib = numlib)
    return numlib.sum((fluxes - continuum)*deltaEnergies, axis  = 1).reshape((len(fluxes),1))
