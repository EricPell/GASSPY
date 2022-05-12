import numpy as np
import cupy

from gasspy.shared_utils.spectra_functions.continuum import continuum_linear_interp


def integrated_line(Energies, deltaEnergies, fluxes, numlib = np, Eaxis = 1, continuum_method = continuum_linear_interp):
    continuum = continuum_method(Energies, fluxes, numlib = numlib, Eaxis = Eaxis)
    return numlib.sum((fluxes - continuum)*deltaEnergies, axis  = Eaxis).reshape( numlib.take(fluxes,0, axis = Eaxis).shape + (1,))
