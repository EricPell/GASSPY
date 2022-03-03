import numpy as np
import cupy


def continuum_linear_interp(Energies, fluxes, idx0 = 0, idx1 = -1, numlib = np):
    """
    """
    f0 = fluxes[:,idx0]
    f1 = fluxes[:,idx1]
    E0 = Energies[idx0]
    E1 = Energies[idx1]
    delta = (f1 - f0) / (E1 - E0)
    return f0[:,numlib.newaxis] + delta[:,numlib.newaxis] * (Energies[numlib.newaxis, :] - E0)