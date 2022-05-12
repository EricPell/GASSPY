import numpy as np
import cupy


def continuum_linear_interp(Energies, fluxes, idx0 = 0, idx1 = -1, numlib = np, Eaxis = 1):
    """
    """
    f0 = numlib.expand_dims(fluxes.take(idx0, Eaxis),Eaxis)
    f1 = numlib.expand_dims(fluxes.take(idx1, Eaxis),Eaxis)
    E0 = Energies[idx0]
    E1 = Energies[idx1]
    delta = (f1 - f0) / (E1 - E0)

    Eshape = ()
    for iax in range(0, len(f0.shape)):
        if iax == Eaxis:
            Eshape = Eshape + (len(Energies),)
        else:
            Eshape = Eshape + (1,)
    return f0 + delta * (Energies - E0).reshape(Eshape)