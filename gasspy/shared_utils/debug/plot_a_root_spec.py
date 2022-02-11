#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

if len(sys.argv) > 1:
    i = int(sys.argv[1])
else:
    os.chdir(os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/GASSPY/spec")
    i = 136149
x = np.load("windowed_energy.npy")
y = np.load("spec_%i.npy"%i)

from scipy import ndimage, interpolate

f = interpolate.interp1d(x, y, axis = 0)

#newx = np.logspace(np.log10(np.min(x[1:-1])), np.log10(np.max(x[1:-1])), 140000)
#newy = f(newx)
#by = ndimage.gaussian_filter(newy, (5,0))


plt.plot(x, y)
plt.xscale("log")
plt.yscale("log")
plt.show()
