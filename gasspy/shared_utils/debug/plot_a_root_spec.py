#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from gasspy.shared_utils.spec_reader import spec_reader

ap=argparse.ArgumentParser()

#---------------outputs-----------------------------
ap.add_argument('f')
ap.add_argument("--Emin", default = None, type=float)
ap.add_argument("--Emax", default = None, type=float)
ap.add_argument("--xp", required = True, nargs = "+", type = float)
ap.add_argument("--yp", required = True, nargs = "+", type = float)

args=ap.parse_args()

assert len(args.xp) == len(args.yp), "xp and yp are required to have the same shape"

reader = spec_reader(args.f, maxmem_GB=None)
if args.Emin is None:
    Emin = np.min(reader.Energies)
else:
    Emin = args.Emin

if args.Emax is None:
    Emax = np.max(reader.Energies)
else:
    Emax = args.Emax

Elims = np.array([Emin, Emax])

fig = plt.figure()
for i in range(len(args.xp)):
    xp = args.xp[i]
    yp = args.yp[i]
    Eplot, flux = reader.read_spec(xp, yp, Elims = Elims)
    plt.plot(Eplot, flux, label = "xp = %.4e, yp=%.4e"%(xp,yp))
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()
