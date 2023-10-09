#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import astropy.units as u
from gasspy.shared_utils.spec_reader import spec_reader
from gasspy.io import gasspy_io
ap=argparse.ArgumentParser()

#---------------outputs-----------------------------
ap.add_argument('f')
ap.add_argument("--Emin", default = None, type=float)
ap.add_argument("--Emax", default = None, type=float)
ap.add_argument("--xp", required = True, nargs = "+", type = float)
ap.add_argument("--yp", required = True, nargs = "+", type = float)
ap.add_argument("--colors", nargs = "+", default = None)
ap.add_argument("--ylims", nargs = 2, default = None, type = float)
ap.add_argument("--outpath", default = None)
ap.add_argument("--no_legend", action = "store_true")
ap.add_argument("--pos_in_code_units", action = "store_true")
ap.add_argument("--marker", default= None, type = str)
args=ap.parse_args()

assert len(args.xp) == len(args.yp), "xp and yp are required to have the same shape"

reader = spec_reader(args.f, maxmem_GB=None)

# Figure out physical size of the plot
gasspy_config = gasspy_io.read_fluxdef("./gasspy_config.yaml")

xsize = reader.observer_size_x * gasspy_config["sim_unit_length"]/((1*u.pc).cgs.value)
ysize = reader.observer_size_y * gasspy_config["sim_unit_length"]/((1*u.pc).cgs.value)


if args.Emin is None:
    Emin = np.min(reader.Energies)
else:
    Emin = args.Emin

if args.Emax is None:
    Emax = np.max(reader.Energies)
else:
    Emax = args.Emax

energy_limits = np.array([Emin, Emax])

colors = [None for i in range(len(args.xp))]
if args.colors is not None:
    for i, col in enumerate(args.colors):
        colors[i] = args.colors[i]


fig = plt.figure(figsize = (5,4))
for i in range(len(args.xp)):
    if args.pos_in_code_units:
        xp = args.xp[i]
        yp = args.yp[i]
    else:
        xp = args.xp[i]/xsize
        yp = args.yp[i]/ysize
    Eplot, flux, line, bband= reader.read_spec(xp, yp, energy_limits = energy_limits, return_integrated_line = True, return_broadband = True)
    np.save("spec_%f_%f.npy"%(args.xp[i], args.yp[i]),np.array([Eplot,flux]) )
    plt.plot(Eplot, flux, label = "xp = %.3e, yp=%.3e"%(xp,yp), color = colors[i], marker = args.marker)
    print("xp = %.4e, yp = %.4e: line = %.4e bband = %.4e"%(xp, yp, line, bband))
if not args.no_legend:
    plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$\nu F_\nu$ [code units]")
plt.xlim(Emin, Emax)
plt.ylim(args.ylims)
plt.xlabel(r"$E_\gamma$ [Ryd]")
if args.outpath is not None:
    plt.savefig(args.outpath)
    plt.close(fig)
else:
    plt.show()
