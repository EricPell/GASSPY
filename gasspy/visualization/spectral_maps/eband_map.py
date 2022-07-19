import numpy as np
import cupy 
import matplotlib.pyplot as plt
import argparse
import astropy.units as u

from gasspy.shared_utils.spec_reader import spec_reader
from gasspy.shared_utils.spectra_functions import integrated_line, broadband
import gasspy.io.gasspy_io as gasspy_io
"""
    DEFINE WHAT TO PLOT
"""
ap=argparse.ArgumentParser()

#---------------outputs-----------------------------
ap.add_argument('f')
ap.add_argument("--Emin", nargs = "+",required=True, type=float)
ap.add_argument("--Emax", nargs = "+",required=True, type=float)
ap.add_argument("--lines", action="store_true")
ap.add_argument("--xlims", default=None, type = float, nargs = 2)
ap.add_argument("--ylims", default=None, type = float, nargs = 2)
ap.add_argument("--nx", default=None, type = int)
ap.add_argument("--ny", default=None, type = int)
ap.add_argument("--out",default=None)
ap.add_argument("--vlims", default = None, type = float, nargs = 2)
ap.add_argument("--colormaps", default=None, nargs = "+")
args=ap.parse_args()


assert len(args.Emin) == len(args.Emax), "Emin and Emax are required to have the same shape"

reader = spec_reader(args.f, maxmem_GB=None)


# Figure out physical size of the plot
gasspy_config = gasspy_io.read_fluxdef("./gasspy_config.yaml")
x0 = 0
x1 = reader.observer_size_x
y0 = 0
y1 = reader.observer_size_y
x1 *= gasspy_config["sim_unit_length"]/((1*u.pc).cgs.value)
y1 *= gasspy_config["sim_unit_length"]/((1*u.pc).cgs.value)

if args.xlims is not None:
    xplot = [args.xlims[0]*(x1-x0) + x0, args.xlims[1]*(x1-x0) + x0]
else:
    xplot = [x0, x1]
if args.ylims is not None:
    yplot = [args.ylims[0]*(y1-y0) + y0, args.ylims[1]*(y1-y0) + y0]
else:
    yplot = [y0, y1]

if args.lines:
    window_method = integrated_line
else:
    window_method = broadband

cmaps = [None for i in range(len(args.Emin))]
if args.colormaps is not None:
    for i in range(len(args.colormaps)):
        cmaps[i] = args.colormaps[i]

nfields = 1
logscale = [True, False, False]
fscale = 5
for i in range(len(args.Emin)):
    map = reader.create_map(Elims=np.array([args.Emin[i], args.Emax[i]]), window_method = window_method, outmap_nfields= nfields, 
                                            outmap_nx = args.nx, outmap_ny = args.ny, xlims = args.xlims, ylims = args.ylims)
    

    fig, axes = plt.subplots(figsize=(fscale*nfields,fscale), nrows = 1, ncols = nfields, sharex = True, sharey =True)
    if nfields == 1:
        axes = [axes,]
    for iax, ax in enumerate(axes):
        if logscale[iax]:
            field = map[:,:,iax].T
            field[field < 1e-50] = 1e-50
            field = np.log10(field)
            if args.vlims is None:
                fmax = np.max(field)
                vmin = fmax - 4
                vmax = fmax + 0.5
            else:
                vmin = args.vlims[0]
                vmax = args.vlims[1]
        else:
            field = map[:,:,iax].T
            vmin = np.min(field)
            vmax = np.max(field)
        print(vmin, vmax)
        ax.imshow(field, vmin = vmin, vmax = vmax, origin = "lower", extent=[xplot[0],xplot[1],xplot[0],xplot[1]], cmap = plt.get_cmap(cmaps[i]))
        ax.set_xlim(xplot)
        ax.set_ylim(xplot)
        ax.set_xlabel(r"$x$ [pc]")
        ax.set_xlabel(r"$y$ [pc]")
    if args.out is not None:
        plt.savefig(args.out, dpi = 600)
        plt.close(fig)
    else:
        plt.show()
