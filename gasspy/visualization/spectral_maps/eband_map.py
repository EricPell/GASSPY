import numpy as np
import cupy 
import matplotlib.pyplot as plt
import argparse

from gasspy.shared_utils.spec_reader import spec_reader
from gasspy.shared_utils.spectra_functions import integrated_line, broadband

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
ap.add_argument("--nx", default=None)
ap.add_argument("--ny", default=None)
ap.add_argument("--out",default=None)

args=ap.parse_args()


assert len(args.Emin) == len(args.Emax), "Emin and Emax are required to have the same shape"

reader = spec_reader(args.f, maxmem_GB=None)

if args.lines:
    window_method = integrated_line
else:
    window_method = broadband
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
            fmax = np.max(field)
            vmin = fmax - 4
            vmax = fmax + 0.5
            print(fmax)
        else:
            field = map[:,:,iax].T
            vmin = np.min(field)
            vmax = np.max(field)
        ax.imshow(field, vmin = vmin, vmax = vmax, origin = "lower", extent=[0,1,0,1])

    if args.out is not None:
        plt.savefig(args.out, dpi = 600)
    plt.show()
