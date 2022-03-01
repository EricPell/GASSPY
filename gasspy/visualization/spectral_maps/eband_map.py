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
ap.add_argument("--xlims", default=None)
ap.add_argument("--ylims", default=None)
ap.add_argument("--nx", default=None)
ap.add_argument("--ny", default=None)

args=ap.parse_args()


assert len(args.Emin) == len(args.Emax), "Emin and Emax are required to have the same shape"

reader = spec_reader(args.f, maxmem_GB=None)

if args.lines:
    window_method = integrated_line
else:
    window_method = broadband


for i in range(len(args.Emin)):
    map = reader.create_map(Elims=np.array([args.Emin[i], args.Emax[i]]), window_method = window_method, outmap_nfields= 1, 
                                            outmap_nx = args.nx, outmap_ny = args.ny, xlims = args.xlims, ylims = args.ylims)
    
    map = np.log10(map[:,:,0]).T
    maxf = np.max(map)
    plt.imshow(map, vmin = maxf - 4, vmax = maxf + 0.5)
    plt.show()