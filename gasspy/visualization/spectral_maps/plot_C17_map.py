import matplotlib.pyplot as plt
import numpy as np
import argparse

from gasspy.physics.sourcefunction_database.cloudy import select_cloudy_lines
from gasspy.shared_utils.spectra_functions import integrated_line, broadband
from gasspy.shared_utils.spec_reader import spec_reader


"""
    DEFINE WHAT TO PLOT
"""
ap=argparse.ArgumentParser()

#---------------input spectral file-----------------------------
ap.add_argument('f')
#---------------Dimensions and limits of the map----------------
ap.add_argument("--xlims", default=None)
ap.add_argument("--ylims", default=None)
ap.add_argument("--nx", default=None)
ap.add_argument("--ny", default=None)
#---------------Where to put the plots--------------------------
ap.add_argument("--outpath", default = "./")
args=ap.parse_args()


# constants

Ryd_Ang = 911.2266
# unpack all the lines
names = []
window_method = []
emins = []
emaxs = []

labels = select_cloudy_lines.labels()
label_list = [[label, label.split(" ")[-1]] for label in list(labels.line.keys())]
for i, label in enumerate(label_list):
    names.append(label[0])
    lam = label[1]
    if lam.endswith("A"):
        E0 = Ryd_Ang / float(lam.strip("A"))
    elif lam.endswith("m"):
        E0 = Ryd_Ang / (float(lam.strip("m"))*1e4)

    # For now add 5% buffer. In future should be an actual line finder
    emins.append(E0*0.95)
    emaxs.append(E0*1.05)
    window_method.append(integrated_line)

# initialize the reader 
reader = spec_reader(args.f, maxmem_GB=None)

# TODO: make this one call to load it all...
fscale = 5
for i in range(len(names)):
    map = reader.create_map(Elims=np.array([emins[i], emaxs[i]]), window_method = window_method[i], outmap_nfields= 1, 
                                            outmap_nx = args.nx, outmap_ny = args.ny, xlims = args.xlims, ylims = args.ylims)
    

    fig, axes = plt.subplots(figsize=(fscale*1.2,fscale), nrows = 1, ncols = 1, sharex = True, sharey =True)
    axes = [axes,]
    for iax, ax in enumerate(axes):
        
        field = map[:,:,iax].T
        field[field < 1e-50] = 1e-50
        field = np.log10(field)
        fmax = np.max(field)
        vmin = fmax - 4
        vmax = fmax 
        ax.imshow(field, vmin = vmin, vmax = vmax, origin = "lower", extent=[0,1,0,1])
        ax.set_title(names[i])
    plt.savefig(args.outpath + "/" + names[i].replace(" ","_") + ".png")
    plt.close(fig)

