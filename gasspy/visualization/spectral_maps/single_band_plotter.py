import numpy as np
import cupy 
import matplotlib
import matplotlib.pyplot as plt
import argparse
import astropy.units as u
import h5py as hp
import matplotlib.colors as mcol
from scipy.interpolate import interp1d

from gasspy.shared_utils.spec_reader import spec_reader
import gasspy.io.gasspy_io as gasspy_io
from mpl_toolkits.axes_grid1 import make_axes_locatable
"""
    DEFINE WHAT TO PLOT
"""
ap=argparse.ArgumentParser()

#---------------outputs-----------------------------
ap.add_argument('f')
ap.add_argument("--xlims", default=None, type = float, nargs = 2)
ap.add_argument("--ylims", default=None, type = float, nargs = 2)
ap.add_argument("--nx", default=None, type = int)
ap.add_argument("--ny", default=None, type = int)
ap.add_argument("--outdir",default="./")
ap.add_argument("--vlims", default = None, type = float, nargs = 2)
ap.add_argument("--colormap", default="viridis")
ap.add_argument("--idirs", default=None, nargs = "+", type = int)

args=ap.parse_args()



def create_map(h5group, gasspy_config, outmap_nx = None, outmap_ny = None, xlims = None, ylims = None, nbins = 1):
    xp = h5group["xp"][:]
    yp = h5group["yp"][:]
    ray_lrefine = h5group["ray_lrefine"][:]
    ray_fluxes = np.zeros((len(xp), nbins))
    for ibin in range(nbins):
        ray_fluxes[:,ibin] = 10**(h5group["photon_flux_%d"%ibin][:].astype(np.float64))
    
    # if we are reading the entire set of rays set limits to cover entire map
    if xlims is None:
        xlims = np.array([0, gasspy_config["detector_size_x"]])
    if ylims is None:
        ylims = np.array([0, gasspy_config["detector_size_y"]])
    outmap_size_x = xlims[1] - xlims[0]
    outmap_size_y = ylims[1] - ylims[0] 
            
    # If nx or ny are not defined we use the maximum of the providied rays
    outmap_ray_lrefine = np.max(ray_lrefine)
    if outmap_nx is None:
        outmap_nx = np.around(2**outmap_ray_lrefine  * outmap_size_x/gasspy_config["detector_size_x"]).astype(int)
    if outmap_ny is None:
        outmap_ny = np.around(2**outmap_ray_lrefine  * outmap_size_y/gasspy_config["detector_size_y"]).astype(int)
    print(outmap_nx, outmap_ny)
    # Size of each pixel 
    outmap_dx = outmap_size_x/outmap_nx
    outmap_dy = outmap_size_y/outmap_ny
    
    outmap = np.zeros((outmap_nx, outmap_ny, nbins))
    # Treat each ray refinement level seperatly
    for lref in range(np.min(ray_lrefine), np.max(ray_lrefine) +1):
        # Size of rays at this refinement level
        ray_dx = gasspy_config["detector_size_x"] / 2**float(lref)
        ray_dy = gasspy_config["detector_size_y"] / 2**float(lref)
        ray_nx = int(ray_dx/outmap_dx)
        if ray_dx/outmap_dx > ray_nx:
            ray_nx += 1
        ray_ny = int(ray_dy/outmap_dy)
        if ray_dy/outmap_dy > ray_ny:
            ray_ny += 1

        idxs_at_lref = np.where(ray_lrefine == lref)[0]
        if len(idxs_at_lref) == 0:
            print(idxs_at_lref.shape)
            continue
        nray_remaining = len(idxs_at_lref)
        iray = 0

        # get the fluxes for these rays and apply the window method
        flux = ray_fluxes[idxs_at_lref,:]

        # Determine where in the map the ray is
        map_xstart =  xp[idxs_at_lref] - 0.5*ray_dx - xlims[0]
        map_xend   =  xp[idxs_at_lref] + 0.5*ray_dx - xlims[0]
        map_ystart =  yp[idxs_at_lref] - 0.5*ray_dy - ylims[0]
        map_yend   =  yp[idxs_at_lref] + 0.5*ray_dy - ylims[0]           
        map_ixstart = np.floor(map_xstart/outmap_dx).astype(int)
        map_ixend   = np.ceil(map_xend/outmap_dx).astype(int)
        map_iystart = np.floor(map_ystart/outmap_dy).astype(int)
        map_iyend   = np.ceil(map_yend/outmap_dy).astype(int)
        ## Bounds
        map_ixstart[map_ixstart<0] = 0
        map_iystart[map_iystart<0] = 0
        map_ixend[map_ixend >= outmap_nx] = outmap_nx-1
        map_iyend[map_iyend >= outmap_ny] = outmap_ny-1

        # If rays are offset from the plot grid, and/or if we are plotting with odd number of pixels
        # the rays might be covering different number of pixels.
        # Figure out how many pixels are covered by each ray in both x and y
        nx_rays = map_ixend - map_ixstart + 1
        ny_rays = map_ixend - map_ixstart + 1

        # Loop over all combinations of nx and ny number of covered pixels
        for nx_now in range(np.min(nx_rays), np.max(nx_rays)+1):
            with_nx = nx_rays == nx_now
            for ny_now in range(np.min(ny_rays), np.max(ny_rays)+1):
                rays_now = np.where(with_nx*(ny_rays==ny_now))[0]
                if len(rays_now) == 0:
                    continue

                # Create a kernel with these number of pixels
                ixp , iyp = np.meshgrid(np.arange(0,nx_now), np.arange(0,ny_now), indexing = "ij")
                ixp = ixp.ravel()
                iyp = iyp.ravel()
                # Add the lowest pixel number to get position in the map
                ixps = (map_ixstart[rays_now][:,np.newaxis] + ixp[np.newaxis,:]).ravel()
                iyps = (map_iystart[rays_now][:,np.newaxis] + iyp[np.newaxis,:]).ravel()
                # Avoid pixels outside of the map
                in_map = np.where((ixps <outmap_nx) * (iyps < outmap_ny))[0]
                
                # The rays may or may not completely cover the pixels, so we need to figure out the covering area

                # Distance to lower pixel edge in x and y. Must be smaller than outmap pixel size and greater than 0
                fxl = np.maximum(np.minimum(((ixps+1)*outmap_dx-np.repeat(map_xstart[rays_now],nx_now*ny_now)), outmap_dx), 0)               
                fyl = np.maximum(np.minimum(((iyps+1)*outmap_dy-np.repeat(map_ystart[rays_now],nx_now*ny_now)), outmap_dy), 0)
                # Distance to upper pixel edge in x and y
                fxr = np.maximum(np.minimum((-(ixps )*outmap_dx+np.repeat(map_xend[rays_now]  ,nx_now*ny_now)), outmap_dx), 0)
                fyr = np.maximum(np.minimum((-(iyps )*outmap_dy+np.repeat(map_yend[rays_now]  ,nx_now*ny_now)), outmap_dy), 0)
                
                # The covering factor is determined by the smallest of these two 
                fx = np.minimum(fxl,fxr)
                fy = np.minimum(fyl,fyr)

                # Loop over bins and add to corresponding pixels, multiplied by the covering area to get total photon count/energy in the pixel
                for ibin in range(nbins):
                    np.add.at(outmap, (ixps[in_map], iyps[in_map], np.full(in_map.shape, ibin)),  (np.repeat(flux[rays_now, ibin], nx_now*ny_now)*fx*fy)[in_map])

    # Go back to surface brightness
    outmap = outmap/outmap_dx/outmap_dy 
    #print(np.min(outmap), np.max(outmap))
    return outmap    

def plot_map(fig, ax, data, cmap, vlims, label):
    norm =matplotlib.colors.LogNorm(vmin=vlims[0], vmax=vlims[1])
    P = ax.imshow(data.T, origin = "lower", cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "2.5%", pad = 0.03)
    cbar = fig.colorbar(P, cax = cax)
    cbar.set_label(r''+label)

# Figure out physical size of the plot
gasspy_config = gasspy_io.read_fluxdef("./gasspy_config.yaml")
x0 = 0
x1 = gasspy_config["detector_size_x"]
y0 = 0
y1 = gasspy_config["detector_size_y"]
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
cmap = args.colormap

h5file = hp.File(args.f)
ndirs  = h5file.attrs.get("ndirs")
nbands = h5file.attrs.get("nbands")
energy_limits = h5file["energy_limits"][:]
energy_widths = energy_limits[:,1] - energy_limits[:,0]
energy_center = energy_limits[:,0] + 0.5*energy_widths
line_data = None
if "line_data" in h5file:
    line_data = {}
    gasspy_io.read_dict_hdf5("line_data", line_data, h5file)
bband_data = None
if "bband_data" in h5file:
    bband_data = {}
    gasspy_io.read_dict_hdf5("bband_data", bband_data, h5file)

if args.idirs is None:
    idirs = range(ndirs)
else:
    idirs = args.idirs
    if not isinstance(idirs, list):
        idirs = [idirs]

for idir in idirs:
    h5group = h5file["dir_%05d"%idir]

    plotmap = create_map(h5group, gasspy_config, outmap_nx = args.nx, outmap_ny = args.ny, xlims = args.xlims, ylims = args.ylims, nbins = nbands)
   
    if bband_data is not None:
        for iband in bband_data["bband_index"]:
            vmax = h5file.attrs.get("max_flux_%d"%iband)
            vmin = vmax/1e4
            norm =matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            fig, axes = plt.subplots(nrows = 1, ncols = 1, sharex=True, sharey=True, figsize = (6,3))
            axes = [axes]
            P = axes[0].imshow(plotmap[:,:,iband].T, origin= "lower", extent = [x0, x1, y0, y1], norm = norm, cmap = args.colormap)

            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size = "2.5%", pad = 0.03)
            cbar = fig.colorbar(P, cax = cax)
            plt.savefig(args.outdir+"band_%d_flux_%05d"%(iband,idir), dpi = 300)
            plt.close(fig)           
    ##
    # Reduce all lines to the line and underlying continuum
    if line_data is not None:
        for line in line_data:
            # Figure out indexes of lines and continuum in band
            print(line)
            line_dict = line_data[line]
            print(line_dict)

            start_index = int(line_dict["start_index"])
            line_index = start_index + np.atleast_1d(np.array(line_dict["line_index"])).astype(int)
            cont_index = start_index + np.atleast_1d(np.array(line_dict["cont_index"])).astype(int)

            # grab their energies
            line_energy = energy_center[line_index]
            cont_energy = energy_center[cont_index]
            print(line_index)
            print(line_energy)
            print(energy_widths[line_index])
            print(cont_index)
            print(cont_energy)
            print(energy_widths[cont_index])

            # Grap appropriate arrays
            line_bins = plotmap[:,:,line_index]/energy_widths[line_index]
            cont_bins = np.log10(plotmap[:,:,cont_index]/energy_widths[cont_index])

            # Find the interpolation fit of the continuum
            cont_fit = interp1d(cont_energy, cont_bins, axis = 2, kind = "cubic")
            line_cont_flux = 10**cont_fit(line_energy)
            print(line_cont_flux.shape)
            line_flux = np.sum((line_bins - line_cont_flux)*energy_widths[line_index],axis =2)
            line_cont_flux = np.sum(line_cont_flux*energy_widths[line_index],axis =2)
            print(line_flux.shape)
            line_flux[line_flux<0] = np.min(line_flux[line_flux>0])
            ## Do plotting NOTE: CHANGE BEHAVIOUR HERE FOR YOUR NEEDS

            # Plot both for comparison
            # Take the vmax as the sum of both line energy bands
            vmax = 0
            for iline in line_index:
                print("max_flux_%d"%iline,  h5file.attrs.get("max_flux_%d"%iline))
                vmax += h5file.attrs.get("max_flux_%d"%iline)
            print(np.max(line_flux/vmax))
            vmin = vmax/1e4
            norm =matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            fig, axes = plt.subplots(nrows = 1, ncols = 2, sharex=True, sharey=True, figsize = (6,3))
            axes[0].imshow(line_flux.T, origin= "lower", extent = [x0, x1, y0, y1], norm = norm, cmap = args.colormap)
            axes[0].set_title(r"%s line flux"%line)
            P = axes[1].imshow(line_cont_flux.T, origin= "lower", extent = [x0, x1, y0, y1], norm = norm, cmap = args.colormap)
            axes[1].get_yaxis().set_visible(False)
            axes[1].set_title(r"%s cont flux"%line)

            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size = "2.5%", pad = 0.03)
            cbar = fig.colorbar(P, cax = cax)
            plt.savefig(args.outdir+"%s_flux_%05d"%(line,idir), dpi = 300)
            plt.close(fig)
            ##
        
    #np.save("%s/habands_frame_%05d.npy"%(args.outdir, idir), np.log10(plotmap).astype(np.float16))