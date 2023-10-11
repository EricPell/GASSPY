import argparse 
import numpy as np
import h5py as hp
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from scipy import stats
from gasspy.io.gasspy_io import read_yaml, read_dict_hdf5
from gasspy.physics.databasing.database_utils import get_neighbor_idxs
from numba import jit


@jit(nopython = True)
def interpolate_kernel(grid_points, grid_values, cell_points):
    cell_values = np.zeros(cell_points.shape[0])
    for icell in range(cell_values.shape[0]):
        neigh_points = grid_points[icell,:,:]
        neigh_values = grid_values[icell,:]
        cell_point = cell_points[icell,:]

        # Deltas
        deltas = np.zeros(neigh_points.shape[1])
        for idim in range(neigh_points.shape[1]):
            if cell_point[idim] >= 0:
                deltas[idim] = np.max(neigh_points[:,idim])
            else:
                deltas[idim] = np.min(neigh_points[:,idim])
        cell_values[icell] = 0 
        for idir in range(neigh_points.shape[0]):
            coeff = neigh_values[idir]
            not_aligned = False
            for idim in range(neigh_points.shape[1]):
                neigh_coord = neigh_points[idir, idim]

                # If this neighbor is in the wrong hyper quadrant, skip
                if not (np.sign(neigh_coord) == np.sign(cell_point[idim]) or neigh_coord == 0): 
                    not_aligned = True
                    break

    
                if abs(neigh_coord) == 0:
                    coeff = coeff*(deltas[idim] - cell_point[idim])/deltas[idim]
                else:
                    coeff = coeff*cell_point[idim]/deltas[idim]
                

            if not not_aligned:
                cell_values[icell] += coeff
    return cell_values 

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcol.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap
def plot_deltas_error(error, bin_edges, prefix = "", cmap = "binary"):
    fig, axes = plt.subplots(3,2, figsize = (12,6))
    axes = axes.ravel()
    axes[1].set_title(line)

    data = error.copy()
    data[data<bin_edges[0]] = (bin_edges[1] + bin_edges[0])*0.5
    data[data>bin_edges[-1]] = (bin_edges[-1] + bin_edges[-2])*0.5
    
    vmax = -1e99
    for ivar, var_label in enumerate(var_labels[1:]):
        hist, x_edge, y_edge, tmp = stats.binned_statistic_2d(np.abs(cell_models[:,ivar+1] - grid_models[:,ivar+1]), data, 1, statistic="count", bins= [delta_bins[ivar+1], bin_edges]) 
        vmax = max(vmax, np.max(hist)/np.sum(hist))

    norm = mcol.LogNorm(vmin = vmax/1e2, vmax = vmax)

    for ivar, var_label in enumerate(var_labels[1:]):
        ax = axes[ivar]
        hist, x_edge, y_edge, tmp = stats.binned_statistic_2d(np.abs(cell_models[:,ivar+1] - grid_models[:,ivar+1]), data, 1, statistic="count",bins= [delta_bins[ivar+1], bin_edges]) 
        xs, ys = np.meshgrid(x_edge,y_edge, indexing = "ij")

        im = ax.pcolormesh(xs,ys, hist/np.sum(hist), norm = norm, cmap = cmap)

        ax.set_xlabel(r"log($\Delta$" + var_label+")")
        ax.set_ylabel(r"$\Delta j/j_\mathrm{cell}$")
        ax.set_yscale("log")
        ax.set_ylim(1e-4,5e2)
        if ivar%2 == 1:
            cbar = fig.colorbar(im, ax = ax)
        if ivar == 3:
            cbar.set_label("Cell fraction")
    plt.subplots_adjust(hspace = 1)
    if args.plotdir is None:
        plt.show()
    else:
        plt.savefig(args.plotdir +"/" + prefix + "delta_error_" + line.replace(" ", "_").replace("__","_")+ ".png")
        plt.close()

def plot_var_error(error, bin_edges, prefix = "", cmap = "binary"):
    fig, axes = plt.subplots(3,2, figsize = (12,6))
    axes = axes.ravel()
    axes[1].set_title(line)
    norm = mcol.LogNorm(vmin = 1e-4, vmax = 1e1)

    data = error.copy()
    data[data<bin_edges[0]] = (bin_edges[1] + bin_edges[0])*0.5
    data[data>bin_edges[-1]] = (bin_edges[-1] + bin_edges[-2])*0.5
    vmax = -1e99
    for ivar, var_label in enumerate(var_labels[1:]):
        hist, x_edge, y_edge, tmp = stats.binned_statistic_2d(10**cell_models[:,ivar+1], data, 1, statistic="count",bins= [var_bins[ivar+1], bin_edges]) 
        vmax = max(vmax, np.max(hist)/np.sum(hist))

    norm = mcol.LogNorm(vmin = vmax/1e2, vmax = vmax*10)

    for ivar, var_label in enumerate(var_labels[1:]):
        ax = axes[ivar]
        hist, x_edge, y_edge, tmp = stats.binned_statistic_2d(10**cell_models[:,ivar+1] , data, 1, statistic="count",bins= [var_bins[ivar+1], bin_edges]) 
        xs, ys = np.meshgrid(x_edge,y_edge, indexing = "ij")

        im = ax.pcolormesh(xs,ys, hist/np.sum(hist), norm = norm, cmap = cmap)

        ax.set_xlabel(r""+ var_label)
        ax.set_ylabel(r"$\Delta j/j_\mathrm{cell}$")
        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_ylim(1e-4,5e2)
        if ivar%2 == 1:
            cbar = fig.colorbar(im, ax = ax)
        if ivar == 3:
            cbar.set_label("Cell Fraction")

    plt.subplots_adjust(hspace = 1)
    if args.plotdir is None:
        plt.show()
    else:
        plt.savefig(args.plotdir +"/" + prefix +"phase_space_error_" + line.replace(" ", "_").replace("__","_")+ ".png")
        plt.close()


def plot_var_var_error(ivar0, ivar1, error, save_name, cmap = "binary"):
    fig, axes = plt.subplots()
    axes.set_title(line)
    norm = mcol.LogNorm(vmin = 1e-4, vmax=5e1)
    data = error.copy()
    data[data<1e-40] = 1e-40
    error_map, x_edge, y_edge, binnumber = stats.binned_statistic_2d(10**cell_models[:,ivar0], 10**cell_models[:,ivar1], data, "max", bins = [var_bins[ivar0], var_bins[ivar1]])
    xs, ys = np.meshgrid(x_edge,y_edge, indexing = "ij")

    im = axes.pcolormesh(xs,ys, error_map, norm = norm, cmap = cmap)
    cbar = fig.colorbar(im, ax = axes)
    cbar.set_label(r"Maximum $\Delta J/J$")
    axes.set_xlim(var_bins[ivar0][0], var_bins[ivar0][-1])
    axes.set_ylim(var_bins[ivar1][0], var_bins[ivar1][-1])
    axes.set_xlabel(var_labels[ivar0])
    axes.set_ylabel(var_labels[ivar1])
    axes.set_xscale("log")
    axes.set_yscale("log")
    
    if args.plotdir is None:
        plt.show()
    else:
        plt.savefig(args.plotdir +"/" + save_name)
        plt.close()    
    
    


def get_error(true_vals, apprx_vals, minval = 1e-40):
    error = np.abs(true_vals - apprx_vals)
    relative_error = error/np.maximum(true_vals, minval)

    return error, relative_error

    
def get_error_emission_size(error, min_sensitivity):
    size = np.full(error.shape, 1e99)    
    size[error > 0] = 3*min_sensitivity*4.25451703e+10/error[error > 0]
    return size

ap = argparse.ArgumentParser()
ap.add_argument("cell_database")
ap.add_argument("grid_database")
ap.add_argument("--gasspy_config", default= "./gasspy_config")
ap.add_argument("--min_cont_frac", type = float, default=None)
ap.add_argument("--min_sensitivity", type = float, default= None)
ap.add_argument("--plotdir", default= None)
args = ap.parse_args()


if args.plotdir is not None:
    if not os.path.exists(args.plotdir):
        os.makedirs(args.plotdir)


cell_database = hp.File(args.cell_database, "r")
grid_database = hp.File(args.grid_database, "r")

# Current config
gasspy_config = read_yaml(args.gasspy_config)

# Databases config
grid_database_config = {}
read_dict_hdf5("gasspy_config", grid_database_config, grid_database)

database_fields = []
for field in grid_database["database_fields"][:]:
    database_fields.append(field.decode())


# Determine which lines we want to look at
line_labels = []
for line in grid_database["line_labels"][:]:
    line_labels.append(line.decode())
gasspy_config = read_yaml(args.gasspy_config)


if "line_labels" in gasspy_config:
    line_list = gasspy_config["line_labels"]
else:
    line_list = []
    for line in line_labels:
        line_list.append(line.decode())


# load up compression ratio
compression_ratio = grid_database_config["compression_ratio"]
max_delta = np.zeros(len(compression_ratio))
for ivar, var in enumerate(database_fields):
    comp = compression_ratio[var]
    max_delta[ivar] = comp[1]*10**-comp[0] 



# Determine which variables we want to interpolate in
if "interpolate_fields" in gasspy_config:
    interpolate = True
    interpolate_fields = gasspy_config["interpolate_fields"]
    interp_field_idxs = []
    for field in interpolate_fields:
        interp_field_idxs.append(database_fields.index(field))

    grid_interpolate_fields = []
    for field in grid_database_config["interpolate_fields"]:
        grid_interpolate_fields.append(field.decode())
    neighbor_idxs = get_neighbor_idxs(interpolate_fields, grid_interpolate_fields)


else:
    interpolate = False

success = np.where(cell_database["model_successful"][:] == 1)[0]
print(len(success))
# gasspy ids specifically wanted by the cells
cell_gasspy_ids = cell_database["cell_gasspy_ids"][success]
# For sorting and inverse sorting due to h5py's particularity
cell_unique_ids, cell_local_ids = np.unique(cell_gasspy_ids, return_inverse=True) 
cell_gasspy_ids_sorter = cell_unique_ids.argsort()
cell_cell_sorter = cell_gasspy_ids_sorter.argsort()

if interpolate:
    # Determine all the neighbors
    needed_gasspy_ids = grid_database["neighbor_ids"][cell_unique_ids[cell_gasspy_ids_sorter]][cell_cell_sorter,:][cell_local_ids,:]
    needed_gasspy_ids = needed_gasspy_ids[:,neighbor_idxs]

    neighbor_shape    = needed_gasspy_ids.shape
    needed_gasspy_ids = needed_gasspy_ids.ravel()
    unique_ids, local_neighbor_ids = np.unique(needed_gasspy_ids, return_inverse=True)
    # How to sort these
    gasspy_ids_sorter = unique_ids.argsort()
    gasspy_ids_inv_sorter = gasspy_ids_sorter.argsort()


    # Get model_data for cell and gridded models
    cell_models = cell_database["model_data"][success, : ]
    cell_model_data = cell_database["model_data"][:, interp_field_idxs][success]
    grid_models = grid_database["model_data"][cell_unique_ids[cell_gasspy_ids_sorter],:][cell_cell_sorter][cell_local_ids]
    grid_model_data = grid_database["model_data"][unique_ids[gasspy_ids_sorter],:][gasspy_ids_inv_sorter][local_neighbor_ids]
    grid_model_data = grid_model_data.reshape(neighbor_shape + grid_model_data.shape[-1:])
    grid_model_data = grid_model_data[:,:,interp_field_idxs]
    grid_model_deltas = grid_model_data[:,:,:] - grid_model_data[:,0,:][:,None,:] 
    cell_model_deltas = cell_model_data - grid_model_data[:,0,:]


var_labels = ["$\Delta x$ [cm]", "$n$ [cm$^{-3}$]", "$T$ [K]", "$F_{FUV}$ [cm$^{-2}$]", "$F_{HII}$ [cm$^{-2}$]", "$F_{HeII}$ [cm$^{-2}$]", "$F_{HeIII}$ [cm$^{-2}$]"]
#var_lims = [[1e-4,5e-4],
#            [1e-4,5e-2],
#            [1e-4,5e-2],
#            [1e-4,5e-2],
#            [1e-4,5e-2],
#            [1e-4,5e-2],
#            [1e-4,5e-2]]
var_lims = [[None,None],
            [None,None],
            [None,None],
            [None,None],
            [None,None],
            [None,None],
            [None,None],
            ]

var_bins = []
delta_bins = []
for ivar in range(cell_models.shape[1]):
    print(ivar, max_delta[ivar])
    if var_lims[ivar][0] is None:
        vmin = np.min(cell_models[:,ivar])
    else:
        vmin = var_lims[ivar][0]
    if var_lims[ivar][1] is None:
        vmax = np.max(cell_models[:,ivar])
    else:
        vmax = var_lims[ivar][1]

    var_bins.append(10**np.arange(vmin, vmax+max_delta[ivar], max_delta[ivar]))
    delta_bins.append(np.linspace(0,max_delta[ivar]/2, 20))

for line in line_list:
    print(line)
    if line not in line_labels:
        print(line_labels)
        sys.exit("Error: line %s not in database"%line)
    iline = line_labels.index(line)

    cell_line_intensity = cell_database["line_intensity"][success,iline]
    cell_cont_intensity = cell_database["cont_intensity"][success,iline]

    grid_line_intensity = grid_database["line_intensity"][cell_unique_ids[cell_gasspy_ids_sorter], iline][cell_cell_sorter][cell_local_ids]
    # Determine relative error
    grid_error,  grid_relative_error = get_error(cell_line_intensity, grid_line_intensity)
    # Mask away relative errors that are to small in absolute terms
    if args.min_cont_frac is not None:
        grid_error[grid_error < cell_cont_intensity*args.min_cont_frac] = 0.0
        grid_relative_error[grid_error < cell_cont_intensity*args.min_cont_frac] = 0.0

    if args.min_sensitivity is not None:
        emission_size = get_error_emission_size(grid_error, args.min_sensitivity)
        grid_error[emission_size > 10**cell_models[:,0]] = 0.0
        grid_relative_error[emission_size > 10**cell_models[:,0]] = 0.0

    print(np.min(cell_models[grid_relative_error>0.01,4]))
    min_bin = min(np.log10(np.nanmin(grid_relative_error[grid_relative_error > 1e-40])), -5)
    max_bin = np.log10(np.nanmax(grid_relative_error[np.isfinite(grid_relative_error)]))

    mask = np.where((cell_models[:,4]<0) * (cell_models[:,1]>0) * (cell_models[:,2] < 3) * (cell_models[:,3]> 6))[0]
    print(mask)
    idx = np.argmax(grid_relative_error[mask])
    print(mask, idx)
    print(cell_models[mask,:][idx,:], grid_relative_error[mask][idx])
    print(grid_models[mask,:][idx,:], grid_relative_error[mask][idx])

    if interpolate:
        grid_ngbr_intensity = grid_database["line_intensity"][unique_ids[gasspy_ids_sorter], iline][gasspy_ids_inv_sorter][local_neighbor_ids]
        grid_ngbr_intensity = grid_ngbr_intensity.reshape(neighbor_shape)
        # set minimum
        grid_ngbr_intensity[grid_ngbr_intensity<1e-40] = 1e-40 
        print("interpolating")
        intr_line_intensity = 10**interpolate_kernel(grid_model_deltas, np.log10(grid_ngbr_intensity), cell_model_deltas)
        print("interpolation complete")
        # Determine relative error
        intr_error,  intr_relative_error = get_error(cell_line_intensity, intr_line_intensity)


        # Mask away relative errors that are to small in absolute terms
        if args.min_cont_frac is not None:
            intr_error[intr_error < cell_cont_intensity*args.min_cont_frac] = 0.0
            intr_relative_error[intr_error < cell_cont_intensity*args.min_cont_frac] = 0.0
        if args.min_sensitivity is not None:
            emission_size = get_error_emission_size(intr_error, args.min_sensitivity)
            intr_error[emission_size > 10**cell_models[:,0]] = 0.0
            intr_relative_error[emission_size > 10**cell_models[:,0]] = 0.0
        
        min_bin = min(min_bin,np.log10(np.nanmin(intr_relative_error[intr_relative_error>1e-40])))
        max_bin = max(max_bin,np.log10(np.nanmax(intr_relative_error[np.isfinite(intr_relative_error)])))

        diff = intr_relative_error - grid_relative_error
        intr_worse = np.argmax(diff)

    # Simple plot of relative error distribution
    fig, axes = plt.subplots(nrows = 2,ncols = 1, sharex=True)
    bin_edges = 10**np.arange(min_bin, max_bin, 0.05)
    data = grid_relative_error.copy()
    data[data<bin_edges[0]] = (bin_edges[1] + bin_edges[0])*0.5
    data[data>bin_edges[-1]] = (bin_edges[-1] + bin_edges[-2])*0.5
    grid_hist, bin_edges = np.histogram(data, bins = bin_edges, density= True)

    bin_sizes = bin_edges[1:] - bin_edges[:-1]
    bins = (bin_edges[1:] + bin_edges[:-1])*0.5
    
    grid_cumhist = np.cumsum((bin_sizes*grid_hist)[::-1])[::-1]
    axes[0].plot(bins, grid_cumhist, label = "Closest")
    axes[1].plot(bins, grid_hist, label = "Closest")
    if interpolate:
        data = intr_relative_error.copy()
        data[data<10**min_bin] = (bin_edges[1] + bin_edges[0])*0.5
        data[data>10**max_bin] = (bin_edges[-1] + bin_edges[-2])*0.5
        intr_relative_error[intr_relative_error < 10**min_bin] = (bin_edges[1] + bin_edges[0])*0.5
        intr_hist, bin_edges = np.histogram(intr_relative_error, bins = bin_edges, density= True)
        intr_cumhist = np.cumsum((bin_sizes*intr_hist)[::-1])[::-1]
        axes[0].plot(bins, intr_cumhist, label = "Interpolated")
        axes[1].plot(bins, intr_hist, label = "Interpolated")
    
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-3, 5e1)
    axes[0].set_ylim(1e-4, 1.4e0)
    axes[1].set_xlabel(r"$\Delta J/J_\mathrm{cell}$")
    axes[0].set_ylabel(r"$F(>\Delta J/J_\mathrm{cell})$")
    axes[1].set_ylabel(r"$f(\Delta J/J_\mathrm{cell})$")
    
    axes[0].legend()
    axes[0].set_title(line)
    if args.plotdir is None:
        plt.show()
    else:
        plt.savefig(args.plotdir + line.replace(" ", "_").replace("__","_")+ ".png")
        plt.close()


    cmap = truncate_colormap(plt.get_cmap("inferno"), 0, 0.85)
    plot_var_error(grid_relative_error, bin_edges, prefix = "closest_", cmap=cmap)
    plot_var_error(intr_relative_error, bin_edges, prefix = "interp_", cmap=cmap)

    plot_deltas_error(grid_relative_error, bin_edges, prefix = "closest_", cmap=cmap)
    plot_deltas_error(intr_relative_error, bin_edges, prefix = "interp_", cmap=cmap)  

    plot_var_var_error(1,2, grid_relative_error, "closest_dens_temp_"+ line.replace(" ", "_").replace("__","_")+ ".png", cmap=cmap)
    plot_var_var_error(1,4, grid_relative_error, "closest_dens_fion_"+ line.replace(" ", "_").replace("__","_")+ ".png", cmap=cmap)
    plot_var_var_error(2,4, grid_relative_error, "closest_temp_fion_"+ line.replace(" ", "_").replace("__","_")+ ".png", cmap=cmap)
    plot_var_var_error(1,3, grid_relative_error, "closest_dens_fuv_"+ line.replace(" ", "_").replace("__","_")+ ".png", cmap=cmap)
    plot_var_var_error(2,3, grid_relative_error, "closest_temp_fuv_"+ line.replace(" ", "_").replace("__","_")+ ".png", cmap=cmap)

    plot_var_var_error(1,2, intr_relative_error, "interp_dens_temp_"+ line.replace(" ", "_").replace("__","_")+ ".png", cmap=cmap)
    plot_var_var_error(1,4, intr_relative_error, "interp_dens_fion_"+ line.replace(" ", "_").replace("__","_")+ ".png", cmap=cmap)
    plot_var_var_error(2,4, intr_relative_error, "interp_temp_fion_"+ line.replace(" ", "_").replace("__","_")+ ".png", cmap=cmap)
    plot_var_var_error(1,3, intr_relative_error, "interp_dens_fuv_"+ line.replace(" ", "_").replace("__","_")+ ".png", cmap=cmap)
    plot_var_var_error(2,3, intr_relative_error, "interp_temp_fuv_"+ line.replace(" ", "_").replace("__","_")+ ".png", cmap=cmap)

