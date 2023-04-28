import argparse 
import numpy as np
import h5py as hp
import sys
import os
import matplotlib.pyplot as plt
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
        for idim in range(neigh_points.shape[0]):
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
            

def get_error(true_vals, apprx_vals, minval = 1e-40):
    error = np.abs(true_vals - apprx_vals)
    relative_error = error/np.maximum(true_vals, minval)

    return error, relative_error

    

ap = argparse.ArgumentParser()
ap.add_argument("cell_database")
ap.add_argument("grid_database")
ap.add_argument("--gasspy_config", default= "./gasspy_config")
ap.add_argument("--min_cont_frac", type = float, default=0.1)
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

# gasspy ids specifically wanted by the cells
cell_gasspy_ids = cell_database["cell_gasspy_ids"][:]
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
    cell_model_data = cell_database["model_data"][:, interp_field_idxs]
    grid_model_data = grid_database["model_data"][unique_ids[gasspy_ids_sorter],:][gasspy_ids_inv_sorter][local_neighbor_ids]
    grid_model_data = grid_model_data.reshape(neighbor_shape + grid_model_data.shape[-1:])
    grid_model_data = grid_model_data[:,:,interp_field_idxs]
    grid_model_deltas = grid_model_data[:,:,:] - grid_model_data[:,0,:][:,None,:] 
    cell_model_deltas = cell_model_data - grid_model_data[:,0,:]


var_labels = ["$\Delta x$ [cm]", "$n$ [cm$^{-3}$]", "$T$ [K]", "$F_{FUV}$ [cm$^{-2}$]", "$F_{HII}$ [cm$^{-2}$]", "$F_{HeII}$ [cm$^{-2}$]", "$F_{HeIII}$ [cm$^{-2}$]"]
var_lims = [[None,None],
            [-2,5],
            [0,7],
            [-10, None],
            [-10, None],
            [-10, None],
            [-10, None]]

for line in line_list:
    if line not in line_labels:
        print(line_labels)
        sys.exit("Error: line %s not in database"%line)
    iline = line_labels.index(line)

    cell_line_intensity = cell_database["line_intensity"][:,iline]
    cell_cont_intensity = cell_database["cont_intensity"][:,iline]

    grid_line_intensity = grid_database["line_intensity"][cell_unique_ids[cell_gasspy_ids_sorter], iline][cell_cell_sorter][cell_local_ids]
    # Determine relative error
    grid_error,  grid_relative_error = get_error(cell_line_intensity, grid_line_intensity)
    # Mask away relative errors that are to small in absolute terms
    grid_error[grid_error < cell_cont_intensity*args.min_cont_frac] = 0
    grid_relative_error[grid_error < cell_cont_intensity*args.min_cont_frac] = 0

    min_bin = min(np.log10(np.nanmin(grid_relative_error[grid_relative_error > 0])), -3)
    max_bin = max(min_bin + 4, 1)
    
    if interpolate:
        grid_ngbr_intensity = grid_database["line_intensity"][unique_ids[gasspy_ids_sorter], iline][gasspy_ids_inv_sorter][local_neighbor_ids]
        grid_ngbr_intensity = grid_ngbr_intensity.reshape(neighbor_shape)
        intr_line_intensity = 10**interpolate_kernel(grid_model_deltas, np.log10(grid_ngbr_intensity), cell_model_deltas)
        
        # Determine relative error
        intr_error,  intr_relative_error = get_error(cell_line_intensity, intr_line_intensity)


        # Mask away relative errors that are to small in absolute terms
        intr_error[intr_error < cell_cont_intensity*args.min_cont_frac] = 0
        intr_relative_error[intr_error < cell_cont_intensity*args.min_cont_frac] = 0

        min_bin = min(min(min_bin,np.log10(np.nanmin(intr_relative_error[intr_relative_error>0]))),-3)
        max_bin = max(max_bin, min_bin+4)

    # Simple plot of relative error distribution
    fig, axes = plt.subplots(nrows = 2,ncols = 1, sharex=True)
    bin_edges = np.logspace(min_bin, max_bin)

    grid_hist, bin_edges = np.histogram(grid_relative_error, bins = bin_edges, density= True)

    bin_sizes = bin_edges[1:] - bin_edges[:-1]
    bins = (bin_edges[1:] + bin_edges[:-1])*0.5
    
    grid_cumhist = np.cumsum((bin_sizes*grid_hist)[::-1])[::-1]
    axes[0].plot(bins, grid_cumhist, label = "Closest")
    axes[1].plot(bins, grid_hist, label = "Closest")
    if interpolate:
        intr_hist, bin_edges = np.histogram(intr_relative_error, bins = bin_edges, density= True)
        intr_cumhist = np.cumsum((bin_sizes*intr_hist)[::-1])[::-1]
        axes[0].plot(bins, intr_cumhist, label = "Interpolated")
        axes[1].plot(bins, intr_hist, label = "Interpolated")
    
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-3, None)
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
    

    




