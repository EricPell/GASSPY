from inspect import trace
from sys import path
from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from torch import full
import yaml
import cupy
import cupyx
import sys
import h5py
from gasspy.raystructures import global_ray_class
import argparse
import importlib.util


"""
    DEFINE WHAT TO PLOT
"""
ap=argparse.ArgumentParser()

#---------------outputs-----------------------------
ap.add_argument('f', nargs='+')
ap.add_argument("--simdir", default="./")
ap.add_argument("--gasspydir", default="GASSPY")
ap.add_argument("--simulation_reader_dir", default="./", help="directory to the simulation_reader class that describes how to load the simulation")
args=ap.parse_args()


trace_files = args.f
simdir = args.simdir


# What we want to grab
# How many pseudo fields
pseudoList = ["Htotdensity", "HIIdensity"]
Npseudo = len(pseudoList)


"""
    LOAD SIM DATA
"""
mH = 1.675e-24
def arange_indexes(start, end):
    # Method to generate list of indices based on a start and end
    # (magic from stackoverflow...)
    lens = end - start
    cupy.cumsum(lens, out=lens)
    i = cupy.ones(int(lens[-1]), dtype=int)
    i[0] = start[0]
    i[lens[:-1]] += start[1:]
    i[lens[:-1]] -= end[:-1]
    cupy.cumsum(i, out=i)
    return i

# Open up the YAML config
with open(r"%s/gasspy_config.yaml"%simdir) as fil:
    gasspy_config = yaml.load(fil, Loader = yaml.FullLoader)
## Load the simulation data class from directory
spec = importlib.util.spec_from_file_location("simulation_reader", args.simulation_reader_dir + "/simulation_reader.py")
reader_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader_mod)
sim_reader = reader_mod.Simulation_Reader(args.simdir, args.gasspydir, gasspy_config["sim_reader_args"])




# Load denisty from the simulation
cell_dens = cupy.array(sim_reader.get_field("rho"), dtype = cupy.float64)/mH
cell_HIIdensity = cupy.array(sim_reader.get_field("rho"), dtype = cupy.float64)*cupy.array(sim_reader.get_field("xHII"), dtype = cupy.float64)/mH
cell_temperatue = cupy.array(sim_reader.get_field("T"), dtype = cupy.float64)
cell_velocity = cupy.array(sim_reader.get_field("vz"), dtype = cupy.float64)*1e5

# Take the amr refinement and index1D from our own calcualtions to make sure dtypes matches
cell_amr_lrefine = cupy.array(sim_reader.get_field("amr_lrefine"))
cell_index1D     = cupy.array(sim_reader.get_index1D())

# Load properties of the grid from the yaml file
min_lref = gasspy_config["amr_lrefine_min"]
max_lref = gasspy_config["amr_lrefine_max"]
boxlen  = gasspy_config["sim_size_x"]
scalel  = gasspy_config["sim_unit_length"]

# Pre caclualte all cell sizes
dx_lrefs = boxlen/2**(np.arange(min_lref, max_lref+1))

# TODO: DO THIS BEFORE SOMEWHERE
# Add all NULL values at the end
cell_dens = cupy.append(cell_dens, [0])
cell_HIIdensity = cupy.append(cell_HIIdensity, [0])
cell_temperatue = cupy.append(cell_temperatue, [1])
cell_velocity = cupy.append(cell_velocity, [0])

cell_pseudo_fields = cupy.zeros((len(cell_dens), 2))
cell_pseudo_fields[:,0] = cell_dens
cell_pseudo_fields[:,1] = cell_HIIdensity


"""
    LOAD TRACE DATA
"""

for trace_file in trace_files:
    print(trace_file)
    # load the trace object
    h5file = h5py.File(trace_file,"r")
    global_rays = global_ray_class()
    global_rays.load_hdf5(h5file)

    # Figure out which rays are leaf rays (defines as those with no child split event)
    ileafs = cupy.where(global_rays.get_field("cevid") == -1)[0]
    leaf_rays = global_rays.get_subset(ileafs)

    NleafRays = int(len(ileafs))
    print("Number of leafs : %d"%NleafRays)
    leaf_ray_pseudofields    = np.zeros((NleafRays,Npseudo))
    index_in_leaf_arrays     = np.zeros(global_rays.nrays, dtype=int)
    index_in_leaf_arrays[leaf_rays["global_rayid"].get()] = np.arange(NleafRays)


    # Min and max refinement level of the rays
    min_ray_lrefine = int(cupy.min(global_rays.get_field("ray_lrefine")))
    max_ray_lrefine = int(cupy.max(global_rays.get_field("ray_lrefine")))

    # split rays into lrefines
    rays_lrefs = []
    for lref in range(min_ray_lrefine, max_ray_lrefine+1):
        at_lref = cupy.where(global_rays.get_field("ray_lrefine") == lref)[0]
        rays_lrefs.append(global_rays.get_subset(at_lref))




    #Refinement level and index1D
    ray_amr_lrefine = h5file["ray_segments"]["amr_lrefine"][:,:]
    ray_index1D     = h5file["ray_segments"]["index1D"][:,:]
    ray_cell_index  = h5file["ray_segments"]["cell_index"][:,:]


    # Previously everything has been in boxlen=1 units, so we need to scale that
    ray_pathlength  = h5file["ray_segments"]["pathlength"][:,:]*scalel

    # Number of cells in a segment (eg the buffer size)
    NcellPerSeg = ray_amr_lrefine.shape[1]

    # Grab the list of global ray ids
    ray_global_rayid = cupy.array(h5file["ray_segments"]["global_rayid"][:])

    # Grab the list of split events
    splitEvents= cupy.array(h5file["splitEvents"][:,:])


    # maximum memory usage
    maxMemory  = 4 * 1024**3

    # bytes per ray RT storage
    Nfields = Npseudo
    perRay = Nfields * np.float64(1).itemsize 

    # bytes per ray segment
    # Number of fields + rayid + amr_lrefine + index1D
    perSeg = (2* Nfields * np.float64(1).itemsize + ray_index1D.itemsize + ray_amr_lrefine.itemsize + ray_pathlength.itemsize)* NcellPerSeg

    # Arrays to keep track of where things are
    ray_idx_in_gpu = cupy.full(global_rays.nrays, -1, dtype = int)
    ray_idx_in_cpu = np.full(global_rays.nrays, -1, dtype = int)

    # Init parent stuff
    parent_rays = None
    counter = 0
    for ray_lref in range(min_ray_lrefine, max_ray_lrefine+1):
        # How many rays do we have at this refinement level?
        rays_at_lref = rays_lrefs[ray_lref - min_ray_lrefine]
        nrays_at_lref = len(rays_at_lref["xp"])
        print(ray_lref, nrays_at_lref, global_rays.nrays)

        # Can we fit all of them?
        # If not take some predefined amount of memory
        if nrays_at_lref*perRay < 0.75*maxMemory:
            nrays_at_a_time = nrays_at_lref

        else:
            nrays_at_a_time = int(0.5*maxMemory/perRay)

        # Allocate CPU memory for all rays    
        ray_pseudofields_cpu    = cupyx.zeros_pinned((nrays_at_lref,Npseudo))

        # Determine rayid mapping to cpu arrays
        ray_idx_in_cpu[rays_at_lref["global_rayid"].get()] = np.arange(nrays_at_lref)

        # Add from parents
        if parent_rays is not None:
            # get the children IDs from the child split events of the parents
            child_rayid = splitEvents[parent_rays["cevid"].get(),1:].ravel().astype(int).get()

            # Duplicate all values by four and add to the cpu fields
            ray_pseudofields_cpu[ray_idx_in_cpu[child_rayid],:] = parent_pseudofields.repeat(4, axis = 0)

        iray_start = 0
        while iray_start < nrays_at_lref:
            # Figure out how many rays we have in this iteration
            iray_end = iray_start + nrays_at_a_time
            iray_end = min(iray_end, nrays_at_lref)
            nrays_now = (iray_end - iray_start)

            # Move the rays at this current iteration to the gpu
            ray_pseudofields_gpu    = cupy.array(ray_pseudofields_cpu[iray_start : iray_end, :])


            # Determine rayid mapping to cpu arrays
            ray_idx_in_gpu[rays_at_lref["global_rayid"]] = cupy.arange(nrays_now)


            # Memory for segments
            memory_remaining = maxMemory - nrays_now*perRay

            # Take the segments corresponding to the current sets of rays 
            idx = np.where(np.isin(ray_global_rayid.get(), rays_at_lref["global_rayid"][iray_start:iray_end].get()))[0]

            seg_amr_lrefine_at_ray_lref = ray_amr_lrefine[idx,:]
            seg_index1D_at_ray_lref = ray_index1D[idx,:]
            seg_pathlength_at_ray_lref = ray_pathlength[idx,:]
            seg_cell_index_at_ray_lref = ray_cell_index[idx,:]
            seg_global_rayid_at_ray_lref = ray_global_rayid[idx]

            # Total number of ray segments in the trace
            Nsegs = len(seg_pathlength_at_ray_lref)


            # How many we calculate at a time (Memory requiremenets)
            Nseg_at_a_time =min(Nsegs, int(memory_remaining/perSeg))

            # How many segments we have left
            Nrem = Nsegs

            # Index of the current segment
            iseg_start = 0

            while Nrem > 0:


                # Determine en index of list of segments
                iseg_end = min((iseg_start + Nseg_at_a_time), Nsegs)
                Nrem = Nrem - (iseg_end - iseg_start)

                # Grab all that you need onto the GPU
                amr_lrefine     = cupy.array(seg_amr_lrefine_at_ray_lref[iseg_start:iseg_end, : ])
                index1D     = cupy.array(seg_index1D_at_ray_lref[iseg_start:iseg_end, : ])

                cell_index     = cupy.array(seg_cell_index_at_ray_lref[iseg_start:iseg_end, : ])
                pathlength  = cupy.array(seg_pathlength_at_ray_lref[iseg_start:iseg_end,:])
                global_rayid = cupy.array(seg_global_rayid_at_ray_lref[iseg_start:iseg_end]) 


                field = pathlength[:,:,cupy.newaxis] * cell_pseudo_fields[cell_index,:]

                if cupy.any(field < 0):
                    print("??????????") 

                # Total in segments
                field = cupy.sum(field, axis = 1)

                #Gather all segments that belong to the same ray
                # The list is ordered so we just need to find where they differ           
                rayids, idxm, counts = cupy.unique(global_rayid, return_index=True, return_counts = True)
                idxp = idxm + counts
                # Determine all indexes
                idxes = arange_indexes(idxm, idxp)

                # Make a cumulative sum and then remove the previous value where we change rays
                tot_field = cupy.cumsum(field, axis = 0)
                # if we only have one ray, no need to do subtract previous values
                if len(idxm) > 1:
                #    # We only need to do this for the 2nd ray and beyond
                    tot_field[idxes[counts[0]:],:] -= cupy.repeat(tot_field[idxm[1:]-1,:], counts[1:].tolist(), axis = 0)



                # Now add the final bits of each ray to its values and 
                # Determine the indexes in the gpu arrays
                idx_in_gpu = ray_idx_in_gpu[rayids]
                ray_pseudofields_gpu[idx_in_gpu,:] += tot_field[idxp - 1,:]

                iseg_start = iseg_end
                counter += 1
            # Move off to the CPU arrays
            ray_pseudofields_cpu[iray_start: iray_end, :] = ray_pseudofields_gpu.get()


            iray_start = iray_end  

        # All rays at this refinement level are done!
        # Save all that are leafs
        is_leaf = rays_at_lref["cevid"] == -1
        idx_leaf = np.where(is_leaf.get())[0]
        idx_in_larr = index_in_leaf_arrays[rays_at_lref["global_rayid"][is_leaf].get()]
        leaf_ray_pseudofields[idx_in_larr, :] = ray_pseudofields_cpu [idx_leaf, :]

        # keep the parent for the next step
        is_parent  = ~is_leaf
        idx_parent = np.where(is_parent.get())[0]
        if len(idx_parent) == 0:
            parent_rays = None
        else:
            parent_rays = {}
            for key in rays_at_lref.keys():
                parent_rays[key] = rays_at_lref[key][is_parent]
            parent_pseudofields = ray_pseudofields_cpu [idx_parent, :] 







    dx_plot = 2**(-float(max_ray_lrefine))
    Nplot = int(2**max_ray_lrefine)

    plot_dens = np.zeros((Nplot,Nplot))
    plot_HII  = np.zeros((Nplot,Nplot))
    plot_nrays = np.zeros((Nplot,Nplot))    
    plot_lrefine = np.zeros((Nplot, Nplot))
    for lref in range(min_ray_lrefine, max_ray_lrefine + 1):
        dx_ray = 2**(-float(lref))
        Ncell_per_ray = 4**(max_ray_lrefine - lref)

        idx_at_lref = cupy.where(leaf_rays["ray_lrefine"] == lref)[0]
        rays_at_lref = {}
        for key in leaf_rays.keys():
            rays_at_lref[key] = leaf_rays[key][idx_at_lref]

        xplot_min = ((rays_at_lref["xp"] - 0.5*dx_ray)/dx_plot).astype(int).get()
        xplot_max = ((rays_at_lref["xp"] + 0.5*dx_ray)/dx_plot).astype(int).get()
        yplot_min = ((rays_at_lref["yp"] - 0.5*dx_ray)/dx_plot).astype(int).get()
        yplot_max = ((rays_at_lref["yp"] + 0.5*dx_ray)/dx_plot).astype(int).get()

        # get indexes in plot
        dens = leaf_ray_pseudofields[index_in_leaf_arrays[rays_at_lref["global_rayid"].get()],0]
        HII = leaf_ray_pseudofields[index_in_leaf_arrays[rays_at_lref["global_rayid"].get()],1]
        #dens = np.sum(leaf_ray_emissivity[index_in_leaf_arrays[rays_at_lref["global_rayid"].values.get()],:], axis = 1)

        for iray in range(len(xplot_min)):
            plot_dens[xplot_min[iray]:xplot_max[iray], yplot_min[iray]:yplot_max[iray]] += dens[iray]
            plot_HII[xplot_min[iray]:xplot_max[iray], yplot_min[iray]:yplot_max[iray]] += HII[iray]
            plot_nrays[xplot_min[iray]:xplot_max[iray], yplot_min[iray]:yplot_max[iray]] += 1
            plot_lrefine[xplot_min[iray]:xplot_max[iray], yplot_min[iray]:yplot_max[iray]] = lref


    mask = plot_nrays > 1
    plot_dens[mask] = plot_dens[mask]/plot_nrays[mask]
    plot_HII[mask]  = plot_HII[mask]/plot_nrays[mask]
    del(plot_nrays)
    #del(leaf_ray_pseudofields)
    pathdir = "/home/loki/Runs/spectra_test/"

    import matplotlib as mpl
    import matplotlib.colors as mcol
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def add_norm(data, rgb_vals, zmin = None, zmax = None):
        if zmin is None:
            zmin = max(np.min(data), 1e10)
        if zmax is None:
            zmax = np.max(data)

        norm = mcol.LogNorm(vmin = zmin, vmax = zmax)
        rgb_map = np.zeros((data.shape[0],data.shape[1],3))
        data = norm(data).data
        rgb_map = data[:,:,np.newaxis]*rgb_vals
        return rgb_map


    print(np.min(plot_dens), np.max(plot_dens))
    rgb_map = add_norm(plot_dens, np.array([0.55,0.25,0.75]), zmax = 5e22)
    rgb_map += add_norm(plot_HII, np.array([0.8,0.8,0.15]))
    rgb_map[rgb_map >= 1] = 1
    rgb_map[rgb_map <= 0] = 0
    rgb_map  = rgb_map.transpose(1,0,2)
    plot_dens = plot_dens.transpose()
    del(plot_HII)

    figsize = (14,6)
    fig, ax = plt.subplots(nrows= 1, ncols = 2, sharex = True, sharey = True, figsize = figsize)
    ax[0].imshow(rgb_map, origin = "lower", extent = [0,1,0,1])
    ax[1].imshow(plot_lrefine.T, origin = "lower", extent = [0,1,0,1])
    plotfile = trace_file[:-11]+ "_plot.png"
    plt.savefig(plotfile)
    plt.show()
    del(rgb_map)
    plt.cla()
    plt.clf()
    plt.close("all")
    plt.close(fig)