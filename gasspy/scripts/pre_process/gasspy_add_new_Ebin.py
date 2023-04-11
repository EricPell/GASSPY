"""
    Author: Loke Ohlin & Eric Pellegrini
    Date: 05-2022
    Purpose: wrapper script to build, run and collect the database in one call rather than 3
    Usage: Fire and forget: call using arguments to describe the wanted directory structure. 
           If other spectra_mesh wanted, copy and write your own
    TODO:
        1) Allow more arguments to be specified in the gasspy_config.yaml
        2) create a class, and move each task into that class for better readability
        3) Make the spectra_mesh window definition into a separate class/function. allow user to supply their own
"""

import os
import sys
import numpy as np
import cupy

import argparse
import importlib.util

from gasspy.raytracing.ray_processors import Flux_calculator
from gasspy.raytracing.raytracers import Raytracer_AMR_neighbor
from gasspy.raytracing.observers import observer_healpix_class

from gasspy.io import gasspy_io

ap = argparse.ArgumentParser()
#-------------DIRECTORIES AND FILES---------------#
ap.add_argument("--simdir", default="./", help="Directory of the simulation and also default work directory")
ap.add_argument("--workdir", default= None, help="work directory. If not specified its the same as simdir")
ap.add_argument("--gasspydir", default="GASSPY", help="directory inside of simdir to put the GASSPY files")
ap.add_argument("--simulation_reader_dir", default="./", help="directory to the simulation_reader class that describes how to load the simulation")
ap.add_argument("--sim_prefix", default = None, help="prefix to put before all snapshot specific files")
#-------------Run parameters-----------#
ap.add_argument("--liteVRAM", action="store_true", help = "Force rerun of raytrace even if the trace file already exists")
ap.add_argument("--fieldname", default = "Fuv")
#############################################
# I) Initialization of the script
#############################################
## parse the commandline argument
args = ap.parse_args()

## move to workdir
if args.workdir is not None:
    workdir = args.workdir
else:
    workdir = args.simdir
os.chdir(workdir)

## create GASSPY dir where all files specific to this snapshot is kept
if not os.path.exists(args.gasspydir):
    os.makedirs(args.gasspydir)

## set prefix to snapshot specific files
if args.sim_prefix is not None:
    ## add an underscore
    sim_prefix = args.sim_prefix + "_"
else:
    sim_prefix = ""

##############################
# II) Load config files and simulation reader 
##############################

## Load the gasspy_config yaml
gasspy_config = gasspy_io.read_fluxdef("./gasspy_config.yaml")

## Load the simulation data class from directory
spec = importlib.util.spec_from_file_location("simulation_reader", args.simulation_reader_dir + "/simulation_reader.py")
reader_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader_mod)
sim_reader = reader_mod.Simulation_Reader(args.simdir, args.gasspydir, gasspy_config["sim_reader_args"])

## Determine maximum memory usage
if "max_mem_GPU" in gasspy_config.keys():
    max_mem_GPU = gasspy_config["max_mem_GPU"]
else:
    max_mem_GPU = 6

if "max_mem_CPU" in gasspy_config.keys():
    max_mem_CPU = gasspy_config["max_mem_CPU"]
else:
    max_mem_CPU = 14


###########################
# III) Define opacity function
###########################
dens_stream = cupy.cuda.stream.Stream(non_blocking=True)
temp_stream = cupy.cuda.stream.Stream(non_blocking=True)
def dust_uv_opacity(cell_index, field_dict):
    cell_index_cpu = cell_index.get()
    with temp_stream:
        cell_temperature_fact = cupy.exp(-cupy.asarray(field_dict["temp"][cell_index_cpu])/1e5)
    with dens_stream:
        cell_number_density = cupy.asarray(field_dict["nden"][cell_index_cpu]) * 1.615096e-21
    dens_stream.synchronize()
    temp_stream.synchronize()
    return cell_number_density*cell_temperature_fact 

neighs = sim_reader.get_cell_neighbors()
neighs = np.sort(neighs, axis = 1)
num_neighs = (neighs[:,1:] != neighs[:,:-1]).sum(axis=1)+1# - np.any(neighs == -1, axis  = 1)


###########################
# III) Calculate the raytrace
###########################
print("Raytracing")
# sourcetable should be a (N,4) array with per source N we have (x, y, z, Nphotons)
sourcetable = np.atleast_2d(np.load(gasspy_config["sourcetable_path"]))
nrots_per_source = gasspy_config["nrots_per_source"]

profiling = True
if profiling:
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()

for isource in range(len(sourcetable)):
    source_pos = sourcetable[isource,:3]/gasspy_config["sim_unit_length"]
    source_Nphoton = sourcetable[isource, 3]

    for irot in range(nrots_per_source) :
        ## Initialize the raytracer TODO: Check if we can just keep this instance across all traces
        observer = observer_healpix_class(gasspy_config, observer_center = source_pos, pov_center = np.random.rand(3))
        #observer = observer_plane_class(gasspy_config, observer_center = np.array([0.25,0.25,0.5]))

        raytracer = Raytracer_AMR_neighbor(sim_reader, gasspy_config, bufferSizeCPU_GB = max_mem_CPU, bufferSizeGPU_GB = max_mem_GPU, no_ray_splitting=False, liteVRAM = args.liteVRAM)
        ray_processor = Flux_calculator(raytracer, sim_reader, source_Nphoton/nrots_per_source, dust_uv_opacity, ["nden", "temp"])
        raytracer.set_ray_processor(ray_processor)
        #"""
        ## set observer
        raytracer.update_observer(observer = observer)
        ## run
        print(" - running raytrace")
        raytracer.raytrace_run()
        ## save to array
        if isource == 0 and irot == 0:
            cell_fluxes = ray_processor.get_fluxes()
        else:
            cell_fluxes += ray_processor.get_fluxes()
        
        ## clean up a bit
        rays = raytracer.global_rays
        #"""
        del raytracer
        del observer
        del ray_processor

if profiling:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.dump_stats("raytrace_precise_neighbor.prof")

sim_reader.save_new_field(args.fieldname, cell_fluxes)
'''
dxs   = sim_reader.get_field("dx") 
lrefs = sim_reader.get_field("amr_lrefine") 
xs    = sim_reader.get_field("x") 
ys    = sim_reader.get_field("y") 
zs    = sim_reader.get_field("z") 
index1D = sim_reader.get_index1D()
cell_index = np.arange(len(xs))
"""
debug_id = 6224012
idirs = range(12,15)
print(debug_id,neighs[debug_id,idirs])
print(xs[debug_id]/gasspy_config["sim_unit_length"], xs[neighs[debug_id,idirs]]/gasspy_config["sim_unit_length"])
print(ys[debug_id]/gasspy_config["sim_unit_length"], ys[neighs[debug_id,idirs]]/gasspy_config["sim_unit_length"])
print(zs[debug_id]/gasspy_config["sim_unit_length"], zs[neighs[debug_id,idirs]]/gasspy_config["sim_unit_length"])
print(lrefs[debug_id], lrefs[neighs[debug_id,idirs]])
"""
# Initialize per refinement level lists
from gasspy.shared_utils.functions import sorted_in1d

cell_index_lref = []
index1D_lref = []
amr_lrefine_min = sim_reader.sim_info["minref"]
amr_lrefine_max = sim_reader.sim_info["maxref"]
for lref in range(amr_lrefine_min, amr_lrefine_max + 1):
    at_lref = lrefs == lref
    idx_sort = index1D[at_lref].argsort()
    # Add arrays sorted in index1D
    index1D_lref.append(index1D[at_lref][idx_sort].astype(int))
    cell_index_lref.append(cell_index[at_lref][idx_sort].astype(int))

boxsize = gasspy_config["sim_unit_length"]
boxhalf = gasspy_config["sim_unit_length"]*0.5 + 0.1*np.min(dxs)

lmin = np.min(lrefs) 
lmax = np.max(lrefs) 
 
nmax1D = int(2**lmax) 
ionSlice  = np.zeros((nmax1D, nmax1D)) 

slice_pos = sourcetable[isource,0]
in_slice = (xs-0.5*dxs < slice_pos) * (xs+0.5*dxs > slice_pos) 

# loop over refinement levels 
for lref in range(lmin, lmax +1): 
    at_lref = np.where((lrefs == lref)*in_slice)
    fluxes = cell_fluxes[at_lref] 
    dx = dxs[at_lref] 
    ix = (zs[at_lref]/dx).astype(int) 
    iy = (ys[at_lref]/dx).astype(int) 
    z = zs[at_lref] 
    npx = nmax1D//int(2**(lref))
    idx_add = np.meshgrid(np.arange(npx), np.arange(npx)) 
    ixs = (idx_add[0] + ix[:,np.newaxis, np.newaxis]*npx).ravel() 
    iys = (idx_add[1] + iy[:,np.newaxis, np.newaxis]*npx).ravel() 

    np.add.at(ionSlice, (ixs, iys), np.repeat(fluxes, npx*npx)) 

md = np.max(ionSlice)
limS = np.array([1e0, 1e15]) 
print(limS)
#md = np.max(ionMap)
#limM = np.array([md/1e4, md]) 

import matplotlib.pyplot as plt
import matplotlib.colors as mcol
fig, ax  = plt.subplots(nrows = 1, ncols = 1, figsize = (5,5), sharex = True, sharey = True) 
if not isinstance(ax, list):
    ax = [ax]
ax[0].imshow(ionSlice.T, cmap = "viridis", origin = "lower", extent = [0,boxsize, 0, boxsize], norm = mcol.LogNorm(vmin = limS[0], vmax = limS[1])) 

plt.show()
'''