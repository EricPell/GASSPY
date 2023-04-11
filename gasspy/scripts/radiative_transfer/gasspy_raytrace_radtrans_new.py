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
import torch
import gc

import argparse
import importlib.util


from gasspy.raytracing.raytracers import Raytracer_AMR_neighbor
from gasspy.raytracing.ray_processors import Raytrace_saver
from gasspy.raytracing.observers import observer_plane_class, observer_healpix_class
from gasspy.radtransfer.rt_trace import Trace_processor
from gasspy.io import gasspy_io
from gasspy.shared_utils.gpu_util.gpu_memory import free_memory

ap = argparse.ArgumentParser()
#-------------DIRECTORIES AND FILES---------------#
ap.add_argument("--simdir", default="./", help="Directory of the simulation and also default work directory")
ap.add_argument("--workdir", default= None, help="work directory. If not specified its the same as simdir")
ap.add_argument("--gasspydir", default="GASSPY", help="directory inside of simdir to put the GASSPY files")
ap.add_argument("--modeldir" , default="GASSPY", help = "directory inside of workdir where to read, put and run the cloudy models")
ap.add_argument("--simulation_reader_dir", default="./", help="directory to the simulation_reader class that describes how to load the simulation")
ap.add_argument("--sim_prefix", default = None, help="prefix to put before all snapshot specific files")
ap.add_argument("--trace_file", default = None, help="name of trace file. If it does not exist we need to recreate it")
ap.add_argument("--spec_save_name", default = None,  help = "Path to file where to save the spectra. Default to sim_prefix+\"_spec.hdf5\"")
#-------------Run parameters-----------#
ap.add_argument("--rerun_raytrace", action="store_true", help = "Force rerun of raytrace even if the trace file already exists")
ap.add_argument("--liteVRAM", action="store_true", help = "Force rerun of raytrace even if the trace file already exists")

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
    sys.exit("ERROR : cant find directory %s"%args.gasspydir)

if not os.path.exists(args.gasspydir+"/projections/"):
    os.makedirs(args.gasspydir+"/projections/")

if not os.path.exists(args.modeldir):
    sys.exit("ERROR : cant find directory %s"%args.modeldir)

## set prefix to snapshot specific files
if args.sim_prefix is not None:
    ## add an underscore
    sim_prefix = args.sim_prefix + "_"
else:
    sim_prefix = ""

##############################
# II) Load config files and simulation reader 
##############################

## Load the fluxdef yaml file
fluxdef = gasspy_io.read_fluxdef("./gasspy_fluxdef.yaml")

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
    max_mem_GPU = 4

if "max_mem_CPU" in gasspy_config.keys():
    max_mem_CPU = gasspy_config["max_mem_CPU"]
else:
    max_mem_CPU = 14

###########################
# III) Calculate the raytrace
###########################
if args.trace_file is not None:
    trace_file = args.gasspydir+"/projections/"+args.trace_file
else:
    trace_file = args.gasspydir+"/projections/"+sim_prefix+"trace.hdf5"
if not os.path.exists(trace_file) or args.rerun_raytrace:
    print("Raytracing")
    ## Define the observer class   
    if "observer_type" in gasspy_config.keys() and gasspy_config["observer_type"] == "healpix":
        observer = observer_healpix_class(gasspy_config)
    else:
        observer = observer_plane_class(gasspy_config)

    ## Initialize the raytracer and ray_processer
    print(" - initializing raytracer")
    raytracer = Raytracer_AMR_neighbor(sim_reader, gasspy_config, bufferSizeCPU_GB = max_mem_CPU, bufferSizeGPU_GB = max_mem_GPU, no_ray_splitting=False, liteVRAM = args.liteVRAM)
    print(" - initializing ray_processor")
    ray_processor = Raytrace_saver(raytracer)
    raytracer.set_ray_processor(ray_processor)
    ## set observer
    raytracer.update_observer(observer = observer)

    ## run
    print(" - running raytrace")
    raytracer.raytrace_run()

    ## save TODO: stop this and just keep in memory
    print(" - saving trace")
    raytracer.save_trace(trace_file)

    ## clean up a bit
    del raytracer
    del observer
    ray_processor.clean()
    del ray_processor
    free_memory()
    gc.collect()
##########################
# IV) Radiative transfer
##########################

## Memory management
cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()

cuda_device = torch.device('cuda:0')
Elims = None
if "Elims" in gasspy_config.keys():
    Elims = np.array(gasspy_config["Elims"]).astype(float)

if args.spec_save_name is None:
    spec_save_name = sim_prefix + "spec.hdf5"
else:
    spec_save_name = args.spec_save_name
print("Radiative transfer")
mytree = Trace_processor(
    root_dir="./",
    modeldir = args.modeldir,
    gasspy_subdir=args.gasspydir,
    config_yaml=gasspy_config,
    traced_rays=trace_file,
    sim_reader = sim_reader, 
    energy_lims=Elims,
    h5database=args.modeldir + "/gasspy_database.hdf5",
    cell_index_to_gasspydb = args.gasspydir + "/cell_data/" + sim_prefix+"cell_gasspy_index.npy",
    opc_per_NH=False,
    accel="torch",
    liteVRAM=args.liteVRAM,
    Nraster=4,
    spec_save_name=spec_save_name,
    dtype=np.float32,
    cuda_device=cuda_device,
    doppler_shift=False,
    doppler_shift_est_bin_ratio=5
)
print(" - Loading files")
mytree.load_all()
print(" - Processing rays")
mytree.process_trace()
