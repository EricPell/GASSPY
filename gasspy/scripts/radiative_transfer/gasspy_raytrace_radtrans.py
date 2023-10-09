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
ap.add_argument("gasspy_config")
ap.add_argument("--simdir", default="./", help="Directory of the simulation and also default work directory")
#-------------Run parameters-----------#
ap.add_argument("--rerun_raytrace", action="store_true", help = "Force rerun of raytrace even if the trace file already exists")

#############################################
# I) Initialization of the script
#############################################
## parse the commandline argument
args = ap.parse_args()

## Load the gasspy_config yaml
gasspy_config = gasspy_io.read_gasspy_config(args.gasspy_config)

## create gasspy_subdir where all files specific to this snapshot is kept
snapshots = gasspy_config["snapshots"]
assert len(snapshots.keys()) == 1, "Can only specify one snapshot at a time for radiative transfer"
snaphot = snapshots[list(snapshots.keys())[0]]
gasspy_subdir = snaphot["gasspy_subdir"]
if not os.path.exists(gasspy_subdir):
    sys.exit("ERROR : cant find directory %s. This needs to be here before doing RT to know the cell-gasspymodel mapping"%gasspy_subdir)

if not os.path.exists(gasspy_subdir+"/projections/"):
    os.makedirs(gasspy_subdir+"/projections/")


##############################
# II) Load simulation reader 
##############################

## Simulation reader directory
simulation_readerdir = gasspy_io.check_parameter_in_config(gasspy_config, "simulation_reader_dir", None, "./")

## Load the simulation data class from directory
spec = importlib.util.spec_from_file_location("simulation_reader", simulation_readerdir + "/simulation_reader.py")
reader_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader_mod)
sim_reader = reader_mod.Simulation_Reader(gasspy_config, snaphot)


###########################
# III) Calculate the raytrace
###########################
# Get prefix (if any) and observer type
trace_prefix = gasspy_io.check_parameter_in_config(gasspy_config, "trace_prefix", None, "")
if trace_prefix != "" and not trace_prefix.endswith("_"):
    trace_prefix += "_"
observer_type = gasspy_io.check_parameter_in_config(gasspy_config, "observer_type", None, "plane")
assert observer_type in ["plane", "healpix"], "Error: invalid observer_type provided. Current options are \"healpix\" and \"plane\" (Default)"

# Name of the trace file
trace_file = gasspy_subdir+"/projections/%s%s_trace.hdf5"%(trace_prefix, observer_type)

if not os.path.exists(trace_file) or args.rerun_raytrace:
    print("Raytracing")
    ## Define the observer class   
    if observer_type == "healpix":
        observer = observer_healpix_class(gasspy_config)
    else:
        observer = observer_plane_class(gasspy_config)

    ## Initialize the raytracer and ray_processer
    print(" - initializing raytracer")
    raytracer = Raytracer_AMR_neighbor(gasspy_config, sim_reader)
    print(" - initializing ray_processor")
    ray_processor = Raytrace_saver(gasspy_config, raytracer)
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

spec_prefix = gasspy_io.check_parameter_in_config(gasspy_config, "spec_prefix", None, "")
if spec_prefix != "" and not spec_prefix :
    spec_prefix += "_"
spec_file = "%s_%s_%s_spec.hdf5"%(spec_prefix, trace_prefix, observer_type)
print("Radiative transfer")
mytree = Trace_processor(
    gasspy_config,
    traced_rays=trace_file,
    sim_reader = sim_reader, 
    accel="torch",
    spec_save_name=spec_file,
    cuda_device=cuda_device,
)
print(" - Loading files")
mytree.load_all()
print(" - Processing rays")
mytree.process_trace()
