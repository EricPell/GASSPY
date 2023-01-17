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
import h5py as hp
import argparse
import importlib.util
from gasspy.raytracing.raytracers import Raytracer_AMR_neighbor
from gasspy.raytracing.ray_processors import Single_band_radiative_transfer
from gasspy.raytracing.observers import observer_plane_class, observer_healpix_class
from gasspy.io import gasspy_io

ap = argparse.ArgumentParser()
#-------------DIRECTORIES AND FILES---------------#
ap.add_argument("--simdir", default="./", help="Directory of the simulation and also default work directory")
ap.add_argument("--config_yaml", default="./gasspy_config.yaml")
ap.add_argument("--workdir", default= None, help="work directory. If not specified its the same as simdir")
ap.add_argument("--gasspydir", default="GASSPY", help="directory inside of simdir to put the GASSPY files")
ap.add_argument("--modeldir" , default="GASSPY", help = "directory inside of workdir where to read, put and run the cloudy models")
ap.add_argument("--simulation_reader_dir", default="./", help="directory to the simulation_reader class that describes how to load the simulation")
ap.add_argument("--sim_prefix", default = None, help="prefix to put before all snapshot specific files")
ap.add_argument("--outfile", default = None, help="name of trace file. If it does not exist we need to recreate it")
#-------------Run parameters-----------#
ap.add_argument("--rerun_raytrace", action="store_true", help = "Force rerun of raytrace even if the trace file already exists")
ap.add_argument("--liteVRAM", action="store_true", help = "All arrays except for those associated with the rays in system memory")
ap.add_argument("--save_opacity", action="store_true", help = "Save opacity of each band along with flux")

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
if not os.path.exists(args.modeldir):
    sys.exit("ERROR : cant find directory %s"%args.modeldir)


## set prefix to snapshot specific files
if args.sim_prefix is not None:
    ## add an underscore
    sim_prefix = args.sim_prefix + "_"
else:
    sim_prefix = ""

if args.outfile is None:
    if not os.path.exists(args.gasspydir+"/single_bands/"):
        os.makedirs(args.gasspydir+"/single_bands/")
    outfile = args.gasspydir + "/single_bands/%ssingle_bands.hdf5"%sim_prefix
else:
    outfile = args.outfile
##############################
# II) Load config files and simulation reader 
##############################

## Load the fluxdef yaml file
fluxdef = gasspy_io.read_fluxdef("./gasspy_fluxdef.yaml")

## Load the gasspy_config yaml
gasspy_config = gasspy_io.read_fluxdef(args.config_yaml)

## Load the simulation data class from directory
print("Initializing simulation reader")
spec = importlib.util.spec_from_file_location("simulation_reader", args.simulation_reader_dir + "/simulation_reader.py")
reader_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader_mod)
sim_reader = reader_mod.Simulation_Reader(args.simdir, args.gasspydir, gasspy_config["sim_reader_args"])

## Determine maximum memory usage
if "max_mem_GPU" in gasspy_config.keys():
    max_mem_GPU = gasspy_config["max_mem_GPU"]
else:
    max_mem_GPU = 8

if "max_mem_CPU" in gasspy_config.keys():
    max_mem_CPU = gasspy_config["max_mem_CPU"]
else:
    max_mem_CPU = 14

###########################
# III) Database information
###########################
gasspy_database   = hp.File(args.modeldir + "/gasspy_database.hdf5", "r")
cell_gasspy_index = np.load(args.gasspydir + "/cell_data/" + sim_prefix+"cell_gasspy_index.npy") 


###########################
# IV) figure out observers and energy bins
###########################
if not "observer_center" in gasspy_config.keys():
    observer_center = np.array([[0.5,0,5,1]])
else:
    observer_center = gasspy_config["observer_center"]
    if isinstance(observer_center, str):
        observer_center = np.atleast_2d(np.load(observer_center))
    else:
        observer_center = np.atleast_2d(observer_center)

if not "pov_center" in gasspy_config.keys():
    pov_center = np.array([[0.5,0,5,0.5]])
else:
    pov_center = gasspy_config["pov_center"]
    if isinstance(pov_center, str):
        pov_center = np.atleast_2d(np.load(pov_center))
    else:
        pov_center = np.atleast_2d(pov_center)   

# Make sure that observer_center and pov_center either has the same length or atleast on of them have the length of 1
if len(observer_center) != 1 and len(pov_center) != 1:
    if len(observer_center) != len(pov_center):
        sys.exit("observer center and pov center must either have the same length or atleast one of them have length 1")

# If one of the two have a length greater than 1, extent the other such that they are equal
elif len(observer_center) > 1:
    pov_center = np.repeat(pov_center, len(observer_center), axis = 0)
elif len(pov_center) > 1:
    observer_center = np.repeat(observer_center, len(pov_center), axis = 0)

# Load the limits of the energy bands
energy_limits = np.atleast_2d(gasspy_config["Elims"])
huh = gasspy_config["Elims"]

# Load the data for the lines (if any)
line_data = None
if "line_data" in gasspy_config.keys():
    line_data = gasspy_config["line_data"]

# Load the data for the broadbands (if any)
bband_data = None
if "bband_data" in gasspy_config.keys():
    bband_data = gasspy_config["bband_data"]

###########################
# V) Initialize hdf5 file with output rays
###########################
h5File = hp.File(outfile, "w")
h5File.create_dataset("energy_limits", data = energy_limits)
h5File.attrs.create("ndirs", data = len(pov_center))
h5File.attrs.create("nbands", data = len(energy_limits))
if line_data is not None:
    gasspy_io.save_dict_hdf5("line_data", line_data, h5File)
if bband_data is not None:
    gasspy_io.save_dict_hdf5("bband_data", bband_data, h5File)


##########################
# V) loop over directions and raytrace
##########################
maxflux = np.zeros(len(energy_limits))
## Initialize the raytracer and ray_processer
raytracer = Raytracer_AMR_neighbor(sim_reader, gasspy_config, bufferSizeCPU_GB = max_mem_CPU, bufferSizeGPU_GB = max_mem_GPU, no_ray_splitting=False, liteVRAM = args.liteVRAM, NcellBuff=16)
print("Initializing ray_processor")
ray_processor = Single_band_radiative_transfer(raytracer, gasspy_database, cell_gasspy_index, energy_limits, liteVRAM=args.liteVRAM)
print("Setting ray_processor")
raytracer.set_ray_processor(ray_processor)

for idir in range(len(pov_center)):
    ## Create a group for this POV
    grp_name = "dir_%05d"%idir
    grp = h5File.create_group(grp_name)


    print("\nRaytracing idir%d\n"%idir)
    ## Define the observer class   
    if "observer_type" in gasspy_config.keys() and gasspy_config["observer_type"] == "healpix":
        observer = observer_healpix_class(gasspy_config, observer_center = observer_center[idir,:], pov_center = pov_center[idir,:])
    else:
        observer = observer_plane_class  (gasspy_config, observer_center = observer_center[idir,:], pov_center = pov_center[idir,:])
    ## set observer
    print("Updating observer")
    raytracer.update_obsplane(obs_plane = observer)
    ## run
    print("Running raytrace")
    raytracer.raytrace_run()
    
    
    ## Save final flux for each ray
    # Find all leaf rays
    leaf_rays = cupy.where(raytracer.global_rays.get_field("cevid") == -1)[0]
    # xp and yp 
    grp.create_dataset("xp", data = raytracer.global_rays.get_field("xp", index = leaf_rays).get())
    grp.create_dataset("yp", data = raytracer.global_rays.get_field("yp", index = leaf_rays).get())
    # refinement level   
    grp.create_dataset("ray_lrefine", data = raytracer.global_rays.get_field("ray_lrefine", index = leaf_rays).get())
    
    # final flux in each band
    for iband in range(len(energy_limits)):
        # Currently we only have two observers, plane and healpix. Plane has a physical size to a pixel, so we should rescale that to get to 1/cm^2 rather than 1/sim_unit_length^2
        # TODO: THIS HAS TO BE REDONE SO THAT OBSERVER KNOWS ITS OWN SCALING IF WE WANT MORE TYPES OF OBSERVERS

        if ("observer_type" in gasspy_config.keys() and gasspy_config["observer_type"] == "healpix"):
            flux = raytracer.global_rays.get_field("photon_count_%d"%iband, index = leaf_rays).get()
        else:
            flux = raytracer.global_rays.get_field("photon_count_%d"%iband, index = leaf_rays).get()/gasspy_config["sim_unit_length"]**2

        grp.create_dataset("photon_flux_%d"%iband, data = np.log10(flux).astype(np.float16))

        if args.save_opacity:
            # final opacity
            grp.create_dataset("optical_depth_%d"%iband, data = raytracer.global_rays.get_field("optical_depth_%d", index = leaf_rays).get())
        maxflux[iband] = max(maxflux[iband], np.max(flux))
        
    del observer
for iband in range(len(energy_limits)):
    h5File.attrs.create("max_flux_%d"%iband, data = maxflux[iband])



