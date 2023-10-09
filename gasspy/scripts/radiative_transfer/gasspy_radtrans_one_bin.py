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
ap.add_argument("gasspy_config", default="./gasspy_config.yaml")
ap.add_argument("outfile", default = None, help="name of trace file. If it does not exist we need to recreate it")
ap.add_argument("--save_opacity", action = "store_true", help = "Save total opacity along with flux")

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
    sys.exit("ERROR : cant find snapshot specific gasspy sub directory %s."%gasspy_subdir)

if not os.path.exists(gasspy_subdir+"/single_bands/"):
    os.makedirs(gasspy_subdir+"/single_bands/")

outfile = gasspy_subdir+"/single_bands/" + args.outfile 
##############################
# II) Load config files and simulation reader 
##############################

## Simulation reader directory
simulation_readerdir = gasspy_io.check_parameter_in_config(gasspy_config, "simulation_reader_dir", None, "./")

## Load the simulation data class from directory
spec = importlib.util.spec_from_file_location("simulation_reader", simulation_readerdir + "/simulation_reader.py")
reader_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader_mod)
sim_reader = reader_mod.Simulation_Reader(gasspy_config, snaphot)

###########################
# III) figure out observers and energy bins
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
energy_limits = np.atleast_2d(gasspy_config["energy_limits"])

# Load the data for the lines (if any)
line_data = None
if "line_data" in gasspy_config.keys():
    line_data = gasspy_config["line_data"]

# Load the data for the broadbands (if any)
bband_data = None
if "bband_data" in gasspy_config.keys():
    bband_data = gasspy_config["bband_data"]

###########################
# IV) Initialize hdf5 file with output rays
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
raytracer = Raytracer_AMR_neighbor(gasspy_config, sim_reader)
print("Initializing ray_processor")
ray_processor = Single_band_radiative_transfer(gasspy_config, raytracer, sim_reader)
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
    raytracer.update_observer(observer = observer)
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
        flux = raytracer.global_rays.get_field("photon_count_%d"%iband, index = leaf_rays).get()

        grp.create_dataset("photon_flux_%d"%iband, data = np.log10(flux).astype(np.float16))

        if args.save_opacity:
            # final opacity
            grp.create_dataset("optical_depth_%d"%iband, data = raytracer.global_rays.get_field("optical_depth_%d", index = leaf_rays).get())
        maxflux[iband] = max(maxflux[iband], np.max(flux))
        
    del observer
for iband in range(len(energy_limits)):
    h5File.attrs.create("max_flux_%d"%iband, data = maxflux[iband])



