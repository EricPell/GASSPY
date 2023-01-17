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

import time
import os
import numpy as np
from astropy.io import fits
import argparse
import importlib.util

from gasspy.physics.sourcefunction_database.cloudy import gasspy_cloudy_db_classes, generate_mesh, cloudy_run, cloudy_model_collector
from gasspy.io import gasspy_io

ap = argparse.ArgumentParser()
#-------------DIRECTORIES AND FILES---------------#
ap.add_argument("--simdir", default="./", help="Directory of the simulation and also default work directory")
ap.add_argument("--workdir", default= None, help="work directory. If not specified its the same as simdir")
ap.add_argument("--gasspydir", default="GASSPY", help="directory inside of simdir to put the GASSPY files")
ap.add_argument("--modeldir" , default="GASSPY", help = "directory inside of workdir where to read, put and run the cloudy models")
ap.add_argument("--cloudy_path", default = None, help="Path to the cloudy installation. If not specified, use environment variable 'CLOUDY_PATH'")
ap.add_argument("--simulation_reader_dir", default="./", help="directory to the simulation_reader class that describes how to load the simulation")
ap.add_argument("--sim_prefix", default = None, help="prefix to put before all snapshot specific files")
#-------------Run parameters-----------#
ap.add_argument("--recompile_cloudy_spectra_mesh", action="store_true", help="Recompile the cloudy with the defined spectra mesh")
ap.add_argument("--cloudy_spectra_mesh_mode", default="select_C17", help="What list of lines and windows do we draw from (NOTE: only one option implemented) \n\tselect_C17 lines from physics/sourcefunction_database/cloudy/select_cloudy_lines")
ap.add_argument("--Ncores", default=1, type=int,help="Number of processes to use")

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


if not os.path.exists(args.modeldir):
    os.makedirs(args.modeldir)

## unpack the cloudy path
if args.cloudy_path is None:
    cloudy_path = os.environ["CLOUDY_PATH"]
else:
    cloudy_path = args.cloudy_path
## unpack number of processes to use
Ncores = args.Ncores

## set prefix to snapshot specific files
if args.sim_prefix is not None:
    ## add an underscore
    sim_prefix = args.sim_prefix + "_"
else:
    sim_prefix = ""
###########################################
# II) Cloudy recompilation with specified spectra mesh
###########################################

if args.recompile_cloudy_spectra_mesh:
    Ryd_Ang = 911.2266
    if args.cloudy_spectra_mesh_mode == "select_C17":
        # Angstrom to rydbergs
        Ryd_Ang = 911.2266
        # Load the select cloudy lines
        from gasspy.physics.sourcefunction_database.cloudy import select_cloudy_lines 
        labels = select_cloudy_lines.labels()

        # Grab the wavelengths
        label_list = [label.split(" ")[-1] for label in list(labels.line.keys())]

        # Convert the wavelengths into energies
        E0 = np.zeros(len(label_list))
        for i, label in enumerate(label_list):
            if label.endswith("A"):
                E0[i] = Ryd_Ang / float(label.strip("A"))
            elif label.endswith("m"):
                E0[i] = Ryd_Ang / (float(label.strip("m"))*1e4)

        # Size of the high resolution window
        delta = E0 * 0.5*(1/(1-1000.0/3e5) - 1/(1+1000/3e5))
        # Resolving power
        R = 10000

        # Create the mesh generator and recompile cloudy
        generator = generate_mesh.mesh_generator(nproc = Ncores, cloudy_data_dir=cloudy_path + "/data/") 
        generator.regrid(E0, delta, R)
    

##############################
# III) Load config files and simulation reader 
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



###############################
# IV) Build the database
###############################
print("\nBuilding database of models")
## initialize the creator class
creator = gasspy_cloudy_db_classes.uniq_dict_creator(
    save_compressed3d = True,
    gasspy_modeldir = args.modeldir,
    gasspy_subdir = args.gasspydir,
    sim_prefix = sim_prefix
)

## load the required fields
creator.simdata = {
    "temp": np.log10(sim_reader.get_field("temp")),
    "dens": np.log10(sim_reader.get_number_density()),
    "dx"  : np.log10(sim_reader.get_field("dx")), 
    "fluxes" : fluxdef
}
## load the fluxes of the radiation fields
for field in fluxdef.keys():
    creator.simdata["fluxes"][field]["data"] = np.log10(np.maximum(sim_reader.get_field(field),1e-40))    

## Specify the cutoff limit for the fluxes
creator.log10_flux_low_limit = {}
for field in fluxdef.keys():
    assert field in gasspy_config["log10_flux_low_limit"].keys(), "log10_flux_low_limit not defined in gasspy_config.yaml for field %s"%field
    creator.log10_flux_low_limit[field] = gasspy_config["log10_flux_low_limit"][field]

## Specify the compression ratios
creator.compression_ratio = {}
for field in ["temp", "dens", "dx"]:
    assert field in gasspy_config["compression_ratio"].keys(), "Compression ratio not defined in gasspy_config.yaml for field %s"%field
    creator.compression_ratio[field] = (gasspy_config["compression_ratio"][field][0], gasspy_config["compression_ratio"][field][1])
# for the fluxes
creator.compression_ratio["fluxes"] = {}
assert "fluxes" in gasspy_config["compression_ratio"].keys(), "fluxes must be specified as their own dictionary inside of the compression ratio dictionary"
for field in fluxdef.keys():
    assert field in gasspy_config["compression_ratio"]["fluxes"].keys(), "Compression ratio not defined in gasspy_config.yaml for flux field %s"%field
    creator.compression_ratio["fluxes"][field] = (gasspy_config["compression_ratio"]["fluxes"][field][0], gasspy_config["compression_ratio"]["fluxes"][field][1])

## compress and trim the simulation data
print(" - Compressing simulation data")
creator.process_simdata()

## Initialize the gasspy_to_cloudy converter
gasspy_to_cloudy = gasspy_cloudy_db_classes.gasspy_to_cloudy(gasspy_config, gasspy_modeldir=creator.gasspy_modeldir, CLOUDY_INIT_FILE="spec_postprocess-c17.ini")
## Process the trimmed simulation data
print(" - Creating cloudy .in files")
gasspy_to_cloudy.process_grid()

## Do some cleanup
del(creator)
del(gasspy_to_cloudy)

################################
# V) Run the cloudy models
################################
print("\nStarting the run on the grid of cloudy models")

## Make a class that takes a dictionary and takes the entries as attributes
# the cloudy run process is coded for using and argparse argument parser, which we need to recreate. Easiest is to just make a class
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

cloudy_run_dict = {
    "Ncores": Ncores,
    "indirs": [args.modeldir + "/cloudy-output/"],
    "cloudy_path": args.cloudy_path,
    "log" : True
}
cloudy_run_args = AttrDict(cloudy_run_dict)


## intitialize the process class
starttime = time.time()
processor = cloudy_run.processor_class(cloudy_run_args)
exec_list = processor.preproc(starttime)

## Start threads to run all needed models
if len(exec_list) > 0:
    processor.pool_handler(exec_list)
    endtime = time.time()

## print to stats to logfile
if not processor.args.log:
    if not processor.args.log:
        print("Total threads used: %i"%min(processor.args.Ncores, len(exec_list)))
        print("Total models calculated: %i"%len(exec_list))
        print("Total time  (s): %0.2f"%(endtime-starttime))
        print("Models / second : %0.4f"%( len(exec_list)/float(endtime-starttime)) )

    else:
        log_file = "cloudy_run.log"
        if os.path.exists(log_file):
            out = open(log_file,"a+")
            print("Append log...")
        else:
            print("New log...")
            out = open(log_file,"w+")
            out.writelines("\t".join(["N_cores","N_mods", "time(s)","mod/s\n"]))
        out.writelines("%i\t%i\t%0.2f\t%0.4f"%(min(processor.args.Ncores, len(exec_list)), len(exec_list),endtime-starttime, len(exec_list)/float(endtime-starttime)))
        out.close()

## clean up
del(processor)

####################################
#VI) Collect the models
####################################
profiling = True
if profiling:
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()
print("\nCollecting all the cloudy models into individual files")
collector = cloudy_model_collector.ModelCollector(cloudy_dir=args.modeldir + "/cloudy-output/", out_dir=args.modeldir,
                                                out_files = {
                                                                "avg_em": True,
                                                                "grn_opc": True,
                                                                "tot_opc": True,
                                                                "mol":True
                                                            }
                                                    )
collector.all_opacities = False
collector.clear_energy = False
collector.collect()
if profiling:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.dump_stats("cloudy_model_collector_prof")