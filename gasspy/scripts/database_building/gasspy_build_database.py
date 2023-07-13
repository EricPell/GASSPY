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
import numpy as np
import argparse
import importlib.util
from mpi4py import MPI
import time

import gasspy
from gasspy.physics.sourcefunction_database.cloudy import generate_mesh, CloudyModelRunner
from gasspy.io import gasspy_io
from gasspy.shared_utils.mpi_utils.mpi_os import mpi_makedirs
from gasspy.shared_utils.mpi_utils.mpi_print import mpi_print

start_time = time.time()
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.size

ap = argparse.ArgumentParser()
#-------------DIRECTORIES AND FILES---------------#
ap.add_argument("--gasspy_config", default="./gasspy_config.yaml", help = "Path to gasspy config yaml file")
ap.add_argument("--fluxdef", default="./gasspy_fluxdef.yaml", help = "Path to flux definition yaml file")
ap.add_argument("--rundir", default = "./rundir/", help = "Path to directory where models are to be run")
ap.add_argument("--simulation_reader_dir", default="./", help="directory to the simulation_reader class that describes how to load the simulation")
#-------------Run parameters-----------#
ap.add_argument("--recompile_cloudy_spectra_mesh", action="store_true", help="Recompile the Cloudy with the defined spectra mesh")
ap.add_argument("--cloudy_spectra_mesh_mode", default="select_C17", help="What list of lines and windows do we draw from (NOTE: only one option implemented) \n\tselect_C17 lines from physics/sourcefunction_database/cloudy/select_cloudy_lines")
ap.add_argument("--cloudy_compile_ncores", default=1, type = int, help="how many threads should be used to compile Cloudy")
ap.add_argument("--max_walltime", default=1e99, type = float, help = "Maximum walltime for run")
args = ap.parse_args()

##############################
# I ) Load config files and simulation reader 
##############################
## Load the gasspy_config yaml
gasspy_config = gasspy_io.read_yaml(args.gasspy_config)

## Load fluxdef
fluxdef = gasspy_io.read_yaml(args.fluxdef)

## Load the simulation reader class from directory
spec = importlib.util.spec_from_file_location("simulation_reader", args.simulation_reader_dir + "/simulation_reader.py")
reader_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader_mod)

if not os.path.exists(gasspy_config["gasspy_modeldir"]):
    mpi_makedirs(gasspy_config["gasspy_modeldir"])

# Create a simulation reader for each snapshot 
sim_readers = []
for snapshot in gasspy_config["snapshots"]:
    if mpi_rank == 0:
        sim_readers.append(reader_mod.Simulation_Reader(gasspy_config, gasspy_config["snapshots"][snapshot]))
    else:
        sim_readers.append(None)

###########################################
# II) Cloudy recompilation with specified spectra mesh
###########################################

if args.recompile_cloudy_spectra_mesh and mpi_rank == 0:
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
        generator = generate_mesh.mesh_generator(nproc = args.cloudy_compile_ncores, cloudy_data_dir=gasspy_config["cloudy_path"] + "/data/") 
        generator.regrid(E0, delta, R)




##############################
# IV) Initialize the database creator
##############################

model_runner = CloudyModelRunner(gasspy_config, args.rundir, fluxdef)

# initialize database creator
database_creator = gasspy.DatabaseCreator(gasspy_config, model_runner)

for sim_reader in sim_readers:
    database_creator.add_snapshot(sim_reader)

#############################
# V) Run all required models
#############################
mpi_comm.barrier()
new_maxtime = args.max_walltime - (time.time()- start_time)
database_creator.set_max_walltime(new_maxtime)
database_creator.run_models()

database_creator.finalize()
