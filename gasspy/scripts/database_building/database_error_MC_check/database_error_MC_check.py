import gasspy
from gasspy.physics.sourcefunction_database.cloudy.cloudy_model_runner import CloudyModelRunner
from gasspy.shared_utils.mpi_utils.mpi_print import mpi_print, mpi_all_print
from gasspy.io.gasspy_io import read_yaml, save_dict_hdf5, check_parameter_in_config
from gasspy.scripts.database_building.database_error_MC_check.cell_database_populator import CellDatabasePopulator
from mpi4py import MPI
import sys
import os
import h5py as hp
import numpy as np 
import astropy.units as apyu
import astropy.constants as apyc
import argparse
import importlib
import matplotlib.pyplot as plt
import time
ap = argparse.ArgumentParser()
ap.add_argument("--gasspy_config", default = "./gasspy_config.yaml")
ap.add_argument("--fluxdef", default = "./gasspy_fluxdef.yaml")
ap.add_argument("--cells_outfile", default = "./MC_cells.hdf5")
ap.add_argument("--rundir", default = "./cloudy_output/")
ap.add_argument("--simulation_reader_dir", default="./", help="directory to the simulation_reader class that describes how to load the simulation")
ap.add_argument("--max_walltime", default = None, type = float)
args = ap.parse_args()

wall_start = time.time()
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()


gasspy_config = read_yaml(args.gasspy_config)
fluxdef = args.fluxdef
indir = args.rundir


    
"""
    Model generation and analyzis
"""



class Unified_Simulation_Reader:
    def __init__(self, gasspy_config, h5outfile, sim_readers):
        self.sim_readers = sim_readers
        self.sim_N_cells = np.zeros(len(sim_readers), dtype=int)
        for ireader, reader in enumerate(self.sim_readers):
            self.sim_N_cells[ireader] = len(reader.get_field("density"))
        
        self.sim_icell = np.cumsum(self.sim_N_cells) - self.sim_N_cells
        self.tot_N_cells = np.sum(self.sim_N_cells)
        self.database_fields = gasspy_config["database_fields"]
        self.models = None
        self.h5outfile = h5outfile
        if os.path.exists(self.h5outfile):
            h5out = hp.File(self.h5outfile, "r+")  
        else:
            h5out = hp.File(self.h5outfile, "w")
        if "database_fields" not in h5out:
            h5out["database_fields"] = self.database_fields
        h5out.close()
        return

    def append_cells(self, N_cells):
        cell_index = np.arange(N_cells)
        # Grab random cells
        icells = np.random.randint(self.tot_N_cells, size = N_cells)
        # Determine which reader they correspond to
        ireaders = np.searchsorted(self.sim_icell, icells, side="right") - 1
        # allocate storage for their model data
        models = np.zeros((N_cells, len(self.database_fields)))

        # Make sure that there's enough allocated in the hdf5 output
        h5out = hp.File(self.h5outfile, "r+")
        if not "model_data" in h5out:
            N_previous = 0
            h5out.create_dataset("model_data", shape = (N_cells, len(self.database_fields)), maxshape = (None, len(self.database_fields)), dtype=np.float64)
        else:
            N_previous = h5out["model_data"].shape[0]
            cell_index+=N_previous
            h5out["model_data"].resize((N_previous+N_cells), axis = 0)
        
        # get the models from the different readers
        for ireader, reader in enumerate(self.sim_readers):
            cells_now = np.where(ireaders == ireader)[0]
            for ifield, field in enumerate(self.database_fields):
                models[cells_now, ifield] = reader.get_field(field)[icells[cells_now] - self.sim_icell[ireader]].astype(np.float64)

        models[models <= 0] = 1e-40
        # Save the the hdf5
        h5out["model_data"][N_previous:,:] = np.log10(models)
        h5out.close()



    def get_field(self, field):
        ifield = self.database_fields.index(field)
        h5out = hp.File(self.h5outfile, "r+")
        data = 10**h5out["model_data"][:,ifield]
        h5out.close()
        return data
    
    def save_new_field(self, field, data, dtype = None):
        if dtype is None:
            dtype = data.dtype
        h5out = hp.File(self.h5outfile, "r+")
        if field in h5out:
            if data.shape[0] > h5out[field].shape[0]:
                h5out[field].resize(data.shape[0], axis = 0)
        else:
            h5out.create_dataset(field, shape = data.shape, maxshape = (None,), dtype = dtype)

        h5out[field][:] = data.astype(dtype)
        h5out.close()

def get_line_indexes(line_labels, spec_energy):
    line_indexes = np.zeros(len(line_labels), dtype = int)
    Ryd_Ang = 911.2266

    for iline, line_label in enumerate(line_labels):
        wav_str = line_label.split(" ")[-1] 
        if wav_str.endswith("A"):
            energy = Ryd_Ang/(float(wav_str.strip("A")))
        else:
            energy = Ryd_Ang/(float(wav_str.strip("m"))*1e4)
        line_indexes[iline] = np.argmin(np.abs(energy-spec_energy))
    return line_indexes


model_runner = CloudyModelRunner(gasspy_config, indir, fluxdef)

# initialize database creator
database_creator = gasspy.DatabaseCreator(gasspy_config, model_runner)

if mpi_rank == 0:
    ## Load the simulation data class from directory
    spec = importlib.util.spec_from_file_location("simulation_reader", args.simulation_reader_dir + "/simulation_reader.py")
    reader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader_mod)

    # Create a list of readers for all required snapshots
    sim_readers = []
    for key in gasspy_config["snapshots"]:
        snapshot = gasspy_config["snapshots"][key]
        sim_readers.append(reader_mod.Simulation_Reader(snapshot["simdir"], snapshot["simdir"]+"/GASSPY", snapshot["sim_args"]))

    sim_reader = Unified_Simulation_Reader(gasspy_config, args.cells_outfile, sim_readers)
    N_cells = gasspy_config["N_cells"]
    sim_reader.append_cells(N_cells)
    
else:
    h5out = None
    sim_reader = None


max_walltime = check_parameter_in_config(gasspy_config, "max_walltime", args.max_walltime, 1e99)

# Start by adding to the database of snapshots so that it is stored
database_creator.add_snapshot(sim_reader)

# Next run all models for the actual cells
populator = CellDatabasePopulator(gasspy_config, model_runner, gasspy_modeldir="./", database_name=args.cells_outfile)
populator.set_max_walltime(max_walltime - (time.time()- wall_start))
populator.run_models()

# Finally run the actual models
database_creator.set_max_walltime(max_walltime - (time.time()- wall_start))
database_creator.run_models()
database_creator.finalize()




