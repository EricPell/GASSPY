import numpy as np
import h5py as hp
from mpi4py import MPI
import pathlib
import argparse
import os
import gasspy
from gasspy.physics.databasing.database_generator import DatabaseGenerator
from gasspy.physics.databasing.database_populator import DatabasePopulator
from gasspy.physics.sourcefunction_database.cloudy import CloudyModelRunner
from gasspy.shared_utils.mpi_utils.mpi_os import mpi_makedirs
from simulation_reader import Simulation_Reader
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.Get_size()
gasspy_path = str(pathlib.Path(gasspy.__file__).resolve().parent)

ap = argparse.ArgumentParser()
ap.add_argument("cloudy_path")
args = ap.parse_args()

testdir = gasspy_path + "/scripts/database_building/database_test/"
modeldir = gasspy_path + "/scripts/database_building/database_test/database/"
if not os.path.exists(modeldir):
    mpi_makedirs(modeldir)
os.chdir(testdir)
database_name = "test_database.hdf5"
fluxdef_file = gasspy_path + "/scripts/database_building/database_test/gasspy_fluxdef.yaml"

gasspy_config = {
    "database_fields" :[ 
        "number_density",
        "temperature",
        "cell_size",
        "ionizing_flux",
        "FUV_flux"
    ],
    "compression_ratio": {
        "number_density"  :  [1, 2],
        "temperature"  :  [1, 2],
        "cell_size"    :  [1, 2],
        "ionizing_flux" :   [1, 2],
        "FUV_flux" :   [1, 2]
    },
    "log10_field_limits":{
        "temp" : {
            "max": 7
        },
        "flux" : {
            "min": -8,
            "min_cutoff_value" : -99
        }
    },
    "interpolate_fields": [
        "number_density",
        "temperature"
    ],
    "cloudy_ini" : gasspy_path + "/physics/sourcefunction_database/cloudy/init_files/spec_postprocess_atomic_no_qheat-c17.ini",
    "cloudy_path" : args.cloudy_path
}


def get_simulation(n_cells):
    sim_data = {}
    # number density between 1e3 and 2e3 cm^-3
    sim_data["number_density"] = 1e3*(np.random.rand(n_cells) +1)
    # temperature between 1e4 and 2e4 K
    sim_data["temperature"]    = 1e4*(np.random.rand(n_cells) +1)
    # sizes all same of 1e17 cm
    sim_data["cell_size"]      = np.full(n_cells, 1e17)
    # ionizing flux between 1e12 and 2e12
    sim_data["ionizing_flux"]  = 1e12*(np.random.rand(n_cells) +1)
    # FUV flux between 1e12 and 2e12
    sim_data["FUV_flux"]  = 1e12*(np.random.rand(n_cells) +1)

    sim_reader = Simulation_Reader(sim_data)
    return sim_reader

if mpi_rank == 0:
    # Start with 10 cells
    sim_reader = get_simulation(10)
    database_generator = DatabaseGenerator(gasspy_config, 
                                           database_name=database_name,
                                           gasspy_modeldir= modeldir)
    database_generator.add_snapshot(sim_reader)
    database_generator.finalize()

mpi_comm.barrier()
model_runner = CloudyModelRunner(gasspy_config, modeldir+"rundir/", fluxdef_file=fluxdef_file)
database_populator = DatabasePopulator(gasspy_config, model_runner, 
                                       database_name = database_name,
                                       gasspy_modeldir = modeldir)
database_populator.run_models()
database_populator.finalize()



