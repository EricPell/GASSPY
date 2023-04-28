import numpy as np
import pathlib
import argparse
import os
import gasspy
from gasspy.physics.sourcefunction_database.cloudy import CloudyModelRunner
from gasspy.shared_utils.mpi_utils.mpi_os import mpi_makedirs
from simulation_reader import Simulation_Reader

gasspy_path = str(pathlib.Path(gasspy.__file__).resolve().parent)

ap = argparse.ArgumentParser()
ap.add_argument("cloudy_path")
ap.add_argument("--lines_only", action= "store_true")
ap.add_argument("--n_models", type=int, default=100)

args = ap.parse_args()

testdir = gasspy_path + "/scripts/database_building/database_test/"
modeldir = gasspy_path + "/scripts/database_building/database_test/database/"
if not os.path.exists(modeldir):
    mpi_makedirs(modeldir)
os.chdir(testdir)
database_name = "test_database.hdf5"
fluxdef_file = gasspy_path + "/scripts/database_building/database_test/gasspy_fluxdef.yaml"

gasspy_config = {
    "database_name" : database_name,
    "gasspy_modeldir" : modeldir,
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
    "cloudy_path" : args.cloudy_path,

    "line_labels" : [
        "H  1 6562.80A",
        "H  1 4861.32A"],
    
    "populator_dump_time" : 600,
    "est_model_time" : 10

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
# initialize model_runner
model_runner = CloudyModelRunner(gasspy_config, modeldir+"rundir/", fluxdef_file=fluxdef_file, lines_only=args.lines_only)
# initialize database creator
database_creator = gasspy.DatabaseCreator(gasspy_config, model_runner)

# Add a snapshot of n_models cells to the database
sim_reader = get_simulation(args.n_models)
database_creator.add_snapshot(sim_reader)
database_creator.run_models()

# Add another 2 snapshot of n_models/2 models
sim_readers = [get_simulation(args.n_models//2), get_simulation(args.n_models//2)]
for sim_reader in sim_readers:
    database_creator.add_snapshot(sim_reader)
database_creator.run_models()

#Finalize
database_creator.finalize()
