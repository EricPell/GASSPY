import h5py as hp
import numpy as np
import argparse
import matplotlib.pyplot as plt
from gasspy.io.gasspy_io import read_dict_hdf5

ap = argparse.ArgumentParser()
ap.add_argument('--cell_values', nargs = "+")
ap.add_argument('--cell_gasspy_index', nargs = "+")
ap.add_argument('--h5_database', default="GASSPY_DATABASE/gasspy_database.hdf5")
ap.add_argument('--separate_by_error', action = "store_true")

args = ap.parse_args()

# Load the h5 database and find the models that failed
database = hp.File(args.h5_database, "r")
problem_dict = {}
read_dict_hdf5("problem_dict", problem_dict, database)

def get_problem_cells(problem_id, gasspy_id_file, compressed_values_file):
    compressed_values, cell_index = np.unique(np.load(compressed_values_file), axis = 0, return_index = True)
    gasspy_id = np.load(gasspy_id_file)[cell_index]
    is_wrong = np.isin(gasspy_id, problem_id)
    return compressed_values[np.where(is_wrong)[0],:]

def print_uniques(problem_values):
    # Make unique
    unique = np.unique(problem_values, axis = 0)
    print(unique)
def plot_uniques(problem_values):
    unique = np.unique(problem_values, axis = 0)
    labels = ["dx", "dens", "temp"]
    for ilabel in range(3, unique.shape[1]):
        print("flux_%d"%(ilabel-3))
        labels.append("flux_%d"%(ilabel-3))
    
    for ifield in range(unique.shape[1]):
        plt.hist(unique[:,ifield])
        plt.title(labels[ifield])
        plt.show()

cell_values_files = args.cell_values
if not isinstance(cell_values_files, list):
    cell_values_files = [cell_values_files]
cell_gasspy_index_files  = args.cell_gasspy_index
if not isinstance(cell_gasspy_index_files, list):
    cell_gasspy_index_files = [cell_gasspy_index_files]

if args.separate_by_error:
    for key in problem_dict.keys:
        problem_id = np.array(problem_dict[key][...])
        i = 0
        problem_values = get_problem_cells(problem_id, cell_gasspy_index_files[i], cell_values_files[i])
        if len(args.cell_values) > 1:
            for i in range(1, args.cell_values):
                problem_values = np.append(problem_values, get_problem_cells(problem_id, cell_gasspy_index_files[i], cell_values_files[i]), axis = 0)
        
        print_uniques(problem_values)
else:
    problem_id = np.array([], dtype = int)
    for i, key in enumerate(problem_dict.keys()):
        problem_id = np.append(problem_id, problem_dict[key][...])
    problem_id = np.unique(np.array(problem_id))

    i = 0
    problem_values = get_problem_cells(problem_id, cell_gasspy_index_files[i], cell_values_files[i])
    if len(args.cell_values) > 1:
        for i in range(1, args.cell_values):
            problem_values = np.append(problem_values, get_problem_cells(problem_id, cell_gasspy_index_files[i], cell_values_files[i]), axis = 0)

    plot_uniques(problem_values)