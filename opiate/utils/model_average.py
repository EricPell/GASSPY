"""calculate the average emissivity and opacity of a cell"""
import glob
import os
import sys
import __model_average__
from multiprocessing import Pool
import pandas
nprocs = 32

if os.path.isfile(sys.argv[1]):
    if sys.argv[1].endswith(".ems"):
        __model_average__.average_emissivity(sys.argv[1])
    
    elif sys.argv[1].endswith(".opt"):
        __model_average__.average_opacity(sys.argv[1])


else:
    # If the provided file is a regexp ending in ems, search for matches since provided path is not a file:

    unique_pdf = pandas.read_pickle("/".join(sys.argv[1].split("/")[:-2])+"/opiate_unique.pkl")


    #Provided string matches only to worker_list ending with ems
    if sys.argv[1].endswith(".ems"):

        worker_list = glob.glob(sys.argv[1])
        worker_list = [[file, unique_pdf] for file in worker_list]
        with Pool(nprocs) as p:
            p.starmap(__model_average__.average_emissivity, worker_list)      

    # or provided string matches only to worker_list ending with opt
    elif sys.argv[1].endswith(".opt"):
        worker_list = glob.glob(sys.argv[1])
        worker_list = [[file] for file in worker_list]
        with Pool(nprocs) as p:
            p.starmap(__model_average__.average_opacity, worker_list)

    # Or search string matches neither, and we should search for both.
    else:
        # Get all the files
        worker_list = glob.glob(sys.argv[1]+"*.ems")

        #Make the dictionary
        df = pandas.read_csv(worker_list[0], delimiter="\t")

        # Make a list of each file to work with starmap. I anticipate other parameters in the future, like a dictionary to write to.
        worker_list = [[file, unique_pdf] for file in worker_list]
        with Pool(nprocs) as p:
            dicts = p.starmap(__model_average__.average_emissivity, worker_list)      

        lines = dicts[0].keys()
        emissivity_dict = {}
        for line in lines:
            emissivity_dict[line] = {}
            for i in range(len(dicts)):
                for key in dicts[i][line].keys():
                    emissivity_dict[line][key] = dicts[i][line][key]
        worker_list = glob.glob(sys.argv[1]+"*.opt")
        worker_list = [[file] for file in worker_list]
        with Pool(nprocs) as p:
            p.starmap(__model_average__.average_opacity, worker_list)
