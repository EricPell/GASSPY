"""calculate the average emissivity and opacity of a cell"""
import glob
import os
import sys
from gasspy.utils import __model_average__
from multiprocessing import Pool
import pandas
import pickle
nprocs = 32

# If the provided file is a regexp ending in ems, search for matches since provided path is not a file:
if len(sys.argv) >= 2:
    unique_pdf = pandas.read_pickle(sys.argv[2])
storage_keys = [tuple(unique_pdf.iloc[index]) for index in range(len(unique_pdf))]

# VS Code hack for debugging. Add an escape charecter to the wild card, and replace if present
sys.argv[1] = sys.argv[1].replace("\\","")

#Provided string matches only to worker_list ending with ems
for search_string in sys.argv[1].split(","):
    if search_string.endswith(".ems"):
        file_list = glob.glob(search_string)
        avg_emissvity_dict = __model_average__.build_avgem_dict(file_list,storage_keys)
        
        outfile = open(sys.argv[2].replace(".pkl","_avge_missivity_dictionary.pkl"),'wb')
        pickle.dump(avg_emissvity_dict,outfile)
        outfile.close()
        pass

# # or provided string matches only to worker_list ending with opt
#     elif search_string.endswith(".opt"):
#         worker_list = glob.glob(sys.argv[1])
#         worker_list = [[file] for file in worker_list]
#         with Pool(nprocs) as p:
#             avg_opacity_dict = p.starmap(__model_average__.build_avgopac_dict(file_list,storage_keys)
