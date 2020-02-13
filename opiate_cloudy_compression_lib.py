""" Read a model and collect it's data into a single binary file """
from __future__ import print_function
import sys
import os.path
import numpy as np
from astropy.io import ascii as ascii
from astropy import units as u
from pathlib import Path
import gc
import time
import pickle
import bz2

default_file_types = ["ems"]
default_prefixes = ["opiate"]

def find_data_files(root, file_types=None, prefixes=None, recursive=True):
    import glob
    found_files = []

    if file_types is None:
        file_types = default_file_types

    if prefixes is None:
        prefixes = default_prefixes

    elif prefixes is False:
        prefixes = [""]

    for file_type in file_types:
        for prefix in prefixes:
            found = glob.glob(root+prefix+"*."+file_type, recursive=recursive)
            found_files = found_files + found

    print(root+": found %i files"%len(found_files))
    return(found_files)        

def read_data(file_list, raw_data_dict=None, guess=False):
    if raw_data_dict is None:
        # If dict_store is None, then we are not appending to an existing dictionary, and must create a new one.
        raw_data_dict = {}
        table_is_touched = True

    else:
        table_is_touched = False

    for fname in file_list:

        if os.path.isfile(fname):
            # Get the ending of the file by splitting on '.' and taking the last object.
            suffix = fname.split(".")[-1]

            # Check that the split worked.
            if len(suffix) >0:
                # Get the root of the file, including path by taking the string characters from the beginning to the end-len("."+suffix) = len(suffix)+1
                root = fname[:-(len(suffix)+1)]

                # This is for performance, despite the fact that I have exceptions. They are hard to debug.
                # Check if this model directory has already been read into the dictionary. 
                # If not, read it in.
                try:
                    raw_data_dict[root][suffix]

                except:

                    try:
                        table = ascii.read(fname,format="commented_header", delimiter="\t", guess=guess)

                    except:
                        table = "read error"
                        print("%s.%s read error"%(root,suffix))

                    # Read data into dictionary         
                    # Check if this root already exists in the dictionary, and create new level if not
                    try:
                        raw_data_dict[root][suffix] = table

                    except:
                        raw_data_dict[root] = {}
                        raw_data_dict[root][suffix] = table


                    table_is_touched = True

            else:
                sys.exit("File with no ending suffix found. File:%s"%(fname))
    
    return(raw_data_dict, table_is_touched)

def merge_data(raw_data_dict, reduced_data_dict=None):
    from astropy.table import Table, dstack
    
    # If reduced_data_dict is None we are not appending to an existing dictionary, and must create a new one.
    reduced_data_dict = {}

    #Loop over all top levels of the dictionary. These are defined by the "root" given to create the dictionary.
    for root in raw_data_dict.keys():
        reduced_data_dict[root] 

def save_dict(dictionary, outfile, compress_output=True):
    #Save the cloud models in a dictionary using pickle.

    Path(os.path.dirname(outfile)).mkdir(parents=True, exist_ok=True)

    if compress_output:
        fh_out = bz2.BZ2File(outfile, "wb")
    else:
        fh_out = open(outfile, "wb")

    pickle.dump(dictionary, fh_out)
    fh_out.close()

    return(True)

def worker_compress_cloudy_dir(data_dir, lock1, lock2, existing_store=False, bz2comp=True, save_dir=False, save_name=False):
    """
    Run all processes to compress a group of files, suitable for threading.
    """
    if save_dir is False:
        save_dir = "./"+data_dir
    if save_name is False:
        save_name = "cloudy_struct_models"
    pickle_file_name = save_dir+"%s.pckl"%(save_name)

    # The flag "existing_store" means check for an existing storage pickle, and if
    # it exists, open it. Setting to false is equivlant to overwrite.

    if existing_store and os.path.isfile(pickle_file_name):
        # try:
        #     # print ("Trying to read existing store")
        #     if bz2comp:
        #         store = pickle.load(bz2.open(pickle_file_name,"rb"))
        #     else:
        #         store = pickle.load(open(pickle_file_name, 'rb'))
        #     # print ("existing store succesfully read")
        # except:
        #     store = {}
        table_modified = False
    else:
        # Waiting is OVER! It's ok if more than one process reads at a time so we don't need to lock
        with lock1:
            datafiles = find_data_files(data_dir)

        # print("No existing store")
        store = {}

        with lock2:
            store, table_modified = read_data(datafiles,raw_data_dict=store)

    # Writing is done to an SSD, so it's actually more efficient to write randomly
    # so I don't care if multiple threads write their pickles at the same time. 
    # There should be a flag for both to allow the user to set this appropriately
    if table_modified:
        # print ("Table modified.")
        save_dict(store,pickle_file_name, compress_output=bz2comp)
    # else:
    #     # print("Table unmodified; not saving")

    del(store)
    # gc.collect()

def average_emissivities(x, y, N_levels):
    """ 
    take a profile with independent axis X and values Y, and calculate N averages
    """
    max_x = np.max(x)

    depth = [ 0.5**level*max_x for level in N_levels]

    avg_y = np.zeros(N_levels)

    dx = np.zeros(len(x))
    dx[0] = x[0]
    dx[1:] = x[1:] - x[:-1]

    # Initialize the index of of x to the maximum value.
    stop_x_i = len(x) - 1

    # Calculate the volume averaged emissivity of the whole cell.
    avg_y[0] = np.sum(dx * y) / x[-1]

    # Next for each refignment level cut the cell in half.
    for level in range(1, N_levels):
        # Calculate desired depth.
        stop_x = 0.5**level * max_x

        # Find the index where the depth is closest to the desired depth
        stop_x_i = np.argmin( np.abs(x[:stop_x_i]-stop_x) )

        # Calculate the volume averaged y
        avg_y[level] = np.sum(dx[:stop_x_i] * y[:stop_x_i]) / x[stop_x_i]
        depth[level] = stop_x

    return(depth, avg_y)    
