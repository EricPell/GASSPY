"""calculate the average emissivity and opacity of a cell"""
import numpy
import pandas
import os
from multiprocessing import Pool

def average_emissivity(file_name, storage_key=None):
    #Unique pdf is a pandas data frame that contains a list of values that will be used as a key in the dictionary.

    avg_file = file_name+".avg"

    if storage_key is None:
        index = int(file_name.split("/")[-1].replace(".ems","").replace("opiate-",""))
        storage_key = (index,)

    df = pandas.read_csv(file_name, delimiter="\t")

    emissivity_dictionary = {}
    for column in df.columns:
        if column != "#depth":
            emissivity_dictionary[column] = {}

    if df.shape[0]==1:
        df = df[df.columns[1:]].squeeze()
    elif df.shape[0]>1:
        df['dr'] = df['#depth'].diff(periods=1)
        df['dr'][0] = df['#depth'][0]
        
        # Get the last element of the first column which is total depth
        total_depth = df.iloc[-1,0]

        #Depth and dr are first and last. Calculate the product of the emissivity
        # This is a messy one liner, so lets take it step by step.
        # First, access the colums of df that are emission lines, which excludes the 0th and last (-1) column
        # Multiply these by the "dr" column to get a differential flux per depth element.
        # "sum" this to get the total flux per cell, and divide by the total path length to get the volume emissivity
        df = df[df.columns[1:-1]].multiply(df['dr'], axis="index").sum(axis="index")/total_depth
        pass

    # Save the file
    for line in emissivity_dictionary:
        emissivity_dictionary[line][storage_key] = df[line]
    df.to_csv(avg_file)
    return(emissivity_dictionary)

def build_avgem_dict(file_list, storage_keys,nprocs=None):
    if nprocs == None:
        nprocs = min(len(os.sched_getaffinity(0)),len(file_list))
    else:
        assert(type(nprocs)==int)
        assert(nprocs>0)
    #Make the dictionary
    df = pandas.read_csv(file_list[0], delimiter="\t")

    # Make a list of each file to work with starmap. I anticipate other parameters in the future, like a dictionary to write to.
    worker_list = [[file_list[i], storage_keys[i]] for i in range(len(file_list))]
    with Pool(nprocs) as p:
        dicts = p.starmap(average_emissivity, worker_list)      

    lines = dicts[0].keys()
    emissivity_dict = {}
    for line in lines:
        emissivity_dict[line] = {}
        for i in range(len(dicts)):
            for key in dicts[i][line].keys():
                emissivity_dict[line][key] = dicts[i][line][key]

    return(emissivity_dict)

def average_opacity(file_name):
    df = pandas.read_csv(file_name, delimiter="\t")
    if df.shape[0]>=1:
        #Should check if 
        pass

if __name__ == 'main':
    average_emissivity(None)
    average_opacity(None)
    pass