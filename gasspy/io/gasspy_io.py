"""IO Routines"""
import yaml
import pickle
import numpy as np
import h5py as hp

def read_compressed3d(file):
    with open(file,'rb') as f:
        data = np.load(f)
    return(data)

def read_avg_em(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        return(data)

def read_dict(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        return(data)

def write_dict(MyDict, file):
    pickle.dump( MyDict, open( file, "wb" ) )

def read_yaml(file_path):
    with open(r'%s'%file_path) as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    return dictionary

def read_fluxdef(fluxdef_file):
    """
        Wrapper around read yaml file
    """
    fluxdef = read_yaml(fluxdef_file)
    return(fluxdef)

def read_gasspy_config(gasspy_config_file):
    """
        Wrapper around read yaml file
    """
    gasspy_config = read_yaml(gasspy_config_file)
    return(gasspy_config)

def save_value_to_group(key, val, grp):
    # If the entry is a dict, then create a sub group and save entries of the sub dictionary into the sub group
    if isinstance(val, dict):
        if key in grp.keys():
            sub_grp = grp[key]
            assert isinstance(sub_grp, hp.Group), "Trying to save %s as a group, but the h5instance already contains %s as a dataset"%(key,key)
        else:
            sub_grp = grp.create_group(key)
        for sub_key in val.keys():
            save_value_to_group(sub_key, val[sub_key], sub_grp)

    # Otherwise save them as datasets
    else:
        if key in grp.keys():
            del grp[key]
        grp[key] = val
    return

def read_value_to_dict(key, dict, grp):
    # If the entry is a group, create a sub dictionary and save entries of sub group into sub dictionary
    if isinstance(grp[key], hp.Group):
        sub_dict = {}
        sub_grp =  grp[key]
        for sub_key in sub_grp.keys():
            read_value_to_dict(sub_key, sub_dict, sub_grp)
        dict[key] = sub_dict

    # Otherwise its a dataset (hopefully)
    elif isinstance(grp[key], hp.Dataset):
        # If the dataset is scalar, the loading syntax is slightly different
        if grp[key].shape == ():
            val = grp[key][()]
        else:
            val = grp[key][:]

        # For whatever reason Strings are saved as bytes objects. Decode them
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        
        # Add entry to dictionary
        dict[key] = val
    else:
        print("key %s in group %s is neither a group or dataset. Ignoring"%(key, grp.name)) 
    return


def save_dict_hdf5(name, dict, h5id):
    #open the group
    if name not in h5id.keys():
        grp = h5id.create_group(name)
    else:
        grp = h5id[name]
    # Loop through all entries in the dict and save them to the group
    for key in dict.keys():
        save_value_to_group(key, dict[key], grp)

def read_dict_hdf5(name, dict, h5id):
    #Open the group
    grp = h5id[name]
    # Loop through all entries in the group and add them to the dict
    for key in grp.keys():
        read_value_to_dict(key, dict, grp)

def save_gasspy_config_hdf5(gasspy_config, h5file):
    """
        Saves a gasspy config into an hdf5 file as a seperate group
        input:
            gasspy_config - dict (Dict containing the gasspy config)
            h5file - hdf5 file identifier (Hdf5 file to save gasspy_config into)
    """
    # Open the group
    config_grp = h5file.create_group("gasspy_config")
    # Loop through all entries in the gasspy config and save them to the group
    for key in gasspy_config.keys():
        save_value_to_group(key, gasspy_config[key], config_grp)
    return

def read_gasspy_config_hdf5(gasspy_config, h5file):
    """
        Reads the gasspy config from a hdf5 file into a supplied dictionary
        input:
            gasspy_config - dict (Dict to save config to)
            h5file - hdf5 file identifier (Hdf5 file that contains the gasspy_config)
    """
    # Open the group
    config_grp = h5file["gasspy_config"]
    # Loop through all of entries into the group and add them to the dictionary
    for key in config_grp.keys():
        read_value_to_dict(key, gasspy_config, config_grp)