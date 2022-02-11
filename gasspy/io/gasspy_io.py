"""IO Routines"""
import yaml
import pickle
import numpy as np

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

def read_fluxdef(fluxdef_file):
    with open(r'%s'%fluxdef_file) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        fluxdef = yaml.load(file, Loader=yaml.FullLoader)
    return(fluxdef)
