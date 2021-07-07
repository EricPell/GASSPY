"""IO Routines"""
def read_compressed3d(file):
    import numpy as np
    with open(file,'rb') as f:
        data = np.load(f)
    return(data)

def read_avg_em(file):
    import pickle
    with open(file, 'rb') as f:
        data = pickle.load(f)
        return(data)



