import pickle
import numpy as np
emissivity_dict = pickle.load( open( "/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/opiate_unique_emissivity_dictionary.pkl", "rb" ) )
param_dict = pickle.load(open("/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/opiate_unique.pkl", "rb"))

line = list(emissivity_dict.keys())[30]

print(list(emissivity_dict[line].keys())[0])
print(list(param_dict.keys()))
stacked = np.vstack([
param_dict['dx'], 
param_dict['dens'], 
param_dict['temp'], 
param_dict['FUV'], 
param_dict['HII'], 
param_dict['HeII'], 
param_dict['HeIII']]
).T

import time
t0 = time.time()
repeats = 100
for i in range(repeats):
    b = [tuple(p) for p in stacked]
    for c in b:
        d = emissivity_dict[line][c]
    print(d)
print(time.time()-t0)
print(len(b)*100)
