#%%
from numpy.lib.function_base import average
from opiate.utils import io
import numpy as np
import pickle as pkl

import cupy
import cudf
import rmm

cudf.set_allocator("managed")
rmm.reinitialize(managed_memory=True)
assert(rmm.is_initialized())

#%%
indir = "/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/"

### TODO:
# SAVE PARAM_TO_INDEX which is the original opiate index idea. (should be generated when creating unique)
# Save index_to_param: used if recreating a new grid with different values. (Should be generated when creating unique)
# Save indexed_avg_em (should be generated using avg_em)

# Create the lookup index, and store in both directions if you want to recover cell-params from index
try:
    param_to_index = io.read_dict(indir+"opiate.param_to_index")
    param_to_index = io.read_dict(indir+"opiate.index_to_param")
    indexed_avg_em = io.read_dict(indir+"opiate.indexed_avg_em")
    emlines = [line for line in indexed_avg_em]

except:
    # Clean the emissivity dictionary of bad values
    avg_em = io.read_avg_em(indir+"opiate_unique_avg_emissivity_dictionary.pkl")
    emlines = [line for line in avg_em]

    for emline in emlines:
        for i, key in enumerate(avg_em[emline]):
            try:
                float(avg_em[emline][key])
            except:
                avg_em[emline][key] = 0.0

    param_to_index = {}
    index_to_param = {}
    for i, key in enumerate(avg_em[emlines[0]]):
        param_to_index[key] = i
        index_to_param[i] = key

    # Initialize the average indexed emissivity list for each line
    indexed_avg_em ={}
    for line in emlines:
        indexed_avg_em[line] = {}

    # Loop over each physical parameter key in the first dictionary, which works for the other emission lines as well
    # save the emissivity using an index
    for i, key in enumerate(avg_em[emlines[0]]):
        for line in emlines:
            indexed_avg_em[line][i] = avg_em[line][key]

    io.write_dict(param_to_index, indir+"opiate.param_to_index")
    io.write_dict(param_to_index, indir+"opiate.index_to_param")
    io.write_dict(indexed_avg_em, indir+"opiate.indexed_avg_em")

#%%
try:
    # Try and load an exisiting simulation
    index3d = np.load(indir+"opiate_indices3d.npy")
    Nx,Ny,Nz,Nd = index3d.shape
except:
    comp3d= io.read_compressed3d(indir+"opiate_compressed3d.npy")
    index3d = np.ndarray(shape=comp3d.shape[:-1],dtype="int32")
    
    Nx,Ny,Nz,Nd = comp3d.shape
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                index3d[ix,iy,iz] = param_to_index[tuple(comp3d[ix,iy,iz,:])]

    np.save(indir+"opiate_indices3d.npy",index3d)
    del(comp3d)


#%%
df = cudf.DataFrame()
df["opiate_index"] = index3d.ravel()

#%%
xx,yy,zz = np.meshgrid(np.arange(Nx),np.arange(Ny),np.arange(Nz))

df["x"] = xx.ravel()
df["vx"] = xx.ravel()
del(xx)
df["y"] = yy.ravel()
df["vy"] = yy.ravel()
del(yy)
df["z"] = zz.ravel()
df["vz"] = zz.ravel()
del(zz)

#%%
line = "H  1 6562.81A"

#%%
emdf = cudf.DataFrame({"index":np.array(list(indexed_avg_em[line].keys()), dtype="int32"),\
    "data":np.array(list(indexed_avg_em[line].values()), dtype="float32")})
#%%
emdf.min()
# %%
