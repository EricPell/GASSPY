"""Populate Cube with Emissivity"""
import time
import numpy as np
import cupy
import pickle
from opiate.utils import io

outdir = "/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/"

comp3d= io.read_compressed3d("/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/opiate_compressed3d.npy")
avg_em = io.read_avg_em("/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/opiate_unique_avg_emissivity_dictionary.pkl")

Nx, Ny, Nz, Nfield = np.shape(comp3d)

tuple_dict ={}
t0 = time.time()
for ix in range(int(Nx/2)):
    for iy in range(int(Ny)):
        for iz in range(int(Nz)):
            tuple_dict[ix,iy,iz] = tuple(comp3d[ix,iy,iz])
del(comp3d)
print("Time to tuple : %f"%(time.time()-t0))

t0 = time.time()
for line in avg_em.keys():
    t0_line = time.time()
    for ix in range(int(Nx/2)):
        for iy in range(int(Ny)):
            for iz in range(int(Nz)):
                avg_em[line][tuple_dict[ix,iy,iz]]
    dt = time.time() - t0_line
    print("Total time for %s = %fs"%(line,dt))

dt = time.time() - t0
print("Total time for %s = %fs"%(line,dt))
