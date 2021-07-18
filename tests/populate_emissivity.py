"""Populate Cube with Emissivity"""
import sys
import time
import numpy as np
import pickle
from gasspy.utils import io
import multiprocessing
from multiprocessing import Pool
from multiprocessing import shared_memory

outdir = "/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/"

comp3d= io.read_compressed3d("/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/gasspy_compressed3d.npy")
avg_em = io.read_avg_em("/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/gasspy_unique_avg_emissivity_dictionary.pkl")

comp3d_shape = comp3d.shape
comp3d_dtype = comp3d.dtype

# if sys.argv[1]:
# shm0 = shared_memory.SharedMemory(create=True, size=comp3d.nbytes, name="comp3d")
# shm_comp3d = np.ndarray(comp3d_shape, dtype=comp3d_dtype, buffer=shm0.buf)
# shm_comp3d[:] = comp3d[:]

# shm_comp3d.close()
# shm_comp3d.unlink()


Nx, Ny, Nz, Nfield = comp3d_shape

del(comp3d)

#shm1 = shared_memory.SharedMemory(create=True, size=avg_em.__sizeof__(), name="avg_em")


tuple_dict ={}
t0 = time.time()

final = np.zeros((Nx,Ny,Nfield))

t0 = time.time()

def worker(ix, iy, comp3d_shape, comp3d_dtype, avg_em):

    shm_b = shared_memory.SharedMemory("comp3d")

    shm_comp3d = np.ndarray(comp3d_shape, dtype=comp3d_dtype, buffer=shm_b.buf)
    sliced = shm_comp3d[ix,iy,:]
    Nz, N_parms = np.shape(sliced)

    flux = np.zeros(len(avg_em))
    
    for iz in range(int(Nz)):
        tup = tuple(sliced[iz])
        #The first variable is log_depth_cm
        dx = 10**tup[0]
        for line_i, line in enumerate(avg_em):
            try:
                flux[line_i] += avg_em[line][tup]*dx
            except:
                pass
    
    shm_b.close()
    return(flux)

ray_list = []


flux_array = np.zeros((Nx,Ny,len(avg_em)))

if len(sys.argv) == 3:
    ix0 =int(sys.argv[1])
    ix1 =int(sys.argv[2])+1
    outfile = "/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/flux_array_%i_%i.npy"%(ix0,ix1)
else:
    ix0 = 0
    ix1 = Nx
    outfile = "/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/flux_array.npy"

for ix in range(ix0,ix1):
    for iy in range(int(Ny)):
        ray_list.append((ix,iy))

N_proc = len(ray_list)
N_groups = len(ray_list)//N_proc
if len(ray_list)%N_proc > 0:
    N_groups+=1


for i_Group in range(N_groups):
    range_start = i_Group*N_proc
    range_end =  (i_Group+1)*N_proc
    exec_list = [ [ix_iy[0], ix_iy[1], comp3d_shape, comp3d_dtype, avg_em] for ix_iy in ray_list[range_start:range_end] ]

    print(ray_list[range_start:range_end])

    # for i_ray, ix_iy in enumerate(ray_list[range_start:range_end]):
    #     flux_array[ix_iy[0],ix_iy[1],:] = worker(exec_list[i_ray][0],exec_list[i_ray][1])
    #     print(i_ray)


    with Pool(3) as p:
        ray_fluxes_list = p.starmap(worker, exec_list)
    
    for i_ray, ix_iy in enumerate(ray_list[range_start:range_end]):
        flux_array[ix_iy[0],ix_iy[1],:] = ray_fluxes_list[i_ray]

flux_array.tofile(outfile)

shm_comp3d.close()
shm_comp3d.unlink()

dt = time.time() - t0
print("Total time for %fs"%(dt))
