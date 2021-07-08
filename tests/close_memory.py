import sys
import time
import numpy as np
import pickle
from opiate.utils import io
import multiprocessing
from multiprocessing import Pool
from multiprocessing import shared_memory

outdir = "/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/"

comp3d= io.read_compressed3d("/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/opiate_compressed3d.npy")
avg_em = io.read_avg_em("/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/opiate_unique_avg_emissivity_dictionary.pkl")

comp3d_shape = comp3d.shape
comp3d_dtype = comp3d.dtype

shm0 = shared_memory.SharedMemory(size=comp3d.nbytes, name="comp3d")

shm0.close()
shm0.unlink()

