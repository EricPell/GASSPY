import os
import shutil
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.Get_size()

def mpi_mkdir(path, mode = 0o777, dir_fd  = None):
    if mpi_rank != 0:
        os.mkdir(path, mode=mode, dir_fd = dir_fd)

def mpi_makedirs(path, mode = 0o777, exist_ok = False):
    if mpi_rank != 0:
        os.makedirs(path, mode=mode, exist_ok = exist_ok)

def mpi_copy(src, dst, follow_symlinks=True):
    if mpi_rank != 0:
        shutil.copy(src, dst, follow_symlinks=True)