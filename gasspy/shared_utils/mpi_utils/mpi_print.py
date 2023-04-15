from mpi4py import MPI
import sys
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.Get_size()

def mpi_print(*args, **kwargs):
    if mpi_rank == 0:
        print(*args, **kwargs)
        sys.stdout.flush()
        
def mpi_all_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()