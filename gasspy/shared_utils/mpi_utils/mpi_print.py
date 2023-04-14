from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.Get_size()

def mpi_print(print_statement):
    if mpi_rank == 0:
        print(print_statement)