from mpi4py import MPI
import numpy as np

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.Get_size()


def mpi_any_sync(to_sync, tag = 99):
    """
    Used to sync in loops were each rank independently do several iterations without communication.
    Call in every iteration, if no rank wants to sync (all pass 0) nothing happens.
    If any wants to sync (any passes 1), all ranks will sync within function and communication can happen
    afterwards
    input:
        int : to_sync (0 if this rank does not need to sync, 1 if this rank needs to sync)
    return
        ndarray of ints: all_sync (Array of size mpi_size where 1 or 0 says if rank does or does not want to sync respectively)
    """
    all_sync = np.zeros(mpi_size, dtype = int)
    # if rank has status == 1, tell the others
    if to_sync == 1:
        for irank in range(mpi_size):
            if irank == mpi_rank:
                continue
            mpi_comm.isend(to_sync, irank, tag = tag)
    # Check if any other rank has status == 1
    for irank in range(mpi_size):
        if irank == mpi_rank: 
            all_sync[irank] = to_sync
        else:
            if mpi_comm.iprobe(source = irank, tag = tag):
                all_sync[irank] = mpi_comm.recv(source = irank, tag = tag)
                
    # If any rank had status == 1, gather to make sure we know of all of them
    if np.any(all_sync == 1):
        all_sync[:] = mpi_comm.allgather(to_sync)
        # Catch any lingering messages as we sould be synced and reset at this point
        for irank in range(mpi_size):
            if irank == mpi_rank: 
                continue
            else:
                if mpi_comm.iprobe(source = irank, tag = tag): 
                    tmp = mpi_comm.recv(source = irank, tag = tag)
    return all_sync