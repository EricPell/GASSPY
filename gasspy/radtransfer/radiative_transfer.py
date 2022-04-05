#!/usr/bin/env python3
"""
This routine performs a spectra RT on an AMR grid
"""
import os
import time
import numpy as np
import torch
import gasspy.radtransfer.__rt__ as __rt__
import time
from astropy import constants as const

import cProfile, pstats

root_dir = os.getenv("HOME")+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/"
cuda_device = 'cuda:0'

for trace in ("000002_trace.hdf5",):
    print(trace)
    mytree = __rt__.FamilyTree(
        root_dir=root_dir,
        gasspy_subdir="GASSPY",
        traced_rays=trace,
        energy="gasspy_ebins.pkl.npy",
        energy_lims=[[0.003, 0.3],],
        em="gasspy_avg_em.pkl.npy",
        op="gasspy_grn_opc.pkl.npy",
        saved3d="saved3d_cloudyfields.npy",
        vel=root_dir+"/vx/celllist_vx_00051.fits",
        den=root_dir+"/rho/celllist_rho_00051.fits",
        opc_per_NH=True,
        mu=1.1,
        accel="torch",
        liteVRAM=False,
        Nraster=4,
        spec_save_name="gasspy_spec"+trace,
        dtype=np.float64,
        spec_save_type="hdf5",
        cuda_device=cuda_device
    )

    t = time.time()

    profiling = False

    mytree.load_all()
    t_start = t
    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    mytree.process_all()

    del(mytree)

    print ("total time = ",time.time()-t_start)
    if profiling:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('ncalls')
        stats.dump_stats("radtran_torch")
