#!/usr/bin/env python3
"""
This routine performs a spectra RT on an AMR grid
"""
import os
import time
import numpy as np
import torch
import gasspy.radtransfer.__rt_branch__ as __rt_branch__
import time
from astropy import constants as const

import cProfile, pstats

root_dir = os.getenv("HOME")+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/"
cuda_device = torch.device('cuda:0')

mytree = __rt_branch__.FamilyTree(
    root_dir=root_dir,
    gasspy_subdir="GASSPY",
    traced_rays="000000_trace",
    energy="gasspy_ebins.pkl.npy",
    energy_lims=None,
    em="gasspy_avg_em.pkl.npy",
    op="gasspy_grn_opc.pkl.npy",
    saved3d="saved3d_cloudyfields.npy",
    vel=root_dir+"/vx/celllist_vx_00051.fits",
    den=root_dir+"/rho/celllist_rho_00051.fits",
    opc_per_NH=True,
    mu=1.1,
    accel="cuda",
    liteVRAM=False,
    Nraster=4,
    spec_save_name="gasspy_spec_000000_trace_cuda",
    dtype=np.float32,
    spec_save_type='hdf5',
    cuda_device=cuda_device
)

t = time.time()
t_start = t

profiling = True
if profiling:
    profiler = cProfile.Profile()
    profiler.enable()

mytree.load_all()
mytree.process_all(i0=79000)

print ("total time = ",time.time()-t_start)
if profiling:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.dump_stats("radtran_cuda")
