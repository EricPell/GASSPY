#!/usr/bin/env python3
"""
This routine performs a spectra RT on an AMR grid
"""
import os
import time
import cupy
import cupyx
import numpy as np
import torch
import gasspy.radtransfer.__rt_branch__ as __rt_branch__
import time
from astropy import constants as const

import cProfile, pstats

cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)

mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()

root_dir = os.getenv("HOME")+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/"

print_status_every_N = 1000

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
    accel="TORCH",
    liteVRAM=False,
    Nraster=4,
    dtype=np.float32
)

mytree.load_all()

print("N_energybins = %i"%(len(mytree.energy)))

t = time.time()
t_start = t

profiling = True
if profiling:
    profiler = cProfile.Profile()
    profiler.enable()

if mytree.accel == "TORCH":
    cuda_device = torch.device('cuda:0')

# for root_i in range(len(mytree.ancenstors)):
#     mytree.set_branch(root_i)
#     if root_i % 1000 == 0:
#         print(root_i)
#     if len(mytree.branch) > 1:
#         print(mytree.branch)

for root_i in range(len(mytree.ancenstors)):
    t = mytree.get_spec_root(root_i, cuda_device)
    if root_i % 1000 == 0:
        print(root_i)

print ("total time = ",time.time()-t_start)
if profiling:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.dump_stats("radtran_newrays")
