from calendar import c
from re import I
import numpy as np
import h5py
import cupy
from gasspy.raystructures import global_ray_class
import sys

simdir = "/mnt/data/research/cinn3d/inputs/ramses/old/SEED1_35MSUN_CDMASK_WINDUV2"
min_lref = 6
max_lref = 11
def get_struct(xp, yp, trace_file):
    h5file = h5py.File(trace_file,"r")
    global_rays = global_ray_class()
    global_rays.load_hdf5(h5file)

    # Figure out which rays are leaf rays (defines as those with no child split event)
    ileafs = cupy.where(global_rays.get_field("cevid") == -1)[0]
    leaf_rays = global_rays.get_subset(ileafs)

    NleafRays = int(len(ileafs))
    print("Number of leafs : %d"%NleafRays)
    index_in_leaf_arrays     = np.zeros(global_rays.nrays, dtype=int)
    index_in_leaf_arrays[leaf_rays["global_rayid"].get()] = np.arange(NleafRays)


    # Min and max refinement level of the rays
    min_ray_lrefine = int(cupy.min(global_rays.get_field("ray_lrefine")))
    max_ray_lrefine = int(cupy.max(global_rays.get_field("ray_lrefine")))

    # split rays into lrefines
    rays_lrefs = []
    for lref in range(min_ray_lrefine, max_ray_lrefine+1):
        at_lref = cupy.where(global_rays.get_field("ray_lrefine") == lref)[0]
        rays_lrefs.append(global_rays.get_subset(at_lref))

    #Refinement level and index1D
    ray_amr_lrefine = h5file["ray_segments"]["amr_lrefine"][:,:]
    ray_index1D     = h5file["ray_segments"]["index1D"][:,:]
    ray_cell_index  = h5file["ray_segments"]["cell_index"][:,:]


    # Previously everything has been in boxlen=1 units, so we need to scale that
    ray_pathlength  = h5file["ray_segments"]["pathlength"][:,:]

    # Number of cells in a segment (eg the buffer size)
    NcellPerSeg = ray_amr_lrefine.shape[1]

    # Grab the list of global ray ids
    ray_global_rayid = np.array(h5file["ray_segments"]["global_rayid"][:])

    # Grab the list of split events
    splitEvents= cupy.array(h5file["splitEvents"][:,:])
 


    rayid     = int(leaf_rays["global_rayid"][cupy.argmin((cupy.square(yp - leaf_rays["yp"]) + 
                                          cupy.square(xp - leaf_rays["xp"])))])

    print(rayid, global_rays.xp[rayid], global_rays.yp[rayid], global_rays.aid[rayid])
    ray_lrefine = np.array([], dtype = int)
    amr_lrefine = np.array([], dtype = int)
    index1D = np.array([], dtype = int)
    pathlength = np.array([])
    ray_rayid = np.array([],dtype = int)
    next_rayid = rayid

    lref_of_ray = int(global_rays.get_field("ray_lrefine")[rayid])
    lref = lref_of_ray
    while lref >= min_ray_lrefine:
        current_rayid = next_rayid
        segs = np.where(ray_global_rayid == current_rayid)[0]
        amr_lrefine_c = ray_amr_lrefine[segs,:].ravel()
        index1D_c = ray_index1D[segs,:].ravel()
        pathlength_c = ray_pathlength[segs,:].ravel()

        print(current_rayid, len(segs))

        idx_to_keep = np.where(index1D_c >= 0)[0]
        #idx_to_keep = np.arange(len(segs))
        ray_lrefine = np.append(ray_lrefine, np.full(amr_lrefine_c[idx_to_keep].shape, lref))
        amr_lrefine = np.append(amr_lrefine, np.flip(amr_lrefine_c[idx_to_keep]))
        index1D     = np.append(index1D, np.flip(index1D_c[idx_to_keep]))
        pathlength  = np.append(pathlength, np.flip(pathlength_c[idx_to_keep]))
        ray_rayid   = np.append(ray_rayid, np.full(index1D_c[idx_to_keep].shape, current_rayid))

        lref = lref - 1
        next_rayid = int(splitEvents[global_rays.get_field("pevid")[current_rayid], 0])

    return np.flip(ray_rayid), np.flip(ray_lrefine), np.flip(amr_lrefine), np.flip(index1D), np.flip(pathlength)




xp = 2017/4096
yp = 1674/4096
trace_file = "/home/loki/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/GASSPY/projections/000001_trace.hdf5"
print(xp, yp)
rgid_s, rref_s, aref_s, ind1_s, path_s = get_struct(xp, yp, trace_file)

num_s = len(rref_s)
for i in range(num_s):
    print(i, "rgid:",rgid_s[i], "rref:",rref_s[i], "aref:",aref_s[i],"ind1:", ind1_s[i],"path:", path_s[i], np.sum(path_s[:i+1]))
  

