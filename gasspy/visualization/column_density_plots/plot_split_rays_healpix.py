from sys import path
from cudf.core import multiindex
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy.lib.function_base import insert
from pandas.core.indexes.multi import MultiIndex
import yaml
import cupy
import cudf
import pickle
import cupyx
import healpy as hp
import sys




simdir = "/mnt/data/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2"



# What we want to grab
# How many pseudo fields
pseudoList = ["Htotdensity", "HIIdensity"]
Npseudo = len(pseudoList)

Nspect  = 1
EmisList = ["HIIdensity"]
OpacList = ["density"]

def ha_em(nHp, ne=None, Tgas=1e4, Z=1.0):
    if ne == None:
        return(cupy.square(nHp)*a_ots(Tgas, Z))

def a_ots(Tgas, Z=1.0):
    # Return effective recombination rate excluding ground state in cm3 s-1 
    return 2.6e-13 * cupy.square(Z) * cupy.power(Tgas/1e4, -0.8)

clght = 29979245800.0
boltz = 1.38065e-16
f0 = 3.288599214800749e15
mH = 1.675e-24
def get_HalphaEmission(nHp, Tgas, velocity, freqBins, dfreq):
    Nphot = ha_em(nHp, Tgas = Tgas)[:,cupy.newaxis]
    f0_shifted = f0/(1 + velocity/clght)[:,cupy.newaxis]  
    width = f0_shifted/clght*cupy.sqrt(2*boltz*Tgas[:,cupy.newaxis]/mH)
    spectra = Nphot * cupy.exp(-cupy.power((freqBins-f0_shifted)/width,2)) / width /cupy.sqrt(cupy.pi)*dfreq

    return spectra

def get_Opacity(nH, Tgas, freqBins, dfreq):
    #return 2.6737967914438504e-26 * nH[:,cupy.newaxis]*cupy.ones(freqBins.shape)
    return 2.6737967914438504e-21 * (nH * np.exp(-Tgas/2e4))[:,cupy.newaxis]* cupy.ones(freqBins.shape)


def arange_indexes(start, end):
    # Method to generate list of indices based on a start and end
    # (magic from stackoverflow...)
    lens = end - start
    cupy.cumsum(lens, out=lens)
    i = cupy.ones(int(lens[-1]), dtype=int)
    i[0] = start[0]
    i[lens[:-1]] += start[1:]
    i[lens[:-1]] -= end[:-1]
    cupy.cumsum(i, out=i)
    return i



def load_field(field_name):
    # Loads a field from the simdir
    with fits.open(simdir + "/" + field_name + "/celllist_"+field_name+"_00051.fits") as F:
        data = F[0].data
    return data


# Open up the YAML config
with open(r"%s/GASSPY/gasspy_config.yaml"%simdir) as fil:
    yamlfile = yaml.load(fil, Loader = yaml.FullLoader)


# Load denisty from the simulation
cell_dens = cupy.array(load_field("rho"), dtype = cupy.float64)/mH
cell_HIIdensity = cupy.array(load_field("rho"), dtype = cupy.float64)*cupy.array(load_field("xHII"), dtype = cupy.float64)/mH
cell_temperatue = cupy.array(load_field("T"), dtype = cupy.float64)
cell_velocity = cupy.array(load_field("vz"), dtype = cupy.float64)*1e5

# Take the amr refinement and index1D from our own calcualtions to make sure dtypes matches
cell_amr_lrefine = cupy.load(simdir+"/GASSPY/amr_lrefine.npy")
cell_index1D     = cupy.load(simdir+"/GASSPY/index1D.npy")

# Load properties of the grid from the yaml file
min_lref = yamlfile["amr_lrefine_min"]
max_lref = yamlfile["amr_lrefine_max"]
boxlen  = yamlfile["sim_size_x"]
scalel  = yamlfile["sim_unit_length"]

# Pre caclualte all cell sizes
dx_lrefs = boxlen/2**(np.arange(min_lref, max_lref+1))

# TODO: DO THIS BEFORE SOMEWHERE
# Add all NULL values that are needed.
# We either need one for all refinement levels, or set the refinement levels of all rays with NULL index1D to a specified NULL value
cell_index1D = cupy.append(cell_index1D, [-1 for lref in range(min_lref, max_lref+3)])
cell_amr_lrefine = cupy.append(cell_amr_lrefine, [lref for lref in range(min_lref, max_lref +3)])
cell_amr_lrefine[-1] = -1
cell_amr_lrefine[-2] = 0
cell_dens = cupy.append(cell_dens, [0 for lref in range(min_lref, max_lref+3)])
cell_HIIdensity = cupy.append(cell_HIIdensity, [0 for lref in range(min_lref, max_lref+3)])
cell_temperatue = cupy.append(cell_temperatue, [1 for lref in range(min_lref, max_lref+3)])
cell_velocity = cupy.append(cell_velocity, [0 for lref in range(min_lref, max_lref+3)])
cell_Ha = ha_em(cell_HIIdensity,Tgas= cell_temperatue)

# Collect to data frame
cell_dataFrame = cudf.DataFrame({"index1D"     : cell_index1D, 
                                 "amr_lrefine" : cell_amr_lrefine, 
                                 "Htotdensity" : cell_dens,
                                 "Temperature" : cell_temperatue,
#                                 "VelocityZ"   : cupy.zeros(cell_velocity.shape),
                                 "VelocityZ"   : cell_velocity,
                                 "HIIdensity"  : cell_HIIdensity
                                 })
# Set indexing
cell_dataFrame.set_index(["amr_lrefine", "index1D"], inplace = True)

# load up on rays
global_rayDF = cudf.read_hdf(simdir+"/GASSPY/healpix_global_rayDF.ray")

print(global_rayDF[global_rayDF["aid"] == 79086])
# set up frequency bins

minVel = cupy.min(cell_velocity[cell_Ha > 1e-8])
maxVel = cupy.max(cell_velocity[cell_Ha > 1e-8])

freqBins = cupy.linspace(f0/(1 + 2*maxVel/clght), f0/(1 + 2*minVel/clght), Nspect)
dfreq = cupy.zeros(freqBins.shape)

#dfreq[:-1] = (freqBins[1:] + freqBins[:-1])/2
#dfreq[1:-1] -= dfreq[:-2]
#dfreq[0] = 2*(dfreq[0] - freqBins[0])
#dfreq[-1] = dfreq[-2]   


# Figure out which rays are leaf rays (defines as those with no child split event)
leafRays = global_rayDF[global_rayDF["cevid"] == -1]
NleafRays = int(len(leafRays))
print("Number of leafs : %d"%NleafRays)
leaf_ray_pseudofields    = np.zeros((NleafRays,Npseudo))
#leaf_ray_emissivity      = np.zeros((NleafRays,Nspect))
index_in_leaf_arrays     = np.zeros(len(global_rayDF), dtype=int)
index_in_leaf_arrays[leafRays["global_rayid"].values.get()] = np.arange(NleafRays)


# Min and max refinement level of the rays
min_ray_lrefine = global_rayDF["ray_lrefine"].min()
max_ray_lrefine = global_rayDF["ray_lrefine"].max()
# split into sets of lrefine
global_rayDF_lrefs = [global_rayDF[global_rayDF["ray_lrefine"]==lref] for lref in range(int(min_ray_lrefine),int(max_ray_lrefine)+1)]


raydump_dict = {}
#with open(simdir+"/GASSPY/000000_traced_rays.ray", "rb") as f:

with open(simdir+"/GASSPY/healpix_loke_devel.ray", "rb") as f:
    tmp = pickle.load(f)
    raydump_dict.update(tmp)


#Refinement level and index1D
ray_amr_lrefine = raydump_dict["amr_lrefine"]
ray_index1D     = raydump_dict["index1D"]
# Just as a caution, set index1D of all non valid refinement levels to its NULL value
ray_index1D[ray_amr_lrefine < min_lref] = -1

# Previously everything has been in boxlen=1 units, so we need to scale that
ray_pathlength  = raydump_dict["pathlength"]*scalel

# Number of cells in a segment (eg the buffer size)
NcellPerSeg = ray_amr_lrefine.shape[1]

# Grab the list of global ray ids
ray_global_rayid = cupy.array(raydump_dict["global_rayid_ofSegment"])

# Grab the list of split events
splitEvents= raydump_dict["splitEvents"]


# maximum memory usage
maxMemory  = 4 * 1024**3

# bytes per ray RT storage
Nfields = 2*Nspect + Npseudo
perRay = Nfields * np.float64(1).itemsize 

# bytes per ray segment
# Number of fields + rayid + amr_lrefine + index1D
perSeg = (2* Nfields * np.float64(1).itemsize + ray_index1D.itemsize + ray_amr_lrefine.itemsize + ray_pathlength.itemsize)* NcellPerSeg

# Arrays to keep track of where things are
ray_idx_in_gpu = cupy.full(len(global_rayDF), -1, dtype = int)
ray_idx_in_cpu = np.full(len(global_rayDF), -1, dtype = int)

# Init parent stuff
parent_DF = None
counter = 0
for ray_lref in range(min_ray_lrefine, max_ray_lrefine+1):
    # How many rays do we have at this refinement level?
    rays_at_lref = global_rayDF_lrefs[ray_lref - min_ray_lrefine]
    nrays_at_lref = len(rays_at_lref)
    print(ray_lref, nrays_at_lref,len(global_rayDF))

    # Can we fit all of them?
    # If not take some predefined amount of memory
    if nrays_at_lref*perRay < 0.75*maxMemory:
        nrays_at_a_time = nrays_at_lref

    else:
        nrays_at_a_time = int(0.5*maxMemory/perRay)

    # Allocate CPU memory for all rays    
    ray_pseudofields_cpu    = cupyx.zeros_pinned((nrays_at_lref,Npseudo))
    #ray_emissivity_cpu      = cupyx.zeros_pinned((nrays_at_lref,Nspect))
    #ray_opacity_cpu         = cupyx.zeros_pinned((nrays_at_lref,Nspect))

    # Determine rayid mapping to cpu arrays
    ray_idx_in_cpu[rays_at_lref["global_rayid"].values.get()] = np.arange(nrays_at_lref)

    # Add from parents
    if parent_DF is not None:
        # get the children IDs from the child split events of the parents
        child_rayid = splitEvents[parent_DF["cevid"].values,1:].ravel().astype(int).get()

        # Duplicate all values by four and add to the cpu fields
        ray_pseudofields_cpu[ray_idx_in_cpu[child_rayid],:] = parent_pseudofields.repeat(4, axis = 0)
        #ray_emissivity_cpu[ray_idx_in_cpu[child_rayid],:] = parent_emissivity.repeat(4, axis = 0)
        #ray_opacity_cpu[ray_idx_in_cpu[child_rayid],:] = parent_opacity.repeat(4, axis = 0)




    iray_start = 0
    while iray_start < nrays_at_lref:
        # Figure out how many rays we have in this iteration
        iray_end = iray_start + nrays_at_a_time
        iray_end = min(iray_end, nrays_at_lref)
        nrays_now = (iray_end - iray_start)

        # Move the rays at this current iteration to the gpu
        ray_pseudofields_gpu    = cupy.array(ray_pseudofields_cpu[iray_start : iray_end, :])
        #ray_emissivity_gpu      = cupy.array(ray_emissivity_cpu[iray_start : iray_end, :])
        #ray_opacity_gpu         = cupy.array(ray_opacity_cpu[iray_start : iray_end, :])


        # Determine rayid mapping to cpu arrays
        ray_idx_in_gpu[rays_at_lref["global_rayid"].values] = cupy.arange(nrays_now)


        # Memory for segments
        memory_remaining = maxMemory - nrays_now*perRay

        # Take the segments corresponding to the current sets of rays 
        idx = np.where(np.isin(ray_global_rayid.get(), rays_at_lref.iloc[iray_start:iray_end]["global_rayid"].to_array()))[0]

        seg_amr_lrefine_at_ray_lref = ray_amr_lrefine[idx,:]
        seg_index1D_at_ray_lref = ray_index1D[idx,:]
        seg_pathlength_at_ray_lref = ray_pathlength[idx,:]
        seg_global_rayid_at_ray_lref = ray_global_rayid[idx]

        # Total number of ray segments in the trace
        Nsegs = len(seg_pathlength_at_ray_lref)


        # How many we calculate at a time (Memory requiremenets)
        Nseg_at_a_time =min(Nsegs, int(memory_remaining/perSeg))

        # How many segments we have left
        Nrem = Nsegs

        # Index of the current segment
        iseg_start = 0

        while Nrem > 0:


            # Determine en index of list of segments
            iseg_end = min((iseg_start + Nseg_at_a_time), Nsegs)
            Nrem = Nrem - (iseg_end - iseg_start)

            # Grab all that you need onto the GPU
            amr_lrefine = cupy.array(seg_amr_lrefine_at_ray_lref[iseg_start:iseg_end, : ])
            index1D     = cupy.array(seg_index1D_at_ray_lref[iseg_start:iseg_end, : ])
            pathlength  = cupy.array(seg_pathlength_at_ray_lref[iseg_start:iseg_end,:])
            global_rayid = cupy.array(seg_global_rayid_at_ray_lref[iseg_start:iseg_end]) 

            # Ravel the index1D and amr_lrefine. They need to be used for indexing, and the best way I've found 
            # is to put them in a dataframe and use cudf.MultiIndex.from_frame()
            print(counter)

            index1D = index1D.ravel()
            amr_lrefine = amr_lrefine.ravel()
            indexing_df = cudf.DataFrame({"amr_lrefine" : amr_lrefine, "index1D" : index1D})
            index = cudf.MultiIndex.from_frame(indexing_df)
            pathlength = pathlength[:,:,cupy.newaxis]
            #if(counter == 21):
            #    for alrf in cupy.unique(amr_lrefine):
            #        r_at_lref = index1D[amr_lrefine==alrf]
            #        c_at_lref = cell_index1D[cell_amr_lrefine == alrf]
            #        not_here = cupy.isin(r_at_lref, c_at_lref) 
            #        print(alrf, r_at_lref[~not_here]) 
            #    sys.exit(0)              
            # Grab the density fields needed
            field = pathlength * cupy.array(cell_dataFrame[pseudoList].loc[index].values).reshape((iseg_end - iseg_start, NcellPerSeg, Npseudo))
            #opac = cupy.array(cell_dataFrame[EmisList].loc[index].values).reshape((iseg_end - iseg_start, NcellPerSeg, Nspect))
            #emis = cupy.array(cell_dataFrame[OpacList].loc[index].values).reshape((iseg_end - iseg_start, NcellPerSeg, Nspect))




            #nHp = cell_dataFrame["HIIdensity"].loc[index].values
            #Temp = cell_dataFrame["Temperature"].loc[index].values
            #velo = cell_dataFrame["VelocityZ"].loc[index].values
            #emis = pathlength * get_HalphaEmission(nHp, Temp, velo, freqBins, dfreq).reshape((iseg_end - iseg_start, NcellPerSeg, Nspect))

            #nH = cell_dataFrame["Htotdensity"].loc[index].values
            #opac = pathlength * get_Opacity(nH, Temp, freqBins, dfreq).reshape((iseg_end - iseg_start, NcellPerSeg, Nspect))
            # Next bits are a bit memory intensive so clean up a bit
            #del(nHp)
            #del(velo)
            #del(Temp)
            #del(nH)
            #del(index1D)
            #del(amr_lrefine)
            #del(pathlength)
            # Total in segments
            # transform pathlength to to a 3D matrix (Nseg, NcellPerSeg,1)
            #cumopac = cupy.cumsum(opac, axis = 1)  - opac
            field = cupy.sum(field, axis = 1)
            #emis = cupy.sum(emis*cupy.exp(-cumopac), axis = 1)
            #if cupy.any(emis < 0):
            #    print("??????????") 
            #opac = cupy.sum(opac, axis = 1)
            #del(cumopac)

            #Gather all segments that belong to the same ray
            # The list is ordered so we just need to find where they differ           
            rayids, idxm, counts = cupy.unique(global_rayid, return_index=True, return_counts = True)
            idxp = idxm + counts
            # Determine all indexes
            idxes = arange_indexes(idxm, idxp)


            #tot_opac  = cupy.zeros(opac.shape)
            #tot_field = cupy.zeros(field.shape)
            #tot_emis  = cupy.zeros(emis.shape)

            #for iray in range(len(rayids)):
                
            #    tot_opac [idxm[iray] : idxp[iray], :]  = cupy.cumsum(opac [idxm[iray] : idxp[iray],:], axis = 0)
            #    tot_field[idxm[iray] : idxp[iray], :]  = cupy.cumsum(field[idxm[iray] : idxp[iray],:], axis = 0)
            #    tot_emis [idxm[iray] : idxp[iray], :]  = cupy.cumsum(emis [idxm[iray] : idxp[iray],:] * cupy.exp(-(tot_opac[idxm[iray] : idxp[iray],:] - opac[idxm[iray] : idxp[iray],:])), axis = 0)


            # We need to deal with emissivities differently, so first we do opacities and pseudo fields
            # Make a cumulative sum and then remove the previous value where we change rays
            #tot_opac  = cupy.cumsum(opac, axis = 0)
            tot_field = cupy.cumsum(field, axis = 0)
            # if we only have one ray, no need to do subtract previous values
            if len(idxm) > 1:
            #    # We only need to do this for the 2nd ray and beyond
            #    tot_opac [idxes[counts[0]:],:] -= cupy.repeat(tot_opac [idxm[1:]-1,:], counts[1:].tolist(), axis = 0)
                tot_field[idxes[counts[0]:],:] -= cupy.repeat(tot_field[idxm[1:]-1,:], counts[1:].tolist(), axis = 0)
            #    for line in range(len(rayids)):
            #        print(tot_opac[idxm[line] : idxp[line],0]-opac[idxm[line] : idxp[line],0])
            # With emissivities We sum at the same time as we sum the opacity
            #tot_emis = cupy.cumsum(emis*np.exp(-(tot_opac - opac)), axis = 0)
            # and do the same subtraction for multiple rays
            #if len(idxm) > 1:
            #    tot_emis [idxes[counts[0]:],:] -= cupy.repeat(tot_emis [idxm[1:]-1,:], counts[1:].tolist(), axis = 0)

            # Check for floating point errors
            #tot_emis[tot_emis < 0] = 0
            # Now add the final bits of each ray to its values and 
            # Determine the indexes in the gpu arrays
            idx_in_gpu = ray_idx_in_gpu[rayids]
            ray_pseudofields_gpu[idx_in_gpu,:] += tot_field[idxp - 1,:]
            #ray_emissivity_gpu[idx_in_gpu,:] += tot_emis[idxp-1,:] * cupy.exp(-ray_opacity_gpu[idx_in_gpu, :])
            #ray_opacity_gpu[idx_in_gpu, :] += tot_opac[idxp-1,:]
            
            iseg_start = iseg_end
            counter += 1
        # Move off to the CPU arrays
        ray_pseudofields_cpu[iray_start: iray_end, :] = ray_pseudofields_gpu.get()
        #ray_opacity_cpu[iray_start: iray_end, :]      = ray_opacity_gpu.get()
        #ray_emissivity_cpu[iray_start: iray_end, :]   = ray_emissivity_gpu.get()

        iray_start = iray_end  

    # All rays at this refinement level are done!
    # Save all that are leafs
    is_leaf = rays_at_lref["cevid"] == -1
    idx_leaf = np.where(is_leaf.values)[0].get()
    leaf_ray_pseudofields[index_in_leaf_arrays[rays_at_lref[is_leaf]["global_rayid"].values.get()], :] = ray_pseudofields_cpu [idx_leaf, :]
    #leaf_ray_emissivity[index_in_leaf_arrays[rays_at_lref[is_leaf]["global_rayid"].values.get()], :] = ray_emissivity_cpu [idx_leaf, :]

    # keep the parent for the next step
    is_parent = ~is_leaf
    idx_parent = np.where(is_parent.values)[0].get()
    parent_DF=  rays_at_lref[is_parent]
    parent_pseudofields = ray_pseudofields_cpu [idx_parent, :] 
    #parent_opacity = ray_opacity_cpu[idx_parent, :] 
    #parent_emissivity = ray_emissivity_cpu[idx_parent, :] 




# We now have the column density of each ray, but now we need to connect the parents to their children
# Make it plottable
#ray_dens = ray_dens.reshape((2048,2048))
# Make it plot
#plt.imshow(cupy.log10(ray_dens.T).get())
#plt.show()



import matplotlib as mpl
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def add_norm(data, rgb_vals, zmin = None, zmax = None):
    if zmin is None:
        zmin = max(np.min(data), 1e10)
    if zmax is None:
        zmax = np.max(data)

    norm = mcol.LogNorm(vmin = zmin, vmax = zmax)
    rgb_map = np.zeros((data.shape[0],data.shape[1],3))
    data = norm(data).data
    rgb_map = data[:,:,np.newaxis]*rgb_vals
    return rgb_map


# Create a list at highest level of refinement... (WTF????)
Npix_max = hp.nside2npix(2**(leafRays["ray_lrefine"].max() - 1))
plot_map = np.zeros(Npix_max)
max_ray_lrefine = leafRays["ray_lrefine"].max()

for lref in range(leafRays["ray_lrefine"].min(), leafRays["ray_lrefine"].max() + 1):
    ndifference = int(4**(max_ray_lrefine - lref))
    
    at_lref = np.where(leafRays["ray_lrefine"].values.get() == lref)[0]
    ipxs_old = cupy.array(leafRays["xp"].astype(int).values[at_lref])
    ipxs_new = cupy.repeat(ipxs_old*ndifference, ndifference)
    ipxs_new += cupy.tile(cupy.arange(ndifference), len(ipxs_old))
    print(ipxs_old[0], ipxs_new[:ndifference])
    plot_map[ipxs_new.get()] = np.repeat(leaf_ray_pseudofields[at_lref,0], ndifference)

print(np.sum(plot_map == 0))
plot_map[plot_map == 0] = np.min(leaf_ray_pseudofields[at_lref,0])
hp.mollview(np.log10(plot_map), nest = True, cmap="BuPu_r")
plt.show()
#rgb_map = add_norm(leaf_ray_pseudofields[:,0], np.array([0.55,0.25,0.75]), zmax = 5e22)
#rgb_map += add_norm(leaf_ray_pseudofields[:,1], np.array([0.8,0.8,0.15]))
#rgb_map[rgb_map >= 1] = 1
#rgb_map[rgb_map <= 0] = 0
#del(leaf_ray_pseudofields)

#fig = plt.figure()
#for lref in range(leafRays["ray_lrefine"].min(), leafRays["ray_lrefine"].max() + 1):
#    at_lref = leafRays["ray_lrefine"].values == lref

