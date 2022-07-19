import h5py
from matplotlib.cbook import delete_masked_points
import numpy as np
import cupy
import psutil
from pathlib import Path
import sys

from gasspy.shared_utils.spectra_functions import broadband, integrated_line
from gasspy.io import gasspy_io
class spec_reader:
    def __init__(self, spec_file,
        sim_dir="./",
        gasspy_subdir="GASSPY",
        gasspy_spec_subdir="spec",
        maxmem_GB = None,
    ):
        """
            Arguments:
            spec_file: string or open hdf5 file (hdf5 file containing the spectra and their position)
            root_dir : string (root directory of simulation)
            gasspy_subdir : string (name of the GASSPY sub-directory within root_dir)
            gasspy_spec_subdir : string (name of the spectra subdirectory within GASSPY sub-directory)
            maxmem_GB: float (maximum amount of system memory we will try to use. NOTE: this is just and estimate so be conservative)
        """
        self.sim_dir = sim_dir
        self.gasspy_subdir = gasspy_subdir
        self.gasspy_spec_subdir = gasspy_spec_subdir
        if not isinstance(spec_file, h5py._hl.files.File):
            assert isinstance(spec_file, str), "provided spec_file is neither a string or open hd5 file"
            if not spec_file.endswith(".hdf5"):
                spec_file += ".hdf5"
            if Path(spec_file).is_file():
                tmp_path = spec_file
            elif Path(self.sim_dir+self.gasspy_subdir+self.gasspy_spec_subdir+spec_file).is_file():
                tmp_path = self.sim_dir+self.gasspy_subdir+self.gasspy_spec_subdir+spec_file
            else:
                sys.exit("Could not find the traced rays file\n"+\
                "Provided path: %s"%spec_file+\
                "Try looking in \"./\" and %s\n"%(self.sim_dir+self.gasspy_spec_subdir)+\
                "Aborting...")            

            self.spec_file = h5py.File(tmp_path, "r")
        else:
            self.spec_file = spec_file

        ## Load the gasspy_config yaml
        gasspy_config = gasspy_io.read_fluxdef(self.sim_dir+"/gasspy_config.yaml")

        # Load the posional information of the rays
        self.xp = self.spec_file["x"][:]
        self.yp = self.spec_file["y"][:]
        self.ray_lrefine = self.spec_file["ray_lrefine"][:]

        # Load the energy bins
        self.Energies = self.spec_file["E"][:]
        # TODO: calculate size of energy bins prior to this, where we actually have all the information needed....
        self.deltaEnergies = np.zeros(self.Energies.shape)
        self.deltaEnergies[1:-1] = self.Energies[2:] - self.Energies[:-2]
        self.deltaEnergies[0] = self.deltaEnergies[1]
        self.deltaEnergies[-1] = self.deltaEnergies[-2]
        
        # set maximum used memory
        if maxmem_GB is not None:
            self.maxmem = maxmem_GB * 1024**3
        else:
            # otherwise try to use available memory with a 0.5 GB to spare
            self.maxmem = psutil.virtual_memory().free - 0.5*1024**3

        # TODO: This should be save parameters both inside of traced rays and spec
        self.observer_size_x = gasspy_config["detector_size_x"]
        self.observer_size_y = gasspy_config["detector_size_y"]

    def read_rays(self, idxs, Eidxs):
        if isinstance(idxs, int):
            idxs = np.array([idxs], dtype = int)
        if isinstance(Eidxs, int):
            Eidxs = np.array([Eidxs], dtype = int)
        #flux = np.zeros((len(idxs), len(Eidxs)))
        #for i, idx in enumerate(idxs):
        #    flux[i,:] = self.spec_file["flux"][idx, Eidxs]
        flux = np.vstack([self.spec_file["flux"][idx, Eidxs] for idx in idxs])
        return flux

    def create_map(self, 
                xlims = None, ylims = None, 
                Elims = None, window_method = broadband, 
                outmap_nfields = 1, outmap_dtype = np.float64, 
                outmap_nx = None, outmap_ny = None):
        """
            Creates a uniform map with the desired fields based on the ray fluxes
            Arguments:
                xlims: array of size 2 (minimum and maximum size of the map in x)
                ylims: array of size 2 (minimum and maximum size of the map in y)
                Elims: array of size 2 (minimum and maximum energy to use in flux)
                window_method: optional method (method used to derive the fields from the specified fluxes, default is integrated)
                outmap_nfields: int (Number of fields of the map, same as the number of fields returned by the window_method, default is 1)
                outmap_dtype: type (dtype of the outmap)
                outmap_nx: number of cells in the x direction of the map
                outmap_ny: number of cells in the y direction of the map
        """

        # if we are reading the entire set of rays set limits to cover entire map
        if xlims is None:
            xlims = np.array([0, self.observer_size_x])
        if ylims is None:
            ylims = np.array([0, self.observer_size_y])
        outmap_size_x = xlims[1] - xlims[0]
        outmap_size_y = ylims[1] - ylims[0] 
        
        # get rays within limits
        print(xlims)
        idxs = np.where((self.xp >= xlims[0]) * (self.xp < xlims[1]) *
                        (self.yp >= ylims[0]) * (self.yp < ylims[1]))[0]
        # if we are reading the entire energy range, set limits to cover everything
        if Elims is None:
            Elims = np.array([np.min(self.Energies)-1, np.max(self.Energies) + 1])
        Eidxs = np.where( (self.Energies >= Elims[0]) * (self.Energies < Elims[1]))[0]
        
        # If nx or ny are not defined we use the maximum of the providied rays
        outmap_ray_lrefine = np.max(self.ray_lrefine)
        if outmap_nx is None:
            outmap_nx = np.around(2**outmap_ray_lrefine  * outmap_size_x/self.observer_size_x).astype(int)
        if outmap_ny is None:
            outmap_ny = np.around(2**outmap_ray_lrefine  * outmap_size_y/self.observer_size_y).astype(int)
        print(outmap_nx, outmap_ny)
        # Size of each pixel 
        outmap_dx = outmap_size_x/outmap_nx
        outmap_dy = outmap_size_y/outmap_ny


        # Total number of cells in the outmap
        outmap_ncells = outmap_nx*outmap_ny
        # Memory load of the map
        outmap_size = outmap_ncells * outmap_nfields * outmap_dtype(1).itemsize

        # Male sure it fits
        if outmap_size > self.maxmem:
            sys.exit("Cant allocate map with current memory limits. map memory size : %f GB, available memory : %f GB\n try using a map on a lower ray_refine level or less pixels"%(outmap_size/1024**3, self.maxmem/1024**3))

        outmap = np.zeros((outmap_nx, outmap_ny, outmap_nfields), outmap_dtype)
        free_mem = self.maxmem - outmap_size
        
        # figure out the size of a ray
        raysize = len(Eidxs) * self.spec_file["flux"].dtype.itemsize
        print(len(Eidxs), self.spec_file["flux"].dtype.itemsize)
        rays_at_a_time = int(free_mem/raysize)
        
        # grab used energies and deltaEnergies
        Energies = self.Energies[Eidxs]
        # TODO: This should be calculated previously on a per bin basis
        deltaEnergies = (Energies[-1] - Energies[0])/len(Energies)
        
        # Treat each ray refinement level seperatly
        for lref in range(np.min(self.ray_lrefine), np.max(self.ray_lrefine) +1):
            # Size of rays at this refinement level
            ray_dx = self.observer_size_x / 2**float(lref)
            ray_dy = self.observer_size_y / 2**float(lref)
            ray_nx = int(ray_dx/outmap_dx)
            if ray_dx/outmap_dx > ray_nx:
                ray_nx += 1
            ray_ny = int(ray_dy/outmap_dy)
            if ray_dy/outmap_dy > ray_ny:
                ray_ny += 1
            print(lref, ray_ny, ray_nx, ray_dx, ray_dy, outmap_dx, outmap_dy)

            idxs_at_lref = idxs[np.where(self.ray_lrefine[idxs] == lref)]
            nray_remaining = len(idxs_at_lref)
            iray = 0
            while(nray_remaining > 0):
                # grab as many rays as you can
                nrays_now = min(rays_at_a_time, nray_remaining)
                idxs_now = idxs_at_lref[iray:iray+nrays_now]

                # get the fluxes for these rays and apply the window method
                fields = np.zeros((nrays_now, outmap_nfields))
                flux = window_method(Energies, deltaEnergies, self.read_rays(idxs_now, Eidxs))
                fields[:,0] = flux[:,0]
                
                # Determine where in the map the ray is
                map_xstart = self.xp[idxs_now] - 0.5*ray_dx - xlims[0]
                map_xend   = self.xp[idxs_now] + 0.5*ray_dx - xlims[0]

                map_ystart = self.yp[idxs_now] - 0.5*ray_dy - ylims[0]
                map_yend   = self.yp[idxs_now] + 0.5*ray_dy - ylims[0]           

                map_ixstart = np.floor(map_xstart/outmap_dx).astype(int)
                map_ixend   = np.ceil(map_xend/outmap_dx).astype(int)

                map_iystart = np.floor(map_ystart/outmap_dy).astype(int)
                map_iyend   = np.ceil(map_yend/outmap_dy).astype(int)

                ## Bounds
                map_ixstart[map_ixstart<0] = 0
                map_iystart[map_iystart<0] = 0

                map_ixend[map_ixend >= outmap_nx] = outmap_nx-1
                map_iyend[map_iyend >= outmap_ny] = outmap_ny-1

                #determine left and right covering fraction
                fxleft  = ((map_ixstart + 1)*outmap_dx - map_xstart)
                fxright = (map_xend - map_ixend*outmap_dx )
                fyleft  = ((map_iystart + 1)*outmap_dy - map_ystart)
                fyright = (map_yend - map_iyend*outmap_dy)

                fxleft[fxleft < 0 ] = 0
                fyleft[fyleft < 0 ] = 0
                fxright[fxright < 0 ] = 0
                fyright[fyright < 0 ] = 0
            
                fxleft[fxleft > outmap_dx ] = outmap_dx
                fyleft[fyleft > outmap_dy ] = outmap_dy
                fxright[fxright > outmap_dx ] = outmap_dx
                fyright[fyright > outmap_dy ] = outmap_dy

                #print("x", map_ixstart[0], map_xstart[0]/outmap_dx, map_ixend[0], map_xend[0]/outmap_dx, fxleft[0]/outmap_dx, fxright[0]/outmap_dx, fxright[0])
                #print("y", map_iystart[0], map_ystart[0]/outmap_dy, map_iyend[0], map_yend[0]/outmap_dy, fyleft[0]/outmap_dy, fyright[0]/outmap_dy)
 
                # If rays are large enough to fully cover more than one map cell, we need to deal with these seperatly. 
                # TODO: Currently we need to loop over rays for this, but we should find a way to avoid this loop
                #if ray_nx >= 2 or ray_ny >= 2: 
                nx_rays = map_ixend - map_ixstart + 1
                ny_rays = map_ixend - map_ixstart + 1

                for nx_now in range(np.min(nx_rays), np.max(nx_rays)+1):
                    with_nx = nx_rays == nx_now
                    for ny_now in range(np.min(ny_rays), np.max(ny_rays)+1):
                        rays_now = np.where(with_nx*(ny_rays==ny_now))[0]
                        if len(rays_now) == 0:
                            continue
                        ixp , iyp = np.meshgrid(np.arange(0,nx_now), np.arange(0,ny_now), indexing = "ij")
                        ixp = ixp.ravel()
                        iyp = iyp.ravel()
                        ixps = (map_ixstart[rays_now][:,np.newaxis] + ixp[np.newaxis,:]).ravel()
                        iyps = (map_iystart[rays_now][:,np.newaxis] + iyp[np.newaxis,:]).ravel()

                        in_map = (ixps <outmap_nx) * (iyps < outmap_ny)

                        
                        fxl = np.maximum(np.minimum(((ixps+1)*outmap_dx-np.repeat(map_xstart[rays_now],nx_now*ny_now)), outmap_dx), 0)
                        fyl = np.maximum(np.minimum(((iyps+1)*outmap_dy-np.repeat(map_ystart[rays_now],nx_now*ny_now)), outmap_dy), 0)
                        fxr = np.maximum(np.minimum((-(ixps )*outmap_dx+np.repeat(map_xend[rays_now]  ,nx_now*ny_now)), outmap_dx), 0)
                        fyr = np.maximum(np.minimum((-(iyps )*outmap_dy+np.repeat(map_yend[rays_now]  ,nx_now*ny_now)), outmap_dy), 0)
                        fx = np.minimum(fxl,fxr)
                        fy = np.minimum(fyl,fyr)
                        for ifl in range(outmap_nfields):
                            np.add.at(outmap, (ixps[in_map], iyps[in_map], np.full(ixps.shape, ifl)[in_map]),  (np.repeat(fields[rays_now,ifl], nx_now*ny_now)*fx*fy)[in_map])
                            #outmap[ixps, iyps, ifl] += np.repeat(fields[rays_now,ifl], nx_now*ny_now)*fx*fy
#                for iray in range(nrays_now):
#                    # Cells completely covered by the ray
#                    nx_now = map_ixend[iray] - map_ixstart[iray] + 1
#                    ny_now = map_iyend[iray] - map_iystart[iray] + 1
#                    #if nx_now <=2 and ny_now <=2:
#                    #    continue
#
#
#                    xx = np.arange(map_ixstart[iray],map_ixend[iray]+1)
#                    yy = np.arange(map_iystart[iray],map_iyend[iray]+1)
#                    ixps, iyps = np.meshgrid(xx,yy,indexing = "ij")
#                    #print(ixps.shape,nx_now, ny_now)
#                    fxl = np.maximum(np.minimum(((ixps+1)*outmap_dx-map_xstart[iray]),np.ones((nx_now,ny_now))*outmap_dx), np.zeros((nx_now,ny_now)))
#                    fyl = np.maximum(np.minimum(((iyps+1)*outmap_dy-map_ystart[iray]),np.ones((nx_now,ny_now))*outmap_dy), np.zeros((nx_now,ny_now)))
#                    fxr = np.maximum(np.minimum((-(ixps )*outmap_dx+map_xend[iray]  ),np.ones((nx_now,ny_now))*outmap_dx), np.zeros((nx_now,ny_now)))
#                    fyr = np.maximum(np.minimum((-(iyps )*outmap_dy+map_yend[iray]  ),np.ones((nx_now,ny_now))*outmap_dy), np.zeros((nx_now,ny_now)))
#                    fx = np.minimum(fxl,fxr)
#                    fy = np.minimum(fyl,fyr)
#                                     
#                    for ifl in range(outmap_nfields):
#                        outmap[map_ixstart[iray]:map_ixend[iray]+1, map_iystart[iray]:map_iyend[iray]+1, ifl] += fields[iray,ifl]*fx*fy
                nray_remaining = nray_remaining - nrays_now                 

        # Go back to surface brightness
        outmap = outmap/outmap_dx/outmap_dy 
        #print(np.min(outmap), np.max(outmap))
        return outmap
    def read_spec(self, x, y, Elims = None, return_integrated_line = False, return_broadband = True):
        """
            Finds the closest matching ray to a set of xy coordinates and returns its spectra and energies
            arguments:
                x: float (x position to find ray of)
                y: float (y position to find ray of)
                Elims: optional array of two floats (minimum and maximum energy of spectra to return)
            TODO: Make this work for arrays of x and y
        """
        # if we are reading the entire energy range, set limits to cover everything
        if Elims is None:
            Elims = np.array([np.min(self.Energies)-1, np.max(self.Energies) + 1])
        Eidxs = np.where( (self.Energies >= Elims[0]) * (self.Energies < Elims[1]))[0]
        
        # find the closest matching ray
        idx = np.argmin((self.xp - x)**2 + (self.yp - y)**2)
        fluxes = self.read_rays([idx], Eidxs)
        if return_integrated_line :
            if return_broadband:
                return self.Energies[Eidxs], fluxes[0,:], integrated_line(self.Energies[Eidxs], self.deltaEnergies[Eidxs], fluxes)[0], broadband(self.Energies[Eidxs], self.deltaEnergies[Eidxs], fluxes)[0]
            else:
                return self.Energies[Eidxs], fluxes[0,:], integrated_line(self.Energies[Eidxs], self.deltaEnergies[Eidxs], fluxes)[0]
        else:
            if return_broadband:
                return self.Energies[Eidxs], fluxes[0,:], broadband(self.Energies[Eidxs], self.deltaEnergies[Eidxs], fluxes)[0]
            else:
                return self.Energies[Eidxs], fluxes[0,:]