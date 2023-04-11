from importlib.resources import path
import cupy
import cupyx
import numpy as np
import h5py as hp 
import astropy.constants as apyc
import astropy.units as apyu

import sys

from gasspy.settings.defaults import ray_dtypes
from gasspy.raytracing.raytracers import Raytracer_AMR_Base
from gasspy.raytracing.ray_processors.ray_processor_base import Ray_processor_base
from gasspy.io import gasspy_io
from gasspy.raystructures import active_ray_class
class Single_band_radiative_transfer(Ray_processor_base):
    def __init__(self, raytracer : Raytracer_AMR_Base, gasspy_database, cell_gasspy_index, energy_lims, liteVRAM = False):
        self.liteVRAM = liteVRAM
        
        # set and load the database
        self.energy_lims = np.atleast_2d(energy_lims)
        self.nbands = self.energy_lims.shape[0]
        self.gasspy_database = gasspy_database
        self.cell_gasspy_index = cell_gasspy_index
        self.load_database()

        self.raytracer = raytracer
    def load_database(self):
        print("loading database")
        # Load the required energies
        self.energies = self.gasspy_database["energy"][:]
        deltaEnergies = np.zeros(self.energies.shape)
        edgesEnergies = np.zeros((self.energies.shape[0]+1))
        edgesEnergies[1:-1] = (self.energies[1:] + self.energies[:-1])*0.5
        edgesEnergies[0]  = 2*self.energies[0]  - edgesEnergies[1]
        edgesEnergies[-1] = 2*self.energies[-1] - edgesEnergies[-1] 
        
        deltaEnergies[:] = edgesEnergies[1:] - edgesEnergies[:-1]
        
        # Initialize array of models
        em_models = np.zeros((self.gasspy_database["avg_em"].shape[0],  self.nbands))
        op_models = np.zeros((self.gasspy_database["tot_opc"].shape[0], self.nbands))
        for iband in range(self.nbands):
            Eidxs = np.where((edgesEnergies[1:] >= self.energy_lims[iband][0])*(edgesEnergies[:-1]<=self.energy_lims[iband][1]))[0]

            current_energies = self.energies[Eidxs]
            energy_left  = np.maximum(edgesEnergies[Eidxs]    , self.energy_lims[iband][0])
            energy_right = np.minimum(edgesEnergies[Eidxs + 1], self.energy_lims[iband][1])
            current_deltaEnergies = energy_right - energy_left
            # Figure out which gasspyIDs we need and remap to the new "snap specific" database 
            unique_gasspy_ids, self.cell_local_index = np.unique(self.cell_gasspy_index, return_inverse=True)
            # NOTE: np.unique returns sorted indexes. IF THIS EVER CHANGES WE NEED TO SORT MANUALLY

            # Total photons/s/cm^3 emitted from cell within energy range
            em_models[:,iband] = np.sum(current_deltaEnergies*(1*apyu.rydberg).cgs.value*self.gasspy_database["avg_em"] [unique_gasspy_ids, Eidxs[0]:Eidxs[-1]+1]/(current_energies*(1*apyu.rydberg).cgs.value)**2, axis = 1)/(4*np.pi)
            # Average opacity of energy range
            op_models[:,iband] = np.sum(current_deltaEnergies*self.gasspy_database["tot_opc"][unique_gasspy_ids, Eidxs[0]:Eidxs[-1]+1], axis = 1)/(energy_right[-1]-energy_left[0])
        
        self.em = em_models[self.cell_local_index,:]
        self.op = op_models[self.cell_local_index,:]
        if not self.liteVRAM:
            self.em = cupy.asarray(self.em)
            self.op = cupy.asarray(self.op)

        return  

    def process_buff(self, active_rays_indexes_todump, full = False, null_only = False):
        if null_only:
            # This ray processer does not need to save information of dead rays, so nothing happens here
            return
        ## Gather the data from the active_rays that is to be piped to system memory
        # Check if there are any rays to dump (filled or terminated)
        if len(active_rays_indexes_todump) == 0:
            return
        # Get get buffer indexes of finished rays into a cupy array
        indexes_in_buffer = self.raytracer.active_rays.get_field("active_rayDF_to_buffer_map", index = active_rays_indexes_todump, full = full)
        
        # How many ray segments we have in this dump
        NraySegInDump = len(active_rays_indexes_todump)

        # Current number of photons and global_rayid of the dumped rays
        global_rayids   = self.raytracer.active_rays.get_field("global_rayid", index = active_rays_indexes_todump, full = full)
        current_flux             = cupy.zeros((global_rayids.shape[0],self.nbands), dtype = cupy.float64)
        current_optical_depth    = cupy.zeros((global_rayids.shape[0],self.nbands), dtype = cupy.float64)
        for iband in range(self.nbands):
            current_flux[:,iband]    = self.raytracer.global_rays.get_field("photon_count_%d"%iband, index = global_rayids)
            current_optical_depth[:,iband] = self.raytracer.global_rays.get_field("optical_depth_%d"%iband, index = global_rayids)

        # Extract pathlength and cell index from buffer
        pathlength  = self.raytracer.buff_pathlength[indexes_in_buffer,:,cupy.newaxis]*self.raytracer.gasspy_config["sim_unit_length"]
        cell_index  = self.raytracer.buff_cell_index[indexes_in_buffer,:]
        # Solid angle is currently 1/area so no need to scale ray_area by sim_unit_length
        ray_area    = self.raytracer.buff_ray_area[indexes_in_buffer,:, cupy.newaxis]#*self.raytracer.gasspy_config["sim_unit_length"]**2
        solid_angle = self.buff_solid_angle[indexes_in_buffer,:,cupy.newaxis]
        if self.liteVRAM:
            cell_index_cpu = cell_index.get()
            optical_depth  = cupy.cumsum(cupy.asarray(self.op[cell_index_cpu,:])*pathlength,axis = 1) + current_optical_depth[:,cupy.newaxis,:]
            emisivity = cupy.asarray(self.em[cell_index_cpu,:])*ray_area*solid_angle
        else:
            optical_depth  = cupy.cumsum(cupy.asarray(self.op[cell_index,:])*pathlength,axis = 1) + current_optical_depth[:,cupy.newaxis,:]
            emisivity = self.em[cell_index,:]*ray_area*solid_angle
        del ray_area
        current_flux += cupy.sum(emisivity*pathlength*cupy.exp(-optical_depth), axis = 1)
        current_optical_depth = optical_depth[:,-1,:]
        
        # Update the number of photons and opacity of the ray
        for iband in range(self.nbands):
            self.raytracer.global_rays.set_field("photon_count_%d"%iband, current_flux[:,iband],index = global_rayids)
            self.raytracer.global_rays.set_field("optical_depth_%d"%iband, current_optical_depth[:,iband],index = global_rayids)
            
        pass
    
    def init_global_ray_fields(self):
        # Initialize the total number of photons and optical depth of each energy bin
        for iband in range(self.nbands):
            self.raytracer.global_rays.append_field("photon_count_%d" %iband, default_value = 0.0, dtype = cupy.float64)
            self.raytracer.global_rays.append_field("optical_depth_%d"%iband, default_value = 0.0, dtype = cupy.float64)


    def update_sizes(self):
        # We require extra variables (solid angle of pixel + opacity and emissivity for each band)
        self.raytracer.oneRayCell += 64 + 2*64*self.nbands
        self.raytracer.NrayBuff = int(self.raytracer.bufferSizeGPU_GB * 8*1024**3 / (self.raytracer.NcellBuff * self.raytracer.oneRayCell))

        return 

    def alloc_buffer(self):
        self.buff_solid_angle = cupy.zeros((self.raytracer.NrayBuff, self.raytracer.NcellBuff), dtype = ray_dtypes["ray_area"])  
        return 
    def store_in_buffer(self):
        self.buff_solid_angle[self.raytracer.active_rays.get_field("active_rayDF_to_buffer_map"), self.raytracer.active_rays.get_field("buffer_current_step")] = self.raytracer.observer.get_pixel_solid_angle(self.raytracer.active_rays, back_half = True)
        return

    def clean_buff(self, indexes_to_clean):
        self.buff_solid_angle[indexes_to_clean,:] = 0
        return
    
    def reset_buffer(self):
        self.buff_solid_angle[:,:] = 0
        return

    def init_active_ray_fields(self):
        # Reset the active ray data structure as we might be adding a lot of memory
        # TODO: do this smarter
        # set total number of rays
        maxmem_GPU = self.raytracer.bufferSizeGPU_GB
        self.raytracer.oneRayCell = self.raytracer.oneRayCell + 2*64*self.nbands
        self.raytracer.NrayBuff = int(maxmem_GPU*8*1024**3/(4*self.raytracer.NcellBuff*self.raytracer.oneRayCell))
        del self.raytracer.active_rays
        self.raytracer.active_rays = active_ray_class(nrays = self.raytracer.NrayBuff)
        return

    def create_child_fields(self, child_rays, parent_rays):
        for iband in range(self.nbands):
            # Total photon count split into the four children
            field_name = "photon_count_%d"%iband
            child_rays[field_name] = cupy.repeat(self.raytracer.global_rays.get_field(field_name, index = parent_rays["global_rayid"]),4)/4.0
            # Total opacity remains the same as it is an intrinsic variable
            field_name = "optical_depth_%d"%iband
            child_rays[field_name] = cupy.repeat(self.raytracer.global_rays.get_field(field_name, index = parent_rays["global_rayid"]),4)
        return 
        
    def finalize(self):
        # Change to "surface density" of photons, eg the number of photons/s going through a unit solid angle/pixel_area (NOTE STILL IN CODE UNITS)
        for iband in range(self.nbands):
            self.final_flux = (self.raytracer.global_rays.get_field("photon_count_%d"%iband)/self.raytracer.observer.get_ray_area_fraction(self.raytracer.global_rays))
            self.raytracer.global_rays.set_field("photon_count_%d"%iband, self.final_flux)
        return
    