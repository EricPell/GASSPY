import cupy
import cupyx
import numpy as np
import h5py as hp 
import astropy.constants as apyc
import sys

from gasspy.settings.defaults import ray_dtypes
from gasspy.raytracing.ray_processors.ray_processor_base import Ray_processor_base
from gasspy.io import gasspy_io

class Flux_calculator(Ray_processor_base):
    def __init__(self, gasspy_config, raytracer, sim_reader, source_Nphoton, cell_opacity_function, opacity_function_needed_fields, liteVRAM = False):
        
        if isinstance(gasspy_config, str):
            self.gasspy_config = gasspy_io.read_yaml(gasspy_config)
        else:
            self.gasspy_config = gasspy_config
        
        # Should we try to use VRAM sparingly
        self.liteVRAM = gasspy_io.check_parameter_in_config(gasspy_config, "liteVRAM", liteVRAM, True)

        
        # Save opacity function reference
        self.cell_opacity_function = cell_opacity_function
        
        # Load fields needed by opacity funciton
        self.sim_reader = sim_reader
        self.opacity_fields = {}
        for field in opacity_function_needed_fields:
            arr = np.append(sim_reader.get_field(field),0)
            self.opacity_fields[field] = cupyx.zeros_like_pinned(arr)
            self.opacity_fields[field][:] = arr[:]
        self.source_Nphotons = source_Nphoton
        # Create arrays to save photon counts in. size =  Number of cells +1  with the final for NULL values
        if self.liteVRAM:
            self.photon_counts = np.zeros(self.sim_reader.Ncells+1)
        else:
            self.photon_counts = cupy.zeros(self.sim_reader.Ncells+1)

        
        # Tell the raytracer, global_rays and active rays to keep track of all photons, and that its a shared field
        self.raytracer = raytracer
        self.raytracer.shared_column_keys.append("Nphotons")
        self.NcellBuff = self.raytracer.NcellBuff
        self.NrayBuff  = self.raytracer.NrayBuff
        self.NraySegs  = self.raytracer.NraySegs

        self.clght = apyc.c.cgs.value    
        self.sim_unit_length = self.gasspy_config["sim_unit_length"] 

    def process_buff(self, active_rays_indexes_todump, full = False, null_only = False):
        if null_only:
            # This ray processer does not need to save information of dead rays, so nothing happens here
            return
        ## Gather the data from the active_rays that is to be piped to system memory
        # Check if there are any rays to dump (filled or terminated)
        if len(active_rays_indexes_todump) == 0:
            return
        # Get get buffer indexes of finished rays into a cupy array
        indexes_in_buffer = self.raytracer.active_rays.get_field("active_rays_to_buffer_map", index = active_rays_indexes_todump, full = full)
        
        # How many ray segments we have in this dump
        NraySegInDump = len(active_rays_indexes_todump)

        # Current number of photons and global_rayid of the dumped rays
        global_rayids = self.raytracer.active_rays.get_field("global_rayid", index = active_rays_indexes_todump, full = full)
        Nphotons = self.raytracer.active_rays.get_field("Nphotons", index = active_rays_indexes_todump)

        # Find the parents last cell_index
        parent_cell_index = cupy.full(len(active_rays_indexes_todump), -1, dtype = cupy.int64)
        parent_id = self.raytracer.global_rays.get_field("pid", index = global_rayids)
        parent_cell_index[parent_id != -1] = self.raytracer.global_rays.get_field("cell_index", index=parent_id[parent_id !=- 1])


        # Extract pathlength and cell index from buffer
        pathlength  = self.raytracer.buff_pathlength[indexes_in_buffer,:]
        cell_index  = self.raytracer.buff_cell_index[indexes_in_buffer,:]
        ray_area    = self.raytracer.buff_ray_area[indexes_in_buffer,:]

        # If the first cell of the trace matches the parents last cell, then we have already traced this cell with this branch. set to null value
        #same_as_parent = cupy.where(cell_index[:,0]==parent_cell_index)[0]
        #pathlength[same_as_parent,0]=0
        #cell_index[same_as_parent,0]=-1
        #ray_area[same_as_parent,0]=0

        # Calculate the totat number of photons in each cell
        opacity = self.cell_opacity_function(cell_index, self.opacity_fields)
        tau = cupy.cumsum(opacity*pathlength*self.sim_unit_length, axis  = 1)
        total_photons_in_cell = Nphotons[:,np.newaxis]*cupy.exp(-tau)*pathlength*self.sim_unit_length/self.clght 

        if self.liteVRAM:
            np.add_at(self.photon_counts, cell_index.ravel().get(), total_photons_in_cell.ravel().get())
        else:
            cupyx.scatter_add(self.photon_counts, cell_index, total_photons_in_cell)
        # Update photon of ray

        self.raytracer.active_rays.set_field("Nphotons", Nphotons*cupy.exp(-tau[:,-1]), index = active_rays_indexes_todump)
        pass
    
    def init_global_ray_fields(self):
        self.raytracer.global_rays.append_field("Nphotons", default_value = 0.0, dtype = cupy.float64)
        rays_Nphoton = self.source_Nphotons*self.raytracer.observer.get_ray_area_fraction(self.raytracer.global_rays)
        self.raytracer.global_rays.set_field("Nphotons", rays_Nphoton)


    def init_active_ray_fields(self):
        self.raytracer.active_rays.append_field("Nphotons", default_value = 0.0, dtype = cupy.float64)
        return


    def create_child_fields(self, child_rays, parent_rays):
        child_rays["Nphotons"] = cupy.repeat(parent_rays["Nphotons"],4)/4.0
        return 
        
    def finalize(self):
        if self.liteVRAM:
            self.photon_counts = np.maximum(self.photon_counts, 1e-20)
            self.cell_fluxes = 10**(np.log10(self.photon_counts[:-1]*self.clght) - 3*np.log10(self.sim_reader.get_field("dx")))
        else:
            self.photon_counts = cupy.maximum(self.photon_counts, 1e-20)
            self.cell_fluxes = 10**(np.log10(self.photon_counts[:-1].get()*self.clght) - 3*np.log10(self.sim_reader.get_field("dx")))
    
        del self.photon_counts
        pass

    def get_fluxes(self):
        return self.cell_fluxes
