import cupy
import numpy
import h5py as hp 

from gasspy.raystructures import traced_ray_class
from gasspy.raytracing.utils.gpu2cpu import pipeline as gpu2cpu_pipeline
from gasspy.settings.defaults import ray_dtypes
from gasspy.io import gasspy_io
class Raytrace_saver:
    def __init__(self, raytracer):
        self.raytracer = raytracer

        self.NcellBuff = self.raytracer.NcellBuff
        self.NrayBuff  = self.raytracer.NrayBuff
        self.NraySegs  = self.raytracer.NraySegs

        # LOKE CODING: I HAVE NO IDEA OF WHERE WE WANT TO PUT THIS THING....
        # initialize the traced_rays object which stores the trace data on the cpu
        self.traced_rays = traced_ray_class(self.NraySegs, self.NcellBuff, ["pathlength", "index1D","amr_lrefine", "cell_index", "ray_area"])

        # create gpu2cache pipeline objects
        # Instead of calling the internal dtype dictionary, explicitly call the global_ray_dtype to ensure a match.  
        self.pathlength_pipe   = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["pathlength"], self.NcellBuff, "pathlength", self.traced_rays)
        self.index1D_pipe      = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["index1D"],self.NcellBuff, "index1D", self.traced_rays)
        self.cell_index_pipe   = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["cell_index"],self.NcellBuff, "cell_index", self.traced_rays)
        self.amr_lrefine_pipe  = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["amr_lrefine"],self.NcellBuff, "amr_lrefine", self.traced_rays)
        self.ray_area_pipe     = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["ray_area"],self.NcellBuff, "ray_area", self.traced_rays)

    

    def process_buff(self, active_rays_indexes_todump, full = False):
        ## Gather the data from the active_rays that is to be piped to system memory
        # Check if there are any rays to dump (filled or terminated)
        if len(active_rays_indexes_todump) == 0:
            return
        # Get get buffer indexes of finished rays into a cupy array
        indexes_in_buffer = self.raytracer.active_rays.get_field("active_rayDF_to_buffer_map", index = active_rays_indexes_todump, full = full)
        
        # How many ray segments we have in this dump
        NraySegInDump = len(active_rays_indexes_todump)

        # dump number and global_rayid of the dumped rays
        global_rayids = self.raytracer.active_rays.get_field("global_rayid", index = active_rays_indexes_todump, full = full)
        dump_number   = self.raytracer.active_rays.get_field("dump_number",  index = active_rays_indexes_todump, full = full)

        # set the dump number and global_rayid in the traced_rays object
        self.traced_rays.append_indexes(global_rayids, dump_number, NraySegInDump)

        # Extract pathlength and cell 1Dindex from buffer
        tmp_pathlength  = self.raytracer.buff_pathlength[indexes_in_buffer,:]
        tmp_index1D     = self.raytracer.buff_index1D[indexes_in_buffer,:]
        tmp_cell_index  = self.raytracer.buff_cell_index[indexes_in_buffer,:]
        tmp_amr_lrefine = self.raytracer.buff_amr_lrefine[indexes_in_buffer,:]
        tmp_ray_area    = self.raytracer.buff_ray_area[indexes_in_buffer,:]

        # Dump into the raytrace data into the pipelines which then will put it on host memory
        #self.global_rayid_pipe.push(tmp_global_rayid)
        self.pathlength_pipe.push(tmp_pathlength)
        self.index1D_pipe.push(tmp_index1D)
        self.amr_lrefine_pipe.push(tmp_amr_lrefine)
        self.cell_index_pipe.push(tmp_cell_index)
        self.ray_area_pipe.push(tmp_ray_area)
        pass

    def finalize(self):
        self.amr_lrefine_pipe.finalize()
        self.index1D_pipe.finalize()
        self.cell_index_pipe.finalize()
        self.pathlength_pipe.finalize()
        self.ray_area_pipe.finalize()

        # Tell the traced rays object that the trace is done, such that it can trim the data and move it out of pinned memory
        self.traced_rays.finalize_trace()        
        # Generate the mapping from a global_rayid to its ray segment dumps
        self.traced_rays.create_mapping_dict(self.raytracer.global_rays.nrays)
        pass

    def save_trace(self, filename):
        # Open the hdf5 file
        h5file = hp.File(filename, "w")
        # Save the traced rays object for later use in RT
        self.traced_rays.save_hdf5(h5file)
        # Save the global_rays
        self.raytracer.global_rays.save_hdf5(h5file)       
        # Save the gasspy config
        gasspy_io.save_gasspy_config_hdf5(self.raytracer.gasspy_config, h5file)
        # close the file
        h5file.close()

        return

    def add_to_splitEvents(self, split_events):
        self.traced_rays.add_to_splitEvents(split_events)