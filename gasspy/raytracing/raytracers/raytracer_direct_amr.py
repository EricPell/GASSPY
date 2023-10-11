import cupy
import numpy as np
import sys
import h5py
from .raytracer_amr_base import Raytracer_AMR_Base

from gasspy.raytracing.utils.gpu2cpu import pipeline as gpu2cpu_pipeline
from gasspy.raystructures import active_ray_class, traced_ray_class
from gasspy.settings.defaults import ray_dtypes, ray_defaults
from gasspy.raytracing.utils.cuda_kernels import raytrace_low_mem_code_string, raytrace_amr_neighbor_code_string, get_index1D_code_string
from gasspy.shared_utils.functions import sorted_in1d
from gasspy.io import gasspy_io

class Raytracer_AMR(Raytracer_AMR_Base):
     

    def set_raytrace_kernel(self):
        # Initialize the raw kernel for the raytracing (and index1D calculations)
        if self.raytrace_method == "low_memory":
            self.raytrace_code_string = raytrace_low_mem_code_string.format(
            sim_size_half_x = self.sim_size_half[0], sim_size_half_y = self.sim_size_half[1], sim_size_half_z = self.sim_size_half[2])
            self.active_rays.append_field("next_index1D", default_value=ray_defaults["index1D"], dtype=ray_dtypes["index_1D"])
        elif self.raytrace_methd == "neighbor":
            self.raytrace_code_string = raytrace_amr_neighbor_code_string.format(
                sim_size_half_x = self.sim_size_half[0], sim_size_half_y = self.sim_size_half[1], sim_size_half_z = self.sim_size_half[2])  
            self.active_rays.append_field("next_cell_index", default_value=ray_defaults["cell_index"], dtype=ray_dtypes["cell_index"])
        else:
            print("ERROR: non valid raytrace method %s"%self.raytrace_method)
            sys.exit(0)      
        self.raytrace_kernel = cupy.RawKernel(self.raytrace_code_string, '__raytrace_kernel__')

        self.get_index1D_code_string = get_index1D_code_string
        self.get_index1D_kernel = cupy.RawKernel(self.get_index1D_code_string, '__get_index1D__')

        # save reference to sim_data
        self.sim_reader = sim_reader 

        #Save a local GASSPY config
        self.gasspy_config = gasspy_config

    def activate_new_rays(self, N=None):
        # If there are no more rays to add, dont add empty arrays (NaN-tastic)
        if self.global_index_of_last_ray_added + 1 == self.global_rays.nrays:
            return
        if self.global_index_of_last_ray_added + 1+ N >= self.global_rays.nrays:
            # and get more using self.global_index_of_last_ray_added:self.global_index_of_last_ray_added+Navail
            N = self.global_rays.nrays - (self.global_index_of_last_ray_added + 1)

        # Get the global_rayid's of the rays to add to the active_rays
        global_rayids = cupy.arange(self.global_index_of_last_ray_added + 1, self.global_index_of_last_ray_added + 1 + N)
        # create a dict to store the fields in
        new_rays_fields = {}
        for field in self.shared_column_keys:
            new_rays_fields[field] = self.global_rays.get_field(field, index = global_rayids)
        for field in self.new_keys_for_active_rays:
            new_rays_fields[field] = cupy.full(N, ray_defaults[field], dtype = ray_dtypes[field])
        # set the intial guess of the global rayID 
        amr_lrefine_index = new_rays_fields["amr_lrefine"] - self.amr_lrefine_min
        new_rays_fields["next_index1D"] = cupy.zeros(new_rays_fields["xi"].shape, dtype = ray_dtypes["next_index1D"])
        
        blocks_per_grid = ((N  + self.threads_per_block - 1)//self.threads_per_block)

        #find_index1D[blocks_per_grid, self.threads_per_block](
        self.get_index1D_kernel( (blocks_per_grid,), (self.threads_per_block,),(
                                            new_rays_fields["xi"],
                                            new_rays_fields["yi"], 
                                            new_rays_fields["zi"],
                                            new_rays_fields["next_index1D"],
                                            new_rays_fields["amr_lrefine"],
            
                                            self.dx_lref,
                                            self.Nmax_lref,
                                            ray_dtypes["amr_lrefine"](self.amr_lrefine_min),
                                            cupy.int64(N))
        )
        # Get information of where in the buffer these rays will write to
        available_buffer_slot_index = cupy.where(self.buff_slot_occupied == 0)[0][:N]

        # Put buffer slot information into the newRays to be added to the active_rays
        # Returned indexes of where by default will have ... a type that probably is int64, but could change.
        # To ensure that the type uses no more memory than necessary we convert it to the desired buffer_slot_index_type
        new_rays_fields["active_rays_to_buffer_map"] = available_buffer_slot_index.astype(ray_dtypes["active_rays_to_buffer_map"])

        # Set occupation status of the buffer
        self.buff_slot_occupied[available_buffer_slot_index] = 1
        
        # Send the arrays to the active_ray data structure
        indexes = self.active_rays.activate_rays(N, fields = new_rays_fields)

        # Set the current area of the rays
        # given the observer plane, we might have different ways of setting the area of a ray
        # so we set this as a function of the observer plane
        self.observer.set_ray_area(self.active_rays)

        # Let ray processors do their things
        self.ray_processor.update_active_ray_fields()

        # Include the global_rayid of the buffer slot
        #rayids = cupy.array(newRays["global_rayid"].values)
        #self.buff_global_rayid[available_buffer_slot_index,:] = cupy.vstack((rayids for i in range(self.NcellBuff))).T
   
        # validate choice of next index1D
        self.check_amr_level(index = indexes)

        # update the last added ray
        self.global_index_of_last_ray_added+=N

        return

    def raytrace_onestep(self):

        # Determine how many blocks to run 
        blocks_per_grid = ((self.active_rays.nactive  + self.threads_per_block - 1)//self.threads_per_block)
        self.raytrace_kernel((blocks_per_grid,), (self.threads_per_block,), (
                        # Per ray variables
                        self.active_rays.xi, 
                        self.active_rays.yi, 
                        self.active_rays.zi, 
                        self.active_rays.raydir_x, 
                        self.active_rays.raydir_y, 
                        self.active_rays.raydir_z, 
                        self.active_rays.ray_status, 
                        self.active_rays.index1D, 
                        self.active_rays.next_index1D, 
                        self.active_rays.amr_lrefine, 
                        self.active_rays.pathlength,

                        # Global parameters
                        self.Nmax_lref, 
                        self.dx_lref.ravel(), 
                        ray_dtypes["amr_lrefine"](self.amr_lrefine_min), 
                        ray_dtypes["global_rayid"](self.active_rays.nactive)
                        ))
        # Set the cell index
        self.active_rays.set_field("cell_index", self.find_cell_index(self.active_rays.get_field("amr_lrefine"), self.active_rays.get_field("index1D")))
    
        self.get_index1D_kernel( (blocks_per_grid,), (self.threads_per_block,),(
                                            self.active_rays.xi, 
                                            self.active_rays.yi,
                                            self.active_rays.zi,
                                            self.active_rays.next_index1D,
                                            self.active_rays.amr_lrefine,
                                            self.dx_lref.ravel(),
                                            self.Nmax_lref,
                                            ray_dtypes["amr_lrefine"](self.amr_lrefine_min),
                                            ray_dtypes["global_rayid"](self.active_rays.nactive)))
    
        # Update area to be half way point through cell before saving to buffers
        self.observer.update_ray_area(self.active_rays, back_half = True)
       

    
    def store_in_buffer(self):
        # store in buffer        
        self.buff_index1D    [self.active_rays.get_field("active_rays_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("index1D")
        self.buff_pathlength [self.active_rays.get_field("active_rays_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("pathlength")
        self.buff_amr_lrefine[self.active_rays.get_field("active_rays_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("amr_lrefine")
        self.buff_ray_area   [self.active_rays.get_field("active_rays_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("ray_area")
        self.buff_cell_index [self.active_rays.get_field("active_rays_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("cell_index")
        
        self.active_rays.field_add("buffer_current_step", 1)
        # Use a mask, and explicitly set mask dtype. This prevents creating a mask value with the default cudf/cupy dtypes, and saving them to arrays with different dtypes.
        # Currently this just throws warnings if they are different dtypes, but this behavior could be subject to change which may produce errors or worse...
        
        filled_buffer = self.active_rays.get_field("buffer_current_step") == ray_dtypes["buffer_current_step"](self.NcellBuff)
        self.active_rays.set_field("ray_status", ray_dtypes["ray_status"](1), index = cupy.where(filled_buffer))
        pass


    def reset_trace(self):
        # Reset the active_ray structure
        self.active_rays.remove_rays(cupy.arange(self.active_rays.nactive))
        
        # reset the buffers
        self.reset_buffer()

        pass
    def reset_buffer(self):
        # set the buffered fields to their default values and tell their pipes to reset
        for field in ["index1D", "pathlength", "amr_lrefine", "cell_index", "ray_area"]:
            self.__dict__["buff_"+field][:,:] = ray_dtypes[field](ray_defaults[field]) 
            self.__dict__[field+"_pipe"].reset()
        
        # Set all buffer slots as un occupied
        self.buff_slot_occupied[:] = ray_dtypes["buff_slot_occupied"](0)

        # tell the traced rays reset
        self.traced_rays.reset()
        pass

if __name__ == "__main__":
    import numpy as np
    import os
    import importlib.util
    import argparse
    
    from gasspy.raytracing.observers import observer_plane_class
    from gasspy.io import gasspy_io

    ap = argparse.ArgumentParser()
    #-------------DIRECTORIES AND FILES---------------#
    ap.add_argument("--simdir", default="./", help="Directory of the simulation and also default work directory")
    ap.add_argument("--workdir", default= None, help="work directory. If not specified its the same as simdir")
    ap.add_argument("--gasspydir", default="GASSPY", help="directory inside of simdir to put the GASSPY files")
    ap.add_argument("--simulation_reader_dir", default="./", help="directory to the simulation_reader class that describes how to load the simulation")
    ap.add_argument("--sim_prefix", default = None, help="prefix to put before all snapshot specific files")

    ## parse the commandline argument
    args = ap.parse_args()
    
    ## move to workdir
    if args.workdir is not None:
        workdir = args.workdir
    else:
        workdir = args.simdir
    os.chdir(workdir)
    
    ## create GASSPY dir where all files specific to this snapshot is kept
    if not os.path.exists(args.gasspydir):
        sys.exit("ERROR : cant find directory %s"%args.gasspydir)
    
    if not os.path.exists(args.gasspydir+"/projections/"):
        os.makedirs(args.gasspydir+"/projections/")
    
    if not os.path.exists(args.modeldir):
        sys.exit("ERROR : cant find directory %s"%args.modeldir)
    
    ## set prefix to snapshot specific files
    if args.sim_prefix is not None:
        ## add an underscore
        sim_prefix = args.sim_prefix + "_"
    else:
        sim_prefix = ""
    
    ## Load the gasspy_config yaml
    gasspy_config = gasspy_io.read_fluxdef("./gasspy_config.yaml")

    ## Load the simulation data class from directory
    spec = importlib.util.spec_from_file_location("simulation_reader", args.simulation_reader_dir + "/simulation_reader.py")
    reader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader_mod)
    sim_reader = reader_mod.Simulation_Reader(args.simdir, args.gasspydir, gasspy_config["sim_reader_args"]) 
    ## Determine maximum memory usage
    if "max_mem_GPU" in gasspy_config.keys():
        max_mem_GPU = gasspy_config["max_mem_GPU"]
    else:
        max_mem_GPU = 4
    
    if "max_mem_CPU" in gasspy_config.keys():
        max_mem_CPU = gasspy_config["max_mem_CPU"]
    else:
        max_mem_CPU = 14
    
    ## Initialize the raytracer
    raytracer = Raytracer_AMR(sim_reader, gasspy_config, maxMemoryCPU_GB = max_mem_CPU, maxMemoryGPU_GB = max_mem_GPU)

    ## Define the observer class   
    observer = observer_plane_class(gasspy_config)

    ## set observer
    raytracer.update_observer(observer = observer)

    ## run
    print(" - running raytrace")
    raytracer.raytrace_run()

    if args.trace_file is not None:
        trace_file = args.gasspydir+"/projections/"+args.trace_file
    else:
        trace_file = args.gasspydir+"/projections/"+sim_prefix+"trace.hdf5"
    ## save TODO: stop this and just keep in memory
    print(" - saving trace")
    raytracer.save_trace(trace_file)

