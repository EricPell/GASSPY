import cupy
import numpy as np
import sys
import h5py
from .raytracer_amr_base import Raytracer_AMR_Base

from gasspy.raytracing.utils.gpu2cpu import pipeline as gpu2cpu_pipeline
from gasspy.raystructures import active_ray_class, traced_ray_class
from gasspy.settings.defaults import ray_dtypes, ray_defaults
from gasspy.raytracing.utils.cuda_kernels import raytrace_amr_neighbor_code_string, verify_cell_index_code_string
from gasspy.shared_utils.functions import sorted_in1d
from gasspy.io import gasspy_io

class Raytracer_AMR_neighbor(Raytracer_AMR_Base):
     

    def set_raytrace_kernel(self):
        # Initialize the raw kernel for the raytracing (and index1D calculations)

        self.raytrace_code_string = raytrace_amr_neighbor_code_string.format(
            sim_size_half_x = self.sim_size_half[0], sim_size_half_y = self.sim_size_half[1], sim_size_half_z = self.sim_size_half[2])  

        self.raytrace_kernel = cupy.RawKernel(self.raytrace_code_string, '__raytrace_kernel__')
        
        self.verify_cell_code_string = verify_cell_index_code_string.format(
            sim_size_half_x = self.sim_size_half[0], sim_size_half_y = self.sim_size_half[1], sim_size_half_z = self.sim_size_half[2])  
        self.verify_cell_kernel = cupy.RawKernel(self.verify_cell_code_string, '__verify_cell_kernel__')

    def __kernel_specific_init_active_rays__(self):
        self.active_rays.append_field("outdir", default_value=-1, dtype=ray_dtypes["amr_lrefine"])
        self.active_rays.append_field("cell_center_x", default_value=-1, dtype=ray_dtypes["xi"])
        self.active_rays.append_field("cell_center_y", default_value=-1, dtype=ray_dtypes["xi"])
        self.active_rays.append_field("cell_center_z", default_value=-1, dtype=ray_dtypes["xi"])

        pass


    def __kernel_specific_set_new_sim_data__(self, sim_reader, gasspy_config):
        if self.liteVRAM:
            self.cell_neighbors = sim_reader.get_cell_neighbors()
            self.cell_center_x = sim_reader.get_field("x")/self.sim_unit_length
            self.cell_center_y = sim_reader.get_field("y")/self.sim_unit_length
            self.cell_center_z = sim_reader.get_field("z")/self.sim_unit_length

        else:
            self.cell_neighbors = cupy.array(sim_reader.get_cell_neighbors())
            self.cell_center_x = cupy.array(sim_reader.get_field("x"))/self.sim_unit_length
            self.cell_center_y = cupy.array(sim_reader.get_field("y"))/self.sim_unit_length
            self.cell_center_z = cupy.array(sim_reader.get_field("z"))/self.sim_unit_length
        return

    def verify_cell_position(self, indexes):
        # Get the current cell indexes of these cells
        cell_indexes = self.active_rays.get_field("cell_index", index = indexes)
        # Determine the current cell center position
        if self.liteVRAM:
            cell_indexes_cpu = cell_indexes.get()
            current_cell_center_x = cupy.array(self.cell_center_x[cell_indexes_cpu])
            current_cell_center_y = cupy.array(self.cell_center_y[cell_indexes_cpu])
            current_cell_center_z = cupy.array(self.cell_center_z[cell_indexes_cpu])
        else:
            current_cell_center_x = self.cell_center_x[cell_indexes]
            current_cell_center_y = self.cell_center_y[cell_indexes]
            current_cell_center_z = self.cell_center_z[cell_indexes]
        
        neigh_id = cupy.zeros(indexes.shape, dtype = ray_dtypes["amr_lrefine"])
        N = len(neigh_id)
        blocks_per_grid = ((N  + self.threads_per_block - 1)//self.threads_per_block)

        # verify the position        
        self.verify_cell_kernel( (blocks_per_grid,), (self.threads_per_block,),(
                                            self.active_rays.get_field("xi", index = indexes),
                                            self.active_rays.get_field("yi", index = indexes),
                                            self.active_rays.get_field("zi", index = indexes),
                                            current_cell_center_x, 
                                            current_cell_center_y, 
                                            current_cell_center_z, 
                                            neigh_id,
                                            self.active_rays.get_field("amr_lrefine", index = indexes),
                                            self.dx_lref,
                                            ray_dtypes["amr_lrefine"](self.amr_lrefine_min),
                                            cupy.int64(N)))
        # If distance is larger than half a cell in any direction, we should look for a neighbor
        wrong = cupy.where(neigh_id != -1)[0]
        if(len(wrong > 0)):
            # Update the wrong ones
            if self.liteVRAM:
                new_cell_index = cupy.asarray(self.cell_neighbors[cell_indexes[wrong].get(),neigh_id[wrong].get()])
            else:
                new_cell_index = self.cell_neighbors[cell_indexes[wrong],neigh_id[wrong]]

            self.active_rays.set_field("cell_index", new_cell_index, index = indexes[wrong])
            self.active_rays.set_field("amr_lrefine", self.grid_amr_lrefine[new_cell_index], index = indexes[wrong])

            # Recursive call here to make sure
            self.verify_cell_position(indexes[wrong])

    def set_new_cell_index(self, indexes):
        # set the intial guess of the global rayID 
        index1D = cupy.zeros(indexes.shape, dtype = ray_dtypes["index1D"])
        N = len(index1D)
      
        blocks_per_grid = ((N  + self.threads_per_block - 1)//self.threads_per_block)

        # Calculate the index1D identifier        
        self.get_index1D_kernel( (blocks_per_grid,), (self.threads_per_block,),(
                                            self.active_rays.get_field("xi", index = indexes),
                                            self.active_rays.get_field("yi", index = indexes),
                                            self.active_rays.get_field("zi", index = indexes),
                                            index1D,
                                            self.active_rays.get_field("amr_lrefine", index = indexes),
                                            self.dx_lref,
                                            self.Nmax_lref,
                                            ray_dtypes["amr_lrefine"](self.amr_lrefine_min),
                                            cupy.int64(N))
        )
        # Set it in the active rays
        self.active_rays.set_field("index1D", index1D, index = indexes)
        # validate choice of index1D
        self.check_amr_level(index1D, self.active_rays.get_field("amr_lrefine", index = indexes), index = indexes)
        self.active_rays.set_field("cell_index", self.find_cell_index( 
                                                    self.active_rays.get_field("amr_lrefine", index = indexes),
                                                    self.active_rays.get_field("index1D", index = indexes)), 
                                                 index = indexes)
        return
    def __kernel_specific_activate_new_rays__(self, indexes):

        # Rays where we know of the cell and should just loop through neighbors
        old_rays = cupy.where(self.active_rays.get_field("cell_index", index = indexes) != -1)[0]
        if len(old_rays) > 0:
            self.verify_cell_position(indexes[old_rays])
        # Rays which we need to determine the cell index from scratch
        new_rays = cupy.where(self.active_rays.get_field("cell_index", index = indexes) == -1)[0]
        if len(new_rays) > 0:
            self.set_new_cell_index(indexes[new_rays])

        return


    def __kernel_specific_raytrace_onestep__(self):
        #debug_ray = cupy.where(self.active_rays.get_field("global_rayid") == 8613159)[0]
        #if len(debug_ray)>0:
        #    print("Before")
        #    self.active_rays.debug_ray(debug_ray[0], ["global_rayid", "cell_index","xi", "yi", "zi","raydir_x", "raydir_y", "raydir_z", "cell_center_x", "cell_center_y", "cell_center_z", "pathlength", "outdir"] )
        #    print("dx = ", self.dx_lref[self.active_rays.get_field("amr_lrefine", index = debug_ray[0])-self.amr_lrefine_min,0])
        
        # Set cell centers corresponding to each ray
        if self.liteVRAM:
            cell_indexes_cpu = self.active_rays.get_field("cell_index").get()
            self.active_rays.set_field("cell_center_x", cupy.asarray(self.cell_center_x[cell_indexes_cpu]))
            self.active_rays.set_field("cell_center_y", cupy.asarray(self.cell_center_y[cell_indexes_cpu]))
            self.active_rays.set_field("cell_center_z", cupy.asarray(self.cell_center_z[cell_indexes_cpu]))
        else:
            self.active_rays.set_field("cell_center_x", self.cell_center_x[self.active_rays.get_field("cell_index")])
            self.active_rays.set_field("cell_center_y", self.cell_center_y[self.active_rays.get_field("cell_index")])
            self.active_rays.set_field("cell_center_z", self.cell_center_z[self.active_rays.get_field("cell_index")])
        
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
                        self.active_rays.cell_index,
                        self.active_rays.cell_center_x,
                        self.active_rays.cell_center_y,
                        self.active_rays.cell_center_z,
                        self.active_rays.outdir,
                        self.active_rays.amr_lrefine, 
                        self.active_rays.pathlength,

                        # Global parameters
                        self.Nmax_lref, 
                        self.dx_lref.ravel(), 
                        ray_dtypes["amr_lrefine"](self.amr_lrefine_min), 
                        ray_dtypes["global_rayid"](self.active_rays.nactive)
                        ))
 
        # Update area to be half way point through cell before saving to buffers
        self.observer.update_ray_area(self.active_rays, back_half = True)
  
        # Store in the buffer
        self.store_in_buffer()


        #Update cells of those where ray_status is ok
        #still_alive = cupy.where((self.active_rays.get_field("ray_status")!=2))

        # Update cell index and center
        if self.liteVRAM:
            self.active_rays.set_field("cell_index", cupy.asarray(self.cell_neighbors[self.active_rays.get_field("cell_index").get(), self.active_rays.get_field("outdir").get()]))
        else:
            self.active_rays.set_field("cell_index", self.cell_neighbors[self.active_rays.get_field("cell_index"), self.active_rays.get_field("outdir")])

        # Update amr_lrefine
        self.active_rays.set_field("amr_lrefine", self.grid_amr_lrefine[self.active_rays.get_field("cell_index")])
        
        # If cell index is negative here, it means we've hit a border
        #exit = cupy.where(self.active_rays.get_field("cell_index") == -1)[0]
        #self.active_rays.set_field("ray_status", 2, index = exit)

        # Determine all rays which are not sufficiently resolving the grid    
        if not self.no_ray_splitting:
            self.find_unresolved_rays() 

    def any(self,arr):
        return cupy.any(arr)

    def __store_in_buffer__(self):
        # store in buffer        
        self.buff_pathlength [self.active_rays.get_field("active_rays_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("pathlength")
        self.buff_amr_lrefine[self.active_rays.get_field("active_rays_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("amr_lrefine")
        self.buff_ray_area   [self.active_rays.get_field("active_rays_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("ray_area")
        self.buff_cell_index [self.active_rays.get_field("active_rays_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("cell_index")
        
    def store_in_buffer(self):
        # store in buffer        
        self.__store_in_buffer__()
        self.ray_processor.store_in_buffer()
        self.active_rays.field_add("buffer_current_step", 1)
        # Use a mask, and explicitly set mask dtype. This prevents creating a mask value with the default cudf/cupy dtypes, and saving them to arrays with different dtypes.
        # Currently this just throws warnings if they are different dtypes, but this behavior could be subject to change which may produce errors or worse...

        filled_buffer = (self.active_rays.get_field("buffer_current_step") == ray_dtypes["buffer_current_step"](self.NcellBuff))
        still_alive = (self.active_rays.get_field("ray_status") == 0)
        self.active_rays.set_field("ray_status", ray_dtypes["ray_status"](1), index = cupy.where(filled_buffer*still_alive))
        pass

    def __kernel_specific_create_child_rays__(self, parent_rays, peid, children_global_rayid, parent_aid, child_rays):
        return

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
    raytracer = Raytracer_AMR_neighbor(sim_reader, gasspy_config, bufferSizeCPU_GB = max_mem_CPU, bufferSizeGPU_GB = max_mem_GPU)

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

