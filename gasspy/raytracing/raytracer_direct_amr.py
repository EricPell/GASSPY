import cupy
import numpy as np
import sys
import h5py

from gasspy.raytracing.utils.gpu2cpu import pipeline as gpu2cpu_pipeline
from gasspy.raystructures import active_ray_class, traced_ray_class
from gasspy.settings.defaults import ray_dtypes, ray_defaults
from gasspy.raytracing.utils.cuda_kernels import raytrace_low_mem_code_string, get_index1D_code_string
from gasspy.shared_utils.functions import sorted_in1d



debug_ray = 856890
class raytracer_class:
    def __init__(self, sim_data, obs_plane = None, line_lables = None, savefiles = True, bufferSizeGPU_GB = 4, bufferSizeCPU_GB = 20, NcellBuff  = 64, raster=1, no_ray_splitting = False):
        self.set_new_sim_data(sim_data, line_lables)
        """
            Input:
                sim_data, required - simulation_data_class object containing the needed data from the simulation
                obs_plane          - initial obs_plane definition, can be set later 
                line_labels        - Names of the wanted lines
                savefiles          - Boolean flag if user wants to save the resulting fluxes or not as fits and npys
                NcellBuff          - integer describing the number of cells a ray will hold in the buffer before
                                     before it calculates the cumulative emissions and opacities
        """
        self.threads_per_block = 64
        if obs_plane is not None:
            self.set_obsplane(obs_plane)
        else:
            self.obs_plane = None
        self.savefiles = savefiles

        # These keys need to be intialized at the transfer of an array from the global_rays to the active_rays data structures
        self.new_keys_for_active_rayDF = ["pathlength", "ray_status", "buffer_current_step", "index1D", "next_index1D", "dump_number", "ray_area"]
        # These keys are shared between the global and active rayDF's
        self.shared_column_keys = ["xp","yp","xi", "yi", "zi", "raydir_x", "raydir_y", "raydir_z", "global_rayid", "ray_lrefine", "amr_lrefine", "ray_lrefine"]
        
        # How much memory (in bits) is one element (eg one cell for one ray) in the buffer
        # Bit size per buffer element:
        #   refinement_level : 8 bit int
        #   cell_index       : 64b bit int to describe highly sims with some safety index
        #   path_length      : 64b or... 8 bit in log if someone wants to sacrifies a flop for a lot of memory
        oneRayCell = 208

        # Use this to determine the number of buffer cells we can afford for a given total memory and number of buffered rays.
        self.NcellBuff = NcellBuff
        self.NrayBuff = int(bufferSizeGPU_GB * 8*1024**3 / (4*NcellBuff * oneRayCell))
        self.NraySegs = int(bufferSizeCPU_GB * 8*1024**3 / (NcellBuff * oneRayCell)) 
        
        # Call the method to allocate the buffers and the gpu2cpu pipes
        self.init_active_rays()

        self.no_ray_splitting = no_ray_splitting

    """
        Externally used methods    
    """

    def raytrace_run(self, save = True, saveprefix = None):
        """
            Main method to run the ray tracing     
        """
        assert self.global_rays is not None, "Need to define rays with an observer class"
        # start by moving all cells outside of box to the first intersecting cell
        self.move_to_first_intersection()
        
        # Send as many rays as possible to the active_rayDF and associated buffers
        self.activate_new_rays(N = min(self.NrayBuff, self.global_rays.nrays))

        # transport rays until all rays are outside the box
        i = 0
        self.first_step = cupy.int8(1)
        # changed to a wrapper method call for easier optmization
        while(self.check_trace_status()):
            # transport the rays through the current cell
            self.raytrace_onestep()

            # Move the trace to the buffers
            self.store_in_buffer()

            # Check for consistency with amr grid
            self.check_amr_level(index = None)

            # Determine all rays which are not sufficiently resolving the grid
            if not self.no_ray_splitting:
                self.find_unresolved_rays()

            self.update_rays()
            i+=1

            if i%100 == 0:
                print(i, self.active_rays.nactive, self.global_Nraysfinished, self.global_rays.nrays)

        # Dump the buffer one last time. Should be unneccesary depending on the stopping condition
        self.dump_buff(cupy.arange(self.active_rays.nactive))
        
        # Finalize the pipe 
        self.amr_lrefine_pipe.finalize()
        self.index1D_pipe.finalize()
        self.cell_index_pipe.finalize()
        self.pathlength_pipe.finalize()


        # Tell the traced rays object that the trace is done, such that it can trim the data and move it out of pinned memory
        self.traced_rays.finalize_trace()
        # Generate the mapping from a global_rayid to its ray segment dumps
        self.traced_rays.create_mapping_dict(self.global_rays.nrays)

    def save_trace(self, filename):
        # Open the hdf5 file
        h5file = h5py.File(filename, "w")

        # Save the traced rays object for later use in RT
        self.traced_rays.save_hdf5(h5file)
        # Save the global_rays
        self.global_rays.save_hdf5(h5file)

        # close the file
        h5file.close()

        return
    def set_new_sim_data(self, sim_data, line_lables = None):
        """
            Method to take an observer plane set internal values 
        """

        # AMR and grid definitions
        self.amr_lrefine_min = sim_data.config_yaml["amr_lrefine_min"]
        self.amr_lrefine_max = sim_data.config_yaml["amr_lrefine_max"]
        self.sim_size_x = sim_data.config_yaml["sim_size_x"]
        self.sim_size_y = sim_data.config_yaml["sim_size_y"]
        self.sim_size_z = sim_data.config_yaml["sim_size_z"]

        # midpoint of the simulation
        self.sim_size_half = cupy.array([self.sim_size_x/2, self.sim_size_y/2, self.sim_size_z/2])


        # required resolution of the raytrace
        self.ray_max_area_frac = sim_data.config_yaml["ray_max_area_frac"]

        # Maximum number of cells in a given direction and the size of a cell for a given refinement level
        self.Nmax_lref = 2**(cupy.arange(self.amr_lrefine_min, self.amr_lrefine_max+1)).astype(ray_dtypes["index1D"])
        self.dx_lref = cupy.zeros((len(self.Nmax_lref),3), dtype=ray_dtypes["xi"])  
        self.dx_lref[:,0] = (self.sim_size_x/self.Nmax_lref).astype(ray_dtypes["xi"])    
        self.dx_lref[:,1] = (self.sim_size_y/self.Nmax_lref).astype(ray_dtypes["xi"])
        self.dx_lref[:,2] = (self.sim_size_z/self.Nmax_lref).astype(ray_dtypes["xi"])

        # smallest possible visible area of the cell
        self.cell_smallest_area = cupy.min(self.dx_lref, axis = 1)**2

        # get per cell variables
        self.grid_index1D     = cupy.array(sim_data.get_index1D())
        self.grid_amr_lrefine = cupy.array(sim_data.get_amr_lrefine())
        # remember the associated cell index
        self.grid_cell_index = cupy.arange(len(self.grid_amr_lrefine))

        # split up into per level for easier searching
        self.grid_index1D_lref = []
        self.grid_cell_index_lref = []
        for lref in range(self.amr_lrefine_min, self.amr_lrefine_max + 1):
            at_lref = self.grid_amr_lrefine == lref
            idx_sort = self.grid_index1D[at_lref].argsort()
            self.grid_index1D_lref.append(self.grid_index1D[at_lref][idx_sort].astype(ray_dtypes["index1D"]))        
            self.grid_cell_index_lref.append(self.grid_cell_index[at_lref][idx_sort])        

        # Initialize the raw kernel for the raytracing (and index1D calculations)
        self.raytrace_code_string = raytrace_low_mem_code_string.format(
        sim_size_half_x = self.sim_size_half[0], sim_size_half_y = self.sim_size_half[1], sim_size_half_z = self.sim_size_half[2])
        self.raytrace_kernel = cupy.RawKernel(self.raytrace_code_string, '__raytrace_kernel__')

        self.get_index1D_code_string = get_index1D_code_string
        self.get_index1D_kernel = cupy.RawKernel(self.get_index1D_code_string, '__get_index1D__')

        # save reference to sim_data
        self.sim_data = sim_data  

    def set_obsplane(self, obs_plane):
        """
            Method to take an observer plane set global_rays
        """

        self.update_obsplane(obs_plane)

    def update_obsplane(self, obs_plane, prefix = None):
        """
            Method to take an observer plane and set global_rays if the observer has changed
        """

        self.global_Nraysfinished = 0
        self.global_Nsplit_events = -1
        self.global_index_of_last_ray_added = -1

        # get the first rays from the observation plane classes definitions
        self.global_rays = obs_plane.get_first_rays()

        # save reference to observer plane
        self.obs_plane = obs_plane

        # initialize the number of split
        self.total_Nsplits = 0


    """
        Internally used methods    
    """
    def check_trace_status(self):
        """
            Method to check if we are done with the raytrace
        """

        # See if number of rays finished is less than the number of rays (initial+split) to process
        return (self.global_Nraysfinished < self.global_rays.nrays)

    def init_active_rays(self):
        """
            super call to initialize the active_rayDF and allocate the buffers and pipes for GPU2CPU transfer and storage
        """

        self.active_rays = active_ray_class(nrays = self.NrayBuff)
        self.Nactive = 0
        self.alloc_buffer()


    def update_rays(self):
        # Get list of deactivated ray from the list in the activate data frame called "to_be_deactivated"
        # 
        # set deactivated in global
        # 
        # split rays that have encountered a higher AMR level 
        # and prune from active_rayDF and set number of available slots

        # Get all rays that need to have their buffers dumped, regardles of whether the ray has filled its buffer or is just done
        # and dump these rays to the system memory using the gpu2cpu_pipeline objects
        active_indexes_to_dump = cupy.where(self.active_rays.get_field("ray_status") > 0)[0]
        self.dump_buff(active_indexes_to_dump, full = True)

        # Get all rays which end where a segement interesects a cell which causes the ray to refine
        # and split them
        if not self.no_ray_splitting:
            self.split_rays() 
           
        # find all rays that have terminated
        finished_active_indexes = cupy.where(self.active_rays.get_field("ray_status") > 1)[0]

        # Set the status in the global dataframe of the rays that are to be dropped from the active data frame to finished
        self.global_rays.set_field("trace_status", 2, index = self.active_rays.get_field("global_rayid", finished_active_indexes))
        
        # remove therse rays from the buffer and the active_rays object
        self.prune_active_rays(finished_active_indexes)
        navail = self.active_rays.nrays - self.active_rays.nactive
        if(navail == 0):
            return
        self.activate_new_rays(N=navail)
        pass
    
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
        for field in self.new_keys_for_active_rayDF:
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

        # Put buffer slot information into the newRays to be added to the active_rayDF
        # Returned indexes of where by default will have ... a type that probably is int64, but could change.
        # To ensure that the type uses no more memory than necessary we convert it to the desired buffer_slot_index_type
        new_rays_fields["active_rayDF_to_buffer_map"] = available_buffer_slot_index.astype(ray_dtypes["active_rayDF_to_buffer_map"])
        
        # Set the current area of the rays
        # given the observer plane, we might have different ways of setting the area of a ray
        # so we set this as a function of the observer plane

        # Set occupation status of the buffer
        self.buff_slot_occupied[available_buffer_slot_index] = 1
        
        # Send the arrays to the active_ray data structure
        indexes = self.active_rays.activate_rays(N, fields = new_rays_fields)

        self.obs_plane.set_ray_area(self.active_rays)

        # Include the global_rayid of the buffer slot
        #rayids = cupy.array(newRays["global_rayid"].values)
        #self.buff_global_rayid[available_buffer_slot_index,:] = cupy.vstack((rayids for i in range(self.NcellBuff))).T
   
        # validate choice of next index1D
        self.check_amr_level(index = indexes)

        # update the last added ray
        self.global_index_of_last_ray_added+=N


    def dump_buff(self, active_rays_indexes_todump, full = False):
        ## Gather the data from the active_rays that is to be piped to system memory
        # Check if there are any rays to dump (filled or terminated)
        if len(active_rays_indexes_todump) == 0:
            return
        # Get get buffer indexes of finished rays into a cupy array
        indexes_in_buffer = self.active_rays.get_field("active_rayDF_to_buffer_map", index = active_rays_indexes_todump, full = full)
        
        # How many ray segments we have in this dump
        NraySegInDump = len(active_rays_indexes_todump)

        # dump number and global_rayid of the dumped rays
        global_rayids = self.active_rays.get_field("global_rayid", index = active_rays_indexes_todump, full = full)
        dump_number   = self.active_rays.get_field("dump_number",  index = active_rays_indexes_todump, full = full)

        # set the dump number and global_rayid in the traced_rays object
        self.traced_rays.append_indexes(global_rayids, dump_number, NraySegInDump)

        # Extract pathlength and cell 1Dindex from buffer
        tmp_pathlength  = self.buff_pathlength[indexes_in_buffer,:]
        tmp_index1D     = self.buff_index1D[indexes_in_buffer,:]
        tmp_cell_index  = self.buff_cell_index[indexes_in_buffer,:]
        tmp_amr_lrefine = self.buff_amr_lrefine[indexes_in_buffer,:]

        # Dump into the raytrace data into the pipelines which then will put it on host memory
        #self.global_rayid_pipe.push(tmp_global_rayid)
        self.pathlength_pipe.push(tmp_pathlength)
        self.index1D_pipe.push(tmp_index1D)
        self.amr_lrefine_pipe.push(tmp_amr_lrefine)
        self.cell_index_pipe.push(tmp_cell_index)

        # reset the buffers 
        self.buff_index1D[indexes_in_buffer, :]     = ray_defaults["index1D"]
        self.buff_pathlength[indexes_in_buffer, :]  = ray_defaults["pathlength"]
        self.buff_amr_lrefine[indexes_in_buffer, :] = ray_defaults["amr_lrefine"]
        self.buff_cell_index[indexes_in_buffer,:]   = ray_defaults["cell_index"]

        #TODO: Change the raystatus flag to a bitmask (https://www.sdss.org/dr12/algorithms/bitmasks/) so that we can know both if a ray buffer is filled AND terminated due to boundary or refined, terminated for any reason while it's buffer is full
        # This would be a nice improvement, as the next bit of code would then be removable because we wouldn't need to check for pathlengths when dumping buffers
        #TODO: If above todo is finished remove next
        # rays that are not pruned but are dumped due to filling their buffer has to have their status updated somewhere
        # reset the buffer index of the rays that have been dumped and add one to the number of dumps the ray have done
        self.active_rays.buffer_current_step[active_rays_indexes_todump] = 0
        self.active_rays.dump_number[active_rays_indexes_todump] += 1

        not_terminated = cupy.where(self.active_rays.get_field("ray_status", index = active_rays_indexes_todump) == ray_dtypes["ray_status"](1))[0]       
        self.active_rays.set_field("ray_status", ray_dtypes["ray_status"](0), index = active_rays_indexes_todump[not_terminated])
        pass


    def prune_active_rays(self, indexes_to_drop):
        """
            Gather all rays that need to be deactivatd.
            Deactivate them in the active and global rayDFs
        """
        #If no rays deactivated: do nothing
        if(len(indexes_to_drop) == 0):
            return
        # Use flag to delete part of the buffers
        # get globalIDs of rays to be deactivated.
        # These rays are DONE, no need to refine

        # Newlly opened, but not cleaned slots in the buffer
        buff_slot_newly_freed = self.active_rays.get_field("active_rayDF_to_buffer_map", index = indexes_to_drop)

        self.global_Nraysfinished += len(buff_slot_newly_freed)

        self.buff_slot_occupied[buff_slot_newly_freed] = 0
        # Buffer to store the ray id, since these will become unordered as random rays are added and removed
        #self.buff_global_rayid[buff_slot_newly_freed, :] = -1

        # Remove rays that have terminated
        self.active_rays.remove_rays(indexes_to_drop)

    def move_to_first_intersection(self):
        """
            finds the closest intersection to the simulation cube for each ray outside the cube
        """
        # define origion and normal vector of each of the 6 planes of the data cube
        planes = cupy.array([[ self.sim_size_x, 0, 0, 1, 0, 0],
                             [ 0, 0, 0, -1,  0, 0],
                             [ 0, self.sim_size_y, 0, 0, 1, 0],
                             [ 0, 0, 0, 0, -1, 0],
                             [ 0, 0, self.sim_size_z, 0, 0, 1],
                             [ 0, 0, 0, 0, 0., -1.]])
        # Allocate a minimum pathlength and an array to hold pathlengths to individual planes
        min_pathlength = cupy.full(self.global_rays.nrays, 1e30)
        pathlength = cupy.zeros(self.global_rays.nrays)
        
        # grab the positions and directions of the rays
        xi = self.global_rays.get_field("xi")
        yi = self.global_rays.get_field("yi")
        zi = self.global_rays.get_field("zi")

        raydir_x = self.global_rays.get_field("raydir_x")
        raydir_y = self.global_rays.get_field("raydir_y")
        raydir_z = self.global_rays.get_field("raydir_z")

        # preallocate temporary positions
        tmp_xi = cupy.zeros(xi.shape)
        tmp_yi = cupy.zeros(yi.shape)
        tmp_zi = cupy.zeros(zi.shape)

        for plane in planes:
            p0     = plane[:3]
            nplane = plane[3:]

            #determine the alignment between the rays and the sides of the plane
            align = nplane[0] * raydir_x + nplane[1] * raydir_y + nplane[2] * raydir_z
            # if nplane*p0 = 0 , they never intersect, so skip
            nohit = align == 0
            pathlength[:] = 1e30
            # find the pathlength to the points where the rays intersect the plane
            pathlength[~nohit] = cupy.abs(((p0[0] - xi[~nohit]) * nplane[0] + 
                                          (p0[1] - yi[~nohit]) * nplane[1] + 
                                          (p0[2] - zi[~nohit]) * nplane[2])/align[~nohit])
            
            if cupy.sum(~cupy.isinf(pathlength)) == 0:
                print( "no intersect found with plane with non parallell normal vector", plane, align )
                sys.exit() 
            
            tmp_xi = xi + raydir_x * pathlength*1.001
            tmp_yi = yi + raydir_y * pathlength*1.001
            tmp_zi = zi + raydir_z * pathlength*1.001

            # identify all intersection outside of the simulation domain
            mask = ((tmp_xi < 0) | (tmp_xi > self.sim_size_x) |
                    (tmp_yi < 0) | (tmp_yi > self.sim_size_y) | 
                    (tmp_zi < 0) | (tmp_zi > self.sim_size_z))
            # set these to an unreasonable high number
            pathlength[mask] = 1e30
        
            # if pathlength to current plane is smaller than currently shortest path, replace them
            mask = pathlength < min_pathlength
            min_pathlength[mask] = pathlength[mask]
        # if rays not already in the box,  move rays. If the ray does not intersect the box, it will be put outside and pruned in later stages
        #TODO: This inbox flag should be set with the "domain_check" logic
        inbox = ((xi >= 0) & (xi <= self.sim_size_x) &
                 (yi >= 0) & (yi <= self.sim_size_y) &
                 (zi >= 0) & (zi <= self.sim_size_z))

        idx_outside = cupy.where(~inbox)
        #TODO: This is an important todo. We must resolve starting rays outside of the box in a graceful manner which allows the remainder of the code to assume that rays are in the box until they leave.
        # Identify rays which start outside of the box, and move their position to the box edge, with a buffer for floating point rounding errors.
        # This is for Loke to remember how "where" works: move rays outside of box. cudf where replaces where false     

        for i, ix in enumerate(["x", "y", "z"]):
            self.global_rays.set_field(ix+"i", self.global_rays.get_field(ix+"i", index  = idx_outside) + self.global_rays.get_field("raydir_"+ix, index = idx_outside) * (min_pathlength + 0.001*cupy.sign(min_pathlength)), index = idx_outside) # some padding to ensure cell boundarys are crossed

        #delete unused vairable
        del(tmp_xi)
        del(tmp_yi)
        del(tmp_zi)
        del(pathlength)
        del(min_pathlength)

        return



    def alloc_buffer(self):
        """
            Allocate buffers and pipes to store and transfer the ray-trace from the GPU to the CPU.
        """
        
        dtype_dict = {
            "buff_slot_occupied": ray_dtypes["buff_slot_occupied"],
            "buff_global_rayid":  ray_dtypes["global_rayid"],
            "buff_index1D":       ray_dtypes["index1D"],
            "buff_amr_lrefine":   ray_dtypes["amr_lrefine"],
            "buff_pathlength":    ray_dtypes["pathlength"],
            "buff_cell_index":    ray_dtypes["index1D"]
        }

        # Array to store the occupancy, and inversly the availablity of a buffer
        self.buff_slot_occupied = cupy.zeros(self.NrayBuff, dtype=dtype_dict["buff_slot_occupied"])

        # Buffer to store the ray id, since these will become unordered as random rays are added and removed
        # LOKE finally agreed that Eric was right hahaha: changed to a 2D array such that each ray-cell-intersection has a rayID assosiated with it
        #self.buff_global_rayid = cupy.full((self.NrayBuff, self.NcellBuff), -1, dtype=dtype_dict["buff_global_rayid"])

        # only occupy available buffers with rays to create new buffer
        # changed default index1D from 0 to point to the null value (-1)
        self.buff_index1D    = cupy.full((self.NrayBuff, self.NcellBuff), -1, dtype=dtype_dict["buff_index1D"])
        self.buff_cell_index = cupy.full((self.NrayBuff, self.NcellBuff), -1, dtype=dtype_dict["buff_cell_index"])
        self.buff_pathlength = cupy.zeros((self.NrayBuff, self.NcellBuff), dtype=dtype_dict["buff_pathlength"])
        self.buff_amr_lrefine = cupy.full((self.NrayBuff, self.NcellBuff), -1, dtype=dtype_dict["buff_amr_lrefine"])

        #TODO The following is a total crap shoot. It's a guess for the typical number of times a ray dumpts it's temp buffer while tracing]
        self.guess_ray_dumps = 30


        # LOKE CODING: I HAVE NO IDEA OF WHERE WE WANT TO PUT THIS THING....
        # initialize the traced_rays object which stores the trace data on the cpu
        self.traced_rays = traced_ray_class(self.NraySegs, self.NcellBuff, ["pathlength", "index1D","amr_lrefine", "cell_index"])

        # create gpu2cache pipeline objects
        # Instead of calling the internal dtype dictionary, explicitly call the global_ray_dtype to ensure a match.  
        #self.global_rayid_pipe = gpu2cpu_pipeline(buff_NraySegs, self.global_ray_dtypes["global_rayid"], "global_rayid", buff_elements*self.guess_ray_dumps)
        self.pathlength_pipe   = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["pathlength"], self.NcellBuff, "pathlength", self.traced_rays)
        self.index1D_pipe      = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["index1D"],self.NcellBuff, "index1D", self.traced_rays)
        self.cell_index_pipe   = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["cell_index"],self.NcellBuff, "cell_index", self.traced_rays)
        self.amr_lrefine_pipe  = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["amr_lrefine"],self.NcellBuff, "amr_lrefine", self.traced_rays)
        pass

    def check_amr_level(self, index = None):
        """
            Method to look through all rays in rayDF and find those whose position and current amr_lrefine does not match that of the grid
            and sets the correct amr_lrefine and next_index1D
        """

        # Fill the cupy arrays with the current value of the amr_lrefine and index1D
        active_index1D     = self.active_rays.get_field("next_index1D", index = index)
        active_amr_lrefine = self.active_rays.get_field("amr_lrefine",  index = index)

        # the indexes in the active_rayDF which corresponds to rays that has entered a different amr_lrefine
        lref_incorrect = cupy.full(active_index1D.shape, True, dtype = cupy.bool8)

        #set all rays with nonsensical amr levels to the minimum
        #active_amr_lrefine[active_amr_lrefine < self.amr_lrefine_min]  = self.amr_lrefine_min

        # Loop over all refinement levels and find the rays who dont have a matching cell at their amr level
        for lref in range(self.amr_lrefine_min,self.amr_lrefine_max+1):
            # Grab the rays at the current refinement level
            # self.active_amr_lrefine is a array of refinement levels for all active arrays, and the comparison produces a mask of True/False values
            at_lref = cupy.where(active_amr_lrefine==lref)[0]
            # If there are none, skip
            if len(at_lref) == 0:
                continue
            # grab the indexes of those who we need to find their new amr_lrefine by identifying those that have no matching cell at their current amr_lrefine in the grid
            correct_lref = sorted_in1d(active_index1D[at_lref], self.grid_index1D_lref[lref-self.amr_lrefine_min]) 
            lref_incorrect[at_lref[correct_lref]] = False

        indexes_to_find = cupy.where(lref_incorrect)[0]
        # If we had no rays wrong, return                                                                                                                               
        if len(indexes_to_find) == 0:
            return
        
        # Pre allocate arrays to store the new values of index1D and amr_lrefine in
        index1D_to_find = cupy.zeros(len(indexes_to_find), dtype = ray_dtypes["index1D"])
        amr_lrefine_to_find = cupy.zeros(len(indexes_to_find), dtype = ray_dtypes["amr_lrefine"])
        
        # A mask showing which of the mismatched rays still need to have their host cell amr_lrefine determined
        not_found = cupy.full(len(indexes_to_find), True, dtype = cupy.bool8)
        # Store the current coordinates of these rays in cupy arrays for faster calculations
        if index is None:
            indexes_in_active_rays = indexes_to_find
        else:
            indexes_in_active_rays = index[indexes_to_find]
        xi = self.active_rays.get_field("xi", index = indexes_in_active_rays)
        yi = self.active_rays.get_field("yi", index = indexes_in_active_rays)
        zi = self.active_rays.get_field("zi", index = indexes_in_active_rays)

        # Loop over refinement levels and determine the new amr_lrefine and index1D
        for lref_new in range(self.amr_lrefine_min,self.amr_lrefine_max+1):
            # If there are no more to be found, quit the loop
            N_not_found = int(cupy.sum(not_found))
            if(N_not_found == 0):
                break
            ilref = lref_new - self.amr_lrefine_min
            #Nmax_lref = self.Nmax_lref[ilref]

            index1D_at_lref = cupy.zeros(N_not_found, dtype = ray_dtypes["index1D"])
            blocks_per_grid = ((N_not_found  + self.threads_per_block - 1)//self.threads_per_block)

#            find_index1D[blocks_per_grid, self.threads_per_block](
            self.get_index1D_kernel( (blocks_per_grid,), (self.threads_per_block,),(
                                                xi[not_found], 
                                                yi[not_found],
                                                zi[not_found],
                                                index1D_at_lref,
                                                cupy.full(N_not_found, lref_new, dtype = ray_dtypes["amr_lrefine"]),

                                                self.dx_lref.ravel(),
                                                self.Nmax_lref,
                                                ray_dtypes["amr_lrefine"](self.amr_lrefine_min),
                                                ray_dtypes["global_rayid"](N_not_found))
            )

            # Determine the index1D of the ray at the current amr refinement level 
            index1D_to_find[not_found] = index1D_at_lref
        
            # Determine if this index1D and amr_lrefine has a match in the simulation 
            # From testing it is faster to do the following calculations on all the rays, even if some of them have already been found
            # rather than masking those out before hand. However...
            matches = sorted_in1d(index1D_to_find, self.grid_index1D_lref[ilref])
            # .. We must make sure that we dont accedentially have a match here as an index1D could exist on multiple refinement levels, 
            # just pointing to different parts of the domain
            matches[~not_found] = False

            # If we have no matches on this amr level : skip
            if cupy.sum(matches) == 0:
                continue
            # otherwise, set the refinement level and set the ray to found
            amr_lrefine_to_find[matches] = lref_new
            not_found[matches] = False

        # Grab all of those that were found and update the amr_lrefine and next_index1D.. We still have the rays that strictly speaking has left the box
        # these dont have any matching cells, so we can only grab found and not all
        found = ~not_found
        self.active_rays.set_field("amr_lrefine",  amr_lrefine_to_find[found], index = indexes_in_active_rays[found])
        self.active_rays.set_field("next_index1D", index1D_to_find[found], index = indexes_in_active_rays[found])

        return

    def find_cell_index(self, amr_lrefine, index1D):
        """
            Returns the cell_indexes corresponding to the amr_lrefine and index1D pairs.
        """
        # Initialize an array 
        cell_index = cupy.full(index1D.shape, ray_defaults["cell_index"], dtype = ray_dtypes["cell_index"])

        # Loop over all refinement levels and find the rays who dont have a matching cell at their amr level
        for lref in range(self.amr_lrefine_min,self.amr_lrefine_max+1):
            # Grab the rays at the current refinement level
            at_lref = cupy.where(amr_lrefine==lref)[0]

            # If there are none, skip
            if len(at_lref) == 0:
                continue
            # Find all the valid cells here
            valid = cupy.where(sorted_in1d(index1D[at_lref], self.grid_index1D_lref[lref-self.amr_lrefine_min]))[0]
            # grab the indexes of those who we need to find their new amr_lrefine by identifying those that have no matching cell at their current amr_lrefine in the grid
            cell_index[at_lref[valid]] = self.grid_cell_index_lref[lref - self.amr_lrefine_min][cupy.searchsorted(self.grid_index1D_lref[lref-self.amr_lrefine_min], index1D[at_lref[valid]])]
        return cell_index

    def find_unresolved_rays(self):
        """
            Method to find all rays whos current level of ray_lrefine is not sufficient to resolve
            the local grid and flag them
        """
        # in the case of non-parallell rays, we may need to update the area of each ray as we move along
        # if so, this is also a function that belongs to the observer
        self.obs_plane.update_ray_area(self.active_rays)
        #print(self.cell_smallest_area[self.active_rays.get_field("amr_lrefine") - self.amr_lrefine_min])
        #print(self.active_rays.get_field("ray_area"))
        # check where the area covered by the ray is larger than some fraction of its cell
        unresolved = self.active_rays.get_field("ray_area") > self.ray_max_area_frac * self.cell_smallest_area[self.active_rays.get_field("amr_lrefine") - self.amr_lrefine_min]
        self.active_rays.set_field("ray_status", 3, index = cupy.where(unresolved)[0])
        pass

    def create_child_rays(self, parent_rays, peid, children_global_rayid, parent_aid):
        """
            Method to spawn child rays from a set of parent rays
            parent_rays : Dataframe containing the active parent rays
            peid        : parent split event id 
        """
        nchild = len(peid)*4
        # Since we might have non-paralell rays, we let the observer class do the actual splitting
        child_rays = self.obs_plane.create_child_rays(parent_rays)

        # Set variables related to the raytrace such as ID numbers and refinement levels
        # Any variable that are copied from the parent needs be repeated 4 times 
        
        # Set the parent event id, global id and initialize the child event ID
        child_rays["pevid"] = cupy.repeat(peid, repeats = 4)
        child_rays["pid"]   = cupy.repeat(parent_rays["global_rayid"] , repeats = 4)
        child_rays["cevid"] = cupy.full( nchild, -1, dtype = ray_dtypes["cevid"])
        child_rays["aid"]   = cupy.repeat(parent_aid, repeats = 4)
        
        # Copy the variables that are identical or similar to the parent
        fields_from_parent = ["amr_lrefine", "ray_lrefine"]

        for field in fields_from_parent:
            child_rays[field] = cupy.repeat(parent_rays[field],4)
        
        # Advance the ray_lrefine by 1
        child_rays["ray_lrefine"] += 1

        # append the child dataframe to the global datafram
        self.global_rays.append(nchild, fields = child_rays)
        pass

    def split_rays(self):
        """
            Take all rays that need to be split, spawn their children and add them to the global_rays
            and add to the split event array 
            TODO: currently we assume that a ray wont have to be split twice to match the grid
                  this should be changed 
        """
        # Find out which rays need to be split
        split_termination_indexes = cupy.where(self.active_rays.get_field("ray_status") == 3)[0]
        # and how many we have
        Nsplits = len(split_termination_indexes)

        # if there are no rays to be split, return
        if Nsplits == 0:
            return

        # Grab the parent rays
        parent_rays = self.active_rays.get_subset(split_termination_indexes)
        # Find the global_rayid's of the rays to be split and their ancestral id
        parent_aid  = self.global_rays.get_field("aid", index = parent_rays["global_rayid"])

        # Generate the split event ids (sequential)
        split_event_ids = cupy.arange(self.total_Nsplits, self.total_Nsplits + Nsplits)
        
        # Set the parent child split event id
        self.global_rays.set_field("cevid", split_event_ids, index = parent_rays["global_rayid"])
        
        # allocate the array for the split events. shape = (Nsplit, 5), one parent, 4 children
        split_events = cupy.zeros((Nsplits, 5))

        # set the parent ID
        split_events[:,0] = parent_rays["global_rayid"]

        # Generate the childrens id
        children_global_rayid = cupy.arange(self.global_rays.nrays, self.global_rays.nrays + 4*Nsplits)
        
        # set them in the split events array
        split_events[:,1:] = children_global_rayid.reshape(Nsplits,4)        
        
        # Add to the split event array inside of traced_rays
        self.traced_rays.add_to_splitEvents(split_events)

        # Create the new children arrays and add them to the global_rays
        self.create_child_rays(parent_rays, split_event_ids, children_global_rayid, parent_aid)
        
        # Increment the number of splits
        self.total_Nsplits += Nsplits
        pass            
    
    def raytrace_onestep(self):
        # Determine how many blocks to run 
        blocks_per_grid = ((self.active_rays.nactive  + self.threads_per_block - 1)//self.threads_per_block)

        # Launch kernel to take one step
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
    
        ##self.active_rays.print(cupy.arange(10))
        #if cupy.isin(cupy.array([debug_ray]), self.active_rays.get_field("global_rayid")):
        #    idx = cupy.where(self.active_rays.get_field("global_rayid") == debug_ray)[0]
        #    self.active_rays.debug_ray(idx, ["global_rayid", "zi","pathlength", "amr_lrefine", "index1D", "next_index1D", "ray_status", "active_rayDF_to_buffer_map", "buffer_current_step"])
        #    print(int(self.active_rays.zi[idx]/2**-11)+ int(2**11)*int(self.active_rays.yi[idx]/2**-11)+ int(2**11*2**11)*int(self.active_rays.xi[idx]/2**-11))
        #    sys.exit(0)
    
    def store_in_buffer(self):
        # store in buffer        
        self.buff_index1D    [self.active_rays.get_field("active_rayDF_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("index1D")
        self.buff_pathlength [self.active_rays.get_field("active_rayDF_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("pathlength")
        self.buff_amr_lrefine[self.active_rays.get_field("active_rayDF_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.active_rays.get_field("amr_lrefine")
        self.buff_cell_index [self.active_rays.get_field("active_rayDF_to_buffer_map"), self.active_rays.get_field("buffer_current_step")] = self.find_cell_index(self.active_rays.get_field("amr_lrefine"), self.active_rays.get_field("index1D"))
        
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
        for field in ["index1D", "pathlength", "amr_lrefine", "cell_index"]:
            self.__dict__["buff_"+field][:,:] = ray_dtypes[field](ray_defaults[field]) 
            self.__dict__[field+"_pipe"].reset()
        
        # Set all buffer slots as un occupied
        self.buff_slot_occupied[:] = ray_dtypes["buff_slot_occupied"](0)

        # tell the traced rays reset
        self.traced_rays.reset()
        pass

if __name__ == "__main__":
    from gasspy.shared_utils.simulation_data_lib import simulation_data_class
    from gasspy.raytracing.observers import observer_plane_class
    import numpy as np
    import cProfile 

    no_ray_splitting = True
    save = True
    datadir = "/home/loki/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/GASSPY"
    sim_data = simulation_data_class(datadir = datadir)
    raytracer = raytracer_class(sim_data, savefiles = True, bufferSizeGPU_GB = 4, bufferSizeCPU_GB = 10, NcellBuff  = 32, raster=1, no_ray_splitting=no_ray_splitting)

    nframes = 19

    pitch = np.zeros(nframes)
    yaw   = np.linspace(0,180, nframes) 
    roll  = np.zeros(nframes)

    N_frames = 1
    pr = cProfile.Profile()
    pr.enable()
    for i in range(nframes):
        print(i)
        obsplane = observer_plane_class(sim_data, pitch = pitch[i], yaw = yaw[i], roll = roll[i])
        raytracer.update_obsplane(obs_plane=obsplane)
        raytracer.raytrace_run()
        #raytracer.save_trace(datadir+"/projections/%06d_trace.hdf5"%i)
        raytracer.reset_trace()
    pr.disable()
    pr.dump_stats('profile_ray_struct')