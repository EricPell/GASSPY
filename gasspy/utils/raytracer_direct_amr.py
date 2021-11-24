from threading import active_count
import cudf
import cupy
import cupyx
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
from gasspy.utils.gpu2cpu import pipeline as gpu2cpu_pipeline
from gasspy.utils.savename import get_filename 
from ..raystructures.traced_rays import cpu_traced_rays
from ..settings.defaults import ray_dtypes
#from gasspy.utils.reconstructor_test import plot_rays

def __raytrace_kernel__(xi, yi, zi, ray_status, index1D, next_index1D, amr_lrefine, pathlength, raydir, sim_size_half, Nmax_lref, dx_lref, amr_lrefine_min, first_step, verbose):
    # raytrace kernel is the fuction called by cudf.DataFrame.apply_rows
    # xi, yi, zi are the inpuyt columns from a DataFrame
    # pathlength, index1D are the output columns 
    # check if inside the box, otherwise return 0

    for i, (x,y,z) in enumerate(zip(xi, yi, zi)):
        # if we know we are outside the box domain set index1D to NULL value (TODO: fix this value in parameters)

        if ray_status[i] > 0:
            continue


        # Wow, ok, so this line is an if statement to determine if a coordinate position is outside of the simulation domain.
        # It returns 1 if inside the rectangle defined by Nx,Ny,Nz, and 0 if outside, no matter the direction.
        # Domain boundary check using position
        if (abs(x-sim_size_half[0]) <= sim_size_half[0]) * (abs(y-sim_size_half[1]) <= sim_size_half[1]) *  (abs(z-sim_size_half[2]) <= sim_size_half[2]):

                index1D[i] = next_index1D[i]
        else:
            index1D[i] = -1
            if first_step[0] == 0 :
                pathlength[i] = -1
                # Ray status 2 is domain exit
                ray_status[i] = 2
                continue

        # Get the grid data relevant to the current amr level
        iamr = amr_lrefine[i] - amr_lrefine_min
        Nmax = Nmax_lref[iamr]

        # Figure out cell size index on refinement level
        dx = dx_lref[iamr, 0]
        dy = dx_lref[iamr, 1]
        dz = dx_lref[iamr, 2]
        
        ix = x/dx
        iy = y/dy
        iz = z/dz

        # init to unreasonably high number
        pathlength[i] = 1e30
        mindir = -1
        # check for closest distance to cell boundary by looking for the closest int in each cardinal axis away from the current position
        # a thousand of a cell width is added as padding such that the math is (almost) always correct
        # NOTE: this could be wrong if a ray is very close to an interface. So depending on the angle of raydir
        # With respect to the cells, errors can occur

        # in x
        if(raydir[0] > 0):
            newpath = (math.floor(ix) + 1 - ix)*dx/raydir[0]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 0
        elif(raydir[0] < 0):
            newpath = (math.ceil(ix) - 1 - ix)*dx/raydir[0]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 0
        
        # in y
        if(raydir[1] > 0):
            newpath = (math.floor(iy) + 1 - iy)*dy/raydir[1]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 1
        elif(raydir[1] < 0):
            newpath = (math.ceil(iy) - 1 - iy)*dy/raydir[1]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 1

        # in z
        if(raydir[2] > 0):
            newpath = (math.floor(iz) + 1 - iz)*dz/raydir[2]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 2
        elif(raydir[2] < 0):
            newpath = (math.ceil(iz) - 1 - iz)*dz/raydir[2]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 2

        if(mindir == 0):
            # move to next int
            if(raydir[0] > 0):
                xi[i] = (math.floor(ix) + 1)*dx
            else:
                xi[i] = (math.ceil(ix) - 1)*dx
            yi[i] = yi[i] + pathlength[i]*raydir[1]
            zi[i] = zi[i] + pathlength[i]*raydir[2]
            # Set proposed index of the next cell

        if(mindir == 1):
            # move to next int
            if(raydir[1] > 0):
                yi[i] = (math.floor(iy) + 1)*dy
            else:
                yi[i] = (math.ceil(iy) - 1)*dy
            xi[i] = xi[i] + pathlength[i]*raydir[0]
            zi[i] = zi[i] + pathlength[i]*raydir[2]
            

        if(mindir == 2):
            # move to next int
            if(raydir[2] > 0):
                zi[i] = (math.floor(iz) + 1)*dz
            else:
                zi[i] = (math.ceil(iz) - 1)*dz
            xi[i] = xi[i] + pathlength[i]*raydir[0]
            yi[i] = yi[i] + pathlength[i]*raydir[1]

            
        next_index1D[i] = int(iz) + Nmax*int(iy) + Nmax*Nmax*int(ix)
            

class raytracer_class:
    def __init__(self, sim_data, obs_plane = None, line_lables = None, savefiles = True, raytraceBufferSize_GB = 4, NrayBuff  = 1048576, NsysBuff = 4, raster=1):
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
        if obs_plane is not None:
            self.set_obsplane(obs_plane)
        else:
            self.set_empty() 
        self.savefiles = savefiles

        ## Keys and dtypes of the columns in global_rayDF
        self.global_ray_vars = ["xp", "yp", 
            "xi", "yi", "zi",
            "global_rayid",
            "trace_status",
            "pid",         
            "pevid",       
            "cevid",      
            "ray_lrefine",
            "amr_lrefine"
        ]
        self.global_ray_dtypes = { var: ray_dtypes[var] for var in self.global_ray_vars }

        ## Keys and dtypes of the columns in active_rayDF
        self.active_ray_vars = [
            "global_rayid",
            "xp", "yp",
            "xi", "yi", "zi",
            "amr_lrefine",
            "ray_lrefine",
            "index1D",
            "pathlength",
            "ray_status",
            "active_rayDF_to_buffer_map",
            "buffer_current_step",
            "dump_number", 
            "ray_area"
        ]
        self.active_ray_dtypes = { var: ray_dtypes[var] for var in self.active_ray_vars }
        self.active_ray_dtypes["next_index1D"] = ray_dtypes["index1D"]

        # These keys need to be intialized at the transfer of an array from the gloal_rayDF to the active_rayDF
        self.new_keys_for_active_rayDF = ["pathlength", "ray_status", "buffer_current_step", "index1D", "next_index1D", "dump_number", "ray_area"]
        # These keys are shared between the global and active rayDF's
        self.shared_column_keys = ["xp","yp","xi", "yi", "zi", "global_rayid", "ray_lrefine", "amr_lrefine", "ray_lrefine"]
        
        # How much memory (in bits) is one element (eg one cell for one ray) in the buffer
        # Bit size per buffer element:
        #   refinement_level : 8 bit int
        #   cell_index       : 64b bit int to describe highly sims with some safety index
        #   path_length      : 64b or... 8 bit in log if someone wants to sacrifies a flop for a lot of memory
        oneRayCell = 136

        # Use this to determine the number of buffer cells we can afford for a given total memory and number of buffered rays.
        self.NcellBuff = int(raytraceBufferSize_GB * 1024**3 / (oneRayCell * NrayBuff)/NsysBuff)
        self.NrayBuff = NrayBuff

        self.Nraytot_est = 4**(sim_data.amr_lrefine_max - sim_data.amr_lrefine_min) * int((sim_data.Nx * sim_data.Ny * sim_data.Nz)**(2/3)) * raster
        
        # Call the method to allocate the buffers and the gpu2cpu pipes
        self.init_active_rays()

    """
        Externally used methods    
    """

    def raytrace_run(self, saveprefix = None):
        """
            Main method to run the ray tracing     
        """

        assert (self.xps is not None) and (self.yps is not None), "ERROR: and observer needs to be set before you can run ray tracing"

        # start by moving all cells outside of box to the first intersecting cell
        self.move_to_first_intersection()
        
        # Do a soft pruning outside of the box
        self.prune_outside_sim(soft = True)
        
        # Send as many rays as possible to the active_rayDF and associated buffers
        self.activate_new_rays(N = min(self.NrayBuff, self.global_Nrays))

        # transport rays until all rays are outside the box
        i = 0
        self.first_step = cupy.ones(1)
        # changed to a wrapper method call for easier optmization
        while(self.check_trace_status()):
            # transport the rays through the current cell
            self.raytrace_onestep()
            # Check for consistency with amr grid
            self.check_amr_level(self.active_rayDF)
            # Determine all rays which are not sufficiently resolving the grid
            self.find_unresolved_rays()
            # update the active_rayDF if needed
            self.update_rays()
            i+=1
            # Turn off flag for first step
            self.first_step[0] = 0
            print(i, len(self.active_rayDF), self.global_Nraysfinished, self.global_Nrays)
        # Dump the buffer one last time. Should be unneccesary depending on the stopping condition
        self.dump_buff(self.active_rayDF.index)
        
        # Tell the traced rays object that the trace is done, such that it can trim the data and move it out of pinned memory
        self.traced_rays.finalize_trace()
        # Generate the mapping from a global_rayid to its ray segment dumps
        self.traced_rays.create_mapping_dict(self.global_Nrays)
        # Save the traced rays object for later use in RT
        self.traced_rays.save(self.sim_data.datadir + "/" + saveprefix + "_loke_devel.ray")

        # Save the global_rayDF 
        # TODO this should be gathered in some way... Maybe combine put in the traced_rays object and saved in its pickle?
        self.global_rayDF.to_hdf(self.sim_data.datadir + "/" + saveprefix + "_global_rayDF.ray", key = "global_rayDF")
        pass

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

        # split up into per level for easier searching
        self.grid_index1D_lref = [self.grid_index1D[self.grid_amr_lrefine == lref] for lref in range(self.amr_lrefine_min, self.amr_lrefine_max+1)]

        # query string used for dropping rays outside bounds 
        # self.inside_query_string  = "(xi >= 0 and xi <= {0} and yi >= 0 and yi <= {1} and zi >= 0 and zi <= {2})".format(int(self.Nmax[0]),int(self.Nmax[1]), int(self.Nmax[2]))
        self.inside_query_string  = "(pathlength >= 0)"

        self.inside_soft_query_string  = "(xi > {0} and xi < {1} and yi > {2} and yi < {3} and zi > {4} and zi < {5})".format(
                                                                                                                             -float(self.dx_lref[0,0]),                     #xmin - dx
                                                                                                                             self.sim_size_x+float(self.dx_lref[0,0]),      #xmax + dx
                                                                                                                             -float(self.dx_lref[0,1]),                     #ymin - dy 
                                                                                                                             self.sim_size_y+float(self.dx_lref[0,1]),      #ymax + dy
                                                                                                                             -float(self.dx_lref[0,2]),                     #zmin - dz      
                                                                                                                             self.sim_size_z+float(self.dx_lref[0,2])       #zmax + dz
                                                                                                                             )
        # save reference to sim_data
        self.sim_data = sim_data  

    def set_obsplane(self, obs_plane):
        """
            Method to take an observer plane set global_rayDF
        """
        self.xps = 0
        self.yps = 0

        self.update_obsplane(obs_plane)

    def update_obsplane(self, obs_plane, prefix = None):
        """
            Method to take an observer plane and set global_rayDF if the observer has changed
        """

        # if the xps and yps arrays have changed, we need to regenerate everything since the xp yp indices are everywhere
        if not (np.array_equal(obs_plane.xps, cupy.asnumpy(self.xps)) 
            and np.array_equal(obs_plane.yps, cupy.asnumpy(self.yps))):
            self.xps = cupy.array(obs_plane.xps)
            self.yps = cupy.array(obs_plane.xps)
            self.xp, self.yp = cupy.meshgrid(self.xps, self.yps)
            
            self.xp = self.xp.ravel()
            self.yp = self.yp.ravel()

        # Trace status key : 0 is unstarted; 1 is active, 2 is finished
        # parent_global_rayid: -1 no parent; >=0 the int of the globalID
        # parent_event_key : -1 no parent; >=0 
        # child_event_key  : -1 no children; >=0 
        # Rays terminating because of AMR splitting will have children linked to an "split event" which is the child id.
        # Rays originating in an AMR split will have a "split event" which is the parent id.
        self.global_Nrays = len(self.xp)
        self.global_Nraysfinished = 0
        self.global_Nsplit_events = -1
        self.global_index_of_last_ray_added = -1
        self.global_rayDF = cudf.DataFrame({"xp" : self.xp, "yp" : self.yp, 
                                    "xi" : cupy.zeros(self.global_Nrays, dtype=self.global_ray_dtypes["xi"]),
                                    "yi" : cupy.zeros(self.global_Nrays, dtype=self.global_ray_dtypes["yi"]),
                                    "zi" : cupy.zeros(self.global_Nrays, dtype=self.global_ray_dtypes["zi"]),
                                    "global_rayid":cupy.arange(self.global_Nrays, dtype=self.global_ray_dtypes["global_rayid"]),
                                    "trace_status":cupy.zeros(self.global_Nrays, dtype=self.global_ray_dtypes["trace_status"]),
                                    "pid":cupy.full(self.global_Nrays , self.global_Nsplit_events, dtype=self.global_ray_dtypes["pid"]),
                                    "pevid":cupy.full(self.global_Nrays , self.global_Nsplit_events, dtype=self.global_ray_dtypes["pevid"]),
                                    "cevid":cupy.full(self.global_Nrays , self.global_Nsplit_events, dtype=self.global_ray_dtypes["cevid"]),
                                    "amr_lrefine":cupy.full(self.global_Nrays, self.sim_data.amr_lrefine_min, dtype = self.global_ray_dtypes["amr_lrefine"]),
                                    "ray_lrefine":cupy.full(self.global_Nrays, int(np.log2(len(self.xps))), dtype = self.global_ray_dtypes["ray_lrefine"])})
        
        # we assume that everything else has been modified, so ray rotation + translation + first hits are recalculated
        for i, xi in enumerate(["xi", "yi", "zi"]) :

            self.global_rayDF[xi] = cupy.full(len(self.global_rayDF),  obs_plane.xp0_r * float(obs_plane.rotation_matrix[i][0]) +
                                                       obs_plane.yp0_r * float(obs_plane.rotation_matrix[i][1]) + 
                                                       obs_plane.zp0_r * float(obs_plane.rotation_matrix[i][2]) + obs_plane.rot_origin[i])

            self.global_rayDF[xi] += (self.global_rayDF["xp"] * float(obs_plane.rotation_matrix[i][0]) + 
                                      self.global_rayDF["yp"] * float(obs_plane.rotation_matrix[i][1]) )

        # rotate direction
        #TODO: merging different codes, we can encompass the non-planar detector by assigning a per-ray-ray-direction where the normal of each ray is 
        #TODO: contained inside the globl_rayDF.
        self.raydir     = cupy.zeros(3)
        self.raydir_inv = cupy.zeros(3)
        #TODO: End
        ####
        for i in range(3):
            # self.raydir[i] = (obs_plane.view_dir[0] * float(obs_plane.rotation_matrix[i][0]) +
            #                   obs_plane.view_dir[1] * float(obs_plane.rotation_matrix[i][1]) +
            #                   obs_plane.view_dir[2] * float(obs_plane.rotation_matrix[i][2]))

            self.raydir[i] =  1.0 * float(obs_plane.rotation_matrix[i][2])
            
        self.raydir_inv = 1/self.raydir
        
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
        return (self.global_Nraysfinished < self.global_Nrays)

    def init_active_rays(self):
        """
            super call to initialize the active_rayDF and allocate the buffers and pipes for GPU2CPU transfer and storage
        """

        self.active_rayDF = cudf.DataFrame()
        self.alloc_buffer()

    def allocate_global_rayDF(self, N):
        """
            Method to allocate new spots for rays
        """
        # initialize dataframe
        newRays = cudf.DataFrame()
        # initialize arrays for all variables
        for var in self.global_ray_vars:
            newRays[var] = cupy.zeros(N, dtype = ray_dtypes[var])
        # By definition we know the rayid already, so set it
        newRays["global_rayid"] = cupy.arange(self.global_Nrays, self.global_Nrays + N)
        
        # append to the new stuff
        self.global_rayDF = self.global_rayDF.append(newRays, ignore_index=True)

        # increase the global number of rays
        self.global_Nrays += N
    def update_rays(self):
        # Get list of deactivated ray from the list in the activate data frame called "to_be_deactivated"
        # 
        # set deactivated in global
        # 
        # split rays that have encountered a higher AMR level 
        # and prune from active_rayDF and set number of available slots

        # Get all rays that need to have their buffers dumped, regardles of whether the ray has filled its buffer or is just done
        # and dump these rays to the system memory using the gpu2cpu_pipeline objects
        self.dump_buff(self.active_rayDF.query("ray_status > 0").index)

        # Get all rays which end where a segement interesects a cell which causes the ray to refine
        # and split them
        self.split_rays() 
           
        finished_active_rayDF_indexes = self.active_rayDF.query("ray_status > 1").index
        Navail = len(finished_active_rayDF_indexes)
        # If no rays are split or finished, there is nothing left to do
        if Navail == 0:
            return

        # Set the status in the global dataframe of the rays that are to be dropped from the active data frame to finished
        self.global_rayDF["trace_status"].loc[self.active_rayDF["global_rayid"].loc[finished_active_rayDF_indexes]] = 2
        self.prune_active_rays(finished_active_rayDF_indexes)
        self.activate_new_rays(N=Navail)
        pass
    
    def activate_new_rays(self, N=None):
        # If there are no more rays to add, dont add empty arrays (NaN-tastic)
        if self.global_index_of_last_ray_added + 1 == len(self.global_rayDF):
            return
        if self.global_index_of_last_ray_added + 1+ N >= len(self.global_rayDF):
            # and get more using self.global_index_of_last_ray_added:self.global_index_of_last_ray_added+Navail
            N = len(self.global_rayDF) - (self.global_index_of_last_ray_added + 1)

        # Save the new rays into a separate dataframe to easily access them
        newRays = self.global_rayDF[self.shared_column_keys].iloc[self.global_index_of_last_ray_added+1:self.global_index_of_last_ray_added+N+1]

        for key in self.new_keys_for_active_rayDF:
            newRays[key] = cupy.zeros(N, dtype= self.active_ray_dtypes[key])

        # set the intial guess of the global rayID 
        newRays["next_index1D"] = ((newRays["xi"].values/self.dx_lref[newRays["amr_lrefine"].values- self.amr_lrefine_min,0]).astype(int) * self.Nmax_lref[newRays["amr_lrefine"].values - self.amr_lrefine_min]**2 +
                                   (newRays["yi"].values/self.dx_lref[newRays["amr_lrefine"].values- self.amr_lrefine_min,1]).astype(int) * self.Nmax_lref[newRays["amr_lrefine"].values - self.amr_lrefine_min] +
                                   (newRays["zi"].values/self.dx_lref[newRays["amr_lrefine"].values- self.amr_lrefine_min,2]).astype(int))

        # Get information of where in the buffer these rays will write to
        available_buffer_slot_index = cupy.where(self.buff_slot_occupied == 0)[0][:N]

        # Put buffer slot information into the newRays to be added to the active_rayDF
        # Returned indexes of where by default will have ... a type that probably is int64, but could change.
        # To ensure that the type uses no more memory than necessary we convert it to the desired buffer_slot_index_type
        newRays["active_rayDF_to_buffer_map"] = available_buffer_slot_index.astype(self.active_ray_dtypes["active_rayDF_to_buffer_map"])
        
        # Set the current area of the rays
        # given the observer plane, we might have different ways of setting the area of a ray
        # so we set this as a function of the observer plane
        self.obs_plane.set_ray_area(newRays)

        # Set occupation status of the buffer
        self.buff_slot_occupied[available_buffer_slot_index] = 1
        
        # Include the global_rayid of the buffer slot
        rayids = cupy.array(newRays["global_rayid"].values)
        self.buff_global_rayid[available_buffer_slot_index,:] = cupy.vstack((rayids for i in range(self.NcellBuff))).T
   
        # validate choice of next index1D
        self.check_amr_level(newRays)
        # Append newrays to activeDF and save result
        self.active_rayDF = self.active_rayDF.append(newRays, ignore_index=True)
        
        # update the last added ray
        self.global_index_of_last_ray_added+=N


    def dump_buff(self, active_rayDF_indexes_todump):
        ## Gather the data from the active_rayDF that is to be piped to system memory

        # Find the memory indexes (accessed via iloc rather than loc) corresponding to the indexes
        active_rayDF_ilocs_todump = cupy.where(cupy.isin(self.active_rayDF.index.values,active_rayDF_indexes_todump.values))[0]

        # Get get buffer indexes of finished rays into a cupy array
        indexes_in_buffer = cupy.array(self.active_rayDF["active_rayDF_to_buffer_map"].iloc[active_rayDF_ilocs_todump].values)
        
        # How many ray segments we have in this dump
        NraySegInDump = len(active_rayDF_indexes_todump)

        # Check if there are any rays to dump (filled or terminated)
        if NraySegInDump== 0:
            return
        # dump number and global_rayid of the dumped rays
        global_rayids = cupy.array(self.active_rayDF["global_rayid"].iloc[active_rayDF_ilocs_todump].values)
        dump_number   = cupy.array(self.active_rayDF["dump_number"].iloc[active_rayDF_ilocs_todump].values)

        # set the dump number and global_rayid in the traced_rays object
        self.traced_rays.append_indexes(global_rayids, dump_number, NraySegInDump)

        # Extract pathlength and cell 1Dindex from buffer
        tmp_pathlength  = self.buff_pathlength[indexes_in_buffer,:]
        tmp_index1D     = self.buff_index1D[indexes_in_buffer,:]
        tmp_amr_lrefine = self.buff_amr_lrefine[indexes_in_buffer,:]

        # Dump into the raytrace data into the pipelines which then will put it on host memory
        #self.global_rayid_pipe.push(tmp_global_rayid)
        self.pathlength_pipe.push(tmp_pathlength)
        self.index1D_pipe.push(tmp_index1D)
        self.amr_lrefine_pipe.push(tmp_amr_lrefine)

        # reset the buffers 
        self.buff_index1D[indexes_in_buffer, :]    = -1
        self.buff_pathlength[indexes_in_buffer, :] = 0 
        self.buff_amr_lrefine[indexes_in_buffer, :] = 0

        #TODO: Change the raystatus flag to a bitmask (https://www.sdss.org/dr12/algorithms/bitmasks/) so that we can know both if a ray buffer is filled AND terminated due to boundary or refined, terminated for any reason while it's buffer is full
        # This would be a nice improvement, as the next bit of code would then be removable because we wouldn't need to check for pathlengths when dumping buffers
        #TODO: If above todo is finished remove next
        # rays that are not pruned but are dumped due to filling their buffer has to have their status updated somewhere
        # reset the buffer index of the rays that have been dumped and add one to the number of dumps the ray have done
        self.active_rayDF["buffer_current_step"].iloc[active_rayDF_ilocs_todump] = 0
        self.active_rayDF["dump_number"].iloc[active_rayDF_ilocs_todump] += 1 
        self.active_rayDF["ray_status"].mask(self.active_rayDF["ray_status"] == self.active_ray_dtypes["ray_status"](1), self.active_ray_dtypes["ray_status"](0), inplace = True)
        pass

    def prune_outside_sim(self, soft = False):
        """
            Removes all rays that are outside the box 
        """
        if(soft) :
            self.global_rayDF = self.global_rayDF.query(self.inside_soft_query_string)
        else:
            self.global_rayDF = self.global_rayDF.query(self.inside_query_string)

    def prune_active_rays(self, indexes_to_drop):
        """
            Gather all rays that need to be deactivatd.
            Deactivate them in the active and global rayDFs
        """
        # Use flag to delete part of the buffers
        # get globalIDs of rays to be deactivated.
        # These rays are DONE, no need to refine

        # Newlly opened, but not cleaned slots in the buffer
        buff_slot_newly_freed = cupy.array(self.active_rayDF["active_rayDF_to_buffer_map"].loc[indexes_to_drop].values)

        self.global_Nraysfinished += len(buff_slot_newly_freed)

        self.buff_slot_occupied[buff_slot_newly_freed] = 0
        # Buffer to store the ray id, since these will become unordered as random rays are added and removed
        self.buff_global_rayid[buff_slot_newly_freed, :] = -1

        # Remove rays that have terminated
        self.active_rayDF.drop(index=indexes_to_drop, inplace=True)

    def get_remaining(self):
        return len(self.rays.query(self.inside_query_string))



    def set_empty(self):
        """
            Method to initialise all used quantities to None
        """
        self.xps = None
        self.yps = None


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

        min_pathlength = cupy.full(self.xp.shape, 1e30)
        for plane in planes:
            p0     = plane[:3]
            nplane = plane[3:]
            # if nplane*p0 = 0 , they never intersect, so skip
            align = cupy.sum(nplane*self.raydir)
            if align == 0:
                continue
            # find the pathlength to the points where the rays intersect the plane
            pathlength = cupy.abs(((p0[0] - self.global_rayDF["xi"].values) * nplane[0] + 
                                   (p0[1] - self.global_rayDF["yi"].values) * nplane[1] + 
                                   (p0[2] - self.global_rayDF["zi"].values) * nplane[2])/align)
            
            if cupy.sum(~cupy.isinf(pathlength)) == 0:
                print( "no intersect found with plane with non parallell normal vector", plane, align )
                sys.exit() 
            
            self.global_rayDF["tmp_xi"] = self.global_rayDF["xi"] + cudf.Series(self.raydir[0] * pathlength*1.001, index = self.global_rayDF.index)
            self.global_rayDF["tmp_yi"] = self.global_rayDF["yi"] + cudf.Series(self.raydir[1] * pathlength*1.001, index = self.global_rayDF.index)
            self.global_rayDF["tmp_zi"] = self.global_rayDF["zi"] + cudf.Series(self.raydir[2] * pathlength*1.001, index = self.global_rayDF.index)

            # identify all intersection outside of the simulation domain
            mask = ((self.global_rayDF["tmp_xi"].values < 0) | (self.global_rayDF["tmp_xi"].values >= self.sim_size_x) |
                    (self.global_rayDF["tmp_yi"].values < 0) | (self.global_rayDF["tmp_yi"].values >= self.sim_size_y) | 
                    (self.global_rayDF["tmp_zi"].values < 0) | (self.global_rayDF["tmp_zi"].values >= self.sim_size_z))
            # set these to an unreasonable high number
            pathlength[mask] = 1e30
        
            # if pathlength to current plane is smaller than currently shortest path, replace them
            mask = pathlength < min_pathlength
            min_pathlength[mask] = pathlength[mask]
        
        # if rays not already in the box,  move rays. If the ray does not intersect the box, it will be put outside and pruned in later stages
        #TODO: This inbox flag should be set with the "domain_check" logic
        inbox = ((self.global_rayDF["xi"] >= 0) & (self.global_rayDF["xi"] <= self.sim_size_x) &
                 (self.global_rayDF["yi"] >= 0) & (self.global_rayDF["yi"] <= self.sim_size_y) &
                 (self.global_rayDF["zi"] >= 0) & (self.global_rayDF["zi"] <= self.sim_size_z))

        #TODO: This is an important todo. We must resolve starting rays outside of the box in a graceful manner which allows the remainder of the code to assume that rays are in the box until they leave.
        # Identify rays which start outside of the box, and move their position to the box edge, with a buffer for floating point rounding errors.
        # This is for Loke to remember how "where" works: move rays outside of box. cudf where replaces where false     
        for i, ix in enumerate(["xi", "yi", "zi"]):
            self.global_rayDF[ix].where(inbox, self.global_rayDF[ix] + cudf.Series(self.raydir[i] * (min_pathlength + 0.001*cupy.sign(min_pathlength)), index = self.global_rayDF.index), inplace = True) # some padding to ensure cell boundarys are crossed
        
        # Now we clean the data frame of temporary lists we no longer need.
        self.global_rayDF.drop(columns=["tmp_xi","tmp_yi","tmp_zi"], inplace=True)


    def alloc_buffer(self):
        """
            Allocate buffers and pipes to store and transfer the ray-trace from the GPU to the CPU.
        """
        
        dtype_dict = {
            "buff_slot_occupied": ray_dtypes["buff_slot_occupied"],
            "buff_global_rayid":  ray_dtypes["global_rayid"],
            "buff_index1D":       ray_dtypes["index1D"],
            "buff_amr_lrefine":   ray_dtypes["amr_lrefine"],
            "buff_pathlength":    ray_dtypes["pathlength"]
        }

        # Array to store the occupancy, and inversly the availablity of a buffer
        self.buff_slot_occupied = cupy.zeros(self.NrayBuff, dtype=dtype_dict["buff_slot_occupied"])

        # Buffer to store the ray id, since these will become unordered as random rays are added and removed
        # LOKE finally agreed that Eric was right hahaha: changed to a 2D array such that each ray-cell-intersection has a rayID assosiated with it
        self.buff_global_rayid = cupy.full((self.NrayBuff, self.NcellBuff), -1, dtype=dtype_dict["buff_global_rayid"])

        # only occupy available buffers with rays to create new buffer
        # changed default index1D from 0 to point to the null value (-1)
        self.buff_index1D    = cupy.full((self.NrayBuff, self.NcellBuff), -1, dtype=dtype_dict["buff_index1D"])
        self.buff_pathlength = cupy.zeros((self.NrayBuff, self.NcellBuff), dtype=dtype_dict["buff_pathlength"])
        self.buff_amr_lrefine = cupy.full((self.NrayBuff, self.NcellBuff), -1, dtype=dtype_dict["buff_amr_lrefine"])

        #TODO The following is a total crap shoot. It's a guess for the typical number of times a ray dumpts it's temp buffer while tracing]
        self.guess_ray_dumps = 30


        # LOKE CODING: I HAVE NO IDEA OF WHERE WE WANT TO PUT THIS THING....
        # initialize the traced_rays object which stores the trace data on the cpu
        self.traced_rays = cpu_traced_rays(self.Nraytot_est*self.guess_ray_dumps, self.NcellBuff, ["pathlength", "index1D","amr_lrefine"])

        # create gpu2cache pipeline objects
        # Instead of calling the internal dtype dictionary, explicitly call the global_ray_dtype to ensure a match.  
        #self.global_rayid_pipe = gpu2cpu_pipeline(buff_NraySegs, self.global_ray_dtypes["global_rayid"], "global_rayid", buff_elements*self.guess_ray_dumps)
        self.pathlength_pipe   = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["pathlength"], self.NcellBuff, "pathlength", self.traced_rays)
        self.index1D_pipe      = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["index1D"],self.NcellBuff, "index1D", self.traced_rays)
        self.amr_lrefine_pipe  = gpu2cpu_pipeline(self.NrayBuff, ray_dtypes["amr_lrefine"],self.NcellBuff, "amr_lrefine", self.traced_rays)
        pass


    def check_amr_level(self, rayDF):
        """
            Method to look through all rays in rayDF and find those whose position and current amr_lrefine does not match that of the grid
            and sets the correct amr_lrefine and next_index1D
        """
        # These are (will be) the indexes in the active_rayDF which corresponds to rays that has entered a different amr_lrefine
        index_to_find = cudf.Int64Index(data=[])

        # Fill the cupy arrays with the current value of the amr_lrefine and index1D
        self.active_index1D     = rayDF["next_index1D"].values
        self.active_amr_lrefine = rayDF["amr_lrefine"].values
        
        #set all rays with nonsensical amr levels to the minimum
        self.active_amr_lrefine[self.active_amr_lrefine < self.amr_lrefine_min]  = self.amr_lrefine_min

        # Loop over all refinement levels and find the rays who dont have a matching cell at their amr level
        for lref in range(self.amr_lrefine_min,self.amr_lrefine_max+1):
            # Grab the rays at the current refinement level
            # self.active_amr_lrefine is a array of refinement levels for all active arrays, and the comparison produces a mask of True/False values
            at_lref = self.active_amr_lrefine==lref
            rays_at_lref = rayDF[at_lref]
            # If there are none, skip
            if len(rays_at_lref) == 0:
                continue
            # grab the indexes of those who we need to find their new amr_lrefine by identifying those that have no matching cell at their current amr_lrefine in the grid
            index_to_find = index_to_find.append(rays_at_lref[cupy.isin(self.active_index1D[at_lref], self.grid_index1D_lref[lref-self.amr_lrefine_min]) == False].index)
        
        # If we had no rays wrong, return                                                                                                                               
        if len(index_to_find) == 0:
            return

        # Find the indexes where rays-current amr_lrefine differ from the current host-cell amr_lrefine inside of the acttive ray dataframe by matching indexes
        # rays with a refinement level amr_lrefine, but are not in a cell with that refinement level
        iloc_to_find = cupy.where(cupy.isin(rayDF.index.values,index_to_find.values))[0]
        
        # Pre allocate arrays to store the new values of index1D and amr_lrefine in
        index1D_to_find = cupy.zeros(len(iloc_to_find))
        amr_lrefine_to_find = cupy.zeros(len(iloc_to_find), dtype = cupy.int16)
        
        # A mask showing which of the mismatched rays still need to have their host cell amr_lrefine determined
        not_found = cupy.full(len(iloc_to_find), True)
        # Store the current coordinates of these rays in cupy arrays for faster calculations
        coords = rayDF[["xi","yi","zi"]].iloc[iloc_to_find].values
        xi = coords[:,0]
        yi = coords[:,1]
        zi = coords[:,2]

        # Loop over refinement levels and determine the new amr_lrefine and index1D
        for lref_new in range(self.amr_lrefine_min,self.amr_lrefine_max+1):
            # If there are no more to be found, quit the loop
            if(cupy.sum(not_found) == 0):
                break

            # Determine the index1D of the ray at the current amr refinement level 
            index1D_to_find[not_found] = ((xi[not_found]/self.dx_lref[lref_new - self.amr_lrefine_min,0]).astype(int) * self.Nmax_lref[lref_new - self.amr_lrefine_min]**2 +
                                          (yi[not_found]/self.dx_lref[lref_new - self.amr_lrefine_min,1]).astype(int) * self.Nmax_lref[lref_new - self.amr_lrefine_min] +
                                          (zi[not_found]/self.dx_lref[lref_new - self.amr_lrefine_min,2]).astype(int))
        
            # Determine if this index1D and amr_lrefine has a match in the simulation 
            # From testing it is faster to do the following calculations on all the rays, even if some of them have already been found
            # rather than masking those out before hand. However...
            matches = cupy.isin(index1D_to_find, self.grid_index1D_lref[lref_new-self.amr_lrefine_min])
            # .. We must make sure that we dont accedentially have a match here as an index1D could exist on multiple refinement levels, 
            # just pointing to different parts of the domain
            matches[~not_found] = False
            
            # If we have no matches on this amr level : skip
            if cupy.sum(matches) == 0:
                continue
            # otherwise, set the refinement level and set the ray to found
            amr_lrefine_to_find[matches] = lref_new
            not_found[matches] = 0

        # Grab all of those that were found and update the amr_lrefine and next_index1D.. We still have the rays that strictly speaking has left the box
        # these dont have any matching cells, so we can only grab found and not all
        found = ~not_found
        rayDF["amr_lrefine"].iloc[iloc_to_find[found]] = amr_lrefine_to_find[found]
        rayDF["next_index1D"].iloc[iloc_to_find[found]] = index1D_to_find[found]

    def find_unresolved_rays(self):
        """
            Method to find all rays whos current level of ray_lrefine is not sufficient to resolve
            the local grid and flag them
        """
        # in the case of non-parallell rays, we may need to update the area of each ray as we move along
        # if so, this is also a function that belongs to the observer
        self.obs_plane.update_ray_area(self.active_rayDF)

        # check where the area covered by the ray is larger than some fraction of its cell
        unresolved = self.active_rayDF["ray_area"].values > self.ray_max_area_frac * self.cell_smallest_area[self.active_rayDF["amr_lrefine"].values - self.amr_lrefine_min]
        self.active_rayDF["ray_status"][unresolved] = 3 
        pass

    def create_child_rays(self, parent_rays, peid, pid, children_global_rayid):
        """
            Method to spawn child rays from a set of parent rays
            parent_rays : Dataframe containing the active parent rays
            peid        : parent split event id 
        """
        # Since we might have non-paralell rays, we let the observer class do the actual splitting
        childrenDF = self.obs_plane.create_child_rays(parent_rays)
        
        # allocate new rays to the global ray dataframe
        self.allocate_global_rayDF(len(parent_rays)*4)

        # Update the keys we already calculated
        for key in childrenDF.keys():
            self.global_rayDF[key].iloc[children_global_rayid] = childrenDF[key].values

        # Set the parent event id and parent global_ray id for the children
        # Since we have 4 children per ray, we need to repeat each element 4 times
        peid_c = cupy.repeat(peid, repeats = 4)
        self.global_rayDF["pevid"].iloc[children_global_rayid] = peid_c
        pid_c = cupy.repeat(pid, repeats = 4)
        self.global_rayDF["pid"].iloc[children_global_rayid] = pid_c
        # TODO: THIS SHOULD BE AUTOMATIC
        self.global_rayDF["cevid"].iloc[children_global_rayid] = -1
        pass

    def split_rays(self):
        """
            Take all rays that need to be split, spawn their children and add them to the global_rayDF
            and add to the split event array 
            TODO: currently we assume that a ray wont have to be split twice to match the grid
                  this should be changed 
        """
        # Find out which rays need to be split
        split_termination_ilocs = cupy.where(self.active_rayDF["ray_status"].values == 3)[0]
        # and how many we have
        Nsplits = len(split_termination_ilocs)

        # if there are no rays to be split, return
        if Nsplits == 0:
            return

        # Find the global_rayid's of the rays to be split
        parent_global_rayid = self.active_rayDF["global_rayid"].iloc[split_termination_ilocs].values

        # Generate the split event ids (sequential)
        split_event_ids = cupy.arange(self.total_Nsplits, self.total_Nsplits + Nsplits)
        
        # Set the parent child split event id
        self.global_rayDF["cevid"].iloc[parent_global_rayid] = split_event_ids
        
        # allocate the array for the split events. shape = (Nsplit, 5), one parent, 4 children
        split_events = cupy.zeros((Nsplits, 5))

        # set the parent ID
        split_events[:,0] = parent_global_rayid

        # Generate the childrens id
        children_global_rayid = cupy.arange(self.global_Nrays, self.global_Nrays + 4*Nsplits)
        
        # set them in the split events array
        split_events[:,1:] = children_global_rayid.reshape(Nsplits,4)        
        
        # Add to the split event array inside of traced_rays
        self.traced_rays.add_to_splitEvents(split_events)

        # Create the new children arrays and add them to the global_rayDF
        self.create_child_rays(self.active_rayDF.iloc[split_termination_ilocs], split_event_ids, parent_global_rayid, children_global_rayid)
        
        # Increment the number of splits
        self.total_Nsplits += Nsplits
        pass            

    def raytrace_onestep(self):
        verbose = (len(self.active_rayDF) <= 3)
        # find the next intersection to a cell boundary by finding the closest distance to an integer for all directions
        self.active_rayDF = self.active_rayDF.apply_rows(__raytrace_kernel__,
                incols = ["xi", "yi", "zi","ray_status", "index1D", "next_index1D", "amr_lrefine", "pathlength"],
                outcols={},
                kwargs = dict(sim_size_half = self.sim_size_half, raydir = self.raydir, amr_lrefine_min = self.amr_lrefine_min, 
                              Nmax_lref = self.Nmax_lref, dx_lref = self.dx_lref, first_step = self.first_step, verbose = verbose))
        
        # store in buffer
        self.buff_index1D   [self.active_rayDF["active_rayDF_to_buffer_map"].values, self.active_rayDF["buffer_current_step"].values]    = self.active_rayDF["index1D"].values[:]
        self.buff_pathlength[self.active_rayDF["active_rayDF_to_buffer_map"].values, self.active_rayDF["buffer_current_step"].values]    = self.active_rayDF["pathlength"].values[:]
        self.buff_amr_lrefine[self.active_rayDF["active_rayDF_to_buffer_map"].values, self.active_rayDF["buffer_current_step"].values]   = self.active_rayDF["amr_lrefine"].values[:]
        
        self.active_rayDF["buffer_current_step"] += 1
        # Use a mask, and explicitly set mask dtype. This prevents creating a mask value with the default cudf/cupy dtypes, and saving them to arrays with different dtypes.
        # Currently this just throws warnings if they are different dtypes, but this behavior could be subject to change which may produce errors or worse...
        self.active_rayDF["ray_status"].mask(self.active_rayDF["buffer_current_step"] == self.active_ray_dtypes["buffer_current_step"](self.NcellBuff), self.active_ray_dtypes["ray_status"](1), inplace = True)

        pass
