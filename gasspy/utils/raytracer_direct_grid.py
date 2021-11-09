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
from gasspy.utils.reconstructor_test import plot_rays

def __raytrace_kernel__(xi, yi, zi, ray_status, pathlength, index1D, raydir, Nmax, first_step):
    # raytrace kernel is the fuction called by cudf.DataFrame.apply_rows
    # xi, yi, zi are the inpuyt columns from a DataFrame
    # pathlength, index1D are the output columns 
    # check if inside the box, otherwise return 0
    Nx = Nmax[0]
    Ny = Nmax[1]
    Nz = Nmax[2]
    
    Nxhalf = Nx/2
    Nyhalf = Ny/2
    Nzhalf = Nz/2

    for i, (x,y,z) in enumerate(zip(xi, yi, zi)):
        # if we know we are outside the box domain set index1D to NULL value (TODO: fix this value in parameters)

        # Wow, ok, so this line is an if statement to determine if a coordinate position is outside of the simulation domain.
        # It returns 1 if inside the rectangle defined by Nx,Ny,Nz, and 0 if outside, no matter the direction.
        if ray_status[i] > 0:
            continue
        # Domain boundary check using position
        if (abs(x-Nxhalf) <= Nxhalf) * (abs(y-Nyhalf) <= Nyhalf) *  (abs(z-Nzhalf) <= Nzhalf):
            index1D[i] = int(z) + Nz*int(y) + Ny*Nz*int(x)
        else:
            #print(x,y,z)
            index1D[i] = -1
            if first_step[0] == 0 :
                pathlength[i] = -1
                # Ray status 2 is domain exit
                ray_status[i] = 2
                continue

        # init to unreasonably high number
        pathlength[i] = 1e30
        mindir = -1
        # check for closest distance to cell boundary by looking for the closest int in each cardinal axis away from the current position
        # a thousand of a cell width is added as padding such that the math is (almost) always correct
        # NOTE: this could be wrong if a ray is very close to an interface. So depending on the angle of raydir
        # With respect to the cells, errors can occur

        # in x
        if(raydir[0] > 0):
            newpath = (math.floor(x) + 1 - x)/raydir[0]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 0
        elif(raydir[0] < 0):
            newpath = (math.ceil(x) - 1 - x)/raydir[0]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 0
        
        # in y
        if(raydir[1] > 0):
            newpath = (math.floor(y) + 1 - y)/raydir[1]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 1
        elif(raydir[1] < 0):
            newpath = (math.ceil(y) - 1 - y)/raydir[1]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 1

        # in z
        if(raydir[2] > 0):
            newpath = (math.floor(z) + 1 - z)/raydir[2]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 2
        elif(raydir[2] < 0):
            newpath = (math.ceil(z) - 1 - z)/raydir[2]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 2

        if(mindir == 0):
            # move to next int
            
            if(raydir[0] > 0):
                xi[i] = math.floor(x) + 1
            else:
                xi[i] = math.ceil(x) - 1
            yi[i] = yi[i] + pathlength[i]*raydir[1]
            zi[i] = zi[i] + pathlength[i]*raydir[2]
            continue 

        if(mindir == 1):
            # move to next int
            if(raydir[1] > 0):
                yi[i] = math.floor(y) + 1
            else:
                yi[i] = math.ceil(y) - 1
            xi[i] = xi[i] + pathlength[i]*raydir[0]
            zi[i] = zi[i] + pathlength[i]*raydir[2]
            continue

        if(mindir == 2):
            # move to next int
            if(raydir[2] > 0):
                zi[i] = math.floor(z) + 1
            else:
                zi[i] = math.ceil(z) - 1
            xi[i] = xi[i] + pathlength[i]*raydir[0]
            yi[i] = yi[i] + pathlength[i]*raydir[1]
            continue


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

        self.global_ray_dtypes = {
            "xp" : cupy.float64, "yp" : cupy.float64, 
            "xi" : cupy.float64, "yi" : cupy.float64, "zi" : cupy.float64,
            "global_rayid" : cupy.int32,
            "trace_status": cupy.int8,
            "index1D"     : cupy.int64,
            "pathlength"  : cupy.float64,
            "pid"         : cupy.int32,
            "pevid"       : cupy.int32,
            "cevid"       : cupy.int32,
            "refinement_level": cupy.int8
        }

        ## Keys and dtypes of the columns in active_rayDF
        self.active_ray_dtypes = {
            "global_rayid": self.global_ray_dtypes["global_rayid"],
            "xi" : self.global_ray_dtypes["xi"], "yi" : self.global_ray_dtypes["yi"], "zi" : self.global_ray_dtypes["zi"],
            "index1D" : cupy.int64,
            "refinement_level": self.global_ray_dtypes["refinement_level"],
            "pathlength": cupy.float64,
            "ray_status": cupy.int8,
            "active_rayDF_to_buffer_map" : cupy.int32,
            "buffer_current_step": cupy.int16
        }

        # These keys need to be intialized at the transfer of an array from the gloal_rayDF to the active_rayDF
        self.new_keys_for_active_rayDF = ["pathlength", "ray_status", "buffer_current_step", "index1D"]
        # These keys are shared between the global and active rayDF's
        self.shared_column_keys = ["xi", "yi", "zi", "global_rayid", "refinement_level"]
        
        # How much memory (in bits) is one element (eg one cell for one ray) in the buffer
        # Bit size per buffer element:
        #   refinement_level : 8 bit int
        #   cell_index       : 64b bit int to describe highly sims with some safety index
        #   path_length      : 64b or... 8 bit in log if someone wants to sacrifies a flop for a lot of memory
        oneRayCell = 136
        # Use this to determine the number of buffer cells we can afford for a given total memory and number of buffered rays.
        self.NcellBuff = int(raytraceBufferSize_GB * 1024**3 / (oneRayCell * NrayBuff)/NsysBuff)
        self.NrayBuff = NrayBuff

        self.Nraytot_est = 4**(sim_data.maxRef - sim_data.minRef) * int((sim_data.Nx * sim_data.Ny * sim_data.Nz)**(2/3)) * raster
        
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

        #reset fluxes
        for line in self.line_lables :
            self.fluxes[line] = 0

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
            # update the active_rayDF if needed
            self.update_rays()
            i+=1
            # Turn off flag for first step
            self.first_step[0] = 0
        
        self.dump_buff(self.active_rayDF.index)
        pathlengths = self.pathlength_pipe.get_output_array()
        index1D     = self.index1D_pipe.get_output_array()
        print(pathlengths.shape)
        print(np.sum(pathlengths > 0))
        print(index1D.shape)
        print(self.aggregate_toc)
        pass

        return 
    def set_new_sim_data(self, sim_data, line_lables = None):
        """
            Method to take an observer plane set internal values 
        """
        self.line_lables = line_lables
        if self.line_lables is None:
            """Try and read from sim_data config"""
            self.line_lables = sim_data.config_yaml["line_labels"]

        self.subphys_id_df = cudf.DataFrame(sim_data.get_subcell_model_id().ravel())

        #
        self.avg_em_df = cudf.DataFrame(sim_data.subcell_models.DF_from_dict(self.line_lables))
        #avg_ab_df   = cudf.DataFrame(sim_data.subcell_models.avg_ab(line_lables))
        
        # save local variables
        self.Nmax = cupy.array(sim_data.Ncells)
        
        # query string used for dropping rays outside bounds 
        # self.inside_query_string  = "(xi >= 0 and xi <= {0} and yi >= 0 and yi <= {1} and zi >= 0 and zi <= {2})".format(int(self.Nmax[0]),int(self.Nmax[1]), int(self.Nmax[2]))
        self.inside_query_string  = "(pathlength >= 0)"

        self.inside_soft_query_string  = "(xi > -1 and xi < {0} and yi > -1 and yi < {1} and zi > -1 and zi < {2})".format(int(self.Nmax[0]+1),int(self.Nmax[1]+1), int(self.Nmax[2]+1))
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
            
            # regenerate ray data frame, and fluxes and opacities
            self.fluxes   = cudf.DataFrame({"xp": self.xp, "yp" : self.yp})
            #self.opacity = cudf.DataFrame({"xp": xp.ravel(), "yp" : yp.ravel()})
            
            for line_lable in self.line_lables :
                self.fluxes[line_lable] = cupy.zeros(self.xp.shape)
                #self.opacity[line_lable] = cupy.zeros(self.xp.shape)
            self.fluxes.set_index(["xp", "yp"], inplace = True) 
            #self.opacity.set_index(["xp", "yp"], inplace = True) 

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
                                    "refinement_level":cupy.full(self.global_Nrays, self.sim_data.minRef, dtype=self.global_ray_dtypes["refinement_level"])})
        
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

    def split_rays(self,split_termination_indexes):
        pass

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
        split_termination_indexes = self.active_rayDF.query("ray_status == 3").index

        if len(split_termination_indexes) > 0:
           self.split_rays(split_termination_indexes) # whatever it is
           
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
        if self.global_index_of_last_ray_added + N > len(self.global_rayDF):
            # and get more using self.global_index_of_last_ray_added:self.global_index_of_last_ray_added+Navail
            N = len(self.global_rayDF) - self.global_index_of_last_ray_added

        # Save the new rays into a separate dataframe to easily access them
        newRays = self.global_rayDF[self.shared_column_keys].iloc[self.global_index_of_last_ray_added+1:self.global_index_of_last_ray_added+N+1]

        for key in self.new_keys_for_active_rayDF:
            newRays[key] = cupy.zeros(N, dtype= self.active_ray_dtypes[key])

        # Get information of where in the buffer these rays will write to
        available_buffer_slot_index = cupy.where(self.buff_slot_occupied == 0)[0][:N]

        # Put buffer slot information into the newRays to be added to the active_rayDF
        # Returned indexes of where by default will have ... a type that probably is int64, but could change.
        # To ensure that the type uses no more memory than necessary we convert it to the desired buffer_slot_index_type
        newRays["active_rayDF_to_buffer_map"] = available_buffer_slot_index.astype(self.active_ray_dtypes["active_rayDF_to_buffer_map"])

        # Set occupation status of the buffer
        self.buff_slot_occupied[available_buffer_slot_index] = 1
        
        # Include the global_rayid of the buffer slot
        rayids = cupy.array(newRays["global_rayid"].values)
        self.buff_global_rayid[available_buffer_slot_index,:] = cupy.vstack((rayids for i in range(self.NcellBuff))).T
        # Append newrays to activeDF and save result
        self.active_rayDF = self.active_rayDF.append(newRays)
        
        # update the last added ray
        self.global_index_of_last_ray_added+=N


    def dump_buff(self, active_rayDF_indexes_todump):
        ## Gather the data from the active_rayDF that is to be piped to system memory
        # Get get buffer indexes of finished rays into a cupy array
        indexes_in_buffer = cupy.array(self.active_rayDF["active_rayDF_to_buffer_map"].loc[active_rayDF_indexes_todump].values)

        # Check if there are any rays to dump (filled or terminated)
        if len(indexes_in_buffer) == 0:
            return
        # Extract pathlength and cell 1Dindex from buffer
        tmp_pathlength = self.buff_pathlength[indexes_in_buffer,:]
        tmp_index1D = self.buff_index1D[indexes_in_buffer,:]
        tmp_global_rayid = self.buff_global_rayid[indexes_in_buffer,:]

        # get global rayid of rays that are in the current dumps
        transfered_global_rayid = tmp_global_rayid[:, 0]

        # How many cells each ray has traced in the dump
        # LOKE DEBUG: TODO: THIS DOES NOT WORK. WE NEED TO CALCULATE THIS AND REMOVE CELLS THAT ARE MASKED OUT DUE TO BEING EMPTY
        # reality check: There is no reason we have to cut or mask out the last empties. In a ray with multiple dumps, only the last dump will have any padding. That's an acceptable amount of trimming to fix later
        # for the benefit of dumping uniform buffer sections. Those can be dealt with in ray-reconstruction, or never, using the index to for the null-physics cell.
        Ntransfered= cupy.array(self.active_rayDF["buffer_current_step"].loc[active_rayDF_indexes_todump].values)

        # How many rays we have in this dump
        NraysInDump = len(active_rayDF_indexes_todump)

        # ravel the arrays to be transfered into 1d
        tmp_pathlength = tmp_pathlength.ravel()
        tmp_index1D = tmp_index1D.ravel()
        tmp_global_rayid = tmp_global_rayid.ravel()

        # get the filled portion of the buffer
        mask = tmp_pathlength > 0

        # mask them out the unfilled
        tmp_pathlength = tmp_pathlength[mask]
        tmp_index1D = tmp_index1D[mask]
        tmp_global_rayid = tmp_global_rayid[mask]

        #TODO: Change the raystatus flag to a bitmask (https://www.sdss.org/dr12/algorithms/bitmasks/) so that we can know both if a ray buffer is filled AND terminated due to boundary or refined, terminated for any reason while it's buffer is full
        # This would be a nice improvement, as the next bit of code would then be removable because we wouldn't need to check for pathlengths when dumping buffers
        #TODO: If above todo is finished remove next
        # So if no cells are there to dump, just return
        if(len(tmp_pathlength) == 0):
            return
        ##

        # Start and end of the current ray dump
        ray_start = cupy.cumsum(Ntransfered) + self.start_dump
        # explicitly calculate ray_end here as it is used in multiple locations
        ray_end = ray_start + Ntransfered
        # save the information of the dumped ray segments into the table of contents
        # each ray segment is saved with the following information
        # 1) transfered_global_rayid: The global ray ID of the ray
        # 2) ray_start: The start index of this segment in the final output arrays on the host memory
        # 3) ray_end: = ray_start + Ntransfered: The end index of this segment in the final output arrays on the host memory 
        self.aggregate_toc[self.toc_length: self.toc_length + NraysInDump, :] = cupy.array([transfered_global_rayid, ray_start, ray_end]).transpose()
        self.toc_length += NraysInDump

        # Dump into the raytrace data into the pipelines which then will put it on host memory
        self.global_rayid_pipe.push(tmp_global_rayid)
        self.pathlength_pipe.push(tmp_pathlength)
        self.index1D_pipe.push(tmp_index1D)
        #self.lrefine_pipe.push(tmp_lrefine)

        # added forgotten counter for how many cells we have dumped in total
        # The next index in the dump will be were the last ray ended
        self.start_dump = ray_end[-1]

        # rays that are not pruned but are dumped due to filling their buffer has to have their status updated somewhere
        # reset the buffer index of the rays that have been dumped
        self.active_rayDF["buffer_current_step"].loc[active_rayDF_indexes_todump] = 0
        self.active_rayDF["ray_status"].mask(self.active_rayDF["ray_status"] == self.active_ray_dtypes["ray_status"](1), self.active_ray_dtypes["ray_status"](0), inplace = True)
        pass

    def prune_outside_sim(self, soft = False):
        """
            Removes all rays that are outside the box 
            DEPRICATED
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

        # only occupy available buffers with rays to create new buffer
        self.buff_index1D[buff_slot_newly_freed, :]    = -1
        self.buff_pathlength[buff_slot_newly_freed, :] = 0 

        # Remove rays that have terminated
        self.active_rayDF.drop(index=indexes_to_drop, inplace=True)

    def get_remaining(self):
        return len(self.rays.query(self.inside_query_string))

    def save_lines_fluxes(self, saveprefix = None):
        os.makedirs("%s/gasspy_output/"%(self.sim_data.datadir), exist_ok=True)
        for line in self.line_lables:
            flux_array = cupy.array(self.fluxes[line])
            flux_array = cupy.asnumpy(flux_array)
            fname = "%s/gasspy_output/%s.npy"%(self.sim_data.datadir, get_filename(line, self.sim_data, self.obs_plane, saveprefix = saveprefix))
            print("saving " + fname)
            np.save(fname, flux_array.reshape(self.obs_plane.Nxp, self.obs_plane.Nyp))
        del(flux_array)


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
        planes = cupy.array([[ float(self.Nmax[0]), 0, 0, 1, 0, 0],
                             [ 0, 0, 0, -1,  0, 0],
                             [ 0, float(self.Nmax[1]), 0, 0, 1, 0],
                             [ 0, 0, 0, 0, -1, 0],
                             [ 0, 0, float(self.Nmax[2]), 0, 0, 1],
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
            mask = ((self.global_rayDF["tmp_xi"].values < 0) | (self.global_rayDF["tmp_xi"].values >= self.Nmax[0]) |
                    (self.global_rayDF["tmp_yi"].values < 0) | (self.global_rayDF["tmp_yi"].values >= self.Nmax[1]) | 
                    (self.global_rayDF["tmp_zi"].values < 0) | (self.global_rayDF["tmp_zi"].values >= self.Nmax[2]))
            # set these to an unreasonable high number
            pathlength[mask] = 1e30
        
            # if pathlength to current plane is smaller than currently shortest path, replace them
            mask = pathlength < min_pathlength
            min_pathlength[mask] = pathlength[mask]
        
        # if rays not already in the box,  move rays. If the ray does not intersect the box, it will be put outside and pruned in later stages
        #TODO: This inbox flag should be set with the "domain_check" logic
        inbox = ((self.global_rayDF["xi"] >= 0) & (self.global_rayDF["xi"] <= int(self.Nmax[0])) &
                 (self.global_rayDF["yi"] >= 0) & (self.global_rayDF["yi"] <= int(self.Nmax[1])) &
                 (self.global_rayDF["zi"] >= 0) & (self.global_rayDF["zi"] <= int(self.Nmax[2])))

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
            "buff_slot_occupied":cupy.int8,
            "buff_global_rayid":self.global_ray_dtypes["global_rayid"],
            "buff_index1D": self.global_ray_dtypes["index1D"],
            "buff_pathlength":self.global_ray_dtypes["pathlength"],
            "aggregate_toc" : cupy.int64
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

        #TODO The following is a total crap shoot. It's a guess for the typical number of times a ray dumpts it's temp buffer while tracing]
        self.guess_ray_dumps = 30

        buff_elements = self.NrayBuff * self.NcellBuff

        # create gpu2cache pipeline objects
        # Instead of calling the internal dtype dictionary, explicitly call the global_ray_dtype to ensure a match.  
        self.global_rayid_pipe = gpu2cpu_pipeline(buff_elements, self.global_ray_dtypes["global_rayid"], buff_elements*self.guess_ray_dumps)
        self.pathlength_pipe   = gpu2cpu_pipeline(buff_elements, self.global_ray_dtypes["pathlength"], buff_elements*self.guess_ray_dumps)
        self.index1D_pipe      = gpu2cpu_pipeline(buff_elements, self.global_ray_dtypes["index1D"], buff_elements*self.guess_ray_dumps)
        #self.lrefine          = gpu2cpu_pineline(buff_elements, cupy.int16)

        # Number of ray segments that have been pushed to the cpug2pu pipes
        self.toc_length = 0
        # Information of where ray segments have been pushed in the cpu2gpu pipes
        # Init to negative so that we can easily remove empty values
        self.aggregate_toc = cupy.full((self.Nraytot_est*self.guess_ray_dumps, 3), -1, dtype=dtype_dict["aggregate_toc"])

        # Index to keep track of how many ray-cell intersection we have sent to the pipe buffers
        self.start_dump = 0

    def i_dont_know(self):
        # Dataframe to save physical values to
        self.rays.reset_index(inplace = True)
        self.ray_buff = cudf.DataFrame({"xp" : self.rays.xp.repeat(self.NcellBuff), "yp" : self.rays.yp.repeat(self.NcellBuff)})
        self.ray_buff["ibuff"] = cupy.mod(cupy.arange(0,len(self.ray_buff)), self.NcellBuff)
        self.ray_buff["pathlength"] = cupy.zeros(len(self.ray_buff))
        for line in self.line_lables :
            self.ray_buff[line] = 0.0
        # for opac in opac_labels:
        #   self.ray_buff[opac] = 0.0
        self.rays.set_index(["xp", "yp"], inplace = True)
        self.ray_buff.set_index(["xp", "yp","ibuff"], inplace = True)


    def raytrace_onestep(self):
        # find the next intersection to a cell boundary by finding the closest distance to an integer for all directions
        self.active_rayDF = self.active_rayDF.apply_rows(__raytrace_kernel__,
                incols = ["xi", "yi", "zi","ray_status"],
                outcols = dict( pathlength = np.float64, index1D=np.int32),
                kwargs = dict(raydir = self.raydir, Nmax = self.Nmax, first_step = self.first_step))
        # self.findrefined()

        
        # store in buffer
        self.buff_index1D   [self.active_rayDF["active_rayDF_to_buffer_map"].values, self.active_rayDF["buffer_current_step"].values]    = self.active_rayDF["index1D"].values[:]
        self.buff_pathlength[self.active_rayDF["active_rayDF_to_buffer_map"].values, self.active_rayDF["buffer_current_step"].values]    = self.active_rayDF["pathlength"].values[:]
        self.active_rayDF["buffer_current_step"] += 1
        # Use a mask, and explicitly set mask dtype. This prevents creating a mask value with the default cudf/cupy dtypes, and saving them to arrays with different dtypes.
        # Currently this just throws warnings if they are different dtypes, but this behavior could be subject to change which may produce errors or worse...
        self.active_rayDF["ray_status"].mask(self.active_rayDF["buffer_current_step"] == self.active_ray_dtypes["buffer_current_step"](self.NcellBuff), self.active_ray_dtypes["ray_status"](1), inplace = True)

    def setBufferDF(self):
        """
            Gathers the data stored in the buff arrays. gets their subphysics d
        """
        self.ray_buff[self.line_lables] = cudf.DataFrame(self.avg_em_df.iloc[self.subphys_id_df.iloc[self.buff_index1D.ravel()].values].values, index = self.ray_buff.index)
        self.ray_buff["pathlength"] = self.buff_pathlength.ravel()

    def addToFlux(self):
        for line in self.line_lables:
            self.ray_buff[line] = self.ray_buff[line].mul(self.ray_buff["pathlength"])
        self.fluxes[self.line_lables] = self.fluxes[self.line_lables].add(self.ray_buff[self.line_lables].groupby([cudf.Grouper(level = 'xp'), cudf.Grouper(level = 'yp')]).sum(), fill_value = 0.0) # * self.opacity[line_labels].exp())

    def get_subphysics_cells(self):
        self.setBufferDF()
        self.addToFlux()


