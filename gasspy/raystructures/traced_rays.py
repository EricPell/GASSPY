import numpy as np
import cupyx 
import cupy
import pickle
from gasspy.settings.defaults import ray_dtypes, ray_defaults

debug_ray = 856890

class traced_ray_class(object):
    def __init__(self, NraySegMax, NcellPerRaySeg, traced_vars, pinned=True):
        self.NraySegMax = NraySegMax
        self.NcellPerRaySeg = NcellPerRaySeg
        self.pinned = pinned
        self.traced_vars = traced_vars
        self.alloc()
        pass


    def alloc(self):
        # Allocate the arrays to buffer the incoming data on system memory
        self.alloc_cpu_arrays()

        # Allocate array for rayid and dump number
        self.global_rayid_ofSegment = np.full(self.NraySegMax, -1, dtype = ray_dtypes["global_rayid"])
        self.dump_number_ofSegment = np.full(self.NraySegMax, -1 , dtype = ray_dtypes["dump_number"])
        
        # counter for the number of ray segments that have been used up
        self.NraySegUsed = 0

        # Arrays to store the split events of the raytrace
        self.splitEvents = None
        pass

    def alloc_cpu_arrays(self):
        # For all of the variable that are traced (Default index1D, pathlength, amr_level, vlos)
        # allocate arrays for the maximum possible number of ray segments and the cells for these ray segments
        if self.pinned:
            for var in self.traced_vars:
                self.__dict__[var + "_cpu_array"] = cupyx.zeros_pinned((self.NraySegMax, self.NcellPerRaySeg), dtype  = ray_dtypes[var])
                self.__dict__[var + "_cpu_array"][:,:] = ray_defaults[var]
        else:
            for var in self.traced_vars:
                self.__dict__[var + "_cpu_array"] = np.zeros((self.NraySegMax, self.NcellPerRaySeg), dtype  = ray_dtypes[var])
                self.__dict__[var + "_cpu_array"][:,:] = ray_defaults[var]
 
    def data_from_cupy(self, varname, data, isegment, NraySeg_transfered, stream = None):
        """ 
            Method to push data to the (pinned cpu) array corresponding to varname

            varname            : (str) name of the variable 
            data               : (cupy.ndarray) 2D array corresponding to ray segments
            isegment           : (int) index of where in the CPU array to put the data
            NraySeg_transfered : (int) number of raysegments transfered
            stream             : (cupy.cuda.stream.Stream) stream to use in the transfer
        """
        assert isegment + NraySeg_transfered < self.NraySegMax, "More ray segments pushed to the CPU than allocated... In future this should result in allocating more memory"
        
        self.__dict__[varname + "_cpu_array"][isegment : isegment + NraySeg_transfered] = data.get(stream = stream)
        pass

    def append_indexes(self, global_rayid, dump_number, NraySeg_transfered):
        """
            Method to set the global_rayid and dump number of the next (specified) number of ray segments
        """
        if(self.NraySegUsed + NraySeg_transfered > self.NraySegMax):
            print(self.NraySegMax, self.NraySegUsed + NraySeg_transfered, NraySeg_transfered) 
      
        assert self.NraySegUsed + NraySeg_transfered <= self.NraySegMax, "No more ray segments are available"
      
        # These arrays are equal in length to the number of segments.
        self.global_rayid_ofSegment[self.NraySegUsed: self.NraySegUsed + NraySeg_transfered] = global_rayid.get()
        self.dump_number_ofSegment [self.NraySegUsed: self.NraySegUsed + NraySeg_transfered] = dump_number.get()
        self.NraySegUsed += NraySeg_transfered
        pass

    def create_mapping_dict(self, Nrays):
        """
            Method to take the current (filled) ray slots and associate their rayid & dump number to the index in a dictionary
        """
        # intitialize the mapping dictionary
        self.RaySegment_mapping_dict = {}
        # Create an entry for every ray
        for iray in range(Nrays):
            self.RaySegment_mapping_dict[iray] = {}
        
        # Loop over all filled ray slots, and create an entry in the corresponding rayid
        for iseg in range(self.NraySegUsed):
            self.RaySegment_mapping_dict[int(self.global_rayid_ofSegment[iseg])][int(self.dump_number_ofSegment[iseg])] = iseg

        pass 

    def finalize_trace(self, delete_pinned = True):
        idx_sort = cupy.lexsort(cupy.array([self.dump_number_ofSegment[:self.NraySegUsed], self.global_rayid_ofSegment[:self.NraySegUsed]]))
        self.global_rayid_ofSegment = self.global_rayid_ofSegment[idx_sort.get()]   
        self.dump_number_ofSegment  = self.dump_number_ofSegment[idx_sort.get()]   

        for var in self.traced_vars:
            self.__dict__[var] = self.__dict__[var+"_cpu_array"][idx_sort.get(),:]

            if delete_pinned:
                # release pinned memory
                del self.__dict__[var+"_cpu_array"]       


    def move_to_pinned_memory(self):
        """
            Copies over all arrays (traced vars & header arrays) to  pinned data spaces
            NOTE: This does not check if the arrays are already on the pinned memory
        """
        
        for arr in ["global_rayid_ofSegment", "dump_number_ofSegment"] + self.traced_vars:
            tmp = self.__dict__[arr][:]
            self.__dict__[arr] = cupyx.zeros_pinned(tmp.shape)
            self.__dict__[arr][:] = tmp[:]
            del tmp


    def move_from_pinned_memory(self):
        """
            Copies over all arrays (traced vars & header arrays) to unpinned numpy arrays
            NOTE: This does not check if the arrays are already unpinned
        """
        
        for arr in ["global_rayid_ofSegment", "dump_number_ofSegment"] + self.traced_vars:
            tmp = self.__dict__[arr][:]
            self.__dict__[arr] = np.zeros(tmp.shape)
            self.__dict__[arr][:] = tmp[:]
            del tmp

    def add_to_splitEvents(self, split_events):
        """
            Appends a list of split_events to the list of the raytrace
        """
        # TODO Make this a preallocated array with large appends if overflow
        # For now: If not defined, set the list as the current list
        #           elsewise append to it
        if self.splitEvents is None:
            self.splitEvents = split_events
        else:
            self.splitEvents = cupy.append(self.splitEvents, split_events, axis = 0)
        return


    def save(self, file="traced_rays.ray"):
        f = open(file, 'wb')
        pickle.dump(self.__dict__,f)
        f.close()

    def load(self, file="traced_rays.ray"):
        with open(file, 'rb') as f:
            tmp = pickle.load(f)
        self.__dict__.update(tmp)
        if self.pinned:
            self.move_to_pinned_memory()


    def save_hdf5(self, h5file):
        """
            Saves the ray structure object as a group within an hdf5 file
            arguments:
                h5file: hdf5 file (An open hdf5 file in which to create the group)
                fields: optional list of strings (subset of fields to save)
        """
        
        # Save all the ray segment fields as their own group
        grp = h5file.create_group("ray_segments")
        grp.create_dataset("global_rayid", self.NraySegUsed, dtype = ray_dtypes["global_rayid"], data = self.global_rayid_ofSegment)
        grp.create_dataset("dump_number", self.NraySegUsed, dtype = ray_dtypes["dump_number"], data = self.dump_number_ofSegment)

        for field in self.traced_vars:
            assert field in self.__dict__.keys(), "Field %s has not been created in traced_rays. Has the finalize trace method been called?" % (field)
            grp.create_dataset(field, (self.NraySegUsed, self.NcellPerRaySeg), dtype = ray_dtypes[field], data = self.__dict__[field])

        
        # Save the split events and the linage information as its own dataset
        h5file.create_dataset("splitEvents", data = self.splitEvents.get())

    def reset(self):
        # Delete all the reduced arrays and reallocate the large _cpu arrays if needed
        cpu_arrays_exists = True
        for field in self.traced_vars:
            del self.__dict__[field]
            cpu_arrays_exists = cpu_arrays_exists and (field+"_cpu" in self.__dict__.keys())

        # Reset splitEvent history
        self.splitEvents = None

        # Reset dump number and global id 
        self.global_rayid_ofSegment = np.full(self.NraySegMax, -1, dtype = ray_dtypes["global_rayid"])
        self.dump_number_ofSegment  = np.full(self.NraySegMax, -1 , dtype = ray_dtypes["dump_number"])

        # Reset all counters
        self.NraySegUsed = 0

        if cpu_arrays_exists:
            return

        # If any of the cpu arrays are missing, reallocate them
        self.alloc_cpu_arrays()
        
    def clean(self):
        # Delete all the reduced arrays and delete the large _cpu arrays 
        cpu_arrays_exists = True
        for field in self.traced_vars:
            if field in self.__dict__:
                del self.__dict__[field]
            if field + "_cpu_array" in self.__dict__:
                del self.__dict__[field+ "_cpu_array"]
        del self.global_rayid_ofSegment
        del self.dump_number_ofSegment
        # Reset splitEvent history
        self.splitEvents = None