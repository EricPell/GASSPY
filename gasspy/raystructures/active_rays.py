#from re import I
import cupy
from numba import cuda
from .base_rays import base_ray_class
from gasspy.settings.defaults import ray_dtypes, ray_defaults
"""
This class contains arrays relevant to active rays
eg rays that are currently being raytraced.
It does not require all the lineage information, but does need information in regards to what buffer slots etc is associated with each list

NOTE: This class always works on the set of active rays, which may or may not be as many as the number of allocated slots
      The currently active indexes are specified in the boolian array "index_active"
"""


class active_ray_class(base_ray_class):
    contained_fields = [
            "global_rayid",
            "xp", "yp",
            "xi", "yi", "zi",
            "raydir_x", "raydir_y", "raydir_z",
            "amr_lrefine",
            "ray_lrefine",
            "index1D",
            "next_index1D",
            "pathlength",
            "ray_status",
            "active_rayDF_to_buffer_map",
            "buffer_current_step",
            "dump_number", 
            "ray_area"]

    class_name = "active_rays"

    def __init__(self, nrays = 0):
        # set the number of rays in this instance
        self.nrays = nrays
        self.nactive = 0
        self.__init_arrays__()

        return

    def __init_arrays__(self):
        # Initialize all arrays to their default value
        for field in self.contained_fields:
            self.__dict__[field] = cupy.full(self.nrays, ray_defaults[field], dtype=ray_dtypes[field])

        self.__init_active__()
        return

    def __init_active__(self):
        # Initialize boolean array corresponding to if an index is active or not, along with the array of active indexes
        self.index_active = cupy.full(self.nrays, False, dtype = cupy.bool8)
        self.active_indexes = cupy.array([])

    def __set_active__(self, index):
        """
            Sets the specified indexes to be active
            arguments:
                    index: integer or array of intefers (indexes to be set to active)
        """
        self.index_active[index] = True
        self.active_indexes = cupy.where(self.index_active)[0]
        self.nactive = len(self.active_indexes)
        return

    def __set_inactive__(self, index):
        """
            Sets the specified indexes to be inactive
            arguments:
                    index: integer or array of intefers (indexes to be set to inactive)
        """
        self.index_active[index] = False
        self.active_indexes = cupy.where(self.index_active)[0]
        self.nactive = len(self.active_indexes)
        return


    def set_field(self, field, value, index = None, full = False):
        """
            Sets the values of a field 
            arguments:
                    field: string (name of field to set)
                    value: any dtype (array of values to set, must be convertable to the same dtype as the field)
                    index: optional integer or array of integer (indexes where values are set, must be same size as value)
                    full : optional bool8 (flag to set all ray slots, not only the active ones)
        """
        
        # Make sure field exists in the raystructure
        assert field in self.contained_fields, "Field %s does not exist in %s data structure" % (field, self.class_name)
        
        if index is None:
            # If no index is supplied, set all the active indexes or all indexes
            if full:
                self.__dict__[field][:] = value
            else:
                self.__dict__[field][:self.nactive] = value
        else:
            # Otherwise just the specified parts of it
            self.__dict__[field][index] = value

        return
        
    def get_field(self, field, index = None, full = False):
        """
            Gets the values of a field 
            arguments:
                    field: string (name of field to get)
                    index: optional integer or array of integer (indexes of values to get)
                    full : optional bool8 (flag to get all ray slots, not only the active ones)
        """
        # Make sure field exists in the raystructure
        assert field in self.contained_fields, "Field %s does not exist in %s data structure" % (field, self.class_name)
       
        if index is None:
             # If no index is supplied, get all active indexes
            return self.__dict__[field][:self.nactive]
        else:
            # Otherwise just the specified parts of it
            return self.__dict__[field][index]   

    def update_nrays(self, nrays):
        """
            Updates the total number of ray slots (eg number of rays that can be active at a time)
            arguments:
                    nrays: integer (new total number of ray slots)
        """
        # If there are no currently active rays, we can just reinitialize
        if len(self.active_indexes) == 0:
            # Set the new number of rays
            self.nrays = nrays
            # (re)initialize the arrays
            self.__init_arrays__()
        else:
            # If there are active rays, we need to find these in the list 
            assert len(self.active_indexes) <= nrays, "New number of active ray slots must be LARGER than the current number of active rays"
            self.nrays = nrays
            for field in self.contained_fields:
                copy = self.__dict__[field][self.active_indexes]
                self.__dict__[field] = cupy.full(self.nrays, ray_defaults[field], dtype=ray_dtypes[field])
                self.__dict__[field][:len(self.active_indexes)] = copy[:]
                del(copy) 
            self.__init_active__()
            self.__set_active__(cupy.arange(len(self.active_indexes)))              
        return



    def remove_rays(self, index, full = False):
        """
            Removes the specified rays, and restructures the list of rays
            arguments:
                    index: integer or array of integers (Indexes of active_indexes corresponding rays to be pruned)
                    full : optional bool8 (flag to specify that the indexes passed are of the full (active and inactive) slots and not of the active_indexes arrays)       
        """
        # first figure out which indexes are to be kept
        to_keep = cupy.where(cupy.in1d(cupy.arange(self.nactive), index, invert=True))[0]
#            to_keep = self.active_indexes[cupy.where(cupy.in1d(cupy.arange(len(self.active_indexes)), index, invert=True))[0]]

        Nremaining = len(to_keep)

        # Loop over all fields, copy the rays we want to keep and put them on the top
        for field in self.contained_fields:
            self.__dict__[field][:Nremaining] = self.__dict__[field][to_keep]

        # set all inactive and active the new slots
        self.index_active[:] = False
        self.__set_active__(cupy.arange(Nremaining))
            
        return


    def activate_rays(self, nrays_to_activate, fields = None):
        """
            Sets a number of rays to be active, fills their arrays with the selected values, and returns the full indexes for usage later
            arguments:
                    nrays_to_activate: integer (number of rays to activate)
                    fields: optional dictionary of (field name, array) pairs to fill out fields here
            returns:
                    indexes_to_activate: array of integers (indexes in the full arrays that has been activated. NOTE: These should be used with the full flag when using get/set_field functions)
        """
        #assert (self.nactive + nrays_to_activate <= self.nrays), "The requested number of rays %d exceeds the total number of ray slots %d" (self.nactive + nrays_to_activate, self.nrays) 
              
        # find inactive slots
        indexes_to_activate = cupy.arange(self.nactive, self.nactive + nrays_to_activate)
        self.__set_active__(indexes_to_activate)
        
        if fields is not None:
            for field, value in fields.items():
                self.set_field(field, value, indexes_to_activate)

        return indexes_to_activate


    def field_add(self, field, value, index = None, full = False):
        """
        Adds a value to a one field
        arguments:
                field: string (Name of field to add to)
                value:        (Value to add to the field)
                index: optional integer or array of integers (indexes where to add value)
                full : optional boolean ((switch if index refers to the index in the full arrays or in the active_index array)    
        """
        # Make sure field exists in the raystructure
        assert field in self.contained_fields, "Field %s does not exist in %s data structure" % (field, self.class_name)

        if index is None:
            self.__dict__[field][:self.nactive] += value
        else:
            self.__dict__[field][index] += value

    def field_mul(self, field, value, index = None, full = False):
        """
        Multiplies one field by a value
        arguments:
                field: string (Name of field to multply)
                value:        (Value to multiply with the field)
                index: optional integer or array of integers (indexes where to multiply value)
                full : optional boolean (switch if index refers to the index in the full arrays or in the active_index array)    
        """
        # Make sure field exists in the raystructure
        assert field in self.contained_fields, "Field %s does not exist in %s data structure" % (field, self.class_name)

        if index is None:
            self.__dict__[field][:self.nactive] *= value
        else:
            self.__dict__[field][index] *= value


    def print(self, idx = None):
        for field in self.contained_fields:
            print(field + ": ")
            if idx is None:
                print("\t\t", self.__dict__[field])
            else:
                print("\t\t", self.__dict__[field][idx])

    def move_to_numba(self, fields):
        for field in fields:
            assert field in self.contained_fields, "Field %s does not exist in %s data structure" % (field, self.class_name)
            self.__dict__[field] = cuda.as_cuda_array(self.__dict__[field])


    def move_to_cupy(self, fields):
        for field in fields:
            assert field in self.contained_fields, "Field %s does not exist in %s data structure" % (field, self.class_name)
            self.__dict__[field] = cupy.asarray(self.__dict__[field])


    def debug_ray(self, idx, fields):
        string = ""

        for field in fields:
            string += field +": "+str(self.__dict__[field][idx]) + ", " 
        print(string)

    

    def save_hdf5(self, h5file, fields = None):
        """
            Saves the ray structure object as a group within an hdf5 file
            arguments:
                h5file: hdf5 file (An open hdf5 file in which to create the group)
                fields: optional list of strings (subset of fields to save)
        """
        grp = h5file.create_group(self.class_name)
        if fields is None:
            fields = self.contained_fields

        for field in fields:
            assert field in self.contained_fields, "Field %s does not exist in %s data structure" % (field, self.class_name)
            grp.create_dataset(field, self.nrays, dtype = ray_dtypes[field], data = self.__dict__[field].get())

    def load_hdf5(self, h5file):
        """
            loads the ray structure object from its group within an hdf5 file. All non specified fields are set to default values
            arguments:
                h5file: hdf5 file (An open hdf5 file in which to create the group)
        """
        grp = h5file[self.class_name]
        self.nrays = len(grp[list(grp.keys())[0]][:])
        for field in self.contained_fields:
            if field not in grp.keys():
                self.__dict__[field] = cupy.zeros(self.nrays, dtype = ray_dtypes[field])
            else:
                self.__dict__[field] = cupy.array(grp[field][:])

        self.nactive = self.nrays
        self.__set_active__(cupy.arange(self.nactive))