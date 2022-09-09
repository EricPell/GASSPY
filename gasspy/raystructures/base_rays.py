import cupy
import numpy
from gasspy.settings.defaults import ray_dtypes, ray_defaults
import pickle
"""
    A parent class containing methods common to all or most rays
    Also used as a flexible container for random rays
"""

class base_ray_class:
    contained_fields = []
    class_name = "base_rays"
    field_dtypes = {}
    field_defaults = {}
    def __init__(self, nalloc, contained_fields, on_cpu = False):
        self.nalloc = nalloc
        self.nrays  = 0
        self.not_allocated = True
        self.contained_fields = contained_fields
        for field in self.contained_fields:
            self.field_dtypes[field] = ray_dtypes[field]
            self.field_defaults[field] = ray_defaults[field]

        self.on_cpu = on_cpu
        if self.on_cpu:
            self.numlib = numpy
        else:
            self.numlib = cupy


        return


    def append_field(self, field, default_value=None, dtype =None):
        """
            Appends a field to the current list of fields
            arguments:
                field: string (name of field)
                default_value: optional object
        """
        if field in self.contained_fields:
            return
        else:
            self.contained_fields.append(field)
            if dtype is None:
                dtype = self.field_dtypes[field]
            if default_value is None:
                default_value = self.field_defaults[field] 
            self.__dict__[field] = self.numlib.full(self.nalloc, default_value, dtype)

            self.field_defaults[field] = default_value
            self.field_dtypes[field] = dtype
    
    def set_field(self, field, value, index = None):
        """
            Sets the values of a field 
            arguments:
                    field: string (name of field to set)
                    value: any dtype (array of values to set, must be convertable to the same dtype as the field)
                    index: optional integer or array of integer (indexes where values are set, must be same size as value)
        """
        
        # Make sure field exists in the raystructure
        assert field in self.contained_fields, "Field %s does not exist in %s data structure" % (field, self.class_name)
        
        if index is None:
            # If no index is supplied, set the entire array
            self.__dict__[field][:self.nrays] = value
        else:
            # Otherwise just the specified parts of it
            self.__dict__[field][index] = value
        return
        
    def get_field(self, field, index = None):
        """
            Gets the values of a field 
            arguments:
                    field: string (name of field to get)
                    index: optional integer or array of integer (indexes of values to get)
        """
        # Make sure field exists in the raystructure
        assert field in self.contained_fields, "Field %s does not exist in %s data structure" % (field, self.class_name)
       
        if index is None:
             # If no index is supplied, get the entire occupied array
            return self.__dict__[field][:self.nrays]
        else:
            # Otherwise just the specified parts of it
            return self.__dict__[field][index]    


    def get_subset(self, index):
        """
            Returns a dictionary containing all fields for a subset of the active rays
            arguments:
                    index: integer or array of integer (indexes of rays to grab)
                    full : optional boolean (switch if index refers to the index in the full arrays or in the active_index array)
            returns:
                    ray_subset: dictionary of arrays (all fields for the specified active rays)
        """
        # initialize dictionary
        ray_subset = {}

        # Add fields
        for field in self.contained_fields:
            ray_subset[field] = self.__dict__[field][index]
        return ray_subset


    def allocate_rays(self, nalloc):
        """
            Allocates arrays for all the associated fields if they have yet to be so, otherwise appends nalloc slots to them
            arguments:
                    nalloc : integer (number of slots to allocate)
        """
        if self.not_allocated:
            for field in self.contained_fields:
                self.__dict__[field] = self.numlib.full(nalloc, self.field_defaults[field], dtype=self.field_dtypes[field])
            self.nalloc = nalloc
            self.not_allocated = False
        else:
            for field in self.contained_fields:
                self.__dict__[field] = self.numlib.append(self.__dict__[field], self.numlib.full(nalloc, self.field_defaults[field], dtype=self.field_dtypes[field]))
            self.nalloc += nalloc

        return

    def append(self, nrays, fields = None, over_alloc_factor = 1.5):
        """
            Appends a set of rays to the global_ray structure
            arguments:
                    nrays: integer (number of rays to append)
                    fields: optional dictionary of arrays (The fields of the rays are to be appended)
                    over_alloc_factor: optional integer (In case of there note being enough slots allocated, over_alloc_factor*len(ray_dict) new rays will be allocated)
            returns:
                    new_indexes : array of integers (indexes of the newly )
        """

        if nrays + self.nrays > self.nalloc:
            self.allocate_rays(int(nrays*over_alloc_factor))
        
        if fields is not None:
            indexes = self.numlib.arange(self.nrays, self.nrays + nrays)
            for field in fields.keys():
                self.set_field(field, fields[field], index = indexes)
        self.nrays += nrays


    def save(self, filename="rays.pickle"):
        ''' 
            saves the ray structure as a pickle
        '''
        f = open(filename, 'wb')
        pickle.dump(self.__dict__,f)
        f.close()
    
    def load(self, filename = "rays.pickle"):
        """
            Loads a pickled ray structure object
            arguments:
                filename: string (Path to file to load)
        """
        with open(filename, 'rb') as f:
            tmp = pickle.load(f)
        self.__dict__.update(tmp)


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
            if self.on_cpu:
                grp.create_dataset(field, self.nrays, dtype = self.field_dtypes[field], data = self.__dict__[field][:self.nrays])
            else:
                grp.create_dataset(field, (self.nrays,), dtype = self.field_dtypes[field], data = self.__dict__[field][:self.nrays].get())

    def load_hdf5(self, h5file):
        """
            loads the ray structure object from its group within an hdf5 file. All non specified fields are set to default values
            arguments:
                h5file: hdf5 file (An open hdf5 file in which to create the group)
        """
        assert self.class_name in h5file.keys(), "%s not in hdf5 file" % self.class_name
        grp = h5file[self.class_name]
        self.nrays = len(grp[list(grp.keys())[0]][:])
        for field in self.contained_fields:
            if field not in grp.keys():
                self.__dict__[field] = self.numlib.zeros(self.nrays, dtype = self.field_dtypes[field])
            else:
                self.__dict__[field] = self.numlib.array(grp[field][:])

        self.nalloc = self.nrays


    def print(self, idx = None):
        for field in self.contained_fields:
            print(field + ": ")
            if idx is None:
                print("\t\t", self.__dict__[field])
            else:
                print("\t\t", self.__dict__[field][idx])