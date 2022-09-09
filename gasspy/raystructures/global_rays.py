import cupy
import numpy
import sys
from .base_rays import base_ray_class
from gasspy.settings.defaults import ray_dtypes, ray_defaults
"""
This class contains arrays relevant to keeping track of all rays, active or not (eg global rays)
It requires all the lineage information, and the current postion and direction of the rays
"""


class global_ray_class(base_ray_class):

    
    class_name = "global_rays"

    def __init__(self, nalloc = None, on_cpu = False, contained_fields = None):
        self.not_allocated = True
        self.nrays = 0
        self.contained_fields = [
            "xp", "yp", 
            "xi", "yi", "zi",
            "raydir_x", "raydir_y", "raydir_z",
            "global_rayid",
            "trace_status",
            "pid",         
            "pevid",       
            "cevid",      
            "aid",
            "ray_lrefine",
            "amr_lrefine",
            "cell_index"
        ]
        self.on_cpu = on_cpu
        if self.on_cpu:
            self.numlib = numpy
        else:
            self.numlib = cupy

        if contained_fields is not None:
            if "global_rayid" not in contained_fields:
                contained_fields.append("global_rayid")
            self.contained_fields = contained_fields
        for field in self.contained_fields:
            self.field_dtypes[field] = ray_dtypes[field]
            self.field_defaults[field] = ray_defaults[field]
        if nalloc is not None:
            self.allocate_rays(nalloc)
        else:
            self.nalloc = 0
        return

    def append(self, nrays, fields = None, over_alloc_factor = 1.2):
        """
            Appends a set of rays to the global_ray structure
            arguments:
                    nrays: integer (number of rays to append)
                    fields: optional dictionary of arrays (The fields of the rays are to be appended)
                    over_alloc_factor: optional integer (In case of there note being enough slots allocated, over_alloc_factor*len(ray_dict) new rays will be allocated)
            returns:
                    global_rayid : array of integers (indexes of the new rays in the global_ray class )
        """

        # If the new number of rays exceed the allocated ones, allocate more
        if nrays + self.nrays > self.nalloc:
            self.allocate_rays(int(nrays*over_alloc_factor))

        # Determine the global_rayids
        new_global_rayid = self.numlib.arange(self.nrays, self.nrays + nrays, dtype=ray_dtypes["global_rayid"])
        self.global_rayid[new_global_rayid] = new_global_rayid
        if fields is not None:
            for field in fields.keys():
                if "global_rayid"  == field:
                    assert self.numlib.array_equal(fields["global_rayid"], new_global_rayid), "ERROR: Supplied global ray id does not match the ones determined internally"
                    continue
                self.__dict__[field][new_global_rayid] = fields[field]

        self.nrays += nrays
        return new_global_rayid
            

