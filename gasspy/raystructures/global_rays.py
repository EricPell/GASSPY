import cupy
import numpy

from .base_rays import base_ray_class
from gasspy.settings.defaults import ray_dtypes, ray_defaults
"""
This class contains arrays relevant to keeping track of all rays, active or not (eg global rays)
It requires all the lineage information, and the current postion and direction of the rays
"""


class global_ray_class(base_ray_class):
    contained_fields = [
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
            "amr_lrefine"
        ]
    class_name = "global_rays"

    def __init__(self, nalloc = None, on_gpu = True, contained_fields = None):
        self.not_allocated = True
        self.nrays = 0

        self.on_gpu = on_gpu
        if self.on_gpu:
            self.numlib = cupy
        else:
            self.numlib = numpy
        
        if contained_fields is not None:
            if "global_rayid" not in contained_fields:
                contained_fields.append("global_rayid")
            self.contained_fields = contained_fields
        
        if nalloc is not None:
            self.allocate_rays(nalloc)
        else:
            self.nalloc = 0
        return

    def append(self, nrays, fields = None, over_alloc_factor = 4):
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
            self.allocate_rays(nrays*over_alloc_factor)
        
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
            

