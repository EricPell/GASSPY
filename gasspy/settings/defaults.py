import cupy

"""
    This file contains a dictonary specifying variables and their dtypes associated with the raytrace
"""

# Dtypes
ray_dtypes = {
    # Variables specific to the ray
    "xp" : cupy.float64, "yp" : cupy.float64,                       # x and y coordinate on the detector plane 
    "global_rayid" : cupy.int32,                                    # unique id of the ray, common and fixed throughout the code 
    "trace_status": cupy.int8,                                      # flag for if a ray is (0)waiting/(1)active/(2)finished in its raytrace
    "pid"         : cupy.int32,                                     # unique id of parent ray
    "pevid"       : cupy.int32,                                     # id of the "parent split event" causing the creation of the ray
    "cevid"       : cupy.int32,                                     # id of the "child split event" causing the termination of the ray
    "ray_lrefine" : cupy.int16,                                     # ray refinement level
    "refinement_level" : cupy.int8,                                 # TODO: DEPRICATED - REMOVE
    
    # Variables specific to the current location of the ray
    "xi" : cupy.float64, "yi" : cupy.float64, "zi" : cupy.float64,  # x,y and z coordinates in the simulation domain
    "index1D"     : cupy.int64,                                     # 1D raveled index of the cell currently containing the ray
    "amr_lrefine" : cupy.int16,                                  # local amr refinement level
    "pathlength"  : cupy.float64,                                   # the pathlength of the ray through the cell
    "ray_status": cupy.int8,                                        # flag for the ray being (0)fine/(1)filled its buffer/(2)terminated for leaving the domain
                                                                    # /(3)split due to refinement
    "active_rayDF_to_buffer_map" : cupy.int32,                      # Mapping index from the active_rayDF to the 2D buffer arrays
    "buffer_current_step": cupy.int16,                              # Current cell in the buffer
    "dump_number" : cupy.int16,                                     # Number of dumps the current ray has done
    "ray_area"    : cupy.float64,                                    # current area covered by the solid angle of the ray
    # Variables specific to the buffer
    "buff_slot_occupied":cupy.int8                                  # Flag if a spot in the buffer is occupied 
}


## Default initial values 
# TODO fix these in the code
ray_defaults = {
    # Variables specific to the ray
    "xp" : -1, "yp" : -1, 
    "global_rayid" : -1,  
    "trace_status": 0,    
    "pid"         : -1,   
    "pevid"       : -1,   
    "cevid"       : -1,   
    "ray_lrefine" : -1,             
    "refinement_level": 0,

    # Variables specific to the current location of the ray
    "xi" : -1, "yi" : -1, "zi" : -1,
    "index1D"     : -1,
    "amr_lrefine" : -1,
    "pathlength"  : -1,
    "ray_status": 0,
                                                           
    "active_rayDF_to_buffer_map" : -1,
    "buffer_current_step": 0,
    "dump_number" : 0,
    "ray_area"    : 0,

    # Variables specific to the buffer
    "buff_slot_occupied": 0

}
