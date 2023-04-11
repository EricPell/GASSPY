from turtle import pos
import cupy

"""
    This file contains a dictonary specifying variables and their dtypes associated with the raytrace
"""
# Default float used
default_float = cupy.float64


rayid_dtype = cupy.int32
cellid_dtype = cupy.int64
amr_lref_dtype = cupy.int16

pos_dtype = default_float


# Dtypes
ray_dtypes = {
    # Variables specific to the ray
    "xp" : pos_dtype, "yp" : pos_dtype,                       # x and y coordinate on the detector plane 
    "global_rayid" : rayid_dtype,                                   # unique id of the ray, common and fixed throughout the code 
    "trace_status": cupy.int8,                                      # flag for if a ray is (0)waiting/(1)active/(2)finished in its raytrace
    "pid"         : rayid_dtype,                                    # unique id of parent ray
    "pevid"       : rayid_dtype,                                    # id of the "parent split event" causing the creation of the ray
    "cevid"       : rayid_dtype,                                    # id of the "child split event" causing the termination of the ray
    "aid"        : rayid_dtype,                                    # id of the "ancestral" ray, used as the id of the branch 
    "ray_lrefine" : amr_lref_dtype,                                     # ray refinement level
    
    # Variables specific to the current location of the ray
    "xi" : pos_dtype, "yi" : pos_dtype, "zi" : pos_dtype,                # x,y and z coordinates in the simulation domain
    "raydir_x": pos_dtype, "raydir_y": pos_dtype, "raydir_z": pos_dtype, # x, y, z normal directions of the ray path
    "index1D"     : cellid_dtype,                                                   # 1D raveled index of the cell currently containing the ray
    "next_index1D"     : cellid_dtype,                                                   # 1D raveled index of the next cell containing the ray
    "amr_lrefine" : amr_lref_dtype,                                                   # local amr refinement level
    "pathlength"  : pos_dtype,                                   # the pathlength of the ray through the cell
    "ray_status": cupy.int8,                                        # flag for the ray being (0)fine/(1)filled its buffer/(2)terminated for leaving the domain
                                                                    # /(3)split due to refinement
    "active_rayDF_to_buffer_map" : rayid_dtype,                      # Mapping index from the active_rayDF to the 2D buffer arrays
    "buffer_current_step": cupy.int16,                              # Current cell in the buffer
    "dump_number" : cupy.int16,                                     # Number of dumps the current ray has done
    "ray_area"    : pos_dtype,                                    # current area covered by the solid angle of the ray
    "solid_angle" : pos_dtype,                                    # current area covered by the solid angle of the ray
    "ray_fractional_area"    : pos_dtype,                                    # current area covered by the solid angle of the ray
    # Variables specific to the buffer
    "buff_slot_occupied":cupy.int8,                                  # Flag if a spot in the buffer is occupied 
    
    "cell_index": cellid_dtype,                                       # Index of the cell in the raveled list supplied by the user


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
    "aid"        : -1,
    "ray_lrefine" : -1,             
    "refinement_level": 0,

    # Variables specific to the current location of the ray
    "xi" : -1, "yi" : -1, "zi" : -1,
    "raydir_x": -1, "raydir_y": -1, "raydir_z": -1,
    "index1D"     : -1,
    "next_index1D"     : -1,
    "amr_lrefine" : -1,
    "pathlength"  : 0,
    "ray_status": 0,
                                                           
    "active_rayDF_to_buffer_map" : -1,
    "buffer_current_step": 0,
    "dump_number" : -1,
    "ray_area"    : 0,
    "solid_angle"    : 0,
    "ray_fractional_area"    : 0,

    # Variables specific to the buffer
    "buff_slot_occupied": 0,

    # Default maximum and minimum refinement levels for the rays
    "ray_lrefine_min" : 1,
    "ray_lrefine_max" : 30,
    
    "cell_index" : -1
}
