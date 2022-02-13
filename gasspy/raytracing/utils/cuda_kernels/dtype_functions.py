import cupy
from gasspy.settings.defaults import ray_dtypes
"""
    This file contains the raw kernels used in the raytracing and various utility functions for that purpose
"""


def python_dtype_to_nvcc(value):
    dtype_strings = [ [cupy.int8,  "char"],
                      [cupy.int16, "short"],
                      [cupy.int32, "int"],
                      [cupy.int64, "long long"],
                      [cupy.float16, "half"],
                      [cupy.float32, "float"],
                      [cupy.float64, "double"],
                      [int, "int"]
                   ]
    # loop over all possible dtypes and return matching c type
    for dtype in dtype_strings:
        if isinstance(value, dtype[0]):
            return dtype[1]
    # if nothing matches.. return void and hope the user is ok...
    return "void"


def get_argument_string(input_vars):
    argument_string = ""
    for i, var in enumerate(input_vars):    
        name, key, arr = var[0], var[1], var[2]
        c_type = python_dtype_to_nvcc(ray_dtypes[key](1))
        if i > 0:
            argument_string += ", " 
        argument_string += c_type + arr + " "+ name
    return argument_string