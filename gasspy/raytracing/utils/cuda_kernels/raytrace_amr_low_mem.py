import cupy 
import numpy as np
from gasspy.settings.defaults import ray_dtypes, ray_defaults
from gasspy.raytracing.utils.cuda_kernels.dtype_functions import get_argument_string, python_dtype_to_nvcc


"""
    Define the string containing c code for the raytrace stepping
"""

# input vars is a list over the variables that are to be used as arguments
# each entry contains another list of three strings 
# 1) name - name of the argument in the c code
# 2) ray_key - the key corresponding to the correct dtype in ray_dtypes
# 3) pointer - either "" or "*" depending on if we are passing a pointer (eg array) or scalar value
input_vars = [["xi", "xi", "*"],
              ["yi", "yi", "*"], 
              ["zi", "zi", "*"],
              ["raydir_x", "raydir_x", "*"], 
              ["raydir_y", "raydir_y", "*"], 
              ["raydir_z", "raydir_z", "*"],
              ["ray_status", "ray_status", "*"],  
              ["index1D", "index1D", "*"],
              ["next_index1D", "next_index1D", "*"],  
              ["amr_lrefine", "amr_lrefine", "*"],
              ["pathlength", "pathlength", "*"],
              ["Nmax_lref", "index1D", "*"],  
              ["dx_lref", "xi", "*"],
              ["amr_lrefine_min", "amr_lrefine", ""],
              ["nrays", "global_rayid", "" ]]


# Loop over all of arguments and generate an appropriate argument string
argument_string = get_argument_string(input_vars)

# Names of types that need to be consistent between c and python
dx_type =  python_dtype_to_nvcc(ray_dtypes["xi"](0))
index1D_type = python_dtype_to_nvcc(ray_dtypes["index1D"](0))
amr_lrefine_type = python_dtype_to_nvcc(ray_dtypes["amr_lrefine"](0))
rayid_dtype = python_dtype_to_nvcc(ray_dtypes["global_rayid"](0))
# We want this string to be formatable using kword argments, as in string.format(key = val)
# However, this means that any {} will be interpated as a format entry. To avoid this being an issue each
# { and } needs to be double
raytrace_low_mem_code_string = r'''
extern "C" __global__
void __raytrace_kernel__('''+argument_string+'''){{
    int mindir;
    '''+dx_type+''' dx, dy, dz, ix, iy, iz, newpath;
    '''+amr_lrefine_type+''' iamr;
    '''+index1D_type+''' Nmax;     

    unsigned '''+rayid_dtype+''' tid = blockDim.x * blockIdx.x + threadIdx.x;

    '''+dx_type+''' sim_size_half_x = {sim_size_half_x};
    '''+dx_type+''' sim_size_half_y = {sim_size_half_y};
    '''+dx_type+''' sim_size_half_z = {sim_size_half_z};
    if(tid < nrays){{
        if(ray_status[tid] > 0) {{
            return;
        }}
        if( (fabs(xi[tid] - sim_size_half_x) <= sim_size_half_x) && (fabs(yi[tid] - sim_size_half_y) <= sim_size_half_y) && (fabs(zi[tid] - sim_size_half_z) <= sim_size_half_z)) {{
            index1D[tid] = next_index1D[tid];
        }} else {{
            index1D[tid] = -1;
            pathlength[tid] = -1;
                
            // Ray status 2 is domain exit
            ray_status[tid] = 2;
            return;
        }}

        // Get the grid data relevant to the current amr level
        iamr = amr_lrefine[tid] - amr_lrefine_min;
        Nmax = Nmax_lref[iamr];

        // Figure out cell size index on refinement level
        dx = dx_lref[iamr*3 + 0];
        dy = dx_lref[iamr*3 + 1];
        dz = dx_lref[iamr*3 + 2];
        
        ix = xi[tid]/dx;
        iy = yi[tid]/dy;
        iz = zi[tid]/dz;

        // init to unreasonably high number
        pathlength[tid] = 1e30;
        mindir = -1;
        // check for closest distance to cell boundary by looking for the closest int in each cardinal axis away from the current position
        // a thousand of a cell width is added as padding such that the math is (almost) always correct
        // NOTE: this could be wrong if a ray is very close to an interface. So depending on the angle of raydir
        // With respect to the cells, errors can occur

        // in x
        if(raydir_x[tid] > 0){{
            newpath = (floor(ix) + 1 - ix)*dx/raydir_x[tid];
            if(pathlength[tid] > newpath){{
                pathlength[tid] = newpath;
                mindir = 0;
            }}
        }} else if(raydir_x[tid] < 0) {{
            newpath = (ceil(ix) - 1 - ix)*dx/raydir_x[tid];
            if(pathlength[tid] > newpath) {{
                pathlength[tid] = newpath;
                mindir = 0;
            }}
        }}
        // in y
        if(raydir_y[tid] > 0) {{
            newpath = (floor(iy) + 1 - iy)*dy/raydir_y[tid];
            if(pathlength[tid] > newpath){{
                pathlength[tid] = newpath;
                mindir = 1;
            }}
        }} else if(raydir_y[tid] < 0) {{
            newpath = (ceil(iy) - 1 - iy)*dy/raydir_y[tid];
            if(pathlength[tid] > newpath){{
                pathlength[tid] = newpath;
                mindir = 1;
            }}
        }}

        // in z
        if(raydir_z[tid] > 0){{
            newpath = (floor(iz) + 1 - iz)*dz/raydir_z[tid];
            if(pathlength[tid] > newpath){{
                pathlength[tid] = newpath;
                mindir = 2;
            }}
        }} else if(raydir_z[tid] < 0) {{
            newpath = (ceil(iz) - 1 - iz)*dz/raydir_z[tid];
            if(pathlength[tid] > newpath) {{
                pathlength[tid] = newpath;
                mindir = 2;
            }}
        }}
        if(mindir == 0){{
            // move to next int
            if(raydir_x[tid] > 0){{
                xi[tid] = (floor(ix) + 1)*dx;
            }} else {{
                xi[tid] = (ceil(ix) - 1)*dx;
            }}
            yi[tid] = yi[tid] + pathlength[tid]*raydir_y[tid];
            zi[tid] = zi[tid] + pathlength[tid]*raydir_z[tid];
        }}
        if(mindir == 1){{
            // move to next int
            if(raydir_y[tid] > 0){{
                yi[tid] = (floor(iy) + 1)*dy;
            }} else {{
                yi[tid] = (ceil(iy) - 1)*dy;
            }}
            xi[tid] = xi[tid] + pathlength[tid]*raydir_x[tid];
            zi[tid] = zi[tid] + pathlength[tid]*raydir_z[tid];
        }}   

        if(mindir == 2){{
            // move to next int
            if(raydir_z[tid] > 0){{
                zi[tid] = (floor(iz) + 1)*dz;
            }} else {{
                zi[tid] = (ceil(iz) - 1)*dz;
            }}
            xi[tid] = xi[tid] + pathlength[tid]*raydir_x[tid];
            yi[tid] = yi[tid] + pathlength[tid]*raydir_y[tid];
        }}
        //next_index1D[tid] = ('''+index1D_type+''')iz + Nmax*('''+index1D_type+''')iy + Nmax*Nmax*('''+index1D_type+''')ix;
    }}
}}
'''