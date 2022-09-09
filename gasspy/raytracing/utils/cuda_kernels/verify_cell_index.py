from gasspy.settings.defaults import ray_dtypes
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
              ["cell_center_x", "xi", "*"],
              ["cell_center_y", "yi", "*"], 
              ["cell_center_z", "zi", "*"],
              ["outdir", "amr_lrefine", "*"],
              ["amr_lrefine", "amr_lrefine", "*"],
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
verify_cell_index_code_string = r'''
extern "C" __global__
void __verify_cell_kernel__('''+argument_string+'''){{
    '''+dx_type+''' dx, dy, dz, ix, iy, iz, diff, diff_x, diff_y, diff_z; 
    '''+amr_lrefine_type+''' iamr, sign_x, sign_y, sign_z,mindir;

    unsigned '''+rayid_dtype+''' tid = blockDim.x * blockIdx.x + threadIdx.x;

    '''+dx_type+''' sim_size_half_x = {sim_size_half_x};
    '''+dx_type+''' sim_size_half_y = {sim_size_half_y};
    '''+dx_type+''' sim_size_half_z = {sim_size_half_z};
    if(tid < nrays){{
        outdir[tid] = -1;

        if( !((fabs(xi[tid] - sim_size_half_x) <= sim_size_half_x) && (fabs(yi[tid] - sim_size_half_y) <= sim_size_half_y) && (fabs(zi[tid] - sim_size_half_z) <= sim_size_half_z))) {{
            return;
        }}

        // Get the grid data relevant to the current amr level
        iamr = amr_lrefine[tid] - amr_lrefine_min;

        // Figure out cell size index on refinement level
        dx = dx_lref[iamr*3 + 0];
        dy = dx_lref[iamr*3 + 1];
        dz = dx_lref[iamr*3 + 2];
        
        ix = xi[tid];
        iy = yi[tid];
        iz = zi[tid];

        // init to zero
        diff = 0;
        mindir = -1;
        // in x
        diff_x = (ix - cell_center_x[tid]);
        if(abs(diff_x) >= (0.5*(1+1e-12)*dx)){{
            mindir = 0;
            diff = abs(diff_x);

        }}
        // in y
        diff_y = (iy - cell_center_y[tid]);
        if(abs(diff_y) >= (0.5*(1+1e-12)*dy)){{
            if(abs(diff_y) > diff){{
                mindir = 8;
                diff = abs(diff_y);
            }}

        }}
        // in z
        diff_z = (iz - cell_center_z[tid]);
        if(abs(diff_z) >= (0.5*(1+1e-12)*dz)){{
            if(abs(diff_z) > diff) {{
                mindir = 16;
                diff = abs(diff_z);
            }}
        }}
        if(mindir == -1){{
            outdir[tid] = mindir;
            return;
        }}

        sign_x = diff_x < 0 ? 0:1;
        sign_y = diff_y < 0 ? 0:1;
        sign_z = diff_z < 0 ? 0:1;
       
        // Figure out exiting quadrant
        // Leaving in either x
        if(mindir == 0){{
            mindir = mindir + 4*sign_x + 2*sign_y + sign_z;
        // Leaving in either y
        }} else if(mindir == 8){{
            mindir = mindir + 4*sign_y + 2*sign_x + sign_z;
        // Leaving in either z
        }} else if(mindir == 16){{
            mindir = mindir + 4*sign_z + 2*sign_x + sign_y;
        }}
        outdir[tid] = mindir;
    }}
}}
'''