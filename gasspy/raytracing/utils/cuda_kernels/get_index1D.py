from gasspy.settings.defaults import ray_dtypes
from gasspy.raytracing.utils.cuda_kernels.dtype_functions import python_dtype_to_nvcc, get_argument_string

# Input variable dtypes (as described in settings.defaults.py) and and extra string showing their size
input_vars = [["xi", "xi", "*"],
              ["yi", "yi", "*"], 
              ["zi", "zi", "*"],
              ["index1D", "index1D", "*"],
              ["amr_lrefine", "amr_lrefine", "*"],
              ["dx_lref", "xi", "*"],
              ["Nmax_lref", "index1D", "*"],  
              ["amr_lrefine_min", "amr_lrefine", ""],
              ["nrays", "global_rayid", "" ]]

# Loop over each input variable, find corresponding dtypes and name of dtype in the compile string
argument_string = get_argument_string(input_vars)

# Names of types that need to be consistent between c and python
dx_type =  python_dtype_to_nvcc(ray_dtypes["xi"](0))
index1D_type = python_dtype_to_nvcc(ray_dtypes["index1D"](0))
amr_lrefine_type = python_dtype_to_nvcc(ray_dtypes["amr_lrefine"](0))
rayid_dtype = python_dtype_to_nvcc(ray_dtypes["global_rayid"](0))

get_index1D_code_string = r'''
extern "C" __global__
void __get_index1D__('''+argument_string+'''){{
    int mindir;
    '''+dx_type+''' dx, dy, dz, ix, iy, iz;
    '''+amr_lrefine_type+''' iamr;
    '''+index1D_type+''' Nmax;     

    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < nrays){{
        
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

        index1D[tid] = ('''+index1D_type+''')iz + Nmax*('''+index1D_type+''')iy + Nmax*Nmax*('''+index1D_type+''')ix;
    }}
}}
'''