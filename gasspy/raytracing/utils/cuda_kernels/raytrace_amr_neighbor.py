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
              ["raydir_x", "raydir_x", "*"], 
              ["raydir_y", "raydir_y", "*"], 
              ["raydir_z", "raydir_z", "*"],
              ["ray_status", "ray_status", "*"],  
              ["cell_index", "cell_index", "*"],
              ["cell_center_x", "xi", "*"],
              ["cell_center_y", "yi", "*"], 
              ["cell_center_z", "zi", "*"],
              ["outdir", "amr_lrefine", "*"],
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
raytrace_amr_neighbor_code_string = r'''
extern "C" __global__
void __raytrace_kernel__('''+argument_string+'''){{
    int mindir;
    '''+dx_type+''' dx, dy, dz, ix, iy, iz, newpath, plane_x, plane_y, plane_z;
    '''+amr_lrefine_type+''' iamr;
    '''+index1D_type+''' Nmax;     

    unsigned '''+rayid_dtype+''' tid = blockDim.x * blockIdx.x + threadIdx.x;

    '''+dx_type+''' sim_size_half_x = {sim_size_half_x};
    '''+dx_type+''' sim_size_half_y = {sim_size_half_y};
    '''+dx_type+''' sim_size_half_z = {sim_size_half_z};
    if(tid < nrays){{
        outdir[tid] = -1;
        if(ray_status[tid] > 0) {{
            return;
        }}
        if( !((fabs(xi[tid] - sim_size_half_x) <= sim_size_half_x) && (fabs(yi[tid] - sim_size_half_y) <= sim_size_half_y) && (fabs(zi[tid] - sim_size_half_z) <= sim_size_half_z))) {{
            cell_index[tid] = -1;
            pathlength[tid] = 0;
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
        
        ix = xi[tid];
        iy = yi[tid];
        iz = zi[tid];

        // init to unreasonably high number
        pathlength[tid] = 1e30;
        mindir = -1;
        // check for closest distance to cell boundary by looking for the closest int in each cardinal axis away from the current position
        // a thousand of a cell width is added as padding such that the math is (almost) always correct
        // NOTE: this could be wrong if a ray is very close to an interface. So depending on the angle of raydir
        // With respect to the cells, errors can occur

        // in x
        if(raydir_x[tid] > 0){{
            plane_x = cell_center_x[tid] + 0.5*dx;
            if(abs(plane_x -ix)<1e-14*ix){{
                newpath = 0.0;
            }} else {{
                newpath = (plane_x-ix)/raydir_x[tid];
            }}
            if((pathlength[tid] > newpath) && (newpath >= 0)){{
                pathlength[tid] = newpath;
                mindir = 4;
            }}
        }} else if(raydir_x[tid] < 0) {{            
            plane_x = cell_center_x[tid] - 0.5*dx;
            if(abs(plane_x -ix)<1e-14*ix){{
                newpath = 0.0;
            }} else {{
                newpath = (plane_x-ix)/raydir_x[tid];
            }}
            if((pathlength[tid] > newpath) && (newpath >= 0)) {{
                pathlength[tid] = newpath;
                mindir = 0;
            }}
        }}
        // in y
        if(raydir_y[tid] > 0) {{
            plane_y = cell_center_y[tid] + 0.5*dy;
            if(abs(plane_y - iy)<1e-14*iy){{
                newpath = 0.0;
            }} else {{
                newpath = (plane_y-iy)/raydir_y[tid];
            }}
            if((pathlength[tid] > newpath) && (newpath >= 0)){{
                pathlength[tid] = newpath;
                mindir = 12;
            }}
        }} else if(raydir_y[tid] < 0) {{
            plane_y = cell_center_y[tid] - 0.5*dy;
            if(abs(plane_y -iy )<1e-14*iy){{
                newpath = 0.0;
            }} else {{
                newpath = (plane_y-iy)/raydir_y[tid];
            }}
            if((pathlength[tid] > newpath) && (newpath >=0)){{
                pathlength[tid] = newpath;
                mindir = 8;
            }}
        }}

        // in z
        if(raydir_z[tid] > 0){{
            plane_z = cell_center_z[tid] + 0.5*dz;
            if(abs(plane_z -iz)<1e-14*iz){{
                newpath = 0.0;
            }} else {{
                newpath = (plane_z-iz)/raydir_z[tid];
            }}
            if((pathlength[tid] > newpath) && (newpath >=0)){{
                pathlength[tid] = newpath;
                mindir = 20;
            }}
        }} else if(raydir_z[tid] < 0) {{
            plane_z = cell_center_z[tid] - 0.5*dz;
            if(abs(plane_z -iz)<1e-14*iz){{
                newpath = 0.0;
            }} else {{
                newpath = (plane_z-iz)/raydir_z[tid];
            }}
            if((pathlength[tid] > newpath) && (newpath >=0)){{
                pathlength[tid] = newpath;
                mindir = 16;
            }}
        }}
        if(mindir == -1){{
            outdir[tid] = mindir;
            pathlength[tid] = 0;
            return;
        }}
        // Update ray positions
        xi[tid] = xi[tid] + pathlength[tid]*raydir_x[tid];
        yi[tid] = yi[tid] + pathlength[tid]*raydir_y[tid];
        zi[tid] = zi[tid] + pathlength[tid]*raydir_z[tid];
        
        // Figure out exiting quadrant
        // Leaving in either x
        if((mindir == 0) || (mindir == 4)){{
            if((yi[tid]-cell_center_y[tid]) > 0){{
                mindir = mindir + 2;
            }}
            if((zi[tid]-cell_center_z[tid]) > 0){{
                mindir = mindir + 1;
            }}
        // Leaving in either y
        }} else if((mindir == 8) || (mindir == 12)){{
            if((xi[tid]-cell_center_x[tid]) > 0){{
                mindir = mindir + 2;
            }}
            if((zi[tid]-cell_center_z[tid]) > 0){{
                mindir = mindir + 1;
            }}
        // Leaving in either z
        }} else if((mindir == 16) || (mindir == 20)){{
            if((xi[tid]-cell_center_x[tid]) > 0){{
                mindir = mindir + 2;
            }}
            if((yi[tid]-cell_center_y[tid]) > 0){{
                mindir = mindir + 1;
            }}
        }}
        outdir[tid] = ('''+amr_lrefine_type+''')mindir;

        // Check again if the rays has left the box (We should be more clever here to only do this once)
        if(raydir_x[tid]>0){{
            if((xi[tid] >= 1-1e-14) || (xi[tid]<0)){{
                ray_status[tid] = 2;
            }}
        }} else {{
            if((xi[tid] > 1) || (xi[tid]<=1e-14)){{
                ray_status[tid] = 2;
            }}            
        }}
        if(raydir_y[tid]>0){{
            if((yi[tid] >= 1-1e-14) || (yi[tid]<0)){{
                ray_status[tid] = 2;
            }}
        }} else {{
            if((yi[tid] > 1) || (yi[tid]<=1e-14)){{
                ray_status[tid] = 2;
            }}            
        }}
        if(raydir_z[tid]>0){{
            if((zi[tid] >= 1-1e-14) || (zi[tid]<0)){{
                ray_status[tid] = 2;
            }}
        }} else {{
            if((zi[tid] > 1) || (zi[tid]<=1e-14)){{
                ray_status[tid] = 2;
            }}            
        }}
    }}
}}
'''