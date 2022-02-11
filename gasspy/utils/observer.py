import numpy as np
from scipy.spatial.transform import Rotation as R
import cupy

from gasspy.settings.defaults import ray_dtypes, ray_defaults
from gasspy.raystructures.global_rays import global_ray_class

class observer_plane_class:
    def __init__(self, sim_data, Nxp = None, Nyp = None, 
                                detector_size_x = None, detector_size_y = None,
                                z_subsamples = 3, scale_l_cgs = None, 
                                planeDefinitionMethod = None, **kwargs):
        
        # if not defined assume that the simulation data is cubic, and take the plot grid to use the same dimensions
        if Nxp is not None:
            self.Nxp = Nxp
        elif "Nxp" in sim_data.config_yaml:
            self.Nxp = sim_data.config_yaml["Nxp"]
        else:
            self.Nxp = sim_data.Ncells[0]

        if Nyp is not None:
            self.Nyp = Nyp
        elif "Nyp" in sim_data.config_yaml:
            self.Nyp = sim_data.config_yaml["Nyp"]
        else:
            self.Nyp = sim_data.Ncells[1]

        if detector_size_x is not None:
            self.detector_size_x = detector_size_x
        elif "detector_size_x" in sim_data.config_yaml:
            self.detector_size_x = sim_data.config_yaml["detector_size_x"]
        else:
            self.detector_size_x = 1

        if detector_size_y is not None:
            self.detector_size_y = detector_size_y
        elif "detector_size_y" in sim_data.config_yaml:
            self.detector_size_y = sim_data.config_yaml["detector_size_y"]
        else:
            self.detector_size_y = 1


        # This is immutable. Never shall xps and yps change. They are pixel indicies for a detector
        if planeDefinitionMethod is None :
            self.xps = (cupy.arange(0, self.Nxp) + 0.5)*self.detector_size_x/self.Nxp
            self.yps = (cupy.arange(0, self.Nyp) + 0.5)*self.detector_size_y/self.Nyp
            
            # Define the entire mesh of points 
            self.xp, self.yp = cupy.meshgrid(self.xps, self.yps)
            self.xp = self.xp.ravel()
            self.yp = self.yp.ravel()

            # Total number of original rays
            self.Nrays = len(self.xp)
            # Refinement level of the initial rays
            self.ray_lrefine = int(cupy.log2(self.Nxp))
        else:
            # user defined method for defining the sets of xp and yp for all rays (if for example high rez region already known)
            self.xp, self.yp, self.ray_lrefine, self.Nrays = planeDefinitionMethod()


        if "sim_size_x" in sim_data.config_yaml:
            self.NumZ = np.sqrt(
                                sim_data.config_yaml["sim_size_x"]**2 + 
                                sim_data.config_yaml["sim_size_y"]**2 + 
                                sim_data.config_yaml["sim_size_z"]**2 
                               )
        else:
            self.NumZ = np.sqrt(np.sum(np.square(sim_data.Ncells)))
        
        self.pitch = 0 
        self.yaw   = 0
        self.roll  = 0

        # Note : not coded, but sim_data origin should default cuberoot(ncells)/2 if not defined by user
        self.rot_origin = sim_data.origin
        self.xp0_s =   0.0
        self.yp0_s =   0.0
        self.zp0_s =   -0.5 * self.NumZ 
        
        self.__dict__.update(kwargs)

        # the position on xp = 0 yp = 0 zp = 0 with respect to the rotation origin
        # By default we are not rotated so x and y are at the midplane
        self.xp0_r = self.xp0_s - self.rot_origin[0]
        self.yp0_r = self.yp0_s - self.rot_origin[1]
        # to stay inside the box, the minium distance is 0.5*sqrt(Nx**2+Ny**2+Nz**2) away from origin
        self.zp0_r = self.zp0_s - self.rot_origin[2]

        self.__dict__.update(kwargs)


        self.rotation_matrix = cupy.array(R.from_rotvec(np.array([self.pitch, self.yaw, self.roll])*np.pi/180).as_matrix())
        self.ray_area = self.detector_size_y*self.detector_size_x*4**(-cupy.arange(ray_defaults["ray_lrefine_min"], ray_defaults["ray_lrefine_max"]).astype(ray_dtypes["xi"]))
        self.dxs = self.detector_size_x*2**(-cupy.arange(ray_defaults["ray_lrefine_min"], ray_defaults["ray_lrefine_max"]).astype(ray_dtypes["xi"]))
        self.dys = self.detector_size_y*2**(-cupy.arange(ray_defaults["ray_lrefine_min"], ray_defaults["ray_lrefine_max"]).astype(ray_dtypes["xi"]))



    def get_first_rays(self, old_rays = None):
        """
            Takes a ray dataframe and populates it with the original set of rays corresponding to this observer
        """
        # If no old rays exist initialize the global ray datastructure with a set of rays
        if old_rays is None:
            global_rays = global_ray_class()
        else:
            global_rays = old_rays

        # Append rays defined by this observer
        global_rayids = global_rays.append(len(self.xp))

        # Set the observation plane definitions
        global_rays.set_field("xp", self.xp, index = global_rayids)
        global_rays.set_field("yp", self.yp, index = global_rayids)
        # Transform the xp and yp in the observers coordinate frame to that of the simulation
        for i, xi in enumerate(["xi", "yi", "zi"]):
            global_rays.set_field(xi, cupy.full(self.Nrays, self.xp0_r * float(self.rotation_matrix[i][0]) + self.yp0_r * float(self.rotation_matrix[i][1]) + self.zp0_r * float(self.rotation_matrix[i][2]) + self.rot_origin[i],
                                              dtype = ray_dtypes[xi]) +
                                      self.xp * float(self.rotation_matrix[i][0]) +
                                      self.yp * float(self.rotation_matrix[i][1]), index = global_rayids)

        # Set the direction of the individual rays
        for i, raydir in enumerate(["raydir_x", "raydir_y", "raydir_z"]):
            global_rays.set_field(raydir, cupy.full(self.Nrays, 1.0 * float(self.rotation_matrix[i][2]), dtype = ray_dtypes[raydir]), index = global_rayids)

        # set the refinement level of the rays
        global_rays.set_field("ray_lrefine", cupy.array(self.ray_lrefine), index = global_rayids)

        # IDs of the parents and corresponding split events, set to null values
        for i, id in enumerate(["pid", "pevid", "cevid"]):
            global_rays.set_field(id, cupy.full(self.Nrays, ray_defaults[id], dtype = ray_dtypes[id]), index = global_rayids)
        # ID of the branch, which, since this is the first ray of the branch, is the same as the ID of the ray
        global_rays.set_field("aid", global_rayids, index = global_rayids)

        # Initialize the trace status of the rays
        global_rays.set_field("trace_status", 0, index=global_rayids)
        
        # Initialize the amr refinement of the rays
        global_rays.set_field("amr_lrefine", ray_defaults["amr_lrefine"], index  = global_rayids)
        return global_rays


    def create_child_rays(self, parent_rays):
        """
            Creates children rays for a set of rays that are to be split
        """
        # How many children
        Nchild = 4*len(parent_rays["xp"])
        
        # Fields that are set here
        fields_new = ["xp", "yp", 
                      "xi", "yi", "zi"]

        fields_from_parent = ["raydir_x", "raydir_y", "raydir_z"]
        child_rays = {}        
        # allocate the new fields of the children 
        for field in fields_new:
            child_rays[field] = cupy.zeros(Nchild, dtype = ray_dtypes[field])

        for field in fields_from_parent:
            child_rays[field] = cupy.repeat(parent_rays[field], repeats=4)
        # Set the position and pixel of the children
        self.set_child_fields(child_rays, parent_rays)
    
        return  child_rays

    def set_child_fields(self, child_rays, parent_rays):
        # start by getting the size of the parent pixel 
        dxp = self.dxs[parent_rays["ray_lrefine"] - ray_defaults["ray_lrefine_min"]]
        dyp = self.dys[parent_rays["ray_lrefine"] - ray_defaults["ray_lrefine_min"]]        

        self.set_child_pixels(child_rays, parent_rays, dxp, dyp)
        self.set_child_positions(child_rays, parent_rays, dxp, dyp)

        return

    def set_child_pixels(self, child_rays, parent_rays, dxp, dyp):
        """
            Calculate the xp and yp of the children given the parents
        """

        # allocate the arrays
        xp_new = cupy.zeros((len(parent_rays["xp"]),4), dtype = ray_dtypes["xp"])
        yp_new = cupy.zeros((len(parent_rays["xp"]),4), dtype = ray_dtypes["yp"])

        # split the pixel into four
        # add 0.5 to xp
        xp_new[:,0] = parent_rays["xp"] - dxp*0.25
        xp_new[:,1] = parent_rays["xp"] + dxp*0.25
        xp_new[:,2] = xp_new[:,0]
        xp_new[:,3] = xp_new[:,1]


        # add 0.5 to yp
        yp_new[:,0] = parent_rays["yp"] - dyp*0.25
        yp_new[:,1] = yp_new[:,0]
        yp_new[:,2] = parent_rays["yp"] + dyp*0.25
        yp_new[:,3] = yp_new[:,2]
        
        # and set the pixel positions in the child dataframe
        child_rays["xp"] = xp_new.ravel()
        child_rays["yp"] = yp_new.ravel()

        return

    def set_child_positions(self, child_rays, parent_rays, dxp, dyp):
        """
            Calculate the current positions of the children, given the parents
        """

        # allocate positional arrays
        xi_new = cupy.zeros((len(parent_rays["xp"]),4), dtype = ray_dtypes["xi"]) 
        yi_new = cupy.zeros((len(parent_rays["xp"]),4), dtype = ray_dtypes["yi"]) 
        zi_new = cupy.zeros((len(parent_rays["xp"]),4), dtype = ray_dtypes["zi"])

        xp_shift = cupy.zeros((len(parent_rays["xp"]),4), dtype = ray_dtypes["xp"])
        yp_shift = cupy.zeros((len(parent_rays["xp"]),4), dtype = ray_dtypes["yp"])        

        # get the shift in each direction
        xp_shift[:,0] = -dxp*0.25 
        xp_shift[:,1] = dxp*0.25
        xp_shift[:,2] = xp_shift[:,0]
        xp_shift[:,3] = xp_shift[:,1]
        
        yp_shift[:,0] = -dyp*0.25 
        yp_shift[:,1] = yp_shift[:,0]
        yp_shift[:,2] = dyp*0.25
        yp_shift[:,3] = yp_shift[:,2]

        # Add the rotated shift
        xi_new[:,:] = cupy.array(parent_rays["xi"])[:,cupy.newaxis] + xp_shift*float(self.rotation_matrix[0][0]) + yp_shift*float(self.rotation_matrix[0][1])
        yi_new[:,:] = cupy.array(parent_rays["yi"])[:,cupy.newaxis] + xp_shift*float(self.rotation_matrix[1][0]) + yp_shift*float(self.rotation_matrix[1][1])
        zi_new[:,:] = cupy.array(parent_rays["zi"])[:,cupy.newaxis] + xp_shift*float(self.rotation_matrix[2][0]) + yp_shift*float(self.rotation_matrix[2][1])

        # set the positions in the child dataframe
        child_rays["xi"] = xi_new.ravel()
        child_rays["yi"] = yi_new.ravel()
        child_rays["zi"] = zi_new.ravel()

        return

    def set_ray_area(self, ray_struct):
        """
            Sets the local area of the rays solid angle 
        """
        # In the case of parallel rays this is constant for a given ray refinement level
        ray_struct.set_field("ray_area", self.ray_area[ray_struct.get_field("ray_lrefine") - ray_defaults["ray_lrefine_min"]])
        
        return
    def update_ray_area(self, rayDF):
        """
            In the case of changing area (non paralell rays)
            have a method to update
        """
        # NOT IMPLEMENTED, DO NOTHING AND RETURN
        return