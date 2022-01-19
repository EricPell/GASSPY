from cupy._indexing.generate import CClass
import numpy as np
from scipy.spatial.transform import Rotation as R
import cupy
import cudf
from ..settings.defaults import ray_dtypes, ray_defaults
import sys
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
            self.xps = (np.arange(0, self.Nxp) + 0.5)*self.detector_size_x/self.Nxp
            self.yps = (np.arange(0, self.Nyp) + 0.5)*self.detector_size_y/self.Nyp
            
            # Define the entire mesh of points 
            self.xp, self.yp = np.meshgrid(self.xps, self.yps)
            self.xp = self.xp.ravel()
            self.yp = self.yp.ravel()

            # Total number of original rays
            self.Nrays = len(self.xp)
            # Refinement level of the initial rays
            self.ray_lrefine = int(np.log2(self.Nxp))
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


        self.rotation_matrix = R.from_rotvec(np.array([self.pitch, self.yaw, self.roll])*np.pi/180).as_matrix()

    def get_first_rays(self):
        """
            Takes a ray dataframe and populates it with the original set of rays corresponding to this observer
        """
        # Initialize a dataframe
        rayDF = cudf.DataFrame()

        # Set the observation plane definitions
        rayDF["xp"] = cupy.array(self.xp, dtype = ray_dtypes["yp"])
        rayDF["yp"] = cupy.array(self.yp, dtype = ray_dtypes["xp"])
        # Transform the xp and yp in the observers coordinate frame to that of the simulation
        for i, xi in enumerate(["xi", "yi", "zi"]):
            rayDF[xi] = cupy.full(self.Nrays, self.xp0_r * float(self.rotation_matrix[i][0]) + self.yp0_r * float(self.rotation_matrix[i][1]) + self.zp0_r * float(self.rotation_matrix[i][2]) + self.rot_origin[i],
                                              dtype = ray_dtypes[xi])
            rayDF[xi] += (rayDF["xp"] * float(self.rotation_matrix[i][0]) +
                          rayDF["yp"] * float(self.rotation_matrix[i][1]))

        # Set the direction of the individual rays
        for i, raydir in enumerate(["raydir_x", "raydir_y", "raydir_z"]):
            rayDF[raydir] = cupy.full(self.Nrays, 1.0 * float(self.rotation_matrix[i][2]), dtype = ray_dtypes[raydir])

        # set the refinement level of the rays
        rayDF["ray_lrefine"] = cupy.array(self.ray_lrefine)
        # set the different ID numbers relevant to the rays
        # The ID of the ray 
        rayDF["global_rayid"] = cupy.arange(self.Nrays, dtype = ray_dtypes["global_rayid"])
        # IDs of the parents and corresponding split events, set to null values
        for i, id in enumerate(["pid", "pevid", "cevid"]):
            rayDF[id] = cupy.full(self.Nrays, ray_defaults[id], dtype = ray_dtypes[id])
        # ID of the branch, which, since this is the first ray of the branch, is the same as the ID of the ray
        rayDF["aid"] = cupy.arange(self.Nrays, dtype = ray_dtypes["aid"])


               
        return rayDF


    def create_child_rays(self, parent_rayDF):
        """
            Creates children rays for a set of rays that are to be split
        """
        # How many children
        Nchild = 4*len(parent_rayDF)
        
        # Fields that are set here
        fields_new = ["xp", "yp", 
                      "xi", "yi", "zi"]

        fields_from_parent = ["raydir_x", "raydir_y", "raydir_z"]
        children_rayDF = cudf.DataFrame()        
        # allocate the new fields of the children 
        for field in fields_new:
            children_rayDF[field] = cupy.zeros(Nchild, dtype = ray_dtypes[field])

        for field in fields_from_parent:
            children_rayDF[field] = cupy.repeat(parent_rayDF[field].values, repeats=4)
        # Set the position and pixel of the children
        self.set_child_pixels(children_rayDF, parent_rayDF)
        self.set_child_positions(children_rayDF, parent_rayDF)
    
        return  children_rayDF

    def set_child_pixels(self, children_rayDF, parent_rayDF):
        """
            Calculate the xp and yp of the children given the parents
        """
        
        # start by getting the size of the parent pixel 
        dxp = self.detector_size_x*2**(-parent_rayDF["ray_lrefine"].values.astype(ray_dtypes["xp"]))
        dyp = self.detector_size_y*2**(-parent_rayDF["ray_lrefine"].values.astype(ray_dtypes["yp"]))

        # allocate the arrays
        xp_new = cupy.zeros((len(parent_rayDF),4), dtype = ray_dtypes["xp"])
        yp_new = cupy.zeros((len(parent_rayDF),4), dtype = ray_dtypes["yp"])

        # split the pixel into four
        # add 0.5 to xp
        xp_new[:,0] = parent_rayDF["xp"].values - dxp*0.25
        xp_new[:,1] = parent_rayDF["xp"].values + dxp*0.25
        xp_new[:,2] = xp_new[:,0]
        xp_new[:,3] = xp_new[:,1]


        # add 0.5 to yp
        yp_new[:,0] = parent_rayDF["yp"].values - dyp*0.25
        yp_new[:,1] = yp_new[:,0]
        yp_new[:,2] = parent_rayDF["yp"].values + dyp*0.25
        yp_new[:,3] = yp_new[:,2]
        
        # and set the pixel positions in the child dataframe
        children_rayDF["xp"] = xp_new.ravel()
        children_rayDF["yp"] = yp_new.ravel()

        return

    def set_child_positions(self, children_rayDF, parent_rayDF):
        """
            Calculate the current positions of the children, given the parents
        """
        # In the parallel case this is a simple shift of dxp*0.5 
        # TODO: Change so that rays start from the centre of the pixel and shift by +-0.25*dxp
        # start by getting the size of the parent pixel 
        # TODO: This is a repeat calculation from before... maybe we can save these somehow..
        dxp = self.detector_size_x*2**(-parent_rayDF["ray_lrefine"].values.astype(ray_dtypes["xp"]))
        dyp = self.detector_size_y*2**(-parent_rayDF["ray_lrefine"].values.astype(ray_dtypes["yp"]))

        # allocate positional arrays
        xi_new = cupy.zeros((len(parent_rayDF),4), dtype = ray_dtypes["xi"]) 
        yi_new = cupy.zeros((len(parent_rayDF),4), dtype = ray_dtypes["yi"]) 
        zi_new = cupy.zeros((len(parent_rayDF),4), dtype = ray_dtypes["zi"])

        xp_shift = cupy.zeros((len(parent_rayDF),4), dtype = ray_dtypes["xp"])
        yp_shift = cupy.zeros((len(parent_rayDF),4), dtype = ray_dtypes["yp"])        

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
        xi_new[:,:] = cupy.array(parent_rayDF["xi"].values)[:,cupy.newaxis] + xp_shift*float(self.rotation_matrix[0][0]) + yp_shift*float(self.rotation_matrix[0][1])
        yi_new[:,:] = cupy.array(parent_rayDF["yi"].values)[:,cupy.newaxis] + xp_shift*float(self.rotation_matrix[1][0]) + yp_shift*float(self.rotation_matrix[1][1])
        zi_new[:,:] = cupy.array(parent_rayDF["zi"].values)[:,cupy.newaxis] + xp_shift*float(self.rotation_matrix[2][0]) + yp_shift*float(self.rotation_matrix[2][1])

        # set the positions in the child dataframe
        children_rayDF["xi"] = xi_new.ravel()
        children_rayDF["yi"] = yi_new.ravel()
        children_rayDF["zi"] = zi_new.ravel()

        return

    def set_ray_area(self, rayDF):
        """
            Sets the local area of the rays solid angle 
        """
        # In the case of parallel rays this is constant for a given ray refinement level
        rayDF["ray_area"] = self.detector_size_y*self.detector_size_x * 4**(-rayDF["ray_lrefine"].values.astype(float))
        
        return
    def update_ray_area(self, rayDF):
        """
            In the case of changing area (non paralell rays)
            have a method to update
        """
        # NOT IMPLEMENTED, DO NOTHING AND RETURN
        return