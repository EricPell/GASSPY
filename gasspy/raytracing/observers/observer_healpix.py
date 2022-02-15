import numpy as np
from scipy.spatial.transform import Rotation as R
import cupy
import cudf
import healpy as hp
import sys

from gasspy.settings.defaults import ray_dtypes, ray_defaults

class observer_healpix_class:
    def __init__(self, sim_data, ray_lrefine_min = None, ray_lrefine_max = None, ipix_lmin = None, **kwargs):
        print("HEALPIX OBSERVER HAS NOT BEEN UPDATED WITH THE RAYSTRUCTURE. TELL LOKE TO GET HIS ACT TOGETHER")
        sys.exit(0)
        # Minimum (eg. initial) refinement level of the rays/healpix 
        # If not minimum refinement level is not defined, either in call or in yaml file, 
        # we take it as the lowest possible (eg, 1)
        if ray_lrefine_min is not None:
            self.ray_lrefine_min = ray_lrefine_min
        elif "ray_refine_min" in sim_data.config_yaml:
            self.ray_lrefine_min = sim_data.config_yaml["ray_lrefine_min"]
        else:
            self.ray_lrefine_min = 1

        # Get the the number of sides for the mininmum level of refinement
        self.Nsides_min = 2**(ray_lrefine_min - 1)

        # Maximum refinement level of the rays/healpix 
        # If not maximum refinement level is not defined,
        # we default to a reasonably large number
        if ray_lrefine_max is not None:
            self.ray_lrefine_max = ray_lrefine_max
        elif "ray_refine_max" in sim_data.config_yaml:
            self.ray_lrefine_max = sim_data.config_yaml["ray_lrefine_max"]
        else:
            self.ray_lrefine_max = ray_defaults["ray_lrefine_max"]

        # Use the maximum and minimum level of refinement and construct an array containing 
        # the pixel area corresponding to each level
        self.pixel_area = cupy.zeros(self.ray_lrefine_max - self.ray_lrefine_min + 1)
        for lref in range(self.ray_lrefine_min, self.ray_lrefine_max+1):
            self.pixel_area[lref - self.ray_lrefine_min] = hp.pixelfunc.nside2pixarea(2**(lref-1))

        # pixels at the minimum refinement level to use. NOTE ASSUMES NESTED ORDERING
        # use the healpy.query methods with nest=True to generate appropriate lists
        # If not defined, take as all pixel at the minimum refinement level
        if ipix_lmin is not None:
            self.ipix_lmin = ipix_lmin
        elif "ipix_lmin" in sim_data.config_yaml:
            self.ipix_lmin = sim_data.config_yaml["ipix_min"]
        else:
            self.ipix_lmin = np.arange(hp.nside2npix(self.Nsides_min))

        self.Nrays = len(self.ipix_lmin)
        



        # Get the ray directions the pixels
        self.raydir = np.zeros((self.Nrays,3))
        self.raydir[:,0], self.raydir[:,1], self.raydir[:,2] = hp.pix2vec(self.Nsides_min, self.ipix_lmin, nest = True)



        # TODO: Change both here and in observer.py. sim_size needs to be defined, and 
        # self.Numz should no longer be in units of number of cells, but between 0 and 1
        if "sim_size_x" in sim_data.config_yaml:
            self.NumZ = np.sqrt(
                                sim_data.config_yaml["sim_size_x"]**2 + 
                                sim_data.config_yaml["sim_size_y"]**2 + 
                                sim_data.config_yaml["sim_size_z"]**2 
                               )
        else:
            self.NumZ = np.sqrt(np.sum(np.square(sim_data.Ncells)))
        

        # Note : not coded, but sim_data origin should default cuberoot(ncells)/2 if not defined by user
        # pov_center: the point at which long = lat = 0 from the viewpoint of the observer 
        self.pov_center = sim_data.origin

        # xp0_s is the position of the observer origin in the coordinate frame of the simulation
        self.xp0_s =   0.0
        self.yp0_s =   0.0
        self.zp0_s =   -0.5 * self.NumZ 
        
        self.__dict__.update(kwargs)

        # We now have a positon for the observer and a center point, we need to figure out the rotation matrix
        # such that long = lat = 0 at self.pov_center in the reference frame of the observer
        
        # In the observer coordinate frame
        pov_center_o = self.pov_center - np.array([self.xp0_s, self.yp0_s, self.zp0_s])
        # normalize
        pov_center_o = pov_center_o/np.sqrt(np.sum(np.square(pov_center_o)))
        yaw = np.arctan2(pov_center_o[2], pov_center_o[0])
        pitch = np.arctan2(np.sqrt(pov_center_o[2]**2 + pov_center_o[0]**2), pov_center_o[1])
        # TODO: this is degenerate, set to zero for now, but should return to be user settable
        roll = 0

        self.rotation_matrix = R.from_rotvec(np.array([pitch, yaw, roll])).as_matrix()
        self.xps = -1
        self.yps = -1

    def get_first_rays(self):
        """
            Takes a ray dataframe and populates it with the original set of rays corresponding to this observer
        """
        # Initialize a dataframe
        rayDF = cudf.DataFrame()

        # Set the observation plane definitions
        # NOTE: healpix pixels are defined by one number. we take xp to be the ipix and yp to be 0
        rayDF["xp"] = cupy.array(self.ipix_lmin, dtype = ray_dtypes["yp"])
        rayDF["yp"] = cupy.zeros(self.Nrays, dtype = ray_dtypes["xp"])

        #All rays start from one point
        rayDF["xi"] = cupy.full(self.Nrays, self.xp0_s, dtype = ray_dtypes["xi"])
        rayDF["yi"] = cupy.full(self.Nrays, self.yp0_s, dtype = ray_dtypes["yi"])
        rayDF["zi"] = cupy.full(self.Nrays, self.zp0_s, dtype = ray_dtypes["zi"])

        # The directions of each ray needs to be rotated 
        for i, raydir in enumerate(["raydir_x", "raydir_y", "raydir_z"]):
            rayDF[raydir] = cupy.array(
                                        self.raydir[:,0]*float(self.rotation_matrix[i][0]) +
                                        self.raydir[:,1]*float(self.rotation_matrix[i][1]) +
                                        self.raydir[:,2]*float(self.rotation_matrix[i][2])                                        
                                        ,dtype = ray_dtypes[raydir])

        # set the refinement level of the rays
        rayDF["ray_lrefine"] = cupy.array(self.ray_lrefine_min)
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
        
        # allocate the arrays
        xp_new = cupy.zeros((len(parent_rayDF),4), dtype = ray_dtypes["xp"])
        yp_new = cupy.zeros((len(parent_rayDF),4), dtype = ray_dtypes["yp"])



        # split the pixel into four
        # index at level i+1 ipix_{i+1} = 4*ipix_{i} + [0,1,2,3]
        xp_new[:,0] = parent_rayDF["xp"].values*4 
        xp_new[:,1] = parent_rayDF["xp"].values*4 + 1
        xp_new[:,2] = parent_rayDF["xp"].values*4 + 2
        xp_new[:,3] = parent_rayDF["xp"].values*4 + 3


        
        # and set the pixel positions in the child dataframe
        children_rayDF["xp"] = xp_new.ravel()
        children_rayDF["yp"] = yp_new.ravel()

        return

    def set_child_positions(self, children_rayDF, parent_rayDF):
        """
            Calculate the current positions of the children, given the parents
        """
 
        # First we get the new directions given the child rays
        # Preallocate the array of directions
        raydirs = np.zeros((len(children_rayDF),3))
        # get the new refinement level of the rays and their ipixes
        lrefine = cupy.repeat(parent_rayDF["ray_lrefine"].values,4) + 1
        ipixs = children_rayDF["xp"].values.astype(cupy.int64)
        unique_lrefs = cupy.unique(lrefine)

        # Get the ray directions
        for lref in unique_lrefs.get():
            idxs = cupy.where(lrefine == lref)[0]
            idxs_cpu = idxs.get()
            raydirs[idxs_cpu, 0], raydirs[idxs_cpu, 1], raydirs[idxs_cpu, 2] =  hp.pix2vec(int(2**(lref-1)), ipixs[idxs].get(), nest = True)


        # Rotate the directions to the desired orientation
        for i, raydir in enumerate(["raydir_x", "raydir_y", "raydir_z"]):
            children_rayDF[raydir] = cupy.array(
                                        raydirs[:,0]*float(self.rotation_matrix[i][0]) +
                                        raydirs[:,1]*float(self.rotation_matrix[i][1]) +
                                        raydirs[:,2]*float(self.rotation_matrix[i][2])                                        
                                        ,dtype = ray_dtypes[raydir])

        # Next figure out how far away the parent ray went
        parent_total_pathlength = cupy.sqrt(cupy.square(parent_rayDF["xi"].values - self.xp0_s) + 
                                            cupy.square(parent_rayDF["yi"].values - self.yp0_s) +
                                            cupy.square(parent_rayDF["zi"].values - self.zp0_s))

        # Copy out to each child
        child_pathlength = cupy.repeat(parent_total_pathlength, 4)

        # Add the pathlength to the original position
        xi_new = cupy.array(self.xp0_s + children_rayDF["raydir_x"].values * child_pathlength)
        yi_new = cupy.array(self.yp0_s + children_rayDF["raydir_y"].values * child_pathlength)
        zi_new = cupy.array(self.zp0_s + children_rayDF["raydir_z"].values * child_pathlength)

        # set the positions in the child dataframe
        children_rayDF["xi"] = xi_new
        children_rayDF["yi"] = yi_new
        children_rayDF["zi"] = zi_new

        return

    def set_ray_area(self, rayDF):
        """
            Sets the local area of the rays solid angle 
        """

        # We have a list of areas saved from before, multiply this with the total pathlength squared 
        # TODO: add a check to make sure we dont exceed the array
    
        rayDF["ray_area"] = self.pixel_area[rayDF["ray_lrefine"].values-self.ray_lrefine_min] * (cupy.square(rayDF["xi"].values - self.xp0_s) + 
                                                                                          cupy.square(rayDF["yi"].values - self.yp0_s) +
                                                                                          cupy.square(rayDF["zi"].values - self.zp0_s))
        
        return
    def update_ray_area(self, rayDF):
        """
            In the case of changing area (non paralell rays)
            have a method to update
        """
        # Just call original....
        self.set_ray_area(rayDF)
        return