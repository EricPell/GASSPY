import numpy as np
from scipy.spatial.transform import Rotation as R
import cupy
import healpy as hpix
import sys

from gasspy.raystructures.global_rays import global_ray_class, base_ray_class
from gasspy.settings.defaults import ray_dtypes, ray_defaults
from gasspy.io.gasspy_io import check_parameter_in_config, read_yaml

class observer_healpix_class:
    def __init__(self, gasspy_config, 
                 ray_lrefine_min = None, 
                 ray_lrefine_max = None, 
                 ipix_lmin = None, 
                 min_pix_lmin = None, 
                 max_pix_lmin = None, 
                 ignore_within_distance = None, 
                 pov_center = None,
                 observer_center = None):
       
        if isinstance(gasspy_config, str):
            self.gasspy_config = read_yaml(gasspy_config)
        else:
            self.gasspy_config = gasspy_config
        
        
        self.ignore_within_distance = check_parameter_in_config(self.gasspy_config, "ignore_within_distance", ignore_within_distance, 0)
    
        # Minimum (eg. initial) refinement level of the rays/healpix 
        # If not minimum refinement level is not defined, either in call or in yaml file, 
        # we take it as the lowest possible (eg, 1)
        self.ray_lrefine_min = check_parameter_in_config(self.gasspy_config, "ray_lrefine_min", ray_lrefine_min, 1)

        # Get the the number of sides for the mininmum level of refinement
        self.Nsides_min = 2**(self.ray_lrefine_min - 1)

        # Maximum refinement level of the rays/healpix 
        # If not maximum refinement level is not defined,
        # we default to a reasonably large number
        self.ray_lrefine_max = check_parameter_in_config(self.gasspy_config, "ray_lrefine_max", ray_lrefine_max, ray_defaults["ray_lrefine_max"])

        # Use the maximum and minimum level of refinement and construct an array containing 
        # the pixel area corresponding to each level
        self.pixel_area = cupy.zeros(self.ray_lrefine_max - self.ray_lrefine_min + 1)
        for lref in range(self.ray_lrefine_min, self.ray_lrefine_max+1):
            self.pixel_area[lref - self.ray_lrefine_min] = hpix.pixelfunc.nside2pixarea(2**(lref-1))

        # pixels at the minimum refinement level to use. NOTE ASSUMES NESTED ORDERING
        # use the healpy.query methods with nest=True to generate appropriate lists
        if ipix_lmin is not None:
            self.ipix_lmin = ipix_lmin
        else:
            # If not defined as a list, we take a user defined range, defaulting as all of the lowest refinement level

            # Starting ipix
            if min_pix_lmin is not None:
                self.min_pix_lmin = min_pix_lmin
            elif "min_pix_lmin" in gasspy_config:
                self.min_pix_lmin = gasspy_config["min_pix_lmin"]
            else:
                self.min_pix_lmin = 0
            
            # Final ipix
            if max_pix_lmin is not None:
                self.max_pix_lmin = max_pix_lmin
            elif "max_pix_lmin" in gasspy_config:
                self.max_pix_lmin = gasspy_config["max_pix_lmin"]
            else:
                self.max_pix_lmin = hpix.nside2npix(self.Nsides_min) - 1

            # Create range
            self.ipix_lmin = np.arange(self.min_pix_lmin, self.max_pix_lmin + 1)

        self.Npixels = len(self.ipix_lmin)
        



        # Get the ray directions the pixels
        self.raydir = np.zeros((self.Npixels,3))
        self.raydir[:,0], self.raydir[:,1], self.raydir[:,2] = hpix.pix2vec(self.Nsides_min, self.ipix_lmin, nest = True)



        # TODO: Change both here and in observer.py. sim_size needs to be defined, and 
        # self.Numz should no longer be in units of number of cells, but between 0 and 1
        self.NumZ = np.sqrt(
                            gasspy_config["sim_size_x"]**2 + 
                            gasspy_config["sim_size_y"]**2 + 
                            gasspy_config["sim_size_z"]**2 
                            )



        #  pov_center: the point at which long = lat = 0 from the viewpoint of the observer. Default to box center 
        default = check_parameter_in_config(self.gasspy_config, "origin", None, np.array([0.5,0.5,0.5]))
        self.pov_center = check_parameter_in_config(self.gasspy_config, "pov_center", pov_center, default)

        # observer center is the position of the observer center in the coordinate frame of the simulation
        default[2] = 0
        self.observer_center = check_parameter_in_config(self.gasspy_config, "pov_center", observer_center, default)
        

        # We now have a positon for the observer and a center point, we need to figure out the rotation matrix
        # such that long = lat = 0 at self.pov_center in the reference frame of the observer
        
        # In the observer coordinate frame
        pov_center_o = self.pov_center - self.observer_center
        # determine quaternions:
        # xhat     =                 [1, 0, 0]
        # new_xhat =                 [pxo, pyo, pzo]
        # quat_n = xhat x new_xhat=  [0*pyo - 0*pxo, ]
        quat_n = np.array([0, -pov_center_o[2], pov_center_o[1]])#*np.sign(pov_center_o[2])
        quat_n_mag = np.sqrt(np.sum(np.square(quat_n))) # _zhat = [0*pzo - 1*pyo, 1*pxo - 0*pzo]
        if quat_n_mag > 0:
            quat_n = quat_n/quat_n_mag
        quat_theta = np.arctan2(quat_n_mag, pov_center_o[0])
        quats = np.append(np.sin(quat_theta/2)*quat_n, np.array([np.cos(quat_theta/2)]))
        quats = quats/np.sqrt(np.sum(np.square(quats)))


        self.rotation_matrix = R.from_quat(np.array(quats)).as_matrix()
        # If there is no change in x and y, quat_n_mag is zero and the problem becomes degerate. 
        # make sure that the direction of z is respected
        if quat_n_mag == 0:
            if np.sign(pov_center_o[0]) < 0:
                self.rotation_matrix[0][0] =- self.rotation_matrix[0][0]
                self.rotation_matrix[1][1] =  self.rotation_matrix[1][1]
                self.rotation_matrix[2][2] =- self.rotation_matrix[2][2]

        self.xps = -1
        self.yps = -1

        
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
        global_rayids = global_rays.append(self.Npixels)

        # Set the observation plane definitions
        # NOTE: healpix pixels are defined by one number. we take xp to be the ipix and yp to be 0
        global_rays.set_field("xp", cupy.array(self.ipix_lmin, dtype = ray_dtypes["xp"]), index = global_rayids)
        global_rays.set_field("yp", cupy.zeros(self.Npixels, dtype = ray_dtypes["xp"]), index = global_rayids)

        #All rays start from one point
        global_rays.set_field("xi", cupy.full(self.Npixels, self.observer_center[0], dtype = ray_dtypes["xi"]), index = global_rayids)
        global_rays.set_field("yi", cupy.full(self.Npixels, self.observer_center[1], dtype = ray_dtypes["yi"]), index = global_rayids)
        global_rays.set_field("zi", cupy.full(self.Npixels, self.observer_center[2], dtype = ray_dtypes["zi"]), index = global_rayids)

        # The directions of each ray needs to be rotated 
        for i, raydir in enumerate(["raydir_x", "raydir_y", "raydir_z"]):
            global_rays.set_field(raydir, cupy.array(
                                        self.raydir[:,0]*float(self.rotation_matrix[i][0]) +
                                        self.raydir[:,1]*float(self.rotation_matrix[i][1]) +
                                        self.raydir[:,2]*float(self.rotation_matrix[i][2])                                        
                                        ,dtype = ray_dtypes[raydir]), index = global_rayids)

        # set the refinement level of the rays
        global_rays.set_field("ray_lrefine", cupy.array(self.ray_lrefine_min), index=global_rayids)
        # set the different ID numbers relevant to the rays

        # IDs of the parents and corresponding split events, set to null values
        for i, id in enumerate(["pid", "pevid", "cevid"]):
            global_rays.set_field(id, cupy.full(self.Npixels, ray_defaults[id], dtype = ray_dtypes[id]), index=global_rayids)
        # ID of the branch, which, since this is the first ray of the branch, is the same as the ID of the ray
        global_rays.set_field("aid", cupy.arange(self.Npixels, dtype = ray_dtypes["aid"]), index= global_rayids)

        # Initialize the trace status of the rays
        global_rays.set_field("trace_status", 0, index=global_rayids)
        
        # Initialize the amr refinement of the rays
        global_rays.set_field("amr_lrefine", ray_defaults["amr_lrefine"], index  = global_rayids)

        # Initialize the fractional area of the rays
        global_rays.set_field("ray_fractional_area", self.get_ray_area_fraction(global_rays), index = global_rayids)            
        return global_rays


    def create_child_rays(self, parent_rays):
        """
            Creates children rays for a set of rays that are to be split
        """
        # How many children
        Nchild = 4*len(parent_rays["xp"])
        
        # Fields that are set here
        fields_new = ["xp", "yp", 
                      "xi", "yi", "zi",
                      "ray_fractional_area"]

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
        self.set_child_pixels(child_rays, parent_rays)
        self.set_child_positions(child_rays, parent_rays)
        return

    def set_child_pixels(self, child_rays, parent_rays):
        """
            Calculate the xp and yp of the children given the parents
        """
        
        # allocate the arrays
        xp_new = cupy.zeros((len(parent_rays["xp"]),4), dtype = ray_dtypes["xp"])
        yp_new = cupy.zeros((len(parent_rays["xp"]),4), dtype = ray_dtypes["yp"])



        # split the pixel into four
        # index at level i+1 ipix_{i+1} = 4*ipix_{i} + [0,1,2,3]
        xp_new[:,0] = parent_rays["xp"]*4 
        xp_new[:,1] = parent_rays["xp"]*4 + 1
        xp_new[:,2] = parent_rays["xp"]*4 + 2
        xp_new[:,3] = parent_rays["xp"]*4 + 3


        
        # and set the pixel positions in the child dataframe
        child_rays["xp"] = xp_new.ravel()
        child_rays["yp"] = yp_new.ravel()

        return

    def set_child_positions(self, child_rays, parent_rays):
        """
            Calculate the current positions of the children, given the parents
        """
 
        # First we get the new directions given the child rays
        # Preallocate the array of directions
        raydirs = np.zeros((len(child_rays["xp"]),3))
        # get the new refinement level of the rays and their ipixes
        lrefine = cupy.repeat(parent_rays["ray_lrefine"],4) + 1
        ipixs = child_rays["xp"].astype(cupy.int64)
        unique_lrefs = cupy.unique(lrefine)

        # Get the ray directions
        for lref in unique_lrefs.get():
            idxs = cupy.where(lrefine == lref)[0]
            idxs_cpu = idxs.get()
            raydirs[idxs_cpu, 0], raydirs[idxs_cpu, 1], raydirs[idxs_cpu, 2] =  hpix.pix2vec(int(2**(lref-1)), ipixs[idxs].get(), nest = True)


        # Rotate the directions to the desired orientation
        for i, raydir in enumerate(["raydir_x", "raydir_y", "raydir_z"]):
            child_rays[raydir] = cupy.array(
                                        raydirs[:,0]*float(self.rotation_matrix[i][0]) +
                                        raydirs[:,1]*float(self.rotation_matrix[i][1]) +
                                        raydirs[:,2]*float(self.rotation_matrix[i][2])                                        
                                        ,dtype = ray_dtypes[raydir])

        # Next figure out how far away the parent ray went
        parent_total_pathlength = cupy.sqrt(cupy.square(parent_rays["xi"] - self.observer_center[0]) + 
                                            cupy.square(parent_rays["yi"] - self.observer_center[1]) +
                                            cupy.square(parent_rays["zi"] - self.observer_center[2]))

        # Copy out to each child
        child_pathlength = cupy.repeat(parent_total_pathlength, 4)

        # Add the pathlength to the original position
        xi_new = cupy.array(self.observer_center[0] + child_rays["raydir_x"] * child_pathlength)
        yi_new = cupy.array(self.observer_center[1] + child_rays["raydir_y"] * child_pathlength)
        zi_new = cupy.array(self.observer_center[2] + child_rays["raydir_z"] * child_pathlength)

        # set the positions in the child dataframe
        child_rays["xi"] = xi_new
        child_rays["yi"] = yi_new
        child_rays["zi"] = zi_new

        return

    def set_ray_area(self, ray_struct, back_half = False):
        """
            Sets the local area of the rays solid angle 
        """

        # We have a list of areas saved from before, multiply this with the total pathlength squared 
        # TODO: add a check to make sure we dont exceed the array
        if back_half:
            ray_struct.set_field("ray_area", self.pixel_area[ray_struct.get_field("ray_lrefine") - self.ray_lrefine_min] * (cupy.square((ray_struct.get_field("xi") - 0.5*ray_struct.get_field("pathlength")*ray_struct.get_field("raydir_x")) - self.observer_center[0]) + 
                                                                                                                            cupy.square((ray_struct.get_field("yi") - 0.5*ray_struct.get_field("pathlength")*ray_struct.get_field("raydir_y")) - self.observer_center[1]) +
                                                                                                                            cupy.square((ray_struct.get_field("zi") - 0.5*ray_struct.get_field("pathlength")*ray_struct.get_field("raydir_z")) - self.observer_center[2])))
        else:
            ray_struct.set_field("ray_area", self.pixel_area[ray_struct.get_field("ray_lrefine") - self.ray_lrefine_min] * (cupy.square(ray_struct.get_field("xi") - self.observer_center[0]) + 
                                                                                                                            cupy.square(ray_struct.get_field("yi") - self.observer_center[1]) +
                                                                                                                            cupy.square(ray_struct.get_field("zi") - self.observer_center[2])))
        

        return
    def update_ray_area(self, ray_struct, back_half = False):
        """
            In the case of changing area (non paralell rays)
            have a method to update
        """
        # Just call original....
        self.set_ray_area(ray_struct, back_half = back_half)
        return

    def get_ray_area_fraction(self, ray_struct, index = None):
        if index is not None:
            return self.pixel_area[ray_struct.get_field("ray_lrefine", index = index) - self.ray_lrefine_min]/4*np.pi
        else:
            return self.pixel_area[ray_struct.get_field("ray_lrefine") - self.ray_lrefine_min]/4*np.pi


    def get_pixel_solid_angle(self, ray_struct : base_ray_class, index = None, numlib = cupy, back_half : bool = False) -> np.ndarray:
        # Get the distance the ray has travelled
        distance = numlib.sqrt(numlib.square(self.observer_center[0]-ray_struct.get_field("xi", index = index))+
                               numlib.square(self.observer_center[1]-ray_struct.get_field("yi", index = index))+
                               numlib.square(self.observer_center[2]-ray_struct.get_field("zi", index = index))) 

        if back_half:
            distance -= ray_struct.get_field("pathlength", index = index)*0.5
        solid_angle =1/numlib.square(distance)
        if self.ignore_within_distance > 0:
            ignore = cupy.where(distance<self.ignore_within_distance/self.gasspy_config["sim_unit_length"])
            solid_angle[ignore] = 0
        return solid_angle

