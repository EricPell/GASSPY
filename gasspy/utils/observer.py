import numpy as np
from scipy.spatial.transform import Rotation as R
class observer_plane_class:
    def __init__(self, sim_data, Nxp = None, Nyp = None, z_subsamples = 3, scale_l_cgs = None, planeDefinitionMethod = None, **kwargs):
        

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

        # This is immutable. Never shall xps and yps change. They are pixel indicies for a detector
        if planeDefinitionMethod is None :
            self.xps = np.arange(0, self.Nxp)
            self.yps = np.arange(0, self.Nyp)
        else:
            self.xps, self.yps = planeDefinitionMethod(self.Nxp, self.Nyp)

        self.NumZ = np.sqrt(np.sum(np.square(sim_data.Ncells)))
        self.z_subsamples = z_subsamples
        self.zps = np.linspace(0, self.NumZ, int(self.NumZ*self.z_subsamples))

        
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
        self.zp0_r = self.zp0_s - self.rot_origin[1]

        self.__dict__.update(kwargs)


        self.rotation_matrix = R.from_rotvec(np.array([self.pitch, self.yaw, self.roll])*np.pi/180).as_matrix()

