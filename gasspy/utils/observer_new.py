import numpy as np
import cupy
from scipy.spatial.transform import Rotation as R, rotation
class observer_plane_class:
    def __init__(self, sim_data, Nxp = None, Nyp = None, Nzp=None, ray_substep_parameters=None, scale_l_cgs = None, planeDefinitionMethod = None, **kwargs):
        
        self.pitch = 0 
        self.yaw   = 0
        self.roll  = 0

        # Note : not coded, but sim_data origin should default cuberoot(ncells)/2 if not defined by user
        self.rot_origin_simfrm = sim_data.origin
        self.xp0_s =   0.0
        self.yp0_s =   0.0
        self.zp0_s =   0.0
        

        self.__dict__.update(kwargs)

        self.rotation_matrix = R.from_rotvec(np.array([self.pitch, self.yaw, self.roll])*np.pi/180).as_matrix()

        rotation_matrix = cupy.array(self.rotation_matrix)

        # Normal vectors of a plane rotated by pitch yaw and roll.
        # These will be used every step, so calculate once and save.

        self.rotation_matrix_simtoobs = R.from_rotvec(np.array([-self.pitch, -self.yaw, -self.roll])*np.pi/180).as_matrix()
        rotation_matrix_simtoobs = cupy.array(self.rotation_matrix_simtoobs)
        
        #calculate the projected size of the simulation due to rotation
        # corners are 0 - origin, and box size - origin,
        sim_corners = [[x,y,z] for x in [-self.rot_origin_simfrm[0],sim_data.Ncells[0]-self.rot_origin_simfrm[0]]\
            for y in [-self.rot_origin_simfrm[1],sim_data.Ncells[1]-self.rot_origin_simfrm[1]]\
            for z in [-self.rot_origin_simfrm[2],sim_data.Ncells[2]-self.rot_origin_simfrm[2]]]
        
        for i, xyz in enumerate(sim_corners):
            xsim_simfrm, ysim_simfrm, zsim_simfrm = xyz
            # calculate the full rotated vector in each axis for each corner.
            corner_xsim_obsfrm = xsim_simfrm * rotation_matrix_simtoobs[0][0] + ysim_simfrm * rotation_matrix_simtoobs[0][1] + zsim_simfrm * rotation_matrix_simtoobs[0][2]
            corner_ysim_obsfrm = xsim_simfrm * rotation_matrix_simtoobs[1][0] + ysim_simfrm * rotation_matrix_simtoobs[1][1] + zsim_simfrm * rotation_matrix_simtoobs[1][2]
            corner_zsim_obsfrm = xsim_simfrm * rotation_matrix_simtoobs[2][0] + ysim_simfrm * rotation_matrix_simtoobs[2][1] + zsim_simfrm * rotation_matrix_simtoobs[2][2]
            if i == 0:
                self.xsim_obsfrm_min = corner_xsim_obsfrm
                self.xsim_obsfrm_max = corner_xsim_obsfrm
                self.ysim_obsfrm_min = corner_ysim_obsfrm
                self.ysim_obsfrm_max = corner_ysim_obsfrm
                self.zsim_obsfrm_min = corner_zsim_obsfrm
                self.zsim_obsfrm_max = corner_zsim_obsfrm
            else:
                # Compare value for each axis with the min and maximum, and adjust accordingly
                self.xsim_obsfrm_min = min(self.xsim_obsfrm_min, corner_xsim_obsfrm)
                self.xsim_obsfrm_max = max(self.xsim_obsfrm_max, corner_xsim_obsfrm)
                self.ysim_obsfrm_min = min(self.ysim_obsfrm_min, corner_ysim_obsfrm)
                self.ysim_obsfrm_max = max(self.ysim_obsfrm_max, corner_ysim_obsfrm)
                self.zsim_obsfrm_min = min(self.zsim_obsfrm_min, corner_zsim_obsfrm)
                self.zsim_obsfrm_max = max(self.zsim_obsfrm_max, corner_zsim_obsfrm)

        # Xp and Yp length is maximum projected length of the simulation as seen by the detector plane
        # Zp length is the length of the simulation domain projected along the normal of the detector
        self.projected_Xp_length = (self.xsim_obsfrm_max - self.xsim_obsfrm_min)
        self.projected_Yp_length = (self.ysim_obsfrm_max - self.ysim_obsfrm_min)
        self.projected_Zp_length = (self.zsim_obsfrm_max - self.zsim_obsfrm_min)


        # We are now going to define any offsets from the zero-zero corner in the pixel array, in the simulation frame.
        # This will give the position of zero pixel 
        # the position on xp = 0 yp = 0 zp = 0 with respect to the rotation origin
        # By default we are not rotated so x and y are at the midplane
        self.xp0_r = self.xp0_s - self.rot_origin_simfrm[0]
        self.yp0_r = self.yp0_s - self.rot_origin_simfrm[1]
        # to stay inside the box, the minium distance is 0.5*sqrt(Nx**2+Ny**2+Nz**2) away from origin
        self.zp0_r = self.zp0_s - self.rot_origin_simfrm[2]


        # if not defined assume that the simulation data is cubic, and take the plot grid to use the same dimensions
        if Nxp is not None:
            self.Nxp = Nxp
        elif "Nxp" in sim_data.config_yaml:
            self.Nxp = sim_data.config_yaml["Nxp"]
        else:
            self.Nxp = sim_data.Ncells[0]
        if self.Nxp == "auto":
            self.Nxp = int(self.projected_X_length)

        if Nyp is not None:
            self.Nyp = Nyp
        elif "Nyp" in sim_data.config_yaml:
            self.Nyp = sim_data.config_yaml["Nyp"]
        else:
            self.Nyp = sim_data.Ncells[1]
        if self.Nyp == "auto":
            self.Nyp = int(self.projected_Y_length)

        # This is immutable. Never shall xps and yps change. They are pixel indicies for a detector
        if planeDefinitionMethod is None :
            self.xps = np.arange(0, self.Nxp)
            self.yps = np.arange(0, self.Nyp)

            #Introduce scale factor here which is a scaled normal vector to change detector size. x, y, z are in observer frame,  i, j, k in simulation frame
            self.nx_i = float(rotation_matrix[0][0])
            self.nx_j = float(rotation_matrix[1][0])
            self.nx_k = float(rotation_matrix[2][0])
            self.ny_i = float(rotation_matrix[0][1])
            self.ny_j = float(rotation_matrix[1][1])
            self.ny_k = float(rotation_matrix[2][1])
            self.nz_i = float(rotation_matrix[0][2])
            self.nz_j = float(rotation_matrix[1][2])
            self.nz_k = float(rotation_matrix[2][2])

        else:
            self.xps, self.yps = planeDefinitionMethod(self.Nxp, self.Nyp)

        pass