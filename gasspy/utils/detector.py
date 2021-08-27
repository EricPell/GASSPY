import cupy
from scipy.spatial.transform import Rotation as R, rotation
import numpy as np

class detector(object):
    def __init__(self, sim_data, observer, ray_parameters_df):
        #This is the pitch, yaw and roll converted into a normal vector at the origin of the detector plane, at the base of the simulation, with Z up.
        if type(observer.rotation_matrix) is not cupy._core.core.ndarray:
            #If matrix is in system memory, move it to the GPU
            rotation_matrix = cupy.array(observer.rotation_matrix)

        # Normal vectors of a plane rotated by pitch yaw and roll.
        # These will be used every step, so calculate once and save.
        self.nx_i = float(rotation_matrix[0][0])
        self.nx_j = float(rotation_matrix[1][0])
        self.nx_k = float(rotation_matrix[2][0])
        self.ny_i = float(rotation_matrix[0][1])
        self.ny_j = float(rotation_matrix[1][1])
        self.ny_k = float(rotation_matrix[2][1])
        self.nz_i = float(rotation_matrix[0][2])
        self.nz_j = float(rotation_matrix[1][2])
        self.nz_k = float(rotation_matrix[2][2])

        #For example: Take a simuation coordinate x_simframe, y_simframe, z_simframe.
        # If you would like to move along the normal of the detector in the Z_obsframe axis by 
        # dZ_obsframe, at a detector pixel X,Y in the observer frame.
        # Lot's of repeat operations can be saved, but using X0 and Y0 = 0
        # x = x_simframe + self.nz_i * dZ_obsframe + self.nx_i * (X-X0) + self.ny_i * (Y-Y0)
        # y = y_simframe + self.nz_j * dZ_obsframe + self.nx_j * (X-X0) + self.ny_j * (Y-Y0)
        # z = z_simframe + self.nz_k * dZ_obsframe + self.nx_k * (X-X0) + self.ny_k * (Y-Y0)

        # If you would like to move along the Z_obsframe axis by dZ_obsframe you would do the following
        # x = x_simframe + self.nz_i * dZ
        # y = y_simframe + self.nz_j * dZ
        # z = z_simframe + self.nz_k * dZ

        # 

        self.rotation_matrix_simtoobs = R.from_rotvec(np.array([-observer.pitch, -observer.yaw, -observer.roll])*np.pi/180).as_matrix()
        rotation_matrix_simtoobs = cupy.array(self.rotation_matrix_simtoobs)
        
        #calculate the projected size of the simulation due to rotation
        # corners are 0 - origin, and box size - origin,
        sim_corners = [[x,y,z] for x in [-observer.rot_origin_simfrm[0],sim_data.Ncells[0]-observer.rot_origin_simfrm[0]]\
            for y in [-observer.rot_origin_simfrm[1],sim_data.Ncells[1]-observer.rot_origin_simfrm[1]]\
            for z in [-observer.rot_origin_simfrm[2],sim_data.Ncells[2]-observer.rot_origin_simfrm[2]]]
        
        for i, xyz in enumerate(sim_corners):
            xsim_simfrm, ysim_simfrm, zsim_simfrm = xyz
            # calculate the full rotated vector in each axis for each corner.
            corner_xsim_detfrm = xsim_simfrm * rotation_matrix_simtoobs[0][0] + ysim_simfrm * rotation_matrix_simtoobs[0][1] + zsim_simfrm * rotation_matrix_simtoobs[0][2]
            corner_ysim_detfrm = xsim_simfrm * rotation_matrix_simtoobs[1][0] + ysim_simfrm * rotation_matrix_simtoobs[1][1] + zsim_simfrm * rotation_matrix_simtoobs[1][2]
            corner_zsim_detfrm = xsim_simfrm * rotation_matrix_simtoobs[2][0] + ysim_simfrm * rotation_matrix_simtoobs[2][1] + zsim_simfrm * rotation_matrix_simtoobs[2][2]
            if i == 0:
                self.xsim_detfrm_min = corner_xsim_detfrm
                self.xsim_detfrm_max = corner_xsim_detfrm
                self.ysim_detfrm_min = corner_ysim_detfrm
                self.ysim_detfrm_max = corner_ysim_detfrm
                self.zsim_detfrm_min = corner_zsim_detfrm
                self.zsim_detfrm_max = corner_zsim_detfrm
            else:
                # Compare value for each axis with the min and maximum, and adjust accordingly
                self.xsim_detfrm_min = min(self.xsim_detfrm_min, corner_xsim_detfrm)
                self.xsim_detfrm_max = max(self.xsim_detfrm_max, corner_xsim_detfrm)
                self.ysim_detfrm_min = min(self.ysim_detfrm_min, corner_ysim_detfrm)
                self.ysim_detfrm_max = max(self.ysim_detfrm_max, corner_ysim_detfrm)
                self.zsim_detfrm_min = min(self.zsim_detfrm_min, corner_zsim_detfrm)
                self.zsim_detfrm_max = max(self.zsim_detfrm_max, corner_zsim_detfrm)

        # Xp and Yp length is maximum projected length of the simulation as seen by the detector plane
        # Zp length is the length of the simulation domain projected along the normal of the detector
        self.projected_Xp_length = (self.xsim_detfrm_max - self.xsim_detfrm_min)
        self.projected_Yp_length = (self.ysim_detfrm_max - self.ysim_detfrm_min)
        self.projected_Zp_length = (self.zsim_detfrm_max - self.zsim_detfrm_min)

        xp_obsfrm = cupy.linspace(-ray_parameters_df["Nxmax"][0]/2.,ray_parameters_df["Nxmax"][0]/2.)
        yp_obsfrm = cupy.linspace(-ray_parameters_df["Nymax"][0]/2.,ray_parameters_df["Nymax"][0]/2.)
        zp_obsfrm = cupy.linspace(0,(ray_parameters_df["Nzmax"]/ray_parameters_df["dZslab"])[0], ray_parameters_df["z_subsamples"])
        self.xp_obsfr, self.yp_obsfrm, self.zp_obsfrm = cupy.meshgrid(xp_obsfrm[:,0],yp_obsfrm[:,0],zp_obsfrm[:,0])
        pass
