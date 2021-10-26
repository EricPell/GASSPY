import cudf
import cupy
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
from gasspy.utils.savename import get_filename 


def __raytrace_kernel__(xi, yi, zi, pathlength, index1D, raydir, Nmax):
    # check if inside the box, otherwise return 0
    for i, (x,y,z) in enumerate(zip(xi, yi, zi)):
        # if we know we are outside the box domain set index1D to NULL value (TODO: fix this value in parameters)
        if x < 0:
            index1D[i] = 0
        elif y < 0:
            index1D[i] = 0
        elif z < 0:
            index1D[i] = 0
        elif x >= Nmax[0]:
            index1D[i] = 0
        elif y >= Nmax[1]:
            index1D[i] = 0
        elif z >= Nmax[2]:
            index1D[i] = 0
        else:
            index1D[i] = int(z) + Nmax[2]*int(y) + Nmax[1]*Nmax[2]*int(x)

        # init to unreasonably high number
        pathlength[i] = 1e30
        mindir = -1
        # check for closest distance to cell boundary by looking for the closest int in all directions
        # a thousand of a cell width is added as padding such that the math is (almost) always correct
        # NOTE: this could be wrong if a ray is very close to an interface. So depending on the angle of raydir
        # With respect to the cells, errors can occur

        # in x
        if(raydir[0] > 0):
            newpath = (math.floor(x) + 1 - x)/raydir[0]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 0
        elif(raydir[0] < 0):
            newpath = (math.ceil(x) - 1 - x)/raydir[0]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 0
        
        # in y
        if(raydir[1] > 0):
            newpath = (math.floor(y) + 1 - y)/raydir[1]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 1
        elif(raydir[1] < 0):
            newpath = (math.ceil(y) - 1 - y)/raydir[1]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 1

        # in z
        if(raydir[2] > 0):
            newpath = (math.floor(z) + 1 - z)/raydir[2]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 2
        elif(raydir[2] < 0):
            newpath = (math.ceil(z) - 1 - z)/raydir[2]
            if(pathlength[i] > newpath):
                pathlength[i] = newpath
                mindir = 2

        if(mindir == 0):
            # move to next int
            
            if(raydir[0] > 0):
                xi[i] = math.floor(x) + 1
            else:
                xi[i] = math.ceil(x) - 1
            yi[i] = yi[i] + pathlength[i]*raydir[1]
            zi[i] = zi[i] + pathlength[i]*raydir[2]
            continue 

        if(mindir == 1):
            # move to next int
            if(raydir[1] > 0):
                yi[i] = math.floor(y) + 1
            else:
                yi[i] = math.ceil(y) - 1
            xi[i] = xi[i] + pathlength[i]*raydir[0]
            zi[i] = zi[i] + pathlength[i]*raydir[2]
            continue

        if(mindir == 2):
            # move to next int
            if(raydir[2] > 0):
                zi[i] = math.floor(z) + 1
            else:
                zi[i] = math.ceil(z) - 1
            xi[i] = xi[i] + pathlength[i]*raydir[0]
            yi[i] = yi[i] + pathlength[i]*raydir[1]
            continue


class raytracer_class:
    def __init__(self, sim_data, obs_plane = None, line_lables = None, savefiles = True, NcellBuff  = 64):
        self.set_new_sim_data(sim_data, line_lables)
        """
            Input:
                sim_data, required - simulation_data_class object containing the needed data from the simulation
                obs_plane          - initial obs_plane definition, can be set later 
                line_labels        - Names of the wanted lines
                savefiles          - Boolean flag if user wants to save the resulting fluxes or not as fits and npys
                NcellBuff          - integer describing the number of cells a ray will hold in the buffer before
                                     before it calculates the cumulative emissions and opacities
        """
        if obs_plane is not None:
            self.set_obsplane(obs_plane)
        else:
            self.set_empty() 
    
        self.savefiles = savefiles
        self.NcellBuff = NcellBuff

    """
        Externally used methods    
    """

    def raytrace_run(self, saveprefix = None):
        """
            Main method to run the ray tracing     
        """

        assert (self.xps is not None) and (self.yps is not None), "ERROR: and observer needs to be set before you can run ray tracing"
        assert self.rays is not None, "ERROR: ray's have not been generated. Use defined functions" 

        #reset fluxes
        for line in self.line_lables :
            self.fluxes[line] = 0
        # start by moving all cells outside of box to the first intersecting cell
        self.move_to_first_intersection()
        
        # Do a soft pruning outside of the box
        self.prune_outside_sim(soft = True)
        
        
        self.ibuff = 0
        self.alloc_buffer()
         
        # transport rays until all rays are outside the box
        i = 0
        while(len(self.rays) > 0):
            # transport the rays through the current cell
            self.raytrace_onestep()
            # advance buffer
            self.ibuff += 1
            i+=1
            if self.ibuff == self.NcellBuff:
                # if we are out of space, gather cells subphysics and reset buffer
                self.get_subphysics_cells()
                self.prune_outside_sim()
                self.alloc_buffer()
                # reset buffer index
                self.ibuff = 0

        # at the end, if there are still things in the buffer, save them
        if self.ibuff > 0:
            self.get_subphysics_cells()
        
        # save fluxes to the files
        if self.savefiles:
            self.save_lines_fluxes(saveprefix=saveprefix)
        
    
    def set_new_sim_data(self, sim_data, line_lables = None):
        """
            Method to take an observer plane set internal values 
        """
        self.line_lables = line_lables
        if self.line_lables is None:
            """Try and read from simdata config"""
            self.line_lables = sim_data.config_yaml["line_labels"]

        self.subphys_id_df = cudf.DataFrame(sim_data.get_subcell_model_id().ravel())

        #
        self.avg_em_df = cudf.DataFrame(sim_data.subcell_models.DF_from_dict(self.line_lables))
        #avg_ab_df   = cudf.DataFrame(sim_data.subcell_models.avg_ab(line_lables))
        
        # save local variables
        self.Nmax = cupy.array(sim_data.Ncells)
        
        # query string used for dropping rays outside bounds 
        self.inside_query_string  = "(xi >= 0 and xi <= {0} and yi >= 0 and yi <= {1} and zi >= 0 and zi <= {2})".format(int(self.Nmax[0]),int(self.Nmax[1]), int(self.Nmax[2]))
        self.inside_soft_query_string  = "(xi > -1 and xi < {0} and yi > -1 and yi < {1} and zi > -1 and zi < {2})".format(int(self.Nmax[0]+1),int(self.Nmax[1]+1), int(self.Nmax[2]+1))
        # save reference to sim_data
        self.sim_data = sim_data  

    def set_obsplane(self, obs_plane):
        """
            Method to take an observer plane set internal values 
        """
        self.xps = obs_plane.xps
        self.yps = obs_plane.xps

        self.xp, self.yp = np.meshgrid(self.xps, self.yps)
        self.xp = self.xp.ravel()
        self.yp = self.yp.ravel()
        

        # initialize rays data frame, pre allocate as much as possible (possible extensions to rayspliting might brake this)
        self.rays = cudf.DataFrame({"xp" : self.xp, "yp" : self.yp, 
                                    "xi" : np.zeros(self.xp.shape), "yi" : np.zeros(self.xp.shape), "zi" : np.zeros(self.xp.shape),
                                    "tmp_xi" : np.zeros(self.xp.shape), "tmp_yi" : np.zeros(self.xp.shape), "tmp_zi" : np.zeros(self.xp.shape),
                                    "pathlength": np.zeros(self.xp.shape), "index1D" : np.zeros(self.xp.shape, dtype = int), "ibuff" : np.zeros(self.xp.shape, dtype = int)})





        self.fluxes   = cudf.DataFrame({"xp": self.xp, "yp" : self.yp})
        #self.opacity = cudf.DataFrame({"xp": xp.ravel(), "yp" : yp.ravel()})
        for line_lable in self.line_lables :
            self.fluxes[line_lable] = cupy.zeros(self.xp.shape)
            #opacity[line_lable] = cupy.zeros(self.xp.shape)
        self.fluxes.set_index(["xp", "yp"], inplace = True) 
        #self.opacity.set_index(["xp", "yp"], inplace = True) 


        for i, xi in enumerate(["xi", "yi", "zi"]) :

            self.rays[xi] = cupy.full(len(self.rays), (self.rays["xp"].values + 0.5 + obs_plane.xp0_r ) * float(obs_plane.rotation_matrix[i][0]) + 
                                                 (self.rays["yp"].values + 0.5 + obs_plane.yp0_r ) * float(obs_plane.rotation_matrix[i][1]) +
                                                  obs_plane.zp0_r * float(obs_plane.rotation_matrix[i][2]) + obs_plane.rot_origin[i])
        # rotate direction
        self.raydir = cupy.zeros(3)
        for i in range(3):
            self.raydir[i] = (obs_plane.view_dir[0] * float(obs_plane.rotation_matrix[i][0]) +
                              obs_plane.view_dir[1] * float(obs_plane.rotation_matrix[i][1]) +
                              obs_plane.view_dir[2] * float(obs_plane.rotation_matrix[i][2]))

        # save reference to observer plane
        self.obs_plane =  obs_plane

    def update_obsplane(self, obs_plane, prefix = None):
        """
            Method to take an observer plane and updates internal values that needs to be updated
        """

        # if the xps and yps arrays have changed, we need to regenerate everything since the xp yp indices are everywhere
        if not (np.array_equal(obs_plane.xps, cupy.asnumpy(self.xps)) 
            and np.array_equal(obs_plane.yps, cupy.asnumpy(self.yps))):
            self.xps = cupy.array(obs_plane.xps)
            self.yps = cupy.array(obs_plane.xps)
            self.xp, self.yp = cupy.meshgrid(self.xps, self.yps)
            
            self.xp = self.xp.ravel()
            self.yp = self.yp.ravel()
            
            # regenerate ray data frame, and fluxes and opacities
            self.fluxes   = cudf.DataFrame({"xp": self.xp, "yp" : self.yp})
            #self.opacity = cudf.DataFrame({"xp": xp.ravel(), "yp" : yp.ravel()})
            
            for line_lable in self.line_lables :
                self.fluxes[line_lable] = cupy.zeros(self.xp.shape)
                #self.opacity[line_lable] = cupy.zeros(self.xp.shape)
            self.fluxes.set_index(["xp", "yp"], inplace = True) 
            #self.opacity.set_index(["xp", "yp"], inplace = True) 


        # TODO: explore if we can save an original reduxed frame of rays
        #if self.rays is not None:
        #    del(self.rays)
        self.rays = cudf.DataFrame({"xp" : self.xp, "yp" : self.yp, 
                                    "xi" : cupy.zeros(self.xp.shape), "yi" : cupy.zeros(self.xp.shape), "zi" : cupy.zeros(self.xp.shape),
                                    "tmp_xi" : cupy.zeros(self.xp.shape), "tmp_yi" : cupy.zeros(self.xp.shape), "tmp_zi" : cupy.zeros(self.xp.shape),
                                    "pathlength": cupy.zeros(self.xp.shape), "index1D" : cupy.zeros(self.xp.shape, dtype = int)})


        # we assume that everything else has been modified, so ray rotation + translation + first hits are recalculated
        for i, xi in enumerate(["xi", "yi", "zi"]) :

            self.rays[xi] = cupy.full(len(self.rays),  obs_plane.xp0_r * float(obs_plane.rotation_matrix[i][0]) +
                                                       obs_plane.yp0_r * float(obs_plane.rotation_matrix[i][1]) + 
                                                       obs_plane.zp0_r * float(obs_plane.rotation_matrix[i][2]) + obs_plane.rot_origin[i])

            self.rays[xi] += (self.rays["xp"] * float(obs_plane.rotation_matrix[i][0]) + 
                              self.rays["yp"] * float(obs_plane.rotation_matrix[i][1]) )

        # rotate direction
        self.raydir     = cupy.zeros(3)
        self.raydir_inv = cupy.zeros(3)
        for i in range(3):
            # self.raydir[i] = (obs_plane.view_dir[0] * float(obs_plane.rotation_matrix[i][0]) +
            #                   obs_plane.view_dir[1] * float(obs_plane.rotation_matrix[i][1]) +
            #                   obs_plane.view_dir[2] * float(obs_plane.rotation_matrix[i][2]))

            self.raydir[i] =  1.0 * float(obs_plane.rotation_matrix[i][2])
            
        self.raydir_inv = 1/self.raydir
        
        # save reference to observer plane
        self.obs_plane = obs_plane


    """
        Internally used methods    
    """

    def save_lines_fluxes(self, saveprefix = None):
        os.makedirs("%s/gasspy_output/"%(self.sim_data.datadir), exist_ok=True)
        for line in self.line_lables:
            flux_array = cupy.array(self.fluxes[line])
            flux_array = cupy.asnumpy(flux_array)
            fname = "%s/gasspy_output/%s.npy"%(self.sim_data.datadir, get_filename(line, self.sim_data, self.obs_plane, saveprefix = saveprefix))
            print("saving " + fname)
            np.save(fname, flux_array.reshape(self.obs_plane.Nxp, self.obs_plane.Nyp))
        del(flux_array)


    def set_empty(self):
        """
            Method to initialise all used quantities to None
        """
        self.xps = None
        self.yps = None
        self.rays = None
        self.pathlength = None
        self.raydir = None
        self.observer = None


    
    
    def move_to_first_intersection(self):
        """
            finds the closest intersection to the simulation cube for each ray outside the cube
        """
        # define origion and normal vector of each of the 6 planes of the data cube
        planes = cupy.array([[ float(self.Nmax[0]), 0, 0, 1, 0, 0],
                             [ 0, 0, 0, -1,  0, 0],
                             [ 0, float(self.Nmax[1]), 0, 0, 1, 0],
                             [ 0, 0, 0, 0, -1, 0],
                             [ 0, 0, float(self.Nmax[2]), 0, 0, 1],
                             [ 0, 0, 0, 0, 0., -1.]])

        min_pathlength = cupy.full(self.xp.shape, 1e30)
        for plane in planes:
            p0     = plane[:3]
            nplane = plane[3:]
            # if nplane*p0 = 0 , they never intersect, so skip
            align = cupy.sum(nplane*self.raydir)
            if align == 0:
                continue
            # find the pathlength to the points where the rays intersect the plane
            pathlength = cupy.abs(((p0[0] - self.rays["xi"].values) * nplane[0] + 
                                   (p0[1] - self.rays["yi"].values) * nplane[1] + 
                                   (p0[2] - self.rays["zi"].values) * nplane[2])/align)
            
            if cupy.sum(~cupy.isinf(pathlength)) == 0:
                print( "no intersect found with plane with non parallell normal vector", plane, align )
                sys.exit() 
            
            self.rays["tmp_xi"] = self.rays["xi"] + cudf.Series(self.raydir[0] * pathlength*1.001, index = self.rays.index)
            self.rays["tmp_yi"] = self.rays["yi"] + cudf.Series(self.raydir[1] * pathlength*1.001, index = self.rays.index)
            self.rays["tmp_zi"] = self.rays["zi"] + cudf.Series(self.raydir[2] * pathlength*1.001, index = self.rays.index)

            # identify all intersection outside of the simulation domain
            mask = ((self.rays["tmp_xi"].values < 0) | (self.rays["tmp_xi"].values >= self.Nmax[0]) |
                    (self.rays["tmp_yi"].values < 0) | (self.rays["tmp_yi"].values >= self.Nmax[1]) | 
                    (self.rays["tmp_zi"].values < 0) | (self.rays["tmp_zi"].values >= self.Nmax[2]))
            # set these to an unreasonable high number
            pathlength[mask] = 1e30
        
            # if pathlength to current plane is smaller than currently shortest path, replace them
            mask = pathlength < min_pathlength
            min_pathlength[mask] = pathlength[mask]
        
        # if rays not already in the box,  move rays. If the ray does not intersect the box, it will be put outside and pruned in later stages
        inbox = ((self.rays["xi"] >= 0) & (self.rays["xi"] <= int(self.Nmax[0])) &
                 (self.rays["yi"] >= 0) & (self.rays["yi"] <= int(self.Nmax[1])) &
                 (self.rays["zi"] >= 0) & (self.rays["zi"] <= int(self.Nmax[2])))

        # move rays outside of box. cudf where replaces where false     
        for i, ix in enumerate(["xi", "yi", "zi"]):
            self.rays[ix].where(inbox, self.rays[ix] + cudf.Series(self.raydir[i] * (min_pathlength + 0.001*cupy.sign(min_pathlength)), index = self.rays.index), inplace = True) # some padding to ensure cell boundarys are crossed
        
    def prune_outside_sim(self, soft = False):
        """
            Removes all rays that are outside the box 
        """
        #inbox =  ((self.rays["xi"] >= 0) | (self.rays["xi"] < int(self.Nmax[0])) |
        #          (self.rays["yi"] >= 0) | (self.rays["yi"] < int(self.Nmax[1])) |
        #          (self.rays["zi"] >= 0) | (self.rays["zi"] < int(self.Nmax[2])))
        

               
        #self.rays.drop(outbox.loc[outbox == True].index, inplace = True)
        if(soft) :
            self.rays = self.rays.query(self.inside_soft_query_string)
        else:
            self.rays = self.rays.query(self.inside_query_string)

        #self.tmp_3d_df.drop(outbox.loc[outbox == True].index, inplace = True)
        #del(outbox)
    def get_remaining(self):
        return len(self.rays.query(self.inside_query_string))

    def alloc_buffer(self):
        """
            Allocates buffer to store cells and their pathlength
            This allows for fewer calls where we add to the fluxes arrays, 
            so less time spent on finding the correct indices
        """
        # only use remaining rays to create new buffer
        self.buff_index      = self.rays.index.values
        self.buff_index1D    = cupy.zeros((len(self.rays),self.NcellBuff), dtype = int)
        self.buff_pathlength = cupy.zeros((len(self.rays),self.NcellBuff))
            
        # Dataframe to save physical values to
        self.rays.reset_index(inplace = True)
        self.ray_buff = cudf.DataFrame({"xp" : self.rays.xp.repeat(self.NcellBuff), "yp" : self.rays.yp.repeat(self.NcellBuff)})
        self.ray_buff["ibuff"] = cupy.mod(cupy.arange(0,len(self.ray_buff)), self.NcellBuff)
        self.ray_buff["pathlength"] = cupy.zeros(len(self.ray_buff))
        for line in self.line_lables :
            self.ray_buff[line] = 0.0
        # for opac in opac_labels:
        #   self.ray_buff[opac] = 0.0
        self.rays.set_index(["xp", "yp"], inplace = True)
        self.ray_buff.set_index(["xp", "yp","ibuff"], inplace = True)
    def reset_buffer(self):
        self.buff_pathlength[:,:] = 0
        self.buff_index1D[:,:] = 1
        self.ray_buff["pathlength"] = 0
        self.ray_buff[self.line_lables] = 0


    def raytrace_onestep(self):
        # find the next intersection to a cell boundary by finding the closest distance to an integer for all directions
        self.rays = self.rays.apply_rows(__raytrace_kernel__,
                incols = ["xi", "yi", "zi"],
                outcols = dict( pathlength = np.float64, index1D=np.int32),
                kwargs = dict(raydir = self.raydir, Nmax = self.Nmax))
        # store in buffers
        self.buff_index1D[:,self.ibuff]    = self.rays["index1D"].values[:]
        self.buff_pathlength[:,self.ibuff] = self.rays["pathlength"].values[:]

    
        
    def setBufferDF(self):
        """
            Gathers the data stored in the buff arrays. gets their subphysics data
        """
        self.ray_buff[self.line_lables] = cudf.DataFrame(self.avg_em_df.iloc[self.subphys_id_df.iloc[self.buff_index1D.ravel()].values].values, index = self.ray_buff.index)
        self.ray_buff["pathlength"] = self.buff_pathlength.ravel()

    def addToFlux(self):
        for line in self.line_lables:
            self.ray_buff[line] = self.ray_buff[line].mul(self.ray_buff["pathlength"])
        self.fluxes[self.line_lables] = self.fluxes[self.line_lables].add(self.ray_buff[self.line_lables].groupby([cudf.Grouper(level = 'xp'), cudf.Grouper(level = 'yp')]).sum(), fill_value = 0.0) # * self.opacity[line_labels].exp())

    def get_subphysics_cells(self):
        self.setBufferDF()
        self.addToFlux()


