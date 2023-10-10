"""
    Author: Loke Ohlin 
    Date: 05-2022
    Purpose: This class contains methods to load data from a simulation
    Usage: Copy this template into a separate directory and change the functions and attributes
           to satisfy the required functions to load raveled datasets


    NOTES:
        -If your simulation does not contain a required field natively, you must make sure that
         self.get_field(self, field)
         calls a function that calculates and returns it
        -File MUST be called simulation_reader.py and class Simulation_Reader for the flexible imports to work
"""
import numpy as np
import yaml
import os
import sys

from gasspy.shared_utils.functions import sorted_in1d
from astropy.io import fits
import astropy.units as apyu
import astropy.constants as apyc

class Simulation_Reader():
    #   
    #  This is a dictionary to convert a gasspy variable name to the equivalent one used in the simulation
    #  
    gasspy_to_sim_label = {
        # density, temperatue and amr level have different names, these need to be specified
        "density": "rho",
        "number_density": "number_density",
        "temperature": "T",
        "amr_lrefine": "amrlevel",
        "coordinate_x": "x",
        "coordinate_y": "y",
        "coordinate_z": "z",
        "veloctiy_x": "vx",
        "velocity_y": "vy",
        "velocity_z": "vz",
        # There is no dx for the simulation (its calculated from amr_lrefine) so can be excluded but MUST be defined as a function here
        "cell_size": "dx",
        # In this case the radiation fields has had their names changed
        "HII" : "NpFUV",
        "HeII" : "NpHII",
        "HeIII" : "NpHeII",
    }

    #
    # List of all fields natively contained by the simlation or calculated in this class
    #
    contained_fields = ["rho", "number_density", "T", "P", "amrlevel", "x", "y", "z", "vx", "vy", "vz", "NpFUV", "NpHII", "NpHeII", "NpHeIII", "Bx", "By", "Bz", "dx"]
    def __init__(self, gasspy_config, snapshot):
        """
            arguments:
                arg_dict: dictionary (a dictionary of arguments needed by the reader to load the simulation data)
        """

        self.simdir = snapshot["simdir"]
        self.gasspy_subdir = snapshot["gasspy_subdir"]
        assert "output_number" in snapshot.keys(), "Error: No output number selected!"
        self.output_number = int(snapshot["output_number"])
        if "save_raytrace_vars" in snapshot.keys():
            self.save_raytrace_vars = snapshot["save_raytrace_vars"]
        else:
            self.save_raytrace_vars = False
        if not os.path.exists(self.gasspy_subdir+"/cell_data"):
            os.makedirs(self.gasspy_subdir+"/cell_data/")
        self.__load_info_file__()
        self.Ncells = len(self.get_field("rho"))
        return

    ######
    #
    #   External functions that MUST exist with these EXACT calls
    #
    ######

    # general function to load and retrieve a field
    # - Must take string name for field
    # - Must return the ravelled list of the field for all cells
    def get_field(self, field):
        """
            Returns the simulation data of the provided field name 
            arguments:
                field: String (name of field either according to gasspy or the simulation)
            returns:
                field_data: array of dtype(field) (ravelled list of the field for all cells)
        """
        # Figure out if we either need to change name
        if field in self.gasspy_to_sim_label.keys():
            field = self.gasspy_to_sim_label[field]

        if not field in self.contained_fields:
            return self.load_new_field(field)

        if field == "dx":
            return self.__get_dx__()
        elif field in ["vx", "vy", "vz"]:
            fname = self.__get_filename__(field)
            return 1e5*fits.open(fname)[0].data.ravel()
        elif field == "velocity":
            return self.__get_velocity__()
        elif field in ["NpHeIII", "NpFUV", "NpHII", "NpHeII"]:
            return self.__get_fluxes__(field)
        elif field == "number_density":
            return self.get_number_density()

        else:
            fname = self.__get_filename__(field)

        assert os.path.exists(fname), "ERROR: File %s not found"%fname

        return fits.open(fname)[0].data.ravel()

    # Specific function to load the number density of hydrogen
    # - Cannot have any arguments
    # - Must return the ravelled list of the field for all cells
    def get_number_density(self):
        """
            Returns the number density of all cells
            arguments:

            returns:
                number_density: array of floats (ravelled list of the number density for all cells)
        """
        return self.get_field("rho")/(1.2 *apyc.m_p.cgs.value)

    # Specific fucntion to calculate or load the index1D of the cells
    # Which in is the index of each cell in a raveled array sorted by x,y,z assuming that the entire simulation is on its refinement level
    # - Cannot have any arguments
    # - Must return the ravelled list for all cells
    def get_index1D(self):

        # In this case, we might want to save the derived variable.
        # We can implement a load, and create a file on first use
        # To follow convention, this should be put in gasspy_subdir/cell_data
        index1D_file = self.gasspy_subdir + "/cell_data/%05d_index1D.npy"%self.output_number
        if os.path.exists(index1D_file) and self.save_raytrace_vars:
            return np.load(index1D_file)
        # Otherwise we need to calculate the index
        index1D = self.__calc_index1D__()
        if self.save_raytrace_vars:
            np.save(index1D_file, index1D)
        return index1D

    # Specific function to calculate or load the neighbor list for all cells as an (Ncell, Nneigh) int array
    # 
    # - Cannot have any arguments
    # - Must return the (Ncell, Nneigh) int array

    def get_cell_neighbors(self):
        """
            Returns the neighbors of all cells
            arguments:

            returns:
                cell_neighbors: array of ints
        """
        # In this case, we might want to save the derived variable.
        # We can implement a load, and create a file on first use
        # To follow convention, this should be put in gasspy_subdir/cell_data
        neighbor_file = self.gasspy_subdir + "/cell_data/%05d_cell_neighbors.npy"%self.output_number
        if os.path.exists(neighbor_file) and self.save_raytrace_vars:
            return np.load(neighbor_file)
        cell_neighbors = np.zeros((self.Ncells, 24), dtype = int)

        # Grab cell variables that we need
        dxs = self.__get_dx__()
        amr_lrefine = self.get_field("amr_lrefine")
        index1D = self.__calc_index1D__()
        cell_index = np.arange(self.Ncells)
        xs = self.__get_x__("x")
        ys = self.__get_x__("y")
        zs = self.__get_x__("z")
        # Initialize per refinement level lists
        cell_index_lref = []
        index1D_lref = []

        amr_lrefine_min = self.sim_info["minref"]
        amr_lrefine_max = self.sim_info["maxref"]
        for lref in range(amr_lrefine_min, amr_lrefine_max + 1):
            at_lref = amr_lrefine == lref
            idx_sort = index1D[at_lref].argsort()
            # Add arrays sorted in index1D
            index1D_lref.append(index1D[at_lref][idx_sort].astype(int))
            cell_index_lref.append(cell_index[at_lref][idx_sort].astype(int))

        # Loop over neighbor positions and find the corresponding cells
        idir = 0
        for ix in [-0.6, 0.6]:
            for iy in [-0.25, 0.25]:
                for iz in [-0.25, 0.25]:
                    cell_neighbors[:,idir] = self.__get_cell_index_from_pos__(xs+ix*dxs,ys+iy*dxs,zs+iz*dxs, index1D_lref, cell_index_lref)
                    idir+=1
        for iy in [-0.6, 0.6]:
            for ix in [-0.25, 0.25]:
                for iz in [-0.25, 0.25]:
                    cell_neighbors[:,idir] = self.__get_cell_index_from_pos__(xs+ix*dxs,ys+iy*dxs,zs+iz*dxs, index1D_lref, cell_index_lref)
                    idir+=1
        for iz in [-0.6, 0.6]:
            for ix in [-0.25, 0.25]:
                for iy in [-0.25, 0.25]:
                    cell_neighbors[:,idir] = self.__get_cell_index_from_pos__(xs+ix*dxs,ys+iy*dxs,zs+iz*dxs, index1D_lref, cell_index_lref)
                    idir+=1

        if self.save_raytrace_vars:
            np.save(neighbor_file, cell_neighbors)
        return cell_neighbors
    

    def save_new_field(self, fieldname, data, dtype = None):
        if dtype is None:
            dtype = data.dtype
        field_file = self.gasspy_subdir + "/cell_data/%05d_%s.npy"%(self.output_number, fieldname)
        np.save(field_file, data.astype(dtype))
        return

    def load_new_field(self, fieldname):
        field_file = self.gasspy_subdir + "/cell_data/%05d_%s.npy"%(self.output_number, fieldname)
        return np.load(field_file)


    ######
    #
    #   Internal functions specific to this simulation
    #   change to your hearts content
    #
    #######

    def __get_fluxes__(self, field):

        fname = self.__get_filename__(field)
        assert os.path.exists(fname), "ERROR: File %s not found"%fname

        return fits.open(fname)[0].data.ravel()/0.01
    # Method to load the info file of the snapshot
    def __load_info_file__(self):
        fname = self.simdir+"/infos/info_%05d.yaml"%self.output_number
        with open(fname, "rb") as f:
            self.sim_info = yaml.load(f, Loader=yaml.FullLoader)
        return

    def __get_filename__(self, field):
        return self.simdir + "/%s/celllist_%s_%05d.fits"%(field, field, self.output_number)

    # This simulation does not have the cell size as a field, so it needs to be calculated
    def __get_dx__(self):
        amr_lrefine = self.get_field("amr_lrefine")
        boxlen = self.sim_info["boxlen"]
        return boxlen/2**amr_lrefine.astype(np.float32)

    # Method to calculate index1D
    def __calc_index1D__(self, xs = None, ys = None, zs = None, amrlevel = None):
        if xs is None or ys is None or zs is None or amrlevel is None:
            xs = self.get_field("x")
            ys = self.get_field("y")
            zs = self.get_field("z")
            amrlevel = self.get_field("amrlevel").astype(int)

        dxs = self.sim_info["boxlen"]/2**(amrlevel)

        return (xs/dxs).astype(int)*2**(amrlevel*2) + (ys/dxs).astype(int)*2**(amrlevel) + (zs/dxs).astype(int)
    def __get_velocity__(self):
        vx = self.get_field("vx")*1e5
        vy = self.get_field("vy")*1e5
        vz = self.get_field("vz")*1e5

        return np.array([vx, vy, vz])

    def __get_cell_index_from_pos__(self, xs, ys, zs, index1D_lref, cell_index_lref):

        cell_index = np.full(xs.shape, -1, dtype = np.int64)

        ## Start by finding matching index1D
        index1D_to_find    = np.full(xs.shape, -1, dtype = np.int64)
        amr_lrefine_to_find    = np.full(xs.shape, -1, dtype = np.int64)
        not_found = np.full(xs.shape, True, dtype = np.bool8)
        amr_lrefine_min = self.sim_info["minref"]
        amr_lrefine_max = self.sim_info["maxref"]
        for lref in range(amr_lrefine_min, amr_lrefine_max + 1):
            N_not_found = np.sum(not_found)
            if N_not_found == 0:
                break
            index1D_to_find[not_found] = self.__calc_index1D__(xs = xs[not_found], ys = ys[not_found], zs = zs[not_found], amrlevel = lref)
            matches = sorted_in1d(index1D_to_find, index1D_lref[lref-amr_lrefine_min], numlib = np)
            matches[~not_found] = False
            if np.sum(matches) == 0:
                continue
            amr_lrefine_to_find[matches] = lref
            not_found[matches] = False

        ## See if we couldnt find any matching cells, and make sure that these corresponds to coordinates outside the simulation domain
        missing = not_found * ((xs >= 0) * (xs < self.sim_info["boxlen"])*
                               (ys >= 0) * (ys < self.sim_info["boxlen"])*
                               (zs >= 0) * (zs < self.sim_info["boxlen"]))
        if sum(missing) > 0 :
            print("[Simulation_reader] ERROR: could not find matching cells")
            print(xs[missing]/self.sim_info["boxlen"])
            print(ys[missing]/self.sim_info["boxlen"])
            print(zs[missing]/self.sim_info["boxlen"])
            sys.exit(0)

        ## Now find the correct cell_index
        for lref in range(amr_lrefine_min, amr_lrefine_max + 1):
            at_lref = np.where(amr_lrefine_to_find  == lref)[0]
            if len(at_lref) == 0:
                continue
            valid = np.where(sorted_in1d(index1D_to_find[at_lref], index1D_lref[lref - amr_lrefine_min], numlib = np))[0]
            cell_index[at_lref[valid]] = cell_index_lref[lref - amr_lrefine_min][np.searchsorted(index1D_lref[lref-amr_lrefine_min], index1D_to_find[at_lref[valid]])]
        # Catch any cells outside
        outside = ((xs < 0) | (xs > self.sim_info["boxlen"])|
                   (ys < 0) | (ys > self.sim_info["boxlen"])|
                   (zs < 0) | (zs > self.sim_info["boxlen"]))
        cell_index[outside] = -1
        return cell_index