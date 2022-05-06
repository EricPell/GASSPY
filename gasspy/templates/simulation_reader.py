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
from astropy.io import fits

class Simulation_Reader():
    #   
    #  This is a dictionary to convert a gasspy variable name to the equivalent one used in the simulation
    #  
    gasspy_to_sim_label = {
        # density, temperatue and amr level have different names, these need to be specified
        "dens": "rho",
        "temp": "T",
        "amr_lrefine": "amrlevel",
        # positions and velocity happens to share names, and can be excluded from this dict
        "x": "x", 
        "y": "y",
        "z": "z",
        "vx": "vx",
        "vy": "vy",
        "vz": "vz",
        # There is no dx for the simulation (its calculated from amr_lrefine) so can be excluded but MUST be defined as a function here
        "dx": "dx",
        # In this case the radiation fields has had their names changed
        "FUV" : "NpFUV",
        "HII" : "NpHII",
        "HeII" : "NpHeII",
        "HeIII" : "NpHeIII",
    }

    def __init__(self, simdir, arg_dict):
        """
            arguments:
                arg_dict: dictionary (a dictionary of arguments needed by the reader to load the simulation data)
        """

        self.simdir = simdir
        assert arg_dict["output_number"] is not None, "Error: No output number selected!"
        self.output_number = int(arg_dict["output_number"])
        self.__load_info_file__()
        
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
        if field == "dx":
            return self.__get_dx__()
        elif field in self.gasspy_to_sim_label.keys():
            fname = self.__get_filename__(self.gasspy_to_sim_label[field])
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
        return self.get_field("rho")/1e-24

    
    ######
    #
    #   Internal functions specific to this simulation
    #   change to your hearts content
    #
    #######

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