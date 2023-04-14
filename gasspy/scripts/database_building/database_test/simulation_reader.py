"""
    Author: Loke Ohlin 
    Date: 05-2022
    Purpose: Dummy simulaton reader for the database test
"""
import numpy as np
import yaml
import os
import sys

import astropy.units as apyu
import astropy.constants as apyc

class Simulation_Reader():
    #   
    #  This is a dictionary to convert a gasspy variable name to the equivalent one used in the simulation
    #
    # We share everything, so no need to specify  
    gasspy_to_sim_label = {

    }

    #
    # List of all fields natively contained by the simlation or calculated in this class
    #
    contained_fields = ["number_density", "cell_size", "temperature", "ionizing_flux", "FUV_flux"]
    def __init__(self, sim_data):
        """
            arguments:
                arg_dict: dictionary (a dictionary of arguments needed by the reader to load the simulation data)
        """
        self.number_density = sim_data["number_density"]
        self.temperature = sim_data["temperature"]
        self.cell_size = sim_data["cell_size"]
        self.ionizing_flux = sim_data["ionizing_flux"]
        self.FUV_flux = sim_data["FUV_flux"]

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
        return self.__dict__[field]


    # Specific fucntion to calculate or load the index1D of the cells
    # Which in is the index of each cell in a raveled array sorted by x,y,z assuming that the entire simulation is on its refinement level
    # - Cannot have any arguments
    # - Must return the ravelled list for all cells
    def get_index1D(self):
        # not needed
        return

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
        # Not needed in this case
        return 


    def save_new_field(self, fieldname, data):
        self.__dict__[fieldname] = data
        return

    def load_new_field(self, fieldname):
        return self.__dict__["field"]