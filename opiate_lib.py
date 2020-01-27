#!/usr/bin/python
""" Append launch directory to python path for importing config files """
import os
import sys
import compress
import numpy as np
import pickle
import astropy.table.table as table

sys.path.append(os.getcwd())

LastModDate = "24.05.2016.EWP"

class uniq_dict_creator(object):
    def __init__(self):
        """ Import default and model specific settings """
        import defaults
        try:
            import myconfig
        except:
            import myconfig_example as myconfig
            print("You are reading the example config file. This is not recommended")

        try:
            self.append_db = myconfig.append_db
        except:
            self.append_db = defaults.append_db

        try:
            self.mask_parameters_dict = myconfig.mask_parameters_dict
        except:
            self.mask_parameters_dict = defaults.mask_parameters_dict

        try:
            self.flux_type = myconfig.flux_type
        except:
            self.flux_type = defaults.flux_type

        try:
            self.compression_ratio = myconfig.compression_ratio
        except:
            self.compression_ratio = defaults.compression_ratio

        try: 
            self.log10_flux_low_limit = myconfig.log10_flux_low_limit
        except: 
            self.log10_flux_low_limit = defaults.log10_flux_low_limit

        try:
            self.debug = myconfig.debug
        except:
            self.debug = defaults.debug

        self.unique_param_dict = {}
        self.data = []

        self.dxxyz = ["dx", "x", "y", "z"]
        self.gasfields = ["dens", "temp"]
        # gas mass density, temperature, fraction of atomic H (iha), ionized (ihp) and molecular (ih2),
        # and various gas fractions.

        # Radiation fields: Should possibly be defined based on code type, i.e. FLASH, RAMSES
        try:
            self.radfields = myconfig.radfields
        except:
            self.radfields = defaults.radfields

        #TODO Change the next lines to create a table. Option: Use astropy.tables
        self.cloudyfields = ["dx", "dens", "temp"] + self.radfields[self.flux_type]

    def mask_data(self, simdata):
        """ Set masks based on mask parameters read in by the defaults library, or by myconfig"""
        masks = {}
        self.n_mask = 0
        """ Create a mask which contains all cells, since there is no such thing as negative density"""
        self.full_mask = simdata["density"] < 0
        for mask_name in self.mask_parameters_dict.keys():
            """For each defined mask in mask_parameters_dictionary"""
            n_mask += 1

            partial_mask = simdata["density"] > 0
            mask_parameters = self.mask_parameters_dict[mask_name]
            for parameter in sorted(self.mask_parameters_dict.keys()):
                if self.mask_parameters_dict[mask_name][parameter] != "default":
                    masks[parameter+"min"] = simdata[parameter] > min(self.mask_parameters_dict[mask_name][parameter])
                    masks[parameter+"max"] = simdata[parameter] < max(self.mask_parameters_dict[mask_name][parameter])

                    partial_mask = partial_mask*masks[parameter+"min"]*masks[parameter+"max"]
            
            self.full_mask = self.full_mask + partial_mask

    def extract_data(self, dd):
        """
        extract data from a dd object, typically read from an MHD simulation with yt
        """
        for field in self.dxxyz:
            self.simdata[field] = dd[field][self.full_mask].value

        for field in self.gasfields:
            if field == "dens":
                mH = 1.67e-24 # Mass of the hydrogen atom
                self.simdata[field] = np.log10(dd[field][self.full_mask].value/mH)
            else:
                self.simdata[field] = np.log10(dd[field][self.full_mask].value)

        for field in self.radfields:
            if self.flux_type is "fervent":
                if field == "flge":
                    self.simdata[field] = np.log10(dd[field][self.full_mask].value)
                else:
                    self.simdata[field] = dd[field][self.full_mask].value-2.0*np.log10(dd['dx'][self.full_mask].value)
                    tolowmask = self.simdata[field] < 0.0
                    self.simdata[field][tolowmask] = -99.00

            if self.flux_type is "Hion_excessE":
                    self.simdata[field] = np.log10(dd[field][self.full_mask].value*2.99792e10) # Hion_excessE is an energy density. U*c is flux 
                    #to_low_value =  -np.log10(2.1790E-11)*1000 # energy flux of one ionizing photon == 13.6eV \times 1000 photons per cm-2 which is 100x less than the ISRF. See ApJ 2002, 570, 697
                    #tolowmask = simdata[field] < to_low_value
                    #simdata[field][tolowmask] = -99.00
            
    def collect_den_temp_flux(self):
        """
        SIM DATA MUST BE A TABLE OR DICTIONARY CONTAINING
        'dx': cell size
        'x:y:z': cartisian coordates of the cell, used for optional masking
        'temp': cell gas kinetic temperature

        gas_fields: Any relevant gas field, describing abundances, either realative or absolute in density

        optional fields:
            velocity: (to be) used to calculated compressional heating
        """
        #Loop over every cell in the masked region

        for field in self.simdata["flux"].keys():
            try:
                self.log10_flux_low_limit[field]
            except:
                self.log10_flux_low_limit[field] = self.log10_flux_low_limit["default"]

        N_rad_fields = len(self.simdata["flux"].keys())
        N_cells = len(self.simdata['dx'])
        for cell_i in range(N_cells):
            # initialize the data values array
            cell_data = []

            dx = self.simdata["dx"][cell_i]

            cloudyparm = "%0.3f\t"%(np.log10(dx))

            #extract dxxyz positions
            for field in self.dxxyz:
                cell_data.append("%0.3e"%(self.simdata[field][cell_i]))

            #extract gas properties field
            for field in self.gasfields:
                try:
                    value = "%0.3f"%(compress.number(self.simdata[field][cell_i], self.compression_ratio[field]))
                except:
                    print("could not compress ", simdata[field][cell_i])
                    value = "%0.3f"%(simdata[field][cell_i])
                if value == "-inf" or value == "inf":
                    value = "%0.3f"%(np.log10(1e-99))
                try:
                    self.cloudyfields.index(field)
                    cloudyparm += "%s\t"%(value)
                except:
                    "field not a cloudy param"
                # Append the field numerical value to data
                self.data.append(value)

            #extract intensity radiation fields

            """ Cleaning step to deal with low and fully shielded cells"""
            for field in self.simdata["flux"].keys():
                logflux = np.log10(self.simdata["flux"][field]['data'][cell_i])
                if logflux > self.log10_flux_low_limit[field]:
                    """ Do we have atleast 1 photon per cm-2?"""
                    value = "%0.3f"%compress.number(float(logflux), self.compression_ratio['flux'][field])
                else:
                    value = "-99.000"
                if value == "-inf" or value == "inf":
                    value = "%0.3f"%(np.log10(1e-99))
                # Append the field numerical value to data
                cell_data.append(value)
                cloudyparm += "%s\t"%(value)

            # if cell_data[-N_rad_fields:] != ['-99.000']*N_rad_fields:
            try:
                self.unique_param_dict[cloudyparm] += 1
            except:
                self.unique_param_dict[cloudyparm] = 1

            
            self.data.append(cell_data)
        return(len(self.unique_param_dict.keys())/N_cells)
