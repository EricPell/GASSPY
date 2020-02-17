#!/usr/bin/python
""" Append launch directory to python path for importing config files """
import os
import sys
import compress
import numpy as np
import pickle
import astropy.table.table as table
from astropy.table import QTable
from astropy.table import Table

import opiate_defaults as defaults
sys.path.append(os.getcwd())

LastModDate = "24.05.2016.EWP"

class uniq_dict_creator(object):
    def __init__(self):
        """ Import default and model specific settings """
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
        
        self.N_unique = 0

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
        self.cloudyfields = ["dx", "dens", "temp"] # + self.radfields[self.flux_type]

        self.simdata = {
            "temp"  :None,

            "dens"  :None,

            "dx"    :None,

            "x"     :None,
            "y"     :None,
            "z"     :None,
            
            "flux":{
                0:{"Emin":None,
                    "Emax":None,
                    "shape":None,
                    "data":None}
            }
        }

        self.compressedsimdata = {
            "temp"  :{"data":None},
            "dens"  :{"data":None},
            "dx"    :{"data":None},
            "flux":{
                0:{"Emin":None,
                    "Emax":None,
                    "shape":None,
                    "data":None}
            }
        }


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
            
    def compress_simdata(self):
        """
        SIM DATA MUST BE A TABLE OR DICTIONARY CONTAINING
        'dx': cell size
        'x:y:z': cartisian coordates of the cell, used for optional masking
        'temp': cell gas kinetic temperature

        gas_fields: Any relevant gas field, describing abundances, either realative or absolute in density

        optional fields:
            velocity: (to be) used to calculated compressional heating
        """

        N_rad_fields = len(self.simdata["flux"].keys())
        N_cells = len(self.simdata['dx'])

        for field in self.simdata.keys():
            """
            Do not copy positional data. It has no meaning in compressed data.
            Do not copy flux, it is nested with meta data.
            """
            if field not in ["x","y","z","flux"]:
                self.compressedsimdata[field]["data"] = np.full(N_cells, np.nan) 

        for field in self.simdata["flux"].keys():
            # Intialize the compression storage attribute
            self.compressedsimdata['flux'][field] = {}
            self.compressedsimdata['flux'][field]["Emin"]  = self.simdata["flux"][field]["Emin"]
            self.compressedsimdata['flux'][field]["Emax"]  = self.simdata["flux"][field]["Emax"]
            self.compressedsimdata['flux'][field]["shape"] = self.simdata["flux"][field]["shape"]
            self.compressedsimdata['flux'][field]["data"]  = np.full(N_cells, np.nan)

            try:
                self.log10_flux_low_limit[field]
            except:
                self.log10_flux_low_limit[field] = self.log10_flux_low_limit["default"]

        for cell_i in range(N_cells):
            # initialize the data values array
            cell_data = []
            cloudyfield = {}

            dx = self.simdata["dx"][cell_i]
            
            cloudyfield["dx"] = float("%0.3f\t"%np.log10(dx))
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

                #We use this string to ensure uniquness
                cloudyparm += "%s\t"%(value)

                #We append this to cell data, but this is deprecated
                cell_data.append(value)

                # We temporarily store the value. Used to be in arrays, but they were arbitrary and hard to maintain 
                # with variable fields
                cloudyfield[field] = float(value)

            #extract intensity radiation fields

            cloudyfield["flux"] = {}

            """ Cleaning step to deal with low and fully shielded cells"""
            for field in self.simdata["flux"].keys():

                logflux = np.log10(self.simdata["flux"][field]["data"][cell_i])

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
                cloudyfield["flux"][field] = float(value)

            # if cell_data[-N_rad_fields:] != ['-99.000']*N_rad_fields:
            try:
                self.unique_param_dict[cloudyparm] += 1

            except:
                self.unique_param_dict[cloudyparm] = 1
                # for field in self.simdata.keys():
                #     """
                #     Do not copy positional data. It has no meaning in compressed data.
                #     Do not copy flux, it is nested with meta data.
                #     """
                #     if field is not ["flux"]:
                #         self.compressedsimdata[field][self.N_unique] = cloudyfield[field]
                #     else:
                #         for rad_field in self.compressedsimdata[field].keys():
                #             self.compressedsimdata[field][rad_field][self.N_unique]

                """
                If this is a new cell, then copy it's data to the arrays of unique cells
                """
                for field in cloudyfield.keys():
                    if field != "flux":
                        self.compressedsimdata[field]["data"][self.N_unique] = cloudyfield[field]
                    else:
                        for flux_type in cloudyfield[field].keys():
                            self.compressedsimdata[field][flux_type]["data"][self.N_unique] = cloudyfield[field][flux_type]

                """
                Iterate the unique field counter
                """                
                self.N_unique += 1

        """
        The fields to store compressed data were done with arrays of length equal to the original data.
        The compression ensures that the length is <= len(original data)
        Thus we crop out the data longer than the self.N_unique
        """            
        for field in cloudyfield.keys():
            if field != "flux":
                self.compressedsimdata[field]["data"] = self.compressedsimdata[field]["data"][:self.N_unique]
            else:
                for flux_type in cloudyfield["flux"].keys():
                    self.compressedsimdata["flux"][flux_type]["data"] = self.compressedsimdata["flux"][flux_type]["data"][:self.N_unique]

        return(self.N_unique/N_cells)

class opiate_to_cloudy(object):
    def __init__(self, save_prefix="opiate"):

        try:
            import myconfig
        except:
            import myconfig_example as myconfig
            print("You are reading the example config file. This is not recommended")

        try:
            self.CLOUDY_modelIF = myconfig.CLOUDY_modelIF
        except:
            self.CLOUDY_modelIF = defaults.CLOUDY_modelIF

        try:
            self.flux_type = myconfig.flux_type
        except:
            self.flux_type = defaults.flux_type

        if self.flux_type is "default":
            """ 
            We will proceed to assume a flat spectrum defined in each energy bin. Please note
            this is not a particularly physical choice
            """
            sys.exit("I can not proceed without knowing the type of radiation bands used in the simulation")

        try:
            self.compression_ratio = myconfig.compression_ratio
        except:
            self.compression_ratio = defaults.compression_ratio

        try:
            self.CLOUDY_INIT_FILE = myconfig.CLOUDY_INIT_FILE
        except:
            self.CLOUDY_INIT_FILE = defaults.CLOUDY_INIT_FILE

        #"""Decide to force every model to be calculated with full depth, or default to a single zone"""
        try:
            """Try and read ForceFullDepth from myconfig"""
            self.ForceFullDepth = myconfig.ForceFullDepth
        except:
            """Else ForceFullDepth is defined in defaults as true, set the global to true"""
            self.ForceFullDepth = defaults.ForceFullDepth

        try:
            self.debug = myconfig.debug
        except:
            self.debug = defaults.debug

        self.MaxNumberModels = int(1e5)

        self.save_prefix = save_prefix

        # Assume we output to the current directory, plus cloud-output
        self.outdir = ""

        # Import string containing each continuum shape.
        if self.flux_type is 'fervent':
            import fervent_bands # Import continuum shapes
            self.bands = fervent_bands.bands
        elif self.flux_type is 'Hion_excessE':
            import Hion_excessE_bands
            self.bands = Hion_excessE_bands.bands
        elif self.flux_type is 'fsb99':
            """nothing to import"""
        elif self.flux_type is 'user':
            """
            Use a user defined flux type.
            Requires definition of energy bands using Emin,Emax{,optional:alpha}
            """
            self.flux_type = 'user'
        else:
            sys.exit("You have selected a flux type I do not understand. Flux Type = %s"%(self.flux_type))


    def open_output(self, UniqID):
        """pass this a dictionary of parameters and values. Loop over and create prefix. Open output """
        try:
            os.stat(self.outdir+"./cloudy-output")
        except:
            os.mkdir(self.outdir+"./cloudy-output")

        prefix = "./cloudy-output/%s-%s"%(self.save_prefix, UniqID)
        self.outfile = open(prefix+".in", 'w')
        self.outfile.write("set save prefix \"%s\""%(prefix)+"\n")

    def check_for_if(self, cell_depth, cell_hden, cell_phi_ih, cell_phi_i2, alpha=4.0e-13):
        """Check to see if cell contains a Hydrogen ionization front"""

        if self.ForceFullDepth is True:
            return True

        else:
            # alpha = 4.0e-13 # H recombinations per second cm-6
            ion_depth = (10**(float(cell_phi_ih)) +\
            10**(float(cell_phi_i2)))/(alpha * 10**(float(cell_hden))**2)

        # Change hardcoded 1e13 to mean free path of ionizing photon in H0.
        if ion_depth <= 10**float(cell_depth) and ion_depth > 1e13:
            return True
        else:
            return False

    def set_cloudy_init_file(self, init_file):
        """ Write command to cloudy input file to use custom init file """
        self.outfile.write("init \"%s\"\n"%(init_file))

    def set_depth(self, model_depth):
        """Write command to cloudy input file to set depth of model"""
        self.outfile.write("stop depth %s\n"%(model_depth))

    def set_hden(self, model_hden):
        """Write command to cloudy input file to set hydrogen density"""
        self.outfile.write("hden %s\n"%(model_hden))

    def set_nend(self, model_is_ionization_front):
        """Write command to cloudy input file to set number of zones to simulate"""
        if model_is_ionization_front is True:
            #Do not set constant temperature if IF exists
            self.outfile.write("set nend 5000\n")

        if model_is_ionization_front is False:
            #Set constant temperature if IF does not exist
            self.outfile.write("set nend 1\n")

    def set_temperature(self, temperature, is_ionization_front, force_Teq=False, force_Tconst=False, T_law=False):
        """Set constant temperature if not modeling the actual ionization front temp gradients"""
        if float(temperature) <= 0:
            sys.exit("WHAT DID YOU DO, THERE ARE NO SUCH THINGS AS NEGATIVE TEMPERATURES!!!!! (The temperature is assumed to be in K - thanks Max D.)")
        if T_law is False:
            if (is_ionization_front is False and force_Teq is False) or (is_ionization_front is True and force_Tconst is True):
                self.outfile.write("constant temperature %s\n"%(temperature))
        else:
            sys.exit("T_law not implemented ... yet")

    def set_I_ge(self, I_ge):
        if I_ge != "-99.000":
            self.outfile.write(fervent_bands.flge)
            self.outfile.write("intensity %s, range 0.41 to 0.823 Ryd\n"%(I_ge))

    def set_phi_uv(self,phi_uv):
        if phi_uv != "-99.000":
            self.outfile.write(fervent_bands.fluv)
            self.outfile.write("phi(h) = %s, range 0.823 to 1.0 Ryd\n"%(phi_uv))

    def set_phi_ih(self,phi_ih):
        if phi_ih != "-99.000":
        #if phi_ih > 0:
            self.outfile.write(fervent_bands.flih)
            self.outfile.write("phi(h) = %s, range 1.0 to 1.117 Ryd\n"%(phi_ih))

    def set_phi_i2(self,phi_i2):
        if phi_i2 != "-99.000":
        #if phi_i2 > 0:
            self.outfile.write(fervent_bands.fli2)
            self.outfile.write("phi(h) = %s, range 1.117 to 3 Ryd\n"%(phi_i2))

    def set_fsb99_phi_ih(self, phi_fsb99, SB99model="1e6cluster_norot_Z0014_BH120"):
        # "table star \"%s.mod\" age=%0.1f years \n" % (SB99model, np.max([SB99_age, i.SB99_age_min])))
        self.outfile.write("table star \"%s.mod\" age=%0.1e years \n" % (SB99model, 1e5))
        self.outfile.write("phi(h) = %s, range 1.0 to 3 Ryd\n"%(phi_fsb99))

    def set_user_flux(self, flux_array, user_flux_definition):
        for phi, field in zip(flux_array, user_flux_definition.keys()):
            # "table star \"%s.mod\" age=%0.1f years \n" % (SB99model, np.max([SB99_age, i.SB99_age_min])))
            self.outfile.write("table SED \"./cloudy-output/opiate_user_flux_%s.sed\"\n"%field)
            self.outfile.write("phi(h) = %s, range %f to %f Ryd\n"%(phi, user_flux_definition[field]['Emin']/13.6,user_flux_definition[field]['Emax']/13.6))

    def set_Hion_excessE_phi_ih(self,I_ih):
        if I_ih != "-99.000":
        #if phi_ih > 0:
            self.outfile.write(Hion_excessE_bands.flih)
            self.outfile.write("intensity = %s, range 1.0 to 3.0 Ryd\n"%(I_ih))

    def create_cloudy_input_file(self, model):
        _UniqID=model["UniqID"]
        _depth=model["depth"]
        _hden=model["hden"]
        _T=model["temp"]
        flux_array=model["rad_fluxes"]
        flux_type=model["flux_type"] 
        _cloudy_init_file=model["CLOUDY_INIT_FILE"]

        """ create prefix for models and open Cloudy input file for writing"""
        self.open_output(_UniqID)
        if self.flux_type is "fervent":
            """4 band fervent"""
            (_I_ge, _phi_uv, _phi_ih, _phi_i2) = flux_array

        elif self.flux_type is "Hion_excessE":
            """1 band simple ionizing SED"""
            _phi_ih = flux_array[0]
            (_I_ge, _phi_uv, _phi_i2) = (np.nan, np.nan, np.nan)

        elif self.flux_type is "fsb99":
            _phi_fsb99 = flux_array[0]
            (_phi_ih, _I_ge, _phi_uv, _phi_i2) = (np.nan, np.nan, np.nan, np.nan)

        # CLOUDY_modelIF is set to True by default. Can be changed in parameter file to false,
        # which will prevent isIF from executing

        #check if hden is log of mass density or volume density.
        if float(_hden) < -6.0:
            _hden = str(float(_hden) - np.log10(1.67e-24))

        isIF = self.ForceFullDepth

        if self.ForceFullDepth is False:
            if(self.CLOUDY_modelIF):
                isIF = self.check_for_if(_depth, _hden, _phi_ih, _phi_i2)
            else:
                isIF = False
    
        """ Set common init file """
        self.set_cloudy_init_file(_cloudy_init_file)

        """ Write individual cloudy parameters to input file """
        self.set_depth(_depth)
        self.set_hden(_hden)
        self.set_nend(isIF)
        self.set_temperature(_T, isIF)
        
        if self.flux_type is "fervent":
            self.set_I_ge(_I_ge)
            self.set_phi_uv(_phi_uv)
            self.set_phi_ih(_phi_ih)
            self.set_phi_i2(_phi_i2)
        elif self.flux_type is "Hion_excessE":
            self.set_Hion_excessE_phi_ih(_phi_ih)
        elif self.flux_type is "fsb99":
            self.set_fsb99_phi_ih(_phi_fsb99)
        elif self.flux_type is 'user':
            self.set_user_flux(flux_array, model['user_flux_definition'])

        """ Close input file """
        self.outfile.close()

    def make_model(self, UniqID, depth, hden, temp, rad_fluxes, flux_type, init_file, user_flux_definition=None):

        self.model_dict = {"UniqID":UniqID,
        "depth":depth,
        "hden":hden,
        "temp":temp,
        "rad_fluxes":rad_fluxes,
        "flux_type":flux_type,
        "CLOUDY_INIT_FILE":init_file}

        if flux_type == 'user':
            self.model_dict['user_flux_definition'] = {}
            for field in user_flux_definition.keys():
                self.model_dict['user_flux_definition'][field] = {}
                self.model_dict['user_flux_definition'][field]["Emin"]  = user_flux_definition[field]["Emin"]
                self.model_dict['user_flux_definition'][field]["Emax"]  = user_flux_definition[field]["Emax"]
                self.model_dict['user_flux_definition'][field]["shape"] = user_flux_definition[field]["shape"]

    def make_user_seds(self, opiate_data):
        """
        make user defined seds using the energy bins defined by the user
        """
        try:
            os.stat(self.outdir+"./cloudy-output")
        except:
            os.mkdir(self.outdir+"./cloudy-output")

        for field in opiate_data['flux'].keys():
            sedfile = "./cloudy-output/%s_user_flux_%s.sed"%(self.save_prefix, field)
            outfile = open(sedfile, 'w')
            if opiate_data['flux'][field]["shape"] is 'const':
                outfile.write("%f 1.000 nuFnu\n"%(opiate_data['flux'][field]["Emin"]/13.6) )
                outfile.write("%f 1.000 nuFnu\n"%(opiate_data['flux'][field]["Emax"]/13.6) )
            outfile.close()



    def process_grid(self, opiate_data, model_limit=-1):
        # dx + gas fields, which could be more than den and temp
        max_depth = {}
        N_models = len(opiate_data['dens']['data'])
        uniqueIDs = np.array(range(N_models))

        if self.flux_type is 'user':
            self.make_user_seds(opiate_data)

        # We are going to create arrays that contain everything but depth to find all the unique models besides depth
        # Join all the fluxes into a unique array.
        all_rad_fluxes = np.array([ [ "%0.4f"%opiate_data['flux'][rad_field]['data'][i] for rad_field in opiate_data['flux'].keys() ] for i in range(N_models) ])

        # This sets the order below
        # Store as Temp, then log n
        all_gas  = np.array([ "%0.4f,%0.4f"%(T, n) for T, n in zip(opiate_data['temp']['data'], opiate_data['dens']['data']) ] )

        initial_conditions = np.array([ all_gas_i + "," + ",".join(all_rad_fluxes_i) for (all_gas_i, all_rad_fluxes_i) in zip(all_gas, all_rad_fluxes) ])

        unique_inital_conditions = np.unique(initial_conditions)

        # For each initial condition we will make mask 
        # which will use to get the maximum physical depth of 
        # any model sharing the same gas_properties and radiation fluxes

        max_depths = [ np.max(opiate_data['dx']["data"][initial_conditions == initial_condition]) for initial_condition in unique_inital_conditions ]
        max_depth_uniqueIDs = [ np.max(uniqueIDs[initial_conditions == initial_condition]) for initial_condition in unique_inital_conditions ]

        # The resulting depth array has a shape and size equal to the unique_initial conditions.

        frozen_conditions = unique_inital_conditions.tolist()
        for i, initial_condition in enumerate(frozen_conditions):
            if ( model_limit < 0 ) or (i <= model_limit):
                UniqID = max_depth_uniqueIDs[i]
                depth = max_depths[i]
                parameters = initial_condition.split(",")
                # This reads the order from above
                # Read as Temp, then log n
                [temp, hden] = parameters[:2]
                rad_fluxes = parameters[2:]
                if ",".join(rad_fluxes) != ",".join(["-99.0000"]*len(rad_fluxes)):
                    if self.debug == False:
                        self.make_model("%010i"%UniqID, depth, hden, temp, rad_fluxes, self.flux_type, self.CLOUDY_INIT_FILE, user_flux_definition=opiate_data['flux'])
                        self.create_cloudy_input_file(self.model_dict)
                    if self.debug == True:
                        print(UniqID, depth, hden, temp, rad_fluxes_string)

        save_dictionary = {"max_depths":max_depths, "max_depth_uniqueIDs":max_depth_uniqueIDs, "unique_inital_conditions":frozen_conditions}
        t = Table()
        t['max_depth'] = max_depths
        t['max_depth_uniqueIDs']  = max_depth_uniqueIDs
        t['unique_inital_conditions'] = frozen_conditions
        t.write("opiate_physical_params_of_id.fits")
        
        with open(self.save_prefix+'_max_depth.pckl', 'wb') as handle:
            pickle.dump(save_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
