#!/usr/bin/python
""" Append launch directory to python path for importing config files """
import os
import sys
from .utils import compress
import numpy as np
import pickle
import astropy.table.table as table
from astropy.table import QTable
from astropy.table import Table
import time
from multiprocessing import Pool
import pandas as pd
import gc
import yaml

from . import opiate_defaults as defaults
sys.path.append(os.getcwd())

LastModDate = "2021.06.06.EWP"

class uniq_dict_creator(object):
    def __init__(self, **kwags):
        """ Import default and model specific settings """
        self.mask_parameters = {}

        self.unique_param_dict = {}
        
        self.N_unique = 0

        self.N_cells = 0

        self.outname = "SHELL_CDMASK2"
        self.outdir = "/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2"

        self.dxxyz = ["dx", "x", "y", "z"]
        self.gasfields = ["dens", "temp"]
        # gas mass density, temperature, fraction of atomic H (iha), ionized (ihp) and molecular (ih2),
        # and various gas fractions.

        # Radiation fields: Should possibly be defined based on code type, i.e. FLASH, RAMSES
        self.radfields = None

        #TODO Change the next lines to create a table. Option: Use astropy.tables
        self.cloudyfields = ["dx", "dens", "temp", "fluxes"]
        self.__default_compression_ratio__ = (1,5.0)
    
        self.compression_ratio ={
            'dx':(3, 1.0),\
            'dens':(1, 5.0),\
            'temp':(1, 5.0),\
            'fluxes':{
            'default':(1,5.0)}}
            # 'FUV':(1, 5.0),\
            # 'HII':(1, 5.0),\
            # 'HeII':(1, 5.0),\
            # 'HeIII':(1, 5.0)}

        self.simdata = {
            "temp"  :None,

            "dens"  :None,

            "dx"    :None,

            "x"     :None,
            "y"     :None,
            "z"     :None,
            
            "fluxes":{
                0:{"Emin":None,
                    "Emax":None,
                    "shape":None,
                    "data":None}
            }
        }

        self.log10_flux_low_limit={}

        self.compressedsimdata = {
            "temp"  :{"data":None},
            "dens"  :{"data":None},
            "dx"    :{"data":None},
            "fluxes":{}
        }

        self.__dict__.update(kwags)

    def pool_handler(self,func, exec_list,Ncores=16):
        p = Pool(min(len(exec_list), Ncores))
        p.map(func, exec_list)

    # def compute_cell(self, cell_i):
    #     if cell_i%self.print_every == 0:
    #         current_time = time.time()
    #         elappsed_time = current_time - self.t_start
    #         estimated_total_time = elappsed_time*(self.N_cells/float(cell_i+1))
    #         time_remaining = estimated_total_time - elappsed_time
    #         print("%i of %i: Elapsed time: %f, Est Time Remainint(%f)"%(cell_i,self.N_cells, elappsed_time, time_remaining))
    #     # initialize the data values array
    #     cell_data = []
    #     cloudyfield = {}

    #     dx = self.simdata["dx"][cell_i]
        
    #     cloudyfield["dx"] = float("%0.3f\t"%np.log10(self.simdata["dx"][cell_i]))
    #     cloudyparm = "%0.3f\t"%(np.log10(dx))

    #     #extract dxxyz positions
    #     for field in self.dxxyz:
    #         cell_data.append("%0.3e"%(self.simdata[field][cell_i]))

    #     #extract gas properties field
    #     for field in self.gasfields:
            
    #         try:
    #             value = "%0.3f"%(compress.number(self.simdata[field][cell_i], self.compression_ratio[field]))
    #         except:
    #             print("could not compress ", self.simdata[field][cell_i])
    #             value = "%0.3f"%(self.simdata[field][cell_i])

    #         if value == "-inf" or value == "inf":
    #             value = "%0.3f"%(np.log10(1e-99))

    #         #We use this string to ensure uniquness
    #         cloudyparm += "%s\t"%(value)

    #         #We append this to cell data, but this is deprecated
    #         cell_data.append(value)

    #         # We temporarily store the value. Used to be in arrays, but they were arbitrary and hard to maintain 
    #         # with variable fields
    #         cloudyfield[field] = float(value)

    #     #extract intensity radiation fields

    #     cloudyfield["fluxes"] = {}

    #     """ Cleaning step to deal with low and fully shielded cells"""
    #     for field in self.simdata["fluxes"].keys():

    #         if self.simdata["fluxes"][field]["data"][cell_i] > self.log10_flux_low_limit[field]:
    #             """ Do we have atleast 1 photon per cm-2?"""
    #             compressed = compress.number(float(self.simdata["fluxes"][field]["data"][cell_i]), self.compression_ratio["fluxes"][field])
    #             value = "%0.3f"%(compressed)

    #             # Why is this here?
    #             if float(value) == -1*float(value):
    #                 value = "%0.3f"%(0.0)
    #         else:
    #             value = "-99.000"
    #         if value == "-inf" or value == "inf":
    #             value = "%0.3f"%(np.log10(1e-99))

    #         # Append each flux field numerical value to data
    #         cell_data.append(value)
    #         cloudyparm += "%s\t"%(value)
    #         cloudyfield["fluxes"][field] = float(value)

    #     # if cell_data[-N_rad_fields:] != ['-99.000']*N_rad_fields:
    #     try:
    #         self.unique_param_dict[cloudyparm] += 1

    #     except:
    #         self.unique_param_dict[cloudyparm] = 1
    #         # for field in self.simdata.keys():
    #         #     """
    #         #     Do not copy positional data. It has no meaning in compressed data.
    #         #     Do not copy flux, it is nested with meta data.
    #         #     """
    #         #     if field is not ["fluxes"]:
    #         #         self.compressedsimdata[field][self.N_unique] = cloudyfield[field]
    #         #     else:
    #         #         for rad_field in self.compressedsimdata[field].keys():
    #         #             self.compressedsimdata[field][rad_field][self.N_unique]

    #         """
    #         If this is a new cell, then copy it's data to the arrays of unique cells
    #         """
    #         for field in self.cloudyfields.keys():
    #             if field != "fluxes":
    #                 self.compressedsimdata[field]["data"][self.N_unique] = cloudyfield[field]
    #             else:
    #                 for flux_type in cloudyfield[field].keys():
    #                     self.compressedsimdata[field][flux_type]["data"][self.N_unique] = cloudyfield[field][flux_type]

    #         """
    #         Iterate the unique field counter
    #         """                
    #         self.N_unique += 1

    def mask_data(self, simdata):
        """ Set masks based on mask parameters read in by the defaults library, or by myconfig"""
        masks = {}
        self.n_mask = 0
        """ Create a mask which contains all cells, since there is no such thing as negative density"""
        self.full_mask = simdata["density"] < 0
        for mask_name in self.mask_parameters.keys():
            """For each defined mask in mask_parameters_dictionary"""
            self.n_mask += 1

            partial_mask = simdata["density"] > 0
            mask_parameters = self.mask_parameters[mask_name]
            for parameter in sorted(self.mask_parameters.keys()):
                if self.mask_parameters[mask_name][parameter] != "default":
                    masks[parameter+"min"] = simdata[parameter] > min(self.mask_parameters[mask_name][parameter])
                    masks[parameter+"max"] = simdata[parameter] < max(self.mask_parameters[mask_name][parameter])

                    partial_mask = partial_mask*masks[parameter+"min"]*masks[parameter+"max"]
            
            self.full_mask = self.full_mask + partial_mask
            
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

        N_rad_fields = len(self.simdata["fluxes"].keys())
        self.N_cells = len(self.simdata['dx'])

        for field in self.simdata.keys():
            """
            Do not copy positional data. It has no meaning in compressed data.
            Do not copy flux, it is nested with meta data.
            """
            if field not in ["x","y","z","fluxes"]:
                if not field in self.compression_ratio.keys():
                    if "default" in self.compression_ratio.keys():
                        # Check if user supplied a default compression ratio
                        self.compression_ratio[field] = self.compression_ratio["default"]
                    else:
                        self.compression_ratio[field]=self.__default_compression_ratio__
                self.compressedsimdata[field]["data"] = compress.array(self.simdata[field],self.compression_ratio[field])

                
        for field in self.simdata["fluxes"].keys():
            # Intialize the compression storage attribute
            
            if not field in self.compression_ratio["fluxes"].keys():
                if "default" in self.compression_ratio["fluxes"].keys():
                    #Check if user specified a default flux compression ratio
                    self.compression_ratio["fluxes"][field] = self.compression_ratio["fluxes"]["default"]
                else:
                    self.compression_ratio["fluxes"][field] = self.__default_compression_ratio__
            
            try:
                self.log10_flux_low_limit[field]
            except:
                self.log10_flux_low_limit[field] = -99.0
            self.compressedsimdata["fluxes"][field] = {}
            for subfield in self.simdata["fluxes"][field]:
                if subfield == "data":
                    self.compressedsimdata["fluxes"][field]['data'] = compress.array(self.simdata['fluxes'][field]["data"],self.compression_ratio['fluxes'][field])
                    low_mask = self.compressedsimdata["fluxes"][field]['data'] < self.log10_flux_low_limit[field]
                    self.compressedsimdata["fluxes"][field]['data'][low_mask] = self.log10_flux_low_limit[field]

                else:
                    self.compressedsimdata["fluxes"][field][subfield] = self.simdata['fluxes'][field][subfield]

        # Clear the simdata from memory
        self.simdata=None
        gc.collect()

        """
        TRIM THE DATA!!!!
        The fields to store compressed data were done with arrays of length equal to the original data.
        The compression ensures that the length is <= len(original data)
        Thus we crop out the data longer than the self.N_unique
        """
        N_fields = len(self.cloudyfields) - 1 + N_rad_fields
        field_header = [None for i in range(N_fields)]
        a = [None for i in range(N_fields)]
        i_field = 0
        for field in self.cloudyfields:
            if field != "fluxes":
                #self.compressedsimdata[field]["data"] = self.compressedsimdata[field]["data"][:self.N_unique]
                # keep a record of the field order, including fluxes
                a[i_field] = self.compressedsimdata[field]["data"]
                field_header[i_field]=field
                i_field+=1
            else:
                for flux_type in self.compressedsimdata["fluxes"].keys():
                    # keep a record of the field order, including fluxes
                    a[i_field] = self.compressedsimdata["fluxes"][flux_type]["data"]
                    field_header[i_field]=flux_type
                    # self.compressedsimdata["fluxes"][flux_type]["data"] = [:self.N_unique]
                    i_field+=1

        for key in self.compressedsimdata.keys():
            if key != "fluxes":
                print(key, len(self.compressedsimdata[key]["data"]))
            else:
                for fluxkey in self.compressedsimdata["fluxes"].keys():
                    print(key, len(self.compressedsimdata['fluxes'][fluxkey]["data"]))

        stacked = np.vstack(a).T
        del(a)
        self.unique = pd.DataFrame(stacked).drop_duplicates()
        self.unique.columns = field_header

        # Save the unique dictionary to a pickle
        self.unique.to_pickle(self.outdir+"/"+self.outname+"_unique.pkl")

        # Save the flux definition to a yaml file
        flux_def = {}
        for field in self.compressedsimdata['fluxes'].keys():
            flux_def[field] = {}
            for fluxkey in self.compressedsimdata["fluxes"][field].keys():
                if fluxkey != "data":
                    flux_def[field][fluxkey] = self.compressedsimdata["fluxes"][field][fluxkey]
        with open(self.outdir+"/"+self.outname+"_fluxdef.yaml",'w') as yamlfile:
            data = yaml.dump(flux_def,yamlfile)
            print("flux definition written to yaml file")

        del(self.compressedsimdata)
        self.N_unique = self.unique.shape[0]
        del(stacked)

        return(self.N_unique/self.N_cells)

class opiate_to_cloudy(object):
    def __init__(self,
    outdir=None,
    outname=None,
    fluxdef_file=None,
    unique_panda_pickle_file=None,
    ForceFullDepth=False,
    **kwags):
        self.__dict__.update(defaults.parameters)

        self.MaxNumberModels = int(1e5)
        if outdir != None:
            self.outdir = outdir
        else:
            self.outdir = "./"
        
        if outname != None:
            self.outname = outname
        else:
            self.outname = "opiate"

        if fluxdef_file != None:
            self.fluxdef_file = fluxdef_file
        else:
            self.fluxdef_file = "%s/%s_fluxdef.yaml"%(self.outdir,self.outname)
        
        if unique_panda_pickle_file != None:
            self.unique_panda_pickle_file = unique_panda_pickle_file
        else:
            self.unique_panda_pickle_file = "%s/%s_unique.pkl"%(self.outdir,self.outname)

        try:
            self.__dict__.update(kwags)
        except:
            pass

        if self.unique_panda_pickle_file != None:
            self.unique_panda_pickle = pd.read_pickle(self.unique_panda_pickle_file)
        
        if self.fluxdef_file != None:
            with open(r'%s'%self.fluxdef_file) as file:
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                self.fluxdef = yaml.load(file, Loader=yaml.FullLoader)

        self.ForceFullDepth=ForceFullDepth

    def open_output(self, UniqID):
        """pass this a dictionary of parameters and values. Loop over and create prefix. Open output """
        try:
            os.stat(self.outdir+"./cloudy-output")
        except:
            os.mkdir(self.outdir+"./cloudy-output")

        prefix = "%s-%s"%(self.outname, UniqID)
        self.outfile = open(self.outdir+"./cloudy-output/"+prefix+".in", 'w')
        self.outfile.write("set save prefix \"%s\""%(prefix)+"\n")

    def check_for_if(self, cell_depth, cell_hden, cell_phi_ih, alpha=4.0e-13):
        """Check to see if cell contains a Hydrogen ionization front"""

        if self.ForceFullDepth is True:
            return True

        else:
            # alpha = 4.0e-13 # H recombinations per second cm-6
            ion_depth = cell_phi_ih /(alpha * 10**(cell_hden**2))

        # Change hardcoded 1e13 to mean free path of ionizing photon in H0.
        if ion_depth <= 10**cell_depth and ion_depth > 1e13:
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

    def set_temperature(self, log10_temperature, is_ionization_front, force_Teq=False, force_Tconst=False, T_law=False):
        """Set constant temperature if not modeling the actual ionization front temp gradients"""
        if T_law is False:
            if (is_ionization_front is False and force_Teq is False) or (is_ionization_front is True and force_Tconst is True):
                self.outfile.write("constant temperature %s\n"%(10**log10_temperature))
        else:
            sys.exit("T_law not implemented ... yet")

    def set_fsb99_phi_ih(self, phi_fsb99, SB99model="1e6cluster_norot_Z0014_BH120"):
        # "table star \"%s.mod\" age=%0.1f years \n" % (SB99model, np.max([SB99_age, i.SB99_age_min])))
        self.outfile.write("table star \"%s.mod\" age=%0.1e years \n" % (SB99model, 1e5))
        self.outfile.write("phi(h) = %s, range 1.0 to 3 Ryd\n"%(phi_fsb99))

    def set_fluxes(self, model, flux_definition):
        for field in flux_definition.keys():
            phi = model[field]
            EminRyd = flux_definition[field]['Emin']/13.6
            EmaxRyd = flux_definition[field]['Emax']/13.6
            if flux_definition[field]['shape'].endswith(".sed"):
                self.outfile.write("table SED \"%s\"\n"%flux_definition[field]['shape'])
            else:
                sys.exit("Currently only SEDs defined by a SED file are supported")
            self.outfile.write("phi(h) = %s, range %f to %f Ryd\n"%(phi, EminRyd, EmaxRyd))

    def create_cloudy_input_file(self, uniqueID=None, model=None, init_file=None, flux_definition=None):
        _UniqID=uniqueID


        """ create prefix for models and open Cloudy input file for writing"""
        self.open_output(_UniqID)
        
        # CLOUDY_modelIF is set to True by default. Can be changed in parameter file to false,
        # which will prevent isIF from executing

        #check if hden is log of mass density or volume density.
        if float(model["dens"]) < -6.0:
            model["dens"] = model["dens"] - np.log10(1.67e-24)

        isIF = self.ForceFullDepth

        _phi_ih = 99.99

        if self.ForceFullDepth is False:
            _phi_ih = 0.0
            for radfield in flux_definition.keys():
                if flux_definition[radfield]['Emin'] > 13.5984:
                    _phi_ih += 10**model[radfield]
                
            isIF = self.check_for_if(model["dx"], model["dens"], _phi_ih)
        else:
            isIF = False

        """ Set common init file """
        if init_file is not None:
            self.set_cloudy_init_file(init_file)

        """ Write individual cloudy parameters to input file """
        self.set_depth(model["dx"])
        self.set_hden(model["dens"])
        self.set_nend(isIF)
        self.set_temperature(model["temp"], isIF)

        self.set_fluxes(model, flux_definition)

        """ Close input file """
        self.outfile.close()


    def make_user_seds(self):
        """
        make user defined seds using the energy bins defined by the user
        """
        try:
            os.stat(self.outdir+"./cloudy-output")
        except:
            os.mkdir(self.outdir+"./cloudy-output")

        for field in self.fluxdef.keys():
            sedfile = "./cloudy-output/%s_%s.sed"%(self.save_prefix, field)
            outfile = open(sedfile, 'w')
            if self.unique_panda_pickle["fluxes"][field]["shape"] == 'const':
                outfile.write("%f -35.0 nuFnu\n"%(self.fluxdef[field]["Emin"]*0.99/13.6) )
                outfile.write("%f 1.000 nuFnu\n"%(self.fluxdef[field]["Emin"]/13.6) )
                outfile.write("%f 1.000 nuFnu\n"%(self.fluxdef[field]["Emax"]/13.6) )
                outfile.write("%f -35.0 nuFnu\n"%(self.fluxdef[field]["Emax"]*1.01/13.6) )
            outfile.close()



    def process_grid(self, model_limit=-1, N0=0):
        # dx + gas fields, which could be more than den and temp
        max_depth = {}
        self.unique_panda_pickle = self.unique_panda_pickle.reset_index()
        N_models = self.unique_panda_pickle.shape[0]

        #This is the maximum number of possible unique IDs. In the end we will populate this from 0 to N_unique, and crop the array. This prevents repeated copies that result form appending.
        uniqueIDs = np.array(range(N_models)) + N0

        for i in range(N_models):
            # How to access the density variable for the 0th position.
            # self.unique_panda_pickle['dens'].loc[0]
            current_model_parameters = self.unique_panda_pickle.loc[i]

            # This reads the order from above
            # Read as Temp, then log n
            if self.debug == False:
                self.create_cloudy_input_file(uniqueID=uniqueIDs[i], model=current_model_parameters, init_file=self.CLOUDY_INIT_FILE, flux_definition=self.fluxdef)
            if self.debug == True:
                rad_fluxes_string = " ".join([current_model_parameters[key] for key in self.fluxdef.keys()])
                print(uniqueIDs[i], current_model_parameters["depth"], current_model_parameters["hden"], current_model_parameters["temp"], rad_fluxes_string)
