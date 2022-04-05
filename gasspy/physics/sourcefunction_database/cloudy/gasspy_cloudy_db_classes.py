#!/usr/bin/python
""" Append launch directory to python path for importing config files """
import os
import sys
from gasspy.shared_utils import compress
import numpy as np
import pickle
import astropy.table.table as table
from astropy.table import QTable
from astropy.table import Table
from astropy import units as u
import time
from multiprocessing import Pool
import pandas as pd
import gc
import yaml
import gasspy, pathlib
gasspy_path = pathlib.Path(gasspy.__file__).resolve().parent

from . import gasspy_cloudy_db_defaults as defaults
sys.path.append(os.getcwd())

LastModDate = "2022.02.14.EWP"

class uniq_dict_creator(object):
    def __init__(self, **kwags):
        """ Import default and model specific settings """
        self.mask_parameters = {}

        self.unique_param_dict = {}
        
        self.N_unique = 0

        self.N_cells = 0

        self.unified_fluxes = False

        self.outname = "test"
        self.outdir = "./"
        self.gasspy_subdir="GASSPY"
        self.save_compressed3d = False

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

        assert type(self.gasspy_subdir) == str, "gasspy_subdir not a string"
        if not self.gasspy_subdir.endswith("/"):
            self.gasspy_subdir = self.gasspy_subdir + "/"
        if not self.gasspy_subdir[0] == "/":
            self.gasspy_subdir = "/" + self.gasspy_subdir


    def pool_handler(self,func, exec_list,Ncores=16):
        p = Pool(min(len(exec_list), Ncores))
        p.map(func, exec_list)
    
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
            
    def compress_simdata(self, save_compressed3d=False):
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
            if field not in ["x","y","z","vx","vy","vz","amr","fluxes"]:
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

        N_fields = len(self.cloudyfields) - 1 + N_rad_fields
        self.field_header = [None for i in range(N_fields)]
        self.stacked = [None for i in range(N_fields)]
        i_field = 0
        for field in self.cloudyfields:
            if field != "fluxes":
                #self.compressedsimdata[field]["data"] = self.compressedsimdata[field]["data"][:self.N_unique]
                # keep a record of the field order, including fluxes
                self.stacked[i_field] = self.compressedsimdata[field]["data"]
                self.field_header[i_field]=field
                i_field+=1
            else:
                for flux_type in self.compressedsimdata["fluxes"].keys():
                    # keep a record of the field order, including fluxes
                    self.stacked[i_field] = self.compressedsimdata["fluxes"][flux_type]["data"]
                    self.field_header[i_field]=flux_type
                    # self.compressedsimdata["fluxes"][flux_type]["data"] = [:self.N_unique]
                    i_field+=1

        self.stacked = np.array(self.stacked).T
        if self.save_compressed3d is not False:
            np.save(file=self.outdir+self.gasspy_subdir+self.save_compressed3d+".npy", arr=self.stacked)

            # with open(self.outdir+"/"+self.save_compressed3d,'wb') as f:
            #     np.save(f, self.stacked)
            # Read:
            # with open(save_compressed3d,'rb') as f:
            #    np.load(f)
        
        pass


    def trim(self):
        """
        TRIM THE DATA!!!!
        The fields to store compressed data were done with arrays of length equal to the original data.
        The compression ensures that the length is <= len(original data)
        Thus we crop out the data that is not unique.
        """
        # Save the unique dictionary to a pickle
        if len(np.shape(self.stacked)) == 4:
            # Flatten the stacked image from 3d + N_fields to 1d + N_fields
            # This is required for pandas
            self.stacked = np.reshape(self.stacked,(np.prod(np.shape(self.stacked)[:-1]),np.shape(self.stacked)[-1]))
        N_original = np.shape(self.stacked)[0]

        #TODO: Replace pandas with a numpy.unique(axis=?)
        self.unique = pd.DataFrame(self.stacked).drop_duplicates()
        del(self.stacked)

        self.unique.columns = self.field_header


        self.unique = self.unique.reset_index(drop=True)
        self.unique.to_pickle(self.outdir+self.gasspy_subdir+self.outname+"_unique.pkl")

        # Save the flux definition to a yaml file
        flux_def = {}
        for field in self.compressedsimdata['fluxes'].keys():
            flux_def[field] = {}
            for fluxkey in self.compressedsimdata["fluxes"][field].keys():
                if fluxkey != "data":
                    flux_def[field][fluxkey] = self.compressedsimdata["fluxes"][field][fluxkey]
        with open(self.outdir+self.gasspy_subdir+self.outname+"_fluxdef.yaml",'w') as yamlfile:
            data = yaml.dump(flux_def,yamlfile)
            print("flux definition written to yaml file")

        del(self.compressedsimdata)
        self.N_unique = self.unique.shape[0]

        return(self.N_unique/self.N_cells)

class gasspy_to_cloudy(object):
    def __init__(self,
    outdir=None,
    outname=None,
    gasspy_subdir="GASSPY",
    fluxdef_file=None,
    unique_panda_pickle_file=None,
    ForceFullDepth=False,
    IF_ionfrac=0.1,
    **kwags):
        self.__dict__.update(defaults.parameters)

        self.MaxNumberModels = int(1e5)
        if outdir != None:
            self.outdir = outdir
        else:
            self.outdir = "./"

        if not gasspy_subdir.endswith("/"):
            gasspy_subdir = gasspy_subdir + "/"
        if not gasspy_subdir[0] == "/":
            gasspy_subdir = "/" + gasspy_subdir

        self.gasspy_subdir = gasspy_subdir

        if outname != None:
            self.outname = outname
        else:
            self.outname = "gasspy"

        if fluxdef_file != None:
            self.fluxdef_file = fluxdef_file
        else:
            self.fluxdef_file = "%s/%s_fluxdef.yaml"%(self.outdir,self.outname)
        
        if unique_panda_pickle_file != None:
            self.unique_panda_pickle_file = unique_panda_pickle_file
        else:
            self.unique_panda_pickle_file = "%s%s/%s_unique.pkl"%(self.outdir, self.gasspy_subdir, self.outname)

        self.IF_ionfrac=IF_ionfrac
        try:
            self.__dict__.update(kwags)
        except:
            pass
        
        self.unified_fluxes = False

        if self.unique_panda_pickle_file != None:
            self.unique_panda_pickle = pd.read_pickle(self.unique_panda_pickle_file)
        
        if self.fluxdef_file != None:
            with open(r'%s'%self.fluxdef_file) as file:
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                self.fluxdef = yaml.load(file, Loader=yaml.FullLoader)

        self.unify_flux_defs()

        self.ForceFullDepth=ForceFullDepth

    def unify_flux_defs(self):
        """ 
        All fluxes are to be converted to either Phi or Intensity.
        # Phi = photons per cm^2 per s
        # Intensity = erg per cm^2 per s        
        """

        for field in self.fluxdef:
            if "unit" in self.fluxdef[field]:
                if self.fluxdef[field]["unit"] not in ["phi", "intensity"]:
                    if isinstance(self.fluxdef[field]["unit"], str):
                        self.fluxdef[field]["unit"] = u.Unit(self.fluxdef[field]["unit"])
                    if u.Unit(self.fluxdef[field]["unit"]).is_equivalent(u.Unit("Hertz")):
                        self.unique_panda_pickle[field] +=  1/np.square(self.unique_panda_pickle['dx']) * u.Unit(self.fluxdef[field]["unit"]).to("Hertz")
                        self.fluxdef[field]["unit"] = "phi"
                    elif u.Unit(self.fluxdef[field]["unit"]).is_equivalent(u.Unit("1/(s*cm^2)")):
                        self.unique_panda_pickle[field] +=  u.Unit(self.fluxdef[field]["unit"]).to("1/(s*cm^2)")
                        self.fluxdef[field]["unit"] = "phi"
                    elif u.Unit(self.fluxdef[field]["unit"]).is_equivalent(u.Unit("erg/(s*cm^2)")):
                        self.unique_panda_pickle[field] +=  u.Unit(self.fluxdef[field]["unit"]).to("erg/(s*cm^2)")
                        self.fluxdef[field]["unit"] = "intensity"
                    elif u.Unit(self.fluxdef[field]["unit"]).is_equivalent(u.Unit("erg/s")):
                        self.unique_panda_pickle[field] +=  1/np.square(self.unique_panda_pickle['dx']) * u.Unit(self.fluxdef[field]["unit"]).to("erg/s")
                        self.fluxdef[field]["unit"] = "intensity"
            else:
                self.fluxdef[field]["unit"] = "phi"
                
        self.unified_fluxes = True


    def open_output(self, UniqID):
        """pass this a dictionary of parameters and values. Loop over and create prefix. Open output """
        try:
            os.stat(self.outdir+"/cloudy-output")
        except:
            os.mkdir(self.outdir+"/cloudy-output")

        prefix = "%s-%s"%(self.outname, UniqID)
        self.outfile = open(self.outdir+"/cloudy-output/"+prefix+".in", 'w')
        self.outfile.write("set save prefix \"%s\""%(prefix)+"\n")

    def check_for_if(self, cell_depth, cell_hden, cell_phi_ih, alpha=4.0e-13):
        """Check to see if cell contains a Hydrogen ionization front"""
        if self.ForceFullDepth is True:
            return True
        else:
            cell_depth = 10**cell_depth
            is_IF = False
            for cell_phi_ih_limit in cell_phi_ih:
                # alpha = 4.0e-13 # H recombinations per second cm-6
                ion_depth = cell_phi_ih_limit /(alpha * 10**(cell_hden)**2)

                # Change hardcoded 1e13 to mean free path of ionizing photon in H0.
                if ion_depth <= cell_depth and ion_depth/cell_depth > self.IF_ionfrac:
                    is_IF = True
            return is_IF

    def set_opacity_emiss(self, model_is_ionization_front):
        if model_is_ionization_front:
            """If an IF then we need to average over the entire cell"""
            self.outfile.write("save diffuse continuum last zone \".em\"\n")
            self.outfile.write("save opacity total last every \".opc\"\n")
        else:
            """If not an IF we use the last zone to estimate emissivity and opacity"""
            self.outfile.write("save diffuse continuum last \".em\"\n")
            self.outfile.write("save opacity total last \".opc\"\n")

        self.outfile.write("save opacities grain last \".grnopc\"\n")

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
            self.outfile.write("set nend 10\n")

    def set_temperature(self, log10_temperature, is_ionization_front, force_Teq=False, force_Tconst=False, T_law=False):
        """Set constant temperature if not modeling the actual ionization front temp gradients"""
        if T_law is False:
            if (is_ionization_front is False and force_Teq is False) or (is_ionization_front is True and force_Tconst is True):
                self.outfile.write("constant temperature linear %s\n"%(10**log10_temperature))
        else:
            sys.exit("T_law not implemented ... yet")

    def set_fsb99_phi_ih(self, phi_fsb99, SB99model="1e6cluster_norot_Z0014_BH120"):
        # "table star \"%s.mod\" age=%0.1f years \n" % (SB99model, np.max([SB99_age, i.SB99_age_min])))
        self.outfile.write("table star \"%s.mod\" age=%0.1e years \n" % (SB99model, 1e5))
        self.outfile.write("phi(h) = %s, range 1.0 to 3 Ryd\n"%(phi_fsb99))

    def set_fluxes(self, model, flux_definition):
        for field in flux_definition.keys():
            flux_value = model[field]
            flux_type = flux_definition[field]["unit"]
            EminRyd = flux_definition[field]['Emin']/13.6
            EmaxRyd = flux_definition[field]['Emax']/13.6
            if flux_definition[field]['shape'].endswith(".sed"):
                self.outfile.write("table SED \"%s\"\n"%flux_definition[field]['shape'])
            else:
                sys.exit("Currently only SEDs defined by a SED file are supported")

            if flux_type == "phi":
                self.outfile.write("%s(h) = %s, range %f to %f Ryd\n"%(flux_type, flux_value, EminRyd, EmaxRyd))
            else:
                self.outfile.write("%s = %s, range %f to %f Ryd\n"%(flux_type, flux_value, EminRyd, EmaxRyd))

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
            """
                When flux is defined as an intensity, explicitly calculating the number of ionizing photons requires using the SED.
                To speed that up we calculate a min and max number of photons based on the limits. At the moment since we are not accounting for the
                cross section of hydrogen in the IF calculation this is no more or less inaccurate.
            """
            _phi_ih = [0.0,0.0]
            for radfield in flux_definition.keys():
                if flux_definition[radfield]['Emin'] > 13.5984:
                    if self.fluxdef[radfield] == 'phi':
                        _phi_ih[0] += 10**model[radfield]
                        _phi_ih[1] += 10**model[radfield]
                    elif self.fluxdef[radfield] == 'intensity':
                        phi_from_min = 10**model[radfield] / (flux_definition[radfield]['Emin']*u.eV).to("erg").value
                        phi_from_max = 10**model[radfield] / (flux_definition[radfield]['Emax']*u.eV).to("erg").value
                        _phi_ih[0] += phi_from_min
                        _phi_ih[1] += phi_from_max
                
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

        self.set_opacity_emiss(isIF)
        """ Close input file """
        self.outfile.close()


    def make_user_seds(self):
        """
        make user defined seds using the energy bins defined by the user
        """
        try:
            os.stat(self.outdir+"/cloudy-output")
        except:
            os.mkdir(self.outdir+"/cloudy-output")

        for field in self.fluxdef.keys():
            if self.fluxdef[field]['shape'].endswith(".sed"):
                sedfile = self.outdir+"/cloudy-output/%s"%(self.fluxdef[field]['shape'])
                os.popen("cp %s %s"%(self.fluxdef[field]['shape'], sedfile))
            
            else:
                sedfile = self.outdir+"/cloudy-output/%s_%s.sed"%(self.save_prefix, field)
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

        self.make_user_seds()

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
