"""
    Modified version of gasspy/physics/sourcefunction_database/cloud/gasspy_to_cloudy
    which runs a single supplied model and processes it
"""

import os, sys
import subprocess
import traceback
import astropy.units as apyu
import astropy.constants as apyc
import numpy as np
import pandas
from gasspy.io.gasspy_io import read_fluxdef, read_yaml, check_parameter_in_config
import gasspy.shared_utils.mpi_utils.mpi_os as mpi_os
from gasspy.shared_utils.mpi_utils.mpi_print import mpi_print, mpi_all_print

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.Get_size()
class CloudyModelRunner():

    def __init__(self, gasspy_config, indir, fluxdef_file, 
        IF_ionfrac = None,
        cloudy_path = None,
        line_labels = None,
        lines_only = None):

        try:
            self.__init_inner__(gasspy_config, indir, fluxdef_file,
                                IF_ionfrac=IF_ionfrac,
                                cloudy_path=cloudy_path,
                                line_labels=line_labels,
                                lines_only = lines_only
                                )
        except:
            mpi_all_print(traceback.format_exc())
            mpi_comm.Abort(1) 

    def __init_inner__(self, gasspy_config, indir, fluxdef_file, 
        IF_ionfrac = None,
        cloudy_path = None,
        line_labels = None,
        lines_only = None):
        # Get the gasspy_config
        if isinstance(gasspy_config, str):
            self.gasspy_config = read_yaml(gasspy_config)
        else:
            self.gasspy_config = gasspy_config

        # Set directory where we will run the models and make sure it exists
        self.indir = indir
        if not os.path.exists(self.indir):
            mpi_os.mpi_makedirs(self.indir)
        mpi_comm.barrier()
        

        # Set ionization fraction limit to treat model as an ionization front
        self.IF_ionfrac = check_parameter_in_config(self.gasspy_config, "IF_ionfrac", IF_ionfrac, 0.01)

        # Set flux definition dictionary and parse units
        if isinstance(fluxdef_file, str):
            self.fluxdef = read_fluxdef(fluxdef_file)
        else:
            self.fluxdef = fluxdef_file
        self.unify_flux_defs()
        self.make_user_seds()

        # Set the default .ini file 
        self.get_ini_file()

        # Set the required fields for a given model
        self.required_fields = ["cell_size", "number_density", "temperature"]
        for field in self.fluxdef:
            self.required_fields.append(field)

        # Set the path to the cloudy exe and ensure that it exists
        self.cloudy_path = check_parameter_in_config(self.gasspy_config, "cloudy_path", cloudy_path, None)
        if self.cloudy_path is None:
            sys.exit("ERROR: No cloudy path supplied")

        self.cloudy_exe = self.cloudy_path + "/source/cloudy.exe"
        if not os.path.exists(self.cloudy_exe):
            sys.exit("Error: cloudy.exe could not be found in %s/source"%self.cloudy_path)

        # Set line labels if we want them
        self.line_labels = check_parameter_in_config(self.gasspy_config, "line_labels", line_labels, None)

        # Check if we only want to run with lines
        self.lines_only = check_parameter_in_config(self.gasspy_config, "lines_only", lines_only, False) 


        self.success = True
        self.read_line_data = False
        self.skip = False

        self.energy_bins = None
        self.delta_energy_bins = None

        self.total_depth = None
        self.delta_r = None
        self.n_zones = None

        # Run an easy model to check that everything works and load energy bins
        model_dict = {
            "number_density" : 3,
            "temperature" : 3,
            "cell_size" : 17,
        }
        for field in self.fluxdef:
            model_dict[field] = 0
        self.run_model(model_dict, "test_%d"%mpi_rank)
        if not self.model_successful():
            mpi_print("ERROR: model_runner test model failed")
            mpi_print("Test model parameters:")
            for key in model_dict:
                mpi_print("%s = %f"%(key,model_dict[key]))
            sys.exit("ERROR: model_runner test model failed")
        mpi_comm.barrier()
        self.read_ebins()
        self.delete_files()


    """
        These functions are called by gasspy and are needed in this explicit form 
    """

    def run_model(self, model, model_name):
        """
            Main function to set and run a given model
        """
        root_dir = os.getcwd()
        os.chdir(self.indir)
        
        self.model = model
        self.model_name = model_name

        # set if we want to enforce full ionization front calculation
        
        # Create in file
        self.create_in()

        # Run model
        out = open(model_name+".out", "w")
        with subprocess.Popen([self.cloudy_exe, model_name+".in"], stdout=subprocess.PIPE) as p:
            pass
            out.write(p.stdout.read().decode("utf-8"))
        out.close()

        # Reset quantities that were specific to the previous model
        self.total_depth = None
        self.delta_r = None
        self.n_zones = None

        os.chdir(root_dir)
        # Check for model failure

        self.success = self.check_success()

        self.skip = False
        return
    
    def model_successful(self):
        """
            Function to return success state of model
        """
        return self.success
    
    """
        Methods to get spectral data (energy, intensity opacity)
        Technically only needed if spectral data is utilized
    """

    def get_energy_bins(self):
        """
            Function to get the energy bins of the spectra
        """
        return self.energy_bins
    def get_delta_energy_bins(self):
        """
            Function to get the energy bins sizes of the spectra
        """
        return self.delta_energy_bins
            
    def get_intensity(self):
        """
            Function to return calculated intensity of model
        """
        return self.read_intensity()

    def get_opacity(self):
        """
            Function to return calculated opacity of model
        """      
        return self.read_opacity()
    
    """
        Methods to get line data (energy, label, intensity opacity)
        Technically only needed if line data is utilized
    """
    def get_line_energies(self):
        # Conversion factors as defined in Cloudy (NOTE: if these change in Cloudy these should also change here)
        Ang2Ryd = 1e8/1.0973731568160e5
        Micro2Ryd = 1e4/1.0973731568160e5
    
        line_energies = np.zeros(len(self.line_labels))

        # Loop over line labels and determine the energies in Rydberg
        # NOTE: Hopefully in the future this could be given by Cloudy itself to reduce possibility of errors
        for iline, label in enumerate(self.line_labels):
            wavelength_str = label.split(" ")[-1]
            if wavelength_str.endswith("A"):
                line_energies[iline] = Ang2Ryd/float(wavelength_str.strip("A"))
            elif wavelength_str.endswith("m"):
                line_energies[iline] = Micro2Ryd/float(wavelength_str.strip("m"))     
        return line_energies
    
    def get_line_labels(self):
        # In Cloudy's case the line labels are in the same order as provided
        return self.line_labels

    def get_line_intensity(self):
        """
            Function to return calculated line intensity of model
        """
        return self.read_line_intensity()

    def get_line_opacity(self):
        """
            Function to return calculated line opacity of model
        """      
        return self.read_line_opacity()
  
    def clean_model(self):
        """
            Function to clean model specific data no longer needed after a model has been processed
        """
        self.delete_files()
        return
    

    """
        Methods for reading cloudy output files 
    """
    def check_success(self, suff = ".spec_em"):
        filename = self.indir + "/%s"%self.model_name+suff
        if not os.path.exists(filename) :
            return False
        return os.path.getsize(filename) > 1000

    def read_ebins(self, suff = ".ebins"):
        filename = self.indir + "/%s"%self.model_name+suff
        if not os.path.exists(filename):
            if (self.energy_bins is None or self.delta_energy_bins is None):
                sys.exit("Error: .ebins file not found and energy bins has not been read by a previous model")
            return
        
        if self.n_zones is None:
            self.read_mol()
        df = pandas.read_csv(filename, delimiter ="\t", usecols=["Anu/Ryd", "d(anu)/Ryd"], na_filter = False, dtype = np.float64, low_memory = False)
        self.n_energy_bins = len(df)//self.n_zones - 1
        self.energy_bins = df["Anu/Ryd"].to_numpy()[:self.n_energy_bins]
        self.delta_energy_bins = df["d(anu)/Ryd"].to_numpy()[:self.n_energy_bins]

    def read_mol(self, suff=".mol"):
        if not self.success:
            return 0 
        """Read the molecular data file, mostly to get out the depth array of the model"""
        filename = self.indir + "/%s"%self.model_name+suff
        with open(filename,"r") as f:
            if len(f.readlines()) < 2:
                self.skip = True

        if not self.skip:
            data = pandas.read_csv(filename, delimiter="\t", usecols=["#depth"], na_filter=False, dtype=np.float64, low_memory=False)

            depth = np.asarray(data["#depth"])
            self.total_depth = np.array([np.sum(depth),])

            self.n_zones = len(depth)

            depth[1:] = depth[1:] - depth[:-1]
            self.delta_r = depth

            return 0
        else:
            return 1


    
    def read_intensity(self, suff = ".spec_em"):
        if not self.success:
            return 0 
        filename = self.indir+"/%s"%self.model_name+suff
        #if self.isIF:
        if self.delta_r is None or self.total_depth is None:
            self.read_mol()
        mydf = pandas.read_csv(
            filename,
            delimiter="\t",
            comment="#",
            header=None, na_filter=False, dtype=np.float64, low_memory=False)

        data = np.array(mydf)

        if len(data[0,:]) > 0 and self.energy_bins is None:
            self.read_ebins()

        data = data[1:,:]
        avg_em = (data.T * self.delta_r).sum(axis=1)/float(self.total_depth)/(4*np.pi*self.energy_bins)
        
        return avg_em     

    def read_opacity(self, suff=".spec_op"):
        if not self.success:
            return 0 
        """ Read the multizone opacity spectrum"""
        filename = self.indir+"/%s"%self.model_name+suff

        mydf = pandas.read_csv(filename, delimiter="\t",skip_blank_lines=True, usecols=["Tot opac"], dtype=np.float64, low_memory=False, na_filter=False)
        if self.energy_bins is None and len(mydf["Tot opac"]) > 0 : 
            self.read_ebins()

        if self.delta_r is None or self.total_depth is None:
            self.read_mol()


        tau = (mydf["Tot opac"].iloc[0:self.n_energy_bins] * float(self.delta_r[0])).to_numpy()

        for izone in range(1, self.n_zones):
            tau[:] += (mydf["Tot opac"].iloc[izone*self.n_energy_bins:(izone+1)*self.n_energy_bins] * self.delta_r[izone]).to_numpy()[:]
        
        tot_opc = tau / float(self.total_depth)

        return tot_opc
        
    def read_line_intensity(self, suff = ".line_em"): 
        if not self.success:
            return 0 
        if not self.read_line_data: 
            return None 
        if self.delta_r is None or self.total_depth is None:
            self.read_mol()
        filename = self.indir+"/%s"%self.model_name+suff

        df = pandas.read_csv(filename, delimiter = "\t") 
        if self.line_labels is None: 
            self.line_labels = df.columns[1:].to_list() 
 
        line_em  = df.to_numpy()[:,1:].T 
        avg_line_em = np.sum(line_em*self.delta_r, axis = 1)/float(self.total_depth)/(4*np.pi)
        return avg_line_em 
 
    def read_line_opacity(self, suff = ".line_op"): 
        if not self.success:
            return 0 
        if not self.read_line_data: 
            return None 
        if self.delta_r is None or self.total_depth is None:
            self.read_mol()
        filename = self.indir+"/%s"%self.model_name+suff
        df = pandas.read_csv(filename, delimiter = "\t") 
        if self.line_labels is None: 
            self.line_labels = df.columns[1:].to_list() 
 
        line_op  = df.to_numpy()[-1,1:].T 
        avg_line_op = line_op/float(self.total_depth) 
        return avg_line_op

    def get_error(self, suff = ".out"):
        problems = []
        filename = self.indir+"/%s"%self.model_name+suff
        with open(filename,"r") as f:
            lines = f.readlines()
            if not lines[-1].endswith("something went wrong]\n"):
                print("%s appears to have run correctly but produced unusable results... please check what happened"%filename)
            for line in lines:
                if line.startswith(" PROBLEM"):
                    problem_line = line.strip("\n").strip(" PROBLEM ")
                    if problem_line not in problems:
                        problems.append(problem_line)
        return problems

    

    def read_all(self, modelname):
        self.read_mol(modelname)
        if self.energy_bins is None:
            self.read_ebins(modelname)
        self.avg_spec_em = self.read_intensity(modelname)
        self.avg_spec_op = self.read_opacity(modelname)
        self.avg_line_em = self.read_line_intensity(modelname)
        self.avg_line_op = self.read_line_opacity(modelname)


    """
        Methods for setting properties for all in files
    """
        

    def get_ini_file(self):

        assert "cloudy_ini" in self.gasspy_config, "Need and .ini file to run cloudy"
        path = self.gasspy_config["cloudy_ini"]
        self.CLOUDY_INIT_FILE = path.split("/")[-1] 
        mpi_os.mpi_copy(path, self.indir + "/" + self.CLOUDY_INIT_FILE)
        return


    def unify_flux_defs(self):
        """ 
        All fluxes are to be converted to either Phi or Intensity. Figure out different time and energy units
        Conversion to intensive variables happen later
        # Phi = photons per cm^2 per s
        # Intensity = erg per cm^2 per s        
        """
        for field in self.fluxdef:
            if "unit" in self.fluxdef[field]:
                if self.fluxdef[field]["unit"] not in ["phi", "intensity", "photon_count", "luminosity"]:
                    if isinstance(self.fluxdef[field]["unit"], str):
                        self.fluxdef[field]["unit"] = apyu.Unit(self.fluxdef[field]["unit"])
                    if apyu.Unit(self.fluxdef[field]["unit"]).is_equivalent(apyu.Unit("Hertz")):
                        self.fluxdef[field]["conversion"] =  apyu.Unit(self.fluxdef[field]["unit"]).to("Hertz").value
                        self.fluxdef[field]["unit"] = "photon_count"
                    elif apyu.Unit(self.fluxdef[field]["unit"]).is_equivalent(apyu.Unit("1/(s*cm^2)")):
                        self.fluxdef[field]["conversion"] +=  apyu.Unit(self.fluxdef[field]["unit"]).to("1/(s*cm^2)").value
                        self.fluxdef[field]["unit"] = "phi"
                    elif apyu.Unit(self.fluxdef[field]["unit"]).is_equivalent(apyu.Unit("erg/(s*cm^2)")):
                        self.fluxdef[field]["conversion"] +=  apyu.Unit(self.fluxdef[field]["unit"]).to("erg/(s*cm^2)").value
                        self.fluxdef[field]["unit"] = "intensity"
                    elif apyu.Unit(self.fluxdef[field]["unit"]).is_equivalent(apyu.Unit("erg/s")):
                        self.fluxdef[field]["conversion"] +=  apyu.Unit(self.fluxdef[field]["unit"]).to("erg/s").value
                        self.fluxdef[field]["unit"] = "luminosity"
                self.fluxdef[field]["conversion"] = 1
            else:
                self.fluxdef[field]["unit"] = "phi"
                self.fluxdef[field]["conversion"] = 1
                
        self.unified_fluxes = True


    def make_user_seds(self):
        """
        make user defined seds using the energy bins defined by the user
        """
        for field in self.fluxdef.keys():
            if self.fluxdef[field]['shape'].endswith(".sed"):
                if mpi_rank != 0:
                    continue
                sedfile = self.indir + "/%s"%(self.fluxdef[field]['shape'])
                os.popen("cp %s %s"%(self.fluxdef[field]['shape'], sedfile))
            
            else:
                sedfile = self.indir + "/%s.sed"%(self.save_prefix, field)
                self.fluxdef[field]["shape"] = sedfile

                if mpi_rank != 0:
                    return
                
                outfile = open(sedfile, 'w')
                if self.fluxdef["fluxes"][field]["shape"] == 'const':
                    outfile.write("%f -35.0 nuFnu\n"%(self.fluxdef[field]["Emin"]*0.99/13.6) )
                    outfile.write("%f 1.000 nuFnu\n"%(self.fluxdef[field]["Emin"]/13.6) )
                    outfile.write("%f 1.000 nuFnu\n"%(self.fluxdef[field]["Emax"]/13.6) )
                    outfile.write("%f -35.0 nuFnu\n"%(self.fluxdef[field]["Emax"]*1.01/13.6) )
                outfile.close()


    """
        Methods for setting parameters for the current model
    """
    def create_in(self, model = None, model_name = None, force_Teq = False):
        """
            Main function to create the .in file
        """
        if model is not None:
            self.model = model
        if model_name is not None:
            self.model_name = model_name

        self.outfile = open(self.model_name+".in", "w")
        self.outfile.write("set save prefix \"%s\"\n"%self.model_name)

        _phi_ih = 99.99

        """
            When flux is defined as an intensity, explicitly calculating the number of ionizing photons requires using the SED.
            To speed that up we calculate a min and max number of photons based on the limits. At the moment since we are not accounting for the
            cross section of hydrogen in the IF calculation this is no more or less inaccurate.
        """
        _phi_ih = [0.0,0.0]
        for field in self.fluxdef.keys():
            if self.fluxdef[field]['Emin'] > 13.5984:
                if self.fluxdef[field]["unit"] == 'phi':
                    _phi_ih[0] += 10**self.model[field] * self.fluxdef[field]["conversion"]
                    _phi_ih[1] += 10**self.model[field] * self.fluxdef[field]["conversion"]
                elif self.fluxdef[field]["unit"] == "photon_count":
                    _phi_ih[0] += 10**(self.model[field] - 2*self.model["cell_size"]) * self.fluxdef[field]["conversion"]
                    _phi_ih[1] += 10**(self.model[field] - 2*self.model["cell_size"])* self.fluxdef[field]["conversion"]
                elif self.fluxdef[field] == 'intensity':
                    phi_from_min = 10**self.model[field] / (self.fluxdef[field]['Emin']*apyu.eV).to("erg").value
                    phi_from_max = 10**self.model[field] / (self.fluxdef[field]['Emax']*apyu.eV).to("erg").value
                    _phi_ih[0] += phi_from_min
                    _phi_ih[1] += phi_from_max
            
        self.isIF = self.check_for_if(self.model["cell_size"], self.model["number_density"], _phi_ih)


        """ Set common init file """
        self.set_cloudy_init_file()

        """ Write individual cloudy parameters to input file """
        self.set_depth(self.model["cell_size"])
        self.set_hden(self.model["number_density"])
        self.set_nend(self.isIF)
        self.set_temperature(self.model["temperature"], self.isIF, force_Teq=force_Teq)

        self.set_fluxes(self.model)

        self.set_spec_opacity_emiss(self.isIF)
        if self.line_labels is not None:
            self.set_line_opacity_emiss()
            self.read_line_data = True
        else:
            self.read_line_data = False

        """ Close input file """
        self.outfile.close()



    def check_for_if(self, cell_depth, cell_hden, cell_phi_ih, alpha=4.0e-13):
        """Check to see if cell contains a Hydrogen ionization front"""

        cell_depth = 10**cell_depth
        is_IF = False
        for cell_phi_ih_limit in cell_phi_ih:
            # alpha = 4.0e-13 # H recombinations per second cm-6
            ion_depth = cell_phi_ih_limit /(alpha * (10**(cell_hden))**2)
            # Change hardcoded 1e13 to mean free path of ionizing photon in H0.
            if ion_depth <= cell_depth and ion_depth/cell_depth > self.IF_ionfrac:
                is_IF = True
        return is_IF

    def set_spec_opacity_emiss(self, force_continuum_bins = False):
        if force_continuum_bins or self.energy_bins is None:
            self.outfile.write("save continuum bins \".ebins\" last\n")
        #if model_is_ionization_front:
        #    """If an IF then we need to average over the entire cell"""
        self.outfile.write("save diffuse continuum last zone \".spec_em\"\n")
        self.outfile.write("save opacity total last every\".spec_op\"\n")
        #else:
        #    """If not an IF we use the last zone to estimate emissivity and opacity"""
        #    self.outfile.write("save diffuse continuum last \".spec_em\"\n")
        #    self.outfile.write("save opacity total last \".spec_op\"\n")

    def set_line_opacity_emiss(self):
        # Emissivity
#        if model_is_ionization_front:
        self.outfile.write("save lines, emissivity, \".line_em\" last no hash\n")
#        else:
        #self.outfile.write("save lines, emissivity, \".line_em\" last no hash\n")
        for line in self.line_labels:
            self.outfile.write("%s\n"%line)
        self.outfile.write("end of lines\n")

        # Opacity
        #if model_is_ionization_front:
        self.outfile.write("save lines, optical some, \".line_op\" last no hash\n")
        #else:
        #self.outfile.write("save lines, optical some, \".line_op\" last no hash\n")
        for line in self.line_labels:
            self.outfile.write("%s\n"%line)
        self.outfile.write("end of lines\n")

    def set_cloudy_init_file(self):
        """ Write command to cloudy input file to use custom init file """
        self.outfile.write("init \"%s\"\n"%(self.CLOUDY_INIT_FILE))

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

        #if model_is_ionization_front is False:
            #Set constant temperature if IF does not exist
            #self.outfile.write("set nend 100\n")

    def set_temperature(self, temperature, is_ionization_front, force_Teq=False, force_Tconst=False, T_law=False):
        """Set constant temperature if not modeling the actual ionization front temp gradients"""
        if T_law is False:
            if (is_ionization_front is False and force_Teq is False) or (is_ionization_front is True and force_Tconst is True):
                self.outfile.write("constant temperature linear %s\n"%(10**temperature))
        else:
            sys.exit("T_law not implemented ... yet")


    def set_fluxes(self, model):
        for field in self.fluxdef.keys():
            flux_value = model[field]+np.log10(self.fluxdef[field]["conversion"])
            if self.fluxdef[field]["unit"] in ["luminosity", "photon_count"]:
                flux_value -= 2*model["cell_size"]
            
            flux_type = self.fluxdef[field]["unit"]
            if flux_type == "luminosity":
                flux_type = "intensity"

            if flux_type == "photon_count":
                flux_type = "phi"

            EminRyd = self.fluxdef[field]['Emin']/apyu.rydberg.to("eV")
            EmaxRyd = self.fluxdef[field]['Emax']/apyu.rydberg.to("eV")
            if self.fluxdef[field]['shape'].endswith(".sed"):
                self.outfile.write("table SED \"%s\"\n"%self.fluxdef[field]['shape'])
            else:
                sys.exit("Currently only SEDs defined by a SED file are supported")

            if flux_type == "phi":
                self.outfile.write("%s(h) = %s, range %f to %f Ryd\n"%(flux_type, flux_value, EminRyd, EmaxRyd))
            else:
                self.outfile.write("%s = %s, range %f to %f Ryd\n"%(flux_type, flux_value, EminRyd, EmaxRyd))



    """
        Deleting files
    """
    def delete_files(self, delete_input = True):
        """
            deletes all files associated exclusively with the current model
            arguments:
                delete_input: boolian (Flag for deleting input file along with the outputs)
        """
        suffixes = [".ebins", ".spec_em", ".spec_op", ".line_em", ".line_op", ".mol",".out"]
        for suff in suffixes:
            filename = self.indir + "/%s"%self.model_name + suff
            self.__delete_file__(filename)
        if delete_input:
            filename = self.indir + "/%s"%self.model_name + ".in"
            self.__delete_file__(filename)

    def __delete_file__(self, filename):
        """
            deletes file IF it exists
        """
        if os.path.exists(filename):
            os.remove(filename)



    """

        Methods for handling failed models
    
    """

    def handle_model_failure(self, cloudy_path):
        problems = self.get_error()

        convergence_error = False

        for problem in problems:
            if "not converged" in problem:
                convergence_error = True

        if convergence_error:
            # See if we should rerun this model without fixed temperature 
            if "failure_unfix_temperature_in_ranges" in self.gasspy_config:
                ranges = self.gasspy_config["failure_unfix_temperature_in_ranges"]
                rerun = True
                
                # loop over all supplied model parameters
                for key in ranges.keys():
                    if 10**self.model[key] < ranges[key][0] or self.model[key] > ranges[key][1]:  
                        rerun = False         
                
                # If model lies within the ranges, rerun but force Teq
                if rerun:
                    root_dir = os.getcwd()
                    os.chdir(self.indir)
                    self.create_in(force_Teq= True)

                    # Run model
                    cloudy_exe = cloudy_path + "/source/cloudy.exe"
                    out = open(self.model_name+".out", "w")
                    with subprocess.Popen([cloudy_exe, self.model_name+".in"], stdout=subprocess.PIPE) as p:
                        pass
                        out.write(p.stdout.read().decode("utf-8"))
                    out.close()

                    # Reset quantities that were specific to the previous model
                    self.total_depth = None
                    self.delta_r = None
                    self.n_zones = None
                
                    os.chdir(root_dir)

                    # recheck for model failure
                    self.success = self.check_success()

                    if self.model_successful:
                        problems = ["Convergence error - fixed with Teq"]

        return problems
