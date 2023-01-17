"""
    Modified version of gasspy/physics/sourcefunction_database/cloud/gasspy_to_cloudy
    which runs a single supplied model and processes it
"""

import os, sys
import subprocess
import shutil, glob, pathlib
import astropy.units as apyu
import numpy as np
import pandas

from gasspy.io.gasspy_io import read_fluxdef, read_yaml

class cloudy_model_runner():

    def __init__(self, gasspy_config, indir, fluxdef_file, 
        force_full_depth_all = False,
        IF_ionfrac = 0.01):

        self.energy_bins = None
        self.denergy_bins = None

        self.total_depth = None
        self.delta_r = None
        self.n_zones = None


        self.indir = indir
        if isinstance(gasspy_config, str):
            self.gasspy_config = read_yaml(gasspy_config)
        else:
            self.gasspy_config = gasspy_config
        self.force_full_depth_all = force_full_depth_all
        self.force_full_depth = self.force_full_depth_all
        self.IF_ionfrac = IF_ionfrac

        # Make sure that the directory exists
        try:
            os.stat(self.indir)
        except:
            os.mkdir(self.indir)


        # Set flux definition dictionary and parse units
        self.fluxdef = read_fluxdef(fluxdef_file)
        self.unify_flux_defs()
        self.make_user_seds()

        # Set the default .ini file 
        self.get_ini_file()
        self.line_labels = None

        self.model_successful = True
        self.read_line_data = False
        self.skip = False

    def run_single_model(self, model, cloudy_path, model_name = "gasspy", linelist = None, force_full_depth = False):
        root_dir = os.getcwd()
        os.chdir(self.indir)
        
        # set if we want to enforce full ionization front calculation
        self.force_full_depth = force_full_depth or self.force_full_depth_all
        
        # Create in file
        self.create_in(model, model_name=model_name, linelist=linelist)

        # Run model
        cloudy_exe = cloudy_path + "/source/cloudy.exe"
        out = open(model_name+".out", "w")
        with subprocess.Popen([cloudy_exe, model_name+".in"], stdout=subprocess.PIPE) as p:
            pass
            out.write(p.stdout.read().decode("utf-8"))
        out.close()

        # Reset quantities that were specific to the previous model
        self.total_depth = None
        self.delta_r = None
        self.n_zones = None

        os.chdir(root_dir)
        # Check for model failure

        self.model_successful = self.check_success()

        self.skip = False
        return

    """
        Methods for reading cloudy output files 
    """
    def check_success(self, suff = ".spec_em"):
        filename = self.indir + "/%s"%self.model_name+suff
        return os.path.getsize(filename) > 1000

    def read_ebins(self, suff = ".ebins"):
        filename = self.indir + "/%s"%self.model_name+suff
        if self.n_zones is None:
            self.read_mol()
        df = pandas.read_csv(filename, delimiter ="\t", usecols=["Anu/Ryd", "d(anu)/Ryd"], na_filter = False, dtype = np.float64, low_memory = False)
        self.n_energy_bins = len(df)//self.n_zones - 1
        self.energy_bins = df["Anu/Ryd"].to_numpy()[:self.n_energy_bins]
        self.denergy_bins = df["d(anu)/Ryd"].to_numpy()[:self.n_energy_bins]

    def read_mol(self, suff=".mol"):
        if not self.model_successful:
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


    def read_spec_em(self, suff = ".spec_em"):
        if not self.model_successful:
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
        
        #else:
        #    mydf = pandas.read_csv(
        #        filename,
        #        delimiter="\t",
        #        usecols=["#energy/Ryd", "Total"], na_filter=False, dtype=np.float64, low_memory=False)
        #    data = np.array(mydf)
        #    if len(data[:,0]) > 0 and self.energy_bins is None:
        #        self.read_ebins()
        
        #    avg_em = data[:,1].reshape(len(self.energy_bins))
        
        return avg_em     

    def read_spec_op(self, suff=".spec_op"):
        if not self.model_successful:
            return 0 
        """ Read the multizone opacity spectrum"""
        filename = self.indir+"/%s"%self.model_name+suff

        mydf = pandas.read_csv(filename, delimiter="\t",skip_blank_lines=True, usecols=["Tot opac"], dtype=np.float64, low_memory=False, na_filter=False)
        if self.energy_bins is None and len(mydf["Tot opac"]) > 0 : 
            self.read_ebins()

        #if self.isIF:
        if self.delta_r is None or self.total_depth is None:
            self.read_mol()
        #del mydf["#nu/Ryd"], mydf["elem"], mydf["Albedo"]


        tau = (mydf["Tot opac"].iloc[0:self.n_energy_bins] * float(self.delta_r[0])).to_numpy()

        for izone in range(1, self.n_zones):
            tau[:] += (mydf["Tot opac"].iloc[izone*self.n_energy_bins:(izone+1)*self.n_energy_bins] * self.delta_r[izone]).to_numpy()[:]
        
        tot_opc = tau / float(self.total_depth)

        #else:
        #    #nu/Ryd	Tot opac	Abs opac	Scat opac	Albedo	elem
        #    tot_opc = np.asarray(mydf["Tot opac"])

        return tot_opc
        
    def read_line_em(self, suff = ".line_em"): 
        if not self.model_successful:
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
 
    def read_line_op(self, suff = ".line_op"): 
        if not self.model_successful:
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
        filename = self.indir+"/%s"%self.model_name+suff
        with open(filename,"r") as f:
            lines = f.readlines()
            if not lines[-1].endswith("something went wrong]\n"):
                print("%s appears to have run correctly but produced unusable results... please check what happened"%filename)
            for line in lines:
                if line.startswith(" PROBLEM"):
                    problem_line = line.strip("\n").strip(" PROBLEM ")
                    if problem_line in self.problem_dict.keys():
                        self.problem_dict[problem_line].append(int(filename.split("-")[1]))
                    else:
                        self.problem_dict[problem_line] = [int(filename.split("-")[1])]
    

    def read_all(self, modelname):
        if True:
        #try:
            self.read_in(modelname)
            self.read_mol(modelname)
            if self.energy_bins is None:
                self.read_ebins(modelname)
            self.avg_spec_em = self.read_spec_em(modelname)
            self.avg_spec_op = self.read_spec_op(modelname)
            self.avg_line_em = self.read_line_em(modelname)
            self.avg_line_op = self.read_line_op(modelname)

       # except:
        else:
            self.get_error(name)
            self.skip = True


    """
        Methods for setting properties for all in files
    """
        

    def get_ini_file(self):

        assert "cloudy_ini" in self.gasspy_config, "Need and .ini file to run cloudy"
        path = self.gasspy_config["cloudy_ini"]
        self.CLOUDY_INIT_FILE = path.split("/")[-1] 
        shutil.copy(path, self.indir + "/" + self.CLOUDY_INIT_FILE)
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
                sedfile = self.indir + "/%s"%(self.fluxdef[field]['shape'])
                os.popen("cp %s %s"%(self.fluxdef[field]['shape'], sedfile))
            
            else:
                sedfile = self.indir + "/%s.sed"%(self.save_prefix, field)
                outfile = open(sedfile, 'w')
                if self.fluxdef["fluxes"][field]["shape"] == 'const':
                    outfile.write("%f -35.0 nuFnu\n"%(self.fluxdef[field]["Emin"]*0.99/13.6) )
                    outfile.write("%f 1.000 nuFnu\n"%(self.fluxdef[field]["Emin"]/13.6) )
                    outfile.write("%f 1.000 nuFnu\n"%(self.fluxdef[field]["Emax"]/13.6) )
                    outfile.write("%f -35.0 nuFnu\n"%(self.fluxdef[field]["Emax"]*1.01/13.6) )
                outfile.close()
                self.fluxdef[field]["shape"] = sedfile


    """
        Methods for setting parameters for the current model
    """
    def create_in(self, model, model_name = "gasspy", linelist = None):
        """
            Main function to create the .in file
        """
        self.model_name = model_name
        self.outfile = open(self.model_name+".in", "w")
        self.outfile.write("set save prefix \"%s\"\n"%model_name)

        _phi_ih = 99.99

        if self.force_full_depth is False:
            """
                When flux is defined as an intensity, explicitly calculating the number of ionizing photons requires using the SED.
                To speed that up we calculate a min and max number of photons based on the limits. At the moment since we are not accounting for the
                cross section of hydrogen in the IF calculation this is no more or less inaccurate.
            """
            _phi_ih = [0.0,0.0]
            for field in self.fluxdef.keys():
                if self.fluxdef[field]['Emin'] > 13.5984:
                    if self.fluxdef[field]["unit"] == 'phi':
                        _phi_ih[0] += 10**model[field] * self.fluxdef[field]["conversion"]
                        _phi_ih[1] += 10**model[field] * self.fluxdef[field]["conversion"]
                    elif self.fluxdef[field]["unit"] == "photon_count":
                        _phi_ih[0] += 10**(model[field] - 2*model["dx"]) * self.fluxdef[field]["conversion"]
                        _phi_ih[1] += 10**(model[field] - 2*model["dx"])* self.fluxdef[field]["conversion"]
                    elif self.fluxdef[field] == 'intensity':
                        phi_from_min = 10**model[field] / (self.fluxdef[field]['Emin']*apyu.eV).to("erg").value
                        phi_from_max = 10**model[field] / (self.fluxdef[field]['Emax']*apyu.eV).to("erg").value
                        _phi_ih[0] += phi_from_min
                        _phi_ih[1] += phi_from_max
                
            self.isIF = self.check_for_if(model["dx"], model["dens"], _phi_ih)
        else:
            self.isIF = True

        """ Set common init file """
        self.set_cloudy_init_file()

        """ Write individual cloudy parameters to input file """
        self.set_depth(model["dx"])
        self.set_hden(model["dens"])
        self.set_nend(self.isIF)
        self.set_temperature(model["temp"], self.isIF)

        self.set_fluxes(model)

        self.set_spec_opacity_emiss(self.isIF)
        if linelist is not None:
            self.set_line_opacity_emiss(linelist, self.isIF)
            self.read_line_data = True
        else:
            self.read_line_data = False

        """ Close input file """
        self.outfile.close()



    def check_for_if(self, cell_depth, cell_hden, cell_phi_ih, alpha=4.0e-13):
        """Check to see if cell contains a Hydrogen ionization front"""
        if self.force_full_depth is True:
            return True
        else:
            cell_depth = 10**cell_depth
            is_IF = False
            for cell_phi_ih_limit in cell_phi_ih:
                # alpha = 4.0e-13 # H recombinations per second cm-6
                ion_depth = cell_phi_ih_limit /(alpha * (10**(cell_hden))**2)

                # Change hardcoded 1e13 to mean free path of ionizing photon in H0.
                if ion_depth <= cell_depth and ion_depth/cell_depth > self.IF_ionfrac:
                    is_IF = True
            return is_IF

    def set_spec_opacity_emiss(self, model_is_ionization_front):
        self.outfile.write("save continuum bins \".ebins\" last\n")
        #if model_is_ionization_front:
        #    """If an IF then we need to average over the entire cell"""
        self.outfile.write("save diffuse continuum last zone \".spec_em\"\n")
        self.outfile.write("save opacity total last every\".spec_op\"\n")
        #else:
        #    """If not an IF we use the last zone to estimate emissivity and opacity"""
        #    self.outfile.write("save diffuse continuum last \".spec_em\"\n")
        #    self.outfile.write("save opacity total last \".spec_op\"\n")

    def set_line_opacity_emiss(self, linelist, model_is_ionization_front):
        # Emissivity
#        if model_is_ionization_front:
        self.outfile.write("save lines, emissivity, \".line_em\" last no hash\n")
#        else:
        #self.outfile.write("save lines, emissivity, \".line_em\" last no hash\n")
        for line in linelist:
            self.outfile.write("%s\n"%line)
        self.outfile.write("end of lines\n")

        # Opacity
        #if model_is_ionization_front:
        self.outfile.write("save lines, optical some, \".line_op\" last no hash\n")
        #else:
        #self.outfile.write("save lines, optical some, \".line_op\" last no hash\n")
        for line in linelist:
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

    def set_temperature(self, log10_temperature, is_ionization_front, force_Teq=False, force_Tconst=False, T_law=False):
        """Set constant temperature if not modeling the actual ionization front temp gradients"""
        if T_law is False:
            if (is_ionization_front is False and force_Teq is False) or (is_ionization_front is True and force_Tconst is True):
                self.outfile.write("constant temperature linear %s\n"%(10**log10_temperature))
        else:
            sys.exit("T_law not implemented ... yet")


    def set_fluxes(self, model):
        for field in self.fluxdef.keys():
            if model[field] <= self.gasspy_config["log10_flux_low_limit"][field]:
                continue
            flux_value = model[field]+np.log10(self.fluxdef[field]["conversion"])
            if self.fluxdef[field]["unit"] in ["luminosity", "photon_count"]:
                flux_value -= 2*model["dx"]
            
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