"""Read and collect cloudy models"""
import pickle
from re import M
import pandas
import numpy as np
import cupy
from pathlib import Path
import gc
import os 
import glob
import sys
import h5py as hp

from gasspy.shared_utils import loop_progress

class ModelCollector():
    """Worker class for reading and collecting cloudy models into a gasspy db entry"""
    def __init__(
        self, cloudy_dir="cloudy_output",
        out_dir="GASSPY/",
        db_name="gasspy_database.hdf5",
        all_opacities=True,
        clear_energy=False,
        single_files=False,
        out_files = {
            "avg_em": True,
            "grn_opc": True,
            "tot_opc": True,
            "mol":True
        },
        maxMem_GB = 30,
        recollect_all = False
    ):  

        self.clear_energy = clear_energy

        self.single_files = single_files

        assert type(out_dir) == str, "out_dir not a string"
        if not out_dir.endswith("/"):
            out_dir = out_dir + "/"
        self.out_dir = os.path.abspath(out_dir)
        print(self.out_dir)

        if not cloudy_dir.endswith("/"):
            cloudy_dir = cloudy_dir + "/" 
        self.cloudy_dir = cloudy_dir


        self.db_name = db_name

        self.all_opacities = all_opacities
        self.energy_limits = None

        self.opacity_everyzone = None
        self.em_everyzone = None

        self.avg_em = None
        self.tot_opc = None
        self.energy_bins = None
        self.n_energy_bins = None

        self.delta_r = None
        self.total_depth = None

        self.n_zones = None

        self.skip = False
        self.out_files = out_files

        self.maxMem_GB = maxMem_GB

        self.parital_saves = False
        self.recollect_all = recollect_all
    def set_maxfiles_at_a_time(self):
        mem_per_file = 0 
        nfluxes = len(self.energy_bins)
        # Add each saved variable
        for ftype in ["avg_em", "grn_opc", "tot_opc"]:
            if self.out_files[ftype]:
                mem_per_file += 64*nfluxes 
        self.maxfiles_at_a_time = int(self.maxMem_GB*1024**3 * 8 / mem_per_file)
    def read_em(self, filename, suff=".em"):
        """ Read the multizone emissivity spectrum """
        if not self.out_files["avg_em"]:
            return

        if self.em_everyzone:

            mydf = pandas.read_csv(
                self.cloudy_dir+filename+suff,
                delimiter="\t",
                comment="#",
                header=None, na_filter=False, dtype=np.float64, low_memory=False)

            data = np.array(mydf)

            if self.energy_bins is None:
                self.energy_bins = data[0,:]

            data = data[1:,:]
            self.avg_em = (data.T * self.delta_r).sum(axis=1)/float(self.total_depth)
        
        else:
            mydf = pandas.read_csv(
                self.cloudy_dir+filename+suff,
                delimiter="\t",
                usecols=["#energy/Ryd", "Total"], na_filter=False, dtype=np.float64, low_memory=False)

            if self.energy_bins is None:
                self.energy_bins = np.array(mydf["#energy/Ryd"])
            self.avg_em = np.array(mydf["Total"])
        
        if self.n_energy_bins is None:
            self.n_energy_bins = len(self.energy_bins)

        return 0

    def read_mol(self, filename, suff=".mol"):
        """Read the molecular data file, mostly to get out the depth array of the model"""
        if not self.out_files["mol"]:
            return
        with open(self.cloudy_dir+filename+suff,"r") as f:
            if len(f.readlines()) < 2:
                self.skip = True

        if not self.skip:
            data = pandas.read_csv(self.cloudy_dir+filename+suff, delimiter="\t", usecols=["#depth"], na_filter=False, dtype=np.float64, low_memory=False)

            depth = np.asarray(data["#depth"])
            self.total_depth = np.array([np.sum(depth),])

            self.n_zones = len(depth)

            depth[1:] = depth[1:] - depth[:-1]
            self.delta_r = depth

            return 0
        else:
            return 1

    def read_grnopc(self, filename, suff=".grnopc"):
        """ Read the multizone opacity spectrum"""
        if not self.out_files["grn_opc"]:
            return
        mydf = pandas.read_csv(self.cloudy_dir+filename+suff, delimiter="\t", skip_blank_lines=True, low_memory=False, na_filter=False)

        # Currently cloudy outputs a grain opacity file with an extra column and delimiter. This fixes that.
        if mydf.columns[0] == "#grain":
            rename_map = dict(zip(list(mydf.columns[0:-1]), list(mydf.columns[1:])))
            rename_map[mydf.columns[-1]] = "junk"
            mydf.rename(rename_map, axis='columns', inplace=True)
            del mydf["junk"]

        # #grain  nu/Ryd  abs+scat*(1-g)  abs     scat*(1-g)      scat    scat*(1-g)/[abs+scat*(1-g)]
        self.grn_opc = np.asarray(mydf["abs"])

    def read_opc(self, filename, suff=".opc"):
        """ Read the multizone opacity spectrum"""
        if not self.out_files["tot_opc"]:
            return

        if self.all_opacities:
            mydf = pandas.read_csv(self.cloudy_dir+filename+suff, delimiter="\t",skip_blank_lines=True, usecols=["Tot opac", "Scat opac"], dtype=np.float64, low_memory=False, na_filter=False)
        else:
            mydf = pandas.read_csv(self.cloudy_dir+filename+suff, delimiter="\t",skip_blank_lines=True, usecols=["Tot opac"], dtype=np.float64, low_memory=False, na_filter=False)

        if self.opacity_everyzone:

            #del mydf["#nu/Ryd"], mydf["elem"], mydf["Albedo"]

            if self.all_opacities is True:
                tau = (mydf.iloc[0:self.n_energy_bins] * float(self.delta_r[0])).to_numpy()            
            else:
                tau = (mydf["Tot opac"].iloc[0:self.n_energy_bins] * float(self.delta_r[0])).to_numpy()

            for izone in range(1, self.n_zones):
                if self.all_opacities is True:
                    tau[:] += (mydf.iloc[izone*self.n_energy_bins:(izone+1)*self.n_energy_bins] * self.delta_r[izone]).to_numpy()[:]
                else:
                    tau[:] += (mydf["Tot opac"].iloc[izone*self.n_energy_bins:(izone+1)*self.n_energy_bins] * self.delta_r[izone]).to_numpy()[:]
            
            self.tot_opc = tau / float(self.total_depth)

        else:
            #nu/Ryd	Tot opac	Abs opac	Scat opac	Albedo	elem
            self.tot_opc = np.asarray(mydf["Tot opac"])


    def read_in(self, filename, suff=".in"):
        """ Read the model input file
            We are looking to see if the model outputs are per zone, or just the last zone.
            This effects the output structure of the files, and requires two different
            ways of reading them, and one extra step for calculating an averge if every zone is used
        """

        with open(self.cloudy_dir + filename + suff, 'r') as file:
            while (line := file.readline().rstrip()):
                if "save opacity" in line:
                    self.opacity_everyzone = "every" in line
                if "save diffuse continuum" in line:
                    self.em_everyzone = "zone" in line

    def collect(self, single_file=None, delete_files = False):
        # Open up the hdf5 files
        self.open_hdf5_db()
        
        if single_file is not None:
            files = [single_file,]
        else:    
            files = glob.glob(self.cloudy_dir+"/*.out")
            
        cleaned = [int(file.split("/")[-1].strip(".out")[len("gasspy-"):]) for file in files]
        cleaned.sort()
        files = ["gasspy-%i"%file_i for file_i in cleaned]

        # if self.energy_bins == None:
        #     name = files[0]
        #     self.read_in(name)
        #     self.read_mol(name)
        #     self.read_em(name)

        # Figure out how many we will deal with at a time
        self.nfiles = len(files) 
        self.current_nsaves = 0
        # If we already have files, we can skip ahead from the ones we already have
        if not self.new_database:
            self.allocate_hdf5_datasets(self.nfiles)

        self.skip = True
        i = self.n_already_done - 1
        print(self.n_already_done)
        while self.skip:
            i += 1
            if i >= len(files):
                sys.exit("None of the cloudy outputs were readable")
            print("gasspy-%i\r"%(i))
            self.all(name=files[i])

        if self.single_files == False:
            # We need to know how many spectral bins we have to calculate the size of one model (or file)
            # This is the first time we for certain know this information, so set it here
            self.set_maxfiles_at_a_time()
            # Now figure out how many we actaully are doing at a time
            if self.nfiles - self.n_already_done > self.maxfiles_at_a_time:
                self.nfiles_at_a_time = self.maxfiles_at_a_time
            else:
                self.nfiles_at_a_time = self.nfiles

            if self.new_database:
                self.allocate_hdf5_datasets(self.nfiles)
            self.e_bins = self.energy_bins.copy()

            self.save_avg_em  = np.zeros((self.nfiles_at_a_time, self.n_energy_bins))
            self.save_tot_opc = np.zeros((self.nfiles_at_a_time, self.n_energy_bins))
            self.save_grn_opc = np.zeros((self.nfiles_at_a_time, self.n_energy_bins))
            i_now = i - self.n_already_done - self.nfiles_at_a_time*self.current_nsaves
            self.save_avg_em [i_now,:] = self.avg_em[:]
            self.save_tot_opc[i_now,:] = self.tot_opc[:]
            self.save_grn_opc[i_now,:] = self.grn_opc[:]

            self.clear()
            for i in range(i+1, len(files)):
                if i % 100 == 0:
                    gc.collect()
                    #print("gasspy-%i\r"%(i))
                    loop_progress.print_progress(i, len(files), start = "\t ", end = " %06d"%i)
                
                self.all("gasspy-%i"%i)
                i_now = i - self.n_already_done - self.nfiles_at_a_time*self.current_nsaves
                if self.skip is False:
                    self.save_avg_em [i_now,:] = self.avg_em[:]
                    self.save_tot_opc[i_now,:] = self.tot_opc[:]
                    self.save_grn_opc[i_now,:] = self.grn_opc[:]
                else:
                    print("Skipping gasspy-%i"%i)
                self.clear()

                if i_now == self.nfiles_at_a_time - 1:
                    index_start = self.n_already_done + self.nfiles_at_a_time*self.current_nsaves
                    current_model_indexes = np.arange(index_start, i_now)
                    self.write_to_hdf5_datasets(index_start, i, i_now)
                    self.current_nsaves += 1
            # If the loop finished before writing the last models, do so here
            if i_now < self.nfiles_at_a_time - 1:
                index_start = self.n_already_done + self.nfiles_at_a_time*self.current_nsaves
                current_model_indexes = np.arange(index_start, i)
                self.write_to_hdf5_datasets(index_start, i, i_now)
        else:
            if self.skip:
                self.avg_em  = np.zeros(1)
                self.tot_opc = np.zeros(1)
                self.grn_opc = np.zeros(1)

            name = files[0]
            self.save_single(name)

    def save_single(self, name):
        np.save(self.cloudy_dir+"%s_avg_em.pkl"%name, self.avg_em, allow_pickle=True)
        np.save(self.cloudy_dir+"%s_tot_opc.pkl"%name, self.tot_opc, allow_pickle=True)
        np.save(self.cloudy_dir+"%s_grn_opc.pkl"%name, self.grn_opc, allow_pickle=True)

    def open_hdf5_db(self):
        h5path = self.out_dir+"/"+self.db_name
        # If there is no database present, or if we force rerun, open a new one 
        if (not Path(h5path).is_file()) or self.recollect_all :
            self.h5Database = hp.File(h5path, "w")
            self.new_database = True
            self.n_already_done = 0

        else:
            self.h5Database = hp.File(h5path, "r+")
            self.energy_bins = self.h5Database["energy"][:]
            self.new_database = False
            self.n_already_done = self.h5Database["avg_em"].shape[0]


    def allocate_hdf5_datasets(self, Ntot):
        if self.new_database:
            # If this is a new database, just initialize datasets
            self.h5Database.create_dataset("energy",   data = self.energy_bins)
            self.h5Database.create_dataset("avg_em",  shape = (Ntot, len(self.energy_bins)), maxshape = (None, len(self.energy_bins)),chunks = True)
            self.h5Database.create_dataset("tot_opc", shape = (Ntot, len(self.energy_bins)), maxshape = (None, len(self.energy_bins)),chunks = True)
            self.h5Database.create_dataset("grn_opc", shape = (Ntot, len(self.energy_bins)), maxshape = (None, len(self.energy_bins)),chunks = True)
        else:
            # Otherwise, we need to reshape them to fit the new data   
            self.h5Database["avg_em"].resize(Ntot, axis = 0)
            self.h5Database["tot_opc"].resize(Ntot, axis = 0)
            self.h5Database["grn_opc"].resize(Ntot, axis = 0)

    def write_to_hdf5_datasets(self, gasspy_id_start, gasspy_id_end, i_now):
        self.h5Database["avg_em" ][gasspy_id_start : gasspy_id_end+1,:] = self.save_avg_em [:i_now+1,:]
        self.h5Database["tot_opc"][gasspy_id_start : gasspy_id_end+1,:] = self.save_tot_opc[:i_now+1,:]
        self.h5Database["grn_opc"][gasspy_id_start : gasspy_id_end+1,:] = self.save_grn_opc[:i_now+1,:]
        self.save_avg_em [:i_now+1,:] = 0
        self.save_tot_opc[:i_now+1,:] = 0
        self.save_grn_opc[:i_now+1,:] = 0

    def save_to_db(self):
        """Save the model to an gasspy database"""
        np.save(self.out_dir + "/%s_ebins.pkl"%self.db_name, self.energy_bins, allow_pickle=True)
        np.save(self.out_dir + "/%s_avg_em.pkl"%self.db_name, self.save_avg_em, allow_pickle=True)
        np.save(self.out_dir + "/%s_tot_opc.pkl"%self.db_name, self.save_tot_opc, allow_pickle=True)
        np.save(self.out_dir + "/%s_grn_opc.pkl"%self.db_name, self.save_grn_opc, allow_pickle=True)

    def clear(self):
        """Re-init no-global values for use by next model"""
        self.opacity_everyzone = None
        self.em_everyzone = None

        self.avg_em = None
        self.tot_opc = None

        self.delta_r = None
        self.total_depth = None

        if self.clear_energy:
            self.energy_bins = None
            self.n_energy_bins = None
        
        self.skip = False


    def all(self, name):
        """ Run all routines for a single model """
        self.skip = False
        try:
        #if True:
            self.read_in(name)
            self.read_mol(name)
            self.read_em(name)
            self.read_grnopc(name)
            self.read_opc(name)

        except:
        #else:
            self.skip = True
    
if __name__ == "__main__":
    import time
    import cProfile, pstats
    
    if len(sys.argv) > 1:
        cloudy_dir = sys.argv[1]
        out_dir = sys.argv[2]
    else:
        cloudy_dir=os.getenv("HOME")+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/cloudy-output/"

    mc = ModelCollector(cloudy_dir=cloudy_dir, out_dir=out_dir, single_files=False, clear_energy=False)

    mc.use_gpu=False
    mc.all_opacities = False
    assert mc.all_opacities is False, "Scattering opacities not implemented, only total"
    mc.clear_energy = False

    mc.collect()
    mc.save_to_db()

