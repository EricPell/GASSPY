"""Read and collect cloudy models"""
import pickle
from re import M
import pandas
import numpy as np
import cupy
import cudf
import gc
import os 
import glob
import sys

class ModelCollector():
    """Worker class for reading and collecting cloudy models into a gasspy db entry"""
    def __init__(
        self, cloudy_dir="cloudy_output",
        out_dir="GASSPY/",
        db_name="gasspy",
        use_gpu=False,
        all_opacities=True,
        clear_energy=False,
        single_files=False
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

        self.use_gpu = use_gpu
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


    def read_em(self, filename, suff=".em"):
        """ Read the multizone emissivity spectrum """
        if self.use_gpu:
            try:
                mydf = cudf.read_csv(
                    self.cloudy_dir+filename+suff,
                    delimiter="\t",
                    comment="#",
                    header=None)
            except:
                print("failed to read %s"%(self.cloudy_dir+filename+suff))
                self.skip = True
        else:
            mydf = pandas.read_csv(
                self.cloudy_dir+filename+suff,
                delimiter="\t",
                comment="#",
                header=None)

        if self.em_everyzone:
            if self.use_gpu:
                data = cupy.zeros(mydf.shape)
                for col_i, col in enumerate(mydf.columns):
                    data[:,col_i] = cupy.asarray(mydf[col])[:]

                if self.energy_bins is None:
                    self.energy_bins = data[0,:]
            else:
                data = np.array(mydf)

                if self.energy_bins is None:
                    self.energy_bins = data[0,:]

            data = data[1:,:]
            self.avg_em = (data.T * self.delta_r).sum(axis=1)/float(self.total_depth)
        
        else:
            if self.use_gpu:
                if self.energy_bins is None:
                    self.energy_bins = cupy.array(mydf[mydf.columns[0]])
                self.avg_em = cupy.array(mydf[mydf.columns[3]])
            else:
                if self.energy_bins is None:
                    self.energy_bins = np.array(mydf[mydf.columns[0]])
                self.avg_em = np.array(mydf[mydf.columns[3]])
        
        if self.n_energy_bins is None:
            self.n_energy_bins = len(self.energy_bins)

        return 0

    def read_mol(self, filename, suff=".mol"):
        """Read the molecular data file, mostly to get out the depth array of the model"""
        with open(self.cloudy_dir+filename+suff,"r") as f:
            if len(f.readlines()) < 2:
                self.skip = True

        if not self.skip:
            if self.use_gpu:
                data = cudf.read_csv(self.cloudy_dir+filename+suff, delimiter="\t")
                depth = cupy.asarray(data["#depth"])
                self.total_depth = cupy.array([cupy.sum(depth),])
            else:
                data = pandas.read_csv(self.cloudy_dir+filename+suff, delimiter="\t")
                depth = np.asarray(data["#depth"])
                self.total_depth = cupy.array([np.sum(depth),])

            self.n_zones = len(depth)

            depth[1:] = depth[1:] - depth[:-1]
            self.delta_r = depth

            return 0
        else:
            return 1

    def read_grnopc(self, filename, suff=".grnopc"):
        """ Read the multizone opacity spectrum"""
        if self.use_gpu:
            mydf = cudf.read_csv(self.cloudy_dir+filename+suff, delimiter="\t", skip_blank_lines=True)
        else:
            mydf = pandas.read_csv(self.cloudy_dir+filename+suff, delimiter="\t", skip_blank_lines=True)

        # Currently cloudy outputs a grain opacity file with an extra column and delimiter. This fixes that.
        if mydf.columns[0] == "#grain":
            rename_map = dict(zip(list(mydf.columns[0:-1]), list(mydf.columns[1:])))
            rename_map[mydf.columns[-1]] = "junk"
            mydf.rename(rename_map, axis='columns', inplace=True)
            del mydf["junk"]

        # #grain  nu/Ryd  abs+scat*(1-g)  abs     scat*(1-g)      scat    scat*(1-g)/[abs+scat*(1-g)]
        if self.use_gpu:
            self.grn_opc = cupy.asarray(mydf["abs"])
        else:
            self.grn_opc = np.asarray(mydf["abs"])

    def read_opc(self, filename, suff=".opc"):
        """ Read the multizone opacity spectrum"""
        if self.use_gpu:
            mydf = cudf.read_csv(self.cloudy_dir+filename+suff, delimiter="\t",skip_blank_lines=True)
        else:
            mydf = pandas.read_csv(self.cloudy_dir+filename+suff, delimiter="\t",skip_blank_lines=True)

        if self.opacity_everyzone:

            del mydf["#nu/Ryd"], mydf["elem"], mydf["Albedo"]

            if self.use_gpu:
                if self.all_opacities is True:
                    tau = cupy.zeros((self.n_energy_bins,len(mydf.columns)))
                    for col_i, col in enumerate(mydf.columns):
                        tau[:,col_i] = (self.delta_r[0] * cupy.asarray(mydf[col].iloc[0:self.n_energy_bins]))[:]
                else:
                    total_opc = cupy.asarray(mydf["Tot opac"])
                    tau = cupy.zeros(self.n_energy_bins)
                    tau[:] = (self.delta_r[0] * total_opc[0:self.n_energy_bins])[:]

                #delta_tau = (opc.iloc[0:self.n_energy_bins] * float(self.delta_r[0])).to_cupy()
            else:
                if self.all_opacities is True:
                    tau = (mydf.iloc[0:self.n_energy_bins] * float(self.delta_r[0])).to_numpy()            
                else:
                    tau = (mydf["Tot opac"].iloc[0:self.n_energy_bins] * float(self.delta_r[0])).to_numpy()

            for izone in range(1, self.n_zones):
                if self.use_gpu:
                    if self.all_opacities is True:
                        for col_i, col in enumerate(mydf.columns):
                            tau[:,col_i] += cupy.asarray(mydf[col].iloc[izone*self.n_energy_bins:(izone+1)*self.n_energy_bins]) * self.delta_r[izone]
                    else:
                        tau[:] += (self.delta_r[izone] * total_opc[izone*self.n_energy_bins:(izone+1)*self.n_energy_bins])[:]
                else:
                    if self.all_opacities is True:
                        tau[:] += (mydf.iloc[izone*self.n_energy_bins:(izone+1)*self.n_energy_bins] * self.delta_r[izone]).to_numpy()[:]
                    else:
                        tau[:] += (mydf["Tot opac"].iloc[izone*self.n_energy_bins:(izone+1)*self.n_energy_bins] * self.delta_r[izone]).to_numpy()[:]
            
            self.tot_opc = tau / float(self.total_depth)

        else:
            #nu/Ryd	Tot opac	Abs opac	Scat opac	Albedo	elem
            if self.use_gpu:
                self.tot_opc = cupy.asarray(mydf["Tot opac"])
            else:
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

            #self.use_gpu = self.em_everyzone

    def collect(self, single_file=None, delete_files = False):
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

        i = 0
        self.all(name=files[0])

        if self.single_files == False:
            self.e_bins = self.energy_bins.copy()

            self.save_avg_em  = np.zeros((self.n_energy_bins,len(files)))
            self.save_tot_opc = np.zeros((self.n_energy_bins, len(files)))
            self.save_grn_opc = np.zeros((self.n_energy_bins, len(files)))

            if self.use_gpu is True:
                self.save_avg_em[:,i] = cupy.asnumpy(self.avg_em)[:]
                self.save_tot_opc[:,i] = cupy.asnumpy(self.tot_opc)[:]
                self.save_grn_opc[:,i] = cupy.asnumpy(self.grn_opc)[:]
            else:
                self.save_avg_em[:,i] = self.avg_em[:]
                self.save_tot_opc[:,i] = self.tot_opc[:]
                self.save_grn_opc[:,i] = self.grn_opc[:]

            self.clear()
            for i in range(i+1, len(files)):
                if i % 100 == 0:
                    gc.collect()
                    print("gasspy-%i\r"%(i))
                self.all("gasspy-%i"%i)
                if self.skip is False:
                    pass
                    if self.use_gpu is True:
                        self.save_avg_em[:,i] = cupy.asnumpy(self.avg_em)[:]
                        self.save_tot_opc[:,i] = cupy.asnumpy(self.tot_opc)[:]
                        self.save_grn_opc[:,i] = cupy.asnumpy(self.grn_opc)[:]
                    else:
                        self.save_avg_em[:,i] = self.avg_em[:]
                        self.save_tot_opc[:,i] = self.tot_opc[:]
                        self.save_grn_opc[:,i] = self.grn_opc[:]
                else:
                    print("Skipping gasspy-%i"%i)
                self.clear()
        else:
            if self.skip:
                if self.use_gpu is True:
                    self.avg_em  = cupy.zeros(1)
                    self.tot_opc = cupy.zeros(1)
                    self.grn_opc = cupy.zeros(1)
                else:
                    self.avg_em  = np.zeros(1)
                    self.tot_opc = np.zeros(1)
                    self.grn_opc = np.zeros(1)

            name = files[0]
            self.save_single(name)

    def save_single(self, name):
        if self.use_gpu is True:
            cupy.save(self.cloudy_dir+"%s_avg_em.pkl"%name, self.avg_em, allow_pickle=True)
            cupy.save(self.cloudy_dir+"%s_tot_opc.pkl"%name, self.tot_opc, allow_pickle=True)
            cupy.save(self.cloudy_dir+"%s_grn_opc.pkl"%name, self.grn_opc, allow_pickle=True)
        else:
            np.save(self.cloudy_dir+"%s_avg_em.pkl"%name, self.avg_em, allow_pickle=True)
            np.save(self.cloudy_dir+"%s_tot_opc.pkl"%name, self.tot_opc, allow_pickle=True)
            np.save(self.cloudy_dir+"%s_grn_opc.pkl"%name, self.grn_opc, allow_pickle=True)

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

