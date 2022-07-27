"""
This routine performs a spectra RT on an AMR grid
"""
import sys, os
from pathlib import Path
import numpy as np
import cupy
import torch
import h5py
from astropy.io import fits
from astropy import constants as const
from gasspy.raystructures import global_ray_class
from gasspy.settings.defaults import ray_dtypes    
import yaml

class FamilyTree():
    def __init__(
        self, root_dir="./",
        gasspy_subdir="GASSPY",
        gasspy_spec_subdir="spec",
        gasspy_projection_subdir="projections",
        modeldir = "GASSPY_DATABASE",
        traced_rays=None,
        global_rayDF_deprecated=None,
        energy=None,
        energy_lims=None,
        Emask=None,
        em=None,
        op=None,
        opc_per_NH=False,
        cell_index_to_gasspydb=None,
        vel=None,
        den=None,
        massden=True,
        mu=1.1,
        los_angle=None,
        accel="torch",
        dtype=np.float32,
        liteVRAM=True,
        Nraster=4,
        useGasspyEnergyWindows=True,
        make_spec_subdirs=True,
        config_yaml=None,
        spec_save_type="hdf5",
        spec_save_name="gasspy_spec",
        cuda_device=None,
        h5database = None,
        em_field = "avg_em",
        op_field = "tot_opc"
    ):
        self.cuda_device = cuda_device
        self.dtype = dtype
        self.torch_dtype = torch.as_tensor(np.array([],dtype=self.dtype)).dtype

        if liteVRAM:
            self.numlib = np
        else:
            self.numlib = cupy

        if str == type(root_dir):
            assert Path(root_dir).is_dir(), "Provided root_dir \""+root_dir+"\"is not a directory"
            self.root_dir = root_dir

        assert type(gasspy_subdir) == str, "gasspy_subdir not a string"
        if not gasspy_subdir.endswith("/"):
            gasspy_subdir = gasspy_subdir + "/"
        if not gasspy_subdir[0] == "/":
            gasspy_subdir = "/" + gasspy_subdir
        self.gasspy_subdir = gasspy_subdir
        assert Path(self.root_dir+self.gasspy_subdir).is_dir() == True, "GASSPY subdir does not exist..."

        assert type(gasspy_projection_subdir) == str, "gasspy_projection_subdir not a string"
        if not gasspy_projection_subdir[0] == "/":
            gasspy_projection_subdir = "/" + gasspy_projection_subdir
        if not gasspy_spec_subdir.endswith("/"):
            gasspy_projection_subdir = gasspy_projection_subdir + "/"
        self.gasspy_projection_subdir = gasspy_projection_subdir
        assert Path(self.root_dir+self.gasspy_subdir+self.gasspy_projection_subdir).is_dir() == True, "GASSPY projections dir does not exist..."

        assert type(gasspy_spec_subdir) == str, "gasspy_spec_subdir not a string"
        if not gasspy_spec_subdir[0] == "/":
            gasspy_spec_subdir = "/" + gasspy_spec_subdir
        if not gasspy_spec_subdir.endswith("/"):
            gasspy_spec_subdir = gasspy_spec_subdir + "/"
        self.gasspy_spec_subdir = gasspy_spec_subdir
        
        assert type(modeldir) == str, "modeldir not a string"
        if not modeldir.endswith("/"):
            modeldir = modeldir+ "/"
        self.modeldir = modeldir


        if not isinstance(traced_rays, h5py._hl.files.File):
            assert isinstance(traced_rays, str), "provided traced rays is neither a string or open hd5 file"
            if Path(traced_rays).is_file():
                tmp_path = traced_rays
            elif Path(self.root_dir+self.gasspy_subdir+self.gasspy_projection_subdir+traced_rays).is_file():
                tmp_path = self.root_dir+self.gasspy_subdir+self.gasspy_projection_subdir+traced_rays
            else:
                sys.exit("Could not find the traced rays file\n"+\
                "Provided path: %s"%traced_rays+\
                "Try looking in \"./\" and %s\n"%(self.root_dir+self.gasspy_projection_subdir)+\
                "Aborting...")            

            self.traced_rays_h5file = h5py.File(tmp_path, "r")
        else:
            self.traced_rays_h5file = traced_rays

        self.spec_save_name = spec_save_name
        self.spec_save_type = spec_save_type

        self.mu = mu
        self.den = den
        self.massden = massden
        self.opc_per_NH = opc_per_NH

        self.Nraster = Nraster

        # use torch tensors?
        self.accel = accel.lower()

        self.useGasspyEnergyWindows = useGasspyEnergyWindows

        self.energy_lims = energy_lims
        self.raydump_dict = {}

        self.los_angle = los_angle

        self.liteVRAM = liteVRAM

        self.branch = {}

        self.Emask = Emask
        self.energy = energy
        self.em = em
        self.op = op
        self.vel = vel
        self.cell_index_to_gasspydb = cell_index_to_gasspydb
        self.traced_rays = traced_rays
        self.global_rayDF_deprecated = global_rayDF_deprecated
        self.config_yaml = config_yaml

        self.h5database = h5database
        self.em_field = em_field
        self.op_field = op_field
    def process_all(self,):
        print(len(self.energy)) 
        for root_i in range(0, len(self.ancenstors)):
            self.get_spec_root(root_i, self.cuda_device)
            if root_i % 1000 == 0:
                print(root_i)
        
        self.close_spec_save_hdf5()

    def open_spec_save_hdf5(self, init_size=0):
        assert isinstance(self.spec_save_name, str), "hdf5 spec save name is not a string...exiting" 
        if not self.spec_save_name.endswith(".hdf5"):
            self.spec_save_name += ".hdf5"

        if Path(self.gasspy_spec_subdir).is_dir():
            self.spec_outpath = self.gasspy_spec_subdir+self.spec_save_name
        else:
            self.spec_outpath = self.root_dir+self.gasspy_subdir+self.gasspy_spec_subdir+self.spec_save_name

        self.spechdf5_out = h5py.File(self.spec_outpath, "w")
        self.N_spec_written = 0

        if init_size >=0:
            init_size=int(init_size)
        else:
            init_size = self.numlib.int(self.new_global_rays.cevid[self.new_global_rays.cevid == -1].shape[0])

        self.spechdf5_out.create_dataset("flux", (init_size, len(self.energy)), maxshape=(None,len(self.energy)), dtype = self.dtype)
        self.spechdf5_out.create_dataset("x", (init_size,), maxshape=(None,), dtype = ray_dtypes["xp"])
        self.spechdf5_out.create_dataset("y", (init_size,), maxshape=(None,), dtype = ray_dtypes["xp"])
        self.spechdf5_out.create_dataset("ray_lrefine", (init_size,), dtype="int8", maxshape=(None,))
        self.spechdf5_out.create_dataset("E", data=self.energy.cpu().numpy())

    def write_spec_save_hdf5(self, new_data, grow=True):
        n_E, n_spec = new_data['flux'].shape
        for key in new_data.keys():
            new_data_shape = new_data[key].shape

            if not grow:
                if len(new_data_shape) == 1:
                    self.spechdf5_out[key][self.N_spec_written:self.N_spec_written+n_spec] = new_data[key][:]
        
                elif len(new_data_shape) == 2:
                    self.spechdf5_out[key][self.N_spec_written:self.N_spec_written+n_spec,:] = new_data[key].T[:]

            else:
                if len(new_data_shape) == 1:
                    self.spechdf5_out[key].resize((self.spechdf5_out[key].shape[0] + n_spec), axis=0)
                    self.spechdf5_out[key][-n_spec:] = new_data[key][:]
        
                elif len(new_data_shape) == 2:
                    self.spechdf5_out[key].resize((self.spechdf5_out[key].shape[0] + n_spec), axis=0)
                    self.spechdf5_out[key][-n_spec:,:] = new_data[key].T[:]

        self.N_spec_written += n_spec

    def close_spec_save_hdf5(self):
        if self.spec_save_type=='hdf5':
            self.spechdf5_out.close()
        else:
            print("wARNING: You tried to close an output hdf5 file, but are not using hdf5 output.")

    def load_all(self):
        self.load_config_yaml()
        self.load_cell_index_to_gasspydb()
        # Ensure the energy bins are loaded BEFORE the em and op tables to minimize memory used
        self.load_energy_limits()

        if self.h5database is None:
            # LEGACY LOADING SYSTEM
            self.load_energy_bins()
            self.load_em()
            self.load_op()
        else:
            self.load_database()
        #self.load_saved3d()
        self.load_new_global_rays()
        self.load_velocity_data()
        self.load_density_data()
        self.load_traced_rays()

        self.set_precision(self.dtype)
        self.cleaning()
        self.padding()
        self.move_to_GPU()

        if self.spec_save_type == 'hdf5':
            self.open_spec_save_hdf5()

    def set_precision(self, new_dtype):
        self.dtype = new_dtype
        self.torch_dtype = torch.as_tensor(np.array([],dtype=self.dtype)).dtype

        self.em                         = self.em.astype(new_dtype)
        self.op                         = self.op.astype(new_dtype)
        self.raydump_dict["pathlength"] = self.raydump_dict["pathlength"].astype(new_dtype)
        if self.opc_per_NH:
            self.den = self.den.astype(new_dtype)

    def load_density_data(self):
        if isinstance(self.den, str):
            if self.den.endswith(".fits"):
                self.den = fits.open(self.den)
                self.den = self.den[0].data
        if self.massden:
            # We have been provided mass density and need to convert to number
            self.den /= self.mu * const.m_p.cgs.value

    def load_config_yaml(self):
        if isinstance(self.config_yaml, str):
            if self.config_yaml is None:
                assert Path(self.root_dir+self.gasspy_subdir+"gasspy_config.yaml").is_file(), "Error: gasspy_config.yaml is not given and does not exists in simulation GASSPY directory"
                self.config_yaml = self.root_dir+self.gasspy_subdir+"gasspy_config.yaml"
                    
            with open(r'%s'%(self.config_yaml)) as file:
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                self.config_yaml = yaml.load(file, Loader=yaml.FullLoader)
        
        self.sim_unit_length = self.config_yaml["sim_unit_length"]

    def load_new_global_rays(self):
        self.new_global_rays = global_ray_class(on_cpu=self.liteVRAM)
        self.new_global_rays.load_hdf5(self.traced_rays_h5file)
        print(self.new_global_rays.xp.dtype)
        # Select the ancestral GIDs
        self.ancenstors = self.new_global_rays.global_rayid[self.numlib.where(self.new_global_rays.pevid == -1)]

    def load_global_rays_deprecated(self):
        # load up on rays
        if isinstance(self.global_rayDF_deprecated, str):
            if Path(self.global_rayDF_deprecated).is_file():
                tmp_path = self.global_rayDF_deprecated
            elif Path(self.root_dir + self.gasspy_subdir + self.global_rayDF_deprecated).is_file():
                tmp_path = self.root_dir + self.gasspy_subdir + self.global_rayDF_deprecated
            #self.global_rayDF_deprecated = cudf.read_hdf(tmp_path)
        # expressions to find parents and children
        self.pid_expr = "(pid == %i)"
        self.pevid_expr = "(pevid == %i)"

        # Get all first_parents with children
        self.ancenstors = self.global_rayDF_deprecated.query("(pevid == -1)")

    def load_energy_limits(self):
        if self.useGasspyEnergyWindows and self.energy_lims is None:
            if Path(self.root_dir+self.gasspy_subdir+"gasspy_continuum_mesh_windows.txt").is_file():
                self.energy_lims = self.root_dir+self.gasspy_subdir+"gasspy_continuum_mesh_windows.txt"
            elif "CLOUDY_DATA_PATH" in os.environ:
                if Path(os.environ['CLOUDY_DATA_PATH']+"gasspy_continuum_mesh_windows.txt").is_file():
                    self.energy_lims =os.environ['CLOUDY_DATA_PATH']+"gasspy_continuum_mesh_windows.txt"
        
        if self.energy_lims is not None:
            if type(self.energy_lims) == str:
                self.energy_lims = np.loadtxt(self.energy_lims)           


    def sort_and_merge_energy_lims(self):
        if self.energy_lims is None:
            return
        # Sort based of upper index
        sorted_index = np.argsort(self.energy_lims[:,1])
        Elower = self.energy_lims[sorted_index,0]
        Eupper = self.energy_lims[sorted_index,1]

        # We need to merge all ranges that overlap
        # This is done through iteratation
        merged = False
        while not merged:
            mat_diff = Elower[np.newaxis,:] - Eupper[:,np.newaxis]
            has_merged = np.zeros(Elower.shape)
            Elower_new = []
            Eupper_new = []
 
            merged = True
            # Loop through the ranges
            for iE in range(len(Eupper)):
                to_merge = []
                if iE < len(Eupper) -1:
                    # If the upper limit of this range is higher than the lower limit of any range above it
                    # these are to be merged
                    to_merge = np.arange(iE+1, len(Eupper))[np.where(mat_diff[iE,iE+1:]<0)[0]]
                # If we are not merging with anything just append to the new lists
                if len(to_merge)==0:
                    if not has_merged[iE]:
                        Elower_new.append(Elower[iE]) 
                        Eupper_new.append(Eupper[iE]) 
                    continue
                # Otherwise append the minimums of the lower limits and maximum of the upper limits
                Elower_new.append(min(Elower[iE],np.min(Elower[to_merge])))
                Eupper_new.append(max(Eupper[iE],Eupper[to_merge[-1]]))
                has_merged[to_merge] = True
                merged = False

            # Update arrays and sort
            Eupper = np.array(Eupper_new)
            sorted_index = np.argsort(Eupper)
            Elower = np.array(Elower_new)[sorted_index]
            Eupper = Eupper[sorted_index]

        self.energy_lims = np.array([Elower,Eupper]).T
                
    def load_database(self):
        if self.h5database is None:
            return
        # Make sure that we load the hdf5 file or have been given one
        if isinstance(self.h5database, str):
            if Path(self.h5database).is_file():
                tmp_path = self.h5database
            elif Path(self.modeldir + self.h5database).is_file():
                tmp_path = self.modeldir + self.h5database
            else:
                sys.exit("Could not find path to Database %s"%(self.h5database))
            self.h5database = h5py.File(tmp_path, "r")
        else:
            if not isinstance(self.h5database, h5py.File) or isinstance(self.h5database, h5py.Group):
                sys.exit("Supplied h5database must be a h5py.File or a path to one, or an h5py.Group ")
            
        # Load the required energies
        self.energy = self.h5database["energy"][:]
        Eidx_ranges = [[0,-1]]
        if self.energy_lims is not None:
            self.sort_and_merge_energy_lims()
            #Eidxs is all of the indexes
            #Eidx_ranges are the values used for slices in each range
            Eidxs = np.array([], dtype = int)
            Eidx_ranges = []
            for elims in self.energy_lims:
                # Find the indexes in the array
                Eidx = np.where((self.energy >= elims[0])*(self.energy<=elims[1]))[0]
                Eidxs = np.append(Eidxs,Eidx)
                Eidx_ranges.append([Eidx[0],Eidx[-1]+1])
            # Select range from energy
            self.energy = self.energy[Eidxs]
        
        # How many energy bins are we dealing with here?
        self.Nener = len(self.energy)

        # Figure out which gasspyIDs we need and remap to the new "snap specific" database 
        unique_gasspy_ids, self.cell_index_to_gasspydb = np.unique(self.cell_index_to_gasspydb, return_inverse=True)
        # NOTE: np.unique returns sorted indexes. IF THIS EVER CHANGES WE NEED TO SORT MANUALLY

        self.Nmodels = len(unique_gasspy_ids)
        print(self.Nmodels,self.Nener)
        # Initialize arrays
        self.em = np.zeros((self.Nener, self.Nmodels))
        self.op = np.zeros((self.Nener, self.Nmodels))
        
        #Loop over index ranges and load
        Eidx_start = 0 
        Eidx_end = 0
        for Eidx_range in Eidx_ranges:
            Eidx_start = Eidx_end
            Eidx_end = Eidx_start + Eidx_range[1]-Eidx_range[0]
            self.em[Eidx_start:Eidx_end,:] = self.h5database[self.em_field][unique_gasspy_ids, Eidx_range[0]:Eidx_range[1]].T
            self.op[Eidx_start:Eidx_end,:] = self.h5database[self.op_field][unique_gasspy_ids, Eidx_range[0]:Eidx_range[1]].T
        return    

    def load_energy_bins(self, save=False):
        # Processing the energy limits must come first, as it allows us to create a numpy mask and convert less 
        # data to the GPU.
        if str == type(self.energy):
            if Path(self.energy).is_file():
                tmp_path = self.energy

            elif Path(self.root_dir + self.gasspy_subdir + self.energy).is_file():
                tmp_path = self.root_dir + self.gasspy_subdir + self.energy
            else:
                sys.exit("Could not find path to energy ")

            self.energy = np.load(tmp_path).astype(self.dtype)

            if self.energy_lims is not None:
                self.Emask = np.full(self.energy.shape, False, dtype = np.bool8)
                for i in range(len(self.energy_lims)):
                    self.Emask[(self.energy >= self.energy_lims[i,0]) * (self.energy <= self.energy_lims[i,1])] = True

            if self.Emask is not None:
                self.energy = self.energy[self.Emask]
        self.energy_upper = np.zeros_like(self.energy)
        self.energy_upper[1:] = (self.energy[1:] + self.energy[:-1])*0.5
        self.energy_upper[-1] = 2*self.energy[-1] - self.energy_upper[-2]
        if save:
            np.save(self.root_dir + self.gasspy_subdir + self.gasspy_spec_subdir + "windowed_energy.npy", self.energy)


    def load_em(self):
        if str == type(self.em):
            if Path(self.em).is_file():
                tmp_path = self.em
            elif Path(self.root_dir + self.gasspy_subdir + self.em).is_file():
                tmp_path = self.root_dir + self.gasspy_subdir + self.em
            else:
                sys.exit("Could not find path to em")

            if self.Emask is not None:
                fmap = np.load(tmp_path, mmap_mode="r") 
                self.em = np.compress(self.Emask, fmap, axis=0).astype(self.dtype)            
            else:
                self.em = np.load(tmp_path).astype(self.dtype)

            # Because of the different dimensionality of the energy grid and the emissivity tables, we use compress to 
            # produce a reduced op limited to the region of interest
            #if self.Emask is not None:
            #    self.em = np.compress(self.Emask, self.em, axis=0).astype(self.dtype)

    def load_op(self):
        if str == type(self.op):
            if Path(self.op).is_file():
                tmp_path = self.op
            elif Path(self.root_dir + self.gasspy_subdir + self.op).is_file():
                tmp_path = self.root_dir + self.gasspy_subdir + self.op
            else:
                sys.exit("Could not find path to op")

            if self.Emask is not None:
                fmap = np.load(tmp_path, mmap_mode="r") 
                self.op = np.compress(self.Emask, fmap, axis=0).astype(self.dtype)
            else:
                self.op = np.load(tmp_path).astype(self.dtype)

            # Because of the different dimensionality of the energy grid and the opacity tables, we use compress to 
            # produce a reduced op limited to the region of interest

    def load_saved3d(self):
        if str == type(self.saved3d):
            if Path(self.saved3d).is_file():
                tmp_path = self.saved3d
            elif Path(self.root_dir + self.gasspy_subdir + self.saved3d).is_file():
                tmp_path = self.root_dir + self.gasspy_subdir + self.saved3d
            else:
                sys.exit("Could not find path to saved3d")

            self.saved3d = np.load(tmp_path)
            # save only the inverse_indexes to cell_index_to_gasspydb

    def load_cell_index_to_gasspydb(self):
        if str == type(self.cell_index_to_gasspydb):
            if Path(self.cell_index_to_gasspydb).is_file():
                tmp_path = self.cell_index_to_gasspydb
            else:
                sys.exit("Could not find path to cell_gasspy_index: %s"%self.cell_index_to_gasspydb)

            self.cell_index_to_gasspydb = np.load(tmp_path).astype(int)
            print(self.cell_index_to_gasspydb)
            # save only the inverse_indexes to cell_index_to_gasspydb

        self.NSimCells = len(self.cell_index_to_gasspydb)

    def load_velocity_data(self):
        if str == type(self.vel):
            if self.vel.endswith("fits"):
                if Path(self.vel).is_file():
                    self.vel = fits.open(self.vel)
                    self.velocity = np.array([0*self.vel[0].data, 0*self.vel[0].data,self.vel[0].data])/3e10
                else:
                    sys.exit("Could not find path to velocity fits file")
            else:
                sys.exit("Unrecognizized format for velocity data file. Contact developers...")
        else:
            self.velocity = self.vel/3e10
            print(self.velocity.max(), self.velocity.min())
    def load_traced_rays(self):
        self.raydump_dict['segment_global_rayid'] = self.traced_rays_h5file['ray_segments']['global_rayid'][:]
        self.raydump_dict["pathlength"] = self.traced_rays_h5file["ray_segments"]["pathlength"][:,:].astype(self.dtype)*self.dtype(self.sim_unit_length)
        self.raydump_dict["cell_index"] = self.traced_rays_h5file["ray_segments"]["cell_index"][:,:]
        self.raydump_dict["splitEvents"] = self.traced_rays_h5file["splitEvents"][:,:]
 
        maxGID = self.numlib.int(self.new_global_rays.global_rayid.max())

        # Initialize the raydump N_segs and index into ray_buffer_dumps with -1
        self.raydump_dict["ray_index0"] = self.numlib.full(maxGID+1,-1).astype(np.int64)
        self.raydump_dict["Nsegs"] = self.numlib.full(maxGID+1,-1).astype(np.int64)
        
        # Get the unique values and the first index of array with that value, and Counts. 
        # This ONLY works because segment global_rayid is sorted.
        unique_gid, i0, Nsegs = np.unique(self.raydump_dict['segment_global_rayid'], return_counts=True, return_index=True)

        self.raydump_dict["ray_index0"][unique_gid] = i0.astype(np.int64)
        self.raydump_dict["Nsegs"][unique_gid] = Nsegs.astype(np.int64)

        self.raydump_dict['NcellPerRaySeg'] = self.raydump_dict["pathlength"].shape[1]
        self.family_tree = {}
        self.max_level = 0
        self.create_split_event_dict()

    def create_split_event_dict(self):
        self.raydump_dict['splitEvents'] = cupy.asnumpy(self.raydump_dict['splitEvents'])
        # Create a dictionary with each split event keyed by the GID of teh parent
        self.split_by_gid_tree = dict(zip(self.raydump_dict['splitEvents'][:, 0].astype(np.int32), zip(*self.raydump_dict['splitEvents'][:, 1:].astype(np.int32).T)))

    def padding(self):
        # The last element of each of the following arrays is assumed to be zero, and is used for out of bound indexes, which will have values -1.
        self.raydump_dict["pathlength"] = np.vstack([self.raydump_dict["pathlength"], np.zeros(self.raydump_dict["pathlength"].shape[1], dtype = self.dtype)])

        # Padd the last value with zero, so that indexing to -1 is safe when doing RT
        self.em = np.vstack([self.em.T, np.zeros(self.em.shape[0], dtype = self.dtype)]).T

        # Padd the last value with zero, so that indexing to -1 is safe when doing RT
        self.op = np.vstack([self.op.T, np.zeros(self.op.shape[0], dtype = self.dtype)]).T

        self.cell_index_to_gasspydb = np.hstack((self.cell_index_to_gasspydb,[-1]))

        self.raydump_dict["cell_index"] = np.vstack((self.raydump_dict["cell_index"],np.full(self.raydump_dict['NcellPerRaySeg'], -1)))

        self.raydump_dict["ray_index0"] = np.append(self.raydump_dict["ray_index0"],[-1])

        if self.opc_per_NH:
            self.den = np.append(self.den, np.array([0], dtype = self.dtype))

    def move_to_GPU(self):
        if self.accel == "torch" and not self.liteVRAM:
            self.raydump_dict["pathlength"] = cupy.asarray(self.raydump_dict["pathlength"], dtype=self.dtype)
            self.raydump_dict["cell_index"] = cupy.asarray(self.raydump_dict["cell_index"], dtype=cupy.int64)
            self.em = cupy.asarray(self.em, dtype=self.dtype)
            self.op = cupy.asarray(self.op, dtype=self.dtype)
            self.den = cupy.asarray(self.den, dtype=self.dtype)
            self.cell_index_to_gasspydb = cupy.asarray(self.cell_index_to_gasspydb)


        if self.accel == "CUDA":
            self.energy = cupy.asarray(self.energy)

        if self.accel == "torch":
            self.energy = torch.as_tensor(self.energy, device = self.cuda_device)

    def cleaning(self):
        """These cleaning operations ensure that the input data conforms to expectations of the radiative transfer algorithms""" 
        # Cleaning operations
        self.raydump_dict["pathlength"][self.raydump_dict["pathlength"]<0] = 0.

        # A ray may die outside the box, and as a result have an index larger than the simulation. We set that to -1.
        self.raydump_dict["cell_index"][self.raydump_dict["cell_index"] > len(self.cell_index_to_gasspydb) - 1] = -1

        #del(self.raydump_dict['splitEvents'])
        #del(self.saved3d)      

    def set_branch(self, root_i):
        """ Get the branch of root i"""
        self.max_level = 0
        level = 0

        # This initalizes the trace down from parent
        gid = self.ancenstors[root_i]
        new_parent_gids = cupy.array([gid], dtype=cupy.int64)
        self.branch[level] = self.numlib.int(gid)

        eol = -1

        while eol < 4**(level-1):
            level += 1
            eol = 0
            # Initialize the current level
            if level in self.branch:
                self.branch[level][:] = -1
            else:
                self.branch[level] = np.full(4**(level), -1, dtype=np.int64)
 
            # Find all the children associated with these rays
            exists = False
            for parent_i, parent_gid in enumerate(new_parent_gids):
                parent_gid = int(parent_gid)
                if parent_gid != -1:
                    if parent_gid in self.split_by_gid_tree:
                        self.branch[level][4*parent_i:4*parent_i+4] = \
                            np.asarray(self.split_by_gid_tree[parent_gid], dtype=np.int64)[:]
                        exists = True
                    else:
                        eol += 1
                else:
                    eol += 1
            if not exists:
                del(self.branch[level])
                level -=1
                break

            # Set these as new parents
            new_parent_gids = self.branch[level]

        # for record keeping in the class of the maximum level reached relative to the parent
        self.max_level = level

    def info(self):
        for key in self.raydump_dict.keys():
            try:
                if type(self.raydump_dict[key]) == dict:
                    print(key, self.raydump_dict[key].shape)    
                print(key, self.raydump_dict[key].shape)
            except:
                print(key, self.raydump_dict[key])

    def get_spec_root(self, root_i, cuda_device):
        self.set_branch(root_i)

        rt_tetris_maps = {}
        rt_tetris_maps[0] = cupy.arange(1)

        if self.accel == "torch":
            rt_tetris_maps[0] = torch.as_tensor(rt_tetris_maps[0])

        my_segment_IDs = {0: [None]}

        my_N_spect = self.Nraster**self.max_level
        #output_array_cpu = cupyx.zeros_pinned((my_N_spect, self.energy.shape[0]))

        if self.accel == "torch":
            output_array_gpu = torch.zeros((self.energy.shape[0], my_N_spect), requires_grad=False, device=cuda_device, dtype=self.torch_dtype)

        my_l_i = 0
        # These are the ray dumps of the first root

        branch_gid = self.branch[0]
        
        i0, Nsegs = self.raydump_dict["ray_index0"][branch_gid], self.raydump_dict["Nsegs"][branch_gid]
        i1 = i0+Nsegs

        my_cell_indexes = self.raydump_dict["cell_index"][i0:i1]
        gasspy_id = self.cell_index_to_gasspydb[my_cell_indexes]
        save_GIDs = {my_l_i:np.array([branch_gid])}

        # Check if using Tensors
        if self.accel == "torch":
            my_Em = torch.as_tensor(self.em.take(gasspy_id, axis=1), device=cuda_device)
            my_Opc = torch.as_tensor(self.op.take(gasspy_id, axis=1), device=cuda_device)
            spectra_shape = my_Em.shape
            my_pathlenths = torch.as_tensor(self.raydump_dict["pathlength"][i0:i1], device=cuda_device)
            

            if self.opc_per_NH:
                my_den = torch.as_tensor(self.den.take(my_cell_indexes), device=cuda_device)
                my_Opc = torch.mul(my_den, my_Opc)

            cum_Opc = torch.cumsum(torch.flip(torch.multiply(my_Opc, my_pathlenths), dims = [1,2]), dim=2)
            cum_Opc[:,1:,:] = torch.add(cum_Opc[:,1:,:], torch.cumsum(cum_Opc[:,: -1,-1], dim = 1)[:,:,None])
            cum_Opc = torch.flip(cum_Opc, dims = [1,2])
            dF = torch.mul(torch.mul(my_Em, my_pathlenths),torch.exp(-cum_Opc)).sum(axis=[2,1])
            #dTau = torch.exp(-torch.mul(my_Opc,my_pathlenths).sum(axis=[2, 1]))

        output_array_gpu[:, 0] = dF[:]# * dTau[:]
        for my_l_i, my_l in enumerate(list(self.branch.values())[1:]):
            
            # Check for the number of dead rays. Some times dumps contain all dead rays. 
            dead_index = np.argwhere(self.branch[my_l_i+1] == -1)

            # If there are live rays in this level, proceed
            if len(dead_index) != len(self.branch[my_l_i+1]):
                my_l_i += 1

                Nsegs_branch = self.numlib.array([self.raydump_dict["Nsegs"][gid] for gid in my_l])

                save_GIDs[my_l_i] = my_l.copy()

                save_GIDs[my_l_i][dead_index] = save_GIDs[my_l_i-1][dead_index // self.Nraster]

                try:
                    rt_tetris_maps[my_l_i]
                except:
                    rt_tetris_maps[my_l_i] = cupy.arange(int(self.Nraster**(my_l_i)))
                    if self.accel == "torch":
                        rt_tetris_maps[my_l_i] = torch.as_tensor(rt_tetris_maps[my_l_i], device=cuda_device)

                my_segment_IDs[my_l_i] = self.numlib.full((len(self.branch[my_l_i]),self.numlib.int(Nsegs_branch.max())), -1, dtype=np.int64)

                for branch_gid_i, branch_gid in enumerate(my_l):
                    i0, Nsegs = self.raydump_dict["ray_index0"][branch_gid], self.raydump_dict["Nsegs"][branch_gid]
                    i1 = i0+Nsegs                
                    my_segment_IDs[my_l_i][branch_gid_i, :Nsegs] = self.numlib.arange(self.numlib.int(i0),self.numlib.int(i1))[:]
                
                # Using take we can get each branch at at time, returning a self.Nraster^level set of arrays which each contain many different path lengths etc.
                my_cell_indexes = self.raydump_dict["cell_index"].take(my_segment_IDs[my_l_i], axis=0)
                gasspy_id = self.cell_index_to_gasspydb[my_cell_indexes]
                
                if self.accel == "torch":
                    ray_dir = torch.as_tensor(cupy.array([self.new_global_rays.raydir_x[my_l], self.new_global_rays.raydir_y[my_l], self.new_global_rays.raydir_x[my_l]]), device = cuda_device)
                    ray_dir = ray_dir.reshape(ray_dir.shape + (1,1))

                    my_pathlenths = torch.as_tensor(self.raydump_dict["pathlength"].take(my_segment_IDs[my_l_i], axis=0), device=cuda_device)
                    my_Em = torch.as_tensor(self.em.take(gasspy_id, axis=1), device=cuda_device)
                    my_Opc = torch.as_tensor(self.op.take(gasspy_id, axis=1), device=cuda_device)


                    if self.opc_per_NH:
                        my_den = torch.as_tensor(self.den.take(my_cell_indexes), device=cuda_device)
                        my_Opc = torch.mul(my_den, my_Opc)

                    cum_Opc = torch.cumsum(torch.flip(torch.multiply(my_Opc, my_pathlenths), dims = [2,3]), dim=3)
                    cum_Opc[:,:,1:,:] = torch.add(cum_Opc[:,:,1:,:], torch.cumsum(cum_Opc[:,:,: -1,-1], dim = 2)[:,:,:,None])
                    cum_Opc = torch.flip(cum_Opc, dims = [2,3])
                    dF = torch.mul(torch.mul(my_Em, my_pathlenths),torch.exp(-cum_Opc)).sum(axis=[3,2])
                    dTau = torch.exp(-torch.mul(my_Opc, my_pathlenths).sum(axis=[3, 2]))
                    output_array_gpu[:,rt_tetris_maps[my_l_i]] = ((output_array_gpu[:,torch.tile(rt_tetris_maps[my_l_i-1],dims=(self.Nraster,1)).T.ravel()]* dTau[:]) + dF[:])

                else:
                    my_pathlenths = cupy.asarray(self.raydump_dict["pathlength"].take(my_segment_IDs[my_l_i], axis=0))
                    my_Em = cupy.asarray(self.em.take(gasspy_id, axis=1))
                    my_Opc = cupy.asarray(self.op.take(gasspy_id, axis=1))

                    if self.opc_per_NH:
                        my_den = cupy.as_array(self.den.take(my_cell_indexes), device=cuda_device)
                        my_Opc = cupy.multiply(my_den, my_Opc)

                    cum_Opc = cupy.cumsum(cupy.multiply(my_Opc, my_pathlenths), axis=3)
                    cum_Opc[:,:,1:,:] = cupy.add(cum_Opc[:,:,1:,:], cupy.expand_dims(cupy.cumsum(cum_Opc[:,:,: -1,-1], axis = 2),-1))

                    dF = cupy.multiply(cupy.multiply(my_Em[:, :],my_pathlenths),cupy.exp(-cum_Opc)).sum(axis=[2, 3])
                    dTau = cupy.exp((-my_Opc[:, :] * my_pathlenths).sum(axis=[2, 3]))
                    output_array_gpu[:,rt_tetris_maps[my_l_i]] = (output_array_gpu[cupy.tile(rt_tetris_maps[my_l_i-1],self.Nraster)] * dTau[:] + dF[:])

            # Per child...
            # Using the 1D simulation/model index extract the emissivity and opacity
            # my_index1Ds = self.raydump_dict["index1D"][my_segment_IDs[my_l_i],:].ravel()
            # my_pathlenths = self.raydump_dict["pathlength"][my_segment_IDs[my_l_i],:].ravel()
            # my_Em = self.em[:, my_index1Ds.ravel()]
            # my_Opc = self.op[:, my_index1Ds.ravel()]
            # dF = (my_Em[:,:] * my_pathlenths).sum(axis=1)
            # dTau = cupy.exp((-my_Opc[:,:] * my_pathlenths).sum(axis=1))
            # output_array_gpu[rt_tetris_maps[my_l_i]] = (output_array_gpu[rt_tetris_maps[my_l_i]] + dF[:]) * dTau[:]

        if self.spec_save_type == "hdf5":
            out_GIDs, out_GID_i = np.unique(save_GIDs[my_l_i], return_index=True)

            if self.accel.lower() == "torch":
                save_data = {'flux':output_array_gpu[:,out_GID_i].cpu().numpy()}

            if self.liteVRAM:
                save_data.update({'x':self.new_global_rays.xp[out_GIDs],
                'y':self.new_global_rays.yp[out_GIDs],
                'ray_lrefine':self.new_global_rays.ray_lrefine[out_GIDs]})
            else:
                save_data.update({'x':self.new_global_rays.xp[out_GIDs].get(),
                'y':self.new_global_rays.yp[out_GIDs].get(),
                'ray_lrefine':self.new_global_rays.ray_lrefine[out_GIDs].get()})
            self.write_spec_save_hdf5(save_data)
        elif self.spec_save_type == "numpy":
            np.save("%s%s%sspec_%i.npy"%(self.root_dir, self.gasspy_subdir, self.gasspy_spec_subdir, root_i), output_array_gpu.cpu().numpy())

