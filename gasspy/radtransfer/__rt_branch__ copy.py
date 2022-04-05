"""
This routine performs a spectra RT on an AMR grid
"""
from inspect import trace
import sys, os
from pathlib import Path
import pickle
import numpy as np
import cupy
import torch
import h5py
from astropy.io import fits
from astropy import constants as const
from gasspy.raystructures import global_ray_class
import yaml

#cupy.cuda.set_allocator(cupy.cuda.MemoryAsyncPool().malloc)
cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)

class FamilyTree():
    def __init__(
        self, root_dir="./",
        gasspy_subdir="GASSPY",
        gasspy_spec_subdir="spec",
        gasspy_projection_subdir="projections",
        traced_rays=None,
        global_rayDF_deprecated=None,
        energy=None,
        energy_lims=None,
        Emask=None,
        em=None,
        op=None,
        opc_per_NH=False,
        saved3d=None,
        vel=None,
        den=None,
        massden=True,
        mu=1.1,
        los_angle=None,
        accel="cuda",
        dtype=np.float32,
        liteVRAM=False,
        Nraster=4,
        useGasspyEnergyWindows=True,
        make_spec_subdirs=True,
        config_yaml=None,
        spec_save_type="hdf5",
        spec_save_name="gasspy_spec",
        cuda_device=None
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

        if make_spec_subdirs:
            spec_subdir_path = root_dir+gasspy_subdir+gasspy_spec_subdir
            os.makedirs(spec_subdir_path, exist_ok = True) 

        if not isinstance(traced_rays, h5py._hl.files.File):
            assert isinstance(traced_rays, str), "provided traced rays is neither a string or open hd5 file"
            if not traced_rays.endswith(".hdf5"):
                traced_rays += ".hdf5"
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
        self.saved3d = saved3d
        self.traced_rays = traced_rays
        self.global_rayDF_deprecated = global_rayDF_deprecated
        self.config_yaml = config_yaml

        self.multiply = cupy.ElementwiseKernel(
        'float64 x, float64 y', 'float64 z',
        '''
        z = x*y;
        ''', 'multipy')

        self.extinct = cupy.ElementwiseKernel(
        'float64 x, float64 y', 'float64 z',
        '''
        z = exp(-x*y);
        ''', 'extinction')


    def process_all(self,i0=0):
        
        for root_i in range(i0, len(self.ancenstors)):
            self.get_spec_root(root_i, self.cuda_device)
            if root_i % 1000 == 0:
                print(root_i, end="\r")
        
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

        self.spechdf5_out.create_dataset("flux", (init_size, len(self.energy)), maxshape=(None,len(self.energy)))
        self.spechdf5_out.create_dataset("x", (init_size,), maxshape=(None,))
        self.spechdf5_out.create_dataset("y", (init_size,), maxshape=(None,))
        self.spechdf5_out.create_dataset("ray_lrefine", (init_size,), dtype="int8", maxshape=(None,))
        if isinstance(self.energy, cupy._core.core.ndarray):
            self.spechdf5_out.create_dataset("E", data=cupy.asnumpy(self.energy))
        elif isinstance(self.energy, torch.Tensor):
            self.spechdf5_out.create_dataset("E", data=self.energy.cpu().numpy())

        self.spec_stream = cupy.cuda.Stream(non_blocking=True)
        self.compute_stream = cupy.cuda.Stream(non_blocking=True)


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

        # Ensure the energy bins are loaded BEFORE the em and op tables to minimize memory used
        self.load_energy_limits()
        self.load_energy_bins()
        self.load_em()
        self.load_op()
        self.load_saved3d()
        self.load_new_global_rays()
        self.load_cell_index_to_gasspydb()
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
        if self.config_yaml is None:
            assert Path(self.root_dir+self.gasspy_subdir+"gasspy_config.yaml").is_file(), "Error: gasspy_config.yaml is not given and does not exists in simulation GASSPY directory"
            self.config_yaml = self.root_dir+self.gasspy_subdir+"gasspy_config.yaml"
                    
        with open(r'%s'%(self.config_yaml)) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            self.config_dict = yaml.load(file, Loader=yaml.FullLoader)
        
        self.sim_unit_length = self.config_dict["sim_unit_length"]

    def load_new_global_rays(self):
        self.new_global_rays = global_ray_class(on_cpu=self.liteVRAM)
        self.new_global_rays.load_hdf5(self.traced_rays_h5file)
        # Select the ancestral GIDs
        self.ancenstors = self.new_global_rays.global_rayid[self.numlib.where(self.new_global_rays.pevid == -1)]

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

            self.em = np.load(tmp_path).astype(self.dtype)

            # Because of the different dimensionality of the energy grid and the emissivity tables, we use compress to 
            # produce a reduced op limited to the region of interest
            if self.Emask is not None:
                self.em = np.compress(self.Emask, self.em, axis=0).astype(self.dtype)

    def load_op(self):
        if str == type(self.op):
            if Path(self.op).is_file():
                tmp_path = self.op
            elif Path(self.root_dir + self.gasspy_subdir + self.op).is_file():
                tmp_path = self.root_dir + self.gasspy_subdir + self.op
            else:
                sys.exit("Could not find path to op")

            self.op = np.load(tmp_path).astype(self.dtype)

            # Because of the different dimensionality of the energy grid and the opacity tables, we use compress to 
            # produce a reduced op limited to the region of interest
            if self.Emask is not None:
                self.op = np.compress(self.Emask, self.op, axis=0).astype(self.dtype)

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
            self.cell_index_to_gasspydb = np.unique(self.saved3d, axis=0, return_inverse=True)[1]

            self.NSimCells = len(self.cell_index_to_gasspydb)

    def load_velocity_data(self):
        if str == type(self.vel):
            if self.vel.endswith("fits"):
                if Path(self.vel).is_file():
                    self.v = fits.open(self.vel)
                    self.v = self.v[0].data
                else:
                    sys.exit("Could not find path to velocity fits file")
            else:
                sys.exit("Unrecognizized format for velocity data file. Contact developers...")

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

        self.raydump_dict["ray_index0"][unique_gid] = i0
        self.raydump_dict["Nsegs"][unique_gid] = Nsegs

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

    def pinned_array(self, array):
        # first constructing pinned memory
        mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
        src = np.frombuffer(
                    mem, array.dtype, array.size).reshape(array.shape)
        src[...] = array
        return src


    def move_to_GPU(self):
        if self.accel == "torch":
            self.energy = torch.as_tensor(self.energy).cuda(self.cuda_device)
        else:
            self.energy = cupy.asarray(self.energy)

        if self.liteVRAM:
            self.raydump_dict["pathlength"] = self.pinned_array(np.array(self.raydump_dict["pathlength"], dtype=self.dtype))
            # self.raydump_dict["cell_index"] = self.pinned_array(np.array(self.raydump_dict["cell_index"], dtype=np.int64))

        else:
            self.raydump_dict["pathlength"] = cupy.asarray(self.raydump_dict["pathlength"], dtype=self.dtype)

        self.raydump_dict["cell_index"] = cupy.asarray(self.raydump_dict["cell_index"], dtype=cupy.int64)
        self.em = cupy.asarray(self.em, dtype=self.dtype)
        self.op = cupy.asarray(self.op, dtype=self.dtype)
        self.den = cupy.asarray(self.den, dtype=self.dtype)
        self.cell_index_to_gasspydb = cupy.asarray(self.cell_index_to_gasspydb)

    def cleaning(self):
        """These cleaning operations ensure that the input data conforms to expectations of the radiative transfer algorithms""" 
        # Cleaning operations
        self.raydump_dict["pathlength"][self.raydump_dict["pathlength"]<0] = 0.

        # A ray may die outside the box, and as a result have an index larger than the simulation. We set that to -1.
        self.raydump_dict["cell_index"][self.raydump_dict["cell_index"] > len(self.cell_index_to_gasspydb) - 1] = -1

        del(self.raydump_dict['splitEvents'])
        del(self.saved3d)      

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
        with self.compute_stream:
            self.set_branch(root_i)

            rt_tetris_maps = {}

            """
            By default the index map exists on the GPU
            """
            rt_tetris_maps[0] = cupy.arange(1)

            if self.accel == "torch":
                rt_tetris_maps[0] = torch.as_tensor(rt_tetris_maps[0])


            my_segment_IDs = {0: [None]}

            my_N_spect = self.Nraster**self.max_level
            #output_array_cpu = cupyx.zeros_pinned((my_N_spect, self.energy.shape[0]))

            if self.accel == "torch":
                output_array_gpu = torch.zeros((self.energy.shape[0], my_N_spect), requires_grad=False, device=cuda_device, dtype=self.torch_dtype)
            elif self.accel == "cuda":
                output_array_gpu = cupy.zeros((self.energy.shape[0], my_N_spect), dtype=self.dtype)

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

                my_pathlenths = torch.as_tensor(self.raydump_dict["pathlength"][i0:i1], device=cuda_device)

                if self.opc_per_NH:
                    my_den = torch.as_tensor(self.den.take(my_cell_indexes), device=cuda_device)
                    my_Opc = torch.mul(my_den, my_Opc)

                dF = torch.mul(my_Em, my_pathlenths).sum(axis=[2,1])
                dTau = torch.exp(-torch.mul(my_Opc,my_pathlenths).sum(axis=[2, 1]))
                
                output_array_gpu[:, 0] = torch.mul(dF, dTau)[:]
                

            elif self.accel == "cuda":
                my_Em  = cupy.asarray(self.em.take(gasspy_id, axis=1))
                my_Opc = cupy.asarray(self.op.take(gasspy_id, axis=1))

                my_pathlenths = cupy.asarray(self.raydump_dict["pathlength"][i0:i1])

                if self.opc_per_NH:
                    my_den = cupy.asarray(self.den.take(my_cell_indexes))
                    my_Opc = cupy.multiply(my_den, my_Opc)

                dF = cupy.multiply(my_Em, my_pathlenths).sum(axis=[2,1])
                dTau = cupy.exp(-cupy.multiply(my_Opc,my_pathlenths).sum(axis=[2, 1]))

                output_array_gpu[:, 0] = cupy.multiply(dF, dTau)[:]

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
                        my_pathlenths = torch.as_tensor(self.raydump_dict["pathlength"].take(my_segment_IDs[my_l_i], axis=0), device=cuda_device)
                        my_Em = torch.as_tensor(self.em.take(gasspy_id, axis=1), device=cuda_device)
                        my_Opc = torch.as_tensor(self.op.take(gasspy_id, axis=1), device=cuda_device)

                        if self.opc_per_NH:
                            my_den = torch.as_tensor(self.den.take(my_cell_indexes), device=cuda_device)
                            my_Opc = torch.mul(my_den, my_Opc)

                        dF = torch.mul(my_Em, my_pathlenths).sum(axis=[3, 2])
                        dTau = torch.exp(-torch.mul(my_Opc, my_pathlenths).sum(axis=[3, 2]))
                        output_array_gpu[:,rt_tetris_maps[my_l_i]] = torch.mul(torch.add(output_array_gpu[:,torch.tile(rt_tetris_maps[my_l_i-1],dims=(self.Nraster,1)).T.ravel()], dF[:]), dTau[:])

                    else:
                        my_pathlenths = cupy.asarray(self.raydump_dict["pathlength"].take(my_segment_IDs[my_l_i], axis=0))
                        my_Em = cupy.asarray(self.em.take(gasspy_id, axis=1))
                        my_Opc = cupy.asarray(self.op.take(gasspy_id, axis=1))

                        if self.opc_per_NH:
                            my_den = cupy.asarray(self.den.take(my_cell_indexes))
                            my_Opc = cupy.multiply(my_den, my_Opc)

                        output_array_gpu[:,rt_tetris_maps[my_l_i]] = cupy.multiply(cupy.add(output_array_gpu[:,cupy.tile(rt_tetris_maps[my_l_i-1],self.Nraster)], cupy.multiply(my_Em[:, :],my_pathlenths).sum(axis=[2, 3])), cupy.exp((cupy.multiply(-my_Opc[:, :], my_pathlenths)).sum(axis=[2, 3])))[:]

                        # dF = cupy.multiply(my_Em[:, :],my_pathlenths).sum(axis=[2, 3])
                        # dTau = cupy.exp((cupy.multiply(-my_Opc[:, :], my_pathlenths)).sum(axis=[2, 3]))
                        # output_array_gpu[:,rt_tetris_maps[my_l_i]] = cupy.multiply(cupy.add(output_array_gpu[:,cupy.tile(rt_tetris_maps[my_l_i-1],self.Nraster)], dF), dTau)[:]

            if self.spec_save_type == "hdf5":
                out_GIDs, out_GID_i = np.unique(save_GIDs[my_l_i], return_index=True)

                if self.accel == "torch":
                    save_data = {'flux':output_array_gpu[:,out_GID_i].detach().cpu().numpy()}
                elif self.accel == "cuda":
                    # Note: Should be a stream
                    # We do this to ensure that the outarray doesn't get out of sync while streaming
                    save_array = cupy.array(output_array_gpu[:,out_GID_i])
                    save_data = {'flux':cupy.asnumpy(save_array, stream=self.spec_stream)}

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
        cupy.cuda.get_current_stream().synchronize()

