import sys
import os
import cupy
import torch
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy import constants as const
import yaml
from gasspy.raystructures import global_ray_class

# a class to store the data for rad tranning,
# moving it to the GPU or CPU, and setting functinos dependent on where things are_deterministic_algorithms_enabled

# It is intended to optimized the performance of mixed memory loaded calculations.

class RT_DATA_Controller():
    def __init__(self):
        pass

    # To do:
    # If self.em is [tensor (cpu or gpu) or cupy or numpy] then
    # def grab_em = (function to get em including if indicies are on GPU or CPU)
    # same for emissivity and each indicies. Each gets a custom function depending on where they live
    # That way each data product gets a flag about where to put it, (or use flag above)
    # flags: tensor, tensor_cpu, numpy, cupy

    def move_to_GPU(self):
        self.energy = torch.as_tensor(self.energy).cuda(self.cuda_device)
        self.raydump_dict["cell_index"]["cpu"] = torch.Tensor.pin_memory(torch.tensor(self.raydump_dict["cell_index"]["cpu"]))

        if self.liteVRAM:
            self.raydump_dict["pathlength"] = self.pinned_array(
                np.array(self.raydump_dict["pathlength"], dtype=self.dtype))
            self.em = torch.Torch.pinned_memory(torch.Tensor(self.em))
            self.op = torch.Torch.pinned_memory(torch.Tensor(self.op))
            self.den = torch.Torch.pinned_memory(torch.Tensor(self.den))

            self.pathlength_device = "cpu"
            self.em_device = "cpu"
            self.op_device = "cpu"
            self.den_device = "cpu"

        else:
            self.pathlength_device = "cpu"
            self.den_device = "cpu"
            self.em_device = self.cuda_device
            self.op_device = self.cuda_device


            self.raydump_dict["pathlength"] = torch.Tensor.pin_memory(torch.as_tensor(self.raydump_dict["pathlength"].astype(self.dtype), device=self.pathlength_device))

            # self.pathlength_device = self.cuda_device
            # self.em = cupy.asarray(self.em, dtype=self.dtype)
            # self.em_device = self.cuda_device
            # self.op = cupy.asarray(self.op, dtype=self.dtype)
            # self.op_device = self.cuda_device
            # self.den = cupy.asarray(self.den, dtype=self.dtype)
            # self.den_device = self.cuda_device

            self.em = torch.as_tensor(self.em.astype(self.dtype)).cuda()
            self.op = torch.as_tensor(self.op.astype(self.dtype)).cuda()
            self.den = torch.as_tensor(self.den.astype(self.dtype), device=self.den_device)

            self.cell_index_to_gasspydb["cpu"] = torch.Tensor.pin_memory(torch.as_tensor(self.cell_index_to_gasspydb["cpu"]))
            self.cell_index_to_gasspydb[self.cuda_device] = self.cell_index_to_gasspydb["cpu"].cuda()
        
    def padding(self):
        # The last element of each of the following arrays is assumed to be zero, and is used for out of bound indexes, which will have values -1.
        self.raydump_dict["pathlength"] = np.vstack([self.raydump_dict["pathlength"], np.zeros(
            self.raydump_dict["pathlength"].shape[1], dtype=self.dtype)])

        # Padd the last value with zero, so that indexing to -1 is safe when doing RT
        self.em = np.vstack([self.em.T, np.zeros(
            self.em.shape[0], dtype=self.dtype)]).T

        # Padd the last value with zero, so that indexing to -1 is safe when doing RT
        self.op = np.vstack([self.op.T, np.zeros(
            self.op.shape[0], dtype=self.dtype)]).T

        self.cell_index_to_gasspydb['cpu'] = np.hstack(
            (self.cell_index_to_gasspydb['cpu'], [-1]))

        self.raydump_dict["cell_index"]["cpu"] = np.vstack(
            (self.raydump_dict["cell_index"]["cpu"], np.full(self.raydump_dict['NcellPerRaySeg'], -1)))

        self.raydump_dict["ray_index0"] = np.append(
            self.raydump_dict["ray_index0"], [-1])

        if self.opc_per_NH:
            self.den = np.append(self.den, np.array([0], dtype=self.dtype))

    def set_precision(self, new_dtype):
        self.dtype = new_dtype
        self.torch_dtype = torch.as_tensor(
            np.array([], dtype=self.dtype)).dtype

        self.em = self.em.astype(new_dtype)
        self.op = self.op.astype(new_dtype)
        self.raydump_dict["pathlength"] = self.raydump_dict["pathlength"].astype(
            new_dtype)
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
            assert Path(self.root_dir+self.gasspy_subdir+"gasspy_config.yaml").is_file(
            ), "Error: gasspy_config.yaml is not given and does not exists in simulation GASSPY directory"
            self.config_yaml = self.root_dir+self.gasspy_subdir+"gasspy_config.yaml"

        with open(r'%s' % (self.config_yaml)) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            self.config_dict = yaml.load(file, Loader=yaml.FullLoader)

        self.sim_unit_length = self.config_dict["sim_unit_length"]

    def load_global_rays(self):
        self.new_global_rays = global_ray_class(on_cpu=self.liteVRAM)
        self.new_global_rays.load_hdf5(self.traced_rays_h5file)
        # Select the ancestral GIDs
        self.ancenstors = self.new_global_rays.global_rayid[self.numlib.where(
            self.new_global_rays.pevid == -1)]
        self.N_ancestors = len(self.ancenstors)

    def load_energy_limits(self):
        if self.useGasspyEnergyWindows and self.energy_lims is None:
            if Path(self.root_dir+self.gasspy_subdir+"gasspy_continuum_mesh_windows.txt").is_file():
                self.energy_lims = self.root_dir+self.gasspy_subdir + \
                    "gasspy_continuum_mesh_windows.txt"
            elif "CLOUDY_DATA_PATH" in os.environ:
                if Path(os.environ['CLOUDY_DATA_PATH']+"gasspy_continuum_mesh_windows.txt").is_file():
                    self.energy_lims = os.environ['CLOUDY_DATA_PATH'] + \
                        "gasspy_continuum_mesh_windows.txt"

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
                self.Emask = np.full(self.energy.shape, False, dtype=np.bool8)
                for i in range(len(self.energy_lims)):
                    self.Emask[(self.energy >= self.energy_lims[i][0])
                               * (self.energy <= self.energy_lims[i][1])] = True

            if self.Emask is not None:
                self.energy = self.energy[self.Emask]

        if save:
            np.save(self.root_dir + self.gasspy_subdir +
                    self.gasspy_spec_subdir + "windowed_energy.npy", self.energy)

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
                self.em = np.compress(
                    self.Emask, self.em, axis=0).astype(self.dtype)

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
                self.op = np.compress(
                    self.Emask, self.op, axis=0).astype(self.dtype)

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
        self.cell_index_to_gasspydb = {}
        self.cell_index_to_gasspydb["cpu"] = np.unique(
            self.saved3d, axis=0, return_inverse=True)[1]
        
        del(self.saved3d)
        
        self.cell_index_to_gasspydb[self.cuda_device] = cupy.asarray(self.cell_index_to_gasspydb["cpu"])

        self.NSimCells = len(self.cell_index_to_gasspydb["cpu"])

    def load_velocity_data(self):
        if str == type(self.vel):
            if self.vel.endswith("fits"):
                if Path(self.vel).is_file():
                    self.v = fits.open(self.vel)
                    self.v = self.v[0].data
                else:
                    sys.exit("Could not find path to velocity fits file")
            else:
                sys.exit(
                    "Unrecognizized format for velocity data file. Contact developers...")

    def load_traced_rays(self):
        self.raydump_dict['segment_global_rayid'] = self.traced_rays_h5file['ray_segments']['global_rayid'][:]
        self.raydump_dict["pathlength"] = self.traced_rays_h5file["ray_segments"]["pathlength"][:, :].astype(
            self.dtype)*self.dtype(self.sim_unit_length)
        self.raydump_dict["cell_index"] = {}
        self.raydump_dict["cell_index"]["cpu"] = self.traced_rays_h5file["ray_segments"]["cell_index"][:, :]
        self.raydump_dict["splitEvents"] = self.traced_rays_h5file["splitEvents"][:, :]

        maxGID = self.numlib.int(self.new_global_rays.global_rayid.max())

        # Initialize the raydump N_segs and index into ray_buffer_dumps with -1
        self.raydump_dict["ray_index0"] = self.numlib.full(
            maxGID+1, -1).astype(np.int64)
        self.raydump_dict["Nsegs"] = self.numlib.full(
            maxGID+1, -1).astype(np.int64)

        # Get the unique values and the first index of array with that value, and Counts.
        # This ONLY works because segment global_rayid is sorted.
        unique_gid, i0, Nsegs = np.unique(
            self.raydump_dict['segment_global_rayid'], return_counts=True, return_index=True)

        self.raydump_dict["ray_index0"][unique_gid] = i0
        self.raydump_dict["Nsegs"][unique_gid] = Nsegs

        self.raydump_dict['NcellPerRaySeg'] = self.raydump_dict["pathlength"].shape[1]
        self.family_tree = {}
        self.max_level = 0
        self.create_split_event_dict()

    def create_split_event_dict(self):
        self.raydump_dict['splitEvents'] = cupy.asnumpy(
            self.raydump_dict['splitEvents'])
        # Create a dictionary with each split event keyed by the GID of teh parent
        self.split_by_gid_tree = dict(zip(self.raydump_dict['splitEvents'][:, 0].astype(
            np.int32), zip(*self.raydump_dict['splitEvents'][:, 1:].astype(np.int32).T)))
