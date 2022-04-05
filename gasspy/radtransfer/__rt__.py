"""
This routine performs a spectra RT on an AMR grid
"""
import sys
import os
from pathlib import Path
import numpy as np
import cupy
import torch
import h5py
from astropy import constants as const
from torch import device
from .rt_data_class import RT_DATA_Controller
from gasspy.radtransfer.hdf5_writer import HDF5_SAVE #, RT_IO
#from gasspy.radtransfer import RadiativeTransferData

#cupy.cuda.set_allocator(cupy.cuda.MemoryAsyncPool().malloc)
#cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
torch.set_grad_enabled(False)

class FamilyTree(RT_DATA_Controller,HDF5_SAVE):
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
        dtype=np.float64,
        liteVRAM=False,
        Nraster=4,
        useGasspyEnergyWindows=True,
        make_spec_subdirs=True,
        config_yaml=None,
        spec_save_type=None,
        spec_save_name="gasspy_spec",
        cuda_device=None,
        N_max_spec=64
    ):
        self.cuda_device = cuda_device
        self.dtype = dtype
        self.torch_dtype = torch.as_tensor(
            np.array([], dtype=self.dtype)).dtype

        self.mempool = cupy.get_default_memory_pool()

        if liteVRAM:
            self.numlib = np
        else:
            self.numlib = cupy

        if str == type(root_dir):
            assert Path(root_dir).is_dir(), "Provided root_dir \"" + \
                root_dir+"\"is not a directory"
            self.root_dir = root_dir

        assert type(gasspy_subdir) == str, "gasspy_subdir not a string"
        if not gasspy_subdir.endswith("/"):
            gasspy_subdir = gasspy_subdir + "/"
        if not gasspy_subdir[0] == "/":
            gasspy_subdir = "/" + gasspy_subdir
        self.gasspy_subdir = gasspy_subdir
        assert Path(
            self.root_dir+self.gasspy_subdir).is_dir() == True, "GASSPY subdir does not exist..."

        assert type(
            gasspy_projection_subdir) == str, "gasspy_projection_subdir not a string"
        if not gasspy_projection_subdir[0] == "/":
            gasspy_projection_subdir = "/" + gasspy_projection_subdir
        if not gasspy_spec_subdir.endswith("/"):
            gasspy_projection_subdir = gasspy_projection_subdir + "/"
        self.gasspy_projection_subdir = gasspy_projection_subdir
        assert Path(self.root_dir+self.gasspy_subdir+self.gasspy_projection_subdir).is_dir(
        ) == True, "GASSPY projections dir does not exist..."

        assert type(
            gasspy_spec_subdir) == str, "gasspy_spec_subdir not a string"
        if not gasspy_spec_subdir[0] == "/":
            gasspy_spec_subdir = "/" + gasspy_spec_subdir
        if not gasspy_spec_subdir.endswith("/"):
            gasspy_spec_subdir = gasspy_spec_subdir + "/"
        self.gasspy_spec_subdir = gasspy_spec_subdir

        if make_spec_subdirs:
            spec_subdir_path = root_dir+gasspy_subdir+gasspy_spec_subdir
            os.makedirs(spec_subdir_path, exist_ok=True)

        if not isinstance(traced_rays, h5py._hl.files.File):
            assert isinstance(
                traced_rays, str), "provided traced rays is neither a string or open hd5 file"
            if not traced_rays.endswith(".hdf5"):
                traced_rays += ".hdf5"
            if Path(traced_rays).is_file():
                tmp_path = traced_rays
            elif Path(self.root_dir+self.gasspy_subdir+self.gasspy_projection_subdir+traced_rays).is_file():
                tmp_path = self.root_dir+self.gasspy_subdir + \
                    self.gasspy_projection_subdir+traced_rays
            else:
                sys.exit("Could not find the traced rays file\n" +
                         "Provided path: %s" % traced_rays +
                         "Try looking in \"./\" and %s\n" % (self.root_dir+self.gasspy_projection_subdir) +
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

        assert isinstance(N_max_spec, int), "N_max_spec value is not int" 
        self.N_max_spec = N_max_spec

    def process_all(self, i0=0, i_list=None):
        if i_list != None:
            for root_i in i_list:
                print(root_i, end="\r")
                self.get_spec_root(root_i, self.cuda_device)

        else:
            root_i = i0
            while root_i < len(self.ancenstors)-1:
                self.get_spec_root(root_i, self.cuda_device)

                # Add to root_i the length of root branches calculated in this go
                root_i += len(self.branch[0])

        self.close_spec_save_hdf5()

    def load_all(self):
        self.load_config_yaml()

        # Ensure the energy bins are loaded BEFORE the em and op tables to minimize memory used
        self.load_energy_limits()
        self.load_energy_bins()
        self.load_em()
        self.load_op()
        self.load_saved3d()
        self.load_global_rays()
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


    def pinned_array(self, array):
        # first constructing pinned memory
        mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
        src = np.frombuffer(
            mem, array.dtype, array.size).reshape(array.shape)
        src[...] = array
        return src

    def cleaning(self):
        """These cleaning operations ensure that the input data conforms to expectations of the radiative transfer algorithms"""
        # Cleaning operations
        self.raydump_dict["pathlength"][self.raydump_dict["pathlength"] < 0] = 0.

        # A ray may die outside the box, and as a result have an index larger than the simulation. We set that to -1.
        self.raydump_dict["cell_index"]["cpu"][self.raydump_dict["cell_index"]["cpu"] > len(self.cell_index_to_gasspydb) - 1] = -1

        try:
            del(self.saved3d)
        except:
            "I guess it's already gone"
        del(self.raydump_dict['splitEvents'])

    def gather_roots(self, root_i):
        N_branches = 0
        self.set_branch(root_i)
        tmp_branches = self.branch.copy()
        N_spec = self.Nraster**self.max_level

        max_level = self.max_level

        while (N_spec <= self.N_max_spec) and (root_i + N_branches < self.N_ancestors -1):
            N_branches += 1
            self.set_branch(root_i+N_branches)

            # Add the next branches.
            N_spec += self.Nraster**self.max_level

            # Assure that this next branch has the same number of levels as the current
            if len(self.branch) == len(tmp_branches) and (N_spec <= self.N_max_spec):
                for l in self.branch.keys():
                    tmp_branches[l] = np.append(
                        tmp_branches[l], self.branch[l])
            else:
                break
        
        self.max_level = max_level
        self.branch = tmp_branches

    def set_branch(self, root_i):
        """ Get the branch of root i"""
        self.max_level = 0
        level = 0
        self.branch = {}

        if root_i % 1000 == 0:
            print(root_i, end="\r")

        # This initalizes the trace down from parent
        gid = self.ancenstors[root_i]

        new_parent_gids = cupy.array([gid], dtype=cupy.int64)

        self.branch[level] = np.array([root_i])
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
                            np.asarray(
                                self.split_by_gid_tree[parent_gid], dtype=np.int64)[:]
                        exists = True
                    else:
                        eol += 1
                else:
                    eol += 1
            if not exists:
                del(self.branch[level])
                level -= 1
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
        torch.cuda.empty_cache()
        self.mempool.free_all_blocks()
        my_cell_indexes = {"cpu":None, cuda_device:None}

        if True:
            self.gather_roots(root_i)
            rt_tetris_maps = {}

            """
            By default the index map exists on the GPU
            """

            # In v2, or at least v1.5, the root branch is no longer an integer, but an abitrary number of roots
            # to speed up smaller branch calculations. Also, this opens up the possibility of totally arbitrary number of spectra
            self.N_roots = len(self.branch[0])
            rt_tetris_maps[0] = cupy.arange(self.N_roots)

            if self.accel == "torch":
                rt_tetris_maps[0] = {'old': torch.as_tensor(rt_tetris_maps[0])}
                rt_tetris_maps[0]['new'] = rt_tetris_maps[0]['old']

            my_raydump_indexes = {"cpu":{0: [None]}, cuda_device:{0: [None]}}

            my_N_spect = self.N_roots * self.Nraster ** self.max_level
            #output_array_cpu = cupyx.zeros_pinned((my_N_spect, self.energy.shape[0]))

            if self.accel == "torch":
                output_array_gpu = torch.zeros(
                    (self.energy.shape[0], my_N_spect), requires_grad=False, device=cuda_device, dtype=self.torch_dtype)
            elif self.accel == "cuda":
                output_array_gpu = cupy.zeros(
                    (self.energy.shape[0], my_N_spect), dtype=self.dtype)

            save_GIDs = {}

            # Loop over each refinement level of this root
            for my_l_i, my_l in enumerate(list(self.branch.values())):

                # Check for the level for dead rays. Some times dumps contain all dead rays.
                dead_index = np.argwhere(self.branch[my_l_i] == -1)

                # If there are live rays in this level, proceed
                if len(dead_index) != len(self.branch[my_l_i]):

                    # This collects the number of buffer dumps per ray, referred to as a segment
                    Nsegs_per_ray = self.numlib.array(
                        [self.raydump_dict["Nsegs"][gid] for gid in my_l])

                    # We save the GID of each ray in this level. Dead rays have a -1 ID, so 
                    # thir GID need to be set to the previously saved GID of that raster.
                    # First we start by copying all the GIDs of the current level, including dead 
                    save_GIDs[my_l_i] = my_l.copy()

                    # If we are not at the first level, which by definition are not dead rays
                    if my_l_i > 0:
                        # Since each ray is split by Nraster we known what GID it originated from in
                        # the previous level. That is the dead_index // Nraster
                        save_GIDs[my_l_i][dead_index] = save_GIDs[my_l_i -
                                                                  1][dead_index // self.Nraster]

                        # Totally unrelated to GIDs, is the tetris map, which tells us 
                        if self.accel == "torch":
                            rt_tetris_maps[my_l_i] ={}
                            rt_tetris_maps[my_l_i]["old"] = torch.repeat_interleave(rt_tetris_maps[my_l_i-1]["new"], self.Nraster)
                            rt_tetris_maps[my_l_i]["new"] = torch.arange(self.N_roots * self.Nraster**(my_l_i))

                        elif self.accel == "cuda":
                            rt_tetris_maps[my_l_i] = {
                                "old": cupy.repeat(rt_tetris_maps[my_l_i-1]["new"], self.Nraster),
                                "new": cupy.arange(my_N_spect*self.Nraster**my_l_i)
                            }

                    for pair in [[np,"cpu"], [cupy, cuda_device]]:
                        numlib, dev = pair
                        my_raydump_indexes[dev][my_l_i] = torch.full((len(self.branch[my_l_i]), numlib.int(Nsegs_per_ray.max())), -1, device=dev)

                        for branch_gid_i, branch_gid in enumerate(my_l):
                            i0, Nsegs = self.raydump_dict["ray_index0"][branch_gid], self.raydump_dict["Nsegs"][branch_gid]
                            i1 = i0+Nsegs
                            my_raydump_indexes[dev][my_l_i][branch_gid_i, :numlib.int(Nsegs)] = torch.arange(
                                numlib.int(i0), numlib.int(i1))[:]

                    # Only torch currently supported, as it's an order of magnitude faster and nearly memory bandwidth limited
                    my_pathlenths = self.raydump_dict["pathlength"][my_raydump_indexes[self.pathlength_device][my_l_i]].cuda()

                    # While not always the most efficient, reshaping the dumps to a single dump times cells is generally faster.
                    # This also removes one additional cumsum to combine them later.
                    tmp_nrays, tmp_ndumps, tmp_cells = my_pathlenths.shape

                    # Remove on dimension from pathlengths
                    my_pathlenths = my_pathlenths.reshape((tmp_nrays, tmp_ndumps*tmp_cells))

                    # Using take we can get each branch at at time, returning a self.Nraster^level set of arrays which each contain many different path lengths etc.
                    my_cell_indexes["cpu"] = torch.Tensor.pin_memory(self.raydump_dict["cell_index"]["cpu"][my_raydump_indexes["cpu"][my_l_i]].reshape((tmp_nrays, tmp_ndumps*tmp_cells)))


                    # I don't know if it's faster or not to reshape on device, or to copy from host to device. I assume the first. But it's too expensive.
                    # my_cell_indexes[cuda_device] = self.raydump_dict["cell_index"][cuda_device].take(
                    #     my_raydump_indexes[cuda_device][my_l_i], axis=0).reshape((tmp_nrays, tmp_ndumps*tmp_cells))

                    my_cell_indexes[cuda_device] = my_cell_indexes["cpu"].cuda()

                    gasspy_id = {"cpu": self.cell_index_to_gasspydb["cpu"][my_cell_indexes["cpu"]], cuda_device : self.cell_index_to_gasspydb[cuda_device][my_cell_indexes[self.cuda_device]]}

                    # my_Opc = torch.as_tensor(self.op.take(
                    #     gasspy_id[self.op_device], axis=1), device=cuda_device)

                    torch.cuda.empty_cache()
                    
                    if self.opc_per_NH:
                        Ext = torch.multiply(
                            self.op[:,gasspy_id[self.op_device]].cuda(), 
                            self.den[my_cell_indexes[self.den_device]].cuda())
                        torch.multiply(Ext, my_pathlenths, out=Ext)

                        torch.cumsum(Ext.flip(2),2, out=Ext)
                        torch.exp(-Ext.flip(2), out=Ext)

                    else:
                        Ext = torch.exp(
                            -torch.cumsum(
                                torch.mul(self.op[:,gasspy_id[self.op_device]].cuda(), my_pathlenths.cuda()).flip(2)
                                , 2).flip(2))

                    # First get the sum of the opc*path along each cell segment
                    # Make sure to flip the cummulative sum twice
                    # The first flip allows us to sum from the back forward, what happens to the ray in terms of amount of exinction
                    # where the first cell is the most extinct, not the least
                    
                    # But the previous flip has the correct extinction, but from end to start.
                    # We need to flip it once more to get next extinction from start to end.

                    # Kill if works
                    # my_Em = torch.as_tensor(self.em.take(
                    #     gasspy_id[self.em_device], axis=1), device=cuda_device)

                    # 
                    net_new_Flux_per_cell = torch.mul(torch.mul(self.em[:,gasspy_id[self.em_device]].cuda(), my_pathlenths.cuda()), Ext).sum(axis=[2])

                    # In the output array, take the prior levels flux and attenuate it by the 
                    output_array_gpu[:, rt_tetris_maps[my_l_i]["new"]] = torch.mul(output_array_gpu[:, rt_tetris_maps[my_l_i]["old"]], Ext[..., 0][:])

                    output_array_gpu[:, rt_tetris_maps[my_l_i]["new"]] = torch.add(output_array_gpu[:, rt_tetris_maps[my_l_i]["new"]], net_new_Flux_per_cell)

            if self.spec_save_type == "hdf5":
                out_GIDs, out_GID_i = np.unique(
                    save_GIDs[my_l_i], return_index=True)

                if self.accel == "torch":
                    save_data = {
                        'flux': output_array_gpu[:, out_GID_i].detach().cpu().numpy()}

                elif self.accel == "cuda":
                    # Note: Should be a stream
                    # We do this to ensure that the outarray doesn't get out of sync while streaming
                    save_array = cupy.array(output_array_gpu[:, out_GID_i])
                    save_data = {'flux': cupy.asnumpy(save_array)}

                if self.liteVRAM:
                    save_data.update({'x': self.new_global_rays.xp[out_GIDs],
                                      'y': self.new_global_rays.yp[out_GIDs],
                                      'ray_lrefine': self.new_global_rays.ray_lrefine[out_GIDs]})
                else:
                    save_data.update({'x': self.new_global_rays.xp[out_GIDs].get(),
                                      'y': self.new_global_rays.yp[out_GIDs].get(),
                                      'ray_lrefine': self.new_global_rays.ray_lrefine[out_GIDs].get()})
                self.write_spec_save_hdf5(save_data)

            elif self.spec_save_type == "numpy":
                np.save("%s%s%sspec_%i.npy" % (self.root_dir, self.gasspy_subdir,
                        self.gasspy_spec_subdir, root_i), output_array_gpu.cpu().numpy())
