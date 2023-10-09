"""
This routine performs a spectra RT on an AMR grid
"""
import sys, os
from pathlib import Path
import numpy as np
import cupy
import cupyx
import torch
import h5py

from gasspy.raystructures import global_ray_class
from gasspy.settings.defaults import ray_dtypes    
from gasspy.io.gasspy_io import read_yaml, check_parameter_in_config, save_gasspy_config_hdf5
import yaml

class Trace_processor():
    def __init__(
        self,
        gasspy_config,
        sim_reader,
        traced_rays, 
        gasspy_subdir="GASSPY",
        gasspy_modeldir = "GASSPY_DATABASE",
        database_name = None,
        energy_limits=None,
        accel="torch",
        radiative_transfer_precision=None,
        liteVRAM=True,
        spec_save_name="gasspy_spec.hdf5",
        cuda_device=None,
        scale_opacity_by_density=None,
        doppler_shift=None,
        maxMemoryGPU_GB=None,
        maxMemoryCPU_GB=None,
        target_segments_per_ray = None,
        doppler_shift_est_bin_ratio = None

    ):
        self.sim_reader = sim_reader
        self.gasspy_config = gasspy_config
        if isinstance(self.gasspy_config,str):
            self.gasspy_config = read_yaml(self.gasspy_config)
        
        self.sim_unit_length = self.gasspy_config["sim_unit_length"]

        if cuda_device is None:
            self.cuda_device = torch.device('cuda:0')
        else:
            self.cuda_device = cuda_device
        
        self.radiative_transfer_precision = check_parameter_in_config(self.gasspy_config, "radiative_transfer_precision", radiative_transfer_precision, "double")
        if self.radiative_transfer_precision == "double":
            self.dtype = np.float64
        elif self.radiative_transfer_precision == "single":
            self.dtype = np.float32
        else:
            sys.exit("Error: radiative_transfer_precision was given as %s but must be either \"double\" or \"single\""%self.radiative_transfer_precision)
        
        self.torch_dtype = torch.as_tensor(np.array([],dtype=self.dtype)).dtype

        
        self.liteVRAM = check_parameter_in_config(self.gasspy_config, "liteVRAM", liteVRAM, True)
        if liteVRAM:
            self.numlib = np
        else:
            self.numlib = cupy

        # Name of the database which we are using
        self.database_name = check_parameter_in_config(self.gasspy_config, "database_name", database_name, "gasspy_database.hdf5") 
        
        # Path to database directory
        self.gasspy_modeldir = check_parameter_in_config(self.gasspy_config, "gasspy_modeldir", gasspy_modeldir, "gasspy_modeldir") 
        if not self.gasspy_modeldir.endswith("/"):
            self.gasspy_modeldir = self.gasspy_modeldir + "/"
        self.h5database = self.gasspy_modeldir+self.database_name
        # Path to where we put spectra and projection files
        self.gasspy_subdir = sim_reader.gasspy_subdir
        if not self.gasspy_subdir.endswith("/"):
            self.gasspy_subdir = self.gasspy_subdir + "/"  

        # Path to projection files
        self.gasspy_projection_subdir = self.gasspy_subdir + "/projections"
        if not self.gasspy_projection_subdir.endswith("/"):
            self.gasspy_projection_subdir = self.gasspy_projection_subdir + "/"

        # Path to spectra files
        self.gasspy_spectra_subdir = self.gasspy_subdir + "/spectra"
        if not self.gasspy_spectra_subdir.endswith("/"):
            self.gasspy_spectra_subdir = self.gasspy_spectra_subdir + "/"

        # either h5file object of path to an hdf5 database containing the traced projection
        self.traced_rays = check_parameter_in_config(self.gasspy_config, "traced_rays_file", traced_rays, "traced_rays.hdf5")

        # Where to save the spectra
        self.spec_save_name = check_parameter_in_config(self.gasspy_config, "spec_save_name", spec_save_name, "spec.hdf5")

        # Needs to be reconsidered
        self.scale_opacity_by_density = check_parameter_in_config(self.gasspy_config, "scale_opacity_by_density", scale_opacity_by_density, False)

        # use torch tensors?
        self.accel = accel.lower()

        # energy limits to only transfer part of the spectrum
        self.energy_limits = check_parameter_in_config(gasspy_config, "energy_limits", energy_limits, None)
        self.raydump_dict = {}

        self.maxMemoryGPU_GB = check_parameter_in_config(self.gasspy_config,"maxMemoryGPU_GB", maxMemoryGPU_GB, 8)*1024**3
        self.maxMemoryCPU_GB = check_parameter_in_config(self.gasspy_config,"maxMemoryCPU_GB", maxMemoryCPU_GB, 8)*1024**3
        self.target_segments_per_ray = check_parameter_in_config(self.gasspy_config, "target_segments_per_ray", target_segments_per_ray, None)   

        self.doppler_shift = check_parameter_in_config(self.gasspy_config, "doppler_shift", doppler_shift, False)
        self.doppler_shift_est_bin_ratio = check_parameter_in_config(self.gasspy_config, "doppler_shift_est_bin_ratio", doppler_shift_est_bin_ratio, 10)

    def process_all(self,):
        for root_i in range(0, len(self.ancenstors)):
            self.get_spec_root(root_i, self.cuda_device)
            if root_i % 1000 == 0:
                print(root_i)
        

    def open_spec_save_hdf5(self, init_size=0):
        assert isinstance(self.spec_save_name, str), "hdf5 spec save name is not a string...exiting" 

        # Ensure the spectra existst
        if not os.path.exists(self.gasspy_spectra_subdir):
            os.makedirs(self.gasspy_spectra_subdir)

        self.spectra_outpath = self.gasspy_spectra_subdir + self.spec_save_name
        spechdf5_out = h5py.File(self.spectra_outpath, "w")

        save_gasspy_config_hdf5(self.gasspy_config, spechdf5_out)

        self.N_spec_written = 0
        if init_size >=0:
            init_size=int(init_size)
        else:
            init_size = self.numlib.int(self.global_rays.cevid[self.global_rays.cevid == -1].shape[0])

        spechdf5_out.create_dataset("flux", (init_size, len(self.energy)), maxshape=(None,len(self.energy)), dtype = self.dtype)
        spechdf5_out.create_dataset("xp", (init_size,), maxshape=(None,), dtype = ray_dtypes["xp"])
        spechdf5_out.create_dataset("yp", (init_size,), maxshape=(None,), dtype = ray_dtypes["xp"])
        spechdf5_out.create_dataset("ray_lrefine", (init_size,), dtype="int8", maxshape=(None,))
        if self.accel == "torch":
            spechdf5_out.create_dataset("energy", data=self.energy.cpu().numpy())
            spechdf5_out.create_dataset("delta_energy", data=self.delta_energy.cpu().numpy())
        else:
            spechdf5_out.create_dataset("energy", data=self.energy.get())
            spechdf5_out.create_dataset("delta_energy", data=self.delta_energy.get())            

        spechdf5_out.close()
    def write_spec_save_hdf5(self, new_data, grow=True):
        n_E, n_spec = new_data['flux'].shape
        spechdf5_out = h5py.File(self.spectra_outpath, "r+")

        for key in new_data.keys():
            new_data_shape = new_data[key].shape

            if not grow:
                if len(new_data_shape) == 1:
                    spechdf5_out[key][self.N_spec_written:self.N_spec_written+n_spec] = new_data[key][:]
        
                elif len(new_data_shape) == 2:
                    spechdf5_out[key][self.N_spec_written:self.N_spec_written+n_spec,:] = new_data[key].T[:]

            else:
                if len(new_data_shape) == 1:
                    spechdf5_out[key].resize((self.spechdf5_out[key].shape[0] + n_spec), axis=0)
                    spechdf5_out[key][-n_spec:] = new_data[key][:]
        
                elif len(new_data_shape) == 2:
                    spechdf5_out[key].resize((self.spechdf5_out[key].shape[0] + n_spec), axis=0)
                    spechdf5_out[key][-n_spec:,:] = new_data[key].T[:]

        self.N_spec_written += n_spec
        spechdf5_out.close()

    def load_all(self):
        print(" - Loading cell gasspy_indexes")
        self.load_cell_index_to_gasspydb()
        # Ensure the energy bins are loaded BEFORE the em and op tables to minimize memory used
        self.load_energy_limits()
        print(" - loading hdf5 database of models")
        self.load_database()
        print(" - loading velocity")
        self.load_velocity_data()
        print(" - loading density")
        self.load_density_data()
        print(" - loading traced rays")
        self.load_traced_rays()

        self.set_precision(self.dtype)
        self.cleaning()
        self.padding()
        self.move_to_GPU()
        self.open_spec_save_hdf5()

    def set_precision(self, new_dtype):
        self.dtype = new_dtype
        self.torch_dtype = torch.as_tensor(np.array([],dtype=self.dtype)).dtype

        self.em                          = self.em.astype(new_dtype)
        self.op                          = self.op.astype(new_dtype)
        self.raydump_dict["pathlength"]  = self.raydump_dict["pathlength"].astype(new_dtype)
        self.raydump_dict["ray_area"]    = self.raydump_dict["ray_area"].astype(new_dtype)
        self.raydump_dict["solid_angle"] = self.raydump_dict["solid_angle"].astype(new_dtype)

        if self.scale_opacity_by_density:
            self.density = self.density.astype(new_dtype)

    def load_density_data(self):
        self.density = self.sim_reader.get_field("density")
    def load_velocity_data(self):
        if self.doppler_shift:
            self.velocity_x = (self.sim_reader.get_field("velocity_x")/3e10).astype(self.dtype)
            self.velocity_y = (self.sim_reader.get_field("velocity_y")/3e10).astype(self.dtype)
            self.velocity_z = (self.sim_reader.get_field("velocity_z")/3e10).astype(self.dtype)


    def load_energy_limits(self):
        if self.energy_limits is not None:
            if type(self.energy_limits) == str:
                self.energy_limits = np.loadtxt(self.energy_limits)           

        self.energy_limits = np.array(self.energy_limits)

    def sort_and_merge_energy_limits(self):
        if self.energy_limits is None:
            return
        # Sort based of upper index
        sorted_index = np.argsort(self.energy_limits[:,1])
        Elower = self.energy_limits[sorted_index,0]
        Eupper = self.energy_limits[sorted_index,1]

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

        self.energy_limits = np.array([Elower,Eupper]).T
                
    def load_database(self):
        if self.h5database is None:
            return
        # Make sure that we load the hdf5 file or have been given one
        if isinstance(self.h5database, str):
            if Path(self.h5database).is_file():
                tmp_path = self.h5database
            elif Path(self.gasspy_modeldir + self.h5database).is_file():
                tmp_path = self.gasspy_modeldir + self.h5database
            else:
                sys.exit("Could not find path to Database %s"%(self.h5database))
            self.h5database = h5py.File(tmp_path, "r")
        else:
            if not isinstance(self.h5database, h5py.File) or isinstance(self.h5database, h5py.Group):
                sys.exit("Supplied h5database must be a h5py.File or a path to one, or an h5py.Group ")
            
        # Load the required energies
        self.energy = self.h5database["energy"][:]
        # Determine the upper and lower bin edges
        # If we already know the bin size, this is easy
        if "delta_energy" in self.h5database.keys():
            self.delta_energy = self.h5database["delta_energy"]
            self.energy_upper = self.energy + 0.5 * self.delta_energy
            self.energy_lower = self.energy - 0.5 * self.delta_energy
        else:  
            print("Energy bin sizes not found in database. Recalculating from central values")
            # otherwise... approximate as midpoint between bins... which is not accurate, but without info this is the best we can do
            self.energy_upper = np.zeros_like(self.energy)
            self.energy_lower = np.zeros_like(self.energy)
            self.energy_lower[1:]  = (self.energy[1:] + self.energy[:-1])*0.5
            self.energy_upper[:-1] =  self.energy_lower[1:]
            self.energy_lower[0]   =  2*self.energy[0 ] - self.energy_upper[0]
            self.energy_upper[-1]  =  2*self.energy[-1] - self.energy_lower[-1]
            self.delta_energy = self.energy_upper - self.energy_lower

        Eidx_ranges = [[0,-1]]
        if self.energy_limits is not None:
            self.sort_and_merge_energy_limits()
            #Eidxs is all of the indexes
            #Eidx_ranges are the values used for slices in each range
            Eidxs = np.array([], dtype = int)
            Eidx_ranges = []
            for elims in self.energy_limits:
                # Find the indexes in the array
                Eidx = np.where((self.energy >= elims[0])*(self.energy<=elims[1]))[0]
                Eidxs = np.append(Eidxs,Eidx)
                Eidx_ranges.append([Eidx[0],Eidx[-1]+1])
            # Select range from energy
            self.energy = self.energy[Eidxs]
            self.delta_energy = self.delta_energy[Eidxs]
            self.energy_upper = self.energy_upper[Eidxs]
            self.energy_lower = self.energy_lower[Eidxs]
        # How many energy bins are we dealing with here?
        self.Nener = len(self.energy)

        # Figure out which gasspyIDs we need and remap to the new "snap specific" database 
        unique_gasspy_ids, self.cell_index_to_gasspydb = np.unique(self.cell_index_to_gasspydb, return_inverse=True)
        # NOTE: np.unique returns sorted indexes. IF THIS EVER CHANGES WE NEED TO SORT MANUALLY

        self.Nmodels = len(unique_gasspy_ids)
        # Initialize arrays
        self.em = np.zeros((self.Nener, self.Nmodels))
        self.op = np.zeros((self.Nener, self.Nmodels))
        
        #Loop over index ranges and load
        Eidx_start = 0 
        Eidx_end = 0
        for Eidx_range in Eidx_ranges:
            Eidx_start = Eidx_end
            Eidx_end = Eidx_start + Eidx_range[1]-Eidx_range[0]
            self.em[Eidx_start:Eidx_end,:] = self.h5database["intensity"][unique_gasspy_ids, Eidx_range[0]:Eidx_range[1]].astype(self.dtype).T
            self.op[Eidx_start:Eidx_end,:] = self.h5database["opacity"][unique_gasspy_ids, Eidx_range[0]:Eidx_range[1]].astype(self.dtype).T
        return    


    def load_cell_index_to_gasspydb(self):
        self.cell_index_to_gasspydb = self.sim_reader.load_new_field("cell_gasspy_ids")
        self.NSimCells = len(self.cell_index_to_gasspydb)

    def load_traced_rays(self):
        if Path(self.traced_rays).is_file():
            tmp_path = self.traced_rays
        elif Path(self.gasspy_projection_subdir+self.traced_rays).is_file():
            tmp_path = self.gasspy_projection_subdir+self.traced_rays
        else:
            sys.exit("Could not find the traced rays file\n"+\
            "Provided path: %s"%self.traced_rays+\
            "Tried looking in \"./\" and %s\n"%(self.gasspy_projection_subdir)+\
            "Aborting...")            

        traced_rays_h5file = h5py.File(tmp_path, "r")

        # Load global rays
        self.global_rays = global_ray_class(on_cpu=self.liteVRAM)
        self.global_rays.load_hdf5(traced_rays_h5file)
        # Select the ancestral global_rayids
        self.ancenstors = self.global_rays.global_rayid[self.numlib.where(self.global_rays.pevid == -1)]
        self.cevid = self.global_rays.get_field("cevid")

        # Load up the segments
        self.raydump_dict['segment_global_rayid'] = traced_rays_h5file['ray_segments']['global_rayid'][:]
        self.raydump_dict["pathlength"] = traced_rays_h5file["ray_segments"]["pathlength"][:,:].astype(self.dtype)*self.dtype(self.sim_unit_length)
        self.raydump_dict["ray_area"] = traced_rays_h5file["ray_segments"]["ray_area"][:,:]
        self.raydump_dict["solid_angle"] = traced_rays_h5file["ray_segments"]["solid_angle"][:,:]
        self.raydump_dict["cell_index"] = traced_rays_h5file["ray_segments"]["cell_index"][:,:]

        self.raydump_dict["splitEvents"] = traced_rays_h5file["splitEvents"][:,:]
 
        maxGID = self.numlib.int(self.global_rays.global_rayid.max())

        # Initialize the raydump N_segs and index into ray_buffer_dumps with -1
        self.raydump_dict["ray_index0"] = self.numlib.full(maxGID+1,-1).astype(np.int64)
        self.raydump_dict["Nsegs"] = self.numlib.full(maxGID+1,-1).astype(np.int64)
        if self.target_segments_per_ray is None:
            self.target_segments_per_ray = max(int(np.mean(self.raydump_dict["Nsegs"])),2)
        
        # Get the unique values and the first index of array with that value, and Counts. 
        # This ONLY works because segment global_rayid is sorted.
        unique_gid, i0, Nsegs = np.unique(self.raydump_dict['segment_global_rayid'], return_counts=True, return_index=True)

        self.raydump_dict["ray_index0"][unique_gid] = i0.astype(np.int64)
        self.raydump_dict["Nsegs"][unique_gid] = Nsegs.astype(np.int64)

        self.raydump_dict['NcellPerRaySeg'] = self.raydump_dict["pathlength"].shape[1]
        self.max_level = 0
        self.create_split_event_dict()
        self.set_ray_to_segment_memory_ratio()


    def set_ray_to_segment_memory_ratio(self):
        # Get the size of a spectra
        spectra_size = len(self.energy)

        # Size required per leaf ray
        ray_size = 4*spectra_size*8 # Opacity + Flux for leafs and rays

        # Size required per cell
        cell_size = 3*spectra_size*8 # 2*Opacity (op and exp(-op)) + emissivity
        if self.doppler_shift :
            # With current doppler shift method, extra memory is required to determing indexing
            cell_size += 4*self.doppler_shift_est_bin_ratio*spectra_size*8 # indexing nonsense, and with the maximum number of cells being shifted 
        ray_ratio = ray_size/(ray_size + self.target_segments_per_ray*self.raydump_dict["NcellPerRaySeg"]*cell_size)

        # Get a maxium number of leaf rays to work on concurrently
        self.max_number_of_leafs = int(ray_ratio*self.maxMemoryGPU_GB/ray_size)
        # Get a maximum number of segment dumps to work on concurrently
        self.max_number_of_segments = int((1-ray_ratio)*self.maxMemoryGPU_GB/(self.raydump_dict["NcellPerRaySeg"]*cell_size))
        self.allocate_leaf_ray_storage()

    def allocate_leaf_ray_storage(self):
        # Get the size of a spectra
        spectra_size = len(self.energy)

        # Allocate arrays to store fluxes and opacities in
        #if self.accel == "torch":
        #    self.rt_flux        = torch.zeros((self.max_number_of_leafs,spectra_size), dtype=self.torch_dtype, device = self.cuda_device)
        #else:
        self.rt_flux        = cupy.zeros((self.max_number_of_leafs,spectra_size), dtype=self.dtype)            

        return

    def reset_leaf_ray_storage(self):
        self.rt_flux[:,:] = 0
        return

    def create_split_event_dict(self):
        self.raydump_dict['splitEvents'] = cupy.asnumpy(self.raydump_dict['splitEvents'])
        # Create a dictionary with each split event keyed by the GID of teh parent
        self.split_by_gid_tree = dict(zip(self.raydump_dict['splitEvents'][:, 0].astype(np.int32), zip(*self.raydump_dict['splitEvents'][:, 1:].astype(np.int32).T)))

    def padding(self):
        # The last element of each of the following arrays is assumed to be zero, and is used for out of bound indexes, which will have values -1.
        self.raydump_dict["pathlength"] = np.vstack([self.raydump_dict["pathlength"], np.zeros(self.raydump_dict["pathlength"].shape[1], dtype = self.dtype)])
        self.raydump_dict["ray_area"] = np.vstack([self.raydump_dict["ray_area"], np.zeros(self.raydump_dict["ray_area"].shape[1], dtype = self.dtype)])
        self.raydump_dict["solid_angle"] = np.vstack([self.raydump_dict["solid_angle"], np.zeros(self.raydump_dict["solid_angle"].shape[1], dtype = self.dtype)])

        # Padd the last value with zero, so that indexing to -1 is safe when doing RT
        self.em = np.vstack([self.em.T, np.zeros(self.em.shape[0], dtype = self.dtype)]).T

        # Padd the last value with zero, so that indexing to -1 is safe when doing RT
        self.op = np.vstack([self.op.T, np.zeros(self.op.shape[0], dtype = self.dtype)]).T

        self.cell_index_to_gasspydb = np.hstack((self.cell_index_to_gasspydb,[-1]))

        self.raydump_dict["cell_index"] = np.vstack((self.raydump_dict["cell_index"],np.full(self.raydump_dict['NcellPerRaySeg'], -1)))

        self.raydump_dict["ray_index0"] = np.append(self.raydump_dict["ray_index0"],[-1])

        if self.scale_opacity_by_density:
            self.density = np.append(self.density, np.array([0], dtype = self.dtype))

    def move_to_GPU(self):
        if not self.liteVRAM:
            self.raydump_dict["pathlength"] = cupy.asarray(self.raydump_dict["pathlength"], dtype=self.dtype)
            self.raydump_dict["cell_index"] = cupy.asarray(self.raydump_dict["cell_index"], dtype=cupy.int64)

            self.em = cupy.asarray(self.em, dtype=self.dtype)
            self.op = cupy.asarray(self.op, dtype=self.dtype)
            self.den = cupy.asarray(self.den, dtype=self.dtype)
            if self.doppler_shift:
                self.velocity_x = cupy.asarray(self.velocity_x, dtype= self.dtype)
                self.velocity_y = cupy.asarray(self.velocity_y, dtype= self.dtype)
                self.velocity_z = cupy.asarray(self.velocity_z, dtype= self.dtype)

        # Always keep indexes on GPU
        self.cell_index_to_gasspydb = cupy.asarray(self.cell_index_to_gasspydb)


        if self.accel == "torch":
            self.energy = torch.as_tensor(self.energy, device = self.cuda_device, dtype = self.torch_dtype)
            self.delta_energy = torch.as_tensor(self.delta_energy, device=self.cuda_device, dtype = self.torch_dtype) 
            self.energy_upper = torch.as_tensor(self.energy_upper, device=self.cuda_device, dtype = self.torch_dtype)
            self.energy_lower = torch.as_tensor(self.energy_lower, device=self.cuda_device, dtype = self.torch_dtype)
        else:
            self.energy = cupy.asarray(self.energy, dtype = self.dtype)
            self.delta_energy = cupy.asarray(self.delta_energy, dtype = self.dtype)
            self.energy_upper = cupy.asarray(self.energy_upper, dtype = self.dtype)
            self.energy_lower = cupy.asarray(self.energy_lower, dtype = self.dtype)
    def cleaning(self):
        """These cleaning operations ensure that the input data conforms to expectations of the radiative transfer algorithms""" 
        # Cleaning operations
        self.raydump_dict["pathlength"][self.raydump_dict["pathlength"]<0] = 0.

        # A ray may die outside the box, and as a result have an index larger than the simulation. We set that to -1.
        self.raydump_dict["cell_index"][self.raydump_dict["cell_index"] > len(self.cell_index_to_gasspydb) - 1] = -1

        #del(self.raydump_dict['splitEvents'])
        #del(self.saved3d)      

  
    def get_branch(self, root_i):
        """
            For a given branch (eg, a root ray and all of its children, their children etc.)
            get all global_rayids and corresponding leaf_rays to add to

            TODO: This function is messy and probably inefficient. Should be redesigned
        """
        self.max_level = 0
        level = 0
        # This initalizes the trace down from parent
        gid = self.ancenstors[root_i]
        new_parent_gids = np.array([gid], dtype=np.int64)

        branch_global_rayid = {}
        branch_parents = {}
        branch_global_rayid[0] = np.array([self.numlib.int(gid)])
        branch_parents[0] = np.array([-1])
        
        nrays = 1
        eol = -1

        # First get the global_rayid such that for each level of this branch we have a 4^(lrefine-lrefine_min) array
        # Non existent rays get listed as -1
    
        while eol < 4**(level-1):
            level += 1
            eol = 0
            # Initialize the current level
            branch_global_rayid[level] = np.full(4**(level), -1, dtype=np.int64)
            branch_parents[level] = np.full(4**(level), -1, dtype = np.int64)
            # Find all the children associated with these rays
            exists = False
            for parent_i, parent_gid in enumerate(new_parent_gids):
                parent_gid = int(parent_gid)
                if parent_gid != -1:
                    if parent_gid in self.split_by_gid_tree:
                        branch_global_rayid[level][4*parent_i:4*parent_i+4] = \
                            np.asarray(self.split_by_gid_tree[parent_gid], dtype=np.int64)[:]
                        branch_parents[level][4*parent_i:4*parent_i+4]=parent_i
                        exists = True
                        nrays += 4
                    else:
                        eol += 1
                else:
                    eol += 1
            if not exists:
                del(branch_global_rayid[level])
                del(branch_parents[level])
                level -=1
                break

            # Set these as new parents
            new_parent_gids = branch_global_rayid[level]
        # for record keeping in the class of the maximum level reached relative to the parent
        max_level = level


        # Global_rayids of the rays
        global_rayids = np.zeros(nrays, dtype=int)
        # The raveled index of the rays
        branch_iray = {}


        # Next set the index of these rays for this branch (eg. if we would put these rays in one list). Lower the level, lower the index
        iray = 0
        for level in range(max_level+1):
            branch_iray[level] = np.full(4**(level), -1, dtype = np.int64)
            for level_iray, global_rayid in enumerate(branch_global_rayid[level]):
                if global_rayid != -1:
                    global_rayids[iray] = global_rayid
                    branch_iray[level][level_iray] = iray

                    iray += 1

        # Gather into one dictionary
        branch = {
            "branch_global_rayid" : branch_global_rayid,
            "branch_iray" : branch_iray,
            "branch_parent" : branch_parents
        }

        #### Map all rays to corresponding leafs ###
        rays_leafs_istart = np.full((nrays), -1)
        rays_nleafs = np.zeros(nrays)
        leaf_rays_global_rayids = np.zeros(nrays)

        # Start from the root ray and determine which rays belong to which leafs
        branch_nleafs = self.determine_leafs(0,0,0, branch, rays_nleafs, rays_leafs_istart, leaf_rays_global_rayids)
        leaf_rays_global_rayids = leaf_rays_global_rayids[:branch_nleafs]
        if branch_nleafs > self.max_number_of_leafs:
            print("Warning: branch with root %d contains more leafs than the maximum number of leafs set"%root_i)
            print("         attempting to extend the maximum number of leafs to accommodate, ")
            self.max_number_of_leafs = branch_nleafs
            self.allocate_leaf_ray_storage()

        branch_dictionary = {
            "global_rayid"     : global_rayids,
            "rays_leafs_istart" : rays_leafs_istart,
            "rays_nleafs"      : rays_nleafs,
            "nrays"            : nrays,
            "nleafs"           : branch_nleafs,
            "leaf_rays_global_rayids" : leaf_rays_global_rayids
        }

        return branch_dictionary

    def determine_leafs(self, level, level_iray, nleafs, branch, rays_nleafs, rays_leafs_istart, leaf_rays_global_rayids):
        # global_rayid of ray
        global_rayid = branch["branch_global_rayid"][level][level_iray]
        if global_rayid == -1:
            sys.exit("Error [rt_trace.py:determine_leafs]: child ray has no global_rayid")
        
        # Index of ray in branch
        iray = branch["branch_iray"][level][level_iray]

        # If its a leaf, we are done and just return
        if self.cevid[global_rayid] == -1:
            rays_leafs_istart[iray] = nleafs
            rays_nleafs[iray] = 1
            leaf_rays_global_rayids[nleafs] = global_rayid
            return nleafs + 1
        # Otherwise recursive call for each child 
        for ichild in range(4):
            # Set ileaf of the child
            nleafs = self.determine_leafs(level+1, level_iray*4 + ichild, nleafs, branch, rays_nleafs, rays_leafs_istart, leaf_rays_global_rayids)
            # Use the childs values to update the current rays leaf
            iray_child = branch["branch_iray"][level + 1][level_iray*4 + ichild]
            if ichild == 0:
                rays_leafs_istart[iray] = rays_leafs_istart[iray_child]
            rays_nleafs[iray] += rays_nleafs[iray_child]
        return nleafs

    def merge_branches(self, branches):
        """
            Merges branches into a set of 1D arrays containing all rays within the branches.
            Rays from a given branch are sequential as before, but now the branches share a set of corresponding leaf rays
        """
        nleafs_total = 0
        nrays_total  = 0
        # Figure out total number of rays
        for ibranch in branches:
            nleafs_total += branches[ibranch]["nleafs"]
            nrays_total  += branches[ibranch]["nrays"]
        
        leafs_global_rayids = np.zeros((nleafs_total), dtype = np.int64)
        global_rayids = np.zeros((nrays_total), dtype=np.int64)
        rays_leafs_istart  = np.zeros((nrays_total), dtype=np.int64)
        rays_nleafs  = np.zeros((nrays_total), dtype=np.int64)
        istart = 0
        ileaf = 0
        for ibranch in branches:
            branch = branches[ibranch]
            nrays  = branch["nrays"]
            nleafs = branch["nleafs"]

            global_rayids[istart:istart+nrays] = branch["global_rayid"]
            leafs_global_rayids[ileaf:ileaf+nleafs] = branch["leaf_rays_global_rayids"]
            rays_leafs_istart[istart:istart+nrays] = ileaf + branch["rays_leafs_istart"]
            rays_nleafs[istart:istart+nrays] = branch["rays_nleafs"]
            istart += nrays
            ileaf  += nleafs
        ray_dictionary = {
            "global_rayids" : global_rayids,
            "rays_leafs_istart"  : rays_leafs_istart,
            "rays_nleafs"  :   rays_nleafs,
            "leafs_global_rayids" : leafs_global_rayids,
            "nleafs_total"  : nleafs_total,
            "nrays_total"   : nrays_total
        }

        return ray_dictionary

    def process_trace(self):
        total_branches = len(self.ancenstors)
        current_branches = {}
        ibranch = 0
        nleafs = 0
        # Loop over all branches
        while ibranch < total_branches:
            # Gather all rays of this branch
            branch = self.get_branch(ibranch)
            # if this branch would exceed the maximum number of leafs, process those already corrected
            if nleafs + branch["nleafs"] > self.max_number_of_leafs:
                print(ibranch, nleafs)
                # Combine current branches into linear arrays for all rays
                ray_dictionary = self.merge_branches(current_branches)
                self.process_rays(ray_dictionary, self.cuda_device)
                nleafs = 0
                current_branches = {}
            current_branches[ibranch] = branch
            nleafs += branch["nleafs"]
            ibranch += 1
        # Do the last branches if any
        if nleafs > 0:
            # Combine current branches into linear arrays
            ray_dictionary = self.merge_branches(current_branches)
            self.process_rays(ray_dictionary, self.cuda_device)

            
        #self.close_spec_save_hdf5()

    def info(self):
        for key in self.raydump_dict.keys():
            try:
                if type(self.raydump_dict[key]) == dict:
                    print(key, self.raydump_dict[key].shape)    
                print(key, self.raydump_dict[key].shape)
            except:
                print(key, self.raydump_dict[key])

    def get_segments(self,ray_dictionary, iray_start, iray_end, nsegs, cuda_device):
        # Allocate space for the segments
        if self.liteVRAM:
            segment_cell_index   = np.zeros((nsegs, self.raydump_dict["NcellPerRaySeg"]), np.int64)
        else:
            segment_cell_index   = cupy.zeros((nsegs, self.raydump_dict["NcellPerRaySeg"]), np.int64)

        if self.accel == "torch":
            ray_segment_istart   = torch.zeros((iray_end - iray_start), device=cuda_device, dtype=torch.int64)
            ray_segment_nsegs    = torch.zeros((iray_end - iray_start), device=cuda_device, dtype=torch.int64)
            segment_iray         = torch.zeros((nsegs), device=cuda_device, dtype=torch.int64)
            segment_pathlength   = torch.zeros((nsegs, self.raydump_dict["NcellPerRaySeg"]), device=cuda_device, dtype=self.torch_dtype)
            segment_ray_area     = torch.zeros((nsegs, self.raydump_dict["NcellPerRaySeg"]), device=cuda_device, dtype=self.torch_dtype)
            segment_solid_angle  = torch.zeros((nsegs, self.raydump_dict["NcellPerRaySeg"]), device=cuda_device, dtype=self.torch_dtype)

        else:
            ray_segment_istart   = cupy.zeros((iray_end - iray_start), dtype=np.int64)
            ray_segment_nsegs    = cupy.zeros((iray_end - iray_start), dtype=np.int64)
            segment_iray         = cupy.zeros((nsegs), dtype=np.int64)
            segment_pathlength   = cupy.zeros((nsegs, self.raydump_dict["NcellPerRaySeg"]), dtype=self.dtype)
            segment_ray_area     = cupy.zeros((nsegs, self.raydump_dict["NcellPerRaySeg"]), dtype=self.dtype)
            segment_solid_angle  = cupy.zeros((nsegs, self.raydump_dict["NcellPerRaySeg"]), dtype=self.dtype)


        iseg = 0

        # Ray properties
        global_rayids = ray_dictionary["global_rayids"][iray_start:iray_end]
        #if self.accel == "torch":
        #    rays_leafs_istart = torch.as_tensor(ray_dictionary["rays_leafs_istart"][iray_start:iray_end], device=cuda_device)
        #    rays_nleafs       = torch.as_tensor(ray_dictionary["rays_nleafs"][iray_start:iray_end], device=cuda_device)
        #else:
        # Currently there's a bug with torch, so we use cupy
        rays_leafs_istart = cupy.asarray(ray_dictionary["rays_leafs_istart"][iray_start:iray_end])
        rays_nleafs       = cupy.asarray(ray_dictionary["rays_nleafs"][iray_start:iray_end])
        for iray, global_rayid in enumerate(global_rayids):
            # Get the segments corresponding to this ray
            trace_segment_istart, nsegs = self.raydump_dict["ray_index0"][global_rayid], self.raydump_dict["Nsegs"][global_rayid]
            trace_segment_iend = trace_segment_istart + nsegs

            # Which segments in the local arrays that corresponds to each ray
            ray_segment_istart[iray] = iseg
            ray_segment_nsegs[iray]  = nsegs
            
            # Set which ray the segment belongs to
            segment_iray[iseg:iseg+nsegs] = iray

            # Set ray-cell intersection properties
            if self.accel == "torch":
                segment_pathlength[iseg:iseg+nsegs]  = torch.as_tensor(self.raydump_dict["pathlength"][trace_segment_istart:trace_segment_iend,:], device = cuda_device)
                segment_ray_area[iseg:iseg+nsegs]    = torch.as_tensor(self.raydump_dict["ray_area"][trace_segment_istart:trace_segment_iend,:], device = cuda_device)
                segment_solid_angle[iseg:iseg+nsegs] = torch.as_tensor(self.raydump_dict["solid_angle"][trace_segment_istart:trace_segment_iend,:], device = cuda_device)
            else:
                segment_pathlength[iseg:iseg+nsegs]  = cupy.ndarray(self.raydump_dict["pathlength"][trace_segment_istart:trace_segment_iend,:])
                segment_ray_area[iseg:iseg+nsegs]    = cupy.ndarray(self.raydump_dict["ray_area"][trace_segment_istart:trace_segment_iend,:])
                segment_solid_angle[iseg:iseg+nsegs] = cupy.ndarray(self.raydump_dict["solid_angle"][trace_segment_istart:trace_segment_iend,:])


            if self.liteVRAM:
                segment_cell_index[iseg:iseg+nsegs] = self.raydump_dict["cell_index"][trace_segment_istart:trace_segment_iend,:]
            else:
                segment_cell_index[iseg:iseg+nsegs] = cupy.ndarray(self.raydump_dict["cell_index"][trace_segment_istart:trace_segment_iend,:])

            iseg += nsegs

        if self.accel == "torch":
            segment_global_rayids = global_rayids[segment_iray.cpu().numpy()]
        else:
            segment_global_rayids = global_rayids[segment_iray.get()]

        if not self.liteVRAM:
            segment_global_rayids = cupy.asarray(segment_global_rayids)

        segments = {
                "cell_indexes" : segment_cell_index,
                "pathlengths"  : segment_pathlength,
                "ray_areas"    : segment_ray_area,
                "solid_angle"  : segment_solid_angle,
                "segment_iray" : segment_iray,
                "segment_global_rayids" : segment_global_rayids,
                "rays_segment_istart": ray_segment_istart,
                "rays_segment_nsegs": ray_segment_nsegs,
                "rays_leafs_istart" : rays_leafs_istart,
                "rays_global_rayids" : global_rayids,
                "rays_nleafs" : rays_nleafs
        }
        return segments

    def process_rays(self, ray_dictionary, cuda_device):
        nrays_total = ray_dictionary["nrays_total"]

        # loop through the rays from highest to lowest, ie from the furthest most points 
        iray = nrays_total - 1
        iray_end = iray + 1
        nsegs = 0

        while iray >= 0:
            # Figure out how many segments corresponds to this ray
            global_rayid = ray_dictionary["global_rayids"][iray]
            nsegs_for_ray = self.raydump_dict["Nsegs"][global_rayid]
            if (nsegs_for_ray + nsegs > self.max_number_of_segments) and nsegs > 0:
                print("\t", iray+1, iray_end)
                segments = self.get_segments(ray_dictionary, iray+1,iray_end, nsegs, cuda_device)
                self.process_segments(segments, cuda_device)
                iray_end = iray + 1
                nsegs = 0
            # If one single ray exceeds the number of allowed segments we need to deal with it in a special way TODO
            if nsegs_for_ray > self.max_number_of_segments:
                self.process_single_ray(iray)
                iray_end -= 1
                iray -= 1
                continue
            nsegs += nsegs_for_ray
            # if last ray process remaining
            if iray == 0 and  iray_end !=0:
                print("\t", 0,iray_end)
                segments = self.get_segments(ray_dictionary, 0 ,iray_end, nsegs, cuda_device)
                self.process_segments(segments, cuda_device)
                iray_end = iray + 1
                nsegs = 0
                break

            iray -= 1

    
        # Once all rays have been processed, send to output
        self.output_leafs(ray_dictionary)

    def arange_indexes(self, start, end, cuda_device = None, accel = None):
        # Method to generate list of indices based on a start and end
        # (magic from stackoverflow...)
        if accel is None:
            accel =self.accel
        lens = end - start
        if accel == "torch":
            if cuda_device is None:
                cuda_device = self.cuda_device
            torch.cumsum(lens, out=lens, dim=0)
            indexes = torch.ones(int(lens[-1]), dtype=int, device=cuda_device)
        else:
            cupy.cumsum(lens, out=lens)
            indexes = cupy.ones(int(lens[-1]), dtype=int)
        indexes[0] = start[0]
        indexes[lens[:-1]] += start[1:]
        indexes[lens[:-1]] -= end[:-1]
        if accel == "torch":
            torch.cumsum(indexes, out=indexes, dim = 0)
        else:
            cupy.cumsum(indexes, out=indexes, axis = 0)

        return indexes

    def velocity_shift(self, segment_global_rayids, cell_indexes, emissivity, opacity, cuda_device):
        raydir_x = self.global_rays.get_field("raydir_x", index = segment_global_rayids).astype(self.dtype)
        raydir_y = self.global_rays.get_field("raydir_y", index = segment_global_rayids).astype(self.dtype)
        raydir_z = self.global_rays.get_field("raydir_z", index = segment_global_rayids).astype(self.dtype)
        
        cell_velocity_x = self.velocity_x[cell_indexes]
        cell_velocity_y = self.velocity_y[cell_indexes]
        cell_velocity_z = self.velocity_z[cell_indexes]

        los_velocity = raydir_x[:, None]*cell_velocity_x + raydir_y[:, None]*cell_velocity_y + raydir_z[:, None]*cell_velocity_z
        if self.accel == "torch":
            return self.__velocity_shift_torch__(torch.as_tensor(los_velocity, device=cuda_device), emissivity, opacity, cuda_device)
        else :
            print("ERROR: VELOCITY SHIFT NOT IMPLEMENTED FOR CUPY")
            sys.exit(0)
        
    def __velocity_shift_torch__(self,los_velocity, emissivity, opacity, cuda_device):
        # Boost factor
        boost_factor = 1/(1-los_velocity)
        # Figure out the new energy bin edges 
        new_energy_lower = torch.mul(self.energy_lower[:, None, None], boost_factor)
        new_energy_upper = torch.mul(self.energy_upper[:, None, None], boost_factor)
        # Determine which bins these correspond to
        idx_lower = torch.searchsorted(self.energy_upper, new_energy_lower.ravel()).reshape(emissivity.shape)
        torch.minimum(idx_lower, torch.as_tensor(len(self.energy)-1, device = cuda_device), out = idx_lower)

        idx_upper = torch.searchsorted(self.energy_upper, new_energy_upper.ravel()).reshape(emissivity.shape)
        torch.minimum(idx_upper, torch.as_tensor(len(self.energy)-1, device = cuda_device), out = idx_upper)

        
        # Initialize shifted arrays
        emissivity_new = torch.zeros_like(emissivity)
        opacity_new    = torch.zeros_like(opacity)

        # Divide the shifted bins into how many new bins they cover
        dindex = (idx_upper - idx_lower)
        for didx in range(torch.min(dindex), torch.max(dindex)+1):
            with_didx = torch.where(dindex == didx)
            
            # If old bins fits entirely into one bin, just add it
            if didx == 0:
                idx_lower_now = idx_lower[with_didx]
                emissivity_new.index_put_((idx_lower_now, with_didx[1], with_didx[2]), torch.mul(emissivity[with_didx], self.delta_energy[with_didx[0]]), accumulate = True)
                opacity_new.index_put_(   (idx_lower_now, with_didx[1], with_didx[2]), torch.mul(opacity[with_didx], self.delta_energy[with_didx[0]])   , accumulate = True)
                continue

            # old bins now span multiple new bins
            boost_now = boost_factor[with_didx[1], with_didx[2]]
            # Bins at the edges
            idx_lower_now = idx_lower[with_didx]
            idx_upper_now = idx_upper[with_didx]

            # Left edge 
            dE = torch.maximum(self.energy_upper[idx_lower_now] - new_energy_lower[with_didx],torch.as_tensor(0, device = cuda_device, dtype = self.torch_dtype))
            # dE is now the range of energy in the shifted frame that lands in the new bin. Convert that to the old energy by dividing by the boost
            torch.divide(dE, boost_now, out = dE)
            emissivity_new.index_put_((idx_lower_now.ravel(), with_didx[1], with_didx[2]), emissivity[with_didx]*dE, accumulate = True)
            opacity_new.index_put_(   (idx_lower_now.ravel(), with_didx[1], with_didx[2]), opacity[with_didx]*dE,    accumulate = True) 

            # right edge
            dE = torch.maximum(new_energy_upper[with_didx] - self.energy_lower[idx_upper_now],torch.as_tensor(0, device = cuda_device, dtype = self.torch_dtype))
            torch.divide(dE, boost_now, out = dE)
            emissivity_new.index_put_((idx_upper_now.ravel(), with_didx[1], with_didx[2]), emissivity[with_didx]*dE, accumulate = True)
            opacity_new.index_put_(   (idx_upper_now.ravel(), with_didx[1], with_didx[2]), opacity[with_didx]*dE   , accumulate = True)                        

            # If the old bin only spans two bins then both of them are edges and we are therefore done
            if didx == 1:
                continue

            # New bins completely covered by the old bin
            Eindexes_new = (idx_lower_now + 1 +torch.arange(0, didx-1, device = cuda_device)[:,None, None]).ravel()
            Eindexes_old = torch.tile(with_didx[0], dims = (didx-1,)).ravel()
            Sindexes = torch.tile(with_didx[1],dims = (didx-1,)).ravel()
            Cindexes = torch.tile(with_didx[2],dims = (didx-1,)).ravel()

            emissivity_new.index_put_((Eindexes_new, Sindexes, Cindexes), emissivity[Eindexes_old, Sindexes, Cindexes]*self.delta_energy[Eindexes_new]/boost_factor[Sindexes, Cindexes], accumulate = True)
            opacity_new.index_put_(   (Eindexes_new, Sindexes, Cindexes),    opacity[Eindexes_old, Sindexes, Cindexes]*self.delta_energy[Eindexes_new]/boost_factor[Sindexes, Cindexes], accumulate = True)

        # Divide by bin size to take from N to Nnu and average opacity
        torch.divide(emissivity_new, self.delta_energy[:,None,None], out = emissivity_new)
        torch.divide(opacity_new, self.delta_energy[:,None,None], out = opacity_new)

        return emissivity_new, opacity_new
    def __process_segments_torch__(self, segments, cuda_device):
        cell_indexes = segments["cell_indexes"]
        pathlengths  = segments["pathlengths"]
        ray_area     = segments["ray_areas"]
        solid_angle  = segments["solid_angle"]

        if self.liteVRAM:
            # If the emissivity and opacity tables are not on GPU, reduce data transfer by only taking unique gasspy_ids
            gasspy_ids = self.cell_index_to_gasspydb[cell_indexes]
            unique_ids, local_index = cupy.unique(gasspy_ids, return_inverse=True)
            local_index = local_index.reshape(gasspy_ids.shape)
            # Move to cpu for indexing
            unique_ids_cpu = unique_ids.get()
            local_em = cupy.asarray(self.em.take(unique_ids_cpu, axis=1))
            local_op = cupy.asarray(self.op.take(unique_ids_cpu, axis=1))
            
            emissivity = torch.as_tensor(local_em.take(local_index, axis = 1), device=cuda_device)
            opacity    = torch.as_tensor(local_op.take(local_index, axis = 1), device=cuda_device)
            del local_em, local_op, gasspy_ids

        else:
            gasspy_ids = self.cell_index_to_gasspydb[cell_indexes]
            # Grab emissivity and opacity from database
            emissivity = torch.as_tensor(self.em.take(gasspy_ids, axis=1), device=cuda_device)
            opacity    = torch.as_tensor(self.op.take(gasspy_ids, axis=1), device=cuda_device)

        if self.doppler_shift:
            segment_global_rayids = segments["segment_global_rayids"]
            emissivity, opacity = self.velocity_shift(segment_global_rayids, cell_indexes, emissivity, opacity, cuda_device)

        # if the opacity is given as a function of number density, scale it
        if self.scale_opacity_by_density:
            density = torch.as_tensor(self.density.take(cell_indexes), device=cuda_device)
            torch.mul(density, opacity, out = opacity)

        # Take from opacity to attenuation
        torch.multiply(opacity,pathlengths, out = opacity)

        # Take from emissivity to flux 
        torch.multiply(emissivity, pathlengths, out = emissivity)
        # Take from flux to counts
        torch.multiply(emissivity, torch.multiply(ray_area, solid_angle),    out = emissivity)

        # Apply attenuation from local cell (approximation as half the cell)
        # TODO: Should be changed to em*(1-exp(op*pathlength))/op, but where opacity is zero might cause issues
        #       so left for now
        torch.multiply(emissivity, torch.exp(-opacity/2), out=emissivity)

        # Get total attenuation along each segment
        torch.cumsum(opacity, dim=2, out = opacity)
        # Total attenuation in segment
        seg_attenuation = opacity[:,:,-1]
        # Opacity Save opacity of individual segment
        seg_opacity = torch.clone(seg_attenuation)

        # Cumulative opacity for all segments
        torch.cumsum(seg_attenuation, dim = 1, out = seg_attenuation)

        # If we have more than one ray this cumulative sum is also across all rays of these segments.
        # Correct for this by subtracting off the final opacity of the previous ray
        # Start by getting all the segments corresponding to each ray
        ray_segment_indexes = self.arange_indexes(segments["rays_segment_istart"], segments["rays_segment_istart"] + segments["rays_segment_nsegs"])
        if len(segments["rays_segment_istart"])> 1:
            seg_attenuation[:,ray_segment_indexes[segments["rays_segment_nsegs"][0]:]] -= torch.repeat_interleave(seg_attenuation[:,segments["rays_segment_istart"][1:]-1], segments["rays_segment_nsegs"][1:], dim = 1)        
            # If floating point errors occur, just ensure that nothing is negative. 
            # Since we are dealing with attenuation, small errors shouldnt matter (unless Av of previous rays was >1e12)
            seg_attenuation = torch.where(seg_attenuation > self.dtype(0.0), seg_attenuation, torch.tensor(0,dtype = self.torch_dtype, device=cuda_device))

        # Opacity seen by each segment (Cumsum - own contribution)
        torch.subtract(seg_attenuation, seg_opacity, out = seg_opacity)
        # ensure no fp nonsense
        seg_opacity = torch.where(seg_opacity > self.dtype(0.0), seg_opacity, torch.tensor(0,dtype = self.torch_dtype, device=cuda_device))

        # Add the attenuation of the previous segments 
        opacity[:,:,:] += seg_opacity[:,:, None]

        # Attenuate with attenuation of the preceeding cell
        emissivity[:,:,1:] *= torch.exp(-opacity[:,:,:-1])
        # Attenuate the first cell by the preceeding segments (Needs to be done seperatly)
        emissivity[:,:,0] *= torch.exp(-seg_opacity[:,:])

        # Total flux in each segment
        seg_flux = torch.sum(emissivity, dim = 2)

        # Mapping from rays to leafs
        ray_leafs_istart = segments["rays_leafs_istart"]
        ray_nleafs       = segments["rays_nleafs"]
        ray_leafs_iend   = ray_leafs_istart + ray_nleafs

        # Total flux and attenuation in each ray
        ray_flux        = torch.zeros((seg_flux.shape[0], len(ray_nleafs)), device = cuda_device, dtype = self.torch_dtype)
        ray_attenuation = torch.zeros((seg_flux.shape[0], len(ray_nleafs)), device = cuda_device, dtype = self.torch_dtype)

        # Get the fractional size of the rays
        ray_global_rayids = segments["rays_global_rayids"]
        ray_fractional_area = cupy.array(self.global_rays.get_field("ray_fractional_area", index = ray_global_rayids))

        # The flux is already attenuated properly for this segment, just sum up the total flux contribution
        ray_flux.index_add_(dim = 1, index = segments["segment_iray"], source = seg_flux)
        # The total attenuation of the ray is the same as its final segment
        ray_segment_istart = segments["rays_segment_istart"]
        ray_segment_nseg = segments["rays_segment_nsegs" ]
        ray_attenuation[:,:] = seg_attenuation[:,ray_segment_istart + ray_segment_nseg-1]
        ray_attenuation = torch.exp(-ray_attenuation)
        # Transpose to be of the shape (nrays, nener)

        ## Due to a bug in pytorch we need to do this with cupy
        ray_flux = cupy.asarray(ray_flux.T)
        ray_attenuation = cupy.asarray(ray_attenuation.T)

        # maximum number of leafs to be filled by a ray
        # Number of leafs is (inversely) equivalent to ray refinement level. Leaf rays will have nleafs of 1, lrefine_max - 1 will have 4, lrefine_max -2 will have 16
        # Rays with fewer nleafs (more higher refinement level) are further away and should therefore be dealt with first
        # in order for cells closer to the observer to attenuate emission from further away
        # Rays with the same nleafs never share a leaf ray so we can also ensure zero collisions
        # by looping over nleafs
        unique_nleafs= cupy.unique(ray_nleafs).get()
        for nleaf in unique_nleafs:
            rays_now = cupy.where(ray_nleafs == nleaf)[0]
            if len(rays_now) <=0:
                continue
            ileafs = self.arange_indexes(ray_leafs_istart[rays_now], ray_leafs_iend[rays_now], accel="cupy")
            # Attenuate the flux of the leaf with the attenuation of the ray
            self.rt_flux[ileafs,:] *= cupy.repeat(ray_attenuation[rays_now,:], axis = 0, repeats = nleaf)

            # Add the flux of the ray. Assume that the photons split evenly into the leafs 
            cupyx.scatter_add(self.rt_flux, ileafs, cupy.repeat(ray_flux[rays_now,:]/ray_fractional_area[rays_now,None], axis = 0, repeats = nleaf))
            #self.rt_flux[ileafs,:] += cupy.repeat(ray_flux[rays_now,:],axis = 0, repeats = nleaf)
        torch.cuda.empty_cache()
        return

    def process_segments(self, segments, cuda_device):
        if self.accel == "torch":
            self.__process_segments_torch__(segments, cuda_device)
        else:
            print("ERROR: cupy RT has not been integrated. Use the torch accelerator")
            sys.exit(0) 
        
        return

    def output_leafs(self, ray_dictionary):

        # Get leafs
        leaf_global_rayids = ray_dictionary["leafs_global_rayids"]
        nleafs_total = ray_dictionary["nleafs_total"]

        if self.liteVRAM:
            #save_data = {'flux':self.rt_flux[:nleafs_total,:].get()*self.global_rays.ray_fractional_area[leaf_global_rayids][:,None]}
            save_data = {'flux':self.rt_flux[:nleafs_total,:].get()}
            save_data.update({'xp':self.global_rays.xp[leaf_global_rayids],
                              'yp':self.global_rays.yp[leaf_global_rayids],
                              'ray_lrefine':self.global_rays.ray_lrefine[leaf_global_rayids]})
        else:
            #save_data = {'flux':self.rt_flux[:nleafs_total,:].get()*self.global_rays.ray_fractional_area[leaf_global_rayids].get()[:,None]}
            save_data = {'flux':self.rt_flux[:nleafs_total,:].get()[:,None]}
            save_data.update({'xp':self.global_rays.xp[leaf_global_rayids].get(),
                              'yp':self.global_rays.yp[leaf_global_rayids].get(),
                              'ray_lrefine':self.global_rays.ray_lrefine[leaf_global_rayids].get()})
        self.write_spec_save_hdf5(save_data)
        self.reset_leaf_ray_storage()


    def write_spec_save_hdf5(self, new_data, grow=True):
        n_spec, n_e = new_data['flux'].shape
        spechdf5_out = h5py.File(self.spectra_outpath, "r+")

        for key in new_data.keys():
            new_data_shape = new_data[key].shape
            if not grow:
                if len(new_data_shape) == 1:
                    spechdf5_out[key][self.N_spec_written:self.N_spec_written+n_spec] = new_data[key][:]
        
                elif len(new_data_shape) == 2:
                    spechdf5_out[key][self.N_spec_written:self.N_spec_written+n_spec,:] = new_data[key][:]

            else:
                if len(new_data_shape) == 1:
                    spechdf5_out[key].resize((spechdf5_out[key].shape[0] + n_spec), axis=0)
                    spechdf5_out[key][-n_spec:] = new_data[key][:]
        
                elif len(new_data_shape) == 2:
                    spechdf5_out[key].resize((spechdf5_out[key].shape[0] + n_spec), axis=0)
                    spechdf5_out[key][-n_spec:,:] = new_data[key][:]

        self.N_spec_written += n_spec
