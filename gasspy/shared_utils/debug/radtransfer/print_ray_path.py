from matplotlib import pyplot as plt
import numpy as np
import torch
import cupy
import sys
import h5py as hp
import importlib.util
import os 

from gasspy.radtransfer import FamilyTree
from gasspy.shared_utils.argument_parsers.FamilyTree_args import add_Family_tree_arguments
from gasspy.shared_utils.spectra_functions import integrated_line
from gasspy.io import gasspy_io

simdir = "/mnt/data/research/cinn3d/inputs/ramses/old/SEED1_35MSUN_CDMASK_WINDUV2"
min_lref = 6
max_lref = 11

class ray_debugger(FamilyTree):
    def __init__(self, 
        root_dir="./",
        gasspy_subdir="GASSPY",
        gasspy_spec_subdir="spec",
        gasspy_projection_subdir="projections",
        traced_rays=None,
        energy=None,
        energy_lims=None,
        Emask=None,
        h5database = None,
        opc_per_NH=False,
        cell_index_to_gasspydb=None,
        vel=None,
        den=None,
        massden=True,
        mu=1.1,
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
        doppler_shift = False):
        super().__init__(
            root_dir=root_dir,
            gasspy_subdir=gasspy_subdir,
            gasspy_spec_subdir=gasspy_spec_subdir,
            gasspy_projection_subdir=gasspy_projection_subdir,
            traced_rays=traced_rays,
            energy=energy,
            energy_lims=energy_lims,
            Emask=Emask,
            h5database = h5database,
            opc_per_NH=opc_per_NH,
            cell_index_to_gasspydb=cell_index_to_gasspydb,
            vel=vel,
            den=den,
            massden=massden,
            mu=mu,
            accel=accel,
            dtype=dtype,
            liteVRAM=liteVRAM,
            Nraster=Nraster,
            useGasspyEnergyWindows=useGasspyEnergyWindows,
            make_spec_subdirs=make_spec_subdirs,
            config_yaml=config_yaml,
            spec_save_type=spec_save_type,
            spec_save_name=spec_save_name,
            cuda_device=cuda_device,
        )
        self.rayid = -1

    def load_new_global_rays(self):
        super().load_new_global_rays()
        # Get leaf rays
        ileafs = self.numlib.where(self.new_global_rays.get_field("cevid") == -1)[0]
        self.leaf_rays = self.new_global_rays.get_subset(ileafs)
        self.index_in_leaf_arrays = self.numlib.zeros(self.new_global_rays.nrays, dtype=int)
        if self.liteVRAM: 
            self.index_in_leaf_arrays[self.leaf_rays["global_rayid"]] = self.numlib.arange(int(len(ileafs)))
        else:
            self.index_in_leaf_arrays[self.leaf_rays["global_rayid"].get()] = self.numlib.arange(int(len(ileafs)))

    def load_traced_rays(self):
        super().load_traced_rays()
        self.NcellPerSeg = self.raydump_dict["pathlength"].shape[1]
    
    def set_complete_path_leaf(self):

        # Determine the total number off cell intersections in the path
        next_rayid = self.rayid
        Nintersects = 0
        while next_rayid >= 0:
            current_rayid = next_rayid
            Nsegs = int(self.raydump_dict["Nsegs"][current_rayid])
            Nintersects += Nsegs * self.NcellPerSeg
            if self.liteVRAM:
                pevid = self.new_global_rays.pevid[current_rayid]
            else:
                pevid = self.new_global_rays.pevid[current_rayid].get()
            if pevid >= 0 :
                next_rayid = int(self.raydump_dict["splitEvents"][pevid, 0])
            else:
                next_rayid = -1    

        # Information about the ray
        self.path_ray_rayid   = self.numlib.zeros((Nintersects,), dtype = int)
        self.path_ray_lrefine = self.numlib.zeros((Nintersects,), dtype = int)

        # Current cell and pathlength of intersection
        self.path_cell_index  = self.numlib.zeros((Nintersects,), dtype = int)
        self.path_pathlength  = self.numlib.zeros((Nintersects,), dtype = float)
        
        next_rayid = self.rayid

        iinter = 0
        while next_rayid >= 0:
            current_rayid = next_rayid
            iseg_start = self.raydump_dict["ray_index0"][current_rayid]
            Nsegs = self.raydump_dict["Nsegs"][current_rayid]

            pathlength_c  = self.raydump_dict["pathlength"][iseg_start: iseg_start + Nsegs,:].ravel()
            cell_index_c  = self.raydump_dict["cell_index"][iseg_start: iseg_start + Nsegs,:].ravel()
            
            iinter_start = iinter 
            iinter_end   = iinter + Nsegs*self.NcellPerSeg

            self.path_ray_lrefine[iinter_start: iinter_end] = self.new_global_rays.ray_lrefine[current_rayid]
            self.path_ray_rayid[iinter_start: iinter_end]   = current_rayid
            self.path_cell_index[iinter_start: iinter_end]  = self.numlib.flip(cell_index_c)
            self.path_pathlength[iinter_start: iinter_end]  = self.numlib.flip(pathlength_c)

            if self.liteVRAM:
                pevid = self.new_global_rays.pevid[current_rayid]
            else:
                pevid = self.new_global_rays.pevid[current_rayid].get()
            if pevid >= 0 :
                next_rayid = int(self.raydump_dict["splitEvents"][pevid, 0])
            else:
                next_rayid = -1          
            iinter = iinter + Nsegs*self.NcellPerSeg

        #self.path_ray_lrefine = self.numlib.flip(self.path_ray_lrefine)
        #self.path_ray_rayid   = self.numlib.flip(self.path_ray_rayid)
        #self.path_cell_index  = self.numlib.flip(self.path_cell_index)
        #self.path_pathlength  = self.numlib.flip(self.path_pathlength)

        self.path_ray_gasspy_id = self.cell_index_to_gasspydb[self.path_cell_index]

        self.path_totpath = self.numlib.cumsum(self.path_pathlength)

    def set_leaf_ray(self, xp = None, yp = None, rayid = None):
        # If no rayid specified, find the closest matching ray
        if rayid is None:
            assert xp is not None and yp is not None, "ERROR (set_leaf_ray): If no rayid supplied, need BOTH xp and yp"
            minindex = self.numlib.argmin(self.numlib.sqrt(
                                                         self.numlib.square(xp - self.leaf_rays["xp"]) +
                                                         self.numlib.square(yp - self.leaf_rays["yp"])
                                                         ))
            rayid = self.leaf_rays["global_rayid"][minindex]
        
        # If this is a new ray, regenerate complete path
        if rayid != self.rayid:
            self.rayid = rayid
            self.set_complete_path_leaf()
            

    def set_ray_spectra(self, Elims, cuda_device, lines = False):
        if self.accel == "torch":
            energy_np = self.energy.cpu().numpy()
        else: 
            energy_np = self.energy.get()
        if Elims is not None:
            Eidxs = np.where( (energy_np >= Elims[0]) * (energy_np < Elims[1]))[0]
        else:
            Eidxs = np.arange(0,len(energy_np))
        if self.accel == "torch":
            my_Em  = torch.as_tensor(self.em.take(self.path_ray_gasspy_id, axis = 1).take(Eidxs, axis = 0), device = cuda_device)
            my_Opc = torch.as_tensor(self.op.take(self.path_ray_gasspy_id, axis = 1).take(Eidxs, axis = 0), device = cuda_device)
            my_pathlengths = torch.as_tensor(self.path_pathlength, device = cuda_device)

            if self.opc_per_NH:
                my_den = torch.as_tensor(self.den.take(self.path_cell_index), device = cuda_device)
                my_Opc = torch.mul(my_den, my_Opc)

            dF = torch.mul(my_Em, my_pathlengths)[:,:]
            dTau = torch.mul(my_Opc, my_pathlengths)[:,:]
        else:
            my_Em  = cupy.asarray(self.em.take(self.path_ray_gasspy_id, axis = 1).take(Eidxs, axis = 0))
            my_Opc = cupy.asarray(self.op.take(self.path_ray_gasspy_id, axis = 1).take(Eidxs, axis = 0))
            my_pathlengths = cupy.asarray(self.path_pathlength)

            if self.opc_per_NH:
                my_den = cupy.asarray(self.den.take(self.path_cell_index))
                my_Opc = cupy.multiply(my_den, my_Opc)

            dF = cupy.mul(my_Em, my_pathlengths) [:,:]
            dTau = cupy.mul(my_Opc, my_pathlengths)[:,:]

        if args.accel == "torch":
            Tau = torch.cumsum(dTau, dim = 1)
            outFlux = torch.flip(torch.cumsum(torch.flip(dF * torch.exp(-Tau), dims = [1]), dim = 1),dims = [1])
            
            bb_Em  = torch.sum(my_Em, dim = 0)
            bb_Opc = torch.sum(my_Opc, dim = 0)
            Tau = torch.sum(Tau, dim = 0)
            if lines:
                print("OI")
                print(my_Em.cpu().numpy().shape)
                self.bb_em = integrated_line(energy_np[Eidxs], deltaEnergies=1, fluxes = my_Em.cpu().numpy(), Eaxis=0)
            else:
                self.bb_em  = bb_Em.cpu().numpy()


            self.bb_Opc = bb_Opc.cpu().numpy()
            self.bb_em  = bb_Em.cpu().numpy()
            self.Tau = Tau.cpu().numpy()
            self.outFlux = outFlux.cpu().numpy()
            if lines:
                print("OI")
                self.outFlux = integrated_line(energy_np[Eidxs], deltaEnergies=1, fluxes = outFlux.cpu().numpy(), Eaxis=0)
            else:
                outFlux  = torch.sum(outFlux, dim = 0)
                self.outFlux = outFlux.cpu().numpy()

        else:
            Tau = cupy.cumsum(dTau, dim = 1)
            outFlux = cupy.flip(cupy.cumsum(cupy.flip(dF * cupy.exp(-Tau), axis = 1), axis = 1), axis = 1)
            bb_Em = cupy.sum(my_Em, axis = 0)
            bb_Opc = cupy.sum(my_Opc, axis = 0)
            Tau = cupy.sum(Tau, axis = 0)
            
            if lines:
                print("OI")
                self.bb_em = integrated_line(energy_np[Eidxs], deltaEnergies=1, fluxes = my_Em.get(), Eaxis=0)
            else:
                self.bb_em = bb_Em.get()

            self.bb_opc = bb_Opc.get()
            self.Tau = Tau.get()
            
            if lines:
                print("OI")
                self.outFlux = integrated_line(energy_np[Eidxs], deltaEnergies=1, fluxes = outFlux.get(), Eaxis=0)
            else:
                outFlux  = cupy.sum(outFlux,axis = 0)
                self.outFlux = outFlux.get()
        

    def debug_ray(self, Elims, cuda_device, figaxes = None, h5out = None, lines = False):


        self.set_ray_spectra(Elims, cuda_device, lines)

        if self.liteVRAM:
            totpath    = self.path_totpath
            pathlength = self.path_pathlength
            cell_index = self.path_cell_index
            gasspy_id  = self.path_ray_gasspy_id
            path_rayid  = self.path_ray_rayid
            ray_lrefine = self.path_ray_lrefine

        else:
            totpath    = self.path_totpath.get()
            pathlength = self.path_pathlength.get()
            cell_index = self.path_cell_index.get()
            gasspy_id  = self.path_ray_gasspy_id.get()
            path_rayid  = self.path_ray_rayid.get()
            ray_lrefine = self.path_ray_lrefine.get()


        if figaxes is not None:
            P = figaxes[0].plot(totpath, self.bb_em)
            color = P[0].get_color()
            figaxes[1].plot(totpath, self.outFlux, c = color)
            figaxes[2].plot(totpath, self.Tau, c = color)
            figaxes[3].plot(totpath, ray_lrefine, c = color, label = "rayid = %d, x = %.2e, y = %.2e" % (self.rayid, self.new_global_rays.xp[self.rayid], self.new_global_rays.yp[self.rayid]))
                
        if h5out is not None:
            grpname = "/ray-%d"%(int(self.rayid))
            if grpname in h5out.keys():
                del h5out[grpname]
            grp = h5out.create_group("/ray-%d"%(int(self.rayid)))
            grp.attrs.create("global_rayid", int(self.rayid))
            grp.attrs.create("xp", float(self.new_global_rays.xp[self.rayid]))
            grp.attrs.create("yp", float(self.new_global_rays.yp[self.rayid]))
            grp.create_dataset("totpath", data = totpath)
            grp.create_dataset("pathlength", data = pathlength)
            grp.create_dataset("cell_index", data = cell_index)
            grp.create_dataset("gasspy_id", data = gasspy_id)
            grp.create_dataset("path_rayid", data = path_rayid)
            grp.create_dataset("ray_lrefine", data = ray_lrefine)
            grp.create_dataset("bb_em",  data = self.bb_em)
            grp.create_dataset("bb_Opc", data = self.bb_Opc)
            grp.create_dataset("outFlux", data = self.outFlux)
            grp.create_dataset("Tau", data = self.Tau)

        
if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--Emin", default = None, type=float)
    ap.add_argument("--Emax", default = None, type=float)
    ap.add_argument("--xp", nargs = "+", type = float, default = [None,])
    ap.add_argument("--yp", nargs = "+", type = float, default = [None,])
    ap.add_argument("--global_rayid", nargs = "+", type = int, default = [None,])
    ap.add_argument("--out_hdf5", type = str, default=None)
    ap.add_argument("--append_hdf5", action="store_true")
    ap.add_argument("--lines", action = "store_true")
    ap.add_argument("--simdir", default="./", help="Directory of the simulation and also default work directory")
    ap.add_argument("--workdir", default= None, help="work directory. If not specified its the same as simdir")
    ap.add_argument("--gasspydir", default="GASSPY", help="directory inside of simdir to put the GASSPY files")
    ap.add_argument("--modeldir" , default="GASSPY_DATABASE", help = "directory inside of workdir where to read, put and run the cloudy models")
    ap.add_argument("--sim_prefix", default = None, help="prefix to put before all snapshot specific files")
    ap.add_argument("--trace_file", default = None, help="name of trace file. If it does not exist we need to recreate it")    
    ap.add_argument("--simulation_reader_dir", default="./", help="directory to the simulation_reader class that describes how to load the simulation")

    # Add arguments related to the FamilyTree class
    add_Family_tree_arguments(ap)
    args = ap.parse_args()

    ## move to workdir
    if args.workdir is not None:
        workdir = args.workdir
    else:
        workdir = args.simdir
    os.chdir(workdir)

    ## set prefix to snapshot specific files
    if args.sim_prefix is not None:
        ## add an underscore
        sim_prefix = args.sim_prefix + "_"
    else:
        sim_prefix = ""

    ## Load the gasspy_config yaml
    gasspy_config = gasspy_io.read_fluxdef("./gasspy_config.yaml")

    ## Load the simulation data class from directory
    spec = importlib.util.spec_from_file_location("simulation_reader", args.simulation_reader_dir + "/simulation_reader.py")
    reader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader_mod)
    sim_reader = reader_mod.Simulation_Reader(args.simdir, args.gasspydir, gasspy_config["sim_reader_args"])

    if args.trace_file is not None:
        trace_file = args.gasspydir+"/projections/"+args.trace_file
    else:
        trace_file = args.gasspydir+"/projections/"+sim_prefix+"trace.hdf5"

    if args.single_precision:
        dtype = np.float32
    else:
        dtype = np.float64
    cuda_device = torch.device('cuda:0')
    if args.Emax is not None:
        Elims = np.array([args.Emin, args.Emax])
    else:
        Elims = None
    ray_debug = ray_debugger(
        root_dir=args.root_dir,
        gasspy_subdir=args.gasspydir,
        traced_rays=trace_file,
        energy_lims=np.atleast_2d(Elims),
        h5database=args.modeldir + "/gasspy_database.hdf5",
        vel=None, #sim_reader.get_field("velocity"),
        den=sim_reader.get_number_density(),
        massden = False,
        opc_per_NH=args.opc_per_NH,
        mu=args.mu,
        accel=args.accel,
        liteVRAM=args.liteVRAM,
        Nraster=4,
        spec_save_name=args.spec_save_name,
        dtype=dtype,
        spec_save_type=args.spec_save_type,
        config_yaml=args.config_yaml,
        cell_index_to_gasspydb = args.gasspydir + "/cell_data/" + sim_prefix+"cell_gasspy_index.npy",
        cuda_device=cuda_device,
        doppler_shift=False
    )   

    if args.out_hdf5 is not None:
        outfile = args.out_hdf5 
        if args.append_hdf5:
            h5out = hp.File(outfile, "a")
        else:
            h5out = hp.File(outfile, "w")
    else:
        h5out = None

    ray_debug.load_all()
    import matplotlib.pyplot as plt

    fig , axes = plt.subplots(figsize = (8,2), ncols=4, nrows = 1, sharex=True)
    axemis = axes[0]
    axflux = axes[1]
    axTau = axes[2]
    axRayLref = axes[3]

    if args.xp[0] != None:
        assert args.yp[0] != None, "Error: must supply both xp and yp"
        assert len(args.yp) == len(args.xp), "Error: xp and yp must be the same shape"

        for ipair in range(len(args.xp)):
            print(args.xp[ipair], args.yp[ipair])
            ray_debug.set_leaf_ray(xp = args.xp[ipair], yp = args.yp[ipair])
            ray_debug.debug_ray(Elims, cuda_device, axes, h5out, lines =args.lines)


    if args.global_rayid[0] != None:
        for irayid in range(len(args.global_rayid)):
            ray_debug.set_leaf_ray(rayid = args.global_rayid[irayid])
            ray_debug.debug_ray(Elims, cuda_device, axes, h5out, lines = args.lines)

    axemis.set_yscale("log")
    axemis.set_ylabel("Emissivity")
    axflux.set_yscale("log")
    axflux.set_ylabel("Cumulative flux")
    axTau.set_yscale("log")
    axTau.set_ylabel("Optical Depth")
    axRayLref.set_ylabel("ray refinement level")
    axRayLref.legend()
    plt.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.1,
        right=0.97,
        hspace=0.15,
        wspace=0.4)
    plt.show()