   
def add_Family_tree_arguments(ap):
    ap.add_argument("--root_dir", default = "./", help = "root workspace directory")
    ap.add_argument("--gasspy_subdir", type = str, default = "GASSPY", help = "name of gasspy subdirectory inside of root_dir")
    ap.add_argument("--gasspy_projection_subdir", type = str, default = "projections", help = "name of directory inside of gasspy_subdir to search for traced rays")
    ap.add_argument("--traced_rays", type = str,default = "000000_trace.hdf5", help = "name of raytrace file")
    ap.add_argument("--em", type = str, default = "gasspy_avg_em.pkl.npy", help = "name of file containing the gasspy average emissivies")
    ap.add_argument("--op", type = str, default = "gasspy_grn_opc.pkl.npy", help = "name of file containing the gasspy opacities")
    ap.add_argument("--vel", type = str, default = "/vx/celllist_vx_00051.fits", help = "Name of file containing the velocities of the cells")
    ap.add_argument("--den", type = str, default = "/rho/celllist_rho_00051.fits", help = "name of file containing the densiities of the cells")
    ap.add_argument("--saved3d", type = str, default = "saved3d_cloudyfields.npy", help = "name of file containing the cell index to gasspy physics index map")
    ap.add_argument("--gasspy_id", type = str, default = "saved3d_cloudyfields.npy", help = "name of file containing the cell index to gasspy physics index map")

    ap.add_argument("--config_yaml", type = str, default = "./gasspy_config.yaml", help = "name of file containing the cell index to gasspy physics index map")
    
    ap.add_argument("--spec_save_type", type = str, default = 'hdf5', help ="save type to use for the spectra (currently only HDF5 supported)")
    ap.add_argument("--spec_save_name", type = str, default = "gasspy_spec", help ="Name of file to save spectra to")   
    
    ap.add_argument("--energy", type = str,default = "gasspy_ebins.pkl.npy", help = "name of file containing the energy bins")
    ap.add_argument("--energy_lims", type = float, default = [None], nargs = "+", help = "sets of two energies (Emin1 Emax1 Emin2 Emax2) between which spectra is considered")
    ap.add_argument("--opc_per_NH", action = "store_true", help = "Switch for if the opacities are given as a per number density of hydrogen in cgs")
    ap.add_argument("--mu", type = float, default = 1.1, help = "Mean molecular weight of the simulation")

    ap.add_argument("--accel", type = str, default = "torch", help = "GPU accelerator, torch: use torch functions, otherwise use cupy")
    ap.add_argument("--liteVRAM", action = "store_true", help = "Switch for if data tables are to be stored on the system or device memory")
    ap.add_argument("--Nraster", type = int, default = 4, help = "Number of subrays that are spawned from the split of one ray (CURRENTLY ONLY 4 SUPPORTED)")
    ap.add_argument("--single_precision", action = "store_true", help = "Switch for saving all floats as single precision, drastically improving preformance but downgrades precision")

    return