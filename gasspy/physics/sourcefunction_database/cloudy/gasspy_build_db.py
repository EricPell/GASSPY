"""
gasspy LIBRARY TEST
"""
import os
import sys, pathlib
import numpy as np
import scipy.constants
import yaml
from astropy.io import fits
import gasspy
from gasspy.physics.sourcefunction_database.cloudy import gasspy_cloudy_db_classes
from gasspy.io import gasspy_io
gasspy_path = pathlib.Path(gasspy.__file__).resolve().parent

with open(r'%s/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/infos/info_00051.yaml'%os.environ["HOME"]) as file:
    info = yaml.load(file, Loader=yaml.FullLoader)

fluxdef_file = os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/gasspy_fluxdef.yaml"

simdata_files = {
    "fluxes":
    {"FUV":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/NpFUV/celllist_NpFUV_00051.fits",
    "HII":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/NpHII/celllist_NpHII_00051.fits",
    "HeIII":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/NpHeII/celllist_NpHeII_00051.fits",
    "HeII":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/NpHeIII/celllist_NpHeIII_00051.fits",
    },
    "T":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/T/celllist_T_00051.fits",
    "rho":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/rho/celllist_rho_00051.fits",
    "x":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/x/celllist_x_00051.fits",
    "y":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/y/celllist_y_00051.fits",
    "z":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/z/celllist_z_00051.fits",
    "vx":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/vx/celllist_vx_00051.fits",
    "vy":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/vy/celllist_vy_00051.fits",
    "vz":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/vz/celllist_vz_00051.fits",
    "amr":os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/amrlevel/celllist_amrlevel_00051.fits"
}

simdata_dict = {}
simdata_dict['fluxes'] = {}

for field in simdata_files.keys():
    if field != "fluxes":
        simdata_dict[field] = fits.open(simdata_files[field])

for field in simdata_files["fluxes"].keys():
    simdata_dict['fluxes'][field] = fits.open(simdata_files["fluxes"][field])

#dx in cm
dx = np.log10(info["boxlen"] / 2**simdata_dict["amr"][0].data.astype(np.float32))

fluxdef = gasspy_io.read_fluxdef(fluxdef_file)

creator = gasspy_cloudy_db_classes.uniq_dict_creator()

creator.simdata = {
    "temp"  :np.log10(simdata_dict["T"][0].data.ravel()),

    "dens"  :np.log10(simdata_dict["rho"][0].data.ravel() / 1e-24),

    "dx"    :dx.ravel(),

    "fluxes":fluxdef
}

for radiation_field_name in creator.simdata['fluxes']:
    creator.simdata['fluxes'][radiation_field_name]['data'] = np.log10(simdata_dict['fluxes'][radiation_field_name][0].data.ravel())

creator.log10_flux_low_limit={
        "FUV":-5.0,
        "HII":-5.0,
        "HeII":-6.0,
        "HeIII":-7.0
    }
creator.compression_ratio ={
    'dx':(3, 1.0),\
    'dens':(1, 2.0),\
    'temp':(1, 5.0),\
    'fluxes':{
    'FUV':(1, 5.0),\
    'HII':(1, 5.0),\
    'HeII':(1, 5.0),\
    'HeIII':(1, 5.0)}
    }

creator.save_compressed3d = "saved3d_cloudyfields"

creator.outdir=os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/"

creator.outname = "gasspy"

creator.compress_simdata()

creator.trim()

gasspy_to_cloudy = gasspy_cloudy_db_classes.gasspy_to_cloudy(outdir=creator.outdir, outname =creator.outname, CLOUDY_INIT_FILE="spec_postprocess-c17.ini")
gasspy_to_cloudy.process_grid()