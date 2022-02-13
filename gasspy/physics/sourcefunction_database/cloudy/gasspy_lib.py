"""
gasspy LIBRARY TEST
"""
import sys
import numpy as np
import scipy.constants
import yaml
import gasspy, pathlib
gasspy_path = pathlib.Path(gasspy.__file__).resolve().parent
"""
import
"""

def report(test_results):
    all_tests_passed = True
    for key in test_results.keys():
        if test_results[key] == False:
            all_tests_passed = False
        print ("%s: %s"%(key, test_results[key]))
    
    print("ALL TEST PASSED: %s"%(all_tests_passed))

test_results={}
try:
    from gasspy.physics.sourcefunction_database.cloudy import gasspy_cloudy_db_classes
    test_results["import library"] = True
except:
    test_results["import library"] = False
    sys.exit("gasspy IMPORT: FAILED")

"""
create an instance
"""
try:
    creator = gasspy_cloudy_db_classes.uniq_dict_creator(save_compressed3d="saved3d_cloudyfields")
    test_results["initialize creator class"] = True
except:
    test_results["initialize creator class"] = False

with open(r'/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/infos/info_00051.yaml') as file:
    info = yaml.load(file, Loader=yaml.FullLoader)
"""
read test data
"""
try:
    import astropy.io.fits
    test_results["astropy imported"] = True
except:
    test_results["astropy imported"] = False

fits_files = {
    "fluxes":
    {"NpFUV":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/NpFUV/celllist_NpFUV_00051.fits",
    "NpHII":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/NpHII/celllist_NpHII_00051.fits",
    "NpHeIII":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/NpHeII/celllist_NpHeII_00051.fits",
    "NpHeII":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/NpHeIII/celllist_NpHeIII_00051.fits",
    },
    "T":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/T/celllist_T_00051.fits",
    "rho":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/rho/celllist_rho_00051.fits",
    "x":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/x/celllist_x_00051.fits",
    "y":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/y/celllist_y_00051.fits",
    "z":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/z/celllist_z_00051.fits",
    "vx":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/vx/celllist_vx_00051.fits",
    "vy":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/vy/celllist_vy_00051.fits",
    "vz":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/vz/celllist_vz_00051.fits",
    "amr":"/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/amrlevel/celllist_amrlevel_00051.fits"
}

test_results["all fits files read"] = True
fits_dict = {}
fits_dict['fluxes'] = {}

for field in fits_files.keys():
    try:
        if field != "fluxes":
            fits_dict[field] = astropy.io.fits.open(fits_files[field])
            test_results["read %s"%fits_files[field]] = True
    except:
        test_results["read %s"%fits_files[field]] = False
        test_results["all fits files read"] = False

for field in fits_files["fluxes"].keys():
    try:
        fits_dict['fluxes'][field] = astropy.io.fits.open(fits_files["fluxes"][field])
        test_results["read %s"%fits_files["fluxes"][field]] = True

    except:
        test_results["read %s"%fits_files["fluxes"][field]] = False
        test_results["all fits files read"] = False

#dx in cm
dx = np.log10(info["boxlen"] / 2**fits_dict["amr"][0].data.astype(np.float32))

try:
    creator.simdata = {
        "temp"  :np.log10(fits_dict["T"][0].data.ravel()),

        "dens"  :np.log10(fits_dict["rho"][0].data.ravel() / 1e-24),

        "dx"    :dx.ravel(),

        "x"     :fits_dict["x"][0].data.ravel(),
        "y"     :fits_dict["y"][0].data.ravel(),
        "z"     :fits_dict["z"][0].data.ravel(),

        "vx"     :fits_dict["vx"][0].data.ravel(),
        "vy"     :fits_dict["vy"][0].data.ravel(),
        "vz"     :fits_dict["vz"][0].data.ravel(),

        "fluxes":{
            "FUV":{
                "Emin":0.1,
                "Emax":13.59844,
                "shape":"specFUV.sed",
                "data":np.log10(fits_dict['fluxes']["NpFUV"][0].data.ravel()),
                },

            "HII":{
                "Emin":13.59844,
                "Emax":24.58741,
                "shape":"specHII.sed",
                "data":np.log10(fits_dict['fluxes']["NpHII"][0].data.ravel())
                },

            "HeII":{
                "Emin":24.58741,
                "Emax":54.41778,
                "shape":"specHeII.sed",
                "data":np.log10(fits_dict['fluxes']["NpHeII"][0].data.ravel())
                },

            "HeIII":{
                "Emin":54.41778,
                "Emax":100.0000,
                "shape":"specHeIII.sed",
                "data":np.log10(fits_dict['fluxes']["NpHeIII"][0].data.ravel())
                }
        }
    }
    creator.log10_flux_low_limit={
            "FUV":-3.0,
            "HII":-3.0,
            "HeII":-4.0,
            "HeIII":-6.0
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

    creator.outdir="/home/pellegew/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV/"
    creator.outname = "gasspy"


    del(fits_dict)
    test_results["create simdata attribute"] = True
except:
    test_results["create simdata attribute"] = False

try:
    creator.compress_simdata()
    test_results["collect_den_temp_flux"] = True
except:
    test_results["collect_den_temp_flux"] = False

try:
    print("Trimming compressed data to uniqueDF")
    creator.trim()
    test_results["collect_den_temp_flux"] = True
except:
    test_results["collect_den_temp_flux"] = False

try:
    gasspy_to_cloudy = gasspy_cloudy_db_classes.gasspy_to_cloudy(outdir=creator.outdir)
    test_results["init gasspy_to_cloudy class"] = True
except:
    test_results["init gasspy_to_cloudy class"] = False

report(test_results)
