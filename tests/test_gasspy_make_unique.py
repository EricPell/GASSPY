"""
gasspy LIBRARY TEST
"""
import sys
import numpy as np
import scipy.constants

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
    from gasspy import gasspy_classes
    test_results["import library"] = True
except:
    test_results["import library"] = False
    sys.exit("gasspy IMPORT: FAILED")

"""
create an instance
"""
creator = gasspy_classes.uniq_dict_creator()
try:
    creator = gasspy_classes.uniq_dict_creator()
    test_results["initialize creator class"] = True
except:
    test_results["initialize creator class"] = False

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
    {"NpFUV":"/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/NpFUV/cube_NpFUV_00051.fits",
    "NpHII":"/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/NpHII/cube_NpHII_00051.fits",
    "NpHeIII":"/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/NpHeII/cube_NpHeII_00051.fits",
    "NpHeII":"/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/NpHeIII/cube_NpHeIII_00051.fits",
    },
    "T":"/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/T/cube_T_00051.fits",
    "rho":"/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/rho/cube_rho_00051.fits",
    "x":"/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/x/cube_x_00051.fits",
    "y":"/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/y/cube_y_00051.fits",
    "z":"/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/z/cube_z_00051.fits"
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
dx = fits_dict["x"][0].data*3.08e18
dx[:,1:,:] = np.abs(dx[:,1:,:] - dx[:,0:-1,:])
dx[:,0,:] = dx[:,1,:]
dx = np.log10(dx)

try:
    creator.simdata = {
        "temp"  :np.log10(fits_dict["T"][0].data.ravel()),

        "dens"  :np.log10(fits_dict["rho"][0].data.ravel() / 1e-24),

        "dx"    :dx.ravel(),

        "x"     :fits_dict["x"][0].data.ravel(),
        "y"     :fits_dict["y"][0].data.ravel(),
        "z"     :fits_dict["z"][0].data.ravel(),

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

    creator.outdir="/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/"
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
    print("compression ratio: %f"%creator.trim())
    test_results["trim"] = True
except:
    test_results["trim"] = False

del(creator)

# try:
#     gasspy_to_cloudy = gasspy_classes.gasspy_to_cloudy()
    
#     test_results["init gasspy_to_cloudy class"] = True
# except:
#     test_results["init gasspy_to_cloudy class"] = False


# try:
#     # Create a cloud input model for each unique data point
#     gasspy_to_cloudy.process_grid(creator.compressedsimdata)
#     test_results["create cloudy files from grid"] = True
# except:
#     test_results["create cloudy files from grid"] = False


report(test_results)
