"""
OPIATE LIBRARY TEST
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
    import opiate_lib
    test_results["import library"] = True
except:
    test_results["import library"] = False
    sys.exit("OPIATE IMPORT: FAILED")

"""
create an instance
"""
creator = opiate_lib.uniq_dict_creator()
try:
    creator = opiate_lib.uniq_dict_creator()
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


fits_files = {"NpHII":"cube_64_NpHII_00020.fits",
"NpHeIII":"cube_64_NpHeIII_00020.fits",
"NpHeII":"cube_64_NpHeII_00020.fits",
"T":"cube_64_T_00020.fits",
"rho":"cube_64_rho_00020.fits"}

test_results["all fits files read"] = True
fits_dict = {}
for field in fits_files.keys():
    try:
        fits_dict[field] = astropy.io.fits.open("fits_test/"+fits_files[field])
        test_results["read %s"%fits_files[field]] = True
    except:
        test_results["read %s"%fits_files[field]] = False
        test_results["all fits files read"] = False

side_length_box_pc = 54.0544

dx = np.full(np.shape(fits_dict["rho"][0].data), side_length_box_pc * 3.08e18 / np.mean(np.shape(fits_dict["rho"][0].data)))

Nx,Ny,Nz = np.shape(fits_dict["rho"][0].data)
xs = np.linspace(0,side_length_box_pc,Nx)
ys = np.linspace(0,side_length_box_pc,Ny)
zs = np.linspace(0,side_length_box_pc,Nz)

xyz = np.meshgrid(xs,ys,zs)

try:
    
    creator.simdata = {
        "temp"  :fits_dict["T"][0].data.ravel(),

        "dens"  :np.log10(fits_dict["rho"][0].data.ravel() / 1e-24),

        "dx"    :dx.ravel(),

        "x"     :xyz[0].ravel(),
        "y"     :xyz[1].ravel(),
        "z"     :xyz[2].ravel(),
        
        "flux":{
            0:{"Emin":13.59844,
                "Emax":24.58741,
                "shape":"const",
                "data":fits_dict["NpHII"][0].data.ravel() * scipy.constants.c},

            1:{"Emin":24.58741,
                "Emax":54.41778,
                "shape":"const",
                "data":fits_dict["NpHeII"][0].data.ravel() * scipy.constants.c},

            2:{ "Emin":54.41778,
                "Emax":100.0000,
                "shape":"const",
                "data":fits_dict["NpHeIII"][0].data.ravel() * scipy.constants.c}
        }
    }

    test_results["create simdata attribute"] = True
except:
    test_results["create simdata attribute"] = False


try:
    print("compression ratio: %f"%creator.compress_simdata())
    test_results["collect_den_temp_flux"] = True
except:
    test_results["collect_den_temp_flux"] = False


try:
    opiate_to_cloudy = opiate_lib.opiate_to_cloudy()
    
    test_results["init opiate_to_cloudy class"] = True
except:
    test_results["init opiate_to_cloudy class"] = False


try:
    # Create a cloud input model for each unique data point
    opiate_to_cloudy.process_grid(creator.compressedsimdata)
    test_results["create cloudy files from grid"] = True
except:
    test_results["create cloudy files from grid"] = False


report(test_results)
