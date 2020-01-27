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
        if test_results[key] == "FAILED":
            all_tests_passed = False
        print ("%s: %s"%(key, test_results[key]))
    
    print("ALL TEST PASSED: %s"%(all_tests_passed))

test_results={}
try:
    import opiate_lib
    test_results["import library"] = "PASS"
except:
    test_results["import library"] = "FAILED"
    sys.exit("OPIATE IMPORT: FAILED")

"""
create an instance
"""
try:
    creator = opiate_lib.uniq_dict_creator()
    test_results["initialize creator class"] = "PASS"
except:
    test_results["initialize creator class"] = "FAILED"

"""
read test data
"""
try:
    import astropy.io.fits
    test_results["astropy imported"] = "PASS"
except:
    test_results["astropy imported"] = "FAILED"


fits_files = {"NpHII":"cube_64_NpHII_00020.fits",
"NpHeIII":"cube_64_NpHeIII_00020.fits",
"NpHeII":"cube_64_NpHeII_00020.fits",
"T":"cube_64_T_00020.fits",
"rho":"cube_64_rho_00020.fits"}

test_results["all fits files read"] = "PASS"
fits_dict = {}
for field in fits_files.keys():
    try:
        fits_dict[field] = astropy.io.fits.open("fits_test/"+fits_files[field])
        test_results["read %s"%fits_files[field]] = "PASS"
    except:
        test_results["read %s"%fits_files[field]] = "FAILED"
        test_results["all fits files read"] = "FAILED"

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

        "dens"  :fits_dict["rho"][0].data.ravel() / 1e-24,

        "dx"    :dx.ravel(),

        "x"     :xyz[0].ravel(),
        "y"     :xyz[1].ravel(),
        "z"     :xyz[2].ravel(),
        
        "Flux_NpHII":{"Emin":13.59844,
                "Emax":24.58741,
                "data":fits_dict["NpHII"][0].data.ravel() / dx.ravel()**2.0 * scipy.constants.c},

        "Flux_NpHeII":{"Emin":24.58741,
                "Emax":54.41778,
                "data":fits_dict["NpHeII"][0].data.ravel() / dx.ravel()**2.0 * scipy.constants.c},

        "Flux_NpHeIII":{ "Emin":54.41778,
                "Emax":100.0000,
                "data":fits_dict["NpHeIII"][0].data.ravel() / dx.ravel()**2.0 * scipy.constants.c}
    }
    test_results["create simdata attribute"] = "PASS"
except:
    test_results["create simdata attribute"] = "FAILED"


try:
    creator.collect_den_temp_flux()
    test_results["create uniqe data"] = "PASSED"
except:
    test_results["create uniqe data"] = "FAILED"


report(test_results)
