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
    from opiate import opiate_classes
    test_results["import library"] = True
except:
    test_results["import library"] = False
    sys.exit("OPIATE IMPORT: FAILED")

try:
    opiate_to_cloudy = opiate_classes.opiate_to_cloudy(outdir="/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/", outname = "opiate")
    
    test_results["init opiate_to_cloudy class"] = True
except:
    test_results["init opiate_to_cloudy class"] = False


try:
    # Create a cloud input model for each unique data point
    opiate_to_cloudy.process_grid()
    test_results["create cloudy files from grid"] = True
except:
    test_results["create cloudy files from grid"] = False


report(test_results)
