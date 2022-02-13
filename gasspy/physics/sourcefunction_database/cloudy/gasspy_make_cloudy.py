"""
gasspy LIBRARY TEST
"""
import sys
import numpy as np
import scipy.constants
import os

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

if len(sys.argv) == 3:
    outdir  = sys.argv[1]
    outname = sys.argv[2]
else:
    outdir = os.environ["HOME"]+"/research/cinn3d/inputs/ramses/SEED1_35MSUN_CDMASK_WINDUV2/"
    outname = "gasspy"

try:
    gasspy_to_cloudy = gasspy_cloudy_db_classes.gasspy_to_cloudy(outdir=outdir, outname = outname, CLOUDY_INIT_FILE="spec_postprocess-c17.ini")
    
    test_results["init gasspy_to_cloudy class"] = True
except:
    test_results["init gasspy_to_cloudy class"] = False


try:
    # Create a cloud input model for each unique data point
    gasspy_to_cloudy.process_grid()
    test_results["create cloudy files from grid"] = True
except:
    test_results["create cloudy files from grid"] = False


report(test_results)