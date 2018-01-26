lastmoddate = "22/01/2018.EWP"

radfields = ["flge", "fluv", "flih", "fli2"]

mask_parameters_dict = {
    "x":"default",\
    "y":"default",\
    "z":"default",\
    "temp":"default"}

CLOUDY_INIT_FILE = "default_postprocess.ini"

""" Set Cloudy to calculate a full calculation instead of a single zone """
ForceFullDepth = True

""" Set class of energy bands used for radiative transfer. Note: default, which is not defined, will cause the code to abort """
flux_type = "default"


""" DEPRECATED: REMOVE """
# Specifies variable CLOUDY_modelIF for make-cloudy-input.py -> checks model for H/H+ IF and sets models to single zone if not.
CLOUDY_modelIF   = True

debug = False

# Specify the compression ratio of physical parameters in the grid

compression_ratio = {
    'dx':(3, 1.0),\
    'dens':(1, 2.0),\
    'temp':(1, 1.0),\
    'flge':(1, 2.0),\
    'fluv':(1, 2.0),\
    'flih':(1, 2.0),\
    'fli2':(1, 2.0),\
    'euve':(1, 1.0)}

log10_flux_low_limit = {
    'flge':-5,\
    'fluv':0,\
    'flih':0,\
    'fli2':0,\
    'euve':-5}
"""'flge': log erg -s cm-2
'fluv': log photon number flux
'flih': log photon number flux
'fli2': log photon number flux
'euve': log erg -s cm-2 """

append_db = True