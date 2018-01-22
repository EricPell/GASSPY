lastmoddate = "22/01/2018.EWP"

mask_parameters_dict = {
    "x":"default",\
    "y":"default",\
    "z":"default",\
    "temp":"default"}

CLOUDY_INIT_FILE = "default_postprocess.ini"

# Specifies variable CLOUDY_modelIF for make-cloudy-input.py -> sets models to single zone
CLOUDY_modelIF   = False

debug = False

# Specify the compression ratio of physical parameters in the grid

compression_ratio = {
    'dx':1.0,\
    'dens':2.0,\
    'temp':1.0,\
    'flge':2.0,\
    'fluv':2.0,\
    'flih':2.0,\
    'fli2':2.0}

append_db = True