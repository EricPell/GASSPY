parameters = {
    "lastmoddate":"2021.06.05.EWP",

    "mask_parameters_dict" : {
        "x":"default",\
        "y":"default",\
        "z":"default",\
        "temp":"default"},

    "CLOUDY_INIT_FILE" : "default_postprocess.ini",

    """ Set Cloudy to calculate a full calculation instead of a single zone """
    "ForceFullDepth" : True,

    """ Set class of energy bands used for radiative transfer. Note: default, which is not defined, will cause the code to abort """
    "flux_definition" : "default",


    """ DEPRECATED: REMOVE """
    # Specifies variable CLOUDY_modelIF for make-cloudy-input.py -> checks model for H/H+ IF and sets models to single zone if not.
    "CLOUDY_modelIF"   : False,

    "debug" : False,

    # Specify the compression ratio of physical parameters in the grid

    "compression_ratio" : {
        'dx':(3, 1.0),\
        'dens':(1, 5.0),\
        'temp':(1, 5.0),\
        'default':(1, 5.0),\
        'fluxes':{'default':(1,5.0)}
    },

    "log10_flux_low_limit" : {'default':-5.0},

    "append_db" : False
}