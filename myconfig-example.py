# Config parameters
inFile = "SILCC_hdf5_plt_cnt_0403"

# Set min-max limits for dimensions and temperature
mask_parameters_dict={
    "x":[-250*3.08e18, 50*3.08e18],\
    "y":[-250*3.08e18, 50*3.08e18],\
    "z":[-50.*3.08e18, 250*3.08e18],\
    "temp":[1e3,1e5]}

CLOUDY_INIT_FILE = "silcc_flash_postprocess.ini"