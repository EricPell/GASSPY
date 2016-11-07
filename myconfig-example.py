# Config parameters
inFile = "SILCC_hdf5_plt_cnt_0403"

# Set min-max limits for dimensions and temperature
mask_parameters_dict={
    "x":[-250*3.08e18, 50*3.08e18],\
    "y":[-250*3.08e18, 50*3.08e18],\
    "z":[-50.*3.08e18, 250*3.08e18],\
    "temp":[1e3,1e5]}

opiate_library = 'silcc-combined-ems.tbl'
opiate_lookup  = 'silcc.unique_parameters'

line_lables = ["O  3  5007A", "H  1  6563A", "S  2  6720A"]

CLOUDY_INIT_FILE = "silcc_flash_postprocess.ini"