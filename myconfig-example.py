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

line_labels = ["N  2 205.4m", "N  2  6584A", "N  2  5755A",\
               "O  3 88.33m", "O  3  5007A", "TOTL  4363A",\
               "O II  3729A", "O II  3726A", "O II  7323A", "O II  7332A", "TOTL  3727A",\
               "O  1 63.17m", "O  1 145.5m", "O  1  6300A",\
               "S  3 18.67m", "S  3 33.47m", "S  3  9532A", "S  3  9069A", "S  3  6312A",\
               "S II  6716A", "S II  6731A", "S  2  6720A",\
               "H  1  6563A", "H  1  4861A",\
               "CO    1300m", "CO    2600m", "CO   866.7m", "CO   371.5m",\
               "SI 2 34.81m"]

CLOUDY_INIT_FILE = "silcc_flash_postprocess.ini"