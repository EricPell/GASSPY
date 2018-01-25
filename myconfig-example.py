# Config parameters
inFile = "SILCC_hdf5_plt_cnt_0403"
radfields = ["flge", "fluv", "flih", "fli2"]
flux_type = "fervent"

# Set min-max limits for dimensions and temperature
mask_parameters_dict={
    "mask1":{
        "x":[-250*3.08e18, 50*3.08e18],\
        "y":[-250*3.08e18, 50*3.08e18],\
        "z":[-50.*3.08e18, 250*3.08e18],\
        "temp":[1e3,2e4]
        },\

    "mask2":{
        "x":[-250*3.08e18, 50*3.08e18],\
        "y":[-250*3.08e18, 50*3.08e18],\
        "z":[-50.*3.08e18, 250*3.08e18],\
        "temp":[1e3,2e4]
        }    
}

ForceFullDepth = True

opiate_library = 'silcc-combined-ems.tbl'
opiate_lookup  = 'silcc.unique_parameters'


line_labels = ["N  2      121.767m", "N  2      205.244m", "N  2      6583.45A", "Blnd      5755.00A",\
               "O  3      88.3323m", "O  3      5006.84A", "Blnd      4363.00A",\
               "O  2      3726.03A", "Blnd      3727.00A", "O  2      3728.81A", "Blnd      7323.00A", "Blnd      7332.00A",\
               "O  1      63.1679m", "O  1      145.495m", "Blnd      6300.00A",\
               "S  3      18.7078m", "S  3      33.4704m", "S  3      9530.62A", "S  3      9068.62A", "S  3      6312.06A",\
               "S  2      6716.44A", "S  2      6730.82A", "Blnd      6720.00A",\
               "H  1      6562.81A", "H  1      4861.33A",\
               "C  2      157.636m",\
               "C  1      370.269m", "C  1      609.590m",\
               "Blnd      2.12100m",\
               "CO        2600.05m",\
               "CO        1300.05m",\
               "CO        866.727m",\
               "CO        650.074m",\
               "CO        520.089m",\
               "CO        433.438m",\
               "CO        371.549m",\
               "CO        325.137m",\
               "CO        289.041m",\
               "CO        260.169m",\
               "CO        236.549m",\
               "CO        216.868m",\
               "CO        200.218m",\
               "HCO+      373.490m",\
               "HCN       375.844m",\
               "Si 2      34.8046m"]

CLOUDY_INIT_FILE = "silcc_flash_postprocess.ini"

# Old line labels from pre-2017
line_labels = ["N  2 205.4m", "N  2  6584A", "N  2  5755A",\
               "O  3 88.33m", "O  3  5007A", "TOTL  4363A",\
               "O II  3729A", "O II  3726A", "O II  7323A", "O II  7332A", "TOTL  3727A",\
               "O  1 63.17m", "O  1 145.5m", "O  1  6300A",\
               "S  3 18.67m", "S  3 33.47m", "S  3  9532A", "S  3  9069A", "S  3  6312A",\
               "S II  6716A", "S II  6731A", "S  2  6720A",\
               "H  1  6563A", "H  1  4861A",\
               "CO    1300m", "CO    2600m", "CO   866.7m", "CO   371.5m",\
               "SI 2 34.81m"]
