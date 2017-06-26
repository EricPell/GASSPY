#!/usr/bin/python
""" Routine to make minimum models required for post processing """
#lastmoddate = 18.07.2016.EWP

import os
import sys
import pickle
import compress
sys.path.append(os.getcwd())

###############################################################
# Set default parameters that can be overwritten by my config #
###############################################################
# Sets CLOUDY_modelIF to True
CLOUDY_modelIF = True
###############################################################
#                                                             #
###############################################################
import defaults
import myconfig

try:
    CLOUDY_INIT_FILE = myconfig.CLOUDY_INIT_FILE
except:
    CLOUDY_INIT_FILE = defaults.CLOUDY_INIT_FILE

#"""Decide to force every model to be calculated with full depth, or default to a single zone"""
try:
    if myconfig.ForceFullDepth == True:
        #"""If ForceFullDepth is defined in myconfig as true, set the global to true"""
        ForceFullDepth = True
    else:
        #"""Else, if set to false, set global false"""
        ForceFullDepth = False
except:
    #"""If Force depth was not defined, that's ok. We will set it to true"""
    ForceFullDepth = False


MaxNumberModels = int(1e5)

# Import string containing each continuum shape.
import fervent_bands # Import continuum shapes        


def set_output_and_save_prefix(UniqID, depth, hden, T, I_ge, phi_uv, phi_ih, phi_i2):
    """pass this a dictionary of parameters and values. Loop over and create prefix. Open output """
    try:
        os.stat("./cloudy-output")
    except:
        os.mkdir("./cloudy-output")

    prefix = "./cloudy-output/silcc-%s"%UniqID
    outfile = open(prefix+".in", 'w')
    outfile.write("set save prefix \"%s\""%(prefix)+"\n")
    return outfile

def check_for_if(cell_depth, cell_hden, cell_phi_ih, cell_phi_i2):
    """Check to see if cell contains a Hydrogen ionization front"""
    alpha = 4.0e-13 # H recombinations per second cm-6
    ion_depth = (10**(float(cell_phi_ih)) +\
    10**(float(cell_phi_i2)))/(alpha * 10**(float(cell_hden))**2)

    # Change hardcoded 1e13 to mean free path of ionizing photon in H0.
    if ForceFullDepth is True:
        return True
    elif ion_depth <= 10**float(cell_depth) and ion_depth > 1e13:
        return True
    else:
        return False

def set_cloudy_init_file(outfile, init_file):
    """ Write command to cloudy input file to use custom init file """
    outfile.write("init \"%s\"\n"%(init_file))

def set_depth(outfile, model_depth):
    """Write command to cloudy input file to set depth of model"""
    outfile.write("stop depth %s\n"%(model_depth))

def set_hden(outfile, model_hden):
    """Write command to cloudy input file to set hydrogen density"""
    outfile.write("hden %s\n"%(model_hden))

def set_nend(outfile, model_is_ionization_front):
    """Write command to cloudy input file to set number of zones to simulate"""
    if model_is_ionization_front is True:
        #Do not set constant temperature if IF exists
        outfile.write("set nend 1000\n")

    if model_is_ionization_front is False:
        #Set constant temperature if IF does not exist
        outfile.write("set nend 1\n")

def set_temperature(outfile, temperature, is_ionization_front):
    """Set constant temperature if not modeling the actual ionization front temp gradients"""
    if is_ionization_front is False:
        outfile.write("constant temperature %s\n"%(temperature))

def set_I_ge(outfile,I_ge):
    if I_ge != "-99.00":
        outfile.write(fervent_bands.flge)
        outfile.write("intensity %s, range 0.41 to 0.823 Ryd\n"%(I_ge))

def set_phi_uv(outfile,phi_uv):
    #if phi_uv != "-99.00":
    if phi_uv > 1:
        outfile.write(fervent_bands.fluv)
        outfile.write("phi(h) = %s, range 0.823 to 1.0 Ryd\n"%(phi_uv))

def set_phi_ih(outfile,phi_ih):
    #if phi_ih != "-99.00":
    if phi_ih > 0:
        outfile.write(fervent_bands.flih)
        outfile.write("phi(h) = %s, range 1.0 to 1.117 Ryd\n"%(phi_ih))

def set_phi_i2(outfile,phi_i2):
    #if phi_i2 != "-99.00":
    if phi_i2 > 0:
        outfile.write(fervent_bands.fli2)
        outfile.write("phi(h) = %s, range 1.117 to 3 Ryd\n"%(phi_i2))

def create_cloudy_input_file(_UniqID, _depth, _hden, _T, flux_array, _cloudy_init_file=CLOUDY_INIT_FILE):
    """ crate prefix for models and open Cloudy input file for writing"""
    (_I_ge, _phi_uv, _phi_ih, _phi_i2) = flux_array
    cloudy_input_file = set_output_and_save_prefix(\
     _UniqID,\
     _hden,\
     _depth,\
     _T,\
     _I_ge,\
     _phi_uv,\
     _phi_ih,\
     _phi_i2)

    # CLOUDY_modelIF is set to True by default. Can be changed in parameter file to false,
    # which will prevent isIF from executing

    if(CLOUDY_modelIF):
        isIF = check_for_if(_depth, _hden, _phi_ih, _phi_i2)
    else:
        isIF = False


    """ Set common init file """
    set_cloudy_init_file(cloudy_input_file, _cloudy_init_file)

    """ Write individual cloudy parameters to input file """
    set_depth(cloudy_input_file, _depth)
    set_hden(cloudy_input_file, _hden)
    set_nend(cloudy_input_file, isIF)
    set_temperature(cloudy_input_file, _T, isIF)
    set_I_ge(cloudy_input_file, _I_ge)
    set_phi_uv(cloudy_input_file, _phi_uv)
    set_phi_ih(cloudy_input_file, _phi_ih)
    set_phi_i2(cloudy_input_file, _phi_i2)

    """ Close input file """
    cloudy_input_file.close()


""" Begin main part of code """
input = open(myconfig.opiate_lookup,'r')
parameter_data = input.readlines()

#if(parameter_data[0] != "UniqID  dx      dens    temp    flge    fluv    flih    fli2    N\n"):
    # sys.exit("Header file did not match expected format")
#else:

# Create max depth to re-use deep models for shallower ones.
max_depth = {}
for i in range(1, len(parameter_data)):
    [UniqID, depth, hden, temp, flge, fluv, flih, fli2, NumberOfCellsLike] = parameter_data[i].split("\t")

    # WARNING - Experimental - WARNING 
    # The following code block is hard coded and won't reflect changes to the config file
    hden = compress.number(float(hden), 2, 3.)
    temp = compress.number(float(temp), 1, 3.)
    flge = compress.number(float(flge), 1, 3.)
    fluv = compress.number(float(fluv), 1, 3.)
    flih = compress.number(float(flih), 1, 3.)
    fli2 = compress.number(float(fli2), 1, 3.)
    
    try:
        if float(depth) > max_depth[hden, temp, flge, fluv, flih, fli2]["depth"]:
            max_depth[hden, temp, flge, fluv, flih, fli2]["depth"] = float(depth)
            max_depth[hden, temp, flge, fluv, flih, fli2]["UniqID"] = UniqID
    except:
        max_depth[hden, temp, flge, fluv, flih, fli2] = {}
        max_depth[hden, temp, flge, fluv, flih, fli2]["depth"] = float(depth)
        max_depth[hden, temp, flge, fluv, flih, fli2]["UniqID"] = UniqID

for parameters in max_depth:
    [hden, temp, flge, fluv, flih, fli2] = parameters
    depth = max_depth[parameters]["depth"]
    UniqID = max_depth[parameters]["UniqID"]
    if myconfig.debug == False:
        create_cloudy_input_file(UniqID, depth, hden, temp, [flge, fluv, flih, fli2])
    if myconfig.debug == True:
        print(UniqID, depth, hden, temp, [flge, fluv, flih, fli2])

#with open('max_depth.pickle', 'wb') as handle:
#    pickle.dump(max_depth, handle, protocol=pickle.HIGHEST_PROTOCOL)
