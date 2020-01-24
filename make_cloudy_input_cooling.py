#!/usr/bin/python
""" Routine to make minimum models required for post processing """
#lastmoddate = 18.07.2016.EWP

import os
import sys
import pickle
import compress
import numpy as np
sys.path.append(os.getcwd())

###############################################################
# Set default parameters that can be overwritten by my config #
###############################################################
# Sets CLOUDY_modelIF to True
CLOUDY_modelIF = False
###############################################################
#                                                             #
###############################################################
import defaults
import myconfig

try:
    flux_type = myconfig.flux_type
except:
    flux_type = defaults.flux_type
    if flux_type is "default":
        sys.exit("I can not proceed without knowing the type of radiation bands used in the simulation")

try:
    compression_ratio = myconfig.compression_ratio
except:
    compression_ratio = defaults.compression_ratio

try:
    CLOUDY_INIT_FILE = myconfig.CLOUDY_INIT_FILE
except:
    CLOUDY_INIT_FILE = defaults.CLOUDY_INIT_FILE

#"""Decide to force every model to be calculated with full depth, or default to a single zone"""
try:
    """Try and read ForceFullDepth from myconfig"""
    ForceFullDepth = myconfig.ForceFullDepth
except:
    """Else ForceFullDepth is defined in defaults as true, set the global to true"""
    ForceFullDepth = defaults.ForceFullDepth

try:
    debug = myconfig.debug
except:
    debug = defaults.debug

MaxNumberModels = int(1e5)

# Import string containing each continuum shape.
if flux_type is 'fervent':
    import fervent_bands # Import continuum shapes
elif flux_type is 'Hion_excessE':
    import Hion_excessE_bands
elif flux_type is 'fsb99':
    """nothing to import"""
    
else:
    sys.exit("You have selected a flux type I do not understand. Flux Type = %s"%(flux_type))


def set_output_and_save_prefix(UniqID):
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
        outfile.write("set nend 5000\n")

    if model_is_ionization_front is False:
        #Set constant temperature if IF does not exist
        outfile.write("set nend 1\n")

def set_temperature(outfile, temperature, is_ionization_front):
    """Set constant temperature if not modeling the actual ionization front temp gradients"""
    if is_ionization_front is False:
        outfile.write("constant temperature %s\n"%(temperature))

def set_I_ge(outfile,I_ge):
    if I_ge != "-99.000":
        outfile.write(fervent_bands.flge)
        outfile.write("intensity %s, range 0.41 to 0.823 Ryd\n"%(I_ge))

def set_phi_uv(outfile,phi_uv):
    if phi_uv != "-99.000":
        outfile.write(fervent_bands.fluv)
        outfile.write("phi(h) = %s, range 0.823 to 1.0 Ryd\n"%(phi_uv))

def set_phi_ih(outfile,phi_ih):
    if phi_ih != "-99.000":
    #if phi_ih > 0:
        outfile.write(fervent_bands.flih)
        outfile.write("phi(h) = %s, range 1.0 to 1.117 Ryd\n"%(phi_ih))

def set_phi_i2(outfile,phi_i2):
    if phi_i2 != "-99.000":
    #if phi_i2 > 0:
        outfile.write(fervent_bands.fli2)
        outfile.write("phi(h) = %s, range 1.117 to 3 Ryd\n"%(phi_i2))

def set_fsb99_phi_ih(outfile,phi_fsb99):
    # "table star \"%s.mod\" age=%0.1f years \n" % (SB99model, np.max([SB99_age, i.SB99_age_min])))
    SB99model = "1e6cluster_norot_Z0014_BH120"
    outfile.write("table star \"%s.mod\" age=%0.1e years \n" % (SB99model, 1e5))
    outfile.write("phi(h) = %s, range 1.0 to 3 Ryd\n"%(phi_fsb99))

def set_Hion_excessE_phi_ih(outfile,I_ih):
    if I_ih != "-99.000":
    #if phi_ih > 0:
        outfile.write(Hion_excessE_bands.flih)
        outfile.write("intensity = %s, range 1.0 to 3.0 Ryd\n"%(I_ih))

def create_cloudy_input_file(_UniqID, _depth, _hden, _T, flux_array, flux_type, _cloudy_init_file=CLOUDY_INIT_FILE):
    """ create prefix for models and open Cloudy input file for writing"""
    cloudy_input_file = set_output_and_save_prefix(_UniqID)
    if flux_type is "fervent":
        """4 band fervent"""
        (_I_ge, _phi_uv, _phi_ih, _phi_i2) = flux_array

    elif flux_type is "Hion_excessE":
        """1 band simple ionizing SED"""
        _phi_ih = flux_array[0]
        (_I_ge, _phi_uv, _phi_i2) = (np.nan, np.nan, np.nan)
    
    elif flux_type is "fsb99":
        _phi_fsb99 = flux_array[0]
        (_phi_ih, _I_ge, _phi_uv, _phi_i2) = (np.nan, np.nan, np.nan, np.nan)


    # CLOUDY_modelIF is set to True by default. Can be changed in parameter file to false,
    # which will prevent isIF from executing

    #check if hden is log of mass density or volume density.
    if float(_hden) < -6.0:
        _hden = str(float(_hden) - np.log10(1.67e-24))

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
    
    if flux_type is "fervent":
        set_I_ge(cloudy_input_file, _I_ge)
        set_phi_uv(cloudy_input_file, _phi_uv)
        set_phi_ih(cloudy_input_file, _phi_ih)
        set_phi_i2(cloudy_input_file, _phi_i2)
    elif flux_type is "Hion_excessE":
        set_Hion_excessE_phi_ih(cloudy_input_file, _phi_ih)
    elif flux_type is "fsb99":
        set_fsb99_phi_ih(cloudy_input_file, _phi_fsb99)

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
    [UniqID, depth, hden, temp] = parameter_data[i].split("\t")[0:4]
    rad_fluxes = parameter_data[i].split("\t")[4:-1]
    NumberOfCellsLike = parameter_data[i].split("\t")[-1]

    # WARNING - Does depth need to be compressed? Ideally not.... I honestly can't see why it would need to be compressed.

    # WARNING - Experimental - WARNING 
    #hden = compress.number(float(hden), compression_ratio['hden'])
    #temp = compress.number(float(temp), compression_ratio['temp'])
    #flge = compress.number(float(flge), compression_ratio['flge'])
    #fluv = compress.number(float(fluv), compression_ratio['fluv'])
    #flih = compress.number(float(flih), compression_ratio['flih'])
    #fli2 = compress.number(float(fli2), compression_ratio['fli2'])

        #if database_type is str:
    if type(rad_fluxes[0]) is str:
        try:
            if float(depth) > max_depth[hden, temp, ",".join(rad_fluxes)]["depth"]:
                max_depth[hden, temp, ",".join(rad_fluxes)]["depth"] = float(depth)
                max_depth[hden, temp, ",".join(rad_fluxes)]["UniqID_of_maxDepth"] = UniqID
        except:
            max_depth[hden, temp, ",".join(rad_fluxes)] = {}
            max_depth[hden, temp, ",".join(rad_fluxes)]["depth"] = float(depth)
            max_depth[hden, temp, ",".join(rad_fluxes)]["UniqID_of_maxDepth"] = UniqID
    
    elif type(rad_fluxes[0]) is float:
        try:
            if float(depth) > max_depth[hden, temp, ",".join(rad_fluxes)]["depth"]:
                max_depth["%0.3f"%(hden), "%0.3f"%(temp), ",".join(rad_fluxes)]["depth"] = float(depth)
                max_depth["%0.3f"%(hden), "%0.3f"%(temp), ",".join(rad_fluxes)]["UniqID_of_maxDepth"] = UniqID
            else:
                UniquID_of_maxDepth =  max_depth[hden, temp, ",".join(rad_fluxes)]["UniqID_of_maxDepth"]
                #dict[UniquID_of_maxDepth].append(UniqID)
        except:
            max_depth["%0.3f"%(hden), "%0.3f"%(temp), ",".join(rad_fluxes)] = {}
            max_depth["%0.3f"%(hden), "%0.3f"%(temp), ",".join(rad_fluxes)]["depth"] = float(depth)
            max_depth["%0.3f"%(hden), "%0.3f"%(temp), ",".join(rad_fluxes)]["UniqID_of_maxDepth"] = UniqID
            #dict[UniquID_of_maxDepth].append(UniqID)

for parameters in max_depth:
    [hden, temp, rad_fluxes_string] = parameters
    rad_fluxes = rad_fluxes_string.split(",")
    depth = max_depth[hden, temp, rad_fluxes_string]["depth"]
    UniqID = max_depth[hden, temp, rad_fluxes_string]["UniqID_of_maxDepth"]
    if debug == False:
        create_cloudy_input_file(UniqID, depth, hden, temp, rad_fluxes, flux_type)
    if debug == True:
        print(UniqID, depth, hden, temp, rad_fluxes_string)

with open('max_depth.pickle', 'wb') as handle:
    pickle.dump(max_depth, handle, protocol=pickle.HIGHEST_PROTOCOL)