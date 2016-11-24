#!/usr/bin/python
lastmoddate = "18.07.2016.EWP"

import os
import sys
sys.path.append(os.getcwd())

###############################################################
# Set default parameters that can be overwritten by my config #
###############################################################
# Sets CLOUDY_modelIF to True
CLOUDY_modelIF = True
###############################################################
#                                                             #
###############################################################

from myconfig import * # read in mask parameters

MaxNumberModels = int(1e7)

# Import configuration file from this
import myconfig
try:
    CLOUDY_INIT_FILE = myconfig.CLOUDY_INIT_FILE
except:
    CLOUDY_INIT_FILE = defaults.CLOUDY_INIT_FILE
    
# Import string containing each continuum shape.
import fervent_bands # Import continuum shapes

def set_output_and_save_prefix(UniqID, depth, hden, T, I_ge, phi_uv, phi_ih, phi_i2):
    try:
        os.stat("./cloudy-output")
    except:
        os.mkdir("./cloudy-output")
        
    # pass this a dictionary of parameters and values. Loop over and create prefix. Use dictionary elsewhere.
    prefix = "./cloudy-output/silcc-%s"%UniqID  #_dx_%s_hden_%s_T_%s_flge_%s_fluv_%s_flih_%s_fli2_%s"%(UniqID, depth, hden, T, I_ge, phi_uv, phi_ih, phi_i2)
    outfile = open(prefix+".in",'w')
    outfile.write("set save prefix \"%s\""%(prefix)+"\n")
    return(outfile)

def check_for_IF(depth,hden,phi_ih,phi_i2):
    alpha = 4.0e-13 # H recombinations per second cm-6
    ion_depth = (10**(float(phi_ih)) + 10**(float(phi_i2)))/(alpha * 10**(float(hden))**2 )
    # Change hardcoded 1e13 to mean free path of ionizing photon in H0.
    if( (ion_depth <= 10**float(depth)) and (ion_depth > 1e13) ):
        return True
    else:
        return False

def set_cloudy_init_file(outfile, init_file):
    outfile.write("init \"%s\"\n"%(init_file))
                  
def set_depth(outfile,depth):
    outfile.write("stop depth %s\n"%(depth))    

def set_hden(outfile, hden):
    outfile.write("hden %s\n"%(hden))

def set_nend(outfile,isIF):
    
    try:
        if myConfig.ForceDepth == True:
            isIF = True
    except:
        "force depth no set"            
        
    if isIF == True:
        """ Do not set constant temperature if IF exists """
        outfile.write("set nend 1000\n")        
    if isIF == False:
        """ Set constant temperature if IF does not exist """
        outfile.write("set nend 1\n")        

def set_T(outfile,T,isIF):
    if(isIF == False):
        outfile.write("constant temperature %s\n"%(T))

def set_I_ge(outfile,I_ge):
    if(I_ge != "-99.0"):
        outfile.write(fervent_bands.flge)
        outfile.write("intensity %s, range 0.41 to 0.823 Ryd\n"%(I_ge))

def set_phi_uv(outfile,phi_uv):
    if(phi_uv != "-99.0"):
        outfile.write(fervent_bands.fluv)
        outfile.write("phi(h) = %s, range 0.823 to 1.0 Ryd\n"%(phi_uv))

def set_phi_ih(outfile,phi_ih):
    if(phi_ih != "-99.0"):
        outfile.write(fervent_bands.flih)
        outfile.write("phi(h) = %s, range 1.0 to 1.117 Ryd\n"%(phi_ih))

def set_phi_i2(outfile,phi_i2):
    if(phi_i2 != "-99.0"):
        outfile.write(fervent_bands.fli2)
        outfile.write("phi(h) = %s, range 1.117 to 3 Ryd\n"%(phi_i2))

def create_cloudy_input_file(UniqID, depth, hden, T, flux_array, cloudy_init_file=CLOUDY_INIT_FILE):
    (I_ge, phi_uv, phi_ih, phi_i2) = flux_array
    """ crate prefix for models and open Cloudy input file for writing"""
    cloudy_input_file = set_output_and_save_prefix(UniqID, hden, depth, T, I_ge, phi_uv, phi_ih, phi_i2)

    # CLOUDY_modelIF is set to True by default. Can be changed in parameter file to false, which will prevent isIF from executing
    if(CLOUDY_modelIF):
        isIF = check_for_IF(depth,hden,phi_ih,phi_i2)
    else:
        isIF = False


    """ Set common init file """
    set_cloudy_init_file(cloudy_input_file, cloudy_init_file)

    """ Write individual cloudy parameters to input file """
    set_depth(cloudy_input_file, depth)
    set_hden(cloudy_input_file, hden)
    set_nend(cloudy_input_file, isIF)
    set_T(cloudy_input_file, T, isIF)
    set_I_ge(cloudy_input_file, I_ge)
    set_phi_uv(cloudy_input_file, phi_uv)
    set_phi_ih(cloudy_input_file, phi_ih)
    set_phi_i2(cloudy_input_file, phi_i2)

    """ Close input file """
    cloudy_input_file.close()


""" Begin main part of code """
input = open(myconfig.opiate_lookup,'r')
parameter_data = input.readlines()

#if(parameter_data[0] != "UniqID  dx      dens    temp    flge    fluv    flih    fli2    N\n"):
    # sys.exit("Header file did not match expected format")
#else:

if(len(parameter_data) < MaxNumberModels):
    for i in range(1,len(parameter_data)):
        [UniqID, depth, hden, temp, flge, fluv, flih, fli2, NumberOfCellsLike] = parameter_data[i].split("\t")
        create_cloudy_input_file(UniqID, depth, hden, temp, [flge, fluv, flih, fli2])
        
