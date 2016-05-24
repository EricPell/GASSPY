#!/usr/bin/python
lastmoddate = "24.05.2016.EWP"

import os
import sys
sys.path.append(os.getcwd())

# Import string containing each continuum shape.
from flge_shape import *
from fluv_shape import *
from flih_shape import *
from fli2_shape import *

def set_output_and_save_prefix(UniqID, depth, hden, T, I_ge, phi_uv, phi_ih, phi_i2):
    # pass this a dictionary of parameters and values. Loop over and create prefix. Use dictionary elsewhere.
    prefix = "silcc-%s"%UniqID  #_dx_%s_hden_%s_T_%s_flge_%s_fluv_%s_flih_%s_fli2_%s"%(UniqID, depth, hden, T, I_ge, phi_uv, phi_ih, phi_i2)
    outfile = open(prefix+".in",'w')
    outfile.write("set save prefix \"%s\""%(prefix)+"\n")
    return(outfile)

def set_cloudy_init_file(outfile, init_file):
    outfile.write("init \"%s\"\n"%(init_file))
                  
def set_depth(outfile,depth):
    outfile.write("stop depth %s\n"%(depth))

def set_hden(outfile, hden):
    outfile.write("hden %s\n"%(hden))

def set_T(outfile,T):
    outfile.write("constant temperature %s\n"%(T))

def set_I_ge(outfile,I_ge):
    if(I_ge != "-99.0"):
        outfile.write(flge_shape)
        outfile.write("intensity %s, range 0.41 to 0.823 Ryd\n"%(I_ge))

def set_phi_uv(outfile,phi_uv):
    if(phi_uv != "-99.0"):
        outfile.write(fluv_shape)
        outfile.write("phi(h) = %s, range 0.823 to 1.0 Ryd\n"%(phi_uv))

def set_phi_ih(outfile,phi_ih):
    if(phi_ih != "-99.0"):
        outfile.write(flih_shape)
        outfile.write("phi(h) = %s, range 1.0 to 1.117 Ryd\n"%(phi_ih))

def set_phi_i2(outfile,phi_i2):
    if(phi_i2 != "-99.0"):
        outfile.write(fli2_shape)
        outfile.write("phi(h) = %s, range 1.117 to 3 Ryd\n"%(phi_i2))


def create_cloudy_input_file(UniqID, depth, hden, T, I_ge, phi_uv, phi_ih, phi_i2, cloudy_init_file="silcc_flash_postprocess_singlezone.ini"):
    """ crate prefix for models and open Cloudy input file for writing"""
    cloudy_input_file = set_output_and_save_prefix(UniqID, hden, depth, T, I_ge, phi_uv, phi_ih, phi_i2)
    
    """ Set common init file """
    set_cloudy_init_file(cloudy_input_file, cloudy_init_file)

    """ Write individual cloudy parameters to input file """
    set_depth(cloudy_input_file, depth)
    set_hden(cloudy_input_file, hden)
    set_T(cloudy_input_file, T)
    set_I_ge(cloudy_input_file, I_ge)
    set_phi_uv(cloudy_input_file, phi_uv)
    set_phi_ih(cloudy_input_file, phi_ih)
    set_phi_i2(cloudy_input_file, phi_i2)

    """ Close input file """
    cloudy_input_file.close()


""" Begin main part of code """
input = open("tmp.unique_parameters",'r')
parameter_data = input.readlines()

#if(parameter_data[0] != "UniqID  dx      dens    temp    flge    fluv    flih    fli2    N\n"):
    # sys.exit("Header file did not match expected format")
#else:
for i in range(1,len(parameter_data)):
        [UniqID, depth, hden, temp, flge, fluv, flih, fli2, NumberOfCellsLike] = parameter_data[i].split("\t")
        create_cloudy_input_file(UniqID, depth, hden, temp, flge, fluv, flih, fli2)
