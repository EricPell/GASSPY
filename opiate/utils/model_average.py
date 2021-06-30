"""calculate the average emissivity and opacity of a cell"""
import glob
import os
import sys
import __model_average__

for file in glob.glob(sys.argv[1]+"*.ems"):
    __model_average__.average_emissivity(file)

for file in glob.glob(sys.argv[1]+"*.opt"):
    __model_average__.average_opacity(file)
