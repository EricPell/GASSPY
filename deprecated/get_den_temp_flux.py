#!/usr/bin/python
""" Append launch directory to python path for importing config files """
import os
import sys
import compress
sys.path.append(os.getcwd())

LastModDate = "24.05.2016.EWP"

""" Import default and model specific settings """
import defaults
import myconfig

try:
    append_db = myconfig.append_db
except:
    append_db = defaults.append_db

try:
    mask_parameters_dict = myconfig.mask_parameters_dict
except:
    mask_parameters_dict = defaults.mask_parameters_dict

try:
    flux_type = myconfig.flux_type
except:
    flux_type = defaults.flux_type

try:
    compression_ratio = myconfig.compression_ratio
except:
    compression_ratio = defaults.compression_ratio

try: 
    log10_flux_low_limit = myconfig.log10_flux_low_limit
except: 
    log10_flux_low_limit = defaults.log10_flux_low_limit

""" Load dependencies """
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from yt.config import ytcfg; ytcfg["yt", "__withinreason"] = "True"
import yt
import numpy as np

def live_line(stringout):
    """ Define function to write messages to stdout """
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(stringout)


ds = yt.load(myconfig.inFile)
dd = ds.all_data()

try:
    debug = myconfig.debug
except:
    debug = defaults.debug

if debug  == True:
    outFile = open("tmp"+".cloudyparameters", 'w')

unique_param_dict = {}

def mask_data(mask_dict):
    """ Set masks based on mask parameters read in by the defaults library, or by myconfig"""
    masks = {}
    n_mask = 0
    """ Create a mask which contains all cells, since there is no such thing as negative density"""
    full_mask = dd["density"] < 0
    for mask_name in mask_dict.keys():
        """For each defined mask in mask_parameters_dictionary"""
        n_mask += 1

        partial_mask = dd["density"] > 0
        mask_parameters = mask_dict[mask_name]
        for parameter in sorted(mask_parameters.keys()):
            if mask_dict[mask_name][parameter] != "default":
                masks[parameter+"min"] = dd[parameter] > min(mask_dict[mask_name][parameter])
                masks[parameter+"max"] = dd[parameter] < max(mask_dict[mask_name][parameter])

                partial_mask = partial_mask*masks[parameter+"min"]*masks[parameter+"max"]
        
        full_mask = full_mask + partial_mask

    if n_mask == 0:
        print("data is not masked")
    del(masks, partial_mask)
    return full_mask # http://www.imdb.com/title/tt0110475/

mask = mask_data(mask_parameters_dict)

dxxyz = ["dx", "x", "y", "z"]
gasfields = ["dens", "temp", "iha ", "ihp ", "ih2 ", "ico ", "icp "]
# gas mass density, temperature, fraction of atomic H (iha), ionized (ihp) and molecular (ih2),
# and various gas fractions.

# Radiation fields: Should possibly be defined based on code type, i.e. FLASH, RAMSES
try:
    radfields = myconfig.radfields
except:
    radfields = defaults.radfields

cloudyfields = ["dx", "dens", "temp"] + radfields

outstr = "cell_i"
for field in dxxyz:
    outstr += "\t%*s"%(9, field)
for field in gasfields+radfields:
    outstr += "\t%*s"%(4, field)

if debug  is True:
    outFile.write(outstr)
    outFile.write("\n")

Ncells = len(dd['dens'][mask])

# Extract masked cells into arrays
simdata = {}
for field in dxxyz:
    simdata[field] = dd[field][mask].value

for field in gasfields:
    if field == "dens":
        mH = 1.67e-24 # Mass of the hydrogen atom
        simdata[field] = np.log10(dd[field][mask].value/mH)
    else:
        simdata[field] = np.log10(dd[field][mask].value)

for field in radfields:
    if flux_type is "fervent":
        if field == "flge":
            simdata[field] = np.log10(dd[field][mask].value)
        else:
            simdata[field] = dd[field][mask].value-2.0*np.log10(dd['dx'][mask].value)
            tolowmask = simdata[field] < 0.0
            simdata[field][tolowmask] = -99.00

    if flux_type is "Hion_excessE":
            simdata[field] = np.log10(dd[field][mask].value*2.99792e10) # Hion_excessE is an energy density. U*c is flux 
            #to_low_value =  -np.log10(2.1790E-11)*1000 # energy flux of one ionizing photon == 13.6eV \times 1000 photons per cm-2 which is 100x less than the ISRF. See ApJ 2002, 570, 697
            #tolowmask = simdata[field] < to_low_value
            #simdata[field][tolowmask] = -99.00
        
        
#Loop over every cell in the masked region
for cell_i in range(Ncells):
    #initialize the data values array
    data = []

    dx = simdata["dx"][cell_i]

    cloudyparm = "%0.3f\t"%(np.log10(dx))

    #extract dxxyz positions
    for field in dxxyz:
        data.append("%0.3e"%(simdata[field][cell_i]))

    #extract gas properties field
    for field in gasfields:
        try:
            value = "%0.3f"%(compress.number(simdata[field][cell_i], compression_ratio[field]))
        except:
            value = "%0.3f"%(simdata[field][cell_i])
        if value == "-inf" or value == "inf":
            value = "%0.3f"%(np.log10(1e-99))
        try:
            cloudyfields.index(field)
            cloudyparm += "%s\t"%(value)
        except:
            "field not a cloudy param"
        # Append the field numerical value to data
        data.append(value)

    #extract intensity radiation fields

    """ Fervent Radiation cleaning step to deal with low and fully shielded cells"""
   
    for field in radfields:
        logflux = simdata[field][cell_i]
        if logflux > log10_flux_low_limit[field]:
            """ Do we have atleast 1 photon per cm-2?"""
            value = "%0.3f"%compress.number(float(logflux), compression_ratio[field])
        else:
            value = "-99.000"
        if value == "-inf" or value == "inf":
            value = "%0.3f"%(np.log10(1e-99))
        # Append the field numerical value to data
        data.append(value)
        cloudyparm += "%s\t"%(value)

    if (flux_type is "fervent") and (data[-3:-1]+[data[-1]] != ["-99.000", "-99.000", "-99.000"]):
        # Write cell data to output file
        try:
            unique_param_dict[cloudyparm] += 1
        except:
            unique_param_dict[cloudyparm] = 1

    if flux_type is "Hion_excessE" and data[-1] != "-99.000":
        # Write cell data to output file
        try:
            unique_param_dict[cloudyparm] += 1
        except:
            unique_param_dict[cloudyparm] = 1

    if debug  is True:
        outFile.write("\t".join(["%*i"%(6, cell_i)] + data) + "\n")


    #Print progress to stdout
    # Only print every 1% cells.
    if float(cell_i)/(float(Ncells)/100.) == cell_i/(Ncells/100):
        message = "Extracting cell %i:%i (%i percent complete)"%(cell_i, Ncells-cell_i, (int(100.*float(cell_i)/float(Ncells))))
        live_line(message)

if debug  is True:
    #Close output file
    outFile.close()

""" We have now finished looping over all cells """
sys.stdout.write("\n")
live_line("Finished %i cells"%(Ncells)+"\n")

""" Check if we are creating or appending to a data base. Set write mode and initialize uniqueID accordingly."""
if append_db is True:
    if not os.path.exists(myconfig.opiate_lookup):
        sys.stderr.write('You have selected to append to an existing database %s, but I could not find the file. I will make a new database.'%myconfig.opiate_lookup)
        append_db = False
    else:
        uniqueID0 = 0
        outFile = open(myconfig.opiate_lookup, 'a')
               
if append_db is False:
    """ If we are not appending to an existing database set the uniqueIDs to 0 and open the database for writing"""
    uniqueID0 = 0        
    outFile = open(myconfig.opiate_lookup, 'w')
    """ Add a header row to the database """
    outFile.write("\t".join(["UniqID"]+cloudyfields)+"\tN\n")

uniqueID = uniqueID0

""" Loop over each unique, compressed, cloudy input parameters, and print them with a uniqueID"""
for i, key in enumerate(sorted(unique_param_dict.keys())):
    outFile.write("%i"%(uniqueID0+i)+"\t"+key+"%i"%unique_param_dict[key]+"\n")
outFile.close()