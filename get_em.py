#!/usr/bin/python
import yt
import numpy as np
from astropy.table import Table
import os
import sys

sys.path.append(os.getcwd())

import myconfig # read in mask parameters

ds = yt.load(myconfig.inFile)
dd = ds.all_data()

try:
    em_table = Table.read(myconfig.opiate_library,format='ascii')
except:
    raise Exception("A problem loading the OPIATE library")
em_table.sort("ID")

try:
    unique_table = Table.read(myconfig.opiate_lookup,format='ascii')
except:
    raise Exception("Problem loading parameters lookup table")


def mask_data(mask_parameters):
    masks = {}
    n_mask=0
    for key in sorted(mask_parameters.keys()):
        if mask_parameters[key] != "default":
            n_mask+=1
            masks[key+"min"] = dd[key] > min(mask_parameters[key])
            masks[key+"max"] = dd[key] < max(mask_parameters[key])
            
            if n_mask != 1:
                mask = mask*masks[key+"min"]*masks[key+"max"]
            else:
                mask = masks[key+"min"]*masks[key+"max"]
                
    if(n_mask == 0):
        print "data is not masked"
        mask = dd["density"] > 0
    return(mask) # http://www.imdb.com/title/tt0110475/

mask = mask_data(myconfig.mask_parameters_dict)

Ncells = len(dd['dens'][mask])

dxxyz = ["dx","x","y","z"]
gasfields = ["dens","temp","iha ","ihp ","ih2 ","ico ","icp "]

radfields = ["flge","fluv","flih","fli2"]

cloudyfields = ["dx","dens","temp"] + radfields

# Extract masked cells into arrays
simdata={}
for field in dxxyz:
    simdata[field] = dd[field][mask].value

simdata['dx'] = np.log10(dd['dx'][mask].value)

for field in gasfields:
    if field == "dens":
        mH = 1.67e-24 # Mass of the hydrogen atom
        simdata[field] = np.log10(dd[field][mask].value/mH)
    else:
        simdata[field] = np.log10(dd[field][mask].value)
        
for field in radfields:    
    if field == "flge":
        simdata[field] = np.log10(dd[field][mask].value)
    else:
        simdata[field] = dd[field][mask].value-2.0*np.log10(dd['dx'][mask].value)
        tolowmask = simdata[field] < 0.0
        simdata[field][tolowmask] = -99.0

unique_dict = {}
for row in range(len(unique_table)):
    try:
        (UniqID, dx, dens, temp, flge, fluv, flih, fli2,N) = unique_table[row]
    except:
        raise Exception("No matching ID for dx,den,temp and radiation parameter space")

    try:
        unique_dict["%0.3f"%dx, "%0.1f"%dens, "%0.1f"%temp, "%0.1f"%flge, "%0.1f"%fluv, "%0.1f"%flih, "%0.1f"%fli2]= unique_table[row]['UniqID']
    except:
        raise Exception("No matching model for UniqueID %s"%UniqID)

#step one, look up ID (which is the row)
line_label = "O  3  5007A"

def emissivity(line_label,dx,dens,temp,flge,fluv,flih,fli2):
    try:
        id = unique_dict["%0.3f"%dx, "%0.1f"%dens, "%0.1f"%temp, flge, fluv, flih, fli2]
    except:
        #ID not contained in model. Could look for nearby models, or just return 0.0
        id = -1
        return(0.0)
    try:
        return(em_table[id][line_label])
    except:
        raise Exception("Known ID did not return data from library")
    
def get_rad_field(field,cell_i):
    logflux = simdata[field][cell_i]
    if logflux > -4:
        value = "%0.1f"%(logflux)
    else:
        value = "-99.0"
    if value == "-inf" or value == "inf":
        value = "%0.1f"%(np.log10(1e-99))
        # Append the field numerical value to data
    return(value)

emissivity_dict = {}
for cell_i in range(Ncells):
    emissivity_dict[cell_i]={}
    dx = simdata["dx"][cell_i]
    dens = simdata["dens"][cell_i]
    temp = simdata["temp"][cell_i]

    flge = get_rad_field("flge",cell_i)
    fluv = get_rad_field("fluv",cell_i)
    flih = get_rad_field("flih",cell_i)
    fli2 = get_rad_field("fli2",cell_i)

    if ([flge,fluv,flih,fli2] != ['-99.0', '-99.0', '-99.0', '-99.0']):
        #print "%0.3e %0.3e %0.3e"%(simdata["x"][cell_i],simdata["y"][cell_i],simdata["z"][cell_i]),"%0.3f"%dx, "%0.1f"%dens, "%0.1f"%temp, flge, fluv, flih, fli2
        emissivity_dict[cell_i][line_label] = float(emissivity(line_label,dx,dens,temp,flge,fluv,flih,fli2))
    
        print "%0.3e %0.3e %0.3e %s\t"%(simdata["x"][cell_i],simdata["y"][cell_i],simdata["z"][cell_i],emissivity_dict[cell_i][line_label])
    
    
