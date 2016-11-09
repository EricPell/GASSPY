#!/usr/bin/python
import yt
import numpy as np
from astropy.table import Table
import os
import sys

sys.path.append(os.getcwd())

outfile = open("fluxes.data",'w') 

""" Import default and model specific settings """ 
import defaults
import myconfig
try:
    mask_parameters_dict = myconfig.mask_parameters_dict
except:
    mask_parameters_dict = defaults.mask_parameters_dict

# raw_input("Now load ds from file")

ds = yt.load(myconfig.inFile)
dd = ds.all_data()

# raw_input("Now load combined ems table from file")
try:
    em_table = Table.read(myconfig.opiate_library,format='ascii')
except:
    raise Exception("A problem occured loading the OPIATE library")
em_table.sort("ID")

# raw_input("Now ready unique parameters table")
try:
    unique_table = Table.read(myconfig.opiate_lookup,format='ascii')
except:
    raise Exception("Problem loading unique parameters lookup table")

def mask_data(mask_parameters):
    # raw_input("Mask creation routine was called")
    masks = {}
    n_mask=0
    for key in sorted(mask_parameters.keys()):
        debugstring = "Current mask key/value is ",key,mask_parameters[key]
        # raw_input(debugstring)
        if (mask_parameters[key] != "default"):
            n_mask+=1
            masks[key+"min"] = dd[key] > min(mask_parameters[key])
            masks[key+"max"] = dd[key] < max(mask_parameters[key])
            
            if n_mask != 1:
                mask = mask*masks[key+"min"]*masks[key+"max"]
            else:
                mask = masks[key+"min"]*masks[key+"max"]
        elif (key in {"x","y","z"}):
            mask_parameters_dict[key] = [ds.domain_left_edge[{"x":0,"y":1,"z":2}[key]].value,
                                         ds.domain_right_edge[{"x":0,"y":1,"z":2}[key]].value]
    if(n_mask == 0):
        print "data is not masked"
        mask = dd["density"] > 0
    return(mask) # http://www.imdb.com/title/tt0110475/

# raw_input("Now Create masks")
mask = mask_data(mask_parameters_dict)

Ncells = len(dd['dens'][mask])

dxxyz = ["dx","x","y","z"]
gasfields = ["dens","temp","iha ","ihp ","ih2 ","ico ","icp "]

radfields = ["flge","fluv","flih","fli2"]

cloudyfields = ["dx","dens","temp"] + radfields

# Extract masked cells into arrays
# raw_input("Now Mask each field")
simdata={}
for field in dxxyz:
    # raw_input(field)
    simdata[field] = dd[field][mask].value

# raw_input("Now Take log10 of dx array")
simdata['dx'] = np.log10(dd['dx'][mask].value)

# raw_input("Now take log 10 of all other arrays")
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

# raw_input("Now loop over each entry in unique_table and create a dictionary")
unique_dict = {}
for row in range(len(unique_table)):
    (UniqID, dx, dens, temp, flge, fluv, flih, fli2,N) = unique_table[row]
    unique_dict["%0.3f"%dx, "%0.1f"%dens, "%0.1f"%temp, "%0.1f"%flge, "%0.1f"%fluv, "%0.1f"%flih, "%0.1f"%fli2]= unique_table[row]['UniqID']

#step one, look up ID (which is the row)
try:
    line_labels = myconfig.line_labels
except:
    line_labels = defaults.line_labels

def emissivity(line_label,dx,dens,temp,flge,fluv,flih,fli2):
    try:
        id = unique_dict["%0.3f"%dx, "%0.1f"%dens, "%0.1f"%temp, flge, fluv, flih, fli2]
    except:
        #ID not contained in model. Could look for nearby models, or just return 0.0
        id = -1
        return(-99.9)
    try:
        return(em_table[id][line_label])
    except:
        raise Exception("Known ID did not return data from library")

def get_rad_field(field,cell_i,mask=[]):
    if(len(mask) == 0):
        logflux = simdata[field][cell_i]
    else:
        logflux = simdata[field][mask][cell_i]
    if logflux > -4:
        value = "%0.1f"%(logflux)
    else:
        value = "-99.0"
    if value == "-inf" or value == "inf":
        value = "%0.1f"%(np.log10(1e-99))
        # Append the field numerical value to data
    return(value)

emissivity_dict = {}

luminosity = {}
volume = {}
xmax = max(mask_parameters_dict["x"])
xmin = min(mask_parameters_dict["x"])
ymax = max(mask_parameters_dict["y"])
ymin = min(mask_parameters_dict["y"])
dz = max(mask_parameters_dict["z"]) - min(mask_parameters_dict["z"])

""" Minumum number of resolution elements to divide an axis into """
min_frb_N = 10

""" Set length of projection bin on the sky """
def projection_scale(xmin,xmax,ymin,ymax,min_frb_N):
    return (min( ((xmax-xmin)/min_frb_N),((ymax-ymin)/min_frb_N) ))

dl  = projection_scale(xmin,xmax,ymin,ymax,min_frb_N)

outfile.write("\t".join(["x","y"]+line_labels))

# print "xmin = %e"%xmin
# print "xmax = %e"%xmax
# print "ymin = %e"%ymin
# print "ymax = %e"%ymax
# print "min_frb_N = %e"%min_frb_N
# print "dl = %e"%dl
ixmax = int((xmax-xmin)/dl)
iymax = int((ymax-ymin)/dl)
#print "ix_max = %i"%ixmax

for ix in range(0,ixmax):
    x = xmin + dl*ix
    xmask = abs(simdata["x"] - x) < dl
    luminosity[x]={}
    volume[x] = {}
    for iy in range(0,iymax):
        y = ymin + dl*iy
        sys.stderr.write('%i %i\n'%(ix,iy))
        ymask = abs(simdata['y'] - y) < dl
        luminosity[x][y] = {}
        for line_label in line_labels:
            luminosity[x][y][line_label] = 0.0
        volume[x][y]=0.0
        xymask = xmask*ymask
        
        for cell_i in range(len(simdata['x'][xymask])):
            emissivity_dict[cell_i]={}
            dx = simdata["dx"][xymask][cell_i]
            dens = simdata["dens"][xymask][cell_i]
            temp = simdata["temp"][xymask][cell_i]
            
            flge = get_rad_field("flge",cell_i,xymask)
            fluv = get_rad_field("fluv",cell_i,xymask)
            flih = get_rad_field("flih",cell_i,xymask)
            fli2 = get_rad_field("fli2",cell_i,xymask)
            
            if ([flge,fluv,flih,fli2] != ['-99.0', '-99.0', '-99.0', '-99.0']):
                for line_label in line_labels:
                    #print "%0.3e %0.3e %0.3e"%(simdata["x"][cell_i],simdata["y"][cell_i],simdata["z"][cell_i]),"%0.3f"%dx, "%0.1f"%dens, "%0.1f"%temp, flge, fluv, flih, fli2
                    emissivity_dict[cell_i][line_label] = float(emissivity(line_label,dx,dens,temp,flge,fluv,flih,fli2))

                    luminosity[x][y][line_label] += 10**(emissivity_dict[cell_i][line_label] + 3*dx)
            volume[x][y] += 10**(3*dx)
        outstr = "\t".join(["%0.3e"%x,"%0.3e"%y])
        for line_label in line_labels:
            flux = 10**(np.log10(luminosity[x][y][line_label]) - 2*np.log10(dl))
            outstr =  "\t".join([outstr, "%0.1e"%(flux)])
        outfile.write(outstr)
        outfile.write("\n")
    #print "\n"
outfile.close()

    
