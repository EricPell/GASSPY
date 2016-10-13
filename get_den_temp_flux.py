#!/usr/bin/python
lastmoddate = "24.05.2016.EWP"

""" Append launch directory to python path for importing config files """
import os
import sys
sys.path.append(os.getcwd())

""" Import default and model specific settings """ 
import defaults
import myconfig
try:
    mask_parameters_dict = myconfig.mask_parameters_dict
except:
    mask_parameters_dict = defaults.mask_parameters_dict
    

""" Load dependencies """
import yt
import numpy as np


""" Define function to write messages to stdout """
def live_line(str):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(str)

 
ds = yt.load(myconfig.inFile)
dd = ds.all_data()

outFile = open("tmp"+".cloudyparameters",'w')

unique_param_dict={}

def get_turb_velocity(x,y,z,dx):
    # identify all cells within a dx of the current cell. if neighboring cells have higher refinement could return more than 6
    # The code solves motion only along cardinal axis. so only worry about cells that can actually send mass into this cell.
    my neighbor_mask = ( abs(dd["x"] - x) <= dx  ) and (abs(dd["y"] - y) <= dx) and (abs(dd["z"]-z) <= dx))
    
    # Am I doing this with heating or turbulence. I would prefer turbulence. Perhaps there should be an option.
    # Or if turbulence is only a heating term (it's not) I could use the same cloudy formula for heating and calculate it myself
    # from the existing velocity field. But turbulence provides an increased collisional rate. 

    #. # calculate relative velocity to all neighboring cells

    #. # Identify which cells are moving in access of the sound speed TOWARD the current cell in it's referece frame

    #. # The energy due to compressive heating is already accounted for in the current temperature of the gas. 
    
    #. # Calculate momentum flux into current

    #. #   
""" Set masks based on mask parameters read in by the defaults library, or by myconfig"""
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

mask = mask_data(mask_parameters_dict)

dxxyz = ["dx","x","y","z"]
gasfields = ["dens","temp","iha ","ihp ","ih2 ","ico ","icp "]
# gas mass density, temperature, fraction of atomic H (iha), ionized (ihp) and molecular (ih2),
# and various gas fractions.

radfields = ["flge","fluv","flih","fli2"]
# Radiation fields: Should possibly be defined based on code type, i.e. FLASH, RAMSES

cloudyfields = ["dx","dens","temp"] + radfields

outstr = "cell_i"
for field in dxxyz:
    outstr +="\t%*s"%(9,field)
for field in gasfields+radfields:
    outstr +="\t%*s"%(4,field)

outFile.write(outstr)
outFile.write("\n")

Ncells = len( dd['dens'][mask])

# Extract masked cells into arrays
simdata={}
for field in dxxyz:
    simdata[field] = dd[field][mask].value

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
#Loop over every cell in the masked region
for cell_i in range(Ncells):
    #initialize the data values array
    data=[]

    dx = simdata["dx"][cell_i]
    
    cloudyparm = "%0.3f\t"%(np.log10(dx))

    #extract dxxyz positions
    for field in dxxyz:
        data.append("%0.3e"%(simdata[field][cell_i]))

    #extract gas properties field
    for field in gasfields:
        value = "%0.1f"%(simdata[field][cell_i])
        if value == "-inf" or value == "inf":
            value = "%0.1f"%(np.log10(1e-99))
        try:
            cloudyfields.index(field)
            cloudyparm +="%s\t"%(value)
        except:
            "field not a cloudy param"
        # Append the field numerical value to data
        data.append(value)

    #extract intensity radiation fields

    for field in radfields:
        logflux = simdata[field][cell_i]
        if logflux > -4:
            value = "%0.1f"%(logflux)
        else:
            value = "-99.0"
        if value == "-inf" or value == "inf":
            value = "%0.1f"%(np.log10(1e-99))
            # Append the field numerical value to data
        data.append(value)
        cloudyparm +="%s\t"%(value)
    
    # Write cell data to output file
    if data[-3:-1]+[data[-1]] != ["-99.0","-99.0","-99.0"]:
        try:
            unique_param_dict[cloudyparm]+=1
        except:
            unique_param_dict[cloudyparm] =1
    outFile.write("\t".join( [ "%*i" % (6,cell_i) ] + data ) + "\n")

    #Print progress to stdout
    # Only print every 1% cells.
    if float(cell_i)/(float(Ncells)/100.) == cell_i/(Ncells/100) :
        message = "Extracting cell %i:%i (%i percent complete)"%(cell_i,Ncells-cell_i,(int(100.*float(cell_i)/float(Ncells))))
        live_line(message)

#Cloes output file
outFile.close()

sys.stdout.write("\n")
live_line("Finished %i cells"%(Ncells)+"\n")

outFile = open("tmp"+".unique_parameters",'w')
outFile.write("\t".join(["UniqID"]+cloudyfields)+"\tN\n")

uniqueID = 0
for key in sorted(unique_param_dict.keys()):
    outFile.write("%i"%uniqueID+"\t"+key+"%i"%unique_param_dict[key]+"\n")
    uniqueID += 1
outFile.close()
