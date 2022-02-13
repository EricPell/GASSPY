

from numpy.lib.npyio import save


def get_filename(line, sim_data, obs_plane, saveprefix = None):
    if saveprefix is not None :
        # first remove all stray spaces and replace with _, removing duplicates
        filename = saveprefix+"_"+line.replace(" ","_").replace("__","_")
    else:
        filename = line.replace(" ","_").replace("__","_")
    #add rotation angles to 3 decimal precision 
    filename += "_x0_%07.3f_%07.3f_%07.3f_pitch_%07.3f_yaw_%07.3f_roll_%07.3f"%(obs_plane.xp0_s, obs_plane.yp0_s, obs_plane.zp0_s, 
                                                                           obs_plane.pitch, obs_plane.yaw, obs_plane.roll)
    return filename