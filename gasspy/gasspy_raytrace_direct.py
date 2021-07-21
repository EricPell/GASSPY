from numpy.core.shape_base import vstack
from gasspy.utils.simulation_data_lib import simulation_data_class
from gasspy.utils.observer import observer_plane_class
from gasspy.utils.raytracer_direct_grid import raytracer_class
import gasspy.utils.save_to_fits
import numpy as np
import cProfile 

datadir = "/home/loke/SanDiskSSD/Documents/PhD/GRASS_test/"
sim_data = simulation_data_class(datadir = datadir)
raytracer = raytracer_class(sim_data, savefiles = True)

Nframes = 10
pitch = np.linspace(0,90,Nframes)
yaw   = np.linspace(0,90,Nframes) 
#yaw[len(pitch)//4:] = np.linspace(0,180,len(pitch) - len(pitch)//4)[:]
# yaw = np.logspace(-2,0,Nframes//2)*180
# yaw = np.append(yaw,yaw[::-1])

roll = np.zeros(Nframes)

rotangles = np.vstack([pitch, yaw, roll]).T
print(rotangles.shape)
pr = cProfile.Profile()
pr.enable()
for irot, rot in enumerate(rotangles):
    prefix="%06i"%irot
    pitch, yaw, roll = rot
    print(prefix, pitch, yaw, roll)
    obsplane = observer_plane_class(sim_data, pitch = pitch, yaw = yaw, roll = roll, z_subsamples=3)
    
    raytracer.update_obsplane(obsplane)
    
    raytracer.raytrace_run(saveprefix = prefix)

    gasspy.utils.save_to_fits.run(sim_data, obsplane, saveprefix=prefix)
pr.disable()
pr.dump_stats('profile_rays')
