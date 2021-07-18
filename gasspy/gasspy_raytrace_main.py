from numpy.core.shape_base import vstack
from gasspy.utils.simulation_data_lib import simulation_data_class
from gasspy.utils.observer import observer_plane_class
from gasspy.utils import raytracer
import gasspy.utils.save_to_fits
import numpy as np

sim_data = simulation_data_class(datadir="/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/")
rotangles = sim_data.config_yaml["rotang"]

rotangles ={}
Nframes = 1
pitch = np.linspace(0,0,Nframes)
yaw =  np.full(Nframes, 45.0) 
#yaw[len(pitch)//4:] = np.linspace(0,180,len(pitch) - len(pitch)//4)[:]
# yaw = np.logspace(-2,0,Nframes//2)*180
# yaw = np.append(yaw,yaw[::-1])

roll = np.zeros(Nframes)

rotangles = np.vstack([pitch, yaw, roll]).T
print(rotangles.shape)

for irot, rot in enumerate(rotangles):
    prefix="%06i"%irot
    pitch, yaw, roll = rot
    obsplane = observer_plane_class(sim_data, pitch = pitch, yaw = yaw, roll = roll, z_subsamples=3)

    raytracer.traceRays(obsplane=obsplane, sim_data=sim_data, dZslab = 100, saveprefix=prefix)

    gasspy.utils.save_to_fits.run(sim_data, obsplane, saveprefix=prefix)
