from numpy.core.shape_base import vstack
from opiate.utils.simulation_data_lib import simulation_data_class
from opiate.utils.observer import observer_plane_class
from opiate.utils import raytracer
import opiate.utils.save_to_fits
import numpy as np

sim_data = simulation_data_class(datadir="/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/")
rotangles = sim_data.config_yaml["rotang"]

rotangles ={}
Nframes = 90
pitch = np.linspace(0,720,Nframes)
#yaw =  np.zeros(Nframes) 
#yaw[len(pitch)//4:] = np.linspace(0,180,len(pitch) - len(pitch)//4)[:]
yaw = np.logspace(-2,0,Nframes//2)*180
yaw = np.append(yaw,yaw[::-1])

roll = np.zeros(Nframes)

rotangles = np.vstack([pitch, yaw, roll]).T
print(rotangles.shape)

for irot, rot in enumerate(rotangles):
    prefix="%06i"%irot
    pitch, yaw, roll = rot
    obsplane = observer_plane_class(sim_data, pitch = pitch, yaw = yaw, roll = roll, z_subsamples=2)

    raytracer.traceRays(obsplane=obsplane, sim_data=sim_data, dZslab = 200, saveprefix=prefix)

    opiate.utils.save_to_fits.run(sim_data, obsplane, saveprefix=prefix)
