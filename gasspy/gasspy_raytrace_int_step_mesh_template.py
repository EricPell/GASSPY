from numpy.core.shape_base import vstack
from gasspy.utils.simulation_data_lib import simulation_data_class
from gasspy.utils.observer_new import observer_plane_class
from gasspy.utils import raytracer_new, moveable_detector
import gasspy.utils.save_to_fits
import numpy as np
import sys

if len(sys.argv) > 1:
    sim_data = simulation_data_class(datadir=sys.argv[1])
else:
    sim_data = simulation_data_class(datadir="/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/")
rotangles = sim_data.config_yaml["rotang"]

# Read the ray substep type, and select it from the config file.
ray_substep_param = {"type":sim_data.config_yaml["ray_substep"]["type"]}
ray_substep_param.update(sim_data.config_yaml["ray_substep"][ray_substep_param["type"]])

Nframes = rotangles["Nframes"][0]
pitch = np.linspace(rotangles['pitch'][0],rotangles['pitch'][1],Nframes)
yaw   = np.linspace(rotangles["yaw"][0],rotangles["yaw"][1], Nframes) 
roll  = np.linspace(rotangles["roll"][0],rotangles["roll"][1], Nframes) 

rotangles = np.vstack([pitch, yaw, roll]).T

for irot, rot in enumerate(rotangles):
    prefix="%06i"%irot
    pitch, yaw, roll = rot
    obsplane = observer_plane_class(sim_data, pitch = pitch, yaw = yaw, roll = roll, substep_parameters=ray_substep_param)
    # det = moveable_detector.movable_detector()

    raytracer_new.traceRays_by_slab_step(obsplane=obsplane, sim_data=sim_data, saveprefix=prefix)

    gasspy.utils.save_to_fits.run(sim_data, obsplane, saveprefix=prefix)
