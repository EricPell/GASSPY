from numpy.core.shape_base import vstack
from gasspy.utils.simulation_data_lib import simulation_data_class
from gasspy.utils.observer import observer_plane_class
from gasspy.utils import raytracer
import gasspy.utils.save_to_fits
import numpy as np
import cProfile, pstats

sim_data = simulation_data_class(datadir="/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/")
rotangles = sim_data.config_yaml["rotang"]


rotangles = np.array([[0,0,0],
                     [0,0,0],
                     [0,0,0],
                     [45,45,0],
                     [45,45,0],
                     [45,45,0]])
profiler = cProfile.Profile()
profiler.enable()
for irot, rot in enumerate(rotangles):
    pitch, yaw, roll = rot
    obsplane = observer_plane_class(sim_data, pitch = pitch, yaw = yaw, roll = roll, z_subsamples=5)

    raytracer.traceRays(obsplane=obsplane, sim_data=sim_data, dZslab = 64, savefiles = False)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('ncalls')
stats.dump_stats("raytrace_profile")