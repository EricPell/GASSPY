from opiate.utils.simulation_data_lib import simulation_data_class
from opiate.utils.observer import observer_plane_class
from opiate.utils import raytracer
import opiate.utils.save_to_fits

sim_data = simulation_data_class(datadir="/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/")
rotangles = sim_data.config_yaml["rotang"]

rotangles ={}
for i, pitch in enumerate(range(0,90,3)):
    rotangles[i]=[pitch,0,0]

for irot in rotangles:
    pitch, yaw, roll = rotangles[irot]
    obsplane = observer_plane_class(sim_data, pitch = pitch, yaw = yaw, roll = roll, z_subsamples=2)
    raytracer.traceRays(obsplane=obsplane, sim_data=sim_data, dZslab = 256)

    opiate.utils.save_to_fits.run(sim_data, obsplane)
