from astropy.io import fits
import yaml
import numpy as np
from gasspy.utils import savename

def run(simdata, obsplane, config_yaml=None, overwrite=True, saveprefix = None):
    if config_yaml is None:
        """ Use a default file name, assumed to be in datadir"""
        config_yaml = "gasspy_config.yaml"
    with open(r'%s/%s'%(simdata.datadir,config_yaml)) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config_yaml = yaml.load(file, Loader=yaml.FullLoader)

    for line_label in config_yaml["line_labels"]:
        name = savename.get_filename(line_label, simdata, obsplane, saveprefix=saveprefix)
        data = np.load("%s/gasspy_output/%s.npy"%(simdata.datadir,name))
        hdu = fits.PrimaryHDU(data)
        hdul = fits.HDUList([hdu])
        hdul.writeto("%s/gasspy_output/%s.fits"%(simdata.datadir,name), overwrite=overwrite)

