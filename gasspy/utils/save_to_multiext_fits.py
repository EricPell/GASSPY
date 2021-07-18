from astropy.io import fits
import yaml
import numpy as np
from gasspy.utils import savename
import glob
import os

def run(search_string, savesfile, overwrite=True):
    file_list = sorted(glob.glob(search_string))
    file_list.sort(key=os.path.getmtime)

    data = np.load(file_list[0])
    hdu_list = [fits.PrimaryHDU(data)] 

    for file in file_list[1:]:
        data = np.load(file)
        hdu_list.append(fits.ImageHDU(data))

    hdul = fits.HDUList(hdu_list)
    hdul.writeto(savesfile, overwrite=overwrite)

if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2])
