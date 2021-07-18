"""Compress the emp cloudy files"""
import glob
import gasspy_cloudy_compression_lib as occl
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import multiprocessing
import gc
import numpy as np

#Undo for debugging in vs code.
#multiprocessing.set_start_method('spawn', True)

def main(dirs, hdd_lock1, hdd_lock2, multiproc=1, max_files=False):
    """
    Create a processor pool and process each cloudy output directory
    """
    if multiproc > 1:
        with ProcessPoolExecutor(max_workers=multiproc) as executor:
            futures = [executor.submit(occl.worker_compress_cloudy_dir, data_dir, hdd_lock1, hdd_lock2, True) for data_dir in dirs]
            for future in as_completed(futures):
                gc.collect()
                if future.result() is not None:
                    print(future.result())
    else:
        for data_dir in dirs:
            occl.worker_compress_cloudy_dir(data_dir,hdd_lock1,hdd_lock2, True, max_files=max_files)

if __name__ == "__main__":
    import sys, getopt

    root_dir = "cloudy-output"
    outfile = "gasspy_filelist.pckl"

    max_core_count = 8

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('gasspy_cloudy_compress.py -i <inputdirectory> -o <output pckl>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('opate_cloudy_compress.py -i <inputdirectory> -o <output pckl>.pckl')
            print('Note on input seach: by default we search 1 level for sub-directories called group*. If we find them, we use those as the search path.')
            sys.exit()
        elif opt in ("-i", "--inputdir"):
            root_dir = arg
        elif opt in ("-o", "--ofile"):
            outfile = arg

    multiprocessing.set_start_method('spawn', True)
    m = multiprocessing.Manager()

    # try:
    #     m = multiprocessing.Manager()
    # except:
    #     multiprocessing.set_start_method('spawn', True)
    #     m = multiprocessing.Manager()
    
    hdd_lock1 = m.Lock()
    hdd_lock2 = m.Lock()

    import pickle
    reuse_filelist = False
    if reuse_filelist:
        try:
            infile = open("gasspy_filelist.pckl", "rb")
            data_dirs = pickle.load(infile)
            infile.close()
        except:
            """
            To prevent directories from becoming too large, cloudy runs can be grouped into sub-directories.
            We will look for sub-directories in the provided project location. If we find none we then use
            the root directory to search for and compress models. 
            We support one level of grouping at this time.
            """
            data_dirs = glob.glob(root_dir+"//group*//", recursive=True)
            if len(data_dirs) == 0:
                data_dirs = [root_dir]
            pickle.dump(data_dirs, open(outfile, "wb"))

    else:
        data_dirs = glob.glob(root_dir+"//group*//", recursive=True)
        if len(data_dirs) == 0:
            data_dirs = [root_dir]
        pickle.dump(data_dirs, open(outfile, "wb"))

    multi_proc_count = np.min([max_core_count, np.max([1, len(data_dirs)]) ])

    main(data_dirs, hdd_lock1, hdd_lock2, multiproc=1, max_files=False)
