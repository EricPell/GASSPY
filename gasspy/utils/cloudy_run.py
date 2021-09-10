#!/usr/bin/python3
import os, sys, shutil
import time
import argparse
import random
from multiprocessing import Process, Value, Array, Lock, Pool, current_process
import subprocess

import glob, pathlib
import multiprocessing
import time, timeit

from pathlib import Path

class processor_class(object):
    """
    docstring
    """
    def __init__(self, command_line):

        parser = argparse.ArgumentParser(prog=command_line[0])
        
        parser.add_argument('--Ncores', metavar='Ncores', type=int,
            default=len(os.sched_getaffinity(0)) ,
            help="Specify number of CPU cores (default: all cores)")

        parser.add_argument('--cloudy_path', metavar='Cloudy path', type=str, default=None, help="Define path to cloudy executable. Use environ 'CLOUDY_EXE' by default")
        parser.add_argument('--indirs', type=str, nargs='+', help='an integer for the accumulator')
        parser.add_argument('--log', action='store_true')

        self.args = parser.parse_args(command_line[1:])

        if self.args.cloudy_path == None:
            try:
                self.args.cloudy_path = os.environ["CLOUDY_PATH"]
            except:
                sys.exit("No Cloudy executable was defined, or found in the environmental variable CLOUDY_PATH")

        for indir in self.args.indirs:         
            if Path(indir).exists == False:
                sys.exit("Supplied path %s not found"%(indir))


    def preproc(self,starttime):
        exec_list = []
        for indir in self.args.indirs:
            l = glob.glob("%s/*.in"%(indir))
            l = [l[i].split("/")[-1] for i in range(len(l))]
            for cloudy_in in l:
                exec_list.append((indir, cloudy_in, self.args.cloudy_path, starttime))
        return(exec_list)

    def run_proc(self, exec_arg):
        import glob, subprocess
        indir, input, cloudy_path, starttime = exec_arg

        current_p = current_process()
        #print('process counter:', current_p._identity[0], 'pid:', os.getpid())

        cloudy_sub_path = cloudy_path

        cloudy_exe = cloudy_sub_path+"/source/cloudy.exe"
        os.environ["CLOUDY_DATA_PATH"] = cloudy_sub_path+"/data"

        base_dir = os.getcwd()
        os.chdir(indir)

        #"""find inputs each time, just in case we are iterationsing over multiple different directories."""
        #inputs = glob.glob("*.in")
        #for input in inputs:
        out = open(input.replace(".in",".out"), "w")

        with subprocess.Popen([cloudy_exe, input], stdout=subprocess.PIPE) as p:
            pass
            out.write(p.stdout.read().decode("utf-8"))
        out.close()

        os.chdir(base_dir)
        if not self.args.log:
            print("Elapsed time: %0.2f (s) : %s/%s"%(time.time()-starttime, indir,input))

    def pool_handler(self, exec_list):
        p = Pool(min(len(exec_list), self.args.Ncores))
        p.map(self.run_proc, exec_list)


if __name__ == "__main__":
    starttime = time.time()
    processor = processor_class(sys.argv)
    exec_list = processor.preproc(starttime)
    processor.pool_handler(exec_list)
    endtime = time.time()

    if not processor.args.log:
        print("Total threads used: %i"%min(processor.args.Ncores, len(exec_list)))
        print("Total models calculated: %i"%len(exec_list))
        print("Total time  (s): %0.2f"%(endtime-starttime))
        print("Models / second : %0.4f"%( len(exec_list)/float(endtime-starttime)) )

    else:
        log_file = "cloudy_run.log"
        if os.path.exists(log_file):
            out = open(log_file,"a+")
            print("Append log...")
        else:
            print("New log...")
            out = open(log_file,"w+")
            out.writelines("\t".join(["N_cores","N_mods", "time(s)","mod/s\n"]))
        out.writelines("%i\t%i\t%0.2f\t%0.4f"%(min(processor.args.Ncores, len(exec_list)), len(exec_list),endtime-starttime, len(exec_list)/float(endtime-starttime)))

    try:
        if processor.args.single:
            print("")
            print("Single Threaded")
            starttime = time.time()
            processor.args.Ncores = 1
            exec_list = processor.preproc(starttime)
            processor.pool_handler(exec_list)
            endtime = time.time()
            print("Real Total time  (s): %0.2f"%(endtime-starttime))
    except:
        pass
    

