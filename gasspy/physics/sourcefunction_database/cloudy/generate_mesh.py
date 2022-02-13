import os
import shutil
import numpy as np
from numpy.core.fromnumeric import sort
import subprocess

Ryd_Ang = 911.2266

class mesh_generator(object):
    def __init__(self,auto_compile=True, nproc = 1, cloudy_data_dir = None):
        self.auto_compile=auto_compile
        self.nproc = nproc
        if cloudy_data_dir is None :
            self.cloudy_data_dir = os.environ["CLOUDY_DATA_PATH"]
        else:
            self.cloudy_data_dir = cloudy_data_dir
        
        return

    def compile_cloudy(self):
        '''
            Compile cloudy, along with its tables for dust and stars assuming that the spectral bins have been changed
        '''
        # Save the original working directory
        work_dir = os.getcwd()
        
        # Move to the source directory and recompile cloudy
        os.chdir(self.cloudy_data_dir+"../source")
        subprocess.run(["make", "clean"])
        subprocess.run(["make", "-j", "%d" % self.nproc])
        
        # Now move to the data directory and tell cloudy to recompile grains with the new spectral bins
        os.chdir(self.cloudy_data_dir) 

        p1 = subprocess.Popen(["echo", "compile grains"], stdout=subprocess.PIPE)
        p2 = subprocess.run(["../source/cloudy.exe"], stdin=p1.stdout)
        p1 = subprocess.Popen(["echo", "compile stars"], stdout=subprocess.PIPE)
        p2 = subprocess.run(["../source/cloudy.exe"], stdin=p1.stdout)


        # For safety move back to the original working directory
        os.chdir(work_dir)
        return 

    def regrid(self, E0, delta, R):
        #E0 = [900, 0.0]
        R0 = [300, 33.33333]

        assert "CLOUDY_DATA_PATH" in os.environ, "CLOUDY_DATA_PATH is not defined in the environment"
        assert os.path.isdir(self.cloudy_data_dir), "Cloudy is not installed at %s"%(self.cloudy_data_dir.strip("/data/"))

        assert np.issubdtype(type(R), np.number), "Error: To speed up velocity shifting in radtran, multiple resolving powers are not currently supported"

        current = self.cloudy_data_dir+"/continuum_mesh.ini"
        backup = current + "_backup"

        if os.path.isfile(backup):
            print ("File %s exists"%backup)
            pass
        else:
            shutil.copy(current, backup)
            print("Copied file")

        f = open(backup, 'r+')
        lines = []
        for line in f.readlines():
            if line[0] != "#":
                lines.append(line)
        f.close()
        
        f = open(current, "w")
        f.writelines(lines[0])
    
        #E0 = 912. / np.array(lam)
        E0 = np.array(E0)
        #E0 = E0[sorted_index]
        delta = np.array(delta)

        E_l = E0 - delta
        E_r = E0 + delta

        sorted_index = np.argsort(E_r)
        E0  = E0 [sorted_index]
        E_l = E_l[sorted_index]
        E_r = E_r[sorted_index]

        if not np.issubdtype(type(R), np.number):
            R = R[sorted_index]
        else:
            R = np.full_like(E0, R)    
        
        merged = False 
        while not merged:
            mat_diff = -E_r[:, np.newaxis] + E_l[np.newaxis, :]
            
            has_merged = np.zeros(E_r.shape)
            E_l_new = []
            E_r_new = []
            R_new   = []

            merged = True
            for i in range(len(E_r)):
                if i < len(E_r) - 1:
                    to_merge = np.arange(i+1, len(E_r))[np.where(mat_diff[i,i+1:] < 0)[0]]
                else:
                    to_merge = []
                if len(to_merge) == 0:
                    if not has_merged[i]:
                        E_l_new.append(E_l[i])
                        E_r_new.append(E_r[i])
                        R_new.append(R[i])
                    continue
                E_l_new.append(min(E_l[i], np.min(E_l[to_merge])))
                E_r_new.append(max(E_r[i], E_r[to_merge[-1]]))
                R_new.append(max(R[i], np.max(R[to_merge])))
                has_merged[to_merge] = True
                merged = False

            # Update right handed (high) energy limits with the new merged upper limits
            E_r = np.array(E_r_new)
            # Get arguments which sort the Er upper list
            sorted_index = np.argsort(E_r)

            # Sort the new lists
            E_l = np.array(E_l_new)[sorted_index]
            E_r = E_r[sorted_index]
            R = np.array(R_new)[sorted_index]
            print(E_r)

        idx_edge = np.where((E_r > 900) * (E_l < 900))[0]
        if len(idx_edge) == 0:
            E_low_res_lim = 900
            wrote900 = False
        else:
            E_low_res_lim = E_r[idx_edge]*1.00001
            wrote900 = True

        for i in range(len(E_r)):
            if E_r[i] < E_low_res_lim:
                R0i = 0
            else:
                R0i = 1
                if not wrote900:
                    f.writelines("%e     300" % (E_low_res_lim))
                    wrote900 = True

            f.writelines("%0.6e %i\n"%(E_l[i]*0.99999,R0[R0i]))
            f.writelines("%0.6e %i\n"%(E_l[i],R[i]))
            f.writelines("%0.6e %i\n"%(E_r[i],R[i]))
            f.writelines("%0.6e %i\n"%(E_r[i]*1.00001,R0[R0i]))

        if wrote900 == False:
            f.writelines("%e     300\n" % (E_low_res_lim))
        
        f.writelines("0        33.333333\n")
        f.close()
        np.savetxt(current.replace(".ini", "_windows.txt"), np.vstack((E_l, E_r)).T)
        if self.auto_compile:
            self.compile_cloudy()


if __name__ == "__main__":
    from gasspy.physics.sourcefunction_database.cloudy import select_cloudy_lines 
    labels = select_cloudy_lines.labels()
    label_list = [label.split(" ")[-1] for label in list(labels.line.keys())]

    E0 = np.zeros(len(label_list))

    for i, label in enumerate(label_list):
        if label.endswith("A"):
            E0[i] = Ryd_Ang / float(label.strip("A"))
        elif label.endswith("m"):
            E0[i] = Ryd_Ang / (float(label.strip("m"))*1e4)

    delta = E0 * 1000.0/3e5
    R = 10000

    generator = mesh_generator(nproc = 16) 
    generator.regrid(E0, delta, R)

