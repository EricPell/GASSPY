import numpy as np
from gasspy.io import gasspy_io
import pandas
import yaml
import sys

class subcell_model_class :
    def __init__(self, datadir, disableAutoloader=False):
        """ Load a subcell model from datadir"""
        self.datadir = datadir
        self.model_dict = gasspy_io.read_dict(datadir+"/gasspy_avg_em_pkl.npy")
        

    def DF_from_dict(self, labels, missing = 0.0):
        df = pandas.DataFrame()
        for label in labels:
            junk = np.full(len(self.model_dict[label]),missing, dtype="float32")
            for key_i, key in enumerate(self.model_dict[label]):
                junk[key_i] = self.model_dict[label][key]
            df[label] = junk
        return(df)

    def get(self, labels,ranges = None):
        """like data[labels].iloc[ranges]"""
        if ranges is None:
            #return self.DF_from_dict(self.model_dict[labels])          
            return self.DF_from_dict(labels)

        else:
            return self.DF_from_dict(self.model_dict[labels]).iloc[ranges]

class simulation_data_class:
    def __init__(self, datadir,config_yaml=None):
        """ loads data of a simulation from datadir, including subcell models, 
            indices of the simulation cells corresponding to the subcell models and simulation dimensions
        """

        # Set the datadir of all models, data and config files
        self.datadir=datadir
    
        if config_yaml is None:
            """ Use a default file name, assumed to be in datadir"""
            config_yaml = "gasspy_config.yaml"
        with open(r'%s/%s'%(datadir,config_yaml)) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            self.config_yaml = yaml.load(file, Loader=yaml.FullLoader)

        try:
            origin = self.config_yaml["origin"]
            self.config_yaml["origin"] = np.array(origin,dtype="float")
        except:
            sys.exit("YOU FUCKED UP 2")
        
        self.__dict__.update(self.config_yaml)
    
    def get_subcell_model_id(self):
        if self.subcell_model_id is None:
            self.subcell_model_id = np.load(self.datadir+"gasspy_indices3d.npy")
        return self.subcell_model_id

    def get_index1D(self):
        fname = self.datadir + "/index1D.npy"
        return np.load(fname)
    def get_amr_lrefine(self):
        fname = self.datadir + "/amr_lrefine.npy"
        return np.load(fname)
