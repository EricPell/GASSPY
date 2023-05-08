"""
    AUTHORS:  Loke Ohlin
    DATE: 13/04-2023
    PURPOSE:
        DatabaseGenerator class:
            This class takes simulation inputs and creates a database file with the informaton of all required models.
            This can then be loaded and populated with intensity and opacity by the DatabasePopulator class
    USAGE:
        See test example in __name__ == "__main__"
"""


import os
import sys
import numpy as np
import shutil
import h5py as hp

from gasspy.shared_utils import compress, loop_progress
import gasspy, pathlib
from gasspy.io.gasspy_io import check_parameter_in_config, save_gasspy_config_hdf5, read_yaml
from gasspy.shared_utils.mpi_utils.mpi_print import mpi_all_print, mpi_print

LastModDate = "2023.04.12.LO"

class DatabaseGenerator(object):
    def __init__(self, 
        gasspy_config,
        database_name = None,
        database_fields = None,
        gasspy_modeldir = None, 
        compression_ratio = None, 
        log10_field_limits = None,
        **kwargs):

        if isinstance(gasspy_config, str):
            self.gasspy_config = read_yaml(gasspy_config)
        else:
            self.gasspy_config = gasspy_config

        # If we are using interpolation we need neighbor data
        if "interpolate_fields" in self.gasspy_config:
            self.need_neighbors = True
            self.interpolate_fields = self.gasspy_config["interpolate_fields"]
        else:
            self.need_neighbors = False


        # Fields 
        self.database_fields = check_parameter_in_config(self.gasspy_config, "database_fields",database_fields, None)
        if self.database_fields is None:
            sys.exit("Error: No database_fields specified")
        self.__default_compression_ratio__ = (1,5.0)
    
        self.compression_ratio = check_parameter_in_config(self.gasspy_config, "compression_ratio", compression_ratio, None)
        if self.database_fields is None:
            sys.exit("Error: No compression_ratio specified")
        self.log10_field_limits = check_parameter_in_config(self.gasspy_config, "log10_field_limits", log10_field_limits, {})

        ##
        #  Name and path of hdf5 database 
        ## 

        # Name of the database which we are using
        self.database_name = check_parameter_in_config(self.gasspy_config, "database_name", database_name, "gasspy_database.hdf5") 
        
        # Path to database directory
        self.gasspy_modeldir = check_parameter_in_config(self.gasspy_config, "gasspy_modeldir", gasspy_modeldir, "gasspy_modeldir") 

        if not self.gasspy_modeldir.endswith("/"):
            self.gasspy_modeldir = self.gasspy_modeldir + "/"
        self.database_path = self.gasspy_modeldir+self.database_name


        if os.path.exists(self.database_path):
            mpi_print("Appending to database %s"%(self.database_path))
            self.append = True
            self.load_database()

        else:
            mpi_print("Creating new database %s"%(self.database_path))
            self.append = False
            self.open_database()
            self.model_data = None
            if self.need_neighbors:
                self.is_neighbor = None
                self.neighbor_ids = None
            self.N_unique = 0
        self.N_saved = 0

        self.__dict__.update(kwargs)
    
    def load_database(self):
        h5database = hp.File(self.gasspy_modeldir + self.database_name, "r+")
        # Load gasspy ids and models
        self.model_data = h5database["model_data"][:,:]

        # If we need neighbor data, make sure that its there. conversly if we do not but the previous database does, 
        # something has been missconfigured
        if "neighbor_ids" in h5database.keys():
            if not self.need_neighbors:
                sys.exit("ERROR: Old database is configured to use neighbor data, but new config does not. Check config options")
            self.neighbor_ids = h5database["neighbor_ids"][:,:]
            self.is_neighbor = h5database["is_neighbor"][:]
        else:
            if self.need_neighbors:
                sys.exit("ERROR: Old database is not configured to use neighbor data, but new config does. Check config options")

        self.N_unique = h5database["model_data"].shape[0]
        self.original_N_unique = self.N_unique
    
    def open_database(self):
        # Open database
        h5database = hp.File(self.gasspy_modeldir + self.database_name, "w")
        # Create all the neccesary datasets
        h5database.create_dataset("model_data", (1, len(self.database_fields)), maxshape=(None,len(self.database_fields)), dtype = float)
        if self.need_neighbors:
            h5database.create_dataset("neighbor_ids", (1, 3**len(self.interpolate_fields)), maxshape=(None,3**len(self.interpolate_fields)), dtype = int)
            h5database.create_dataset("is_neighbor", (1, ), maxshape=(None,), dtype = int)
        save_gasspy_config_hdf5(self.gasspy_config, h5database)
        h5database["database_fields"] = self.database_fields
        #close the database
        h5database.close()
    def save_database(self):
        """
            save the database
        """
        # Open database
        h5database =  hp.File(self.gasspy_modeldir + self.database_name, "r+")
        # If no new models, dont bother
        if self.N_unique == self.N_saved:
            return
        if self.need_neighbors: 
            keys = ["model_data", "neighbor_ids", "is_neighbor"]
        else:
            keys = ["model_data"]
        # Loop over all required fields ands save
        for key in keys:
            h5database[key].resize((self.N_unique), axis=0)
            h5database[key][self.N_saved:] = self.__dict__[key][self.N_saved:]
        
        # Update number of saved fields
        self.N_saved = self.N_unique

        # close the database
        h5database.close()

    def get_compressed_sim_data(self, sim_reader):
        """
            Loads each field, compres them and enforces limits
        """

        compressed_sim_data = None
        for ifield, field in enumerate(self.database_fields):
            # Load and compress the field (ensure double precision)
            compressed_field_data = compress.array(np.log10(sim_reader.get_field(field).astype(np.float64)), self.compression_ratio[field])
            # If field has lower or upper limits, set these
            if field in self.log10_field_limits:
                if "max" in self.log10_field_limits[field]:
                    maxval = self.log10_field_limits[field]["max"]
                    if "max_cutoff_value" in self.log10_field_limits[field]:
                        compressed_field_data[compressed_field_data > maxval] = self.log10_field_limits[field]["max_cutoff_value"]
                    else:
                        compressed_field_data[compressed_field_data > maxval] = maxval
                if "min" in self.log10_field_limits[field]:
                    minval = self.log10_field_limits[field]["min"]
                    if "min_cutoff_value" in self.log10_field_limits[field]:
                        compressed_field_data[compressed_field_data < minval] = self.log10_field_limits[field]["min_cutoff_value"]
                    else:
                        compressed_field_data[compressed_field_data < minval] = minval

            # If this is the first field, figure out the sizes
            if ifield == 0:
                compressed_sim_data = np.zeros((len(compressed_field_data), len(self.database_fields)))
            compressed_sim_data[:, ifield] = compressed_field_data 
            del compressed_field_data

        return compressed_sim_data

    def __shift_recursive__(self, ishift, ifield, sim_neighbors, non_neighbor_ids, total_shift = None):
        """
            Recursive function to find all neighbors
            For each interpolated field:
                1 Determine the shift based on the compressio_ratio
                2 Add to its entry of the shift
                3 recursive call to the next field
            Once all fields have added their shift: Add shift to original, progress neighbor count "ishift" by one
        """
        if ifield == len(self.interpolate_fields):
            # we are done, add shift and progress ishift
            if total_shift is None:
                sys.exit("Error in neighbor finding. total_shift was given as None")
            sim_neighbors[non_neighbor_ids + ishift, : ] += total_shift
            return ishift + 1 

        # If this is the first call, total_shift SHOULD be NONE, initialize it
        if total_shift is None:
            total_shift = np.zeros(len(self.database_fields))
      
        # Loop over left, centre and right, determine shift and call for the next field
        field = self.interpolate_fields[ifield]
        shift = 10**-self.compression_ratio[field][0] * self.compression_ratio[field][1]
        for local_ishift, local_shift in enumerate([0,-1,1]):
            total_shift[self.database_fields.index(field)] = local_shift* shift
            ishift = self.__shift_recursive__(ishift, ifield + 1, sim_neighbors, non_neighbor_ids, total_shift=total_shift)
        return ishift

    def add_neighbors(self, sim_model_data, sim_unique_ids):
        """
            Adds neighbors, checks for unique models again and returns a new list of id's, and neighbors
        """
        # Create space for neighbors. 3^nshift_field from left, centre and right
        sim_neighbors = np.repeat(sim_model_data, 3**len(self.interpolate_fields), axis = 0)
        
        # Indexes of the original models
        non_neighbor_ids = np.arange(sim_model_data.shape[0])*3**len(self.interpolate_fields)
        
        # Determine fields for all neighbors
        self.__shift_recursive__(0,0, sim_neighbors, non_neighbor_ids)

        # Ensure they are all compressed
        for ifield, field in enumerate(self.database_fields):
            sim_neighbors[:, ifield] = compress.array(sim_neighbors[:, ifield], self.compression_ratio[field])
              
        # Determine all unique entries (neighbors contains the original)
        uniques, all_ids = np.unique(sim_neighbors, axis = 0, return_inverse = True)

        # Update ids for the cells
        sim_unique_ids = all_ids[non_neighbor_ids[sim_unique_ids]]

        # Create an array that identifies the neighbors of a given model
        # Only gives neighbors for those that are ones specifically needed. otherwize set to -1
        sim_neighbor_ids = np.full((uniques.shape[0],3**len(self.interpolate_fields)), -1,dtype = int) 
        for ineigh in range(3**len(self.interpolate_fields)):
            sim_neighbor_ids[all_ids[non_neighbor_ids],ineigh] = all_ids[non_neighbor_ids + ineigh]
        
        # Add a list to easily determine which model is a neighbor and which are ones specifically need
        sim_is_neighbor = np.full(uniques.shape[0], 1, dtype=int)
        sim_is_neighbor[all_ids[non_neighbor_ids]] = 0
        
        # return all needed arrays
        return uniques, sim_unique_ids, sim_neighbor_ids, sim_is_neighbor

    def merge_uniques(self, new_model_data, new_unique_ids, new_neighbor_ids = None, new_is_neighbor = None):
        if self.model_data is None:
            self.model_data = new_model_data
            self.N_unique = self.model_data.shape[0]
            if self.need_neighbors:
                self.neighbor_ids = new_neighbor_ids
                self.is_neighbor = new_is_neighbor
            return new_unique_ids
        # Add to list of uniques
        appended_uniques = np.append(self.model_data, new_model_data, axis = 0)

        # determine all uniques
        all_uniques, all_ids = np.unique(appended_uniques, axis = 0, return_inverse = True)

        # np unique will havee scrambelled the old models. lets make sure these still match their old ids
        uniques = np.zeros(all_uniques.shape)

        # Make sure old ids are still in place
        uniques[:self.N_unique,:] = all_uniques[all_ids[:self.N_unique]]

        # Next figure out where the new_ids and models are
        # Next figure out which of the new ids are in the old list
        in_old = np.in1d(all_ids[self.N_unique:],all_ids[:self.N_unique])
        new_ids = np.zeros(new_model_data.shape[0], dtype = int)
        sort = all_ids[:self.N_unique].argsort()
        rank = np.searchsorted(all_ids[:self.N_unique], all_ids[self.N_unique:][in_old], sorter = sort)
        new_ids[in_old] = sort[rank]
        # Finaly add the new uniques in and match indexes
        newer_uniques = all_uniques[all_ids[self.N_unique:][~in_old],:]
        uniques[self.N_unique:,:] = newer_uniques
        tmp, ids = np.unique(all_ids[self.N_unique:][~in_old], return_inverse = True)

        new_ids[~in_old] = ids + self.N_unique

        # Set the new ids for the cells
        new_unique_ids[:] = new_ids[new_unique_ids]

        # If we need neighbors do the same matching for them
        if self.need_neighbors:
            new_neighbor_ids[:,:] = new_ids[new_neighbor_ids] 
            # Finally we need to update which models are neighbors, since a neighbor in new may not be in old and vice versa
            
            # Expand neighbor list and neighbor id list
            is_neighbor = np.full(uniques.shape[0], 1, dtype = int)
            is_neighbor[:self.N_unique] = self.is_neighbor
            neighbor_ids = np.full((len(uniques), 3**len(self.interpolate_fields)),-1,dtype = int)
            neighbor_ids[:self.N_unique,:] = self.neighbor_ids

            # set for all new models
            is_neighbor[new_ids] = new_is_neighbor & is_neighbor[new_ids]
            # Only set if the new one if its not an neighbor, to avoid overwrite
            neighbor_ids[new_ids[np.where(new_is_neighbor==0)],:] = new_neighbor_ids[np.where(new_is_neighbor==0),:]

            # Set globals
            self.is_neighbor = is_neighbor
            self.neighbor_ids = neighbor_ids

        # Update the models and return the id's of the new_model_data
        self.model_data = uniques
        self.N_unique = self.model_data.shape[0]

        return new_unique_ids

    def add_snapshot(self, sim_reader):
        """
            Adds a snapshot to the database, determening all required unique models
            Simulation_Reader : sim_reader (class that loads/returns fields from the simulation)
        """
        mpi_print("Compressing snapshot")
        compressed_simdata = self.get_compressed_sim_data(sim_reader)
        
        # Determine all new unique models
        sim_model_data, sim_unique_ids = np.unique(compressed_simdata, axis = 0, return_inverse = True)
        if self.need_neighbors:
            sim_model_data, sim_unique_ids, sim_neighbor_ids, sim_is_neighbor = self.add_neighbors(sim_model_data, sim_unique_ids)
        else:
            sim_is_neighbor = None
            sim_neighbor_ids = None

        # Save old_N_unique for book keeping
        old_N_unique = self.N_unique
        # If we do not have any previous data we can just set here
        sim_unique_ids = self.merge_uniques(sim_model_data, sim_unique_ids, new_neighbor_ids = sim_neighbor_ids, new_is_neighbor = sim_is_neighbor)

        mpi_print("Snapshot has %d unique models:"%(len(sim_model_data)))
        mpi_print("\t%d are new"%(self.N_unique- old_N_unique))
        if self.need_neighbors:
            mpi_print("\t%d new neighbors are required"%np.sum(self.is_neighbor[old_N_unique:]))
        mpi_print("")
        # Make snapshot remember its models
        sim_reader.save_new_field("cell_gasspy_ids", sim_unique_ids, dtype = int)
        
    def finalize(self):

        # Save database
        self.save_database()    

        # Print dataset information
        mpi_print("Dataset has %d unique models:"%(self.N_unique))
        if self.need_neighbors:
            mpi_print("\t%d are explicitly needed"%(np.sum(self.is_neighbor==0)))
            mpi_print("\t%d are only needed as neighbors"%(np.sum(self.is_neighbor==1)))
        mpi_print("")


####################################################################################
# TESTING
####################################################################################
"""
    testunit:
    1) Generate multiple simulation_readers with fields
    2) Add one
    3) Add the other
    4) check that it all makes sense
"""
if __name__ == "__main__":

    class simulation_reader:
        def __init__(self):
            self.fields = []
            return
        def set_field(self, field, data):
            """
                function to set a field
            """
            if field not in self.fields:
                self.fields.append(field)
            self.__dict__[field] = data

        def get_field(self, field):
            """
                function the get a field
            """
            return self.__dict__[field]
        
        def save_new_field(self, field, data, dtype = None):
            """
                Function to save a new field, used to save the gasspy_ids, here just store in the class
            """
            self.__dict__[field] = data
            return
        def check_snapshot(self, field, uniques, gasspy_config):
            
            compressed_field = compress.array(np.log10(self.__dict__[field]), gasspy_config["compression_ratio"][field])
            if field in gasspy_config["log10_field_limits"]:
                log10_field_limits =  gasspy_config["log10_field_limits"][field]
                if "max" in log10_field_limits:
                    maxval = log10_field_limits["max"]
                    if "max_cutoff_value" in log10_field_limits:
                        compressed_field[compressed_field > maxval] = log10_field_limits["max_cutoff_value"]
                    else:
                        compressed_field[compressed_field > maxval] = maxval
                if "min" in log10_field_limits:
                    minval = log10_field_limits["min"]
                    if "min_cutoff_value" in log10_field_limits:
                        compressed_field[compressed_field < minval] = log10_field_limits["min_cutoff_value"]
                    else:
                        compressed_field[compressed_field < minval] = minval
            diff = np.sum(compressed_field - uniques[self.cell_gasspy_ids])

            if diff == 0:
                print("\t%s matches"%field)
            else:
                print("\t%s does not match"%field)
                print(compressed_field)
                print(uniques[self.cell_gasspy_ids])


    # Simple gasspy config with needed parameters
    gasspy_config = {
        "database_name" : "test_database.hdf5",
        "gasspy_modeldir" : "./test_database/",
        "database_fields" :[ 
              "dens",
              "temp",
              "dx",
              "flux"
        ],
        "compression_ratio": {
                "dens"  :  [1, 1.0],
                "temp"  :  [1, 1.0],
                "dx"    :  [1, 1.0],
                "flux" :   [1, 1.0]
        },
        "log10_field_limits":{
                "temp" : {
                    "max": 7
                },
                "flux" : {
                    "min": -8,
                    "min_cutoff_value" : -99
                }
        },
        "interpolate_fields": [
            "dens",
            "temp"
        ]
    }

    # First snapshot: 4 cells, 2 identical, 1 exceeding temperature and below minimum of flux
    # start in logspace for clarity
    dens1 = np.array([3.05, 3.12, 3.05, 0.02])
    temp1 = np.array([4.05, 4.12, 4.05, 9.11])
    dx1   = np.array([17.01, 17.01, 17.01, 18.99])
    flux1 = np.array([13.11, 7.01, 13.11, -9.1])
    
    snapshot1 = simulation_reader()
    snapshot1.set_field("dens", 10**dens1)
    snapshot1.set_field("temp", 10**temp1)
    snapshot1.set_field("dx"  , 10**dx1)
    snapshot1.set_field("flux", 10**flux1)

    # Second snapshot: 2 identical to first, 2 new identical
    # start in logspace for clarity
    dens2 = np.array([3.12 , 3.13, 3.13,  3.05])
    temp2 = np.array([4.12 , 4.05, 4.05,  4.05])
    dx2   = np.array([17.01, 18.01, 18.01, 17.01])
    flux2 = np.array([7.01 , 0.12, 0.12,  13.11])

    snapshot2 = simulation_reader()
    snapshot2.set_field("dens", 10**dens2)
    snapshot2.set_field("temp", 10**temp2)
    snapshot2.set_field("dx"  , 10**dx2)
    snapshot2.set_field("flux", 10**flux2)

    # Third snapshot: 1 offset snapshot 2 by 0.1 in dens and temp and one random
    # start in logspace for clarity
    dens3 = np.array([3.12, 3.12,  3.14])
    temp3 = np.array([4.04, 4.04,  4.14])
    dx3   = np.array([18.1, 18.1,  17.01])
    flux3 = np.array([0.21, 0.21,  13.10])

    snapshot3 = simulation_reader()
    snapshot3.set_field("dens", 10**dens3)
    snapshot3.set_field("temp", 10**temp3)
    snapshot3.set_field("dx"  , 10**dx3)
    snapshot3.set_field("flux", 10**flux3)

    # fourth snapshot: 1 we've had before (Test if we have no new models that the appends still work)
    # start in logspace for clarity
    dens4 = np.array([3.14])
    temp4 = np.array([4.14])
    dx4   = np.array([17.01])
    flux4 = np.array([13.10])

    snapshot4 = simulation_reader()
    snapshot4.set_field("dens", 10**dens4)
    snapshot4.set_field("temp", 10**temp4)
    snapshot4.set_field("dx"  , 10**dx4)
    snapshot4.set_field("flux", 10**flux4)

    # fifth snapshot: 1 we've not had before (Test if we only have new models that the appends still work)
    # start in logspace for clarity
    dens5 = np.array([7.14])
    temp5 = np.array([4.14])
    dx5   = np.array([17.01])
    flux5 = np.array([13.10])

    snapshot5 = simulation_reader()
    snapshot5.set_field("dens", 10**dens5)
    snapshot5.set_field("temp", 10**temp5)
    snapshot5.set_field("dx"  , 10**dx5)
    snapshot5.set_field("flux", 10**flux5)

    if os.path.exists("test_database"):
        shutil.rmtree("test_database")
    
    os.makedirs("test_database")
    
    # First two in one go
    database_creator = DatabaseGenerator(gasspy_config)
    database_creator.add_snapshot(snapshot1)
    database_creator.add_snapshot(snapshot2)
    database_creator.finalize()
    del database_creator
    # Next try to load and append
    database_creator = DatabaseGenerator(gasspy_config)
    database_creator.add_snapshot(snapshot3)
    database_creator.add_snapshot(snapshot4)
    database_creator.add_snapshot(snapshot5)

    database_creator.finalize() 

    h5database = hp.File("test_database/test_database.hdf5", "r")
    print("Snapshot 1:")
    for ifield, field in enumerate(gasspy_config["database_fields"]):
        snapshot1.check_snapshot(field, h5database["model_data"][:,ifield], gasspy_config)
    print("Snapshot 2:")
    for ifield, field in enumerate(gasspy_config["database_fields"]):
        snapshot2.check_snapshot(field, h5database["model_data"][:,ifield], gasspy_config)
    print("Snapshot 3:")
    for ifield, field in enumerate(gasspy_config["database_fields"]):
        snapshot3.check_snapshot(field, h5database["model_data"][:,ifield], gasspy_config)  
    print("Snapshot 4:")
    for ifield, field in enumerate(gasspy_config["database_fields"]):
        snapshot3.check_snapshot(field, h5database["model_data"][:,ifield], gasspy_config)  
    print("Snapshot 5:")
    for ifield, field in enumerate(gasspy_config["database_fields"]):
        snapshot3.check_snapshot(field, h5database["model_data"][:,ifield], gasspy_config)  

    # Test if neighbor information makes sense
    is_neighbor = h5database["is_neighbor"][:]
    neighbor_ids = h5database["neighbor_ids"][:,:]


    # Determine all unique required ids, eg models that are specifically needed (and therefore have neighbor information)
    gasspy_ids_1 = snapshot1.cell_gasspy_ids
    gasspy_ids_2 = snapshot2.cell_gasspy_ids
    gasspy_ids_3 = snapshot3.cell_gasspy_ids
    gasspy_ids_4 = snapshot4.cell_gasspy_ids
    gasspy_ids_5 = snapshot5.cell_gasspy_ids
    
    # Append them and unique
    unique_gasspy_ids = None
    for gasspy_ids in [gasspy_ids_1, gasspy_ids_2, gasspy_ids_3, gasspy_ids_4, gasspy_ids_5] :
        if unique_gasspy_ids is None:
            unique_gasspy_ids = gasspy_ids.copy()
        else:
            unique_gasspy_ids = np.append(unique_gasspy_ids, gasspy_ids)
    unique_gasspy_ids = np.unique(unique_gasspy_ids)

    # Test if any of them are considered a neighbor only model (which lack neighbor information)
    if np.sum(is_neighbor[unique_gasspy_ids]) > 0:
        print("Error: Required models are neighbors!")

    # Test if they all have the required neighbor information
    if np.any(neighbor_ids[unique_gasspy_ids,:] == -1):
        print("Error: Some required models dont know all their neighbors" )

    # Ensure that all neighbors are unique 
    for gid in unique_gasspy_ids:
        uniq_neighs = np.unique(neighbor_ids[gid])
        if len(uniq_neighs) < neighbor_ids.shape[1]:
            print("Error: Some reguired models dont have all unique neighbors")

    print(h5database["database_fields"][:])
    shutil.rmtree("test_database")








