"""
    AUTHORS:  Loke Ohlin
    DATE: 13/04-2023
    PURPOSE:
        DatabasePopulator class:
            This class takes a skeleton database with all required models (generated by a DatabaseGenerator object) and used a ModelRunner class
            to run each model and save the resulting intensity and opacity
    USAGE:
        See test example in __name__ == "__main__"
"""


import sys
from mpi4py import MPI
import traceback

from gasspy.io.gasspy_io import check_parameter_in_config, read_yaml
from gasspy.shared_utils.mpi_utils.mpi_print import mpi_print, mpi_all_print
from gasspy.physics.databasing.database_generator import DatabaseGenerator
from gasspy.physics.databasing.database_populator import DatabasePopulator

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.Get_size()

class DatabaseCreator(object):
    def __init__(self, 
                 gasspy_config, 
                 model_runner,
                 database_fields = None,
                 gasspy_modeldir = None, 
                 compression_ratio = None, 
                 log10_field_limits = None,
                 populator_dump_time = None,
                 est_model_time = None,
                 max_walltime = None,
                 database_name = None,
                 h5database = None,
                 ) -> None:

        if isinstance(gasspy_config, str):
            self.gasspy_config = read_yaml(gasspy_config)
        else:
            self.gasspy_config = gasspy_config

        ##
        #  Name and path of hdf5 database 
        ## 

        # Name of the database which we are using
        self.database_name = check_parameter_in_config(self.gasspy_config, "database_name", database_name, "gasspy_database.hdf5") 
        
        # Path to database directory
        self.gasspy_modeldir = check_parameter_in_config(self.gasspy_config, "gasspy_modeldir", gasspy_modeldir, "gasspy_modeldir") 
        if not self.gasspy_modeldir.endswith("/"):
            self.gasspy_modeldir = self.gasspy_modeldir + "/"

        #####
        # Parameters for the generator
        #####
        # Fields 
        self.database_fields = check_parameter_in_config(self.gasspy_config, "database_fields",database_fields, None)
        if self.database_fields is None:
            mpi_print("Error: No database_fields specified", file = sys.stderr)
            mpi_comm.Abort(1)
        self.__default_compression_ratio__ = (1,5.0)
        # Compression ratio
        self.compression_ratio = check_parameter_in_config(self.gasspy_config, "compression_ratio", compression_ratio, None)
        if self.database_fields is None:
            mpi_print("Error: No compression_ratio specified", file = sys.stderr)
            mpi_comm.Abort(1)
        # Limits
        self.log10_field_limits = check_parameter_in_config(self.gasspy_config, "log10_field_limits", log10_field_limits, {})

        #####
        # Parameters for the populator
        #####
        # Set dump timing information
        self.populator_dump_time = check_parameter_in_config(self.gasspy_config, "populator_dump_time", populator_dump_time, 1800) # Default 30 minutes
        self.est_model_time = check_parameter_in_config(self.gasspy_config, "est_model_time", est_model_time, 10) # Estimate 10 seconds per model
        self.max_walltime = check_parameter_in_config(self.gasspy_config, "max_walltime", max_walltime, 1e99) # infinte
        
        # set model_runner
        self.model_runner = model_runner


        # Initialize the generator on the main rank only
        if mpi_rank == 0:
            try:
                self.database_generator = DatabaseGenerator(self.gasspy_config, 
                                                            database_name      = self.database_name,
                                                            database_fields    = self.database_fields,
                                                            gasspy_modeldir    = self.gasspy_modeldir,
                                                            compression_ratio  = self.compression_ratio,
                                                            log10_field_limits = self.log10_field_limits
                                                            )
            except :
                print(traceback.format_exc())
                mpi_comm.Abort(1)
            # get pointer to h5database
            self.h5database = self.database_generator.h5database
        else:
            self.h5database = None

        # Initialize the populator on all ranks
        self.database_populator = DatabasePopulator(self.gasspy_config, self.model_runner,
                                                    populator_dump_time = self.populator_dump_time,
                                                    est_model_time      = self.est_model_time,
                                                    max_walltime        = self.max_walltime,
                                                    gasspy_modeldir     = self.gasspy_modeldir,
                                                    database_name       = self.database_name,
                                                    h5database          = self.h5database
                                                    )
        
        return


    def add_snapshot(self, sim_reader):
        """
            Adds a snapshot to the database, determening all required unique models
            input:
            Simulation_Reader : sim_reader (class that loads/returns fields from the simulation)
        """
        # Only do this on the main rank
        if mpi_rank != 0:
            return    
        # Catch exceptions here to nicely exit the mpi environment
        try:
            # Give the snapshot to the generator to determine all uniques an neighbors if required
            self.database_generator.add_snapshot(sim_reader)

            # Tell the generator to save its findings to the database
            self.database_generator.save_database()
        except:
            mpi_print(traceback.format_exc())
            mpi_comm.Abort(1)            

    def run_models(self):
        """
            Runs the models that have not yet been run
        """
        self.database_populator.run_models()
    
    def finalize(self):
        """ 
            Ensures that everything has been saved properly and closes the hdf5 database file
        """
        if mpi_rank == 0 :
            # Catch exceptions here to nicely exit the mpi environment
            try:
                self.database_generator.finalize(close_hdf5 = False)
            except:
                mpi_print(traceback.format_exc())
                mpi_comm.Abort(1)                 
        
        self.database_populator.finalize(close_hdf5 = True)
        
        if mpi_rank == 0:
            # Catch exceptions here to nicely exit the mpi environment
            try:
                self.h5database.close()
            except:
                mpi_print(traceback.format_exc())
                mpi_comm.Abort(1)     



   