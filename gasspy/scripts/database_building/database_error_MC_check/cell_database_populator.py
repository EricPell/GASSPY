from gasspy.shared_utils.mpi_utils.mpi_print import mpi_all_print, mpi_print
from gasspy.physics.databasing.database_populator import DatabasePopulator
from mpi4py import MPI
import h5py as hp 
import numpy as np
import time 
import sys
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()
class CellDatabasePopulator(DatabasePopulator):
    def __init__(self, gasspy_config, model_runner,
                 populator_dump_time = None,
                 est_model_time = None,
                 max_walltime = None,
                 gasspy_modeldir = None,
                 database_name = None,
                 lines_only = None
                 ) -> None:
        super(CellDatabasePopulator, self).__init__(gasspy_config, model_runner, 
                       populator_dump_time = populator_dump_time,
                       est_model_time = est_model_time,
                       max_walltime = max_walltime,
                       gasspy_modeldir = gasspy_modeldir,
                       database_name = database_name,
                       lines_only = lines_only)
        self.save_spectra = False
        if self.save_lines:
            line_energies = self.model_runner.get_line_energies()
            energy = self.model_runner.get_energy_bins()
            
            # Determine energy indexes around the line to determine continuum level
            line_ienergies = np.searchsorted(energy, line_energies, side = "right")-1
            self.line_ienergies = np.repeat(line_ienergies, 11).reshape((line_ienergies.shape[0],11)) + np.arange(11) - 5
            self.denergy = self.model_runner.get_delta_energy_bins()[line_ienergies]
        
    def check_database(self):
        if mpi_rank != 0:
            return
        
        # open database
        h5database =  hp.File(self.database_path, "r+")

        # Load models
        self.model_data = h5database["model_data"][:,:]
        self.N_unique = self.model_data.shape[0]

        # open database
        h5database =  hp.File(self.database_path, "r+")

        # If we have populated this database before, check if we have added new models since 
        lines_in_database = "line_intensity" in h5database

        if lines_in_database:            
            # How big is the current database, do we need to extend it? 
            n_allocated = h5database["model_successful"].shape[0]
            if n_allocated != self.N_unique:
                # List of all fields that need to be extended
                keys = ["model_completed", "model_successful", 
                        "line_intensity", "line_opacity",
                        "cont_intensity", "cont_opacity"]
                for key in keys:
                    h5database[key].resize((self.N_unique), axis=0)
                    h5database[key][n_allocated:] = 0        
        # If we havent populated this dataset before, create room for model completion and success flags
        if not "model_completed" in h5database:
            h5database.create_dataset("model_completed" , shape = (self.model_data.shape[0],), maxshape=(None,), dtype = int)
            h5database.create_dataset("model_successful", shape = (self.model_data.shape[0],), maxshape=(None,), dtype = int)

        # Close database
        h5database.close()
    def save_models(self,model_successful, gasspy_ids, cont_intensity = None, cont_opacity = None , line_intensity = None, line_opacity = None):
        """
            Method to save a list of models
        """
        # Only main rank does this
        if mpi_rank != 0:
            return
        
        # Open database
        h5database =  hp.File(self.database_path, "r+")

        # If this is the first time we save, we must create the dataset (since we now know the shape)  
        if self.save_lines and not "line_intensity" in h5database:
            h5database.create_dataset("line_intensity", shape = (self.N_unique, self.N_lines),maxshape=(None,self.N_lines))
            h5database.create_dataset("line_opacity"  , shape = (self.N_unique, self.N_lines),maxshape=(None,self.N_lines))
            h5database.create_dataset("cont_intensity", shape = (self.N_unique, self.N_lines),maxshape=(None,self.N_lines))
            h5database.create_dataset("cont_opacity"  , shape = (self.N_unique, self.N_lines),maxshape=(None,self.N_lines))
            # Also save line information here
            h5database["line_labels"] = self.model_runner.get_line_labels()
            h5database["line_energies"] = self.model_runner.get_line_energies()

        # Start by sorting since h5py is picky
        sorter = gasspy_ids.argsort()
        gasspy_ids = gasspy_ids[sorter]

        model_successful = model_successful[sorter]

        if self.save_lines:
            line_intensity  = line_intensity[sorter,:]
            line_opacity    = line_opacity[sorter,:]
            cont_intensity  = cont_intensity[sorter,:]
            cont_opacity    = cont_opacity[sorter,:]                 
            h5database["line_intensity"][gasspy_ids,:] = line_intensity
            h5database["line_opacity"][gasspy_ids,:] = line_opacity    
            h5database["cont_intensity"][gasspy_ids,:] = cont_intensity
            h5database["cont_opacity"][gasspy_ids,:]   = cont_opacity  
        h5database["model_successful"][gasspy_ids] = model_successful
        h5database["model_completed"][gasspy_ids] = 1   
        
        # close database
        h5database.close()

    def gather_results(self):
        """
            Method to gather the results from the run models in the buffers
        """
        # Determine how many models were completed across all ranks
        n_complete_rank = np.zeros(mpi_size, dtype = int)
        n_complete_rank[:] = mpi_comm.allgather(self.local_n_complete)
        n_complete_cum = np.cumsum(n_complete_rank)
        n_complete_total = np.sum(n_complete_rank)
        
        all_cont_intensity = None
        all_cont_opacity = None
        all_line_intensity = None
        all_line_opacity = None

        if n_complete_total == 0:
            return
        if mpi_rank == 0:
            # Create arrays to store these models
            if self.save_lines:
                all_line_intensity = np.zeros((n_complete_total, self.N_lines))
                all_line_opacity = np.zeros((n_complete_total, self.N_lines))
                all_cont_intensity = np.zeros((n_complete_total, self.N_lines))
                all_cont_opacity = np.zeros((n_complete_total, self.N_lines))                               
            all_model_successful = np.zeros((n_complete_total), dtype = int)
            all_gasspy_ids = np.zeros((n_complete_total), dtype = int)

            # Set values for local completed models
            if self.local_n_complete > 0:
                if self.save_lines:
                    all_cont_intensity[:self.local_n_complete,:] = self.buffer_cont_intensity[:self.local_n_complete,:]
                    all_cont_opacity[:self.local_n_complete,:]   = self.buffer_cont_opacity[:self.local_n_complete,:]
                    all_line_intensity[:self.local_n_complete,:] = self.buffer_line_intensity[:self.local_n_complete,:]
                    all_line_opacity[:self.local_n_complete,:]   = self.buffer_line_opacity[:self.local_n_complete,:]
                all_model_successful[:self.local_n_complete] = self.buffer_model_successful[:self.local_n_complete]
                all_gasspy_ids[:self.local_n_complete] = self.buffer_gasspy_ids[:self.local_n_complete]

            # Loop over all ranks and gather completed models from other ranks
            for irank in range(1, mpi_size):
                if n_complete_rank[irank] == 0:
                    continue
                if self.save_lines:
                    all_cont_intensity[n_complete_cum[irank-1]:n_complete_cum[irank],:] = mpi_comm.recv(source = irank, tag = 6*irank + 1)
                    all_cont_opacity[n_complete_cum[irank-1]:n_complete_cum[irank],:] = mpi_comm.recv(source = irank, tag = 6*irank + 2)
                    all_line_intensity[n_complete_cum[irank-1]:n_complete_cum[irank],:] = mpi_comm.recv(source = irank, tag = 6*irank + 3)
                    all_line_opacity[n_complete_cum[irank-1]:n_complete_cum[irank],:] = mpi_comm.recv(source = irank, tag = 6*irank + 4)
                
                all_model_successful[n_complete_cum[irank-1]:n_complete_cum[irank]] = mpi_comm.recv(source = irank, tag = 6*irank + 5)
                all_gasspy_ids[n_complete_cum[irank-1]:n_complete_cum[irank]] = mpi_comm.recv(source = irank, tag = 6*irank + 6)
            
            # Save them
            mpi_print("\tdumping %d models"%n_complete_total)
            self.save_models(all_model_successful, all_gasspy_ids, 
                            cont_intensity=all_cont_intensity, cont_opacity=all_cont_opacity, 
                            line_intensity=all_line_intensity, line_opacity=all_line_opacity)
        else: # if not main rank
            if self.local_n_complete == 0:
                return
            if self.save_lines:
                mpi_comm.send(self.buffer_cont_intensity[:self.local_n_complete,:], dest = 0, tag = 6*mpi_rank + 1)            
                mpi_comm.send(self.buffer_cont_opacity[:self.local_n_complete,:], dest = 0, tag = 6*mpi_rank + 2)            
                mpi_comm.send(self.buffer_line_intensity[:self.local_n_complete,:], dest = 0, tag = 6*mpi_rank + 3)            
                mpi_comm.send(self.buffer_line_opacity[:self.local_n_complete,:], dest = 0, tag = 6*mpi_rank + 4)

            mpi_comm.send(self.buffer_model_successful[:self.local_n_complete], dest = 0, tag = 6*mpi_rank + 5)            
            mpi_comm.send(self.buffer_gasspy_ids[:self.local_n_complete], dest = 0, tag = 6*mpi_rank + 6)           

    def allocate_local_buffers(self):
        """
            Method to allocate the buffers
        """
        # How many do we expect?
        self.n_buffered_models = int(self.populator_dump_time/self.est_model_time) + 1
        if self.save_lines:
            self.buffer_cont_intensity = np.zeros((self.n_buffered_models,self.N_lines), dtype = float)
            self.buffer_cont_opacity = np.zeros((self.n_buffered_models,self.N_lines), dtype = float)
            self.buffer_line_intensity = np.zeros((self.n_buffered_models,self.N_lines), dtype = float)
            self.buffer_line_opacity = np.zeros((self.n_buffered_models,self.N_lines), dtype = float)            
        self.buffer_model_successful = np.zeros(self.n_buffered_models, dtype = int)
        self.buffer_gasspy_ids = np.zeros(self.n_buffered_models, dtype = int)
        self.local_n_complete = 0
        return
    
    def reset_local_buffers(self):  
        """
            Method to reset the buffers
        """
        # if they havent been allocated, do nothing

        if self.save_lines:
            self.buffer_line_intensity[:,:] = 0
            self.buffer_line_opacity[:,:] = 0 
            self.buffer_cont_intensity[:,:] = 0
            self.buffer_cont_opacity[:,:] = 0 

        self.buffer_model_successful[:] = 0
        self.buffer_gasspy_ids[:] = 0
        self.local_n_complete = 0
        return
    

    def __run_model__(self):
        if self.local_n_complete >= len(self.models_to_run):
            # If there is nothing to run locally, do nothing
            return
        # Grab current model
        gasspy_id = self.gasspy_ids[self.local_n_complete]
        model = self.models_to_run[self.local_n_complete,:]

        # Create dictionary of fields needed for the model_runner
        model_dict = {}
        for field in self.model_runner.required_fields:
            if field not in self.database_fields:
                mpi_print("ERROR: could not find field %s required by model_runner in database"%field)
                mpi_print("Fields required by model_runner :")
                mpi_print(self.model_runner.required_fields)
                mpi_print("Fields in database :")
                mpi_print(self.database_fields)                   
                sys.exit(0)

            ifield = self.database_fields.index(field)
            model_dict[field] = model[ifield]
        
        # Send model to model_runner
        self.model_runner.run_model(model_dict, "gasspy_%d"%gasspy_id)

        # set success state and gasspy id
        self.buffer_model_successful[self.local_n_complete] = self.model_runner.model_successful()
        self.buffer_gasspy_ids[self.local_n_complete] = gasspy_id

        if self.model_runner.model_successful():   
            # Get intensity and opacity off model
            if self.save_lines:
                self.buffer_line_intensity[self.local_n_complete,:] = self.model_runner.get_line_intensity()
                self.buffer_line_opacity[self.local_n_complete,:] = self.model_runner.get_line_opacity()

                # Grab smallest intensities and opacity as continuum comparison
                intensity = self.model_runner.get_intensity()[self.line_ienergies]
                opacity = self.model_runner.get_opacity()[self.line_ienergies]
                self.buffer_cont_intensity[self.local_n_complete,:] = np.min(intensity*self.denergy[:,None], axis = 1)
                self.buffer_cont_opacity[self.local_n_complete,:] = np.min(opacity, axis = 1)

        # advance number of completed models
        self.local_n_complete += 1
        self.model_runner.clean_model()

  