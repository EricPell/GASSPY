import h5py as hp
import numpy as np
import psutil
import shutil
import os
import sys 

from gasspy.physics.databasing.cgasspy_tree import CGasspyTree
from gasspy.io.gasspy_io import read_yaml, check_parameter_in_config, read_gasspy_config_hdf5, save_gasspy_config_hdf5
from gasspy.shared_utils.mpi_utils.mpi_print import mpi_print

class GasspyDatabase:
    def __init__(self, 
        gasspy_config,
        database_name = None,
        gasspy_modeldir = None,
        database_fields = None,
        refinement_fields = None,
        discrete_fields = None,
        fields_lrefine = None,
        fields_domain_limits = None
        ):
        if isinstance(gasspy_config, str):
            self.gasspy_config = read_yaml(gasspy_config)
        else:
            self.gasspy_config = gasspy_config

        # Name of the database which we are using
        self.database_name = check_parameter_in_config(self.gasspy_config, "database_name", database_name, "gasspy_database.hdf5") 
        
        # Path to database directory
        self.gasspy_modeldir = check_parameter_in_config(self.gasspy_config, "gasspy_modeldir", gasspy_modeldir, "gasspy_modeldir") 
        if not self.gasspy_modeldir.endswith("/"):
            self.gasspy_modeldir = self.gasspy_modeldir + "/"
        
        self.database_path = self.gasspy_modeldir+self.database_name     

        # Fields 
        self.database_fields = check_parameter_in_config(self.gasspy_config, "database_fields", database_fields, None)
        if self.database_fields is None:
            sys.exit("Error: No database_fields specified")
        self.__default_compression_ratio__ = (1,5.0)
        self.n_database_fields = len(self.database_fields)

        # fields we need for convergence
        self.refinement_fields = check_parameter_in_config(self.gasspy_config, "refinement_fields", refinement_fields, None)
        self.n_refinement_fields = len(self.refinement_fields)

        self.refinement_to_database_field_index = np.array([self.database_fields.index(field) for field in self.refinement_fields], dtype = int)
   
        # index of center cell in neighbor array (i.e the "self" neighbor)
        self.center_cell_index = 0
        for ifield in range(self.n_refinement_fields):
            self.center_cell_index += 3**(self.n_refinement_fields - 1 - ifield)

        # Refinement level min and max for all refinement fields
        default = {}
        self.fields_lrefine = check_parameter_in_config(self.gasspy_config, "fields_lrefine", fields_lrefine, default)
        for field in self.refinement_fields:
            assert field in self.fields_lrefine, "Field %s does not have its refinement levels specified. All refinement fields must have their refinement levels specified."%field
            # make into arrays
            self.fields_lrefine[field] = np.array(self.fields_lrefine[field]) 

        # Grab the maximum lrefine for each of the fields we intend to refine
        self.max_lrefine = np.array([self.fields_lrefine[field][1] for field in self.refinement_fields])


        # Domain limits for refinement fields
        default = {}
        self.fields_domain_limits = check_parameter_in_config(self.gasspy_config, "fields_domain_limits", fields_domain_limits, default)
        for field in self.fields_domain_limits:
            # make into arrays
            self.fields_domain_limits[field] = np.array(self.fields_domain_limits[field])

        # Make sure all refinement fields have their limits specified
        for field in self.refinement_fields:
            assert field in self.fields_domain_limits, "Field %s does not have its domain limits specified"%field

        # Discrete fields that should not be used for refinement
        self.discrete_fields = check_parameter_in_config(self.gasspy_config, "discrete_fields", discrete_fields, [])
        self.n_discrete_fields = len(self.discrete_fields)
        self.discrete_to_database_field_index = np.array([self.database_fields.index(field) for field in self.discrete_fields])


        self.h5GroupName = "Tree_structure"

        if os.path.exists(self.database_path):
            mpi_print("Appending to database %s"%(self.database_path))
            self.verify_h5database()

        else:
            mpi_print("Creating new database %s"%(self.database_path))
            self.append = False
            self.create_new_h5database()
            self.N_unique = 0

        self.tree = None
        self.load_database()
        self.sim_readers = []



    def create_new_h5database(self):
        h5database = hp.File(self.gasspy_modeldir + self.database_name, "w")
        # Create all the neccesary datasets
        save_gasspy_config_hdf5(self.gasspy_config, h5database)
        h5database["database_fields"] = self.database_fields
        h5database["refinement_fields"] = self.refinement_fields
        h5database["discrete_fields"] = self.discrete_fields

        #close the database
        h5database.close()



    def verify_h5database(self):
        h5database = hp.File(self.database_path, "r")
        old_gasspy_config = {}
        read_gasspy_config_hdf5(old_gasspy_config, h5database)
        h5database.close()

        old_database_fields = old_gasspy_config["database_fields"]
        assert old_database_fields == self.database_fields, "database_fields do not match between supplied gasspy_config and database. (NOTE: order matters)" 

        old_refinement_fields = old_gasspy_config["refinement_fields"]
        assert old_refinement_fields == self.refinement_fields, "refinement_fields do not match between supplied gasspy_config and database. (NOTE: order matters)" 

        old_fields_lrefine = old_gasspy_config["fields_lrefine"]
        for field in self.refinement_fields:
            assert np.array_equal(old_fields_lrefine[field], self.fields_lrefine[field]), "fields_lrefine for field %s does not match between supplied gasspy_config and database"%field 

        old_fields_domain_limits = old_gasspy_config["fields_domain_limits"]
        for field in old_fields_domain_limits:
            assert np.array_equal(old_fields_domain_limits[field], self.fields_domain_limits[field]), "fields_domain_limits for field %s does not match between supplied gasspy_config and database"%field 


    def load_sim_data(self, sim_reader):
        h5database = hp.File(self.database_path, "r")

        database_fields =  [field.decode() for field in h5database["database_fields"][:]]
        gasspy_config = {}
        read_gasspy_config_hdf5(gasspy_config, h5database)
        h5database.close()
        for ifield, field in enumerate(database_fields):
            field_data = np.log10(sim_reader.get_field(field))
            # If field has lower or upper limits, set these
            if field in self.fields_domain_limits:
                maxval = self.fields_domain_limits[field][1]
                field_data[field_data > maxval] = maxval
                minval = self.fields_domain_limits[field][0]
                field_data[field_data < minval] = minval
            if ifield == 0:
                N_cells = field_data.shape[0]
                sim_data = np.zeros((N_cells, len(database_fields)))
            sim_data[:,ifield] = field_data
        return sim_data

    def create_new_tree(self):
        # If we already have tree loaded, delete it.. no need for confusion
        if self.tree is not None:
            del self.tree
        
        # Set lrefine and domain limits for each field using 2D arrays
        lrefine_limits = np.zeros((self.n_refinement_fields,2))
        domain_limits  = np.zeros((self.n_refinement_fields,2))
        for ifield, field in enumerate(self.refinement_fields):
            lrefine_limits[ifield] = self.fields_lrefine[field]
            domain_limits[ifield]  = self.fields_domain_limits[field]

        # initialize tree
        self.tree = CGasspyTree(self.n_database_fields, self.n_refinement_fields, self.refinement_to_database_field_index, self.n_discrete_fields, self.discrete_to_database_field_index, domain_limits, lrefine_limits)

        pass

    def clean_tree(self):
        del self.tree
        self.tree = None

    def load_database(self):
        # Create a new tree to populate
        self.create_new_tree()

        # If file does not exist do nothing
        if not os.path.exists(self.database_path):
            return
        
        # Likewise, if the hdf5 file does not have any tree information, do nothing
        h5database = hp.File(self.database_path, "r")
        if not self.h5GroupName in h5database.keys():
            return
        
        # All the setter functions required 
        setters = {
            "nodes_gasspy_ids" : self.tree.set_nodes_gasspy_ids,
            "roots_discrete_data": self.tree.set_roots_discrete_data,
            "nodes_root_id": self.tree.set_nodes_root_id,
            "nodes_is_root": self.tree.set_nodes_is_root,
            "nodes_is_leaf": self.tree.set_nodes_is_leaf,
            "nodes_is_required": self.tree.set_nodes_is_required,
            "nodes_has_converged": self.tree.set_nodes_has_converged,
            "nodes_split_field": self.tree.set_nodes_split_field,
            "nodes_child_node_ids": self.tree.set_nodes_child_node_ids,
            "nodes_neighbor_node_ids": self.tree.set_nodes_neighbor_node_ids
        }

        # The group in which we save
        tree_group = h5database[self.h5GroupName]
        nodes_model_data = tree_group["nodes_model_data"][:]
        nodes_node_lrefine = tree_group["nodes_node_lrefine"][:]
        self.tree.initialize_nodes(nodes_model_data, nodes_node_lrefine)
        for label in setters:
            setters[label](tree_group[label][:])
        # Load and insert the unique gasspy_model_data
        self.tree.set_gasspy_model_data(h5database["gasspy_model_data"][:])
        h5database.close()
    


    def save_database(self):
        mpi_print("Saving database")
        # Create a new file if it doesnt already exist
        if not os.path.exists(self.database_path):
            h5database = hp.File(self.database_path, "w")
        else:
            h5database = hp.File(self.database_path, "r+")



        getters = {
            "nodes_model_data" : self.tree.get_nodes_model_data,
            "nodes_node_lrefine" : self.tree.get_nodes_node_lrefine,
            "nodes_gasspy_ids" : self.tree.get_nodes_gasspy_ids,
            "roots_discrete_data": self.tree.get_roots_discrete_data,
            "nodes_root_id" : self.tree.get_nodes_root_id,
            "nodes_is_root": self.tree.get_nodes_is_root,
            "nodes_is_leaf": self.tree.get_nodes_is_leaf,
            "nodes_is_required": self.tree.get_nodes_is_required,
            "nodes_has_converged": self.tree.get_nodes_has_converged,
            "nodes_split_field": self.tree.get_nodes_split_field,
            "nodes_child_node_ids": self.tree.get_nodes_child_node_ids,
            "nodes_neighbor_node_ids": self.tree.get_nodes_neighbor_node_ids
        }

        # If the Tree structure group doesnt already exist in this database, create it 
        if self.h5GroupName not in h5database.keys():
            tree_group = h5database.create_group(self.h5GroupName)
        else:
            tree_group = h5database[self.h5GroupName]

        for label in getters:
            data = getters[label]()
            if label not in tree_group.keys():
                shape = (0,)+data.shape[1:]
                max_shape = (None,) + data.shape[1:]
                tree_group.create_dataset(label, shape, maxshape = max_shape, dtype = data.dtype)
            
            N_saved = tree_group[label].shape[0]
            N_total = data.shape[0]
            # If we have no new nodes, just break
            if N_total == N_saved:
                continue

            assert N_total > N_saved, "Error, tree structure of dataset is larger than the one we are trying to save"
            tree_group[label].resize((N_total), axis = 0)
            tree_group[label][:] = data[:]
        

        # Save gasspy_model_data
        if "gasspy_model_data" not in h5database.keys():
            h5database.create_dataset("gasspy_model_data", (0, self.n_database_fields), maxshape = (None, self.n_database_fields))
        gasspy_model_data = self.tree.get_gasspy_model_data()
        N_saved = h5database["gasspy_model_data"].shape[0]
        N_total = gasspy_model_data.shape[0]
        if N_saved != N_total:
            assert N_total > N_saved, "Error, tree structure of dataset is larger than the one we are trying to save"
            h5database["gasspy_model_data"].resize((N_total), axis = 0)
            h5database["gasspy_model_data"][N_saved:N_total] = gasspy_model_data[N_saved:N_total]
        h5database.close()






    def add_snapshot(self, sim_reader):
        """
            Adds a SimulationReader object to the database
            input:
                sim_reader : SimulationReader (object that reads the simulation snapshot in question)
            output:
                None
        """
        self.sim_readers.append(sim_reader)
        mpi_print("Adding snapshot to the database")
        self.add_cells(sim_reader)
        # Set new gasspy_ids
        print("set_uniques")
        self.tree.set_unique_gasspy_models()
        self.save_database()
        return    
    
    def add_cells(self, sim_reader):
        """
            Adds cells from a SimulationReader object to the tree
            input:
                sim_reader : SimulationReader (object that reads the simulation snapshot in question)
            output:
                None
        """
        sim_data = self.load_sim_data(sim_reader)
        try: 
            previous_node_ids = sim_reader.load_new_field("cells_node_ids")
        except:
            previous_node_ids = np.full(sim_data.shape[0], -1, dtype=np.int32)
        new_node_ids = self.tree.add_points(sim_data, previous_node_ids)
        sim_reader.save_new_field("cells_node_ids", new_node_ids)
        self.tree.set_neighbors()      

    def get_gasspy_ids(self, sim_reader):
        """
            Adds cells from a SimulationReader object to the tree
            input:
                sim_reader : SimulationReader (object that reads the simulation snapshot in question)
            output:
                cell_gasspy_ids : array of ints (gasspy_ids for every cell)
        """
        mpi_print("Loading snapshot")
        sim_data = self.load_sim_data(sim_reader)
        try: 
            previous_node_ids = sim_reader.load_new_field("cells_node_ids")
        except:
            previous_node_ids = np.full(sim_data.shape[0], -1, dtype=np.int32)

        mpi_print("Matching to in the tree [Gasspyids]")
        gasspy_ids = self.tree.get_gasspy_ids(sim_data, previous_node_ids)
        sim_reader.save_new_field("cells_gasspy_ids", gasspy_ids)
        return gasspy_ids

    def get_node_ids(self, sim_reader):
        """
            Adds cells from a SimulationReader object to the tree
            input:
                sim_reader : SimulationReader (object that reads the simulation snapshot in question)
            output:
                cell_node_ids : array of ints (id of the nodes that each cell was matched to)
        """
        mpi_print("Loading snapshot")
        sim_data = self.load_sim_data(sim_reader)
        try: 
            previous_node_ids = sim_reader.load_new_field("cells_node_ids")
        except:
            previous_node_ids = np.full(sim_data.shape[0], -1, dtype=np.int32)

        mpi_print("Matching to nodes in the tree")
        new_node_ids = self.tree.get_node_ids(sim_data, previous_node_ids)
        sim_reader.save_new_field("cells_node_ids", new_node_ids)
        return new_node_ids

    

    """
        Wrappers for various getters for data of nodes
    """
    def get_nodes_model_data(self):
        """
            Get the model_data of the nodes
            input:
                None
            output:
                nodes_model_data : array of float64 (model_data for each node)
        """
        return self.tree.get_nodes_model_data()
    
    def get_nodes_node_lrefine(self):
        """
            get the lrefine on the nodes
            input:
                None
            output:
                nodes_node_lrefine : array of int16 (node_lrefine for each node)
        """
        return self.tree.get_nodes_node_lrefine()

    def get_nodes_gasspy_ids(self):
        """
            get the gasspy_ids of the nodes
            input:
                None
            output:
                nodes_gasspy_ids : array of int32 (gasspy_ids for each node)
        """
        return self.tree.get_nodes_gasspy_ids()
    
    def get_nodes_root_id(self):
        """
            get the root_id of the nodes
            input:
                None
            output:
                nodes_gasspy_ids : array of int32 (root_ids for each node)
        """
        return self.tree.get_nodes_root_id()
    
    def get_nodes_is_root(self):
        """
            get the is_root status of the nodes
            input:
                None
            output:
                nodes_is_root : array of int8 (is_root for each node)
        """
        return self.tree.get_nodes_is_root()
    
    def get_nodes_is_leaf(self):
        """
            get the is_leaf status of the nodes
            input:
                None
            output:
                nodes_is_leaf : array of int8 (is_leaf for each node)
        """
        return self.tree.get_nodes_is_leaf()
    
    def get_nodes_is_required(self):
        """
            get the is_required status of the nodes
            input:
                None
            output:
                nodes_is_required : array of int8 (is_required for each node)
        """
        return self.tree.get_nodes_is_required()

    def get_nodes_has_converged(self):
        """
            get the is_required status of the nodes
            input:
                None
            output:
                nodes_is_required : array of int8 (is_required for each node)
        """
        return self.tree.get_nodes_has_converged()
    
    def get_nodes_split_field(self):
        """
            get the split_field of the nodes, eg the index of the fieldd along which the nodes have been split
            input:
                None
            output:
                nodes_split_field : array of int16 (split_field for each node)
        """
        return self.tree.get_nodes_split_field()
    
    def get_nodes_child_node_ids(self):
        """
            get the child_node_ids of the nodes
            input:
                None
            output:
                nodes_child_node_ids : array of int32 (child_node_ids for each node)
        """
        return self.tree.get_nodes_child_node_ids()
    
    def get_nodes_neighbor_node_ids(self):
        """
            get the neighbor_node_ids of the node
            input:
                None
            output:
                nodes_neighbor_ids : array of int32 (neighbor_node_ids for each node)
        """
        return self.tree.get_nodes_neighbor_node_ids()


    """
        Various setters
    """
    def check_type(self, array, dtype, method):
        """
            Method to check that supplied array is of correct type, prints a warning if not
            input:
                array : array (array to be checked)
                dtype : dtype object (dtype that is desired)
                method: String (name of the function for debug purposes)
        """
        if not array.dtype == dtype:
            mpi_print("Warning:%s recieved array of type %s, but expecting %s. Converting array"%(method, array.dtype.name, dtype.name))
        return array.astype(dtype)

    def set_nodes_gasspy_ids(self, gasspy_ids):
        """
            Set gasspy_ids of the nodes
            input:
                node_gasspy_ids : array of int32 (gasspy_ids of the nodes)
            output:
                None
        """
        gasspy_ids = self.check_type(gasspy_ids, np.int32, "set_nodes_gasspy_ids")
        self.tree.set_nodes_gasspy_ids(gasspy_ids)
        return


    def set_nodes_root_id(self, root_id):
        """
            set root_id status of the nodes
            input:
                node_root_id : array of int32 (root_id of the nodes)
            output:
                None
        """
        root_id = self.check_type(root_id, np.int32, "set_nodes_root_id")
        self.tree.set_nodes_root_id(root_id)
        return

    def set_nodes_is_root(self, is_root):
        """
            set is_root status of the nodes
            input:
                node_is_root : array of int8 (is_root of the nodes)
            output:
                None
        """
        is_root = self.check_type(is_root, np.int8, "set_nodes_is_root")
        self.tree.set_nodes_is_root(is_root)
        return
    
    def set_nodes_is_leaf(self, is_leaf):
        """
            set is leaf status of the nodes
            input:
                node_gasspy_is_leaf : array of int8 (is_leaf of the nodes)
            output:
                None
        """
        is_leaf = self.check_type(is_leaf, np.int8, "set_nodes_is_leaf")
        self.tree.set_nodes_is_leaf(is_leaf)
        return

    def set_nodes_is_required(self, is_required):
        """
            set is required status of the nodes
            input:
                node_is_required : array of int8 (is_required of the nodes)
            output:
                None
        """
        is_required = self.check_type(is_required, np.int8, "set_nodes_is_required")
        self.tree.set_nodes_is_required(is_required)
        return
    
    def set_nodes_has_converged(self, has_converged):
        """
            set convergence status of the nodes
            input:
                node_has_converged : array of int8 (has_converged of the nodes)
            output:
                None
        """
        has_converged = self.check_type(has_converged, np.int8, "set_nodes_has_converged")
        self.tree.set_nodes_has_converged(has_converged)
        return

    def set_nodes_split_field(self, split_field):
        """
            set split_field of the nodes
            input:
                node_split_field : array of int16 (split_field of the nodes)
            output:
                None
        """
        split_field = self.check_type(split_field, np.int32, "set_nodes_split_field")
        self.tree.set_nodes_split_field(split_field)
        return

    def set_nodes_child_node_ids(self, child_node_ids):
        """
            set child_node_ids of the nodes
            input:
                node_child_node_ids : array of int32 (child_node_ids of the nodes)
            output:
                None
        """
        child_node_ids = self.check_type(child_node_ids, np.int32, "set_nodes_child_node_ids")
        self.tree.set_nodes_child_node_ids(child_node_ids)
        return

    def set_nodes_neighbor_node_ids(self, neighbor_node_ids):
        """
            set neighbor_node_ids of the nodes
            input:
                node_neighbor_node_ids : array of int32 (neighbor_node_ids of the nodes)
            output:
                None
        """
        neighbor_node_ids = self.check_type(neighbor_node_ids, np.int32, "set_nodes_neighbor_node_ids")
        self.tree.set_nodes_neighbor_node_ids(neighbor_node_ids)
        return



    def __recursive_get_parallel_idxs(self, ishift, iparallel, ifield, ifield_ishift, ifield_to_check, idxm, idx, idxp):
        """
            Recursive function to determine which neighbors corresponds to parallell shifts with respect to ifield_to_check
        """
        if ifield == self.n_refinement_fields:
            idxm[iparallel] = ishift
            idx [iparallel] = ishift +   ifield_ishift
            idxp[iparallel] = ishift + 2*ifield_ishift
            return iparallel + 1, ishift + 1 

        if ifield == ifield_to_check:
            ishift_old = ishift
            iparallel, ishift = self.__recursive_get_parallel_idxs(ishift, iparallel, ifield+1, ifield_ishift, ifield_to_check, idxm, idx, idxp)
            ishift = ishift_old + 3*ifield_ishift

            return iparallel, ishift

        for local_ishift in range(3):
            iparallel, ishift = self.__recursive_get_parallel_idxs(ishift, iparallel, ifield+1, ifield_ishift, ifield_to_check, idxm, idx, idxp)
        return iparallel, ishift


        
    def get_norm(self, values, norm_values, ifield_to_check):
        """
            Determine normalization for paralell checks
        """
        idxm = np.zeros(3**(self.n_refinement_fields-1), dtype = int)
        idx  = np.zeros(3**(self.n_refinement_fields-1), dtype = int)
        idxp = np.zeros(3**(self.n_refinement_fields-1), dtype = int)
        ifield_ishift = 3**(self.n_refinement_fields - ifield_to_check - 1)
        self.__recursive_get_parallel_idxs(0, 0, 0, ifield_ishift, ifield_to_check, idxm, idx, idxp)
    
        norm_values[:,idxm] = values[:,idx]
        norm_values[:,idx]  = values[:,idx]
        norm_values[:,idxp] = values[:,idx]

    def __final_check__(self, ishift, ifield_to_check, diagonal, atol, rtol):
        if ishift == self.center_cell_index:
            # If this is the center cell, just pass
            return ishift + 1

        if diagonal:
            # If this is a diagonal, this could be a covariance error or just simply another variable that is causing the error. 
            # Try to see if the diagonal errors are large enough, and if so stricten the convergence on the parallel 
            diagonal_error_large = (self.absolute_error_to_center[:, ishift] > atol)*(self.relative_error_to_center[:, ishift] > rtol)
            parallel_error_large = np.where((self.absolute_error[:, ishift] > 0.5*atol)*(self.relative_error[:, ishift] > 0.5*rtol)*diagonal_error_large)[0]
            parallel_error_max   = np.where((self.relative_error[:, ishift] > self.max_error[:, ishift])*diagonal_error_large )[0]

            # Set those where the parallel error is to large to not ok
            self.current_model_converged[parallel_error_large, ifield_to_check]=0

            # Set the new maximum errors and corresponding ifield
            self.max_error[parallel_error_max, ishift] = self.relative_error[parallel_error_max, ishift]
            self.max_error_ifield[parallel_error_max] = ifield_to_check

        # Regular parallell check
        error_large = np.where((self.absolute_error[:, ishift] > atol)*(self.relative_error[:, ishift] > rtol))[0]
        self.current_model_converged[error_large, ifield_to_check] = 0

        return ishift + 1

    def __recursive_check__(self, ishift, ifield, ifield_to_check, diagonal, atol, rtol):
        # If we've reached the end of our fields, process the error for this shift
        if ifield == self.n_refinement_fields:
            return self.__final_check__(ishift, ifield_to_check, diagonal, atol, rtol)

        # if any of the fields that are currently not being checked are not at their central value, then this is a diagonal model and needs to be treated differently
        if ifield == ifield_to_check:
            # Loop over shifts ([-1,1]) and go to the next field (if we dont change ifield_to_check, then there should not be an error here due to ifield_to_check)
            ishift = self.__recursive_check__(ishift, ifield+1, ifield_to_check, diagonal, atol, rtol)   
            ishift += 3**(self.n_refinement_fields - ifield_to_check - 1) # Skip 0th shift
            ishift = self.__recursive_check__(ishift, ifield+1, ifield_to_check, diagonal, atol, rtol)   

        else:
            diagonal_now = [True, False, True]
            # Loop over all shifts ([-1,0,1]) and go to the next field
            for local_ishift in range(3):
                ishift = self.__recursive_check__(ishift, ifield+1, ifield_to_check, diagonal or diagonal_now[local_ishift], atol, rtol)   
 

        # If this is the first field then we are done with the recursive loop. If so, return nothing, otherwise return the ishift value for progression in the outer loops
        if ifield > 0:
            return ishift
        else:
            return

    def check_model_convergence(self, values, atol, rtol):
        # Ensure that shapes makes sense
        values = np.atleast_2d(values)

        # Determine the absolute and relative errors with respect to the center model 
        self.absolute_error_to_center = np.abs(values[:,:] -values [:,self.center_cell_index][:, np.newaxis])
        self.relative_error_to_center = self.absolute_error_to_center/(values[:,self.center_cell_index][:,np.newaxis] + 1e-40)

        # Initialize arrays to determine the variable who's likely the most responsible for an error in a given direction
        self.max_error = np.zeros(values.shape)
        self.max_error_ifield = np.zeros(values.shape, dtype = int) - 1

        # Initialize arrays for model_convergence
        self.current_model_converged = np.ones((values.shape[0],len(self.refinement_fields)), dtype = np.int8)

        for ifield, field in enumerate(self.refinement_fields):
            # Determine the reference value used for each comparison (eg the central value for the current field)
            norm_values = np.zeros(values.shape)
            self.get_norm(values, norm_values,ifield)

            # Get absolute and relative error with respect to these reference values
            self.absolute_error = np.abs(values - norm_values)
            self.relative_error = self.absolute_error/(norm_values + 1e-40)

            # Chech which errors are too large and which fields need to be refined
            self.__recursive_check__(0, 0, ifield, False, atol, rtol)

        # Set the status of the maximum error field to not converged. Will only change things if multiple variables are ok, but their covariance is too large
        for ifield, field in enumerate(self.refinement_fields):
            max_error_here = np.where(self.max_error_ifield == ifield)[0]
            self.current_model_converged[max_error_here,  self.refinement_fields.index(field)] = 0

        return 
    

    def get_atol_rtol(self, convergence_criterion, model_ids, cont_intensity = None):
        atol = np.full(model_ids.shape,-1, dtype = float)
        rtol = np.full(model_ids.shape,-1, dtype = float)
        if "atol" in convergence_criterion:
            atol[:] = convergence_criterion["atol"]
        if "rtol" in convergence_criterion:
            rtol[:] = convergence_criterion["rtol"]

        if "max_surface_brightness" in convergence_criterion:
            h5database = hp.File(self.gasspy_modeldir + self.database_name, "r")
            # Load cell sizes
            unique_ids, local_ids = np.unique(model_ids, return_inverse = True)

            sorter = unique_ids.argsort()
            inv_sorter = sorter.argsort()
            cell_sizes = 10** h5database["gasspy_model_data"][unique_ids[sorter],self.database_fields.index("cell_size")][inv_sorter][local_ids]
            max_surface_brightness = convergence_criterion["max_surface_brightness"] 
            atol = np.maximum(3*max_surface_brightness/cell_sizes, atol)
            h5database.close()
        if "reference_continuum" in convergence_criterion:
            scale_factor = convergence_criterion["reference_continuum"]["scale_factor"]
            atol = np.maximum(atol, scale_factor*cont_intensity)

        atol[atol < 0.0] = 1e99
        rtol[rtol < 0.0] = 1e99
        return atol, rtol
    
    def check_spectra_ranges_convergence(self, required_models, all_neighbor_ids, all_neighbor_gasspy_ids):
        h5database =  hp.File(self.database_path, "r")

        criterions_max_floats = 1
        energy = h5database["energy"][:]
        delta_energy = h5database["delta_energy"][:]

        # Determine spectra ranges and indexes
        spectra_idxs = []
        for label in self.gasspy_config["convergence_criterions"]["spectra_ranges"]:
            spectra_range = self.gasspy_config["convergence_criterions"]["spectra_ranges"][label]
            emin = spectra_range["energy_range"][0]
            emax = spectra_range["energy_range"][1]
            eidx = np.searchsorted(energy, np.array([emin, emax]))
            eidx[1]+=1
            eidx[1] = min(eidx[1], len(energy))
            spectra_idxs.append(eidx)
            # How many spectral bins will we need for this
            criterions_max_floats = max(criterions_max_floats, eidx[1]-eidx[0])
        spectra_idxs = np.array(spectra_idxs)
        avail_memory = psutil.virtual_memory().available
        # Estimate how many models we can work on at a time
        n_models =  int(0.8*avail_memory/((4+ criterions_max_floats)*8*3**self.n_refinement_fields))
        
        # Loop over all models
        N_required = len(required_models)
        imodel = 0

        while imodel < N_required:
            required_model_idxs = np.arange(imodel, min(N_required, imodel + n_models))
            # Models we want to check now
            node_ids = required_models[required_model_idxs]

            # Grab ids for all the neighbors
            neighbor_gasspy_ids = all_neighbor_gasspy_ids[node_ids,:]

            # Grab center gasspy ids, corresponding to the node in question 
            nodes_gasspy_ids = neighbor_gasspy_ids[:,self.center_cell_index]

            # Determine all unique ones
            unique_ids, local_ids = np.unique(neighbor_gasspy_ids.ravel(), return_inverse = True)
            # Loop over all spectra ranges and grab their 
            for irange, label in enumerate(self.gasspy_config["convergence_criterions"]["spectra_ranges"]):

                spectra_range = self.gasspy_config["convergence_criterions"]["spectra_ranges"][label]
                intensity = np.sum(h5database["intensity"][unique_ids, spectra_idxs[irange][0]:spectra_idxs[irange][1]+1]*delta_energy[spectra_idxs[irange][0]:spectra_idxs[irange][1]+1], axis = 1)
                neighbor_intensity = intensity[local_ids].reshape(neighbor_gasspy_ids.shape)
                atol, rtol = self.get_atol_rtol(spectra_range, nodes_gasspy_ids)
                self.check_model_convergence(neighbor_intensity, atol, rtol)
                self.model_converged[required_model_idxs, :] *= self.current_model_converged[:,:]
            imodel = min(N_required, imodel+n_models)

        h5database.close()

    def check_lines_convergence(self, required_models, all_neighbor_gasspy_ids):
        h5database =  hp.File(self.gasspy_modeldir + self.database_name, "r")

        criterions_max_floats = 1
        energy = h5database["energy"][:]
        delta_energy = h5database["delta_energy"][:]
        line_labels = [line_label.decode() for line_label in h5database["line_labels"]]

    
        continuum_idxs = []
        for line_label in self.gasspy_config["convergence_criterions"]["lines"]:
            line_criterion = self.gasspy_config["convergence_criterions"]["lines"][line_label]
            if "reference_continuum" in line_criterion:
                emin = line_criterion["reference_continuum"]["energy_range"][0]
                emax = line_criterion["reference_continuum"]["energy_range"][1]
                eidx = np.searchsorted(energy, np.array([emin, emax]))
                eidx[1]+=1
                continuum_idxs.append(eidx)
                # How many spectral bins will we need for this
                criterions_max_floats = max(criterions_max_floats, eidx[1]-eidx[0])
        continuum_idxs = np.array(continuum_idxs)
        avail_memory = psutil.virtual_memory().available
        # Estimate how many models we can work on at a time
        n_models =  int(0.8*avail_memory/((4+ criterions_max_floats)*8*3**self.n_refinement_fields))
        

        N_required = len(required_models)
        imodel = 0
        while imodel < N_required:
            required_model_idxs = np.arange(imodel, min(N_required, imodel + n_models))
            # Models we want to check now
            node_ids = required_models[required_model_idxs]
            
            # Grab ids for all the neighbors
            neighbor_gasspy_ids = all_neighbor_gasspy_ids[node_ids,:]

            # Grab center gasspy ids for all the neighbors 
            nodes_gasspy_ids, nodes_local_ids = np.unique(neighbor_gasspy_ids[:,self.center_cell_index], return_inverse=True)
        
            sorter = nodes_gasspy_ids.argsort()
            inv_sorter = sorter.argsort()

            # Determine all unique ones
            unique_ids, local_ids = np.unique(neighbor_gasspy_ids.ravel(), return_inverse = True)


            # Counter for continuum checks
            icont = 0
            # Loop over all line criterions
            for line_label in self.gasspy_config["convergence_criterions"]["lines"]:
                line_criterion = self.gasspy_config["convergence_criterions"]["lines"][line_label]
                # Do we need continuum?
                if "reference_continuum" in line_criterion:
                    cont_intensity = np.sum(h5database["intensity"][nodes_gasspy_ids[sorter], continuum_idxs[icont][0]:continuum_idxs[icont][1]+1]*delta_energy[continuum_idxs[icont][0]:continuum_idxs[icont][1]+1], axis = 1)[inv_sorter][nodes_local_ids]
                else:
                    cont_intensity  = None
                # Which index does this line have in the database
                line_index = line_labels.index(line_label)
                # Grab the intensity of all neighbors and ensure correct shape
                neighbor_line_intensity = h5database["line_intensity"][unique_ids, line_index][local_ids].reshape(neighbor_gasspy_ids.shape)
                # Get absolute and relative tolerances for this criterion
                atol, rtol = self.get_atol_rtol(line_criterion, nodes_gasspy_ids[nodes_local_ids], cont_intensity=cont_intensity)
                # Calculate model convergence
                self.check_model_convergence(neighbor_line_intensity, atol, rtol)
                # Logical and
                self.model_converged[required_model_idxs, :] *= self.current_model_converged[:,:]
            imodel = min(N_required, imodel+n_models)
        h5database.close()

    def check_convergence(self):
        if self.tree is None:
            self.load_database()

        # If no convergence criterions have been specified, then everything is converged
        if not "convergence_criterions" in self.gasspy_config:
            return
        mpi_print("Checking convergence")

        is_required = self.get_nodes_is_required()
        is_leaf = self.get_nodes_is_leaf()
        has_converged = self.get_nodes_has_converged()
        all_neighbor_node_ids = self.get_nodes_neighbor_node_ids()
        all_neighbor_gasspy_ids = self.get_nodes_gasspy_ids()[all_neighbor_node_ids]

        # Take all leaf nodes, that are required by the current snaphsots, and has not already converged
        required_models = np.where((is_required == 1)*(has_converged != 1))[0]

        # Only look at required models that have not converged
        self.model_converged = np.zeros((len(required_models), self.n_refinement_fields))

        # Reset all non-converged required models to converged, and let inner functions tell otherwise
        self.model_converged[:,:] = 1
        if "spectra_ranges" in self.gasspy_config["convergence_criterions"]:
            self.check_spectra_ranges_convergence(required_models, all_neighbor_node_ids, all_neighbor_gasspy_ids)
        if "lines" in self.gasspy_config["convergence_criterions"]:
            self.check_lines_convergence(required_models, all_neighbor_gasspy_ids)    

        # Increase the refinement level 
        nodes_lrefine = self.get_nodes_node_lrefine()[required_models]

        # Which nodes has converged?
        has_converged = np.where(np.all(self.model_converged==1, axis=1))[0]
        self.tree.set_has_converged(required_models[has_converged])

        # Determine which fields for which nodes need to be refined
        need_refinement = np.where((self.model_converged == 0)*(nodes_lrefine < self.max_lrefine[np.newaxis,:]))
        if len(need_refinement[0]) == 0:
            mpi_print("\tNo further refinement needed for now")
            return True

        nodes_to_refine = np.unique(need_refinement[0])
        mpi_print("\t%d models needs to be refined"%len(nodes_to_refine))

        # Increase their refinement level by 1
        nodes_lrefine[need_refinement] += 1
        mpi_print("\tRefining")
        self.tree.refine_nodes(required_models[nodes_to_refine], nodes_lrefine[nodes_to_refine])

        # Loop through all sim readers and find which models their cells corresponds to in order to know which models are required
        mpi_print("\tRe-adding points")
        for sim_reader in self.sim_readers:
            self.add_cells(sim_reader)
        
        # set new neighbors
        mpi_print("\tSetting neighbors")
        self.tree.set_neighbors()    

        # Set new gasspy_ids
        mpi_print("\tDetermening new unique models")
        self.tree.set_unique_gasspy_models()  


        self.save_database()
        return False

    def finalize(self):
        self.save_database()
        return


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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcol
    from mpl_toolkits.axes_grid1 import make_axes_locatable
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
            return self.__dict__[field].copy()
        
        def save_new_field(self, field, data, dtype = None):
            """
                Function to save a new field, used to save the gasspy_ids, here just store in the class
            """
            self.__dict__[field] = data
            return
        
        def load_new_field(self, field):
            return self.__dict__[field].copy()

    def get_intensity1(var1, var2):
        return np.exp(-(2*var1)**10)
    def get_intensity2(var1, var2):
        return np.exp(-(var1 + var2)**10)
    def get_intensity3(var1, var2):
        arr = np.zeros(var1.shape)
        arr[var2<=0.511111111111] = 1.0
        return arr
        
    def populate(h5database_path):
        h5database =  hp.File(h5database_path, "r+")

        if "intensity" in h5database:
            del h5database["intensity"]
            del h5database["energy"]
            del h5database["delta_energy"]

        if "line_intensity" in h5database:
            del h5database["line_intensity"]
            del h5database["line_labels"]

        database_fields = [ field.decode() for field in h5database["database_fields"][:]]

        var1 = h5database["gasspy_model_data"][:,database_fields.index("var1")]
        var2 = h5database["gasspy_model_data"][:,database_fields.index("var2")]

        intensity = np.zeros(var1.shape + (1,))
        intensity[:,:] = get_intensity1(var1,var2)[:,np.newaxis]
        h5database["intensity"] = intensity
        h5database["energy"] = np.array([1])
        h5database["delta_energy"] = np.ones(1)

        line_intensity = np.zeros(var1.shape + (2,))
        line_intensity[:,0] = get_intensity2(var1,var2)
        line_intensity[:,1] = get_intensity3(var1,var2)
        h5database["line_intensity"] = line_intensity
        h5database["line_labels"] = ["line_1", "line_2"]

        h5database.close()

    def add_colorbar(fig, ax, im, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size ="2.5%", pad = 0.03)
        cbar = fig.colorbar(im, cax= cax, orientation = "horizontal", ticklocation = "top")
        cbar.set_label(r""+label)

    def add_points(ax, sim_readers):
        return
        for sim_reader in sim_readers:
            ax.plot(np.log10(sim_reader.get_field("var1")), np.log10(sim_reader.get_field("var2")), c = "w", marker = "x", ls = "")

    def plot(gasspy_database, h5database_path, sim_readers):
        h5database =  hp.File(h5database_path, "r")
        database_fields = [ field.decode() for field in h5database["database_fields"][:]]
        refinement_fields = [ field.decode() for field in h5database["refinement_fields"][:]]

        intensity1 = h5database["intensity"][:,0]
        intensity2 = h5database["line_intensity"][:,0]
        intensity3 = h5database["line_intensity"][:,1]
        h5database.close()
        resolution = 500

        v1 = np.linspace(-0.1,1.1,resolution)
        v2 = np.linspace(-0.1,1.1,resolution)

        var1, var2 = np.meshgrid(v1,v2)

        points = np.zeros((resolution**2,3))
        points[:,database_fields.index("cell_size")] = 0
        points[:,database_fields.index("var1")] = var1.ravel()
        points[:,database_fields.index("var2")] = var2.ravel()
        sim_reader = simulation_reader()
        sim_reader.set_field("cell_size", 10**points[:,database_fields.index("cell_size")])
        sim_reader.set_field("var1", 10**points[:,database_fields.index("var1")])
        sim_reader.set_field("var2", 10**points[:,database_fields.index("var2")])

        gasspy_ids = gasspy_database.get_gasspy_ids(sim_reader)
        node_ids = gasspy_database.get_node_ids(sim_reader)
        node_lrefine = gasspy_database.get_nodes_node_lrefine()[node_ids]

        # lrefine cmap
        lmin = np.min(node_lrefine[:, [refinement_fields.index("var1"),refinement_fields.index("var2")]])
        lmax = np.max(node_lrefine[:, [refinement_fields.index("var1"),refinement_fields.index("var2")]])

        lrefine_cmap = plt.get_cmap("gist_heat_r", lmax-lmin)
        lrefine_norm = mcol.Normalize(vmin = lmin, vmax = lmax)

        one_plot = 2.0

        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize = (3*one_plot,3*(one_plot*1.2)))
        im = axes[2,0].pcolormesh(var1, var2, node_lrefine[:, refinement_fields.index("var1")].reshape(resolution,resolution), cmap = lrefine_cmap, norm = lrefine_norm)
        add_colorbar(fig, axes[2,0], im, "var1_lrefine")
        add_points(axes[2,0], sim_readers)

        im = axes[2,1].pcolormesh(var1, var2, node_lrefine[:, refinement_fields.index("var2")].reshape(resolution,resolution), cmap = lrefine_cmap, norm = lrefine_norm)
        add_colorbar(fig, axes[2,1], im, "var2_lrefine")
        add_points(axes[2,1], sim_readers)

        # convergence in var1 only
        intensity = get_intensity1(var1,var2)
        im = axes[0,0].pcolormesh(var1, var2, intensity, cmap = "viridis")
        add_colorbar(fig, axes[0,0], im, "Intensity1")
        add_points(axes[0,0], sim_readers)
        binned_intensity = intensity1[gasspy_ids]
        binned_intensity[gasspy_ids==-1] = np.nan
        binned_intensity = binned_intensity.reshape(resolution,resolution)
        im = axes[1,0].pcolormesh(var1, var2, binned_intensity, cmap = "viridis")
        add_colorbar(fig, axes[1,0], im, "Binned Intensity1")
        add_points(axes[1,0], sim_readers)


        # convergence in var1 and var2
        intensity = get_intensity2(var1,var2)
        im = axes[0,1].pcolormesh(var1, var2, intensity, cmap = "viridis")
        add_colorbar(fig, axes[0,1], im, "Intensity2")
        add_points(axes[0,1], sim_readers)
        binned_intensity = intensity2[gasspy_ids]
        binned_intensity[gasspy_ids==-1] = np.nan
        binned_intensity = binned_intensity.reshape(resolution,resolution)
        im = axes[1,1].pcolormesh(var1, var2, binned_intensity, cmap = "viridis")
        add_colorbar(fig, axes[1,1], im, "Binned Intensity2")
        add_points(axes[1,1], sim_readers)

        # convergence in var2 only
        intensity = get_intensity3(var1,var2)        
        im = axes[0,2].pcolormesh(var1, var2, intensity, cmap = "viridis")
        add_colorbar(fig, axes[0,2], im, "Intensity3")
        add_points(axes[0,2], sim_readers)
        binned_intensity = intensity3[gasspy_ids]
        binned_intensity[gasspy_ids==-1] = np.nan
        binned_intensity = binned_intensity.reshape(resolution,resolution)
        im = axes[1,2].pcolormesh(var1, var2, binned_intensity, cmap = "viridis")
        add_colorbar(fig, axes[1,2], im, "Binned Intensity3")
        add_points(axes[1,2], sim_readers)

        axes[1,0].set_ylabel(r"var2")
        axes[2,1].set_xlabel(r"var1")

        for ax in axes.ravel():
            ax.set_aspect("equal")
        plt.subplots_adjust(top=0.93,
                            bottom=0.07,
                            left=0.15,
                            right=0.912,
                            hspace=0.355,   
                            wspace=0.145)
        plt.show()

        




    # Simple gasspy config with needed parameters
    gasspy_config = {
        "database_name" : "test_database.hdf5",
        "gasspy_modeldir" : "./test_database/",
        "database_fields" :[ 
              "cell_size",
              "var1",
              "var2"
        ],
        "discrete_fields": [
            "cell_size"
        ],
        "fields_lrefine": {
                "var1"  :  [2, 8],
                "var2"  :  [2, 8],
        },
        "fields_domain_limits":{
                "var1" : [0,1],
                "var2" : [0,1],
        },
        "refinement_fields": [
            "var1",
            "var2"
        ],

        "convergence_criterions" : {
            "spectra_ranges": {
                "range_1" : {
                    "energy_range" : [0.5,1.5],
                    "rtol" : 1e-1,
                    "max_surface_brightness" :  0.05
                }
            },
            "lines" : {
                "line_1" : {
                    "rtol" : 1e-1,
                    "atol" : 0.05,
                    "reference_continuum" : {
                        "energy_range": [0.5, 1.5],
                        "scale_factor": 0.1
                    }
                },
                "line_2" : {
                    "rtol" : 1e-2,
                    "atol" : 0.005
                }
            }
        }
    }   

    np.random.seed(1258021)
    # First snapshot: 100 cells
    # start in logspace for clarity
    N_cells = 100

    var1       = np.random.rand(N_cells)
    var2       = np.random.rand(N_cells)
    cell_size  = np.zeros(N_cells)
    
    snapshot1 = simulation_reader()
    snapshot1.set_field("var2", 10**var2)
    snapshot1.set_field("var1", 10**var1)
    snapshot1.set_field("cell_size"  , 10**cell_size)

    # Second snapshot: 100 More
    # start in logspace for clarity
    
    var1       = np.random.rand(N_cells)
    var2       = np.random.rand(N_cells)
    cell_size  = np.zeros(N_cells)

    snapshot2 = simulation_reader()
    snapshot2.set_field("var2", 10**var2)
    snapshot2.set_field("var1", 10**var1)
    snapshot2.set_field("cell_size"  , 10**cell_size)

    if os.path.exists("test_database"):
        shutil.rmtree("test_database")

    os.makedirs("test_database")
    #"""
    # First two in one go
    database_creator = GasspyDatabase(gasspy_config)
    database_creator.add_snapshot(snapshot1)
    database_creator.add_snapshot(snapshot2)

    # Populate cells
    populate("./test_database/test_database.hdf5")

    # refine and populate untill we are done
    while not database_creator.check_convergence():
        populate("./test_database/test_database.hdf5")

    database_creator.finalize()
    plot(database_creator, "test_database/test_database.hdf5", [snapshot1,snapshot2])
    del database_creator
    #"""
    # Next try to load and append 1 more snapshot that covers most of everything
    var1       = np.random.rand(N_cells)
    var2       = np.random.rand(N_cells)
    cell_size  = np.zeros(N_cells)

    resolution = 1024
    v1 = np.linspace(-0.1,1.1,resolution)
    v2 = np.linspace(-0.1,1.1,resolution)
    var1, var2 = np.meshgrid(v1,v2)
    var1 = var1.ravel()
    var2 = var2.ravel()
    var1 = np.append(var1,0.5)
    var2 = np.append(var2,0.51111111111111)
    cell_size  = np.zeros(var1.shape)

    snapshot3 = simulation_reader()
    snapshot3.set_field("var2", 10**var2)
    snapshot3.set_field("var1", 10**var1)
    snapshot3.set_field("cell_size"  , 10**cell_size)

    database_creator = GasspyDatabase(gasspy_config)

    database_creator.add_snapshot(snapshot1)
    database_creator.add_snapshot(snapshot2)
    database_creator.add_snapshot(snapshot3)
    database_creator.save_database()
    # Populate cells
    populate("./test_database/test_database.hdf5")

    # refine and populate untill we are done
    while not database_creator.check_convergence():
        database_creator.save_database()
        populate("./test_database/test_database.hdf5")


    database_creator.finalize() 
    
    plot(database_creator, "test_database/test_database.hdf5", [snapshot1,snapshot2,snapshot3])

    #shutil.rmtree("test_database")





