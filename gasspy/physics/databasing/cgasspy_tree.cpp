#include <iostream> 
#include <cmath>
#include <bits/stdc++.h>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <exception>
#include <stdexcept>
#include <chrono>
using namespace std;
namespace py = pybind11;

int n_database_fields;
int n_refinement_fields;
int *refinement_field_index;
int center_cell_neighbor_index;

long long finding_existing;
long long creating_new;
long long get_node_ms;


class GasspyException: public std::exception{
    private:
        std::string message = "";
    public:
        explicit GasspyException(const char *m): message{m} {}
        const char *what() const noexcept override {return message.c_str();}
};

class ModelNode{
    public:
        int gasspy_id;
        int node_id;
        int8_t is_leaf;
        int8_t is_root;
        int root_id;
        int16_t split_field;
        double *model_data;
        double *node_delta;
        int16_t *node_lrefine;
        int8_t is_required;
        int8_t has_neighbors;
        int8_t has_converged;
        int8_t in_tree;

        ModelNode* *child_nodes;
        ModelNode* *neighbor_nodes;

        // Node identification and comparisons
        int get_node_id(double* point);
        int find_node(double* point);
        int find_node(double* point, int16_t *node_lrefine);
        
        int nodes_identical(ModelNode* other_node);
        int nodes_identical(double *model_data, int16_t *node_lrefine);

        int nodes_same_data(ModelNode* other_node);
        int nodes_same_data(double *model_data);

        void debug_node();
        void debug_point(double* point);

        ModelNode(double *model_data, int16_t *node_lrefine){
            this->node_id = -1;
            this->gasspy_id = -1;
            this->model_data   = new double[n_database_fields];
            this->node_lrefine = new int16_t[n_database_fields];
            this->node_delta   = new double[n_database_fields];
            for(int ifield = 0; ifield < n_database_fields; ifield++){
                this->model_data[ifield] = model_data[ifield];
                this->node_lrefine[ifield] = node_lrefine[ifield];
                this->node_delta[ifield] = pow(2, -(double) node_lrefine[ifield]);
            }
            this->split_field = -1;
            this->is_leaf = 1;
            this->is_root = 0;
            this->root_id = -1;
            this->has_neighbors = 0;
            this->is_required = 0;
            this->has_converged = 0;

            this->child_nodes = new ModelNode*[3];
            for(int ichild = 0; ichild < 3; ichild++){
                this->child_nodes[ichild] = NULL;
            }
            this->neighbor_nodes = new ModelNode*[(int)pow(3,n_refinement_fields)];
            for(int ineigh = 0; ineigh < (int) pow(3,n_refinement_fields); ineigh++){
                this->neighbor_nodes[ineigh] = NULL;
            }
        }

        ModelNode(){
            this->node_id = -1;

        };
};



int ModelNode::get_node_id(double* point){ 
    int ifield;
    int child_index = 0;

    double point_field_value;
    double node_field_delta;
    double node_field_value;

    // Loop through all the fields. If we are outside the bound in any of them
    // return -1, otherwise determine if which child to ask next
    for(ifield = 0; ifield < n_database_fields; ifield++){
        point_field_value = point[ifield];
        node_field_delta = this->node_delta[ifield];
        node_field_value = this->model_data[ifield];

        if (fabs(point_field_value - node_field_value) > 0.5*node_field_delta){
            return -1;
        }
        if((this->is_leaf == 0) && (ifield == this->split_field)){
            if( (point_field_value - node_field_value) < -0.25*node_field_delta){
                child_index = 0;
            } else if( (point_field_value - node_field_value) > 0.25*node_field_delta){
                child_index = 2;
            } else {
                child_index = 1;
            }
        }

    }
    // if we get here then the point is in this node. If we're a leaf, or the wanted child does not exist return our 
    // node_id, otherwise pass along to the corresponding child
    if((this->is_leaf == 1 ) || (this->child_nodes[child_index] == NULL)){
        return this->node_id;
    } else {
        return this->child_nodes[child_index]->get_node_id(point);
    }
}

int ModelNode::find_node(double* point){ 
    int ifield;
    int look_at_child[3];
    if(this->is_leaf == 0 && this->split_field == -1){
        throw GasspyException("Node is not a leaf but has no split field...");
    }
    
    for(int ichild = 0; ichild<3;ichild++){
        look_at_child[ichild] = 0;
    }

    double point_field_value;
    double node_field_delta;
    double node_field_value;
    // Loop through all the fields. If we are outside the bound in any of them
    // return -1, otherwise determine if which child to ask next
    for(ifield = 0; ifield < n_database_fields; ifield++){

        point_field_value = point[ifield];
        node_field_delta  = this->node_delta[ifield];
        node_field_value  = this->model_data[ifield];
        if (fabs(point_field_value - node_field_value) - 1e-7*node_field_delta > 0.5*node_field_delta){
            return -1;
        }

        if((this->is_leaf == 0) && (ifield == this->split_field)){
            if( (point_field_value - node_field_value) - 1e-7*node_field_delta <= -0.25*node_field_delta){
                look_at_child[0] = 1;
            } 
            if( fabs(point_field_value - node_field_value) - 1e-7*node_field_delta <= 0.25*node_field_delta){
                look_at_child[1] = 1;
            }
            if( (point_field_value - node_field_value) + 1e-7*node_field_delta >= 0.25*node_field_delta){
                look_at_child[2] = 1;
            } 
        }

    }

    //Check if this node matches the data, otherwise look at children
    if(this->nodes_same_data(point)==1){
        return this->node_id;
    }

    // if we get here then the point is in this node. If we're a leaf and lrefines match, return our 
    // node_id, otherwise pass along to the corresponding child. If the child does not exists, them return false
    if(this->is_leaf == 1){
        // this is a leaf but did not match, therefore not found
        return -1;
    } else {
        // If the desired node is within the node boundaries, but is not this node, check children
        for(int ichild = 0; ichild < 3; ichild++){
            if((this->child_nodes[ichild] != NULL) && (look_at_child[ichild]==1)){
                int found_node_id = this->child_nodes[ichild]->find_node(point);
                if(found_node_id >=0){
                    return found_node_id;
                }
            }
        }
        // none of the children found something, just return false
        return -1;
    }
}

int ModelNode::find_node(double* point, int16_t *wanted_node_lrefine){    
    int ifield;
    int look_at_children = 0;
    int look_at_child[3];
    for(int ichild = 0; ichild<3;ichild++){
        look_at_child[ichild] = 1;
    }

    double point_field_value;
    double point_field_delta;
    double node_field_delta;
    double node_field_value;

    // Loop through all the fields. If we are outside the bound in any of them
    // return -1, otherwise determine if which child to ask next
    for(ifield = 0; ifield < n_database_fields; ifield++){
        if (this->node_lrefine[ifield] > wanted_node_lrefine[ifield]){
            // If we have refined to much, the requested node does not exist
            return -1;
        }

        point_field_value = point[ifield];
        point_field_delta = pow(2,-(double)wanted_node_lrefine[ifield]);
        node_field_delta  = this->node_delta[ifield];
        node_field_value  = this->model_data[ifield];
        if (fabs(point_field_value - node_field_value)-1e-7*point_field_delta > 0.5*node_field_delta){
            return -1;
        }
        if(this->node_lrefine[ifield]<wanted_node_lrefine[ifield]){
            look_at_children = 1;
        }
        if((this->is_leaf == 0) && (ifield == this->split_field) && (this->node_lrefine[ifield]<wanted_node_lrefine[ifield])){
            if( (point_field_value - node_field_value) - 1e-7*point_field_delta <= -0.25*node_field_delta){
                look_at_child[0] = 1;
            } 
            if( fabs(point_field_value - node_field_value) - 1e-7*point_field_delta <= 0.25*node_field_delta){
                look_at_child[1] = 1;
            }
            if( (point_field_value - node_field_value) + 1e-7*point_field_delta >= 0.25*node_field_delta){
                look_at_child[2] = 1;
            } 
        }

    }
   
    // if we get here then the point is in this node. If we're a leaf and lrefines match, return our 
    // node_id, otherwise pass along to the corresponding child. If the child does not exists, them return false
    if(this->is_leaf == 1){
        // We want higher lrefines, but this is a leaf, the node does not exist
        if(look_at_children == 1){
            return -1;
        }

        // otherwise check if identical
        if(this->nodes_identical(point, wanted_node_lrefine)==1){
            return this->node_id;
        } 

        return -1;

    } else {
        // We are at the corrrect refinement level, check here
        if(look_at_children == 0){
            if(this->nodes_identical(point, wanted_node_lrefine)==1){
                return this->node_id;
            }
            return -1;
        }

        // otherwise check children
        for(int ichild = 0; ichild < 3; ichild++){
            if((this->child_nodes[ichild] != NULL) && (look_at_child[ichild]==1)){
                int found_node_id = this->child_nodes[ichild]->find_node(point, wanted_node_lrefine);
                if(found_node_id >=0){
                    return found_node_id;
                }
            }
        }
        // none of the children found something, just return false
        return -1;
    }
}


int ModelNode::nodes_identical(ModelNode* other_node){
    // checks if a node is identical to another, This is ONLY true if they have the same 
    int ifield;
    double max_delta;
    for(ifield = 0; ifield < n_database_fields; ifield++){
        if(this->node_lrefine[ifield]!= other_node->node_lrefine[ifield]){
            return -1;
        }
        //everything is based on powers of two, so a given number is only given to the lrefine'th decimal
        max_delta = 1e-7*fmin(this->node_delta[ifield], other_node->node_delta[ifield]);
        if(fabs(this->model_data[ifield]-other_node->model_data[ifield])>max_delta){
            return -1;
        }
    }
    return 1;
}

int ModelNode::nodes_identical(double* wanted_model_data, int16_t* wanted_node_lrefine){
    // checks if a node is identical to another, This is ONLY true if they have the same refinement level AND data
    int ifield;
    double max_delta;
    for(ifield = 0; ifield < n_database_fields; ifield++){
        if(this->node_lrefine[ifield] != wanted_node_lrefine[ifield]){
            return -1;
        }
        //everything is based on powers of two, so a given number is only given to the lrefine'th decimal
        max_delta = 1e-7*pow(2, -(double)wanted_node_lrefine[ifield]); //.1*pow(10,std::min(0,-std::max((int)this->node_lrefine[ifield], (int)wanted_node_lrefine[ifield])));
        if(fabs(this->model_data[ifield]-wanted_model_data[ifield])>max_delta){
            return -1;
        }
    }
    return 1;
}

int ModelNode::nodes_same_data(ModelNode* other_node){
    // checks if a node is identical to another, This is ONLY true if they have the same 
    int ifield;
    double max_delta;
    for(ifield = 0; ifield < n_database_fields; ifield++){
        if(this->node_lrefine[ifield]!= other_node->node_lrefine[ifield]){
            return -1;
        }
        //everything is based on powers of two, so a given number is only given to the lrefine'th decimal
        max_delta = 1e-7*other_node->node_delta[ifield]; //.1*pow(10,std::min(0,-std::max((int)this->node_lrefine[ifield], (int)wanted_node_lrefine[ifield])));

        if(fabs(this->model_data[ifield]-other_node->model_data[ifield])>max_delta){
            return -1;
        }
    }
    return 1;
}

int ModelNode::nodes_same_data(double* wanted_model_data){
    // checks if a node has identicall data as model_data
    int ifield;
    double max_delta;
    for(ifield = 0; ifield < n_database_fields; ifield++){

        //everything is based on powers of two, so a given number is only given to the lrefine'th decimal
        max_delta =  1e-10*this->node_delta[ifield];
        if(fabs(this->model_data[ifield]-wanted_model_data[ifield])>max_delta){
            return -1;
        }
    }
    return 1;
}

void ModelNode::debug_node(){
    py::print("---------------------------");
    py::print("Node id", this->node_id);
    py::print("Model_data, node_lrefine:");
    for(int ifield = 0; ifield < n_database_fields; ifield++){
        py::print("\t",this->model_data[ifield],", ",this->node_lrefine[ifield]);
    }
    py::print("Gasspy_id", this->gasspy_id);
    py::print("is_leaf", this->is_leaf);
    py::print("is_root", this->is_root);
    py::print("root_id", this->root_id);
    py::print("has_neighbors", this->has_neighbors);
    py::print("split_field", this->split_field);
    py::print("is_required", this->is_required);

    py::print("Child node ids:");
    for(int ichild = 0; ichild<3; ichild++){
        if(this->child_nodes[ichild]!=NULL){
            py::print("\t",this->child_nodes[ichild]->node_id);
        } else {
            py::print("\t",-1);
        }
    }

    py::print("Neighbor node ids:");
    for(int ineigh = 0; ineigh < pow(3,n_refinement_fields); ineigh++){
        if(this->neighbor_nodes[ineigh]!=NULL){
            py::print("\t",this->neighbor_nodes[ineigh]->node_id);
        } else {
            py::print("\t",-1);
        }
    }
    py::print("---------------------------");
}

void ModelNode::debug_point(double* point){ 
    int ifield;
    int child_index = -1;

    double point_field_value;
    double node_field_delta;
    double node_field_value;
    py::print("Entering node ", this->node_id);
    this->debug_node();

    // Loop through all the fields. If we are outside the bound in any of them
    // return -1, otherwise determine if which child to ask next
    py::print("\tChecking fields");
    for(ifield = 0; ifield < n_database_fields; ifield++){
        point_field_value = point[ifield];
        node_field_delta = this->node_delta[ifield];
        node_field_value = this->model_data[ifield];
        py::print("\t",ifield," ",node_field_value-node_field_delta, point_field_value, node_field_value+node_field_delta);
        if (fabs(point_field_value - node_field_value) > 0.5*node_field_delta){
            py::print("Point failed for field ", ifield," at node ",this->node_id);
            return;
        }
        if((this->is_leaf == 0) && (ifield == this->split_field)){
            if( (point_field_value - node_field_value) < -0.25*node_field_delta){
                child_index = 0;
            } else if( (point_field_value - node_field_value) > 0.25*node_field_delta){
                child_index = 2;
            } else {
                child_index = 1;
            }
        }

    }
    // if we get here then the point is in this node. If we're a leaf, or the wanted child does not exist return our 
    // node_id, otherwise pass along to the corresponding child
    if((this->is_leaf == 1 ) || (this->child_nodes[child_index] == NULL)){
        py::print("Point successful for node ", this->node_id);
    } else {
        if(child_index==-1){
        
            py::print("?????????????");
            return;
        }
        py::print("\tEntering child ", child_index, " corresponding to node ", this->child_nodes[child_index]->node_id);
        this->child_nodes[child_index]->debug_point(point);
    }
}



int __shift_recursive__(int ishift, int i_refinement_field, vector<vector<double>> &neighbor_model_data, int16_t *node_lrefine, vector<double> &total_shift){
    /*
        Recursive function to find all neighbors
        For each interpolated field:
            1 Determine the shift based on the compressio_ratio
            2 Add to its entry of the shift
            3 recursive call to the next field
        Once all fields have added their shift: Add shift to original, progress neighbor count "ishift" by one
    */
    if(i_refinement_field == n_refinement_fields){
        // we are done, add shift and progress ishift
        for(int ifield = 0; ifield < n_database_fields; ifield++){
            neighbor_model_data[ishift][ifield] += total_shift[ifield];
        }
        return ishift + 1;
    }

    int ifield = refinement_field_index[i_refinement_field];
    // Loop over left, centre and right, determine shift and call for the next field
    double shift = pow(2, -(float)node_lrefine[ifield]);
    for(int local_shift = -1; local_shift < 2; local_shift++){
        total_shift[ifield] = local_shift*shift;
        ishift = __shift_recursive__(ishift, i_refinement_field + 1, neighbor_model_data, node_lrefine, total_shift);
    }
    return ishift;
}


struct listnode{
    ModelNode node;
    listnode* next;
    listnode* previous;
    int inode;
};

class NodeList {
    private:
        listnode *head, *tail;
        int list_has_changed;
    public:
        int nnodes;
        NodeList(){
            this->head = NULL;
            this->tail = NULL;
            this->nnodes = 0;
            list_has_changed = 0;
        }


        listnode* get_head(){
            return this->head;
        }

        listnode* get_tail(){
            return this->tail;
        }

        ModelNode* add_node(double* coords, int16_t *node_lrefine){
            listnode *new_entry = new listnode;
            new_entry->node = ModelNode(coords, node_lrefine);
            new_entry->next = NULL;
            new_entry->inode = this->nnodes;
            new_entry->node.node_id = this->nnodes;    
            if (this->head == NULL){
                new_entry->previous = NULL;
                this->head = new_entry;
                this->tail = new_entry;
            } else {
                new_entry->previous = this->tail;
                this->tail->next = new_entry;
                this->tail = this->tail->next;
            }
            this->nnodes += 1;
            return &(this->tail->node);
        }

        ModelNode* get_node(int node_id){
            int current_node_id;
            listnode* current_node = head;
            for(current_node_id = 0; current_node_id < node_id; current_node_id++){
                current_node = current_node->next;
            }
            return &current_node->node;
        }

        void delete_node(int node_id){
            // find the node that we want to delete
            int current_node_id;
            listnode* current_node = head;
            for(current_node_id = 0; current_node_id < node_id; current_node_id++){
                current_node = current_node->next;
            }

            // make sure heads and tails of the list are still accurate
            if(node_id == 0){
                this->head = current_node->next;
            }
            if(node_id == this->nnodes-1){
                this->tail = current_node->previous;
            }

            // drops the reference for the node in the linked list
            current_node->previous = current_node->next;
            listnode* deleted_node = current_node;
            delete deleted_node;
            deleted_node = NULL;

            // updates the ids of the nodes that were after the deleted node in the list
            current_node_id = node_id;
            current_node = current_node->next;
            while(current_node!=NULL){
                current_node->node.node_id = current_node_id;
                current_node = current_node->next;
                current_node_id+=1;            
            }  
            this->nnodes = current_node_id;
        }


};


class GasspyTree {
    private:
        std::vector<int> root_ids;
        std::vector<ModelNode*> root_nodes;
        std::vector<int> required_leaf_ids;
        std::vector<std::vector<double>> gasspy_model_data; 
        int16_t *fields_lrefine_min;
        int16_t *fields_lrefine_max;
    public:
        NodeList all_nodes;
        void set_required_leaf_ids();
        void get_n_leafs();
        int find_node(double *coords, int16_t *node_lrefine);
        int find_node(double *coords);
        int get_node_id(double *coords);

        int add_point(double *point);
        void refine_node(int16_t *new_node_lrefine, ModelNode* current_node);
        int find_node_neighbors(ModelNode* current_node);
        void set_neighbors();
        void get_node_data();


        GasspyTree(int _n_database_fields, int _n_refinement_fields, int *_refinement_field_index){
            this->all_nodes = NodeList();
            n_database_fields = _n_database_fields;
            n_refinement_fields = _n_refinement_fields;
            refinement_field_index = new int[n_refinement_fields];
            center_cell_neighbor_index = 0;
            for(int ifield = 0; ifield<n_refinement_fields; ifield++){
                refinement_field_index[ifield] = _refinement_field_index[ifield];
                center_cell_neighbor_index += (int)round(pow(3, n_refinement_fields - 1 - ifield));
            }      

            this->fields_lrefine_min = new int16_t[n_refinement_fields];    
            this->fields_lrefine_max = new int16_t[n_refinement_fields];    
        }
        GasspyTree(int _n_database_fields, int _n_refinement_fields, py::array_t<int> _refinement_field_index){
            this->all_nodes = NodeList();
            n_database_fields = _n_database_fields;
            n_refinement_fields = _n_refinement_fields;
            refinement_field_index = new int[n_refinement_fields];
            auto _refinement_field_index_ra = _refinement_field_index.mutable_unchecked();
            center_cell_neighbor_index = 0;
            for(int ifield = 0; ifield<n_refinement_fields; ifield++){
                refinement_field_index[ifield] = _refinement_field_index_ra(ifield);
                center_cell_neighbor_index += (int)round(pow(3, n_refinement_fields - 1 - ifield));
            }   
            py::print(center_cell_neighbor_index);
            this->fields_lrefine_min = new int16_t[n_refinement_fields];    
            this->fields_lrefine_max = new int16_t[n_refinement_fields];           
        }

        // Python interface functions
        py::array_t<int> add_points(py::array_t<double> points, py::array_t<int> previous_node_ids);
        void refine_nodes(py::array_t<int> node_ids, py::array_t<int16_t> wanted_node_lrefine);
        void set_has_converged(py::array_t<int> node_ids);
        void set_unique_gasspy_models();

        // getters
        py::array_t<int> get_node_ids(py::array_t<double> points, py::array_t<int> previous_node_ids);
        py::array_t<int> get_gasspy_ids(py::array_t<double> points, py::array_t<int> previous_node_ids);
        
        py::array_t<double> get_gasspy_model_data();
        py::array_t<int> get_leaf_and_neighbor_gasspy_ids();
        py::array_t<double> get_nodes_model_data();
        py::array_t<int16_t> get_nodes_node_lrefine();
        py::array_t<int> get_nodes_gasspy_ids();
        py::array_t<int8_t> get_nodes_is_root();
        py::array_t<int8_t> get_nodes_is_leaf();
        py::array_t<int8_t> get_nodes_is_required();
        py::array_t<int8_t> get_nodes_has_converged();
        py::array_t<int16_t> get_nodes_split_field();
        py::array_t<int> get_nodes_child_node_ids();
        py::array_t<int> get_nodes_neighbor_node_ids();

        // setters
        void set_lrefine_limits(py::array_t<int16_t> lrefine_limits);
        void set_gasspy_model_data(py::array_t<double> gasspy_model_data);
        void initialize_nodes(py::array_t<double> new_model_data, py::array_t<int16_t> new_node_lrefine);
        void set_nodes_gasspy_ids(py::array_t<int> gasspy_ids);
        void set_nodes_is_root(py::array_t<int8_t> is_root);
        void set_nodes_is_leaf(py::array_t<int8_t> is_leaf);
        void set_nodes_is_required(py::array_t<int8_t> is_required);
        void set_nodes_has_converged(py::array_t<int8_t> has_converged);
        void set_nodes_split_field(py::array_t<int16_t> split_field);
        void set_nodes_child_node_ids(py::array_t<int> child_node_ids);
        void set_nodes_neighbor_node_ids(py::array_t<int> neighbor_node_ids);

        // Debug
        void debug_nodes(py::array_t<int> node_ids);


};


int GasspyTree::get_node_id(double *point){
    // Finds if there's a node with the same coordinates
    int node_id;
    ModelNode* root_node;
    // Loop through all roots and check each one
    for(size_t iroot = 0; iroot < this->root_nodes.size(); iroot++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }

        auto start = std::chrono::high_resolution_clock::now();
        root_node = this->root_nodes.at(iroot);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        get_node_ms += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        int inside = 1;
        for(int ifield = 0; ifield < n_database_fields; ifield++){
            double delta = root_node->node_delta[ifield];
            double value = root_node->model_data[ifield];
            if(fabs(value - point[ifield])>0.5*delta){
               inside = 0;
               break;
            }
        }
        if(inside == 0){
            continue;
        }
        node_id = root_node->get_node_id(point);

        if(node_id < 0){
            root_node->debug_point(point);
            throw GasspyException("Couldn't find matching node even when root contains point.. this is bad..");
        }
        // Found one? use it!
        return node_id;
    }
    return -1;
}

int GasspyTree::find_node(double *coords){
    // Finds if there's a node with the same coordinates
    ModelNode* root_node;
    int node_id;
    // Loop through all roots and check each one
    for(size_t iroot = 0; iroot < this->root_nodes.size(); iroot++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        root_node = this->root_nodes.at(iroot);
        node_id = root_node->find_node(coords);
        if(node_id >= 0){
            return node_id;
        }
    }
    return -1;
}


int GasspyTree::find_node(double *coords, int16_t *node_lrefine){
    // loops through and finds a node with the same coordinates and refinement levels (eg. identical)
    ModelNode* root_node;
    int node_id;
    for(size_t iroot = 0; iroot < this->root_nodes.size(); iroot++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        root_node = this->root_nodes.at(iroot);
        node_id = root_node->find_node(coords, node_lrefine);

        if(node_id >= 0){
            return node_id;
        }
    }
    return -1;
}


int GasspyTree::add_point(double *point){
    int node_id;
    int ifield;

    // See if we can match this point to the existing nodes
    auto start =  std::chrono::high_resolution_clock::now();
    node_id = this->get_node_id(point);
    if(node_id >= 0){
        this->all_nodes.get_node(node_id)->is_required=1;
        return node_id;
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    finding_existing += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();


    start =  std::chrono::high_resolution_clock::now();

    // otherwise start a new root with this point
    double coords[n_database_fields];
    double field_delta;
    int field_nsteps;

    // Find closest root coordinates
    for(ifield = 0; ifield < n_database_fields; ifield++){
        field_delta = pow(2., -(float)this->fields_lrefine_min[ifield]);
        field_nsteps = (int)(point[ifield]/field_delta);
        if(point[ifield] > (field_nsteps+0.5)*field_delta){
            field_nsteps += 1;
        } else if (point[ifield] < (field_nsteps-0.5)*field_delta) {
            field_nsteps -= 1;
        }
        coords[ifield] = field_nsteps*field_delta;
    }

    ModelNode *current_model_node = this->all_nodes.add_node(coords, this->fields_lrefine_min);
    node_id = current_model_node->node_id;
    this->root_ids.push_back(node_id);
    this->root_nodes.push_back(current_model_node);

    current_model_node->is_required=1;
    current_model_node->is_leaf=1;
    current_model_node->is_root=1;
    current_model_node->root_id=this->root_nodes.size()-1;
    elapsed = std::chrono::high_resolution_clock::now() - start;

    creating_new += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return node_id;
}

void GasspyTree::set_neighbors(){ 
    int inode;
    listnode* current_node = this->all_nodes.get_head();
    for(inode = 0; inode < this->all_nodes.nnodes; inode++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        if((current_node->node.is_required==1)&&(current_node->node.has_neighbors!=1)){
            this->find_node_neighbors(&current_node->node);
        }
        current_node = current_node->next;
    }
}

void GasspyTree::set_unique_gasspy_models(){
    int inode;
    size_t imodel;
    int ifield;
    int found = 0;
    vector <double> current_model;
    double max_delta;
    listnode* current_node = this->all_nodes.get_head();
    for(inode = 0; inode < (all_nodes.nnodes); inode++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        vector <double> node_model(n_database_fields);
        for(ifield = 0; ifield < n_database_fields; ifield++){
            node_model[ifield] = current_node->node.model_data[ifield];
        }

        // We have already set the gasspy_id for this node, no need to do so again
        if(current_node->node.gasspy_id >= 0){
            current_node = current_node->next;
            continue;
        }
        // Loop through all known gasspy_models and see if this node already has a matching one 
        for(imodel = 0; imodel < this->gasspy_model_data.size(); imodel++){
            current_model = this->gasspy_model_data.at(imodel);
            found = 1;
            for(ifield = 0; ifield < n_database_fields; ifield++){
                max_delta = 1e-7*0.5*pow(2,-(double)current_node->node.node_lrefine[ifield]);
                if(fabs(node_model.at(ifield)-current_model.at(ifield))>max_delta){
                    found = 0;
                    break;
                }            
            }
            if(found == 1){
                current_node->node.gasspy_id = imodel;
                break;
            }
        }
        // If no matching model was found, add a new one
        if(found == 0){
            this->gasspy_model_data.push_back(node_model);
            current_node->node.gasspy_id = this->gasspy_model_data.size()-1;
        }        
        current_node = current_node->next;

    }
    return;
}


void GasspyTree::set_required_leaf_ids(){
    if(!this->required_leaf_ids.empty()){
        this->required_leaf_ids.clear();
    }
    int node_id;
    listnode* current_node = this->all_nodes.get_head();
    for(node_id = 0; node_id < all_nodes.nnodes; node_id++){
        if((current_node->node.is_leaf==1)&&(current_node->node.is_required==1)){
            this->required_leaf_ids.push_back(node_id);
        }
        current_node = current_node->next;
    }

    return;
}

void GasspyTree::refine_node(int16_t* new_node_lrefine, ModelNode* current_node){
    int ifield, ishift;
    int node_id;
    double* child_model_data;
    int16_t *child_node_lrefine;
    double node_field_delta;
    double model_field_data;
    if(current_node->is_leaf == 0){
        // if this node has already split, just call for its children
        for(ishift = 0; ishift < 3; ishift ++){
            this->refine_node(new_node_lrefine, current_node->child_nodes[ishift]);
        }
        return;
    }
    current_node->split_field = -1;
    // Determine which refinement level we need to split. If multiple start with first
    for(ifield = 0; ifield < n_database_fields; ifield++){
        if(new_node_lrefine[ifield] > current_node->node_lrefine[ifield]){
            current_node->split_field = ifield;   
            break;     
        }
    }

    // This node does not need to be refined, even if it was told to...
    if(current_node->split_field == -1){
        return;
    }
    // This node will refine.
    // We therefore set it as not a leaf and not be required
    current_node->is_leaf = 0;   
    current_node->is_required = 0;

    // Copy over model data into child buffer
    child_model_data = new double [n_database_fields];
    child_node_lrefine = new int16_t [n_database_fields];
    for(ifield = 0; ifield < n_database_fields; ifield++){
        child_model_data[ifield] = current_node->model_data[ifield];
        child_node_lrefine[ifield] = current_node->node_lrefine[ifield];
    }

    // advance the child lrefine
    child_node_lrefine[current_node->split_field] += 1;

    // Shift the coordinate and split into three new nodes
    model_field_data = current_node->model_data[current_node->split_field];
    node_field_delta = 0.5*pow(2,-(double)current_node->node_lrefine[current_node->split_field]);
    for(ishift = 0; ishift < 3; ishift++){
        // Set shifted value of child data
        child_model_data[current_node->split_field] = model_field_data + (ishift-1)* node_field_delta ;

        // check if this exact node already exists in the tree
        node_id = this->find_node(child_model_data, child_node_lrefine);

        // if no matching node was found, create one
        if(node_id>=0){
            // if a match was found, just set to this one
            current_node->child_nodes[ishift] = this->all_nodes.get_node(node_id);
            // also make sure that this no longer registers as a root node
            if(current_node->child_nodes[ishift]->is_root == 1){
                current_node->child_nodes[ishift]->is_root = 0;
                this->root_ids.erase(this->root_ids.begin()+current_node->child_nodes[ishift]->root_id);
                this->root_nodes.erase(this->root_nodes.begin()+current_node->child_nodes[ishift]->root_id);
            }
        } else {

            // add node to list of nodes
            current_node->child_nodes[ishift]=this->all_nodes.add_node(child_model_data, child_node_lrefine);
            // advance refinement level of selected field
            current_node->child_nodes[ishift]->is_leaf=1;
            current_node->child_nodes[ishift]->is_root=0;
        }
        // recursive call to this new node
        this->refine_node(new_node_lrefine, current_node->child_nodes[ishift]);
    }
    return;
}

int GasspyTree::find_node_neighbors(ModelNode* current_node){
    int ifield;
    int ineigh;
    int inode;
    vector<vector<double>> neighbor_model_data(pow(3,n_refinement_fields),vector<double>(n_database_fields));
    double coords[n_database_fields];

    // First determine the position of all neighbors
    for(ineigh = 0; ineigh < pow(3, n_refinement_fields); ineigh++){
        for(ifield = 0; ifield < n_database_fields; ifield++){            
            neighbor_model_data[ineigh][ifield] = current_node->model_data[ifield];
        }
    }
    vector<double> total_shift(n_database_fields);
    int ishift = __shift_recursive__(0, 0, neighbor_model_data, current_node->node_lrefine, total_shift);

    // for each neighbor node
    for(ineigh=0; ineigh<pow(3,n_refinement_fields); ineigh++){
        // if this is the center cell, just set to self
        if(ineigh == center_cell_neighbor_index){
            current_node->neighbor_nodes[ineigh] = current_node;
            continue;
        }

        for(ifield=0; ifield < n_database_fields; ifield++){
            coords[ifield] = neighbor_model_data[ineigh][ifield];
        }

        // first check if we already have a node with these coordinates
        inode = this->find_node(coords);

        // if so set to that node
        if(inode >= 0){
            current_node->neighbor_nodes[ineigh] = this->all_nodes.get_node(inode);
        } else {
            // otherwise create a new node and add node to list of nodes
            current_node->neighbor_nodes[ineigh] = this->all_nodes.add_node(coords, current_node->node_lrefine);
            current_node->neighbor_nodes[ineigh]->is_root = 0;
        }
    }
    current_node->has_neighbors = 1;
    return 1;
}



// Python interface functions
py::array_t<int> GasspyTree::add_points(py::array_t<double> points, py::array_t<int> previous_node_ids){
    int node_id;
    auto point_ra = points.mutable_unchecked();
    auto previous_node_ids_ra = previous_node_ids.mutable_unchecked();
    py::array_t<int> _tmp_node_ids(points.shape(0));
    auto _tmp_node_ids_ra = _tmp_node_ids.mutable_unchecked();
    double point[n_database_fields];

    finding_existing = 0;
    creating_new = 0;
    get_node_ms = 0;
    for(size_t ipoint = 0; ipoint < (size_t) points.shape(0); ipoint++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
            point[ifield] = point_ra(ipoint, ifield);
        }

        node_id = previous_node_ids_ra[ipoint];
        if(node_id >= 0){
            // if this point already had a node associated to it, the point should either belong to the node 
            // or one of its children
            if(node_id < this->all_nodes.nnodes){
                ModelNode* previous_node = this->all_nodes.get_node(previous_node_ids_ra[ipoint]);
                node_id = previous_node->get_node_id(point); 
                this->all_nodes.get_node(node_id)->is_required=1;
            } else {
                node_id = -1;
            }
        } 
        if(node_id < 0) {
            // Otherwise find a node for the point by looking through the tree
            node_id = this->add_point(point);
        }
        _tmp_node_ids_ra(ipoint) = node_id;
        if(ipoint%10000 == 0){
            py::print("ipoint ", ipoint, "nroots", this->root_nodes.size(), py::arg("flush") = true);
            py::print("Finding existing : ", finding_existing, "mu", py::arg("flush")= true);
            py::print("Creating new     : ", creating_new, "mu", py::arg("flush") = true);
            py::print("get_node         : ", get_node_ms, "mu", py::arg("flush") = true);
        }
    }
    py::print("nroots", this->root_nodes.size());
    py::print("Finding existing : ", finding_existing, "mu");
    py::print("Creating new     : ", creating_new, "mu");
    return _tmp_node_ids;
}

void GasspyTree::refine_nodes(py::array_t<int> node_ids, py::array_t<int16_t> wanted_node_lrefines){
    auto node_ids_ra = node_ids.mutable_unchecked();
    auto wanted_node_lrefines_ra = wanted_node_lrefines.mutable_unchecked();
    int node_id;
    int16_t wanted_node_lrefine[n_database_fields];

    for(size_t inode = 0; inode < (size_t) node_ids.size(); inode++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        node_id = node_ids_ra(inode);
        for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
            wanted_node_lrefine[ifield] = wanted_node_lrefines_ra(inode, ifield);
        }
        this->refine_node(wanted_node_lrefine, this->all_nodes.get_node(node_id));
    }
}

void GasspyTree::set_has_converged(py::array_t<int> node_ids){
    auto node_ids_ra = node_ids.mutable_unchecked();
    int node_id;
    for(size_t inode = 0; inode < (size_t) node_ids.size(); inode++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        node_id = node_ids_ra(inode);

        this->all_nodes.get_node(node_id)->has_converged = 1;
    }   
}



// getters
py::array_t<int> GasspyTree::get_node_ids(py::array_t<double> points, py::array_t<int> previous_node_ids){
    py::array_t<int> _tmp_node_ids(points.shape(0));
    auto _tmp_node_ids_ra = _tmp_node_ids.mutable_unchecked();
    auto points_ra = points.mutable_unchecked();
    auto previous_node_ids_ra = previous_node_ids.mutable_unchecked();

    double point[n_database_fields];
    int node_id;
    int warn= 0;
    for(size_t ipoint = 0; ipoint < (size_t) points.shape(0); ipoint++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
            point[ifield] = points_ra(ipoint, ifield);
        }
        node_id = previous_node_ids_ra[ipoint];
        if(node_id >= 0){
            // if this point already had a node associated to it, the point should either belong to the node 
            // or one of its children
            if(node_id < this->all_nodes.nnodes){
                ModelNode* previous_node = this->all_nodes.get_node(previous_node_ids_ra[ipoint]);
                node_id = previous_node->get_node_id(point);
            } else{ 
                node_id = -1;
            }
        }  
        if(node_id < 0){
            // If there's no prevoud node, or we couldnt match to the old node, find a node for the point by looking through the tree
            node_id = this->get_node_id(point);
        }

        if(node_id < 0){
            warn = 1;
        }
        _tmp_node_ids_ra(ipoint) = node_id;
    }
    if(warn){
        py::print("\nWarning: Some points could not be matched to models.\n\t If you havent added this snapshot before, please do so.\n\t If this snapshot has been added before, something has gone wrong and you should blame loke\n");
    }
    return _tmp_node_ids;
}
py::array_t<int> GasspyTree::get_gasspy_ids(py::array_t<double> points, py::array_t<int> previous_node_ids){
    py::array_t<int> _tmp_gasspy_ids(points.shape(0));
    auto _tmp_gasspy_ids_ra = _tmp_gasspy_ids.mutable_unchecked();
    auto points_ra = points.mutable_unchecked();
    auto previous_node_ids_ra = previous_node_ids.mutable_unchecked();

    double point[n_database_fields];
    int node_id;
    int warn= 0;
    for(size_t ipoint = 0; ipoint < (size_t) points.shape(0); ipoint++){
        for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
            point[ifield] = points_ra(ipoint, ifield);
        }
        node_id = previous_node_ids_ra[ipoint];
        if(node_id >= 0){
            // if this point already had a node associated to it, the point should either belong to the node 
            // or one of its children
            if(node_id < this->all_nodes.nnodes){
                ModelNode* previous_node = this->all_nodes.get_node(previous_node_ids_ra[ipoint]);
                node_id = previous_node->get_node_id(point);
            } else {
                node_id = -1;
            }
        } 
        if (node_id < 0) {
            // Otherwise find a node for the point by looking through the tree
            node_id = this->get_node_id(point);
        }
        if(node_id < 0){
            warn = 1;
        }
        _tmp_gasspy_ids_ra(ipoint) = this->all_nodes.get_node(node_id)->gasspy_id;
    }
    if(warn){
        py::print("\nWarning: Some points could not be matched to models.\n\t If you havent added this snapshot before, please do so.\n\t If this snapshot has been added before, something has gone wrong and you should blame loke\n");
    }
    return _tmp_gasspy_ids;
}
        


py::array_t<double> GasspyTree::get_gasspy_model_data(){
    // Allocate an python array
    py::array_t<double> _tmp_gasspy_model_data( {(size_t)this->gasspy_model_data.size(), (size_t)n_database_fields} );
    
    // required to view?
    auto ra = _tmp_gasspy_model_data.mutable_unchecked();

    // copy over data
    for(size_t gasspy_id = 0; gasspy_id < (size_t) this->gasspy_model_data.size(); gasspy_id ++){
        for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
            ra(gasspy_id, ifield) = this->gasspy_model_data[gasspy_id][ifield];
        }
    }
    return _tmp_gasspy_model_data;
}

py::array_t<int> GasspyTree::get_leaf_and_neighbor_gasspy_ids(){
    // TODO
    py::array_t<int> _tmp_nodes_model_data( {(size_t)this->all_nodes.nnodes, (size_t)n_database_fields} );
    return _tmp_nodes_model_data;
}


py::array_t<double> GasspyTree::get_nodes_model_data(){
    // Allocate an python array
    py::array_t<double> _tmp_nodes_model_data( {(size_t)this->all_nodes.nnodes, (size_t)n_database_fields} );
    
    // required to view?
    auto ra = _tmp_nodes_model_data.mutable_unchecked();

    // copy over data
    listnode *current_node = this->all_nodes.get_head();
    for(size_t node_id = 0; node_id < (size_t) this->all_nodes.nnodes; node_id++){
        for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
            ra(node_id, ifield) = current_node->node.model_data[ifield];
        }
        current_node = current_node->next;
    }
    return _tmp_nodes_model_data;
}

py::array_t<int16_t> GasspyTree::get_nodes_node_lrefine(){
    // Allocate an python array
    py::array_t<int16_t> _tmp_nodes_node_lrefine( {(size_t)this->all_nodes.nnodes, (size_t)n_database_fields} );
    
    // required to view?
    auto ra = _tmp_nodes_node_lrefine.mutable_unchecked();

    // copy over data
    listnode *current_node = this->all_nodes.get_head();
    for(size_t node_id = 0; node_id < (size_t) this->all_nodes.nnodes; node_id++){
        for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
            ra(node_id, ifield) = current_node->node.node_lrefine[ifield];
        }
        current_node = current_node->next;
    }
    return _tmp_nodes_node_lrefine;
}

py::array_t<int> GasspyTree::get_nodes_gasspy_ids(){
    // Allocate an python array
    py::array_t<int> _tmp_nodes_gasspy_ids((size_t)this->all_nodes.nnodes);
    
    // required to view?
    auto ra = _tmp_nodes_gasspy_ids.mutable_unchecked();

    // copy over data
    listnode *current_node = this->all_nodes.get_head();
    for(size_t node_id = 0; node_id < (size_t) this->all_nodes.nnodes; node_id++){
        ra(node_id) = current_node->node.gasspy_id;
        current_node = current_node->next;
    }
    return _tmp_nodes_gasspy_ids;
}
py::array_t<int8_t> GasspyTree::get_nodes_is_root(){
    // Allocate an python array
    py::array_t<int8_t> _tmp_nodes_is_root((size_t)this->all_nodes.nnodes);
    
    // required to view?
    auto ra = _tmp_nodes_is_root.mutable_unchecked();

    // copy over data
    listnode *current_node = this->all_nodes.get_head();
    for(size_t node_id = 0; node_id < (size_t) this->all_nodes.nnodes; node_id++){
        ra(node_id) = current_node->node.is_root;
        current_node = current_node->next;
    }
    return _tmp_nodes_is_root;
}

py::array_t<int8_t> GasspyTree::get_nodes_is_leaf(){
    // Allocate an python array
    py::array_t<int8_t> _tmp_nodes_is_leaf((size_t)this->all_nodes.nnodes);
    
    // required to view?
    auto ra = _tmp_nodes_is_leaf.mutable_unchecked();

    // copy over data
    listnode *current_node = this->all_nodes.get_head();
    for(size_t node_id = 0; node_id < (size_t) this->all_nodes.nnodes; node_id++){
        ra(node_id) = current_node->node.is_leaf;
        current_node = current_node->next;
    }
    return _tmp_nodes_is_leaf;
}
py::array_t<int8_t> GasspyTree::get_nodes_is_required(){
    // Allocate an python array
    py::array_t<int8_t> _tmp_nodes_is_required((size_t)this->all_nodes.nnodes);
    
    // required to view?
    auto ra = _tmp_nodes_is_required.mutable_unchecked();

    // copy over data
    listnode *current_node = this->all_nodes.get_head();
    for(size_t node_id = 0; node_id < (size_t) this->all_nodes.nnodes; node_id++){
        ra(node_id) = current_node->node.is_required;
        current_node = current_node->next;
    }
    return _tmp_nodes_is_required;
}

py::array_t<int8_t> GasspyTree::get_nodes_has_converged(){
    // Allocate an python array
    py::array_t<int8_t> _tmp_nodes_has_converged((size_t)this->all_nodes.nnodes);
    
    // required to view?
    auto ra = _tmp_nodes_has_converged.mutable_unchecked();

    // copy over data
    listnode *current_node = this->all_nodes.get_head();
    for(size_t node_id = 0; node_id < (size_t) this->all_nodes.nnodes; node_id++){
        ra(node_id) = current_node->node.has_converged;
        current_node = current_node->next;
    }
    return _tmp_nodes_has_converged;
}

py::array_t<int16_t> GasspyTree::get_nodes_split_field(){
    // Allocate an python array
    py::array_t<int16_t> _tmp_nodes_split_field((size_t)this->all_nodes.nnodes);
    
    // required to view?
    auto ra = _tmp_nodes_split_field.mutable_unchecked();

    // copy over data
    listnode *current_node = this->all_nodes.get_head();
    for(size_t node_id = 0; node_id < (size_t) this->all_nodes.nnodes; node_id++){
        ra(node_id) = current_node->node.split_field;
        current_node = current_node->next;
    }
    return _tmp_nodes_split_field;
}

py::array_t<int> GasspyTree::get_nodes_child_node_ids(){
    // Allocate an python array
    py::array_t<int> _tmp_nodes_child_node_ids( {(size_t)this->all_nodes.nnodes, (size_t)3} );
    
    // required to view?
    auto ra = _tmp_nodes_child_node_ids.mutable_unchecked();

    // copy over data
    listnode *current_node = this->all_nodes.get_head();
    for(size_t node_id = 0; node_id < (size_t) this->all_nodes.nnodes; node_id++){
        for(size_t ichild = 0; ichild < 3; ichild++){
            if(current_node->node.is_leaf==0){
                ra(node_id, ichild) = current_node->node.child_nodes[ichild]->node_id;
            } else {
                ra(node_id, ichild) = -1;
            }
        }
        current_node = current_node->next;
    }
    return _tmp_nodes_child_node_ids;

}
py::array_t<int> GasspyTree::get_nodes_neighbor_node_ids(){
    // Allocate an python array
    py::array_t<int> _tmp_nodes_neighbor_node_ids( {(size_t)this->all_nodes.nnodes, (size_t)pow(3,n_refinement_fields)} );
    
    // required to view?
    auto ra = _tmp_nodes_neighbor_node_ids.mutable_unchecked();

    // copy over data
    listnode *current_node = this->all_nodes.get_head();
    for(size_t node_id = 0; node_id < (size_t) this->all_nodes.nnodes; node_id++){
        for(size_t ineigh = 0; ineigh < (size_t) pow(3,n_refinement_fields); ineigh++){
            if(current_node->node.has_neighbors == 1){
                ra(node_id, ineigh) = current_node->node.neighbor_nodes[ineigh]->node_id;
            } else {
                ra(node_id, ineigh) = -1;
            }
        }
        current_node = current_node->next;
    }
    return _tmp_nodes_neighbor_node_ids;
}

// setters
void GasspyTree::set_lrefine_limits(py::array_t<int16_t> lrefine_limits){
    auto ra = lrefine_limits.mutable_unchecked();
    for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
        this->fields_lrefine_min[ifield] = ra(ifield,0);
        this->fields_lrefine_max[ifield] = ra(ifield,1);
    }
}

void GasspyTree::set_gasspy_model_data(py::array_t<double> new_gasspy_model_data){
    this->gasspy_model_data = vector<vector<double>>(gasspy_model_data.size(), vector<double>(n_database_fields));
    auto ra = new_gasspy_model_data.mutable_unchecked();
    for(size_t gasspy_id = 0; gasspy_id < (size_t) new_gasspy_model_data.size(); gasspy_id++){
        for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
            this->gasspy_model_data[gasspy_id][ifield] = ra(gasspy_id,ifield);
        }
    }
    return;
}

void GasspyTree::initialize_nodes(py::array_t<double> new_model_data, py::array_t<int16_t> new_node_lrefine){
    double current_model_data[n_database_fields];
    int16_t current_node_lrefine[n_database_fields];

    auto new_model_data_ra = new_model_data.mutable_unchecked();
    auto new_node_lrefine_ra = new_node_lrefine.mutable_unchecked();

    for(int node_id = 0; node_id < new_model_data.shape(0); node_id++){
        for(int ifield = 0; ifield < n_database_fields; ifield++){
            current_model_data[ifield] = new_model_data_ra(node_id, ifield);
            current_node_lrefine[ifield] = new_node_lrefine_ra(node_id, ifield);
        }
        this->all_nodes.add_node(current_model_data, current_node_lrefine);
    }
    return;
}

void GasspyTree::set_nodes_gasspy_ids(py::array_t<int> gasspy_ids){ 
    auto gasspy_ids_ra = gasspy_ids.mutable_unchecked();
    listnode *current_node = all_nodes.get_head();
    for(int node_id = 0; node_id < this->all_nodes.nnodes; node_id++){
        current_node->node.gasspy_id = gasspy_ids_ra(node_id);
        current_node = current_node->next;
    }
    return;
}
void GasspyTree::set_nodes_is_root(py::array_t<int8_t> is_root){
    auto is_root_ra = is_root.mutable_unchecked();
    listnode *current_node = all_nodes.get_head();
    for(int node_id = 0; node_id < this->all_nodes.nnodes; node_id++){
        current_node->node.is_root = is_root_ra(node_id);
        if(current_node->node.is_root == 1){
            this->root_nodes.push_back(&(current_node->node));
            this->root_ids.push_back(node_id);
            current_node->node.root_id = this->root_nodes.size()-1;
        }
        current_node = current_node->next;
    }
    return;
}

void GasspyTree::set_nodes_is_leaf(py::array_t<int8_t> is_leaf){
    auto is_leaf_ra = is_leaf.mutable_unchecked();
    listnode *current_node = all_nodes.get_head();
    for(int node_id = 0; node_id < this->all_nodes.nnodes; node_id++){
        current_node->node.is_leaf = is_leaf_ra(node_id);
        current_node = current_node->next;
    }
    return;
}
void GasspyTree::set_nodes_is_required(py::array_t<int8_t> is_required){
    auto is_required_ra = is_required.mutable_unchecked();
    listnode *current_node = all_nodes.get_head();
    for(int node_id = 0; node_id < this->all_nodes.nnodes; node_id++){
        current_node->node.is_required = is_required_ra(node_id);
        current_node = current_node->next;
    }
    return;
}

void GasspyTree::set_nodes_has_converged(py::array_t<int8_t> has_converged){
    auto has_converged_ra = has_converged.mutable_unchecked();
    listnode *current_node = all_nodes.get_head();
    for(int node_id = 0; node_id < this->all_nodes.nnodes; node_id++){
        current_node->node.has_converged = has_converged_ra(node_id);
        current_node = current_node->next;
    }
    return;
}

void GasspyTree::set_nodes_split_field(py::array_t<int16_t> split_field){
    auto split_field_ra = split_field.mutable_unchecked();
    listnode *current_node = all_nodes.get_head();
    for(int node_id = 0; node_id < this->all_nodes.nnodes; node_id++){
        current_node->node.split_field = split_field_ra(node_id);
        current_node = current_node->next;
    }
    return;
}
void GasspyTree::set_nodes_child_node_ids(py::array_t<int> child_node_ids){
    auto child_node_ids_ra = child_node_ids.mutable_unchecked();
    listnode *current_node = all_nodes.get_head();
    for(int node_id = 0; node_id < this->all_nodes.nnodes; node_id++){
        if(child_node_ids_ra(node_id,0)==-1){
            if(current_node->node.is_leaf == 0){
                throw GasspyException("A none-leaf node was assigned no children!");
            }
        } else {
            if(current_node->node.is_leaf == 1){
                throw GasspyException("A leaf node was assigned children!");
            }
            for(int ichild = 0; ichild < 3; ichild++){
                current_node->node.child_nodes[ichild] = all_nodes.get_node(child_node_ids_ra(node_id, ichild));
            }
        }
        current_node = current_node->next;
    }
    return;
}
void GasspyTree::set_nodes_neighbor_node_ids(py::array_t<int> neighbor_node_ids){
    auto neighbor_node_ids_ra = neighbor_node_ids.mutable_unchecked();
    listnode *current_node = all_nodes.get_head();
    for(int node_id = 0; node_id < this->all_nodes.nnodes; node_id++){
        if(neighbor_node_ids_ra(node_id,0)==-1){
            current_node->node.has_neighbors = 0;
            
        } else {
            for(int ineigh = 0; ineigh < pow(3,n_refinement_fields); ineigh++){
                current_node->node.neighbor_nodes[ineigh] = all_nodes.get_node(neighbor_node_ids_ra(node_id, ineigh));
            }
            current_node->node.has_neighbors = 1;
        }
        current_node = current_node->next;
    }
    return;
}


//debug
void GasspyTree::debug_nodes(py::array_t<int> node_ids){
    auto node_ids_ra = node_ids.mutable_unchecked();
    int node_id;
    for(size_t inode = 0; inode < (size_t) node_ids.size(); inode++){
        node_id = node_ids_ra(inode);
        this->all_nodes.get_node(node_id)->debug_node();
    }   
}

PYBIND11_MODULE(cgasspy_tree, m){
    py::class_<GasspyTree>(m, "CGasspyTree")
    .def(py::init<int, int, py::array_t<int>>())
    .def("set_unique_gasspy_models", &GasspyTree::set_unique_gasspy_models)
    .def("add_points", &GasspyTree::add_points, py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>())
    .def("refine_nodes", &GasspyTree::refine_nodes)
    .def("set_has_converged", &GasspyTree::set_has_converged)
    .def("set_neighbors", &GasspyTree::set_neighbors)
    .def("get_gasspy_ids", &GasspyTree::get_gasspy_ids)
    .def("get_node_ids", &GasspyTree::get_node_ids)
    .def("get_gasspy_model_data", &GasspyTree::get_gasspy_model_data)
    .def("get_leaf_and_neighbor_gasspy_ids", &GasspyTree::get_leaf_and_neighbor_gasspy_ids)
    .def("get_nodes_model_data", &GasspyTree::get_nodes_model_data)
    .def("get_nodes_node_lrefine", &GasspyTree::get_nodes_node_lrefine)
    .def("get_nodes_gasspy_ids", &GasspyTree::get_nodes_gasspy_ids)
    .def("get_nodes_is_root", &GasspyTree::get_nodes_is_root)
    .def("get_nodes_is_leaf", &GasspyTree::get_nodes_is_leaf)
    .def("get_nodes_is_required", &GasspyTree::get_nodes_is_required)
    .def("get_nodes_has_converged", &GasspyTree::get_nodes_has_converged)
    .def("get_nodes_split_field", &GasspyTree::get_nodes_split_field)
    .def("get_nodes_child_node_ids",&GasspyTree::get_nodes_child_node_ids)
    .def("get_nodes_neighbor_node_ids",&GasspyTree::get_nodes_neighbor_node_ids)  
    .def("initialize_nodes", &GasspyTree::initialize_nodes)
    .def("set_gasspy_model_data", &GasspyTree::set_gasspy_model_data)
    .def("set_nodes_gasspy_ids", &GasspyTree::set_nodes_gasspy_ids)
    .def("set_nodes_is_root", &GasspyTree::set_nodes_is_root)
    .def("set_nodes_is_leaf", &GasspyTree::set_nodes_is_leaf)
    .def("set_nodes_is_required", &GasspyTree::set_nodes_is_required)
    .def("set_nodes_has_converged", &GasspyTree::set_nodes_has_converged)
    .def("set_nodes_split_field", &GasspyTree::set_nodes_split_field)
    .def("set_nodes_child_node_ids",&GasspyTree::set_nodes_child_node_ids)
    .def("set_nodes_neighbor_node_ids",&GasspyTree::set_nodes_neighbor_node_ids)
    .def("set_lrefine_limits",&GasspyTree::set_lrefine_limits)
    .def("debug_nodes",&GasspyTree::debug_nodes);

    static py::exception<GasspyException> ex(m,"GasspyTreeException");
    py::add_ostream_redirect(m,"ostream_redirect");
    py::register_exception_translator([](std::exception_ptr p){
        try {
            if(p){
                std::rethrow_exception(p);
            }
        } catch (const GasspyException &e) {
            ex(e.what());
        }
    });
}
