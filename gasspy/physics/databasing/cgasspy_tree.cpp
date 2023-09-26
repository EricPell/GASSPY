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
int n_discrete_fields;
int *refinement_field_index;
int *discrete_field_index;
int center_cell_neighbor_index;

long long finding_existing;
long long creating_new;

long long finding_shift;
long long finding_node;
long long creating_node;
long long non_matching;
long long matching;
long long finding_match;


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

        int *child_node_ids;
        int *neighbor_node_ids;

        // Node identification and comparisons
        int get_node(double* point, int converged_only);
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
            this->model_data   = new double[n_refinement_fields];
            this->node_lrefine = new int16_t[n_refinement_fields];
            this->node_delta   = new double[n_refinement_fields];
            for(int ifield = 0; ifield < n_refinement_fields; ifield++){
                this->model_data[ifield] = model_data[ifield];
                this->node_lrefine[ifield] = node_lrefine[ifield];
                this->node_delta[ifield] = pow(2, -(double) node_lrefine[ifield]);
            }
            this->split_field = -1;
            this->is_leaf = 1;
            this->has_neighbors = 0;
            this->is_required = 0;
            this->has_converged = 0;

            this->child_node_ids = new int[3];
            for(int ichild = 0; ichild < 3; ichild++){
                this->child_node_ids[ichild] = -1;
            }
            this->neighbor_node_ids = new int[(int)pow(3,n_refinement_fields)];
            for(int ineigh = 0; ineigh < (int) pow(3,n_refinement_fields); ineigh++){
                this->neighbor_node_ids[ineigh] = -1;
            }
        }

        ModelNode(){
            this->node_id = -1;

        };
};


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
            //this->tail = NULL;
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

vector<ModelNode> all_nodes;


int ModelNode::get_node(double* point, int converged_only = 1){ 
    int ifield;
    int child_index = 0;

    double point_field_value;
    double node_field_delta;
    double node_field_value;

    // Loop through all the fields. If we are outside the bound in any of them
    // return -1, otherwise determine if which child to ask next
    for(ifield = 0; ifield < n_refinement_fields; ifield++){
        point_field_value = point[ifield];
        node_field_delta = this->node_delta[ifield];
        node_field_value = this->model_data[ifield];

        // sum_{n->inf}1/2^n = 1, so nodes at maximum one delta away away
        if (fabs(point_field_value - node_field_value) > node_field_delta){
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
    if((this->is_leaf == 1 ) || ((this->has_converged == 1) && (converged_only==1)) || (this->child_node_ids[child_index] == -1)){
        return this->node_id;
    } else {
        return all_nodes.at(this->child_node_ids[child_index]).get_node(point);
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
    for(ifield = 0; ifield < n_refinement_fields; ifield++){

        point_field_value = point[ifield];
        node_field_delta  = this->node_delta[ifield];
        node_field_value  = this->model_data[ifield];
        if (fabs(point_field_value - node_field_value) - 1e-7*node_field_delta > node_field_delta){
            return -1;
        }

        if((this->child_node_ids[0] != NULL) && (ifield == this->split_field)){
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
    if(this->child_node_ids[0]==-1){
        // this is a leaf but did not match, therefore not found
        return -1;
    } else {
        // If the desired node is within the node boundaries, but is not this node, check children
        for(int ichild = 0; ichild < 3; ichild++){
            if((this->child_node_ids[ichild] != -1) && (look_at_child[ichild]==1)){
                int found_node = all_nodes.at(this->child_node_ids[ichild]).find_node(point);
                if(found_node != -1){
                    return found_node;
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
    for(ifield = 0; ifield < n_refinement_fields; ifield++){
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
        if((this->child_node_ids[0] != NULL) && (ifield == this->split_field) && (this->node_lrefine[ifield]<wanted_node_lrefine[ifield])){
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
    if(this->child_node_ids[0] == -1){
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
            if((this->child_node_ids[ichild] >= 0) && (look_at_child[ichild]==1)){
                int found_node = all_nodes.at(this->child_node_ids[ichild]).find_node(point, wanted_node_lrefine);
                if(found_node >= 0 ){
                    return found_node;
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
    for(ifield = 0; ifield < n_refinement_fields; ifield++){
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
    for(ifield = 0; ifield < n_refinement_fields; ifield++){
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
    for(ifield = 0; ifield < n_refinement_fields; ifield++){
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
    for(ifield = 0; ifield < n_refinement_fields; ifield++){

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
    for(int ifield = 0; ifield < n_refinement_fields; ifield++){
        py::print("\t",this->model_data[ifield],", ",this->node_lrefine[ifield]);
    }
    py::print("Gasspy_id", this->gasspy_id);
    py::print("is_leaf", this->is_leaf);
    py::print("has_neighbors", this->has_neighbors);
    py::print("split_field", this->split_field);
    py::print("is_required", this->is_required);

    py::print("Child node ids:");
    for(int ichild = 0; ichild<3; ichild++){
        py::print("\t",this->child_node_ids[ichild]);
    }

    py::print("Neighbor node ids:");
    for(int ineigh = 0; ineigh < pow(3,n_refinement_fields); ineigh++){
        py::print("\t",this->neighbor_node_ids[ineigh]);
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
    for(ifield = 0; ifield < n_refinement_fields; ifield++){
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
    if((this->is_leaf == 1 ) || (this->child_node_ids[child_index] == -1)){
        py::print("Point successful for node ", this->node_id);
    } else {
        if(child_index==-1){
        
            py::print("?????????????");
            return;
        }
        py::print("\tEntering child ", child_index, " corresponding to node ", this->child_node_ids[child_index]);
        all_nodes.at(child_node_ids[child_index]).debug_point(point);
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
        for(int ifield = 0; ifield < n_refinement_fields; ifield++){
            neighbor_model_data[ishift][ifield] += total_shift[ifield];
        }
        return ishift + 1;
    }

    // Loop over left, centre and right, determine shift and call for the next field
    double shift = pow(2, -(double)node_lrefine[i_refinement_field]);
    for(int local_shift = -1; local_shift < 2; local_shift++){
        total_shift[i_refinement_field] = local_shift*shift;
        ishift = __shift_recursive__(ishift, i_refinement_field + 1, neighbor_model_data, node_lrefine, total_shift);
    }
    return ishift;
}



class GasspyTree {
    private:
        std::vector<int> root_node_ids;
        std::vector<std::vector<double>> root_nodes_discrete_data;
        std::vector<std::vector<double>> gasspy_model_data; 
        int16_t *fields_lrefine_min;
        int16_t *fields_lrefine_max;
        double *fields_domain_limits;
    public:
        ModelNode* find_node(double *coords, int16_t *node_lrefine, int root_id);
        ModelNode* find_node(double *coords, int root_id);
        ModelNode* get_node(double *coords, int root_id);
        ModelNode* add_node(double *coords, int16_t *node_lrefine, int root_id, int is_neighbor);
        int add_point(double *point, int root_id);
        void refine_node(int16_t *new_node_lrefine, ModelNode* current_node);
        int find_node_neighbors(ModelNode* current_node);
        void set_neighbors();
        int find_root_node(double *discrete_data);
        int find_gasspy_model(vector<double> current_model);
        void set_unique_gasspy_model(ModelNode* current_node, vector<double> discrete_data);

        GasspyTree(int _n_database_fields, int _n_refinement_fields, int *_refinement_field_index, int _n_discrete_fields, int *_discrete_field_index, double *_fields_domain_limits, int16_t *_lrefine_limits){
            n_database_fields = _n_database_fields;
            n_refinement_fields = _n_refinement_fields;
            n_discrete_fields = _n_discrete_fields;

            refinement_field_index = new int[n_refinement_fields];
            center_cell_neighbor_index = 0;
            for(int ifield = 0; ifield<n_refinement_fields; ifield++){
                refinement_field_index[ifield] = _refinement_field_index[ifield];
                center_cell_neighbor_index += (int)round(pow(3, n_refinement_fields - 1 - ifield));
            }    

            discrete_field_index = new int[n_discrete_fields];
            for(int ifield = 0; ifield < n_discrete_fields; ifield ++){
                discrete_field_index[ifield] = _discrete_field_index[ifield];
            }  

            this->fields_lrefine_min = new int16_t[n_refinement_fields];    
            this->fields_lrefine_max = new int16_t[n_refinement_fields];
            this->fields_domain_limits = new double[n_database_fields*2];

            for(size_t ifield = 0; ifield < (size_t) n_refinement_fields; ifield++){
                this->fields_lrefine_min[ifield] = _lrefine_limits[2*ifield    ];
                this->fields_lrefine_max[ifield] = _lrefine_limits[2*ifield + 1];
                this->fields_domain_limits[2*ifield  ] = _fields_domain_limits[2*ifield    ];
                this->fields_domain_limits[2*ifield+1] = _fields_domain_limits[2*ifield + 1];
            }       
        }
        GasspyTree(int _n_database_fields, int _n_refinement_fields, py::array_t<int> _refinement_field_index, int _n_discrete_fields, py::array_t<int> _discrete_field_index, py::array_t<double> _fields_domain_limits, py::array_t<int16_t> _lrefine_limits){
            n_database_fields = _n_database_fields;
            n_refinement_fields = _n_refinement_fields;
            n_discrete_fields = _n_discrete_fields;

            refinement_field_index = new int[n_refinement_fields];
            auto _refinement_field_index_ra = _refinement_field_index.mutable_unchecked();
            center_cell_neighbor_index = 0;
            for(int ifield = 0; ifield<n_refinement_fields; ifield++){
                refinement_field_index[ifield] = _refinement_field_index_ra(ifield);
                center_cell_neighbor_index += (int)round(pow(3, n_refinement_fields - 1 - ifield));
            }  

            discrete_field_index = new int[n_discrete_fields];
            auto _discrete_field_index_ra = _discrete_field_index.mutable_unchecked();
            for(int ifield = 0; ifield<n_discrete_fields; ifield++){
                discrete_field_index[ifield] = _discrete_field_index_ra(ifield);
            }   

            this->fields_lrefine_min = new int16_t[n_refinement_fields];    
            this->fields_lrefine_max = new int16_t[n_refinement_fields]; 
            this->fields_domain_limits = new double[n_refinement_fields*2];

            auto _fields_domain_limits_ra = _fields_domain_limits.mutable_unchecked();
            auto _lrefine_limits_ra = _lrefine_limits.mutable_unchecked();
            for(size_t ifield = 0; ifield < (size_t) n_refinement_fields; ifield++){
                this->fields_lrefine_min[ifield] = _lrefine_limits_ra(ifield,0);
                this->fields_lrefine_max[ifield] = _lrefine_limits_ra(ifield,1);
                this->fields_domain_limits[2*ifield  ] = _fields_domain_limits_ra(ifield,0);
                this->fields_domain_limits[2*ifield+1] = _fields_domain_limits_ra(ifield,1);
            }   
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
        py::array_t<double> get_roots_discrete_data();
        py::array_t<int> get_nodes_root_id();
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
        void set_roots_discrete_data(py::array_t<double> root_discrete_data);
        void set_nodes_root_id(py::array_t<int> root_id);
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

int GasspyTree::find_root_node(double *discrete_data){

    // See if we can match it to an existing root
    for(size_t iroot = 0; iroot < this->root_nodes_discrete_data.size(); iroot ++){
        for(size_t ifield = 0; ifield < (size_t) n_discrete_fields; ifield++){
            if(fabs(discrete_data[ifield]-this->root_nodes_discrete_data[iroot][ifield])>1e-8*this->root_nodes_discrete_data[iroot][ifield]){
                goto continue_outer;
            }
        }
        return (int)iroot;
        continue_outer:;
    }

    // Otherwise create a new root

    // Roots start at the center of the hyperdimensional cube with size 1 for all refinement variables
    int16_t* node_lrefine = new int16_t[n_refinement_fields];
    double_t* model_data  = new double[n_refinement_fields];
    for(int ifield = 0; ifield < n_refinement_fields; ifield ++){
        node_lrefine[ifield] = 0;
        model_data[ifield] = 0.5;
    }

    all_nodes.push_back(ModelNode(model_data, node_lrefine));
    ModelNode* root_node = &all_nodes.back();
    root_node->node_id = all_nodes.size() - 1;
    root_node->is_root = 1;
    root_node->root_id = this->root_node_ids.size();
    this->root_node_ids.push_back(root_node->node_id);
    std::vector<double> root_discrete_data;
    std::copy(&discrete_data[0], &discrete_data[sizeof(discrete_data)], std::back_inserter(root_discrete_data));
    this->root_nodes_discrete_data.push_back(root_discrete_data);
    return root_node->root_id;
}

ModelNode* GasspyTree::get_node(double *point, int root_id){
    // find initial node in the tree
    int found_node_id = all_nodes.at(this->root_node_ids.at(root_id)).get_node(point);
    if(found_node_id < 0){
        throw GasspyException("ERROR: [GasspyTree::get_node] Could not find node containing point");
    }
    ModelNode* found_node = &all_nodes.at(found_node_id);
    // Make sure that the node is at a higher refinement level than minimum
    int16_t* node_lrefine = new int16_t[n_refinement_fields];
    int16_t* next_node_lrefine = new int16_t[n_refinement_fields];
    int refine = 0;
    for(int ifield = 0; ifield < n_refinement_fields; ifield++){
        node_lrefine[ifield] = found_node->node_lrefine[ifield];
        next_node_lrefine[ifield] = (int16_t) min((int)(node_lrefine[ifield]+1), (int) fields_lrefine_min[ifield]);
        if(next_node_lrefine[ifield]> node_lrefine[ifield]){
            refine = 1;
        }
    }
    // otherwise refine until the found node is at a minimum
    while(refine == 1){
        int previous_node_id = found_node->node_id;
        this->refine_node(next_node_lrefine, found_node);
        found_node = &all_nodes.at(all_nodes.at(previous_node_id).get_node(point));
        refine = 0;
        for(int ifield = 0; ifield < n_refinement_fields; ifield++){
            node_lrefine[ifield] = found_node->node_lrefine[ifield];
            next_node_lrefine[ifield] = (int16_t) min((int)(node_lrefine[ifield]+1), (int) fields_lrefine_min[ifield]);
            if(next_node_lrefine[ifield]> node_lrefine[ifield]){
                refine = 1;
            }
        }   
    }
    return found_node;
}


ModelNode* GasspyTree::find_node(double *coords, int root_id){
    int node_id = all_nodes.at(this->root_node_ids[root_id]).find_node(coords);
    if(node_id >= 0){
        return &all_nodes.at(node_id);
    }
    return NULL;
}

ModelNode* GasspyTree::find_node(double *coords, int16_t *node_lrefine, int root_id){
    int node_id = all_nodes.at(this->root_node_ids[root_id]).find_node(coords, node_lrefine);
    if(node_id >= 0){
        return &all_nodes.at(node_id);
    }
    return NULL;
}

ModelNode* GasspyTree::add_node(double *coords, int16_t *wanted_node_lrefine, int root_id, int is_neighbor = 0){
    ModelNode* found_node = &all_nodes.at(all_nodes.at(this->root_node_ids[root_id]).get_node(coords));
    if(found_node==NULL){
        throw GasspyException("ERROR: [GasspyTree::add_node] found node was NULL (not found)");
    }
    ModelNode* parent_node = found_node;
    int original_node_id = found_node->node_id;
    // Make sure that the node is at a higher refinement level than minimum
    int16_t* node_lrefine = new int16_t[n_refinement_fields];
    int16_t* next_node_lrefine = new int16_t[n_refinement_fields];
    int refine = 0;
    for(size_t ifield = 0; ifield < (size_t) n_refinement_fields; ifield++){
        node_lrefine[ifield] = found_node->node_lrefine[ifield];
        // only refine one variable at a time
        if(refine == 1){
            next_node_lrefine[ifield] = found_node->node_lrefine[ifield];
            continue;
        }
        next_node_lrefine[ifield] = (int16_t) min((int)(node_lrefine[ifield]+1), (int) wanted_node_lrefine[ifield]);
        if(next_node_lrefine[ifield]> node_lrefine[ifield]){
            refine = 1;
        }
    }
    if((refine == 0)&&(found_node->nodes_same_data(coords)==0)){
        throw GasspyException("Error: [GasspyTree::add_node] Could not find node even at matching refinement level");
    }

    // otherwise refine until the found node is at the desired refinement
    while(refine == 1){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        int parent_node_id = found_node->node_id;
        this->refine_node(next_node_lrefine, found_node);
        parent_node = &all_nodes.at(parent_node_id);
        int found_node_id = parent_node->get_node(coords,0);
        if(found_node_id==-1){
            throw GasspyException("ERROR: [GasspyTree::add_node] Could not find node after refining");
        }
        found_node = &all_nodes.at(found_node_id);

        refine = 0;
        for(int ifield = 0; ifield < n_refinement_fields; ifield++){
            node_lrefine[ifield] = found_node->node_lrefine[ifield];
            // only refine one variable at a time
            if(refine == 1){
                next_node_lrefine[ifield] = found_node->node_lrefine[ifield];
                continue;
            }
            next_node_lrefine[ifield] = (int16_t) min((int)(node_lrefine[ifield]+1), (int) wanted_node_lrefine[ifield]);
            if(next_node_lrefine[ifield] > node_lrefine[ifield]){
                refine = 1;
            }
        }
    }
    if(is_neighbor==1){
        all_nodes.at(original_node_id).is_leaf = 1;
    }
    if(found_node->nodes_identical(coords, wanted_node_lrefine) == 0){
        throw GasspyException("Error: add_node did not add an identical node to the one specified");
    }
    return found_node;
}

int GasspyTree::add_point(double *point, int root_id){
    // See if we can match this point to the existing nodes
    ModelNode* found_node = this->get_node(point, root_id);
    if(found_node != NULL){
        found_node->is_required=1;
        return found_node->node_id;
    }
    return -1;
}

void GasspyTree::set_neighbors(){ 
    int node_id;
    ModelNode* current_node;
    finding_shift = 0;
    finding_node = 0;
    creating_node = 0;
    matching = 0;
    non_matching = 0;
    int counter = 0;
    for(node_id = 0; node_id < all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        if((current_node->is_required==1)&&(current_node->has_neighbors!=1)){
            this->find_node_neighbors(current_node);
            counter += 1;

        }
    }
}


int GasspyTree::find_gasspy_model(vector<double> current_model){

    // Loop through all known gasspy_models and see if this node already has a matching one 
    auto start = chrono::high_resolution_clock::now();

    for(size_t imodel = 0; imodel < this->gasspy_model_data.size(); imodel++){
        vector<double> gasspy_model = this->gasspy_model_data.at(imodel);
        int found = 1;
        for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
            double max_delta = 1e-10*current_model[ifield];
            if(fabs(gasspy_model.at(ifield)-current_model.at(ifield))>max_delta){
                found = 0;
                break;
            }            
        }
        if(found == 1){
            return imodel;
        }
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    finding_match += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return -1;


}


void GasspyTree::set_unique_gasspy_model(ModelNode *current_node, vector<double> discrete_data){
    ModelNode* neighbor_node;
    // If this node is required, find gasspy models for all its neighbors (includes itself) unless they already have one
    if((current_node->is_required == 1)){
        vector<double> current_model(n_database_fields,0);
        int gasspy_id;

        // set discrete fields, (constant for a given root)
        for(int ifield = 0; ifield < n_discrete_fields; ifield++){
            current_model[discrete_field_index[ifield]] = discrete_data[ifield];
        }

        for(int ineigh = 0; ineigh < pow(3,n_refinement_fields); ineigh++){
            if(current_node->neighbor_node_ids[ineigh] == -1){
                std::ostringstream estr;
                estr << "Error: Neighbor node" << ineigh << "of required node" << current_node->node_id << "not set";
                std::string emsg = estr.str();
                throw GasspyException(emsg.c_str());
            }
            // If we already have a model, no need to check
            neighbor_node = &all_nodes.at(current_node->neighbor_node_ids[ineigh]);
            if(neighbor_node->gasspy_id>=0){
                gasspy_id = neighbor_node->gasspy_id;
                goto model_found;
            }

            // set refinement fields
            for(int ifield = 0; ifield < n_refinement_fields; ifield++){
                current_model[refinement_field_index[ifield]] = neighbor_node->model_data[ifield];
            }

            // check if we already have a model matching the current
            gasspy_id = this->find_gasspy_model(current_model);
            // otherwise create one
            if(gasspy_id < 0){
                gasspy_id = this->gasspy_model_data.size();
                this->gasspy_model_data.push_back(current_model);
            }

            model_found:;
            // Just sanity check.. This REALLY should not happen
            if(gasspy_id < 0){
                throw GasspyException("Error: gasspy_id of model less than 0 (not found) even after creating one...");
            }

            // set gasspy_id of node
            neighbor_node->gasspy_id = gasspy_id;

            // by construction, center child will have the same model, so might as well set it
            if(current_node->child_node_ids[1]>=0){
                all_nodes.at(current_node->child_node_ids[1]).gasspy_id=gasspy_id;
            }
        }
    }
    // Check its children next
    int all_null = 0;
    for(int ichild = 0; ichild < 3; ichild++){
        if(current_node->child_node_ids[ichild] >= 0){
            all_null = 1;
            this->set_unique_gasspy_model(&all_nodes.at(current_node->child_node_ids[ichild]), discrete_data);
        }
    }
}

void GasspyTree::set_unique_gasspy_models(){
    for(size_t iroot = 0; iroot < this->root_node_ids.size(); iroot++){

        this->set_unique_gasspy_model(&all_nodes.at(this->root_node_ids[iroot]),this->root_nodes_discrete_data[iroot]);
    }
    return;
}



void GasspyTree::refine_node(int16_t* new_node_lrefine, ModelNode* current_node){
    int ifield, ishift;
    double* child_model_data;
    int16_t *child_node_lrefine;
    double node_field_delta;
    double model_field_data;
    
    // Save id
    int current_node_id = current_node->node_id;
    if(current_node->child_node_ids[0] != -1){
        current_node->is_leaf = 0;
        current_node->is_required = 0;
        // if this node has already split, just call for its children
        for(ishift = 0; ishift < 3; ishift ++){
            this->refine_node(new_node_lrefine, &all_nodes.at(current_node->child_node_ids[ishift]));

            // Since the vector may have been resized, we need to find the current node again
            current_node = &all_nodes.at(current_node_id);
        }
        return;
    }

    current_node->split_field = -1;
    // Determine which refinement level we need to split. If multiple start with first
    for(ifield = 0; ifield < n_refinement_fields; ifield++){
        if(new_node_lrefine[ifield] > current_node->node_lrefine[ifield]){
            current_node->split_field = ifield;   
            break;     
        }
    }

    // This node does not need to be refined, even if it was told to...
    if(current_node->split_field == -1){
        current_node->is_leaf = 1;
        return;
    }
    // This node will refine.
    // We therefore set it as not a leaf and not be required
    current_node->is_leaf = 0;   
    current_node->is_required = 0;

    // Copy over model data into child buffer
    child_model_data = new double [n_refinement_fields];
    child_node_lrefine = new int16_t [n_refinement_fields];
    for(ifield = 0; ifield < n_refinement_fields; ifield++){
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
        ModelNode* found_node = this->find_node(child_model_data, child_node_lrefine, current_node->root_id);

        // if no matching node was found, create one
        if(found_node != NULL){
            // if a match was found, just set to this one
            current_node->child_node_ids[ishift] = found_node->node_id;
        } else {
            // add node to list of nodes
            all_nodes.push_back(ModelNode(child_model_data, child_node_lrefine));    
            found_node = &all_nodes.back();
            found_node->node_id = all_nodes.size() - 1;

            // Since the vector may have been resized, we need to find the current node again
            current_node = &all_nodes.at(current_node_id);

            current_node->child_node_ids[ishift]=found_node->node_id;
            // advance refinement level of selected field
            found_node->is_leaf = 1;
            found_node->root_id = current_node->root_id;
            if(found_node->root_id>=this->root_node_ids.size()){
                throw GasspyException("WTF is this root_id");
            }
        }
        // recursive call to this new node
        this->refine_node(new_node_lrefine, found_node);
        
        // Since the vector may have been resized, we need to find the current node again
        current_node = &all_nodes.at(current_node_id);
    }
    return;
}

int GasspyTree::find_node_neighbors(ModelNode* current_node){
    int ifield;
    int ineigh;
    vector<vector<double>> neighbor_model_data(pow(3,n_refinement_fields),vector<double>(n_refinement_fields));
    double coords[n_refinement_fields];


    auto start = std::chrono::high_resolution_clock::now();

    // First determine the position of all neighbors
    for(ineigh = 0; ineigh < pow(3, n_refinement_fields); ineigh++){
        for(ifield = 0; ifield < n_refinement_fields; ifield++){            
            neighbor_model_data[ineigh][ifield] = current_node->model_data[ifield];
        }
    }
    vector<double> total_shift(n_refinement_fields);
    int ishift = __shift_recursive__(0, 0, neighbor_model_data, current_node->node_lrefine, total_shift);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    finding_shift += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    // for each neighbor node

    for(ineigh=0; ineigh<pow(3,n_refinement_fields); ineigh++){
        // if this is the center cell, just set to self
        if(ineigh == center_cell_neighbor_index){
            current_node->neighbor_node_ids[ineigh] = current_node->node_id;
            continue;
        }

        for(ifield=0; ifield < n_refinement_fields; ifield++){
            coords[ifield] = neighbor_model_data[ineigh][ifield];
        }
        start = std::chrono::high_resolution_clock::now();
        // first check if we already have a node with these coordinates
        ModelNode* found_node = this->find_node(coords, current_node->root_id);
        elapsed = std::chrono::high_resolution_clock::now() - start;
        finding_node += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();        

        // if so set to that node
        if(found_node != NULL){
            current_node->neighbor_node_ids[ineigh] = found_node->node_id;
        } else {
            // otherwise create a new node and add node to list of nodes
            start = std::chrono::high_resolution_clock::now();
            //current_node->neighbor_nodes[ineigh] = this->all_nodes.add_node(coords, current_node->node_lrefine); 
            int current_node_id = current_node->node_id;
            int neighbor_node_id = this->add_node(coords, current_node->node_lrefine, current_node->root_id, 1)->node_id;
            // Since vector might have been resized we need to refind current node
            current_node = &all_nodes.at(current_node_id);
            current_node->neighbor_node_ids[ineigh] = neighbor_node_id; 
        }
    }
    current_node->has_neighbors = 1;
    return 1;
}



// Python interface functions
py::array_t<int> GasspyTree::add_points(py::array_t<double> points, py::array_t<int> previous_node_ids){
    int node_id;
    auto points_ra = points.mutable_unchecked();
    auto previous_node_ids_ra = previous_node_ids.mutable_unchecked();
    py::array_t<int> _tmp_node_ids(points.shape(0));
    auto _tmp_node_ids_ra = _tmp_node_ids.mutable_unchecked();
    double coord[n_refinement_fields];
    double discreete_data[n_discrete_fields];

    finding_existing = 0;
    creating_new = 0;
    for(size_t ipoint = 0; ipoint < (size_t) points.shape(0); ipoint++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }

        // Take discreete variables and find the approapriate root
        for(size_t ifield = 0; ifield < (size_t) n_discrete_fields; ifield ++){
            discreete_data[ifield] = points_ra(ipoint, discrete_field_index[ifield]);
        }
        int root_id = this->find_root_node(discreete_data);

        // Take variable refinement fields, normalize and limit to bounds
        for(size_t ifield = 0; ifield < (size_t) n_refinement_fields; ifield++){
            coord[ifield] = (points_ra(ipoint, refinement_field_index[ifield])-this->fields_domain_limits[2*ifield])/(this->fields_domain_limits[2*ifield+1]-this->fields_domain_limits[2*ifield]);
            coord[ifield] = fmax(0.0, fmin(1.0, coord[ifield]));
        }

        node_id = previous_node_ids_ra[ipoint];
        node_id = -1;
        if(node_id >= 0){
            // if this point already had a node associated to it, the point should either belong to the node 
            // or one of its children
            if(node_id < all_nodes.size()){
                ModelNode* previous_node = &all_nodes.at(previous_node_ids_ra[ipoint]);
                ModelNode* found_node = &all_nodes.at(previous_node->get_node(coord)); 
                found_node->is_required=1;
                node_id = found_node->node_id;
            } else {
                node_id = -1;
            }
        } 
        if(node_id < 0) {
            // Otherwise find a node for the point by looking through the tree
            node_id = this->add_point(coord, root_id);
        }

        _tmp_node_ids_ra(ipoint) = node_id;
        /*
        if(ipoint%100000 == 0){
            py::print("ipoint ", ipoint, "nnodes", all_nodes.size(), py::arg("flush") = true);
            py::print("Finding existing : ", finding_existing, "mu", py::arg("flush")= true);
            py::print("Creating new     : ", creating_new, "mu", py::arg("flush") = true);
        }
        */
    }
    /*
    py::print("nnodes", all_nodes.size(), py::arg("flush") = true);
    py::print("Finding existing : ", finding_existing, "mu", py::arg("flush") = true);
    py::print("Creating new     : ", creating_new, "mu", py::arg("flush") = true);
    */
    return _tmp_node_ids;
}

void GasspyTree::refine_nodes(py::array_t<int> node_ids, py::array_t<int16_t> wanted_node_lrefines){
    auto node_ids_ra = node_ids.mutable_unchecked();
    auto wanted_node_lrefines_ra = wanted_node_lrefines.mutable_unchecked();
    int node_id;
    int16_t wanted_node_lrefine[n_refinement_fields];

    for(size_t inode = 0; inode < (size_t) node_ids.size(); inode++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        node_id = node_ids_ra(inode);
        for(size_t ifield = 0; ifield < (size_t) n_refinement_fields; ifield++){
            wanted_node_lrefine[ifield] = wanted_node_lrefines_ra(inode, ifield);
        }
        this->refine_node(wanted_node_lrefine, &all_nodes.at(node_id));
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

        all_nodes.at(node_id).has_converged = 1;
    }   
}



// getters
py::array_t<int> GasspyTree::get_node_ids(py::array_t<double> points, py::array_t<int> previous_node_ids){
    py::array_t<int> _tmp_node_ids(points.shape(0));
    auto _tmp_node_ids_ra = _tmp_node_ids.mutable_unchecked();
    auto points_ra = points.mutable_unchecked();
    auto previous_node_ids_ra = previous_node_ids.mutable_unchecked();

    double coord[n_refinement_fields];
    double discreete_data[n_discrete_fields];

    int node_id;

    // Make sure that there is a root node
    int warn= 0;
    for(size_t ipoint = 0; ipoint < (size_t) points.shape(0); ipoint++){
        if(PyErr_CheckSignals()!=0){ // make ctrl-c able. This checks if error signals has been passed to python and exits if so
            throw py::error_already_set();
        }
        // Take discreete variables and find the approapriate root
        for(size_t ifield = 0; ifield < (size_t) n_discrete_fields; ifield ++){
            discreete_data[ifield] = points_ra(ipoint, discrete_field_index[ifield]);
        }
        int root_id = this->find_root_node(discreete_data);

        // Take point, normalize and limit to bounds
        for(size_t ifield = 0; ifield < (size_t) n_refinement_fields; ifield++){
            coord[ifield] = (points_ra(ipoint, refinement_field_index[ifield])-this->fields_domain_limits[2*ifield])/(this->fields_domain_limits[2*ifield+1]-this->fields_domain_limits[2*ifield]);
            coord[ifield] = fmax(0.0, fmin(1.0, coord[ifield]));
        }

        node_id = previous_node_ids_ra[ipoint];
        if(node_id >= 0){
            // if this point already had a node associated to it, the point should either belong to the node 
            // or one of its children
            if(node_id < all_nodes.size()){
                ModelNode* previous_node = &all_nodes.at(previous_node_ids_ra[ipoint]);
                ModelNode* found_node = &all_nodes.at(previous_node->get_node(coord));
                node_id = found_node->node_id;
            } else{ 
                node_id = -1;
            }
        }  
        if(node_id < 0){
            // If there's no prevoud node, or we couldnt match to the old node, find a node for the point by looking through the tree
           ModelNode* found_node = this->get_node(coord, root_id);
           node_id = found_node->node_id;
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

    double coord[n_refinement_fields];
    double discreete_data[n_discrete_fields];

    int node_id;

    ModelNode* found_node = NULL;
    int warn= 0;
    for(size_t ipoint = 0; ipoint < (size_t) points.shape(0); ipoint++){

        // Take discreete variables and find the approapriate root
        for(size_t ifield = 0; ifield < (size_t) n_discrete_fields; ifield ++){
            discreete_data[ifield] = points_ra(ipoint, discrete_field_index[ifield]);
        }
        int root_id = this->find_root_node(discreete_data);
        // Take point, normalize and limit to bounds
        for(size_t ifield = 0; ifield < (size_t) n_refinement_fields; ifield++){
            coord[ifield] = (points_ra(ipoint, refinement_field_index[ifield])-this->fields_domain_limits[2*ifield])/(this->fields_domain_limits[2*ifield+1]-this->fields_domain_limits[2*ifield]);
            coord[ifield] = fmax(0.0, fmin(1.0, coord[ifield]));
        }
        node_id = previous_node_ids_ra[ipoint];
        if(node_id >= 0){
            // if this point already had a node associated to it, the point should either belong to the node 
            // or one of its children
            if(node_id < all_nodes.size()){
                ModelNode* previous_node = &all_nodes.at(previous_node_ids_ra[ipoint]);
                found_node = &all_nodes.at(previous_node->get_node(coord));
                if(found_node == NULL) {
                    node_id = -1;
                }else{
                    node_id = found_node->node_id;
                }
            } else {
                found_node = NULL;
                node_id = -1;
            }
        } 
        if(found_node == NULL){
            // Otherwise find a node for the point by looking through the tree
            found_node = this->get_node(coord, root_id);
        }
        if(found_node == NULL){
            warn = 1;
            _tmp_gasspy_ids_ra(ipoint) = -1;
            continue;
        }
        _tmp_gasspy_ids_ra(ipoint) = found_node->gasspy_id;
        found_node = NULL;
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
    py::array_t<int> _tmp_nodes_model_data( {(size_t)all_nodes.size(), (size_t)n_database_fields} );
    return _tmp_nodes_model_data;
}


py::array_t<double> GasspyTree::get_nodes_model_data(){
    // Allocate an python array
    py::array_t<double> _tmp_nodes_model_data( {(size_t)all_nodes.size(), (size_t)n_refinement_fields} );
    
    // required to view?
    auto ra = _tmp_nodes_model_data.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        for(size_t ifield = 0; ifield < (size_t) n_refinement_fields; ifield++){
            ra(node_id, ifield) = current_node->model_data[ifield];
        }
    }
    return _tmp_nodes_model_data;
}

py::array_t<int16_t> GasspyTree::get_nodes_node_lrefine(){
    // Allocate an python array
    py::array_t<int16_t> _tmp_nodes_node_lrefine( {(size_t)all_nodes.size(), (size_t)n_refinement_fields} );
    
    // required to view?
    auto ra = _tmp_nodes_node_lrefine.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        for(size_t ifield = 0; ifield < (size_t) n_refinement_fields; ifield++){
            ra(node_id, ifield) = current_node->node_lrefine[ifield];
        }
    }
    return _tmp_nodes_node_lrefine;
}

py::array_t<int> GasspyTree::get_nodes_gasspy_ids(){
    // Allocate an python array
    py::array_t<int> _tmp_nodes_gasspy_ids((size_t)all_nodes.size());
    
    // required to view?
    auto ra = _tmp_nodes_gasspy_ids.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        ra(node_id) = current_node->gasspy_id;
    }
    return _tmp_nodes_gasspy_ids;
}

py::array_t<double> GasspyTree::get_roots_discrete_data(){
    // Allocate an python array
    py::array_t<double> _tmp_roots_discrete_data( {(size_t)this->root_nodes_discrete_data.size(), (size_t)n_discrete_fields} );
    
    // required to view?
    auto ra = _tmp_roots_discrete_data.mutable_unchecked();

    // copy over data
    for(size_t iroot = 0; iroot < (size_t) this->root_nodes_discrete_data.size(); iroot++){
        for(size_t ifield = 0; ifield < (size_t) n_discrete_fields; ifield++){
            ra(iroot, ifield) = this->root_nodes_discrete_data[iroot][ifield];
        }
    }
    return _tmp_roots_discrete_data;
}

py::array_t<int> GasspyTree::get_nodes_root_id(){
    // Allocate an python array
    py::array_t<int> _tmp_nodes_root_id((size_t)all_nodes.size());
    
    // required to view?
    auto ra = _tmp_nodes_root_id.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        ra(node_id) = current_node->root_id;
    }
    return _tmp_nodes_root_id;
}

py::array_t<int8_t> GasspyTree::get_nodes_is_root(){
    // Allocate an python array
    py::array_t<int8_t> _tmp_nodes_is_root((size_t)all_nodes.size());
    
    // required to view?
    auto ra = _tmp_nodes_is_root.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        ra(node_id) = current_node->is_root;
    }
    return _tmp_nodes_is_root;
}

py::array_t<int8_t> GasspyTree::get_nodes_is_leaf(){
    // Allocate an python array
    py::array_t<int8_t> _tmp_nodes_is_leaf((size_t)all_nodes.size());
    
    // required to view?
    auto ra = _tmp_nodes_is_leaf.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        ra(node_id) = current_node->is_leaf;
    }
    return _tmp_nodes_is_leaf;
}
py::array_t<int8_t> GasspyTree::get_nodes_is_required(){
    // Allocate an python array
    py::array_t<int8_t> _tmp_nodes_is_required((size_t)all_nodes.size());
    
    // required to view?
    auto ra = _tmp_nodes_is_required.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        ra(node_id) = current_node->is_required;
    }
    return _tmp_nodes_is_required;
}

py::array_t<int8_t> GasspyTree::get_nodes_has_converged(){
    // Allocate an python array
    py::array_t<int8_t> _tmp_nodes_has_converged((size_t)all_nodes.size());
    
    // required to view?
    auto ra = _tmp_nodes_has_converged.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        ra(node_id) = current_node->has_converged;
    }
    return _tmp_nodes_has_converged;
}

py::array_t<int16_t> GasspyTree::get_nodes_split_field(){
    // Allocate an python array
    py::array_t<int16_t> _tmp_nodes_split_field((size_t)all_nodes.size());
    
    // required to view?
    auto ra = _tmp_nodes_split_field.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        ra(node_id) = current_node->split_field;
    }
    return _tmp_nodes_split_field;
}

py::array_t<int> GasspyTree::get_nodes_child_node_ids(){
    // Allocate an python array
    py::array_t<int> _tmp_nodes_child_node_ids( {(size_t)all_nodes.size(), (size_t)3} );
    
    // required to view?
    auto ra = _tmp_nodes_child_node_ids.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        for(size_t ichild = 0; ichild < 3; ichild++){
            if(current_node->child_node_ids[ichild]!=-1){
                ra(node_id, ichild) = current_node->child_node_ids[ichild];
            } else {
                ra(node_id, ichild) = -1;
            }
        }
    }
    return _tmp_nodes_child_node_ids;

}
py::array_t<int> GasspyTree::get_nodes_neighbor_node_ids(){
    // Allocate an python array
    py::array_t<int> _tmp_nodes_neighbor_node_ids( {(size_t)all_nodes.size(), (size_t)pow(3,n_refinement_fields)} );
    
    // required to view?
    auto ra = _tmp_nodes_neighbor_node_ids.mutable_unchecked();

    // copy over data
    ModelNode* current_node;
    for(size_t node_id = 0; node_id < (size_t) all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        for(size_t ineigh = 0; ineigh < (size_t) pow(3,n_refinement_fields); ineigh++){
            if(current_node->has_neighbors == 1){
                ra(node_id, ineigh) = current_node->neighbor_node_ids[ineigh];
            } else {
                ra(node_id, ineigh) = -1;
            }
        }
    }
    return _tmp_nodes_neighbor_node_ids;
}

// setters
void GasspyTree::set_lrefine_limits(py::array_t<int16_t> lrefine_limits){
    auto ra = lrefine_limits.mutable_unchecked();
    for(size_t ifield = 0; ifield < (size_t) n_refinement_fields; ifield++){
        this->fields_lrefine_min[ifield] = ra(ifield,0);
        this->fields_lrefine_max[ifield] = ra(ifield,1);
    }
}

void GasspyTree::set_gasspy_model_data(py::array_t<double> new_gasspy_model_data){
    this->gasspy_model_data = vector<vector<double>>(new_gasspy_model_data.shape(0), vector<double>(n_database_fields));
    auto ra = new_gasspy_model_data.mutable_unchecked();
    for(size_t gasspy_id = 0; gasspy_id < (size_t) new_gasspy_model_data.shape(0); gasspy_id++){
        for(size_t ifield = 0; ifield < (size_t) n_database_fields; ifield++){
            this->gasspy_model_data[gasspy_id][ifield] = ra(gasspy_id,ifield);
        }
    }
    return;
}

void GasspyTree::initialize_nodes(py::array_t<double> new_model_data, py::array_t<int16_t> new_node_lrefine){
    // Delete old nodes if here
    all_nodes.clear();
    this->root_node_ids.clear();
    this->root_nodes_discrete_data.clear();
    this->gasspy_model_data.clear();

    double current_model_data[n_refinement_fields];
    int16_t current_node_lrefine[n_refinement_fields];

    auto new_model_data_ra = new_model_data.mutable_unchecked();
    auto new_node_lrefine_ra = new_node_lrefine.mutable_unchecked();
    for(int node_id = 0; node_id < new_model_data.shape(0); node_id++){
        for(int ifield = 0; ifield < n_refinement_fields; ifield++){
            current_model_data[ifield] = new_model_data_ra(node_id, ifield);
            current_node_lrefine[ifield] = new_node_lrefine_ra(node_id, ifield);
        }
        all_nodes.push_back(ModelNode(current_model_data, current_node_lrefine));
        all_nodes.back().node_id = all_nodes.size() - 1;
    }

    return;
}

void GasspyTree::set_nodes_gasspy_ids(py::array_t<int> gasspy_ids){ 
    auto gasspy_ids_ra = gasspy_ids.mutable_unchecked();
    ModelNode* current_node;
    for(int node_id = 0; node_id < all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        current_node->gasspy_id = gasspy_ids_ra(node_id);
    }
    return;
}


void GasspyTree::set_roots_discrete_data(py::array_t<double> root_discrete_data){
    auto root_discrete_data_ra = root_discrete_data.mutable_unchecked();
    for(size_t iroot = 0; iroot < root_discrete_data.shape(0); iroot++){
        std::vector<double> discreete_data(n_discrete_fields,0);
        for(size_t ifield; ifield < n_discrete_fields; ifield++){
            discreete_data[ifield] = root_discrete_data_ra(iroot, ifield);
        }
        this->root_nodes_discrete_data.push_back(discreete_data);
    }
}

void GasspyTree::set_nodes_root_id(py::array_t<int> root_id){ 
    auto root_id_ra = root_id.mutable_unchecked();
    ModelNode* current_node;
    for(int node_id = 0; node_id < all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        current_node->root_id = root_id_ra(node_id);
    }
    return;
}

void GasspyTree::set_nodes_is_root(py::array_t<int8_t> is_root){
    auto is_root_ra = is_root.mutable_unchecked();
    ModelNode* current_node;
    for(int node_id = 0; node_id < all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        current_node->is_root = is_root_ra(node_id);
        if(current_node->is_root == 1){
            this->root_node_ids.push_back(node_id);
            current_node->root_id = this->root_node_ids.size()-1;
        }
    }
    return;
}


void GasspyTree::set_nodes_is_leaf(py::array_t<int8_t> is_leaf){
    auto is_leaf_ra = is_leaf.mutable_unchecked();
    ModelNode* current_node;
    for(int node_id = 0; node_id < all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        current_node->is_leaf = is_leaf_ra(node_id);
    }
    return;
}
void GasspyTree::set_nodes_is_required(py::array_t<int8_t> is_required){
    auto is_required_ra = is_required.mutable_unchecked();
    ModelNode* current_node;
    for(int node_id = 0; node_id < all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        current_node->is_required = is_required_ra(node_id);
    }
    return;
}

void GasspyTree::set_nodes_has_converged(py::array_t<int8_t> has_converged){
    auto has_converged_ra = has_converged.mutable_unchecked();
    ModelNode* current_node;
    for(int node_id = 0; node_id < all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        current_node->has_converged = has_converged_ra(node_id);
    }
    return;
}

void GasspyTree::set_nodes_split_field(py::array_t<int16_t> split_field){
    auto split_field_ra = split_field.mutable_unchecked();
    ModelNode* current_node;
    for(int node_id = 0; node_id < all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        current_node->split_field = split_field_ra(node_id);
    }
    return;
}
void GasspyTree::set_nodes_child_node_ids(py::array_t<int> child_node_ids){
    auto child_node_ids_ra = child_node_ids.mutable_unchecked();
    ModelNode* current_node;
    for(int node_id = 0; node_id < all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);
        for(int ichild = 0; ichild < 3; ichild++){
            if(child_node_ids_ra(node_id,ichild)>=0){
                current_node->child_node_ids[ichild] = child_node_ids_ra(node_id, ichild);
            }
        }
    }
    return;
}
void GasspyTree::set_nodes_neighbor_node_ids(py::array_t<int> neighbor_node_ids){
    auto neighbor_node_ids_ra = neighbor_node_ids.mutable_unchecked();
    ModelNode* current_node;
    for(int node_id = 0; node_id < all_nodes.size(); node_id++){
        current_node = &all_nodes.at(node_id);

        if(neighbor_node_ids_ra(node_id,0)==-1){
            current_node->has_neighbors = 0;
            
        } else {
            for(int ineigh = 0; ineigh < pow(3,n_refinement_fields); ineigh++){
                current_node->neighbor_node_ids[ineigh] = neighbor_node_ids_ra(node_id, ineigh);
            }
            current_node->has_neighbors = 1;
        }
    }
    return;
}


//debug
void GasspyTree::debug_nodes(py::array_t<int> node_ids){
    auto node_ids_ra = node_ids.mutable_unchecked();
    int node_id;
    for(size_t inode = 0; inode < (size_t) node_ids.size(); inode++){
        node_id = node_ids_ra(inode);
        all_nodes.at(node_id).debug_node();
    }   
}

PYBIND11_MODULE(cgasspy_tree, m){
    py::class_<GasspyTree>(m, "CGasspyTree")
    .def(py::init<int, int, py::array_t<int>, int, py::array_t<int>, py::array_t<double>, py::array_t<int16_t>>())
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
    .def("get_roots_discrete_data", &GasspyTree::get_roots_discrete_data)
    .def("get_nodes_root_id", &GasspyTree::get_nodes_root_id)
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
    .def("set_roots_discrete_data", &GasspyTree::set_roots_discrete_data)
    .def("set_nodes_root_id", &GasspyTree::set_nodes_root_id)
    .def("set_nodes_is_root", &GasspyTree::set_nodes_is_root)
    .def("set_nodes_is_leaf", &GasspyTree::set_nodes_is_leaf)
    .def("set_nodes_is_required", &GasspyTree::set_nodes_is_required)
    .def("set_nodes_has_converged", &GasspyTree::set_nodes_has_converged)
    .def("set_nodes_split_field", &GasspyTree::set_nodes_split_field)
    .def("set_nodes_child_node_ids",&GasspyTree::set_nodes_child_node_ids)
    .def("set_nodes_neighbor_node_ids",&GasspyTree::set_nodes_neighbor_node_ids)
    //.def("set_lrefine_limits",&GasspyTree::set_lrefine_limits)
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
