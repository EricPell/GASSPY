class Ray_processor_base:
    """
       Informal interface for ray_processor classes, containing all empty functions that are called externally, for example by the raytracer classes
    """
    def __init__(self):
        return    

    def init_global_ray_fields(self):
        """
            Add fields to the global_rays specific for the ray processor
        """
        return

    def update_global_ray_fields(self):
        """
            Updates/sets fields in global_rays specific for the ray processor
        """
        return

    def update_sizes(self):
        """
            Updates the sizes of buffers and active_rays to accomodate new fields added by the ray_processor
        """
        return
    def init_active_ray_fields(self):
        """
            Add fields to the global_rays specific for the ray processor
        """
        return

    def update_active_ray_fields(self):
        """
            Updates/sets fields in active_rays specific for the ray processor
        """
        return

    def create_child_fields(self, child_rays, parent_rays):
        """
            creates fields specific for the ray processor of child rays given a set of parent rays 
        """
        return 

    def finalize(self):
        """
            Final call to the ray processor, where it cleans up and does any post-processing required
        """
        return

    def process_buff(self, active_rays_indexes_todump, full = False):
        """
            Handles the rays that are about to have their buffer of cell intersections dumped. 
            Here the ray processor process each cell intersection as desired so that they can be removed from the buffer. 
        """
        return

    def raytrace_onestep(self):
        """
            Method to allow the ray processor to influence the ray tracing on a step by step basis
        """
        return

    def add_to_splitEvents(self, split_events):
        """
            Method to save split events if needed. Usefull/required to keep track of ray heritage.
        """
        return


    def alloc_buffer(self):
        """
            Allocates buffers specific to the ray processor
        """
        return
    
    def store_in_buffer(self):
        """
            Stores the desired quantities specific to the ray processor in the buffers
        """
        return

    def clean_buff(self, indexes_to_clean):
        """
            Cleans the buffers if the indexes defined in indexes_to_clean
        """
        return

    def reset_buffer(self):
        """
            Resets the buffers
        """
        return

    def save_trace(self):
        """
            TODO: remove this and make it so its only defined by specific processors...
        """