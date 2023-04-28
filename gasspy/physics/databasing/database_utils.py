import numpy as np




def __get_neighbor_idxs_recursive__(iidx, idx, ifield, interpolate_fields, all_interpolate_fields, neighbor_idxs):
    if ifield == len(all_interpolate_fields):
        neighbor_idxs[iidx] = idx
        return iidx + 1
        
    current_field = all_interpolate_fields[ifield]
    if current_field not in interpolate_fields:
        return __get_neighbor_idxs_recursive__(iidx, idx*3 ,ifield + 1, interpolate_fields, all_interpolate_fields, neighbor_idxs)

    for local_ishift, local_shift in enumerate([0,-1,1]):
        iidx = __get_neighbor_idxs_recursive__(iidx, idx*3 + local_ishift, ifield +1, interpolate_fields, all_interpolate_fields, neighbor_idxs)
    return iidx 

def get_neighbor_idxs(interpolate_fields, all_interpolate_fields):
    """
        Returns the indexes in the neighbor array for all_interpolate_fields that corresponds to shfits ONLY in interpolate_fields
    """
    neighbor_idxs = np.zeros(3**len(interpolate_fields), dtype = int)
    __get_neighbor_idxs_recursive__(0,0,0, interpolate_fields, all_interpolate_fields, neighbor_idxs)

    return neighbor_idxs




if __name__ == "__main__":
    all_interpolate_fields = ["var1", "var2", "var3"]
    interpolate_fields = ["var1", "var3"]

    print(get_neighbor_idxs(interpolate_fields, all_interpolate_fields))

