from opiate.utils import save_to_fits, savename
import cudf
import cupy
import numpy as np
import os

use_shared_memory = 1
try:
    use_shared_memory = int(os.environ["CUDF_USE_SHARED_MEMORY"])
except:
    pass

if use_shared_memory == 1:
    import rmm
    cudf.set_allocator("managed")
    rmm.reinitialize(managed_memory=True)
    assert(rmm.is_initialized())


def coordinate_transform(ray_df, rotation_matrix, ray_parameters_df):
    """
        Given a rotation matrix (eg from scipy.spatial.transform.Rotation) rotates the ray segments
    """
    if type(rotation_matrix) is not cupy._core.core.ndarray:
        #If matrix is in system memory, move it to the GPU
        rotation_matrix = cupy.array(rotation_matrix)

    ray_df["xi"] = (((ray_df["xp"] + ray_parameters_df["xp0_r"].iloc[0])*float(rotation_matrix[0][0])) +
                    ((ray_df["yp"] + ray_parameters_df["yp0_r"].iloc[0])*float(rotation_matrix[0][1])) + 
                    ((ray_df["zp"] + ray_parameters_df["zp0_r"].iloc[0])*float(rotation_matrix[0][2])) + ray_parameters_df["rot_origin_x"].iloc[0]).astype(int) 

    ray_df["yi"] = (((ray_df["xp"] + ray_parameters_df["xp0_r"].iloc[0])*float(rotation_matrix[1][0])) + 
                    ((ray_df["yp"] + ray_parameters_df["yp0_r"].iloc[0])*float(rotation_matrix[1][1])) + 
                    ((ray_df["zp"] + ray_parameters_df["zp0_r"].iloc[0])*float(rotation_matrix[1][2])) + ray_parameters_df["rot_origin_y"].iloc[0]).astype(int)

    ray_df["zi"] = (((ray_df["xp"] + ray_parameters_df["xp0_r"].iloc[0])*float(rotation_matrix[2][0])) + 
                    ((ray_df["yp"] + ray_parameters_df["yp0_r"].iloc[0])*float(rotation_matrix[2][1])) + 
                    ((ray_df["zp"] + ray_parameters_df["zp0_r"].iloc[0])*float(rotation_matrix[2][2])) + ray_parameters_df["rot_origin_z"].iloc[0]).astype(int)
    pass
    
def path_rayCell(ray_df, ray_parameters_df):
    """
        Takes the ray segment in the simulation coordinate frame as ints, which defines a cell.
        Using duplicate values we count the number of ray subsegments in the cells, merging subsegments corresponding to the same cells and rays.
    """
    last = ray_df.drop_duplicates(subset=['xp', 'yp', 'xi', 'yi', 'zi'],keep="last").index.to_array()
    ray_df.drop_duplicates(subset=['xp', 'yp', 'xi', 'yi', 'zi'], inplace=True)
    first = ray_df.index.to_array()
    nray_subsec = last - first
    del(first,last)

    ray_df["Nraysubseg"] = nray_subsec
    del(nray_subsec)
    
    query_string  = "(xi >= 0 and xi < {0} and yi >= 0 and yi < {1} and zi >= 0 and zi < {2})".format(ray_parameters_df["Nxmax"].iloc[0],ray_parameters_df["Nymax"].iloc[0],ray_parameters_df["Nzmax"].iloc[0])
    ray_df = ray_df.query(query_string)
    if len(ray_df) == 0:
          return ray_df
    ray_df["path_length"] = ray_df.Nraysubseg.rtruediv(ray_parameters_df["z_subsamples"].iloc[0])

    return(ray_df)



def coordinate_toIndex(ray_df, ray_parameters_df):
    ray_df["index1D"] = cudf.Series(ray_df["xi"].rmul(ray_parameters_df["Nymax"].iloc[0]*ray_parameters_df["Nzmax"].iloc[0]).radd(ray_df["yi"].rmul(ray_parameters_df["Nzmax"].iloc[0])).radd(ray_df["zi"]))

def getSubphysicsIndex(ray_df, idf):
    ray_df["opiate_j"] = idf.iloc[(ray_df["xi"].values.ravel().tolist(), ray_df["yi"].values.ravel().tolist(), ray_df["zi"].values.ravel().tolist()),:] 

def traceRays(sim_data, obsplane, line_labels=None, dZslab = 52):
    if line_labels is None:
        """Try and read from simdata config"""
        line_labels = sim_data.config_yaml["line_labels"]

    # Ensure that indices to subphysic models are stored on the GPU as a cudf dataframe
    subphys_id_cudf = cudf.DataFrame(sim_data.get_subcell_model_id().ravel())
    # all the subphysics models that are in the simulation, store the emission and opacities on the GPU
    avg_em_df   = cudf.DataFrame(sim_data.subcell_models.DF_from_dict(line_labels))
    # TODO: opacities are currently not complete, commented for TESTING purposes
    #avg_ab_df   = cudf.DataFrame(sim_data.subcell_models.avg_ab(line_labels))

    # grab the 1D arrays of the observer frame and rays, in the coordinate frame of the simulation
    xps = obsplane.xps
    yps = obsplane.yps
    zps = obsplane.zps 
    rotation_matrix = cupy.array(obsplane.rotation_matrix)
    # 2D meshgrid of our points
    xp, yp = np.meshgrid(xps, yps)

    # Predefine 2D data structures to store summed fluxes and opacities in 
    print(xp.ravel())
    fluxes_df    = cudf.DataFrame({"xp": xp.ravel(), "yp" : yp.ravel()})
    for line_label in line_labels:
        fluxes_df[line_label] = cupy.zeros(xp.ravel().shape)
    fluxes_df.set_index(["xp","yp"],inplace=True)
    #opacities_df = cudf.DataFrame({"xp": xp.ravel(), "yp" : yp.ravel(), "opacity" : np.zeros(xp.ravel().shape)}).set_index(["xp","yp"])

    # Need to store a few parameters that are used by the ray tracing on the GPU
    ray_parameters_df = cudf.DataFrame()
    ray_parameters_df["Nxmax"] = [sim_data.Ncells[0]] 
    ray_parameters_df["Nymax"] = [sim_data.Ncells[1]] 
    ray_parameters_df["Nzmax"] = [sim_data.Ncells[2]] 
    ray_parameters_df["z_subsamples"] = [obsplane.z_subsamples]
    ray_parameters_df["xp0_r"] = [obsplane.xp0_r]
    ray_parameters_df["yp0_r"] = [obsplane.yp0_r]
    ray_parameters_df["zp0_r"] = [obsplane.zp0_r]
    ray_parameters_df["rot_origin_x"] = [obsplane.rot_origin[0]]
    ray_parameters_df["rot_origin_y"] = [obsplane.rot_origin[1]]
    ray_parameters_df["rot_origin_z"] = [obsplane.rot_origin[2]]


    # Estimate the number of slabs used for calculation
    nZslab = obsplane.NumZ//dZslab + 1 
    for iz in range(int(nZslab)):
        izmin = iz*dZslab*obsplane.z_subsamples
        if(izmin >= len(zps) - 1):
            break
        izmax = min((iz+1)*dZslab*obsplane.z_subsamples, len(zps))

        # 3D meshgrid
        xp, yp, zp = np.meshgrid(xps, yps, zps[izmin: izmax])

        raydf = cudf.DataFrame({"xp" : xp.ravel()})#, "yp" : yp.ravel(), "zp":zp.ravel()})
        del(xp)
        raydf["yp"] = yp.ravel()
        del(yp)
        raydf["zp"] = zp.ravel()
        del(zp)

        # transform the coordinate from ray grid to simulation grid
        coordinate_transform(raydf, rotation_matrix, ray_parameters_df = ray_parameters_df)
        raydf = path_rayCell(raydf, ray_parameters_df = ray_parameters_df)

        raydf.dropna(inplace = True)

        print(izmin, izmax, len(zps), "number of cells in slab = ", len(raydf))


        if len(raydf) == 0:
            continue
    
        # WE NOW CONVERT COORDINATE into a 1d raveled equivlant index from 3d
        coordinate_toIndex(raydf, ray_parameters_df=ray_parameters_df)

        # Clean the object and reduce memory footprint getting rid of unused coordinates
        raydf.drop(columns=["xi","yi","zi","Nraysubseg"], inplace=True)

        # THIS IS A LIST OF CELL INDEXES WHERE THE INDEX IS AN OPIATE ID
        # IT IS IN 3D, so we ravel it
        # opiate_id = cudf.DataFrame(cupy.load(indir+"opiate_indices3d.npy").ravel())
        raydf.set_index(["xp","yp"], inplace=True)
        raydf[line_labels] = cudf.DataFrame(data = avg_em_df.iloc[subphys_id_cudf.iloc[raydf["index1D"]].values].values, index = raydf.index)


        fluxes_df[line_labels] = fluxes_df[line_labels].add(raydf[line_labels].groupby([cudf.Grouper(level = 'xp'), cudf.Grouper(level = 'yp')]).sum(), fill_value = 0.0)

        del(raydf)    

    os.makedirs("%s/glowviz_output/"%(sim_data.datadir), exist_ok=True)
    for line_label in line_labels:
        flux_array = cupy.array(fluxes_df[line_label])
        flux_array = cupy.asnumpy(flux_array)
        fname = "%s/glowviz_output/%s.npy"%(sim_data.datadir,savename.get_filename(line_label, sim_data, obsplane))
        print("saving " + fname)
        np.save(fname, flux_array.reshape(obsplane.Nxp, obsplane.Nyp))
    del(fluxes_df, flux_array, ray_parameters_df)
