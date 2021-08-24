from gasspy.utils import save_to_fits, savename
import cudf
import cupy
import numpy as np
import os
from gasspy.utils import moveable_detector

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

    ray_df["xi"] = cupy.full(len(ray_df), ray_parameters_df["xp0_r"].iloc[0] * float(rotation_matrix[0][0]) + 
                                          ray_parameters_df["yp0_r"].iloc[0] * float(rotation_matrix[0][1]) +
                                          ray_parameters_df["zp0_r"].iloc[0] * float(rotation_matrix[0][2]) + ray_parameters_df["rot_origin_x_simfrm"].iloc[0])
    ray_df["xi"] += ((ray_df["xp"]*float(rotation_matrix[0][0])) + 
                     (ray_df["yp"]*float(rotation_matrix[0][1])) + 
                     (ray_df["zp"]*float(rotation_matrix[0][2])))
    ray_df["xi"] = ray_df["xi"].astype(int)

    ray_df["yi"] = cupy.full(len(ray_df), ray_parameters_df["xp0_r"].iloc[0] * float(rotation_matrix[1][0]) + 
                                          ray_parameters_df["yp0_r"].iloc[0] * float(rotation_matrix[1][1]) +
                                          ray_parameters_df["zp0_r"].iloc[0] * float(rotation_matrix[1][2]) + ray_parameters_df["rot_origin_y_simfrm"].iloc[0])
    ray_df["yi"] += ((ray_df["xp"]*float(rotation_matrix[1][0])) + 
                     (ray_df["yp"]*float(rotation_matrix[1][1])) + 
                     (ray_df["zp"]*float(rotation_matrix[1][2])))
    ray_df["yi"] = ray_df["yi"].astype(int)

    ray_df["zi"] = cupy.full(len(ray_df), ray_parameters_df["xp0_r"].iloc[0] * float(rotation_matrix[2][0]) + 
                                          ray_parameters_df["yp0_r"].iloc[0] * float(rotation_matrix[2][1]) +
                                          ray_parameters_df["zp0_r"].iloc[0] * float(rotation_matrix[2][2]) + ray_parameters_df["rot_origin_z_simfrm"].iloc[0])
    ray_df["zi"] += ((ray_df["xp"]*float(rotation_matrix[2][0])) + 
                     (ray_df["yp"]*float(rotation_matrix[2][1])) + 
                     (ray_df["zp"]*float(rotation_matrix[2][2])))
    ray_df["zi"] = ray_df["zi"].astype(int)

    pass
def coordinate_toIndex(ray_df, ray_parameters_df):
    ray_df["index1D"] = cudf.Series(ray_df["xi"].rmul(ray_parameters_df["Nymax"].iloc[0]*ray_parameters_df["Nzmax"].iloc[0]).radd(ray_df["yi"].rmul(ray_parameters_df["Nzmax"].iloc[0])).radd(ray_df["zi"]))
    
def path_rayCell(ray_df, ray_parameters_df):
    """
        Takes the ray segment in the simulation coordinate frame as ints, which defines a cell.
        Using duplicate values we count the number of ray subsegments in the cells, merging subsegments corresponding to the same cells and rays.
    """
    query_string  = "(xi >= 0 and xi < {0} and yi >= 0 and yi < {1} and zi >= 0 and zi < {2})".format(ray_parameters_df["Nxmax"].iloc[0],ray_parameters_df["Nymax"].iloc[0],ray_parameters_df["Nzmax"].iloc[0])
    ray_df = ray_df.query(query_string)
    # nothing in box
    if len(ray_df) == 0:
          return ray_df
    
    #get 1 dimensional cell index
    coordinate_toIndex(ray_df, ray_parameters_df)
    last = ray_df.drop_duplicates(subset=['xp', 'yp', 'index1D'],keep="last").index.to_array()
    ray_df.drop_duplicates(subset=['xp', 'yp', 'index1D'], inplace=True)
    #last = ray_df.drop_duplicates(subset=['xp', 'yp', 'xi', 'yi', 'zi'],keep="last").index.to_array()
    #ray_df.drop_duplicates(subset=['xp', 'yp', 'xi', 'yi', 'zi'], inplace=True)
    first = ray_df.index.to_array()
    nray_subsec = last - first
    #del(first,last)

    ray_df["Nraysubseg"] = nray_subsec
    del(nray_subsec)    

    ray_df["path_length"] = ray_df.Nraysubseg.rtruediv(ray_parameters_df["z_subsamples"].iloc[0])

    return(ray_df)

def getSubphysicsIndex(ray_df, idf):
    ray_df["gasspy_j"] = idf.iloc[(ray_df["xi"].values.ravel().tolist(), ray_df["yi"].values.ravel().tolist(), ray_df["zi"].values.ravel().tolist()),:] 

def traceRays(sim_data, obsplane, line_labels=None, ray_substep_parameters={"type": "cell", "nZslabs":10, "cell_subsamples":3}, savefiles = True, saveprefix=None):

    assert ray_substep_parameters["type"] == "cell",  "Trace rays only works with subcell division. Use gasspy.raytracer.traceRays_by_slab "
    # If ray marching is defined by the size of cells, decide how many nZslabs you want 
    assert "nZslabs" in ray_substep_parameters, "Using cell division. nZslabs not defined"
    assert "cell_subsamples" in ray_substep_parameters, "Using cell division. cell_subsamples not defined"
    assert int(ray_substep_parameters["nZslabs"])==ray_substep_parameters["nZslabs"], "nZslabs is not an int"
    assert int(ray_substep_parameters["cell_subsamples"]) == ray_substep_parameters["cell_subsamples"], "cell_subsample is not an int"

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
    ray_parameters_df["dZslab"] = [obsplane.dZslab]
    ray_parameters_df["xp0_r"] = [obsplane.xp0_r]
    ray_parameters_df["yp0_r"] = [obsplane.yp0_r]
    ray_parameters_df["zp0_r"] = [obsplane.zp0_r]
    ray_parameters_df["rot_origin_x_simfrm"] = [obsplane.rot_origin[0]]
    ray_parameters_df["rot_origin_y_simfrm"] = [obsplane.rot_origin[1]]
    ray_parameters_df["rot_origin_z_simfrm"] = [obsplane.rot_origin[2]]

    # Estimate the number of slabs used for calculation
    nZslab = int(obsplane.NumZ//obsplane.dZslab) + 1 
    for iz in range(int(nZslab)):
        izmin = int(iz*obsplane.dZslab*obsplane.z_subsamples)
        if(izmin >= len(zps) - 1):
            break
        izmax = min(int((iz+1)*obsplane.dZslab*obsplane.z_subsamples), len(zps))

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
        moveable_detector(rotation_matrix, ray_parameters_df = ray_parameters_df)
        raydf = path_rayCell(raydf, ray_parameters_df = ray_parameters_df)

        raydf.dropna(inplace = True)

        print(izmin, izmax, len(zps), "number of cells in slab = ", len(raydf))


        if len(raydf) == 0:
            continue
    
        # WE NOW CONVERT COORDINATE into a 1d raveled equivlant index from 3d
        # coordinate_toIndex(raydf, ray_parameters_df=ray_parameters_df)

        # Clean the object and reduce memory footprint getting rid of unused coordinates
        raydf.drop(columns=["xi","yi","zi","Nraysubseg"], inplace=True)

        # THIS IS A LIST OF CELL INDEXES WHERE THE INDEX IS AN gasspy ID
        # IT IS IN 3D, so we ravel it
        # gasspy_id = cudf.DataFrame(cupy.load(indir+"gasspy_indices3d.npy").ravel())
        raydf.set_index(["xp","yp"], inplace=True)
        raydf[line_labels] = cudf.DataFrame(data = avg_em_df.iloc[subphys_id_cudf.iloc[raydf["index1D"]].values].values, index = raydf.index)


        fluxes_df[line_labels] = fluxes_df[line_labels].add(raydf[line_labels].groupby([cudf.Grouper(level = 'xp'), cudf.Grouper(level = 'yp')]).sum(), fill_value = 0.0)

        del(raydf)    

    if savefiles:
        os.makedirs("%s/gasspy_output/"%(sim_data.datadir), exist_ok=True)
        for line_label in line_labels:
            flux_array = cupy.array(fluxes_df[line_label])
            flux_array = cupy.asnumpy(flux_array)
            fname = "%s/gasspy_output/%s.npy"%(sim_data.datadir, savename.get_filename(line_label, sim_data, obsplane, saveprefix = saveprefix))
            print("saving " + fname)
            np.save(fname, flux_array.reshape(obsplane.Nxp, obsplane.Nyp))
            del(flux_array)
    del(fluxes_df, ray_parameters_df)


def traceRays_by_slab_step(sim_data, obsplane, line_labels=None, ray_substep_parameters={"type": "slab", "nZslabs":10, "slab_subsamples":512}, savefiles = True, saveprefix=None):
    assert ray_substep_parameters["type"] == "slab",  "traceRays_by_slab rays only works with slab division. Use gasspy.raytracer.traceRays to specify a dZcell"
    # If ray marching is defined by the size of cells, decide how many nZslabs you want 
    assert "slab_subsamples" in ray_substep_parameters, "slab_subsamples: number of slab sub divisons"
    assert "nZslabs" in ray_substep_parameters, "nZslab: number of Z slabs to divid simulation not provided "

    if line_labels is None:
        """Try and read from simdata config"""
        line_labels = sim_data.config_yaml["line_labels"]

    # Ensure that indices to subphysic models are stored on the GPU as a cudf dataframe
    subphys_id_cudf = cudf.DataFrame(sim_data.get_subcell_model_id().ravel())
    # all the subphysics models that are in the simulation, store the emission and opacities on the GPU
    avg_em_df   = cudf.DataFrame(sim_data.subcell_models.DF_from_dict(line_labels))
    # TODO: opacities are currently not complete, commented for TESTING purposes
    #avg_ab_df   = cudf.DataFrame(sim_data.subcell_models.avg_ab(line_labels))

    # Predefine 2D data structures to store summed fluxes and opacities in 
    fluxes_df    = cudf.DataFrame({"xp": obsplane.xps.ravel(), "yp" : obsplane.yps.ravel()})
    for line_label in line_labels:
        fluxes_df[line_label] = cupy.zeros(obsplane.xps.ravel().shape)
    fluxes_df.set_index(["xp","yp"],inplace=True)
    #opacities_df = cudf.DataFrame({"xp": xp.ravel(), "yp" : yp.ravel(), "opacity" : np.zeros(xp.ravel().shape)}).set_index(["xp","yp"])

    # Adjust the path length through the box, scaling it from the length of the princple Z-axis depending on rotation.
    
    # Need to store a few parameters that are used by the ray tracing on the GPU
    ray_parameters_df = cudf.DataFrame()
    ray_parameters_df["Nxmax"] = [sim_data.Ncells[0]] 
    ray_parameters_df["Nymax"] = [sim_data.Ncells[1]] 
    ray_parameters_df["Nzmax"] = [sim_data.Ncells[2]]
    # deprecated: ray_parameters_df["z_subsamples"] = [obsplane.z_subsamples]
    # deprecated: ray_parameters_df["dZslab"] = [obsplane.dZslab]
    ray_parameters_df["xp0_r"] = [obsplane.xp0_r]
    ray_parameters_df["yp0_r"] = [obsplane.yp0_r]
    ray_parameters_df["zp0_r"] = [obsplane.zp0_r]
    ray_parameters_df["rot_origin_x_simfrm"] = [obsplane.rot_origin_simfrm[0]]
    ray_parameters_df["rot_origin_y_simfrm"] = [obsplane.rot_origin_simfrm[1]]
    ray_parameters_df["rot_origin_z_simfrm"] = [obsplane.rot_origin_simfrm[2]]

    # Estimate the number of slabs used for calculation
    # If the box is not evenly divided into nZslabs, then add 1 more.

    # dSlab is defined depending on an unprojected cube. Rescale depending on projection
    # Delta is the scale factor of which the length of the box is streched from the unrotated frame
    delta = obsplane.projected_Zp_length/sim_data.Ncells[2]
    nZslabs = int(ray_substep_parameters["nZslabs"]*delta) + 1*(ray_substep_parameters["nZslabs"]*delta%1 > 0)
    dZslab = obsplane.projected_Zp_length/nZslabs
  
    for iz in range(nZslabs):
        current_Z_plane_simulation_frame = iz * dZslab + obsplane.Z_plane_start
        # transform the coordinate from ray grid to simulation grid

        current_coordinates = None

        row: ip(i_xp,j_up), i_xp, j_yp, k_zp, em, op, n_zps, RL

        for ip in Nzp:
            if Nzp > len(em[i_ray]):
                big_flag = True
                tau_ip = cupy.zeros(n_zps[ip])

        for row in DF:
            tau[i_ray][k_zp] = row[tau]
            em[i_ray][k_zp] =  row[em]

            if (finshed):
                detec[i_ray] = cupy.sumproduct(em,tau)
                if big_flag:
                    em_ip = cupy.zeros(unrefined_Nzp)
                    tau_ip = cupy.zeros(unrefined_Nzp)

        coordinate_transform(raydf, obsplane.rotation_matrix, ray_parameters_df = ray_parameters_df)

        raydf = path_rayCell(raydf, ray_parameters_df = ray_parameters_df)

        # (xp0,yp0) ----------------> = Nz=64 -> CuPy array 64 long
        # (xp0,yp1) ___.___.___.___.> = Nz=4  -> CuPy array 4 long
        #            0   1   2   3
        # row entry: For each rowwise element, put em,tau into array, at correct index

        # row: 

        # 1) for each zp in {xp, yp, ray_index} cum_opac + cumsum opac                                               : total_opac[xp, yp, ray_index, zp]
        # 2) for each zp in {xp, yp, ray_index} exp(-total_opac[xp,yp, ray_index, zp]) * emis[xp, yp, ray_index, zp] : total_emis[xp, yp, ray_index, zp]
        # 3) for each ray_index in {xp, yp}     sum total_emis                                                       : flux[xp, yp, ray_index]
        # 4) for each ray_index in {xp, yp}     sum total_opac                                                       : cum_opac[xp, yp, ray_index]
        
        
        raydf.dropna(inplace = True)

        if len(raydf) == 0:
            continue
    
        # WE NOW CONVERT COORDINATE into a 1d raveled equivlant index from 3d
        # coordinate_toIndex(raydf, ray_parameters_df=ray_parameters_df)

        # Clean the object and reduce memory footprint getting rid of unused coordinates
        raydf.drop(columns=["xi","yi","zi","Nraysubseg"], inplace=True)

        # THIS IS A LIST OF CELL INDEXES WHERE THE INDEX IS AN gasspy ID
        # IT IS IN 3D, so we ravel it
        # gasspy_id = cudf.DataFrame(cupy.load(indir+"gasspy_indices3d.npy").ravel())
        raydf.set_index(["xp","yp"], inplace=True)
        raydf[line_labels] = cudf.DataFrame(data = avg_em_df.iloc[subphys_id_cudf.iloc[raydf["index1D"]].values].values, index = raydf.index)

        
        fluxes_df[line_labels] = fluxes_df[line_labels].add(raydf[line_labels].groupby([cudf.Grouper(level = 'xp'), cudf.Grouper(level = 'yp')]).sum(), fill_value = 0.0)

        del(raydf)    

    if savefiles:
        os.makedirs("%s/gasspy_output/"%(sim_data.datadir), exist_ok=True)
        for line_label in line_labels:
            flux_array = cupy.array(fluxes_df[line_label])
            flux_array = cupy.asnumpy(flux_array)
            fname = "%s/gasspy_output/%s.npy"%(sim_data.datadir, savename.get_filename(line_label, sim_data, obsplane, saveprefix = saveprefix))
            print("saving " + fname)
            np.save(fname, flux_array.reshape(obsplane.Nxp, obsplane.Nyp))
            del(flux_array)
    del(fluxes_df, ray_parameters_df)
