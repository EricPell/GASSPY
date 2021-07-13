import cudf
import cupy
from cudf.core import column
import numpy as np
import rmm
from opiate.utils import opiate_io

cudf.set_allocator("managed")
rmm.reinitialize(managed_memory=True)
assert(rmm.is_initialized())

indir = "/home/ewpelleg/research/cinn3d/inputs/ramses/SHELL_CDMASK2/"
indexed_avg_em = opiate_io.read_dict(indir+"opiate.indexed_avg_em")


def opiate_avgem_dict_to_DF(indexed_avg_em,linelabel):
    junk = cupy.full(len(indexed_avg_em[linelabel]),0.0, dtype="float32")
    for key_i, key in enumerate(indexed_avg_em[linelabel]):
        junk[key_i] = indexed_avg_em[linelabel][key]
    avg_em_df = cudf.Series(data=junk)
    del(junk)
    return(avg_em_df)

linelabel = "H  1 6562.81A"
avg_em_df = opiate_avgem_dict_to_DF(indexed_avg_em,linelabel)

ray_parameters_series = cudf.Series(data=[512,512,512], index=["Nxmax", "Nymax", "Nzmax"])
ray_parameters_series["z_subsamples"] = 1

def main():
       
    def coordinate_transform(ray_df, xp_transf=0.0, yp_transf=0.0, zp_transf=0.0, rotation_matrix = cupy.array([[1,0,0],[0,1,0],[0,0,1]])):
        ray_df["xi"] = ((ray_df["xp"]*float(rotation_matrix[0,0])) + (ray_df["yp"]*float(rotation_matrix[0,1])) + (ray_df["zp"]*float(rotation_matrix[0,2])) - xp_transf).astype(int) 
        ray_df["yi"] = ((ray_df["xp"]*float(rotation_matrix[1,0])) + (ray_df["yp"]*float(rotation_matrix[1,1])) + (ray_df["zp"]*float(rotation_matrix[1,2])) - yp_transf).astype(int)
        ray_df["zi"] = ((ray_df["xp"]*float(rotation_matrix[2,0])) + (ray_df["yp"]*float(rotation_matrix[2,1])) + (ray_df["zp"]*float(rotation_matrix[2,2])) - zp_transf).astype(int)

        
    def coordinate_toCell(ray_df, ray_parameters=ray_parameters_series):
        last = ray_df.drop_duplicates(subset=['xi','yi','zi'],keep="last").index.to_array()
        first = ray_df.drop_duplicates(subset=['xi','yi','zi']).index.to_array()
        nray_subsec = last - first
        del(first,last)
        ray_df.drop_duplicates(subset=['xi','yi','zi'], inplace=True)

        #query_string  = "(xi >= -1 and xi < {0} and yi >= -1 and yi < {1} and zi >= -1 and zi < {2})".format(Nxmax + 1, Nymax + 1, Nzmax + 1)
        #ray_df = ray_df.query(query_string)


        ray_df["Nraysubseg"] = nray_subsec
        del(nray_subsec)
        
        query_string  = "(xi >= 0 and xi < {0} and yi >= 0 and yi < {1} and zi >= 0 and zi < {2})".format(ray_parameters["Nxmax"],ray_parameters["Nymax"],ray_parameters["Nzmax"])
        ray_df = ray_df.query(query_string)
        ray_df["path_length"] = ray_df.Nraysubseg.rtruediv(ray_parameters["z_subsamples"])

        return(ray_df)



    def coordinate_toIndex(ray_df, ray_parameters=ray_parameters_series):
        ray_df["index1D"] = cudf.Series(ray_df["xi"].rmul(ray_parameters["Nymax"]*ray_parameters["Nzmax"]).radd(ray_df["yi"].rmul(ray_parameters["Nzmax"])).radd(ray_df["zi"]))

    def getOpiateIndex(ray_df, idf):
        ray_df["opiate_j"] = idf.iloc[(ray_df["xi"].values.ravel().tolist(), ray_df["yi"].values.ravel().tolist(), ray_df["zi"].values.ravel().tolist()),:]  
    theta = np.pi*0.01
    phi = np.pi*0.0
    xps = np.arange(0,512)
    yps = np.arange(0,512)

    Nxp = len(xps)
    Nyp = len(yps) 
    
    xp, yp = np.meshgrid(xps, yps)

    fluxes_df    = cudf.DataFrame({"xp": xp.ravel(), "yp" : yp.ravel(), "flux" : np.zeros(xp.ravel().shape)}).set_index(["xp","yp"])
    #opacities_df = cudf.DataFrame({"xp": xp.ravel(), "yp" : yp.ravel(), "opacity" : np.zeros(xp.ravel().shape)}).set_index(["xp","yp"])


    outfluxmap = cupy.full((Nxp,Nxp),0.0,dtype="float32")

    Nz = 512
    zps = np.linspace(0,Nz,int(((Nz)*ray_parameters_series["z_subsamples"]+1)))

    dZStep = 52
    nZStep = Nz//dZStep + 1
    import time
    opiate_id = cudf.DataFrame(cupy.load(indir+"opiate_indices3d.npy").ravel())

    t0 = time.time()
    for iz in range(nZStep):
        xp, yp, zp = np.meshgrid(xps, yps, zps[iz*dZStep: (iz+1)*dZStep])

        df3d = cudf.DataFrame({"xp" : xp.ravel()})#, "yp" : yp.ravel(), "zp":zp.ravel()})
        del(xp)
        df3d["yp"] = yp.ravel()
        del(yp)
        df3d["zp"] = zp.ravel()
        del(zp)

        theta = np.pi*0.01
        phi = np.pi*0.0


        # transform the coordinate from ray grid to simulation grid
        coordinate_transform(df3d)
        df3d = coordinate_toCell(df3d, ray_parameters = ray_parameters_series)

        # WE NOW CONVERT COORDINATE into a 1d raveled equivlant index from 3d
        coordinate_toIndex(df3d, ray_parameters=ray_parameters_series)

        # Clean the object and reduce memory footprint getting rid of unused coordinates
        df3d.drop(columns=["xi","yi","zi","Nraysubseg"], inplace=True)

        # THIS IS A LIST OF CELL INDEXES WHERE THE INDEX IS AN OPIATE ID
        # IT IS IN 3D, so we ravel it
        # opiate_id = cudf.DataFrame(cupy.load(indir+"opiate_indices3d.npy").ravel())

        df3d["avg_em"] = cudf.Series(data = avg_em_df.iloc[opiate_id.iloc[df3d["index1D"]].values].values, index = df3d.index)

        df3d.set_index(["xp","yp"], inplace=True)
        fluxes_df["flux"] += df3d["avg_em"].groupby([cudf.Grouper(level = 'xp'), cudf.Grouper(level = 'yp')]).sum()
        print(iz)

    print(fluxes_df)
'''
    #for xp in range(Nxp):
    t0 = time.time()
    for ixp, xp in enumerate(xps):
        xp_transf = xp
        for iyp, yp in enumerate(yps):
            yp_transf = yp
            npix += 1

            df3d = cudf.DataFrame({"zp":zp.ravel()})
            #df.pipe(multiply,arg1="zp", arg2=-1.0, arg3="-zp")
            df3d.pipe(coordinate_transform, xp_transf=xp_transf, yp_transf=yp_transf)
            # Note: We query and overwrite the dataframe with the expectation that most values are outside the box
            df3d = coordinate_toCell(df3d, ray_parameters = ray_parameters_series)
            
            # Note: coordinate_toIndex in local memory space
            coordinate_toIndex(df3d, ray_parameters = ray_parameters_series)



            opiate_j = op_df.iloc[df3d["index1D"]]

            outfluxmap[ixp,iyp] = avg_em.iloc[opiate_j.values].sum()
        print("%0.2f%% complete"%(100.*(ixp+1)/Nxp), end="\r")

    cupy.save(indir+"outfluxmap.npy", outfluxmap)


    print("\n")
    print("Total time = %f"%(time.time()-t0))
    #print("Time/ray element in ms = %f"%((time.time()-t0)*1e3/(Nz*ray_parameters_series["z_subsamples"]*Nxp*Nyp)))
'''
if __name__ == '__main__':
    
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()

    for i in range(1):
        main()

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.dump_stats('profile-data')

