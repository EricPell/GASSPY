import cudf
import cupy
from cudf.core import column
import numpy as np

def multiply(a_df, arg1="x", arg2=1.0, arg3=None):
    if arg3 is None:
        a_df[arg1] = (a_df[arg1]*arg2)
        a_df["int"] = cudf.DataFrame(a_df[arg1].floordiv(1), dtype=int)

    else:
        a_df[arg3] = (a_df[arg1]*arg2)
        a_df["int"] = cudf.DataFrame(a_df[arg3].floordiv(1), dtype=int)

    
def coordinate_transform(ray_df, xp_transf=0.0, yp_transf=0.0, zp_transf=0.0, rotation_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])):
    ray_df["xi"] = (ray_df["zp"]*rotation_matrix[0,2]) + xp_transf
    ray_df["yi"] = (ray_df["zp"]*rotation_matrix[1,2]) + yp_transf
    ray_df["zi"] = (ray_df["zp"]*rotation_matrix[2,2]) + zp_transf
    
def coordinate_toIndex(ray_df):
    ray_df["xi"] = cudf.DataFrame(ray_df["xi"].floordiv(1), dtype = int)
    ray_df["yi"] = cudf.DataFrame(ray_df["yi"].floordiv(1), dtype = int)
    ray_df["zi"] = cudf.DataFrame(ray_df["zi"].floordiv(1), dtype = int)

def getOpiateIndex(ray_df, idf):
    ray_df["opiate_j"] = idf.iloc[(ray_df["xi"].values.ravel().tolist(), ray_df["yi"].values.ravel().tolist(), ray_df["zi"].values.ravel().tolist()),:]  

Nxp=25
Nyp=512
Nz = 512
z_subsamples = 10
zp = np.linspace(0,Nz,int(Nz*z_subsamples))

ixx, iyy, izz = np.meshgrid(np.arange(0,512),np.arange(0,512),np.arange(0,512))
ixx = ixx.ravel()
iyy = iyy.ravel()
izz = izz.ravel()

opiate_id = np.random.randint(17555, size=ixx.shape)
op_df = cudf.DataFrame(opiate_id, dtype=int, index = cudf.DataFrame({"ix":ixx, "iy":iyy, "iz":izz}))

theta = np.pi/2.0
phi = np.pi/2.0

import time
t0 = time.time()


N_pixels_in_plane = Nxp * Nyp
npix=0
for xp in range(Nxp):
    for yp in range(Nyp):
        npix += 1

        df3d = cudf.DataFrame({"zp":zp.ravel()})
        #df.pipe(multiply,arg1="zp", arg2=-1.0, arg3="-zp")
        df3d.pipe(coordinate_transform)
        df3d.pipe(coordinate_toIndex)
        #df.pipe(getOpiateIndex, op_df)

        try:
            xi_list = cupy.append(xi_list, df3d["xi"])
            yi_list = cupy.append(yi_list, df3d["yi"])
            zi_list = cupy.append(zi_list, df3d["zi"])
        
        except:
            xi_list = cupy.ndarray((Nyp,int(Nz*z_subsamples)),dtype=int)
            yi_list = cupy.ndarray((Nyp,int(Nz*z_subsamples)),dtype=int)
            zi_list = cupy.ndarray((Nyp,int(Nz*z_subsamples)),dtype=int)

            xi_list[yp,:] = df3d["xi"].values
            yi_list[yp,:] = df3d["yi"].values
            zi_list[yp,:] = df3d["zi"].values

        # xi_list = df3d["xi"].values.ravel()
        # yi_list = df3d["yi"].values.ravel()
        # zi_list = df3d["zi"].values.ravel()
        #df3d["opiate_j"] = op_df.iloc[(xi_list, yi_list, zi_list),:]

    opiate_j = op_df.iloc[(xi_list.ravel(), yi_list.ravel(), zi_list.ravel()),:]
    print("%0.2f%% complete"%(npix/N_pixels_in_plane*100), end="\r")
    del(opiate_j,xi_list,yi_list,zi_list)

print("Total time = %f"%(time.time()-t0))
print("Time/op in ns = %f"%(Nz*z_subsamples*Nxp*Nyp/(time.time()-t0)/1e9))

