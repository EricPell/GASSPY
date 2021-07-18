import cudf
import cupy
import numpy as np
def coordinate_toIndex(ray_df, Nxmax, Nymax, Nzmax):
    ray_df["index1D"] = cudf.Series(ray_df["xi"].rmul(Nymax*Nxmax).radd(ray_df["yi"].rmul(Nzmax).radd(ray_df["zi"])))

xps = np.arange(512)
yps = np.arange(512)
zps = np.linspace(0, 64, int(64*5))

xp, yp, zp = np.meshgrid(xps, yps, zps)
xp = xp.ravel()
yp = yp.ravel()
zp = zp.ravel()

df = cudf.DataFrame({"xp" : xp, "yp" : yp, "zp" : zp, "np" : np.zeros(zp.shape), "zi" : zp, "xi" : xp + 0.1, "yi" : yp + 0.1})
df["zi"] = df["zi"].astype(int)
df["yi"] = df["yi"].astype(int)
df["xi"] = df["xi"].astype(int)
coordinate_toIndex(df, 512, 512, 512)

import time

t0 = time.time()
for i in range(5):
    print(i)
    last = df.drop_duplicates(subset=['xp', 'yp', 'index1D'], keep = 'last')
    first = df.drop_duplicates(subset=['xp', 'yp', 'index1D'])
    tmp  = last.index.to_array() - first.index.to_array() + 1
print(tmp, tmp.shape)
print("un grouped = %f s" %(time.time() - t0))

t0 = time.time()
for i in range(5):
    print(i)
    df.set_index(["xp","yp","index1D"],inplace=True)
    grouplist = [cudf.Grouper(level = 'xp'),
                 cudf.Grouper(level = 'yp'),
                 cudf.Grouper(level = 'index1D')]


    tmp = cudf.DataFrame(df["np"].groupby(grouplist,sort=True).count())
    tmp["zp"] = df["zp"].groupby(grouplist,sort=True).min()
    df.reset_index(inplace = True)
print(tmp, len(tmp))
print("grouped = %f s" %(time.time() - t0))


