import numpy as np
import cudf 
import cupy
import cupyx
import time

ntotal  = 512*512
nactive = 512*512
ncellbuff = 128
nbuff  = 5
active_ray_index = np.random.randint(0,ntotal, nactive)
input_array = cupy.ones((nactive, ncellbuff))

# Option A
output_array = cupyx.zeros_pinned(nactive*ncellbuff*nbuff)
start = time.time()
for ibuff in range(nbuff) :
    output_array[ibuff*nactive*ncellbuff : (ibuff + 1)*nactive*ncellbuff] = input_array.ravel().get()

end  = time.time()

print("Option A : %s"%(end - start))




# Option B
output_array = cupyx.zeros_pinned((nbuff, nactive, ncellbuff))
start = time.time()
for ibuff in range(nbuff) :
    output_array[ibuff, :, :] = input_array.get()

end  = time.time()

print("Option B : %s"%(end - start))




# Option C
output_array = cupyx.zeros_pinned((nbuff * nactive, ncellbuff))
start = time.time()
for ibuff in range(nbuff) :
    output_array[ibuff*nactive : (ibuff + 1)*nactive, :] = input_array.get()

end  = time.time()

print("Option C : %s"%(end - start))

# Option D
output_array = cupyx.zeros_pinned(nactive*ncellbuff*nbuff)
start = time.time()
for ibuff in range(nbuff) :
    for iactive in range(nactive):
        output_array[ibuff*ncellbuff*nactive + iactive*ncellbuff : ibuff*nactive*ncellbuff + (iactive + 1)*ncellbuff] = input_array[iactive,:].get()

end  = time.time()

print("Option D : %s"%(end - start))
