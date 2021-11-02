import cupy
import time
N = int(200*200*200)
print("Make dict")
print("M\ttime\ttime/op")
for M in [10,100,1000,10000, 100000]:
    storage = {}
    ids = cupy.arange(N)
    start = time.time()
    for group_i in range(0,N,M):
        rays = cupy.asnumpy(ids[group_i:group_i+M])
        storage.update(dict.fromkeys(rays,{})) 
    dt = time.time() - start
    print("%i\t%0.5f\t%0.9f"%(M, dt, dt/N))


print("Pop dict")
print("M\ttime\ttime/op")
for M in [10,100,1000,10000, 100000]:
    storage = {}
    ids = cupy.arange(N)
    start = time.time()
    for group_i in range(0,N,M):
        rays = cupy.asnumpy(ids[group_i:group_i+M])
        storage.update(dict.fromkeys(rays,{})) 
    dt = time.time() - start
    print("%i\t%0.5f\t%0.9f"%(M, dt, dt/N))
