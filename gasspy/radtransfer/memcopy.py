import numpy
import cupy
import time
dev0 = cupy.cuda.Device(0)
dev1 = cupy.cuda.Device(1)
# with dev0:
#     a = cupy.random.rand(int(0.4e9))
t0 = time.time()
    
multiply = cupy.ElementwiseKernel(
    'float%i x, float%i y'%(64,64), 'float%i z'%(64),
    '''
z = x*y;
''', 'multipy')

add = cupy.ElementwiseKernel(
    'float%i x, float%i y'%(64,64), 'float%i z'%(64),
    '''
z = x+y;
''', 'multipy')

extinct = cupy.ElementwiseKernel(
    'float%i x, float%i y'%(64,64), 'float%i z'%(64),
    '''
z = exp(-x*y);
''', 'extinction')

for i in range(100000):
    with dev1:
        a = cupy.random.rand(int(2048*2048))
        b = cupy.random.rand(int(2048*2048))
        c = multiply(a,b) *extinct(a,b)
print(i, time.time()-t0)    