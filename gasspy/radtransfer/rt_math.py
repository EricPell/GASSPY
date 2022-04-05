import cupy
import numpy

class cumathlib():
    def __init__(self, nbytes=64):
    
        self.multiply = cupy.ElementwiseKernel(
            'float%i x, float%i y'%(nbytes,nbytes), 'float%i z'%(nbytes),
            '''
        z = x*y;
        ''', 'multipy')

        self.add = cupy.ElementwiseKernel(
            'float%i x, float%i y'%(nbytes,nbytes), 'float%i z'%(nbytes),
            '''
        z = x+y;
        ''', 'multipy')

        self.extinct = cupy.ElementwiseKernel(
            'float%i x, float%i y'%(nbytes,nbytes), 'float%i z'%(nbytes),
            '''
        z = exp(-x*y);
        ''', 'extinction')

