'''
snippets
'''

import numpy as np


def func_exponent(p, x):
    return p[0]*x**p[1] + p[2]

def exp_decay(p, x):
    return p[1]*np.exp(-x/p[0])

def biexp_decay(p, x):
    return p[2]*np.exp(-x/p[0]) + p[3]*np.exp(-x/p[1])

def triexp_decay(p, x):
    return p[3]*np.exp(-x/p[0]) + p[4]*np.exp(-x/p[1]) + p[5]*np.exp(-x/p[2])

def sinexp_decay(p, x):
    return np.multiply(p[0]*np.exp(-x/p[1]), np.sin(p[2]*x/5309 + p[3]))

def func_residuals(func, p, x, y):
    return func(p, x) - y

def mean_dim_reduction(array, delta, axis=0):
    rank = np.ndim(array)
    if rank==1:
        nrows = np.size(array)
        new_array = np.empty((int(nrows/delta), ))
        for row in range(int(nrows/delta)):
            new_array[row] = np.mean(array[row*delta:(row+1)*delta])
    else:
        nrows, ncols = np.shape(array)
        new_array = np.empty((int(nrows/delta), ncols))
        if axis==1:
            for col in range(int(ncols/delta)):
                new_array[:,col] = np.mean(array[:,col*delta:(col+1)*delta], axis=1)
        else:
            for row in range(int(nrows/delta)):
                new_array[row,:] = np.mean(array[row*delta:(row+1)*delta,:], axis=0)
    
                
    return new_array
