'''
snippets
'''

import numpy as np


def exponent_func(p, x):
    return p[0]*x**p[1] + p[2]


def exp_func(p, x):
    return p[0]*np.exp(-x/p[1]) + p[2]


def biexp_func(p, x):
    return p[0]*np.exp(-x/p[1]) + p[2]*np.exp(-x/p[3])


def triexp_func(p, x):
    return p[0]*np.exp(-x/p[1]) + p[2]*np.exp(-x/p[3]) + p[4]*np.exp(-x/p[5])


def sinexp_func(p, x):
    '''
    p[0] = A0
    p[1] = tau
    p[2] = omega
    p[3] = phi
    '''
    ### 5309 is a conv factor for omega in cm^-1
    #return np.multiply(p[0]*np.exp(-x/p[1]), np.sin(p[2]*x/5309 + p[3]))
    return np.multiply(p[0]*np.exp(-x/p[1]), np.sin(p[2]*x + p[3]))


def gaussian_func(p, x):
    '''
    p[0] = A0
    p[1] = sigma
    p[2] = x0
    '''
    return p[0]*np.exp(-0.5*((x-p[2])/p[1])**2.0)


def cauchy_func(p, x):
    '''
    p[0] = A0
    p[1] = gamma
    p[2] = x0
    '''
    return p[0]*(1.0/np.pi)*(1.0/p[1])*(1.0/(1.0+((x-p[2])/p[1])**2.0))


def residuals_func(func, p, x, y):
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
