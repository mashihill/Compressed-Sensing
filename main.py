#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
#import cv2
import scipy.io as sio
import scipy.fftpack as fft
import random
import math
from skimage import img_as_float

from numpy import empty,arange,exp,real,imag,pi
from numpy.fft import rfft,irfft

def dct(y):
    N = len(y)
    y2 = empty(2*N,float)
    y2[:N] = y[:]
    y2[N:] = y[::-1]

    c = rfft(y2)
    phi = exp(-1j*pi*arange(N)/(2*N))
    return real(phi*c[:N])

def dct2(y):
    M = y.shape[0]
    N = y.shape[1]
    a = empty([M,N],float)
    b = empty([M,N],float)

    for i in range(M):
        a[i,:] = dct(y[i,:])
    for j in range(N):
        b[:,j] = dct(a[:,j])

    return b

matfn = './data33.mat'
data = sio.loadmat(matfn)
mat4py_load = data['data33']
#u = fft.dct(img_as_float(mat4py_load[:,:,0]))
#u = fft.dct((mat4py_load[:,:,0]).astype(float), type=3)
u = dct2((mat4py_load[:,:,0]).astype(float))
N = u.shape[0]
eps = 1
p = 1
M = N/2

#phi = np.random.rand(M, N)
phi = np.eye(M, N)
phi = ((phi >= .5).astype(int) - (phi < .5).astype(int)) / math.sqrt(M)

print 'phi: ', phi
print 'u0: ', u

b = np.dot(phi, u);
u = np.dot(phi.transpose(), (np.linalg.solve((np.dot(phi, phi.transpose())), b)))

"""
Run the algorithm
IRLS
"""

#print(mat4py_load[:,:,1])
print 'b: ', b, b.shape
print 
print 'u: ', u
