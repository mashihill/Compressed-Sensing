#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy.misc import toimage
import numpy as np
import numpy.matlib
#import cv2
import scipy.io as sio
import scipy.fftpack as fft
import random
import math
import dicom


def get_2D_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fft.dct(fft.dct(img.T, norm='ortho').T, norm='ortho')
def get_2d_idct(coefficients):
    """ Get 2D Inverse Cosine Transform of Image
    """
    return fft.idct(fft.idct(coefficients.T, norm='ortho').T, norm='ortho')

matfn = './data33.mat'
#data = sio.loadmat(matfn)
data = dicom.read_file('./fSer1001_0.dcm')
#mat4py_load = data['data33']
mat4py_load = data.pixel_array
u = np.matrix(get_2D_dct((mat4py_load).astype(float)))
N = u.shape[0]
eps = 1.0
p = 1.0
M = N/2

reconstructedImages = np.matlib.zeros(u.shape)

phi = np.matrix(np.random.rand(M, N))
#phi = np.matlib.eye(M, N)
phi = ((phi >= .5).astype(int) - (phi < .5).astype(int)) / math.sqrt(M)

print 'phi: ', phi
print 'u0: ', u

b = phi * u;
u = phi.T * (np.linalg.solve((phi * phi.T), b))

"""
IRLS Begin
"""
prevObj = None
while eps > 10**(-8):
    weights = np.power((np.power(u, 2)+eps), (p/2 - 1))
    for i in range(N):
        Q = np.diag(np.array(np.power(weights[:,i], -1).ravel())[0])
        u[:,i] = Q * phi.T * (np.linalg.solve((phi * Q * phi.T) , b[:,i]))
    currObj = np.sum(np.multiply(weights,np.power(u, 2)))/10000;

    if prevObj:
        if abs(currObj - prevObj) / currObj < math.sqrt(eps)/100:
            eps = eps/10;
    prevObj=currObj;

reconstructedImages = get_2d_idct(u)
toimage(reconstructedImages).show()

"""
IRLS End
"""
#print(mat4py_load[:,:,1])
print 'b: ', b, b.shape
print 'u: ', u
