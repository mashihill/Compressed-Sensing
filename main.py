#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy.misc import toimage
import numpy as np
import numpy.matlib
#import cv2
import scipy.io as sio
import scipy.fftpack as fft
import scipy.misc as misc
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

ImageCount = 10
FileName = ["CT-MONO2-16-ankle","ILIACSM.bmp","IM_0001","MOVEKNEE.bmp","MR-MONO2-16-head","MR-MONO2-16-knee","ONECSPIN.bmp","ONEHEART.bmp","ONESHLDR.bmp","fSer1001_0.dcm","fSer1001_1.dcm","fSer1001_11.dcm"]
p0 = 1.0
ratio = 2

for i in xrange(ImageCount):
    if (FileName[i][-1] == 'p'):
        data = misc.imread('./' + FileName[i])
    else:
        data = dicom.read_file('./' + FileName[i]).pixel_array
    print data.shape
    u = np.matrix(get_2D_dct((data).astype(float)))
    N = u.shape[0]
    eps = 1.0
    p = p0
    M = N / ratio
    print u.shape
    ResultImages = np.matlib.zeros(u.shape)
    phi = np.matrix(np.random.rand(M, N))
    #phi = np.matlib.eye(M, N)
    phi = ((phi >= .5).astype(int) - (phi < .5).astype(int)) / math.sqrt(M)
    b = phi * u;
    u = phi.T * (np.linalg.solve((phi * phi.T), b))

    """
    IRLS Begin
    """
    print 'IRLS Begin', i
    prevObj = None
    while eps > 10**(1):
        weights = np.power((np.power(u, 2)+eps), (p/2 - 1))
        for i in range(N):
            Q = np.diag(np.array(np.power(weights[:,i], -1).ravel())[0])
            u[:,i] = Q * phi.T * (np.linalg.solve((phi * Q * phi.T) , b[:,i]))
        currObj = np.sum(np.multiply(weights,np.power(u, 2)))/10000;

        if prevObj:
            if abs(currObj - prevObj) / currObj < math.sqrt(eps)/100:
                eps = eps/10;
        prevObj=currObj;

    ResultImages = get_2d_idct(u)
    toimage(ResultImages).show()

"""
IRLS End
"""
