#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy.misc import toimage
import numpy as np
import numpy.matlib
import scipy.io as sio
import scipy.fftpack as fft
import scipy.misc as misc
import random
import math
import dicom

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def get_2D_dct(img):
    return fft.dct(fft.dct(img.T, norm='ortho').T, norm='ortho')

def get_2d_idct(coefficients):
    return fft.idct(fft.idct(coefficients.T, norm='ortho').T, norm='ortho')

FileName = ["CT-MONO2-16-ankle",
        "ILIACSM.bmp",
        "IM_0001.bmp",
        "MOVEKNEE.bmp",
        "MR-MONO2-16-head",
        "ONECSPIN.bmp",
        "ONEHEART.bmp",
        "ONESHLDR.bmp",
        "fSer1001_1.dcm"]
p0 = 1.0
ratio = 2
eps = 1.0
p = p0

rms = []
for j in xrange(len(FileName)):
    data = dicom.read_file('./imgdata/' + FileName[j]).pixel_array

    """ Modeling """
    """ Assume We Get Image through DCT Process """
    X = get_2D_dct(data)


    """ Start Simulate Compressive Sensing Process """
    """ Design Sensing Matrix """
    phi = np.matrix(np.random.normal(0, math.sqrt(M), (M, N)))
    Y = phi * X
    """ u is Traditional Decoded Image """
    u = phi.T * (np.linalg.solve((phi * phi.T), b))

    """ Compute l1 Norm by IRLS """
    prevObj = None
    while eps > 10**(-3):
        weights = np.power((np.power(u, 2)+eps), (p/2 - 1))
        for i in range(N):
            Q = np.diag(np.array(np.power(weights[:,i], -1).ravel())[0])
            u[:,i] = Q * phi.T * (np.linalg.solve((phi * Q * phi.T) , Y[:,i]))
        currObj = np.sum(np.multiply(weights,np.power(u, 2)))/10000
        if prevObj:
            if abs(currObj - prevObj) / currObj < math.sqrt(eps)/100
                eps = eps/10
        prevObj=currObj

    """ Final Result """
    toimage(u).save( str(Filename) )
