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

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def get_2D_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fft.dct(fft.dct(img.T, norm='ortho').T, norm='ortho')
def get_2d_idct(coefficients):
    """ Get 2D Inverse Cosine Transform of Image
    """
    return fft.idct(fft.idct(coefficients.T, norm='ortho').T, norm='ortho')

FileName = ["CT-MONO2-16-ankle","ILIACSM.bmp","IM_0001.bmp","MOVEKNEE.bmp","MR-MONO2-16-head","ONECSPIN.bmp","ONEHEART.bmp","ONESHLDR.bmp","fSer1001_1.dcm"]
p0 = 1.0
ratio = 2
#rms = np.matlib.zeros([3,len(FileName)])
rms_all = []

for ratio in [0.9,0.8,0.7,0.6,0.5]:
    rms = []
    for j in xrange(len(FileName)):
#for i in xrange(3, 4):
        if (FileName[j][-1] == 'p'):
            data = misc.imread('./imgdata/' + FileName[j])
        else:
            data = dicom.read_file('./imgdata/' + FileName[j]).pixel_array
        print data.shape
        u = np.matrix(get_2D_dct((data).astype(float)))
        N = u.shape[0]
        eps = 1.0
        p = p0
        M = int(N * ratio)
        ResultImages = np.matlib.zeros((u.shape[0], u.shape[1] * 3 + 100))
        ResultImages[:,u.shape[1]:u.shape[1]+50] = 256
        ResultImages[:,u.shape[1] * 2 + 50:u.shape[1] * 2 + 100] = 256
        ResultImages[:,:(u.shape[1])] = data
        #phi = np.matrix(np.random.rand(M, N))
        #phi = np.matlib.eye(M, N)
        #phi = ((phi >= .5).astype(int) - (phi < .5).astype(int)) / math.sqrt(M)
        phi = np.matrix(np.random.normal(0, math.sqrt(M), (M, N)))#phi / math.sqrt(M)
        b = phi * u;
        u = phi.T * (np.linalg.solve((phi * phi.T), b))
        u0 = np.matlib.zeros(u.shape)
        u0[:,:] = u[:,:]
        ResultImages[:,(u.shape[1] + 50): u.shape[1] * 2 + 50] = get_2d_idct(u)
        #toimage(phi).show()
        """
        IRLS Begin
        """
        print 'IRLS Begin', j
        prevObj = None
        while eps > 10**(-3):
            weights = np.power((np.power(u, 2)+eps), (p/2 - 1))
            for i in range(N):
                Q = np.diag(np.array(np.power(weights[:,i], -1).ravel())[0])
                u[:,i] = Q * phi.T * (np.linalg.solve((phi * Q * phi.T) , b[:,i]))
            currObj = np.sum(np.multiply(weights,np.power(u, 2)))/10000;
            if prevObj:
                if abs(currObj - prevObj) / currObj < math.sqrt(eps)/100:
                    eps = eps/10;
            prevObj=currObj;
        rms.append ([rmse(u, data), rmse(u, u0), rmse(u0, data)])
        print rms
        ResultImages[:,(u.shape[1]) * 2 + 100:] = get_2d_idct(u)
        #ResultImages[:] = get_2d_idct(u)
        toimage(ResultImages).save( 'no_' + str(j) + '_ratio_' + str(ratio) + '.jpg')
    rms_all.append(rms)
    print rms_all

"""
IRLS End
"""




