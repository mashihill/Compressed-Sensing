#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
#import cv2
import scipy.io as sio
import scipy.fftpack as fft
import random
import math



def get_2D_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fft.dct(fft.dct(img.T, norm='ortho').T, norm='ortho')


matfn = './data33.mat'
data = sio.loadmat(matfn)
mat4py_load = data['data33']
u = get_2D_dct((mat4py_load[:,:,0]).astype(float))
N = u.shape[0]
eps = 1.0
p = 1.0
M = N/2

#phi = np.random.rand(M, N)
phi = np.eye(M, N)
phi = ((phi >= .5).astype(int) - (phi < .5).astype(int)) / math.sqrt(M)

print 'phi: ', phi
print 'u0: ', u

b = np.dot(phi, u);
u = np.dot(phi.transpose(), (np.linalg.solve((np.dot(phi, phi.transpose())), b)))

"""
IRLS Begin
"""

#while(eps > 10**(-8)):
#    weights=(u.^2+eps).^(p/2 - 1);
#    for i in range(N)
#        Q=diag(weights(:,i).^(-1));
#        u(:,i)=Q*phi'*((phi*Q*phi')\b(:,i));
#    end
#    
#    currObj=sum(sum(weights.*(u.^2)))/10000;
#    
#    
#    if abs(currObj-prevObj)/currObj < sqrt(eps)/100
#        eps=eps/10;
#    end
#    prevObj=currObj;
#end


"""
IRLS End
"""
#print(mat4py_load[:,:,1])
print 'b: ', b, b.shape
print 'u: ', u
