#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
import cv2
import scipy.io as sio
import scipy.fftpack as fft

matfn = '/Users/momo/Code/MIP/Project/data33.mat'
data = sio.loadmat(matfn)
mat4py_load = data['data33']
print(mat4py_load[:,:,1])

