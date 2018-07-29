#!/usr/bin/env python3

"""
Implementation of RAISR in Python by Jalali-Lab  

[RAISR](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7744595) (Rapid and Accurate Image Super 
Resolution) is an image processing algorithm published by Google Research in 2016. With sufficient 
trainingData, consisting of low and high resolution image pairs, RAISR algorithm tries to learn
a set of filters which can be applied to an input image that is not in the training set, 
to produce a higher resolution version of it. The source code released here is the Jalali-Lab 
implementation of the RAISR algorithm in Python 3.x. The implementation presented here achieved 
performance results that are comparable to that presented in Google's research paper 
(with ± 0.1 dB in PSNR). 

Just-in-time (JIT) compilation employing JIT numba is used to speed up the Python code. A very 
parallelized Python code employing multi-processing capabilities is used to speed up the testing 
process. The code has been tested on GNU/Linux and Mac OS X 10.13.2 platforms. 

Author: Sifeng He, Jalali-Lab, Department of Electrical and Computer Engineering, UCLA

Copyright
---------
RAISR is developed in Jalali Lab at University of California, Los Angeles (UCLA).  
More information about the technique can be found in our group website: http://www.photonics.ucla.edu


"""

import numpy as np
import numba as nb
import os
import cv2
import warnings
import pickle
import time
from math import floor
from skimage.measure import compare_psnr
from scipy.misc import imresize
from Functions import *
import matplotlib.image as mpimg
from multiprocessing import Pool

warnings.filterwarnings('ignore')

testPath = 'testData'

R = 2  # Upscaling factor=2 R = [ 2 3 4 ]
patchSize = 11  # Pacth Size=11
gradientSize = 9
Qangle = 24  # Quantization factor of angle =24
Qstrength = 3  # Quantization factor of strength =3
Qcoherence = 3  # Quantization factor of coherence =3

with open("Filter/filter"+str(R), "rb") as fp:
    h = pickle.load(fp)

with open("Filter/Qfactor_str"+str(R), "rb") as sp:
    stre = pickle.load(sp)

with open("Filter/Qfactor_coh"+str(R), "rb") as cp:
    cohe = pickle.load(cp)

filelist = make_dataset(testPath)

wGaussian = Gaussian2d([patchSize, patchSize], 2)
wGaussian = wGaussian/wGaussian.max()
wGaussian = np.diag(wGaussian.ravel())
imagecount = 1
patchMargin = floor(patchSize/2)
psnrRAISR = []
psnrBicubic = []
psnrBlending = []

def TestProcess(i):
    if (i < iteration - 1):
        offset_i = offset[i * batch:i * batch + batch]
        offset2_i = offset2[i * batch:i * batch + batch]
        grid = np.tile(gridon[..., None], [1, 1, batch]) + np.tile(offset_i, [patchSize, patchSize, 1])
    else:
        offset_i = offset[i * batch:im.size]
        offset2_i = offset2[i * batch:im.size]
        grid = np.tile(gridon[..., None], [1, 1, im.size - (iteration - 1) * batch]) + np.tile(offset_i,[patchSize, patchSize,1])
    f = im_LR.ravel()[grid]
    gx = im_GX.ravel()[grid]
    gy = im_GY.ravel()[grid]
    gx = gx.reshape((1, patchSize * patchSize, gx.shape[2]))
    gy = gy.reshape((1, patchSize * patchSize, gy.shape[2]))
    G = np.vstack((gx, gy))
    g1 = np.transpose(G, (2, 0, 1))
    g2 = np.transpose(G, (2, 1, 0))
    x = Gaussian_Mul(g1, g2, wGaussian)
    w, v = np.linalg.eig(x)
    idx = (-w).argsort()
    w = w[np.arange(np.shape(w)[0])[:, np.newaxis], idx]
    v = v[np.arange(np.shape(v)[0])[:, np.newaxis, np.newaxis], np.arange(np.shape(v)[1])[np.newaxis, :, np.newaxis], idx[:,np.newaxis,:]]
    thelta = np.arctan(v[:, 1, 0] / v[:, 0, 0])
    thelta[thelta < 0] = thelta[thelta < 0] + pi
    thelta = np.floor(thelta / (pi / Qangle))
    thelta[thelta > Qangle - 1] = Qangle - 1
    thelta[thelta < 0] = 0
    lamda = w[:, 0]
    u = (np.sqrt(w[:, 0]) - np.sqrt(w[:, 1])) / (np.sqrt(w[:, 0]) + np.sqrt(w[:, 1]) + 0.00000000000000001)
    lamda = np.searchsorted(stre, lamda)
    u = np.searchsorted(cohe, u)
    j = thelta * Qstrength * Qcoherence + lamda * Qcoherence + u
    j = j.astype('int')
    offset2_i = np.unravel_index(offset2_i, (H, W))
    t = ((offset2_i[0] - patchMargin) % R) * R + ((offset2_i[1] - patchMargin) % R)
    filtertj = h[t, j]
    filtertj = filtertj[:, :, np.newaxis]
    patch = f.reshape((1, patchSize * patchSize, gx.shape[2]))
    patch = np.transpose(patch, (2, 0, 1))
    result = np.matmul(patch, filtertj)
    return result


print('Begin to process images:')
for image in filelist:
    print('\r', end='')
    print('' * 60, end='')
    print('\r Processing ' + str(imagecount) + '/' + str(len(filelist)) + ' image (' + image + ')')
    im_uint8 = cv2.imread(image)
    im_mp = mpimg.imread(image)
    if len(im_mp.shape) == 2:
        im_uint8 = im_uint8[:,:,0]
    im_uint8 = modcrop(im_uint8, R)
    if len(im_uint8.shape) > 2:
        im_ycbcr = BGR2YCbCr(im_uint8)
        im = im_ycbcr[:, :, 0]
    else:
        im = im_uint8
    im_double = im2double(im)
    H, W = im.shape
    region = (slice(patchMargin, H - patchMargin), slice(patchMargin, W - patchMargin))
    start = time.time()
    imL = imresize(im_double, 1 / R, interp='bicubic', mode='F')
    im_bicubic = imresize(imL, (H, W), interp='bicubic', mode='F')
    im_bicubic = im_bicubic.astype('float64')
    im_bicubic = np.clip(im_bicubic, 0, 1)
    im_LR = np.zeros((H+patchSize-1,W+patchSize-1))
    im_LR[(patchMargin):(H+patchMargin),(patchMargin):(W+patchMargin)] = im_bicubic
    im_result = np.zeros((H, W))
    im_GX, im_GY = np.gradient(im_LR)
    index = np.array(range(im_LR.size)).reshape(im_LR.shape)
    offset = np.array(index[0:H, 0:W].ravel())
    offset2 = np.array(range(im.size))
    gridon = index[0:patchSize, 0:patchSize]
    batch = 2000
    iteration = ceil(im.size / batch + 0.000000000000001)
    im_result = np.array([])

    i = range(iteration)
    p = Pool()
    im_in = p.map(TestProcess, i)

    for i in range(iteration):
        im_result = np.append(im_result, im_in[i])

    im_result = im_result.reshape(H, W)
    im_result = np.clip(im_result, 0, 1)

    end = time.time()
    print(end - start)

    im_blending = Blending2(im_bicubic, im_result)
    # im_blending = Backprojection(imL, im_blending, 50) #Optional: Backprojection, which can slightly improve PSNR, especilly for large upscaling factor.
    im_blending = np.clip(im_blending, 0, 1)

    if len(im_uint8.shape) > 2:
        result_ycbcr = np.zeros((H, W, 3))
        result_ycbcr[:, :, 1:3] = im_ycbcr[:, :, 1:3]
        result_ycbcr[:, :, 0] = im_blending * 255
        result_ycbcr = result_ycbcr[region].astype('uint8')
        result_RAISR = YCbCr2BGR(result_ycbcr)
    else:
        result_RAISR = im_result[region] * 255
        result_RAISR = result_RAISR.astype('uint8')

    im_result = im_result*255
    im_result = np.rint(im_result).astype('uint8')
    im_bicubic = im_bicubic*255
    im_bicubic = np.rint(im_bicubic).astype('uint8')
    im_blending = im_blending * 255
    im_blending = np.rint(im_blending).astype('uint8')

    PSNR_bicubic = compare_psnr(im[region], im_bicubic[region])
    PSNR_result = compare_psnr(im[region], im_result[region])
    PSNR_blending = compare_psnr(im[region], im_blending[region])
    PSNR_blending = max(PSNR_result, PSNR_blending)

    cv2.imwrite('results/' + os.path.splitext(os.path.basename(image))[0] + '_result.bmp', result_RAISR)
    psnrRAISR.append(PSNR_result)
    psnrBicubic.append(PSNR_bicubic)
    psnrBlending.append(PSNR_blending)

    imagecount += 1


RAISR_psnrmean = np.array(psnrBlending).mean()
Bicubic_psnrmean = np.array(psnrBicubic).mean()


print('\r', end='')
print('' * 60, end='')
print('\r RAISR PSNR of '+ testPath +' is ' + str(RAISR_psnrmean))
print('\r Bicubic PSNR of '+ testPath +' is ' + str(Bicubic_psnrmean))

