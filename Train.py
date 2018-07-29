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

Dependencies
All the dependent Python modules needed for the using our RAISR implementation can be installed 
using pip package manager and are the following:

*  NumPy
*  Numba
*  Python Imaging Library (PIL)
*  scipy
*  os
*  pickle
*  skimage

Training
All the training images are to be stored in the **trainingData** directory, before executing the 
Train.py script. In the training script, modify the upscaling factor, R, appropriately. The training 
outputs three files, **filter.txt**, **Qfactor_str.txt**, and **Qfactor_coh.txt**, which are needed in 
the testing phase. The trainingData used in this implementation is the **BSD 200** ([Martin et al. ICCV 2001](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)). A pre-trained filter with upscaling factors x2, x3, and x4 are available for testing in the **Filter** directory.

Copyright
---------
RAISR is developed in Jalali Lab at University of California, Los Angeles (UCLA).  
More information about the technique can be found in our group website: http://www.photonics.ucla.edu
 
"""

import numpy as np
import numba as nb
import os
import cv2
import pickle
import time
from math import floor
from Functions import *

trainPath = 'trainingData'

R = 2                           # Upscaling factor=2
patchSize = 11                  # Pacth Size=11
gradientSize = 9                # Gradient Size = 9
Qangle = 24                     # Quantization factor of angle =24
Qstrength = 3                   # Quantization factor of strength =3
Qcoherence = 3                  # Quantization factor of coherence =3
stre = np.zeros((Qstrength-1))  # Strength boundary
cohe = np.zeros((Qcoherence-1)) # Coherence boundary

Q = np.zeros((R*R, Qangle*Qstrength*Qcoherence, patchSize*patchSize, patchSize*patchSize))  # Eq.4
V = np.zeros((R*R, Qangle*Qstrength*Qcoherence, patchSize*patchSize))                       # Eq.5
h = np.zeros((R*R, Qangle*Qstrength*Qcoherence, patchSize*patchSize))
mark = np.zeros((R*R, Qangle*Qstrength*Qcoherence))                  # statical distribution of patch numbers in each bucket
w = Gaussian2d([patchSize, patchSize], 2)
w = w/w.max()
w = np.diag(w.ravel())                                               # Diagnal weighting matrix Wk in Algorithm 1

filelist = make_dataset(trainPath)

instance = 20000000                          # use 20000000 patches to get the Strength and coherence boundary
patchNumber = 0                              # patch number has been used
quantization = np.zeros((instance,2))        # quantization boundary
for image in filelist:
    print('\r', end='')
    print('' * 60, end='')
    print('\r Quantization: Processing '+ str(instance/2) + ' patches (' + str(200*patchNumber/instance) + '%)')
    im_uint8 = cv2.imread(image)
    if is_greyimage(im_uint8):
        im_uint8 = im_uint8[:,:,0]
    if len(im_uint8.shape)>2:
        im_ycbcr = BGR2YCbCr(im_uint8)
        im = im_ycbcr[:,:,0]
    else:
        im = im_uint8
    im = modcrop(im,R)
    im_LR = Prepare(im,patchSize,R)         # Prepare the cheap-upscaling images (optional: JPEG compression)
    im_GX,im_GY = np.gradient(im_LR)        # Calculate the gradient images
    quantization, patchNumber = QuantizationProcess (im_GX, im_GY,patchSize, patchNumber, w, quantization)  # get the strength and coherence of each patch
    if (patchNumber > instance/2):
        break

# uniform quantization of patches, get the optimized strength and coherence boundaries
quantization = quantization [0:patchNumber,:]
quantization = np.sort(quantization,axis=0)
for i in range(Qstrength-1):
    stre[i] = quantization[floor((i+1)*patchNumber/Qstrength),0]
for i in range(Qcoherence-1):
    cohe[i] = quantization[floor((i+1)*patchNumber/Qcoherence),1]

print('Begin to process images:')
imagecount = 1
for image in filelist:
    print('\r', end='')
    print('' * 60, end='')
    print('\r Processing ' + str(imagecount) +'/' + str(len(filelist))+ ' image ('   + image + ')')
    im_uint8 = cv2.imread(image)
    if is_greyimage(im_uint8):
        im_uint8 = im_uint8[:,:,0]
    if len(im_uint8.shape)>2:
        im_ycbcr = BGR2YCbCr(im_uint8)
        im = im_ycbcr[:,:,0]
    else:
        im = im_uint8
    im = modcrop(im,R)
    im_LR = Prepare(im,patchSize,R)
    im_HR = im2double(im)
    #im_HR = Dog1(im_HR)                # optional: sharpen the image
    im_GX,im_GY = np.gradient(im_LR)
    Q, V, mark = TrainProcess(im_LR, im_HR, im_GX, im_GY,patchSize, w, Qangle, Qstrength,Qcoherence, stre, cohe, R, Q, V, mark)  # get Q, V of each patch
    imagecount += 1

# optional: Using patch symmetry for nearly 8* more learning examples
# print('\r', end='')
# print(' ' * 60, end='')
# print('\r Using patch symmetry for nearly 8* more learning examples:')
# for i in range(Qangle):
#     for j in range(Qstrength*Qcoherence):
#         for r in range(R*R):
#             for t in range(1,8):
#                 t1 = t % 4
#                 t2 = floor(t / 4)
#                 Q1 = Getfromsymmetry1(Q[r, i * Qstrength * Qcoherence + j], patchSize, t1, t2)  # Rotate 90*t1 degree or flip t2
#                 V1 = Getfromsymmetry2(V[r, i * Qstrength * Qcoherence + j], patchSize, t1, t2)
#                 i1 = Qangle*t1/2 + i
#                 i1 = np.int(i1)
#                 if t2 == 1:
#                     i1 = Qangle -1 - i1
#                 while i1 >= Qangle:
#                     i1 = i1 - Qangle
#                 while i1 < 0:
#                     i1 = i1 + Qangle
#                 Q[r, i1 * Qstrength * Qcoherence + j] += Q1
#                 V[r, i1 * Qstrength * Qcoherence + j] += V1


print('Get the filter of RAISR:')
for t in range(R*R):
    for j in range(Qangle*Qstrength*Qcoherence):
        while(True):
            if(Q[t,j].sum()<100):
                break
            if(np.linalg.det(Q[t,j])<1):
                Q[t,j] = Q[t,j] + np.eye(patchSize*patchSize)* Q[t,j].sum()*0.000000005
            else:
                h[t,j] = np.linalg.inv(Q[t,j]).dot(V[t,j])         # Eq.2
                break

with open("filter"+str(R), "wb") as fp:
    pickle.dump(h, fp)

with open("Qfactor_str"+str(R), "wb") as sp:
    pickle.dump(stre, sp)

with open("Qfactor_coh"+str(R), "wb") as cp:
    pickle.dump(cohe, cp)


print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
