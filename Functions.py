import cv2
from scipy.misc import imresize
from scipy.signal import convolve2d
import numpy as np
from math import atan2, floor, pi, ceil, isnan
import numba as nb
import os

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

# Python opencv library (cv2) cv2.COLOR_BGR2YCrCb has different parameters with MATLAB color convertion.
# In order to have a fair comparison with the benchmark, we wrote these functions by ourselves.
def BGR2YCbCr(im):
    mat = np.array([[24.966, 128.553, 65.481],[112, -74.203, -37.797], [-18.214, -93.786, 112]])
    mat = mat.T
    offset = np.array([[[16, 128, 128]]])

    if im.dtype == 'uint8':
        mat = mat/255
        out = np.dot(im,mat) + offset
        out = np.clip(out, 0, 255)
        out = np.rint(out).astype('uint8')
    elif im.dtype == 'float':
        mat = mat/255
        offset = offset/255
        out = np.dot(im, mat) + offset
        out = np.clip(out, 0, 1)
    else:
        assert False
    return out

def YCbCr2BGR(im):
    mat = np.array([[24.966, 128.553, 65.481],[112, -74.203, -37.797], [-18.214, -93.786, 112]])
    mat = mat.T
    mat = np.linalg.inv(mat)
    offset = np.array([[[16, 128, 128]]])

    if im.dtype == 'uint8':
        mat = mat * 255
        out = np.dot((im - offset),mat)
        out = np.clip(out, 0, 255)
        out = np.rint(out).astype('uint8')
    elif im.dtype == 'float':
        mat = mat * 255
        offset = offset/255
        out = np.dot((im - offset),mat)
        out = np.clip(out, 0, 1)
    else:
        assert False
    return out

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    if im.dtype == 'uint8':
        out = im.astype('float') / 255
    elif im.dtype == 'uint16':
        out = im.astype('float') / 65535
    else:
        assert False
    out = np.clip(out, 0, 1)
    return out

def Gaussian2d(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python

    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def modcrop(im,modulo):
    shape = im.shape
    size0 = shape[0] - shape[0] % modulo
    size1 = shape[1] - shape[1] % modulo
    if len(im.shape) == 2:
        out = im[0:size0, 0:size1]
    else:
        out = im[0:size0, 0:size1, :]
    return out

def Prepare(im, patchSize, R):
    patchMargin = floor(patchSize/2)
    H, W = im.shape
    imL = imresize(im, 1 / R, interp='bicubic')
    # cv2.imwrite('Compressed.jpg', imL, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    # imL = cv2.imread('Compressed.jpg')
    # imL = imL[:,:,0]   # Optional: Compress the image
    imL = imresize(imL, (H, W), interp='bicubic')
    imL = im2double(imL)
    im_LR = imL
    return im_LR

def is_greyimage(im):
    x = abs(im[:,:,0]-im[:,:,1])
    y = np.linalg.norm(x)
    if y==0:
        return True
    else:
        return False

@nb.jit(nopython=True, parallel=True)
def Grad(patchX,patchY,weight):

    gx = patchX.ravel()
    gy = patchY.ravel()
    G = np.vstack((gx,gy)).T
    x0 = np.dot(G.T,weight)
    x = np.dot(x0, G)
    w,v = np.linalg.eig(x)
    index= w.argsort()[::-1]
    w = w[index]
    v = v[:,index]

    lamda = w[0]

    u = (np.sqrt(w[0]) - np.sqrt(w[1]))/(np.sqrt(w[0]) + np.sqrt(w[1]) + 0.00000000000000001)

    return lamda,u

@nb.jit(nopython=True, parallel=True)
def HashTable(patchX,patchY,weight, Qangle,Qstrength,Qcoherence,stre,cohe):
    assert (len(stre)== Qstrength-1) and (len(cohe)==Qcoherence-1),"Quantization number should be equal"

    gx = patchX.ravel()
    gy = patchY.ravel()
    G = np.vstack((gx,gy)).T
    x0 = np.dot(G.T,weight)
    x = np.dot(x0, G)
    w,v = np.linalg.eig(x)
    index= w.argsort()[::-1]
    w = w[index]
    v = v[:,index]

    theta = atan2(v[1,0], v[0,0])
    if theta<0:
        theta = theta+pi
    theta = floor(theta/(pi/Qangle))

    lamda = w[0]

    u = (np.sqrt(w[0]) - np.sqrt(w[1]))/(np.sqrt(w[0]) + np.sqrt(w[1]) + 0.00000000000000001)

    if isnan(u):
        u=1

    if theta>Qangle-1:
        theta = Qangle-1
    if theta<0:
        theta = 0

    lamda = np.searchsorted(stre,lamda)

    u = np.searchsorted(cohe,u)

    return theta,lamda,u

@nb.jit(nopython=True, parallel=True)
def Gaussian_Mul(x,y,wGaussian):
    result = np.zeros((x.shape[0], x.shape[1], y.shape[2]))
    for i in range(x.shape[0]):
        # inter = np.matmul(x[i], wGaussian)
        # result[i] = np.matmul(inter,y[i])
        inter = np.dot(x[i], wGaussian)
        result[i] = np.dot(inter, y[i])
    return result

def CT_descriptor(im):
    H, W = im.shape
    windowSize = 3
    Census = np.zeros((H, W))
    CT = np.zeros((H, W, windowSize, windowSize))
    C = np.int((windowSize-1)/2)
    for i in range(C,H-C):
        for j in range(C, W-C):
            cen = 0
            for a in range(-C, C+1):
                for b in range(-C, C+1):
                    if not (a==0 and b==0):
                        if im[i+a, j+b] < im[i, j]:
                            cen += 1
                            CT[i, j, a+C,b+C] = 1
            Census[i, j] = cen
    Census = Census/8
    return Census, CT

def Blending1(LR, HR):
    H,W = LR.shape
    H1,W1 = HR.shape
    assert H1==H and W1==W
    Census,CT = CT_descriptor(LR)
    blending1 = Census*HR + (1 - Census)*LR
    return blending1

def Blending2(LR, HR):
    H,W = LR.shape
    H1,W1 = HR.shape
    assert H1==H and W1==W
    Census1, CT1 = CT_descriptor(LR)
    Census2, CT2 = CT_descriptor(HR)
    weight = np.zeros((H, W))
    x = np.zeros(( 3, 3))
    for i in range(H):
        for j in range(W):
            x  = np.absolute(CT1[i,j]-CT2[i,j])
            weight[i, j] = x.sum()
    weight = weight/weight.max()
    blending2 = weight * LR + (1 - weight) * HR
    return blending2

def Backprojection(LR, HR, maxIter):
    H, W = LR.shape
    H1, W1 = HR.shape

    w = Gaussian2d((5,5), 10)
    w = w**2
    w = w/sum(np.ravel(w))
    for i in range(maxIter):
        im_L = imresize(HR, (H, W), interp='bicubic', mode='F')
        imd = LR - im_L
        im_d = imresize(imd, (H1, W1), interp='bicubic', mode='F')
        HR = HR + convolve2d(im_d, w, 'same')

    return HR


def Dog1(im):
    sigma = 0.85
    alpha = 1.414
    r = 15
    ksize = (3, 3)
    G1 = cv2.GaussianBlur(im, ksize, sigma)
    Ga1 = cv2.GaussianBlur(im, ksize, sigma*alpha)
    D1 = cv2.addWeighted(G1, 1+r, Ga1, -r, 0)

    G2 = cv2.GaussianBlur(Ga1, ksize, sigma)
    Ga2 = cv2.GaussianBlur(Ga1, ksize, sigma*alpha)
    D2 = cv2.addWeighted(G2, 1+r, Ga2, -r, 0)

    G3 = cv2.GaussianBlur(Ga2, ksize, sigma)
    Ga3 = cv2.GaussianBlur(Ga2, ksize, sigma * alpha)
    D3 = cv2.addWeighted(G3, 1+r, Ga3, -r, 0)

    B1 = Blending1(im, D3)
    B1 = Blending1(im, B1)
    B2 = Blending1(B1, D2)
    B2 = Blending1(im, B2)
    B3 = Blending1(B2, D1)
    B3 = Blending1(im, B3)

    output = B3

    return output

def Getfromsymmetry1(V, patchSize, t1, t2):
    V_sym = np.zeros((patchSize*patchSize,patchSize*patchSize))
    for i in range(1, patchSize*patchSize+1):
        for j in range(1, patchSize*patchSize+1):
            y1 = ceil(i/patchSize)
            x1 = i-(y1-1)*patchSize
            y2 = ceil(j/patchSize)
            x2 = j-(y2-1)*patchSize
            if (t1 == 1) and (t2 == 0):
                ig = patchSize * x1 + 1 - y1
                jg = patchSize * x2 + 1 - y2
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 2) and (t2 == 0):
                x = patchSize + 1 - x1
                y = patchSize + 1 - y1
                ig = (y - 1) * patchSize + x
                x = patchSize + 1 - x2
                y = patchSize + 1 - y2
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 3) and (t2 == 0):
                x = y1
                y = patchSize + 1 - x1
                ig =(y - 1) * patchSize + x
                x = y2
                y = patchSize + 1 - x2
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 0) and (t2 == 1):
                x = patchSize + 1 - x1
                y = y1
                ig =(y - 1) * patchSize + x
                x = patchSize + 1 - x2
                y = y2
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 1) and (t2 == 1):
                x0 = patchSize + 1 - x1
                y0 = y1
                x = patchSize + 1 - y0
                y = x0
                ig =(y - 1) * patchSize + x
                x0 = patchSize + 1 - x2
                y0 = y2
                x = patchSize + 1 - y0
                y = x0
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 2) and (t2 == 1):
                x0 = patchSize + 1 - x1
                y0 = y1
                x = patchSize + 1 - x0
                y = patchSize + 1 - y0
                ig =(y - 1) * patchSize + x
                x0 = patchSize + 1 - x2
                y0 = y2
                x = patchSize + 1 - x0
                y = patchSize + 1 - y0
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            elif (t1 == 3) and (t2 == 1):
                x0 = patchSize + 1 - x1
                y0 = y1
                x = y0
                y = patchSize + 1 - x0
                ig =(y - 1) * patchSize + x
                x0 = patchSize + 1 - x2
                y0 = y2
                x = y0
                y = patchSize + 1 - x0
                jg = (y - 1) * patchSize + x
                V_sym[ig - 1, jg - 1] = V[i - 1, j - 1]
            else:
                assert False
    return V_sym

def Getfromsymmetry2(V, patchSize, t1, t2):
    Vp = np.reshape(V, (patchSize, patchSize))
    V1 = np.rot90(Vp, t1)
    if t2 == 1:
        V1 = np.flip(V1, 1)
    V_sym = np.ravel(V1)
    return V_sym

# Quantization procedure to get the optimized strength and coherence boundaries
@nb.jit(nopython=True, parallel=True)
def QuantizationProcess (im_GX, im_GY,patchSize, patchNumber,w , quantization):
    H, W = im_GX.shape
    for i1 in range(H-2*floor(patchSize/2)):
        for j1 in range(W-2*floor(patchSize/2)):
            idx = (slice(i1,(i1+2*floor(patchSize/2)+1)),slice(j1,(j1+2*floor(patchSize/2)+1)))
            patchX = im_GX[idx]
            patchY = im_GY[idx]
            strength, coherence = Grad(patchX, patchY, w)
            quantization[patchNumber, 0] = strength
            quantization[patchNumber, 1] = coherence
            patchNumber += 1
    return quantization, patchNumber

# Training procedure for each image (use numba.jit to speed up)
@nb.jit(nopython=True, parallel=True)
def TrainProcess (im_LR, im_HR, im_GX, im_GY,patchSize, w, Qangle, Qstrength,Qcoherence, stre, cohe, R, Q, V, mark):
    H, W = im_HR.shape
    for i1 in range(H-2*floor(patchSize/2)):
        for j1 in range(W-2*floor(patchSize/2)):
            idx1 = (slice(i1,(i1+2*floor(patchSize/2)+1)),slice(j1,(j1+2*floor(patchSize/2)+1)))
            patch = im_LR[idx1]
            patchX = im_GX[idx1]
            patchY = im_GY[idx1]
            theta,lamda,u=HashTable(patchX, patchY, w, Qangle, Qstrength,Qcoherence, stre, cohe)
            patch1 = patch.ravel()
            patchL = patch1.reshape((1,patch1.size))
            t = (i1 % R) * R +(j1 % R)
            j = theta * Qstrength * Qcoherence + lamda * Qcoherence + u
            tx = np.int(t)
            jx = np.int(j)
            A = np.dot(patchL.T, patchL)
            Q[tx,jx] += A
            b1=patchL.T * im_HR[i1+floor(patchSize/2),j1+floor(patchSize/2)]
            b = b1.reshape((b1.size))
            V[tx,jx] += b
            mark[tx,jx] = mark[tx,jx]+1
    return Q,V,mark

