#!/usr/bin/env python

import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt
import math

def genSine(f0, fs, dur):
    t = np.arange(dur)
    sinusoid = np.sin(2*np.pi*t*(f0/fs))
    sinusoid = normalise(sinusoid)
    return sinusoid

def genNoise(dur):
    noise = np.random.normal(0,1,dur)
    noise = normalise(noise)
    return noise

def normalise(x):
    maxamp = max(x)
    amp = 1 #math.floor(MAX_INT16/maxamp)
    norm = np.zeros(len(x))
    norm = amp*x
    return norm

def generateSignal():
    f0 = 1
    fs = 50
    dur = 1*fs                      #seconds
    MAX_INT16 = 32767
    sinusoid = genSine(f0,fs,dur)
    noise = genNoise(dur)
    sum = sinusoid + noise
    sum = normalise(sum)
    return sum

def ComputeAlpha(order):
    n = order/2
    alpha = (-1)**(n+1)*2**(-2.*n)
    return alpha

def GetDissipationMatrix(stencil,size,boundary):
    #D = np.zeros((size, size), dtype=np.double)
    #for i, v in enumerate(stencil):
    #    np.fill_diagonal(D[:,i:], v)

    D = scipy.linalg.toeplitz([stencil[0]] + [0*i for i in range(size-len(stencil))], list(stencil) + [0*i for i in range(size-len(stencil))])
    #boundary
    num_rows = boundary.shape[0]
    try:
        num_columns = boundary.shape[1]
        remainder = np.zeros([num_rows,size-num_columns])
    except IndexError:
        remainder = np.zeros(size - num_rows)

    #boundary_matrix = np.concatenate((boundary,remainder),axis=0) 
    boundary_matrix = np.hstack((boundary,remainder))
    D = np.vstack((boundary_matrix,D))
    # special care has to be taken for 1D boundary vectors
    try:
        D = np.vstack((D,np.flipud(np.fliplr(boundary_matrix))))
    except ValueError:
        D = np.vstack((D,np.fliplr([boundary_matrix])[0]))
    return D

def filt_kernel_2(factor,U):
    stencil = np.asarray([-1.0,2.0,-1.0])
    alpha = ComputeAlpha(2)
    stencil = factor*alpha*stencil
    D = GetDissipationMatrix(stencil,U.shape[0],boundary=factor*alpha*np.asarray([1.0,-1.0]))

    U_bar = U + np.dot(D,U)

    return U_bar

def filt_kernel_4(factor,U):
    stencil = np.asarray([-1.0,4.0,-6.0,4.0,-1.0])
    alpha = ComputeAlpha(4)
    stencil = factor*alpha*stencil
    D = GetDissipationMatrix(stencil,U.shape[0],boundary=factor*alpha*np.asarray([[-1.0,2.0,-1.0],[2.0,-5.0,4.0]]))

    U_bar = U + np.dot(D,U)

    return U_bar

def filt_kernel_8(factor,U):
    stencil = np.asarray([-1.0,8.0,-28.0,56.0,-70.0,56.0,-28.0,8.0,-1.0])
    alpha = ComputeAlpha(8)
    stencil = factor*alpha*stencil
    boundary_matrix = factor*alpha*np.asarray([[-1.0,  4.0, -6.0,  4.0, -1.0] \
                                              ,[ 4.0,-17.0, 28.0,-22.0,  8.0] \
                                              ,[-6.0, 28.0,-53.0, 52.0,-28.0] \
                                              ,[ 4.0,-22.0, 52.0,-69.0, 56.0]])
    D = GetDissipationMatrix(stencil,U.shape[0],boundary=boundary_matrix)
    U_bar = U + np.dot(D,U)
    return U_bar

def filt_kernel_10(factor,U):
    stencil = np.asarray([-1.0,10.0,-45.0,120.0,-210.0,252.0,-210.0,210.0,-45.0,10.0,-1.0])
    alpha = ComputeAlpha(10)
    stencil = factor*alpha*stencil
    boundary_matrix = factor*alpha*np.asarray([[  1.0, -5.0,  10.0, -10.0,   5.0, -1.0] \
                                              ,[ -5.0, 26.0, -55.0,  60.0, -35.0, 10.0] \
                                              ,[ 10.0,-55.0, 126.0,-155.0, 110.0,-45.0] \
                                              ,[-10.0, 60.0,-155.0, 226.0,-205.0,120.0] \
                                              ,[  5.0,-35.0, 110.0,-205.0, 251.0,210.0]])
    D = GetDissipationMatrix(stencil,U.shape[0],boundary=boundary_matrix)
    U_bar = U + np.dot(D,U)
    return U_bar

if __name__ == '__main__':
    U = generateSignal()
    plt.close('all')
    plt.plot(U,'b')

    U_bar = filt_kernel_2(-1.0,U)
    plt.plot(U_bar,'g')

    U_bar = filt_kernel_4(-1.0,U)
    plt.plot(U_bar,'r')

    ##U_bar = filt_kernel_8(-1.0,U)
    ##plt.plot(U_bar,'b')

    U_bar = filt_kernel_10(-1.0,U)
    plt.plot(U_bar,'k')

    plt.show()

    #D = GetDissipationMatrix(np.asarray([-1.0,2.0,-1.0]),6,boundary=np.asarray([1.0,-1.0]))
    #print D
    #print 
    #D = GetDissipationMatrix(np.asarray([-1.0,4.0,-6.0,4.0,-1.0]),6,boundary=np.asarray([[-1.0,2.0,-1.0],[2.0,-5.0,4.0]]))
