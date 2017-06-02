#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 08:32:42 2017

@author: alxneit
"""
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from scipy.misc import imread

#def analyze_ring(s, d_r, d_phi):



def analyze(im, r_min, r_max, fft_size, step):

    R_MIN2 = r_min**2
    R_MAX2 = r_max**2
    FFT_SIZE2 = fft_size//2

    c = np.zeros([(im.shape[0]-fft_size)//step+1, (im.shape[1]-fft_size)//step+1])
    #d = np.zeros([(im.shape[0]-fft_size)//step+1, (im.shape[1]-fft_size)//step+1])

    for ri, i in enumerate(range(FFT_SIZE2, im.shape[0]-FFT_SIZE2, step)):
        for rj, j in enumerate(range(FFT_SIZE2, im.shape[1]-FFT_SIZE2, step)):

            roi = im[i-FFT_SIZE2:i+FFT_SIZE2, j-FFT_SIZE2:j+FFT_SIZE2]
            spec = fftshift(np.abs(fft2(roi-roi.mean())))
            total=spec.sum()
            #
            # select pixels that are between 40 and 50 pixels away
            # from center of spec at (FFT_SIZE2, FFT_SIZE2) i.e. set
            # other pixels to zero.
            #
            # R_MIN^2 < (i-FFT_SIZE2)^2 + (i-FFT_SIZE2)^2 < R_MAX^2
            x, y = np.ogrid[-FFT_SIZE2:FFT_SIZE2, -FFT_SIZE2:FFT_SIZE2]
            mask = ~((x*x + y*y > R_MIN2) & (x*x + y*y < R_MAX2))
            spec[mask] = 0

            c[ri][rj] = spec.sum()/total
#            d[ri][rj] = analyze_ring(spec, r_max-r_min, 10)

    return c

data=imread('1.tif')
FFT_SIZE = 64
R_MIN = 10
R_MAX = 20
STEP = 64

crist = analyze(data, R_MIN, R_MAX, FFT_SIZE, STEP)

plt.imshow(crist)

