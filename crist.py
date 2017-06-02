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

def direction(s):

    N = 36
    direct = np.zeros(N)
    for x in range(s.shape[0]):
        for y in range(s.shape[1]):
            if np.isclose(s[x][y],0.0):
                continue
            phi = np.arctan2(y-s.shape[1]//2,x-s.shape[0]//2)
            idx =int(np.floor(phi / np.pi) * (N-1))
            direct[idx] = s[x][y]

    phi_max = np.argmax(direct) * 2.0 * np.pi
    return phi_max




def analyze(im, r_min, r_max, fft_size, step):

    R_MIN2 = r_min**2
    R_MAX2 = r_max**2
    FFT_SIZE2 = fft_size//2

    c = np.zeros([(im.shape[0]-fft_size)//step+1, (im.shape[1]-fft_size)//step+1])
    d = np.zeros([(im.shape[0]-fft_size)//step+1, (im.shape[1]-fft_size)//step+1])

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
            d[ri][rj] = direction(spec)
    return c, d

FFT_SIZE = 64
R_MIN = 10
data=imread('../1.tif')
R_MAX = 20
STEP = 64

crist, directions = analyze(data, R_MIN, R_MAX, FFT_SIZE, STEP)

plt.imshow(crist)
plt.imshow(directions)

