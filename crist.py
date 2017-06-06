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
    DX = s.shape[0]//2
    DY = s.shape[1]//2
    F = (N-1) / np.pi
    direct = np.zeros(N)
    for x in range(s.shape[0]):
        for y in range(s.shape[1]):
            if np.isclose(s[x][y], 0.0):
                continue
            phi = np.arctan2(y-DY, x-DX)
            idx = int(np.floor(phi * F))
            direct[idx] += s[x][y]

    phi_max = np.argmax(direct) * 2.0 * np.pi
    return phi_max

def determine_lattice_const(s, r_min, r_max, n_r):
    """Determine lattice constant and coherence lenght from FFT. All calculations
    in pixel numbers.

    Parameters:
        s : np.array
            2D Fourier transform.
        r_min, r_max : float
            only data between ``r_min`` and ``r_max`` are non-zero (valid)
        n_r : int
            number of bins used in interval ``r_min`` - ``r_max``
    """
    FFT_SIZE2 = s.shape[0]//2

    # set number of bins to ensure that ``n_r`` bins cover ``r_min`` - ``r_max``
    N_BINS = int(np.ceil(np.sqrt(2)*FFT_SIZE2*n_r / (r_max - r_min)))
    # ``x, y`` are pixel indices relative to origin (cetner) of ``spec``
    x, y = np.ogrid[-FFT_SIZE2 : FFT_SIZE2,
                    -FFT_SIZE2 : FFT_SIZE2]
    r = np.sqrt(x*x + y*y)

    radius, _ = np.histogram(r.flatten(), bins=N_BINS, weights=s.flatten())
    idx_d = radius.argmax()
    # TODO: Umrechnen von pixel auf nm
    d = 1.0 * idx_d
    # TODO: calculate delta_d
    delta_d = 0

    return d, delta_d




def analyze(im, r_min, r_max, fft_size, step):
    """Analyze local crystallinity of image ``im``

    Parameters:
        im : np array
            Image to be analyzed.
        r_min, r_max : float
            Minimum, maximum of circle (in pixels of FFT) to be analyzed
        fft_size : int
            size of widow to be analyzed (must be 2^N)
        step : int
            Window is translated by ``step`` (x and y)

    Returns:


    """

    R_MIN2 = r_min**2
    R_MAX2 = r_max**2
    FFT_SIZE2 = fft_size//2

    d = np.zeros([(im.shape[0] - fft_size) // step + 1,
                  (im.shape[1] - fft_size) // step + 1])
    delta_d = np.zeros([(im.shape[0] - fft_size) // step + 1,
                        (im.shape[1] - fft_size) // step+ 1])

    for ri, i in enumerate(range(FFT_SIZE2,
                                 im.shape[0] - FFT_SIZE2,
                                 step)):
        for rj, j in enumerate(range(FFT_SIZE2,
                                     im.shape[1] - FFT_SIZE2,
                                     step)):

            roi = im[i-FFT_SIZE2 : i+FFT_SIZE2,
                     j-FFT_SIZE2 : j+FFT_SIZE2]
            spec = fftshift(np.abs(fft2(roi-roi.mean())))

            # select pixels that are between ``r_min`` and ``r_max`` pixels away
            # from center of ``spec`` at ``(FFT_SIZE2, FFT_SIZE2)`` i.e. set
            # other pixels to zero.
            #
            # r_min^2 < (i-FFT_SIZE2)^2 + (i-FFT_SIZE2)^2 < r_max^2

            # ``x, y`` are pixel indices relative to origin (cetner) of ``spec``
            x, y = np.ogrid[-FFT_SIZE2 : FFT_SIZE2,
                            -FFT_SIZE2 : FFT_SIZE2]
            mask = ~((x*x + y*y > R_MIN2) & (x*x + y*y < R_MAX2))
            spec[mask] = 0

            # only pixels between ``r_min`` and ``r_max`` are non-zero

            d[ri][rj], delta_d[ri][rj] = determine_lattice_const(spec, r_min, r_max, 11)

    return d, delta_d

FFT_SIZE = 64
R_MIN = 10
data = imread('../1.tif')
R_MAX = 20
STEP = 64

crist, directions = analyze(data, R_MIN, R_MAX, FFT_SIZE, STEP)

plt.imshow(crist)
plt.imshow(directions)
