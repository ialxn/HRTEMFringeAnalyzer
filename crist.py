#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 08:32:42 2017

@author: alxneit
"""
import warnings

import numpy as np
from numpy.fft import fft2, fftshift

from scipy.misc import imread
from scipy.optimize import curve_fit, OptimizeWarning

from numba import jit

import matplotlib.pyplot as plt

#
# for jacoby
# ddA = 1
# ddoffset = 1
# ddsigma = A*(mu - x)**2*exp(-0.5*(mu - x)**2/sigma**2)/sigma**3
# ddmu = -A*(mu - x)*exp(-0.5*(mu - x)**2/sigma**2)/sigma**2
#
@jit(nopython=True, nogil=True)
def gaussian(x, *p):
    """Fit gaussian and offset
    """
    A, mu, sigma, offset = p
    return A * np.exp(-(x - mu)**2 / (2.0 * sigma**2)) + offset


def FWHH(x, y):
    """Calculate peak and FWHH of ``y(x)`` by fitting a parabola
    to the region around the maximum.

    Parameters:
        x : array of floats
            x values
        y : array of floats
            y values (at x)

    Returns
        max_value : float
            x value for which y=f(x) is maximum
        delta_value : float
            FWHH
    """
    # determine interval that will be fitted
    # (use only data where y > mean)
    idx_max = y.argmax()
    m = np.isfinite(y)
    if len(m) < 4:
        # not enough data for fit, return x for which y is maximum
        max_value = x[idx_max]
        delta_value = float('nan')
    else:
        p0 = [y[idx_max],
              x[idx_max],
              x[m].ptp() / 2,
              np.nanmean(y)]
        try:
            warnings.simplefilter('ignore', OptimizeWarning)
            coeffs, cov = curve_fit(gaussian,
                                    x[m],
                                    y[m],
                                    p0=p0)
        except (ValueError, RuntimeError) as error:
            max_value = float('nan')
            delta_value = float('nan')
        else:
            # successful fit:
            #   maximum: in (validated) x-range and finite error
            #   delta: positive and error smaller than (validated) x-range
            if (coeffs[1] > x[m][0]) and (coeffs[1] < x[m][-1]) and np.isfinite(cov[1, 1]):
                max_value = coeffs[1]
            else:
                max_value = float('nan')
            if (coeffs[2] > 0.0) and (cov[2, 2] < x[m].ptp()):
                delta_value = coeffs[2]
            else:
                delta_value = float('nan')

    return max_value, delta_value


def noise_floor(s):
    """Determine aproximate noise floor of FFT ``s``

    Ad-hoc definition of the noise floor:
        + use all data in cornes of 2D FFT (i.e. outside of circle
          with radius ``R`` = ``FFT_SIZE//2``)
        + calculate mean and standard deviation
        + define noise floor as mean-value + 3*standard deviations
    """
    # ``x, y`` are pixel distances relative to origin (center) of ``spec``
    # the offset of 0.5 makes the center lies between pixels and ensures that
    # the distances from center to any of the sides is equal. with the offset
    # the minimum radius is 0.5 pixels, i.e. we are measuring the distance
    # to the center of the pixels.
    FFT_SIZE2 = s.shape[0]//2
    R2 = FFT_SIZE2 * FFT_SIZE2
    x, y = np.ogrid[-FFT_SIZE2 + 0.5 : FFT_SIZE2 + 0.5,
                    -FFT_SIZE2 + 0.5 : FFT_SIZE2 + 0.5]
    mask = ((x*x + y*y) >= R2)
    mean = s[mask].mean()
    error = s[mask].std()

    return mean + 3.0 * error


def analyze_direction(s):
    """Determine direction of periodicy in image ``s``

    Parameters:
        s : np.array
            2D Fourier transform

    Returns
        phi_max : float
            predominant direction of periodicity (0..pi)
        delta_phi : float
            FWHH of `phi_max``
    """
    FFT_SIZE2 = s.shape[0]//2
    N_BINS = 18 # 10 degrees per bin
    dphi = 2.0 * np.pi / N_BINS
    # ``x, y`` are pixel distances relative to origin (center) of ``spec``
    # the offset of 0.5 makes the center lies between pixels and ensures that
    # the distances from center to any of the sides is equal. with the offset
    # the minimum radius is 0.5 pixels, i.e. we are measuring the distance
    # to the center of the pixels.
    x, y = np.ogrid[-FFT_SIZE2 + 0.5 : FFT_SIZE2 + 0.5,
                    -FFT_SIZE2 + 0.5 : FFT_SIZE2 + 0.5]
    phi = np.arctan2(y, x)
    phi[phi < 0] += np.pi
    d, _ = np.histogram(phi.flatten(), bins=N_BINS, weights=s.flatten())

    if d.max() > 1.5 * np.nanmean(d):
        #   significant peak
        phi_max, delta_phi = FWHH(np.linspace(0.5*dphi, np.pi - 0.5*dphi, num=N_BINS), d)
    else:
        phi_max = float('nan')
        delta_phi = float('nan')

    return phi_max, delta_phi


def determine_lattice_const(s, r_min, r_max):
    """Determine lattice constant and coherence lenght from FFT. All calculations
    in pixel numbers.

    Parameters:
        s : np.array
            2D Fourier transform.
        r_min, r_max : float
            only data between ``r_min`` and ``r_max`` are non-zero (valid)
    """
    FFT_SIZE2 = s.shape[0]//2
    n_r = int(np.around(10 * (r_max - r_min)))  # ad hoc definition (10)
    dr = (r_max - r_min) / n_r
    #
    # bin edges to ensure that ``n_r`` bins cover ``r_min`` - ``r_max`` and
    # include both endpoints
    #
    bins = np.linspace(r_min, r_max, n_r + 1, endpoint=True)

    #
    # ``x, y`` are pixel distances relative to origin (center) of ``spec``
    # the offset of 0.5 makes the center lies between pixels and ensures that
    # the distances from center to any of the sides is equal. with the offset
    # the minimum radius is 0.5 pixels, i.e. we are measuring the distance
    # to the center of the pixels.
    #
    x, y = np.ogrid[-FFT_SIZE2 + 0.5 : FFT_SIZE2 + 0.5,
                    -FFT_SIZE2 + 0.5 : FFT_SIZE2 + 0.5]
    r = np.sqrt(x*x + y*y)

    #
    # weights should  include 1/r^2
    # we integrate azimutally, thus noise at large ``r`` contributes more
    # than noise (or signal) at small ``r``
    radius, _ = np.histogram(r.flatten(),
                             bins=bins,
                             weights=s.flatten() / r.flatten()**2)

    #
    # calculate noise level from mean of last 3 values
    #
    if radius.max() > 2.0 * np.nanmean(radius):
        # significant peak
        d, delta_d = FWHH(np.linspace(r_min + 0.5*dr, r_max -0.5*dr, num=n_r),
                          radius)
        d = 2.0 * FFT_SIZE2 / d
    else:
        d = float('nan')
        delta_d = float('nan')

    return d, delta_d




def analyze(im, r_min, r_max, fft_size, step):
    """Analyze local crystallinity of image ``im``

    Parameters:
        im : np array
            Image to be analyzed.
        r_min, r_max : float
            Minimum, maximum of frequency (in pixels of FFT) to be analyzed.
            Calculated form min/max of period to be analyzed as
            freq = fft_size / period
        fft_size : int
            size of widow to be analyzed (must be 2^N)
        step : int
            Window is translated by ``step`` (x and y)

    Returns:


    """
    R_MIN2 = r_min**2
    R_MAX2 = r_max**2
    FFT_SIZE2 = fft_size//2

    # x-axis is im.shape[1] -> horizontal (left->right)
    # y-axis is im.shape[0] -> vertical (top->down)
    # indices v,h for center of roi in image
    # indices rv, rh for result arrays
    d = np.zeros([(im.shape[0] - fft_size) // step + 1,
                  (im.shape[1] - fft_size) // step + 1])
    delta_d = np.zeros([(im.shape[0] - fft_size) // step + 1,
                        (im.shape[1] - fft_size) // step+ 1])
    phi = np.zeros([(im.shape[0] - fft_size) // step + 1,
                    (im.shape[1] - fft_size) // step + 1])
    delta_phi = np.zeros([(im.shape[0] - fft_size) // step + 1,
                          (im.shape[1] - fft_size) // step+ 1])

    for rv, v in enumerate(range(FFT_SIZE2,
                                 im.shape[0] - FFT_SIZE2,
                                 step)):
        for rh, h in enumerate(range(FFT_SIZE2,
                                     im.shape[1] - FFT_SIZE2,
                                     step)):

            roi = im[v-FFT_SIZE2 : v+FFT_SIZE2,
                     h-FFT_SIZE2 : h+FFT_SIZE2]
            spec = fftshift(np.abs(fft2(roi-roi.mean())))
            level = noise_floor(spec)

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

            # set all pixels below noise floor ``level`` to zero
            spec[spec <= level] = 0

            d[rv, rh], delta_d[rv, rh] = determine_lattice_const(spec, r_min, r_max)
            phi[rv, rh], delta_phi[rv, rh] = analyze_direction(spec)

    return d, delta_d, phi, delta_phi


def sub_imageplot(data, ax, title, vmin, vmax):
    """Plot image ``data`` at axes instance ``ax``. Add title ``title`` and
    use scale ``vmin`` .. ``vmax``
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    im = ax.imshow(data, cmap='jet', vmin=vmin, vmax=vmax, origin='upper')
    # Create divider for existing axes instance
    divider = make_axes_locatable(ax)
    # Append axes to the right of ax, with 20% width of ax
    cax = divider.append_axes("right", size="20%", pad=0.05)
    # Create colorbar in the appended axes
    # Tick locations can be set with the kwarg `ticks`
    # and the format of the ticklabels with kwarg `format`
    plt.colorbar(im, cax=cax)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

if __name__ == '__main__':
    from argparse import ArgumentParser

    supported = ','.join(plt.figure().canvas.get_supported_filetypes())
    plt.close()

    parser = ArgumentParser(description='Analyze local cristallinity of data')
    parser.add_argument('-a', '--autoscale', metavar='KEY',
                        type=str, default='X',
                        help='autoscale color bar [DCPSA]')
    parser.add_argument('-f', '--file', metavar='FILE',
                        type=str, required=True,
                        help='Name of data file')
    parser.add_argument('-F', '--FFT_size', metavar='N',
                        type=int, default=128,
                        help='Size of moving window (NxN) [128].')
    parser.add_argument('-m', '--d_min', metavar='R.r',
                        type=float, default=0.0,
                        help='Minimum period (in pixels) to be evaluated')
    parser.add_argument('-M', '--d_max', metavar='R.r',
                        type=float, default=5.0,
                        help='Maximum period (in pixels) to be evaluated')
    parser.add_argument('-s', '--step', metavar='S',
                        type=int, default=25,
                        help='Step size (x and y) in pixels of moving window')
    parser.add_argument('-o', '--output', metavar='FILE', type=str,
                        help='Output to file. Supported formats: ' + supported)
    args = parser.parse_args()

    data = imread(args.file)
    d_value, coherence, direction, spread = analyze(data,
                                                    args.FFT_size / args.d_max,
                                                    args.FFT_size / args.d_min,
                                                    args.FFT_size,
                                                    args.step)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    if ('A' in args.autoscale) or ('D' in args.autoscale):
        sub_imageplot(d_value, ax1, 'd_values',
                      np.nanmin(d_value), np.nanmax(d_value))
    else:
        sub_imageplot(d_value, ax1, 'd_values',
                      args.d_min, args.d_max)

    if ('A' in args.autoscale) or ('C' in args.autoscale):
        sub_imageplot(coherence, ax2, 'coherence',
                      np.nanmin(coherence), np.nanmax(coherence))
    else:
        sub_imageplot(coherence, ax2, 'coherence',
                      0.0, 1.0)

    if ('A' in args.autoscale) or ('P' in args.autoscale):
        sub_imageplot(direction, ax3, 'direction',
                      np.nanmin(direction), np.nanmax(direction))
    else:
        sub_imageplot(direction, ax3, 'direction',
                      0.0, np.pi)

    if ('A' in args.autoscale) or ('S' in args.autoscale):
        sub_imageplot(spread, ax4, 'spread',
                      np.nanmin(spread), np.nanmax(spread))
    else:
        sub_imageplot(spread, ax4, 'spread',
                      0.0, 1.0)

    plt.tight_layout()

    if args.output is None:
        plt.show()
    else:
        try:
            plt.savefig(args.output)
        except ValueError:
            print('Cannot save figure ({})'.format(args.output))
            print('Supported formats: {}'.format(supported))
            plt.show()
