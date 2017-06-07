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
    N_BINS = 36
    DELTA_PHI = np.pi / N_BINS
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
    idx_d = d.argmax()
    phi_max = idx_d * DELTA_PHI
    #TODO: calculate delta_phi (FWHH)
    delta_phi = 0

    return phi_max, delta_phi


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

    # bin edges to ensure that ``n_r`` bins cover ``r_min`` - ``r_max`` and
    # include both endpoints
    bins = np.linspace(r_min, r_max, n_r + 1, endpoint=True)
    # ``x, y`` are pixel distances relative to origin (center) of ``spec``
    # the offset of 0.5 makes the center lies between pixels and ensures that
    # the distances from center to any of the sides is equal. with the offset
    # the minimum radius is 0.5 pixels, i.e. we are measuring the distance
    # to the center of the pixels.
    x, y = np.ogrid[-FFT_SIZE2 + 0.5 : FFT_SIZE2 + 0.5,
                    -FFT_SIZE2 + 0.5 : FFT_SIZE2 + 0.5]
    r = np.sqrt(x*x + y*y)
    radius, _ = np.histogram(r.flatten(), bins=bins, weights=s.flatten())

    if radius.max() >= 2.0 * radius.mean():
        # significant peak
        idx_d = radius.argmax()
        # TODO: Umrechnen von pixel auf nm
        d = 1.0 * idx_d
        # TODO: calculate delta_d
        delta_d = 0
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
    phi = np.zeros([(im.shape[0] - fft_size) // step + 1,
                    (im.shape[1] - fft_size) // step + 1])
    delta_phi = np.zeros([(im.shape[0] - fft_size) // step + 1,
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

            d[ri][rj], delta_d[ri][rj] = determine_lattice_const(spec, r_min, r_max, 11)
            phi[ri][rj], delta_phi[ri][rj] = analyze_direction(spec)

    return d, delta_d, phi, delta_phi


def sub_imageplot(data, ax, title, vmin, vmax):
    """Plot image ``data`` at axes instance ``ax``. Add title ``title`` and
    use scale ``vmin`` .. ``vmax``
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    im = ax.imshow(data, cmap='bwr', vmin=vmin, vmax=vmax)
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
    parser.add_argument('-f', '--file', metavar='FILE',
                        type=str, required=True,
                        help='Name of data file')
    parser.add_argument('-F', '--FFT_size', metavar='N',
                        type=int, default=128,
                        help='Size of moving window (NxN) [128].')
    parser.add_argument('-r', '--r_min', metavar='R.r',
                        type=float, default=0.0,
                        help='Minimum radius (of FFT in pixels) to be evaluated')
    parser.add_argument('-R', '--r_max', metavar='R.r',
                        type=float, default=10.0,
                        help='Maximum radius (of FFT in pixels) to be evaluated')
    parser.add_argument('-s', '--step', metavar='S',
                        type=int, default=64,
                        help='Step size (x and y) in pixels of moving window')
    parser.add_argument('-o', '--output', metavar='FILE', type=str,
                        help='Output to file. Supported formats: ' + supported)
    args = parser.parse_args()

    data = imread(args.file)
    d_value, coherence, direction, spread = analyze(data,
                                                    args.r_min, args.r_max,
                                                    args.FFT_size,
                                                    args.step)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    sub_imageplot(d_value, ax1, 'd_values', 1.0, 20.0)
    sub_imageplot(coherence, ax2, 'coherence', 0.0, 1.0)
    sub_imageplot(direction, ax3, 'direction', 0.0, np.pi)
    sub_imageplot(spread, ax4, 'spread', 0.0, 1.0)

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
