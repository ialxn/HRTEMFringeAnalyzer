#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 08:32:42 2017

@author: Ivo Alxneit (ivo.alxneit@psi.ch)
"""
import warnings

from argparse import ArgumentParser

from math import sqrt
import numpy as np
from numpy.fft import fft2, fftshift, fftfreq

from scipy.misc import imread
from scipy.optimize import curve_fit, OptimizeWarning

from numba import jit
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

__version__ = ''

# tuning knobs
# ``_tune_threshold_direction``: a valid peak along the azimut must be higher
#                                than ``_tune_threshold_direction`` times the
#                                mean intensity
# ``_tune_threshold_period``: a valid peak along the radius must be higher
#                             than ``_tune_threshold_period`` times the
#                             mean intensity
#
_tune_threshold_direction = 2.5
_tune_threshold_period = 10.0
#
@jit(nopython=True, nogil=True, cache=True)
def gaussian(x, *p):
    """Calculate Gaussian and offset

    Parameters:
        x : float
            x-value
        p : list of (four) float parameters
            A : area
            mu : x_center
            sigma : FWHH
            offset : baseline offset

    Returns
        y = f(x, p)
    """
    A, mu, sigma, offset = p
    return A * np.exp(-(x - mu)**2 / (2.0 * sigma**2)) + offset


def find_peak(x, y):
    """Calculate peak and FWHH of ``y(x)`` by fitting a gaussian plus offset
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
    idx_max = np.argmax(y)
    p0 = [y[idx_max],
          x[idx_max],
          x.ptp() / 10.0,
          y.mean()]
    try:
        warnings.simplefilter('ignore', OptimizeWarning)
        coeffs, cov = curve_fit(gaussian,
                                x,
                                y,
                                p0=p0)
    except (ValueError, RuntimeError):
        max_value = float('nan')
        delta_value = float('nan')
    else:
        # successful fit:
        #   maximum: in (validated) x-range and finite error
        #   delta: positive and error smaller than (validated) x-range
        if (coeffs[1] > x[0]) and (coeffs[1] < x[-1]) and np.isfinite(cov[1, 1]):
            max_value = coeffs[1]
        else:
            max_value = float('nan')

        if coeffs[2] > 0.0:
            try:
                err = sqrt(cov[2, 2])
            except ValueError:
                delta_value = float('nan')
            else:
                if err < x.ptp():
                    delta_value = coeffs[2]
                else:
                    delta_value = float('nan')
        else:
            delta_value = float('nan')

    return max_value, delta_value


def noise_floor(window, radius_squared):
    """Determine aproximate noise floor of FFT ``window``

    Parameters:
        window : np array
            Window to be analyzed
        radius_squared : np.array
            squared distances of data points to center of ``s``

    Ad-hoc definition of the noise floor:
        + use all data in cornes of 2D FFT (i.e. outside of circle
          with radius ``R`` = ``FFT_SIZE//2``)
        + calculate mean and standard deviation
        + define noise floor as mean-value + 3*standard deviations

    Returns
        noise_floor : float
            mean + 3*sigma
    """
    mask = (radius_squared >= (window.shape[0]//2)**2)
    mean = window[mask].mean()
    error = window[mask].std()

    return mean + 3.0 * error


def analyze_direction(window, radius_squared, phi):
    """Find peak in FFT window ``window`` and return its direction
    and angular spread

    Parameters:
        window : np.array
            2D Fourier transform
        radius_squared : np.array
            squared distances of each pixel in the 2D FFT relative to the one
            that represents the zero frequency
        phi : np.array
            angle of each pixel in 2D FFT relative to the pixel that
            represents the zero frequency

    Returns
        omega : float
            predominant direction of periodicity (0..pi)
        delta_phi : float
            FWHH of ``omega``
    """
    bins = 36 # 10 degrees per bin
    warnings.simplefilter('ignore', RuntimeWarning)
    angle, edges = np.histogram(phi.flatten(), bins=bins,
                                weights=window.flatten() / radius_squared.flatten())

    if np.nanmax(angle) > _tune_threshold_direction * np.nanmean(angle):
        #   significant peak
        # replace boundaries by center of bins
        edges += 0.5 * (edges[1] - edges[0])
        # peak could lie close to zero or pi, which makes fitting a gaussian
        # impossible (or problematic at least).
        # if peak is in a unproblematic position (central two quarters, bins 9-26)
        # pi/4 < peak_position < 3pi/4 then do nothing
        # if peak lies in first or last quarter (bins 0-8 or 27-35)
        # do wrap around, i.e. use quarters 2,3,0,1 (in this order) and analyze
        # pi/2 < peak_position < 3pi/2
        idx = np.nanargmax(angle)
        if (idx < 9) or (idx >= 27):
            # note: len(edges) == bins+1! after the shift by half a bin width
            #       edges[-1] is the offset to be added for the wrap-around
            edges = np.append(edges[18:-1], edges[0:18] + edges[-1])
            angle = np.append(angle[18:], angle[0:18])
        else:
            # remove last element of edges to have len(edges) == bins
            edges = np.append(edges[:-1],[])

        # replace 'nan' by linear interpolation between its neighbors
        mask = np.isnan(angle)
        angle[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), angle[~mask])

        omega, delta_omega = find_peak(edges, angle)

        # because of the wrap-around omega could be larger than pi
        if omega > np.pi:
            omega -= np.pi
    else:
        omega = float('nan')
        delta_omega = float('nan')

    return omega, delta_omega


def determine_lattice_const(window, radius_squared):
    """Determine lattice constant and coherence lenght from FFT. All calculations
    in pixel numbers.

    Parameters:
        window : np.array
            2D Fourier transform.
        radius_squared : np.array
            squared distances of each pixel in the 2D FFT relative to the one
            that represents the zero frequency

    Returns
        d : float
            Periode found
        delta_d : float
            Coherence length (length of periodic structure) as A.U.
    """
    bins = window.shape[0]//2  # ad hoc definition
    #
    # weights should  include 1/r^2
    # we integrate azimutally, thus noise at large ``r`` contributes more
    # than noise (or signal) at small ``r``
    warnings.simplefilter('ignore', RuntimeWarning)
    radius, edges = np.histogram(np.sqrt(radius_squared).flatten(),
                                 bins=bins,
                                 weights=window.flatten() / radius_squared.flatten())

    if np.nanmax(radius) > _tune_threshold_period * np.nanmean(radius):
        # significant peak
        # replace boundaries by center of bins
        edges += 0.5 * (edges[1] - edges[0])
        # replace 'nan' by linear interpolation between its neighbors
        mask = np.isnan(radius)
        radius[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), radius[~mask])

        d, delta_d = find_peak(edges[:-1], radius)
        d = 1.0 / np.interp(d,
                            np.arange(window.shape[0]//2),
                            fftfreq(window.shape[0])[0 : window.shape[0]//2])
        delta_d = 1.0 / delta_d
    else:
        d = float('nan')
        delta_d = float('nan')

    return d, delta_d


def inner_loop(v, im, fft_size, step, r2, phi, mask, han2d):
    """
    Analyzes horizontal line ``v`` in image ``im``

    Parameters:
        v : int
            Line number (horizontal index) to be analyzed
        im : np array
            Image to be analyzed
        fft_size : int
            width of window to be analyzed (2^N x 2^N)
        step : int
            Horizontal (and vertical) step size to translate
            window
        r2 : np.array
            squared distances of each pixel in the 2D FFT relative to the one
            that represents the zero frequency
        phi : np.array
            angle of each pixel in 2D FFT relative to the pixel that
            represents the zero frequency
        mask : np.array
            mask that discards very low frequencies (index 1,2) and high
            frequencies (indices above FFT_SIZE2)
        han2d : np.array
            2D Hanning window applied to roi before the 2D FFT

    Returns
        List of np arrays of length ``Nh``
        d : np array
            Periode found
        delta_d : np array
            Coherence length (length of periodic structure) as A.U.
        omega : np array
            Direction of lattice (direction) vector of periodic structure
        delta_omega : np array
            Spread of direction vector
    """
    FFT_SIZE2 = fft_size//2
    Nh = int(np.ceil((im.shape[1] - fft_size) / step))
    d = np.zeros([Nh])
    delta_d = np.zeros([Nh])
    omega = np.zeros([Nh])
    delta_omega = np.zeros([Nh])

    for rh, h in enumerate(range(FFT_SIZE2,
                                 im.shape[1] - FFT_SIZE2,
                                 step)):

        roi = im[v-FFT_SIZE2 : v+FFT_SIZE2,
                 h-FFT_SIZE2 : h+FFT_SIZE2]
        spec = fftshift(np.abs(fft2(han2d * (roi-roi.mean()))))
        level = noise_floor(spec, r2)

        spec[mask] = 0
        spec[spec <= level] = 0

        d[rh], delta_d[rh] = determine_lattice_const(spec, r2)
        omega[rh], delta_omega[rh] = analyze_direction(spec, r2, phi)

    return (d, delta_d, omega, delta_omega)


def analyze(im, fft_size, step, n_jobs):
    """Analyze local crystallinity of image ``im``

    Parameters:
        im : np array
            Image to be analyzed.
        fft_size : int
            Size of window to be analyzed (must be 2^N)
        step : int
            Window is translated by ``step`` (horizontal and vertical)
        n_jobs : int
            Number of parallel running jobs

    Returns
        d : np array
            Periode
        delta_d : np array
            Coherence length (length of periodic structure) as A.U.
        omega : np array
            Direction of lattice (direction) vector of periodic structure
        delta_omega : np array
            Spread of direction vector
    """
    FFT_SIZE2 = fft_size//2
    # x-axis is im.shape[1] -> horizontal (left->right)
    # y-axis is im.shape[0] -> vertical (top->down)
    # indices v,h for center of roi in image
    # indices rv, rh for result arrays
    Nh = int(np.ceil((im.shape[1] - fft_size) / step))
    Nv = int(np.ceil((im.shape[0] - fft_size) / step))

    ###########################################################################
    # prepare arrays that are needed many times to deal with the 2D fourier
    # transforms
    # x, y are indices of the frequency shifted transforms, i.e. the
    # zero frequency (DC term) is now at [0,0]
    x, y = np.ogrid[-FFT_SIZE2 : FFT_SIZE2,
                    -FFT_SIZE2 : FFT_SIZE2]
    # r2: the geometrical distance (index) squared of a given intry in the FFT
    r2 = x*x + y*y
    # phi: the geometrical azimut of a given intry in the FFT with the range
    #      -pi .. phi .. 0 mapped to 0 .. phi .. pi
    phi = np.arctan2(x, y)
    phi[phi < 0] += np.pi

    # mask: discard very low frequencies (index 1,2) and high frequencies
    # (indices above FFT_SIZE2)
    mask = ~((r2 > 16) & (r2 < FFT_SIZE2**2))
    # 2D hanning window
    han = np.hanning(fft_size)
    han2d = np.sqrt(np.outer(han, han))
    ###########################################################################

    with Parallel(n_jobs=n_jobs) as parallel:
        res = parallel(delayed(inner_loop)(v,
                                           im,
                                           fft_size, step,
                                           r2, phi, mask, han2d) \
                                           for rv, v, in enumerate(range(FFT_SIZE2,
                                                                         im.shape[0] - FFT_SIZE2,
                                                                         step)))
        d, delta_d, omega, delta_omega = zip(*res)

    return (np.array(d).reshape(Nv, Nh),
            np.array(delta_d).reshape(Nv, Nh),
            np.array(omega).reshape(Nv, Nh),
            np.array(delta_omega).reshape(Nv, Nh))


def sub_imageplot(data, ax, title):
    """Plot image ``data`` at axes instance ``ax``. Add title ``title`` and
    use scale ``vmin`` .. ``vmax``
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if title == 'direction':
        im = ax.imshow(data, cmap='jet', vmin=0, vmax=np.pi, origin='upper')
    else:
        im = ax.imshow(data, cmap='jet', origin='upper')
    # Create divider for existing axes instance
    divider = make_axes_locatable(ax)
    # Append axes to the right of ax, with 20% width of ax
    cax = divider.append_axes("right", size="20%", pad=0.05)
    # Create colorbar in the appended axes
    # Tick locations can be set with the kwarg `ticks`
    # and the format of the ticklabels with kwarg `format`
    if title == 'direction':
        ticks = np.linspace(0, np.pi, num=9, endpoint=True)
        labels = ['W', '', 'NW', '', 'N', '', 'NE', '', 'E']
        cbar = plt.colorbar(im, cax=cax, ticks=ticks)
        cbar.ax.set_yticklabels(labels)
        cbar.set_label('direction  [-]')
    else:
        cbar = plt.colorbar(im, cax=cax)
        if title == 'd_values':
            cbar.set_label('d value  [pixel]')
        if title == 'coherence':
            cbar.set_label('coherence  [A.U.]')
        if title == 'spread':
            cbar.set_label(r'$\sigma_\mathrm{dir}$  [$^\circ$]')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def main():
    """main function
    """
    supported = ','.join(plt.figure().canvas.get_supported_filetypes())
    plt.close()

    parser = ArgumentParser(description='Analyze local cristallinity of data')
    parser.add_argument('-f', '--file', metavar='FILE',
                        type=str, required=True,
                        help='Name of data file. Remove outliers (bright and dark) before.')
    parser.add_argument('-F', '--FFT_size', metavar='N',
                        type=int, default=128,
                        help='Size of moving window (NxN) [128].')
    parser.add_argument('-j', '--jobs', metavar='N',
                        type=int, default=1,
                        help='Number of threads to be started [1].')
    parser.add_argument('-s', '--step', metavar='S',
                        type=int, default=32,
                        help='Step size (x and y) in pixels of moving window [32]')
    parser.add_argument('-S', '--save',
                        action='store_true', default=False,
                        help='Store result in gzipped text files')
    parser.add_argument('-o', '--output', metavar='FILE', type=str,
                        help='Output to file. Supported formats: ' + supported)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    args = parser.parse_args()

    data = imread(args.file, mode='I')
    d_value, coherence, direction, spread = analyze(data,
                                                    args.FFT_size,
                                                    args.step,
                                                    args.jobs)

    if args.save:
        header = '#\n' \
        + '# crist.py version {}\n'.format(__version__) \
        + '# Results for {} ({}x{} [vxh])\n'.format(args.file, data.shape[0], data.shape[1]) \
        + '# FFT window: {}x{}\n'.format(args.FFT_size, args.FFT_size) \
        + '#       step: {}\n'.format(args.step) \
        + '#\n' \
        + '# To convert from local indices to indices of original image\n' \
        + '#\n' \
        + '#     image_idx = FFT_window/2 + local_idx * step\n' \
        + '# both, for horizontal and vertical index\n' \
        + '#'
        base_name, _ = args.file.rsplit(sep='/')[-1].rsplit(sep='.')
        np.savetxt(base_name + '_period' + '.dat.gz', d_value,
                   delimiter='\t', header=header, comments='')
        np.savetxt(base_name + '_coherence' + '.dat.gz', coherence,
                   delimiter='\t', header=header, comments='')
        np.savetxt(base_name + '_direction' + '.dat.gz', direction,
                   delimiter='\t', header=header, comments='')
        np.savetxt(base_name + '_spread' + '.dat.gz', spread,
                   delimiter='\t', header=header, comments='')

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    sub_imageplot(d_value, ax1, 'd_values')
    sub_imageplot(coherence, ax2, 'coherence')
    sub_imageplot(direction, ax3, 'direction')
    sub_imageplot(np.rad2deg(spread), ax4, 'spread')

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


if __name__ == '__main__':
    #
    # tuning knob for finding direction of the periodic pattern
    _tune_direction_treshhold = 3.0
    # tuning knob for finding the d-value of the periodic pattern
    _tune_period_treshhold = 12.0
    #
    main()
