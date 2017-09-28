#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:05:26 2017

@author: alxneit
"""
import sys
import warnings

import numpy as np
from numpy.fft import fft2, fftshift

import matplotlib.pyplot as plt

from scipy.misc import imread
from scipy.optimize import curve_fit, OptimizeWarning

from numba import jit
from joblib import Parallel, delayed

@jit(nopython=True, nogil=True, cache=True)
def __gaussian(x, *p):
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
    A, x_0, sigma, offset = p
    # 4*ln2 = 2.7725887 ensures that sigma = FWHH
    factor = 2.7725887
    return A * np.exp(-factor*((x - x_0) / sigma)**2) + offset


def __find_peak(x, y):
    """Determines peak position and FWHH

    Parameters:
        x : array of floats
            x values
        y : array of floats
            corresponding y values

    Returns
        x_center : float
            x value for which y=f(x) is maximum
        sigma : float
            FWHH
    """
    idx_max = np.argmax(y)
    p_0 = [y[idx_max],
           x[idx_max],
           x.ptp() * 0.05,
           y.mean() * 0.5]
    try:
        warnings.simplefilter('ignore', OptimizeWarning)
        coeffs, cov = curve_fit(__gaussian,
                                x,
                                y,
                                p0=p_0)
    except (ValueError, RuntimeError):
        x_center = np.nan
        sigma = np.nan
    else:
        # successful fit:
        #   max_value: in (validated) x-range and finite error
        #   delta_value: positive and covariance is positive
        if (coeffs[1] > x[0]) and (coeffs[1] < x[-1]) and np.isfinite(cov[1, 1]):
            x_center = coeffs[1]
        else:
            x_center = np.nan

        if (coeffs[2] > 0.0) and (cov[2, 2] > 0.0):
            sigma = coeffs[2]
        else:
            sigma = np.nan

    return x_center, sigma


def __noise_floor(window, r2, TUNE_NOISE):
    """Determine aproximate noise floor of  ``window``

    Parameters:
        window : np array
            Window to be analyzed
        r2 : np.array
            squared distances of data points to center of ``s``

    Ad-hoc definition of the noise floor:
        + use all data in cornes of 2D FFT (i.e. outside of circle
          with radius ``R`` = ``FFT_SIZE//2``)
        + calculate mean and standard deviation
        + define noise floor as mean-value + ``TUNE_NOISE``*standard deviations

    Returns
        noise_floor : float
            mean + ``TUNE_NOISE`` * sigma
    """
    mask = (r2 >= (window.shape[0] // 2)**2)
    mean = window[mask].mean()
    error = window[mask].std()

    return mean + TUNE_NOISE * error


def __analyze_direction(window, r2, alpha, TUNE_THRESHOLD_DIRECTION):
    """Find peak in FFT window ``window`` and return its direction
    and angular spread

    Parameters:
        window : np.array
            2D Fourier transform
        r2 : np.array
            squared distances of each pixel in the 2D FFT relative to the one
            that represents the zero frequency
        alpha : np.array
            angle of each pixel in 2D FFT relative to the pixel that
            represents the zero frequency
        TUNE_THRESHOLD_DIRECTION : float
            threshold value to discriminate noise peaks

    Returns
        phi : float
            predominant direction of periodicity (0..pi)
        delta_phi : float
            FWHH of ``omega``
    """
    bins = 36 # 10 degrees per bin
    warnings.simplefilter('ignore', RuntimeWarning)
    angle, edges = np.histogram(alpha.flatten(),
                                bins=bins,
                                weights=window.flatten() / r2.flatten())

    if np.nanmax(angle) > TUNE_THRESHOLD_DIRECTION * np.nanmean(angle):
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
            edges = np.append(edges[18 : -1], edges[0 : 18] + edges[-1])
            angle = np.append(angle[18 : ], angle[0 : 18])
        else:
            # remove last element of edges to have len(edges) == bins
            edges = np.append(edges[ : -1], [])

        # replace non-finite entries by linear interpolation between its neighbors
        mask = ~np.isfinite(angle)
        angle[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), angle[~mask])

        phi, sigma_phi = __find_peak(edges, angle)

        if np.isnan(sigma_phi):
            phi = np.nan
        else:
            # because of the wrap-around omega could be larger than pi
            if phi > np.pi:
                phi -= np.pi
    else:
        phi = np.nan
        sigma_phi = np.nan

    return phi, sigma_phi


def __analyze_lattice_const(power_spectrum, r2, TUNE_THRESHOLD_PERIOD):
    """Determine lattice constant and coherence length from FFT. All calculations
    in pixel numbers.

    Parameters:
        power_spectrum : np.array
            abs(2D Fourier transform)
        r2 : np.array
            squared distances of each pixel in the 2D FFT relative to the one
            that represents the zero frequency i.e. frequency^2
        TUNE_THRESHOLD_PERIOD : float
            threshold value to discriminate noise peaks

    Returns
        d : float
            Period found
        sigma_d : float
            Coherence length (length of periodic structure) as A.U.
    """
    bins = power_spectrum.shape[0] // 2  # ad hoc definition
    # build histogram (power as function of radius, bin edges are radius in pixels)
    # weights should  include 1/r^2 i.e. power_spectrum/r^2
    # we integrate azimuthally, thus noise at large ``r`` contributes more
    # than noise (or signal) at small ``r``
    warnings.simplefilter('ignore', RuntimeWarning)
    power, edges = np.histogram(np.sqrt(r2).flatten(),
                                bins=bins,
                                weights=power_spectrum.flatten() / r2.flatten())

    if np.nanmax(power) > TUNE_THRESHOLD_PERIOD * np.nanmean(power):
        # significant peak
        # replace boundaries by centers of bins
        edges += 0.5 * (edges[1] - edges[0])
        # replace non-finite entries by linear interpolation between its neighbors
        mask = ~np.isfinite(power)
        power[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), power[~mask])

        d, sigma_d = __find_peak(edges[ : -1], power)
        # convert to periode
        d = power_spectrum.shape[0] / d
        sigma_d = 1.0 / sigma_d
        if np.isnan(sigma_d):
            d = np.nan
    else:
        d = np.nan
        sigma_d = np.nan

    return d, sigma_d


def inner_loop(v, img, fft_size, step, const, tune):
    """
    Analyzes horizontal line ``v`` in image ``im``

    Parameters:
        v : int
            Line number (horizontal index) to be analyzed
        img : np array
            Image to be analyzed
        fft_size : int
            width of window to be analyzed (2^N x 2^N)
        step : int
            Horizontal (and vertical) step size to translate
            window
        const : tuple of
            r2 : np.array
                squared distances of each pixel in the 2D FFT relative to the one
                that represents the zero frequency
            aplha : np.array
                angle of each pixel in 2D FFT relative to the pixel that
                represents the zero frequency
            mask : np.array
                mask that discards very low frequencies (index 1,2) and high
                frequencies (indices above FFT_SIZE2)
            han2d : np.array
                2D Hanning window applied to roi before the 2D FFT

    Returns
        tuple of np arrays of length ``Nh``
        d : np array
            Period found
        sigma_d : np array
            Coherence length (length of periodic structure) as 1/sigma [A.U.]
        phi : np array
            Direction of lattice (direction) vector of periodic structure
        sigma_phi : np array
            Error of direction vector
    """
    r2, alpha, mask, han2d = const
    TUNE_NOISE, TUNE_THRESHOLD_PERIOD, TUNE_THRESHOLD_DIRECTION = tune
    fft_size2 = fft_size // 2
    Nh = int(np.ceil((img.shape[1] - fft_size) / step))
    d = np.zeros([Nh])
    sigma_d = np.zeros([Nh])
    phi = np.zeros([Nh])
    sigma_phi = np.zeros([Nh])

    for rh, h in enumerate(range(fft_size2,
                                 img.shape[1] - fft_size2,
                                 step)):

        roi = img[v-fft_size2 : v+fft_size2,
                  h-fft_size2 : h+fft_size2]
        # equalize contrast in each roi: -0.5 .. 0.5
        roi = (roi + roi.min()) / roi.max()
        roi -= roi.mean()

        power_spectrum = fftshift(np.abs(fft2(han2d * roi)**2))

        # set very low and very high frequencies to zero (mask)
        # set to zero all frequencies with power smaller than noise floor
        power_spectrum[mask] = 0
        power_spectrum[power_spectrum <= __noise_floor(power_spectrum, r2, TUNE_NOISE)] = 0
        power_spectrum[power_spectrum is 0] = np.nan

        d[rh], sigma_d[rh] = __analyze_lattice_const(power_spectrum, r2,
                                                     TUNE_THRESHOLD_PERIOD)
        phi[rh], sigma_phi[rh] = __analyze_direction(power_spectrum, r2, alpha,
                                                     TUNE_THRESHOLD_DIRECTION)

    return (d, sigma_d, phi, sigma_phi)





__version__ = ''
class HRTEMCrystallinity:
    """
    """
    def __init__(self, fft_size=32, step=1, jobs=1, fname=None):
        """
        Initialized basic attributes.

        Tuning knobs:
            ``TUNE_THRESHOLD_DIRECTION``: a valid peak along the azimuth must be higher
                                          than ``TUNE_THRESHOLD_DIRECTION`` times the
                                          mean intensity
            ``TUNE_THRESHOLD_PERIOD``: a valid peak along the frequency (radius) must
                                       be higher than ``TUNE_THRESHOLD_PERIOD`` times
                                       the mean intensity.
            ``TUNE_NOISE``: everything below mean() + ``TUNE_NOISE``*std is considered
                            noise.
            ``TUNE_MIN_FREQUENCY2``: filter out low frequencies and DC term of FFT.
                                     ``TUNE_MIN_FREQUENCY2`` corresponds to the index
                                     (after flipping quadrants) squared.
            ``TUNE_MAX_FREQUENCY2``: filter out high frequencies of FFT.
                                     ``TUNE_MAX_FREQUENCY2`` corresponds to the index
                                     (after flipping quadrants) squared.
        """

        self.supported = ','.join(plt.figure().canvas.get_supported_filetypes())
        plt.close()

        self.fft_size = fft_size
        self.fft_size2 = fft_size // 2
        self.step = step
        self.jobs = jobs

        self.TUNE_THRESHOLD_DIRECTION = 5.0
        self.TUNE_THRESHOLD_PERIOD = 25.0
        self.TUNE_NOISE = 4.0
        self.TUNE_MIN_FREQUENCY2 = 4**2 # seems a good value
        self.TUNE_MAX_FREQUENCY2 = self.fft_size2**2

        self.__update_precalc()

        try:
            self.image_data = imread(fname, mode='I')
            self.image_fname = fname
        except:
            raise ValueError("Could not read file {}".format(fname))


    def __update_precalc(self):
        """
        Updates/sets all attributes that depend on ``self.fft_size`` and invalidates
        results by setting ``results_are_valid = False``.
        This is used in ``__init__()`` and is triggered by the ``fft_size`` setter.

        Constant data:
            r2 : nparray
                the geometrical distance (index) squared of a given entry.
            alpha : nparray
                the geometrical azimuth of a given entry with the range
                -pi <= alpha < 0 shifted to 0 <= alpha < pi.
            mask : nparray
                  discarded data at very low frequencies (``TUNE_MIN_FREQUENCY``)
                  and high frequencies (``TUNE_MAX_FREQUENCY``).
            han2d : nparray
                2D Hanning window
        """
        self.fft_size2 = self.fft_size // 2
        self.TUNE_MAX_FREQUENCY2 = self.fft_size2**2
        x, y = np.ogrid[-self.fft_size2 : self.fft_size2,
                        -self.fft_size2 : self.fft_size2]
        self.r2 = x*x + y*y
        self.alpha = np.arctan2(x, y)
        self.alpha[self.alpha < 0] += np.pi
        self.mask = ~((self.r2 > self.TUNE_MIN_FREQUENCY2) & (self.r2 < self.TUNE_MAX_FREQUENCY2))
        han = np.hanning(self.fft_size)
        self.han2d = np.sqrt(np.outer(han, han))
        self.results_are_valid = False



    @property
    def fft_size(self):
        return self.__fft_size

    @fft_size.setter
    def fft_size(self, fft_size):
        self.__fft_size = fft_size
        try:
            self.__update_precalc()
        except AttributeError:
            pass

    @property
    def step(self):
        return self.__step

    @step.setter
    def step(self, step):
        self.results_are_valid = False
        self.__step = step

    @property
    def image_fname(self):
        return self.__image_fname

    @image_fname.setter
    def image_fname(self, fname):
        try:
            self.image_data = imread(fname, mode='I')
            self.__image_fname = fname
            self.results_are_valid = False
        except:
            raise ValueError("Could not read file {}".format(fname))

    @staticmethod
    def __sub_imageplot(data, this_ax, title, limits):
        """
        Plot data as image
        Parameters
            data : nparray
                data to be plotted
            this_ax : axes instance
            title : string
            limits : (float, float)
                (minimum, maximum) or (None, None) to autoscale
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        img = this_ax.imshow(data, cmap='jet', vmin=limits[0], vmax=limits[1], origin='upper')
        # Create divider for existing axes instance
        divider = make_axes_locatable(this_ax)
        # Append axes to the right of ax, with 20% width of ax
        cax = divider.append_axes("right", size="20%", pad=0.05)
        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks`
        # and the format of the ticklabels with kwarg `format`
        if title == r'direction ($\phi$)':
            ticks = np.linspace(0, np.pi, num=9, endpoint=True)
            labels = ['W', '', 'NW', '', 'N', '', 'NE', '', 'E']
            cbar = plt.colorbar(img, cax=cax, ticks=ticks)
            cbar.ax.set_yticklabels(labels)
            cbar.ax.invert_yaxis()  # W at top, E at bottom
            cbar.set_label('direction  [-]')
        else:
            cbar = plt.colorbar(img, cax=cax)
            if title == r'spacing ($d$)':
                cbar.set_label('$d$  [pixel]')
            if title == r'coherence ($1/\sigma_d$)':
                cbar.set_label(r'$1/\sigma_d$  [A.U.]')
            if title == r'spread ($np.array(d).reshape(Nv, Nh)\sigma_\phi$)':
                cbar.set_label(r'$\sigma_d$  [$^\circ$]')

        this_ax.set_title(title)
        this_ax.xaxis.set_visible(False)
        this_ax.yaxis.set_visible(False)


    def plot_data(self, outfname=None,
                  limits_d=(None, None),
                  limits_sigma_d=(None, None),
                  limits_phi=(0.0, np.pi),
                  limits_sigma_phi=(None, None)):
        """Summary plot of results to ``outfname`` (if provided) or to pop-up
        window. Individualt limits can be provided.
        """
        if not self.results_are_valid:
            print('No valid data to plot', file=sys.stderr)
            return

        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        self.__sub_imageplot(self.d, ax1,
                             r'spacing ($d$)', limits_d)
        self.__sub_imageplot(self.sigma_d, ax2,
                             r'coherence ($1/\sigma_d$)', limits_sigma_d)
        self.__sub_imageplot(self.phi, ax3,
                             r'direction ($\phi$)', limits_phi)
        self.__sub_imageplot(np.rad2deg(self.sigma_phi), ax4,
                             r'spread ($\sigma_\phi$)', limits_sigma_phi)

        plt.tight_layout()

        if outfname is None:
            plt.show()
        else:
            try:
                plt.savefig(outfname)
            except ValueError:
                print('Cannot save figure ({})'.format(outfname))
                print('Supported formats: {}'.format(self.supported))
                plt.show()



    def save_data(self, compressed=True):
        """Save data in (compressed) ASCII files
        """
        if not self.results_are_valid:
            print('No valid data to save', file=sys.stderr)
            return

        if compressed:
            ext = '.gz'
        else:
            ext = ''
        header = '#\n' \
        + '# crist.py version {}\n'.format(__version__) \
        + '# Tuning knobs:\n' \
        + '#        TUNE_MIN_FREQUENCY2: {}\n'.format(self.TUNE_MIN_FREQUENCY2) \
        + '#        TUNE_MAX_FREQUENCY2: {}\n'.format(self.TUNE_MAX_FREQUENCY2) \
        + '#                 TUNE_NOISE: {}\n'.format(self.TUNE_NOISE) \
        + '#   TUNE_THRESHOLD_DIRECTION: {}\n'.format(self.TUNE_THRESHOLD_DIRECTION) \
        + '#      TUNE_THRESHOLD_PERIOD: {}\n'.format(self.TUNE_THRESHOLD_PERIOD) \
        + '#\n' \
        + '# Results for {} ({}x{} [vxh])\n'.format(self.image_fname,
                                                    self.image_data.shape[0],
                                                    self.image_data.shape[1]) \
        + '# FFT window: {}x{}\n'.format(self.fft_size, self.fft_size) \
        + '#       step: {}\n'.format(self.step) \
        + '#\n' \
        + '# To convert from local indices to indices of original image\n' \
        + '#\n' \
        + '#     image_idx = FFT_window/2 + local_idx * step\n' \
        + '# both, for horizontal and vertical index\n' \
        + '#'
        base_name, _ = self.image_fname.rsplit(sep='/')[-1].rsplit(sep='.')
        np.savetxt(base_name + '_spacing' + '.dat' + ext, self.d,
                   delimiter='\t', header=header, comments='')
        np.savetxt(base_name + '_coherence' + '.dat' + ext, self.sigma_d,
                   delimiter='\t', header=header, comments='')
        np.savetxt(base_name + '_direction' + '.dat' + ext, self.phi,
                   delimiter='\t', header=header, comments='')
        np.savetxt(base_name + '_spread' + '.dat' + ext, self.sigma_phi,
                   delimiter='\t', header=header, comments='')


    def analyze(self):
        """Analyze local crystallinity of ``image_data``
        """
        # x-axis is im.shape[1] -> horizontal (left->right)
        # y-axis is im.shape[0] -> vertical (top->down)
        # indices v,h for center of roi in image
        Nh = int(np.ceil((self.image_data.shape[1] - self.fft_size) / self.step))
        Nv = int(np.ceil((self.image_data.shape[0] - self.fft_size) / self.step))

        with Parallel(n_jobs=self.jobs) as parallel:
            res = parallel(delayed(inner_loop)(v,
                                               self.image_data,
                                               self.fft_size, self.step,
                                               (self.r2, self.alpha, self.mask, self.han2d),
                                               (self.TUNE_NOISE,
                                                self.TUNE_THRESHOLD_PERIOD,
                                                self.TUNE_THRESHOLD_DIRECTION))
                           for v in range(self.fft_size2,
                                          self.image_data.shape[0] - self.fft_size2,
                                          self.step))

            d, sigma_d, phi, sigma_phi = zip(*res)

        self.d = np.array(d).reshape(Nv, Nh)
        self.sigma_d = np.array(sigma_d).reshape(Nv, Nh)
        self.phi = np.array(phi).reshape(Nv, Nh)
        self.sigma_phi = np.array(sigma_phi).reshape(Nv, Nh)
        self.results_are_valid = True



def main():
    """main function
    """
    from argparse import ArgumentParser
    def parse_command_line():
        """Parse command line arguments and return them
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
        return parser.parse_args()


    args = parse_command_line()
    H = HRTEMCrystallinity(fft_size=args.FFT_size, step=args.step, jobs=args.jobs, fname=args.file)
    print(H.fft_size, H.step)
    H.analyze()
    H.plot_data('test32.pdf')
    H.save_data(compressed=False)
    H.fft_size = 128
    H.plot_data('invalid.pdf')
    H.step = 30
    print(H.fft_size, H.step)
    H.analyze()
    H.save_data()
    H.plot_data('test128.pdf')
    H.plot_data('test128_b.pdf', limits_d=(15.0, 20.0))

if __name__ == '__main__':
    main()
