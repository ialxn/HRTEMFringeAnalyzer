#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:44:23 2017

@author: alxneit
"""
from argparse import ArgumentParser
import gzip
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

__version__ = ''

def read_header(fname):
    """Reads ``fft_size`` and ``step`` from compressed output ```fname``
    """
    with gzip.open(fname, 'rb') as file:
        for b_line in file:
            # convert to string and skip terminal '\n'
            line = str(b_line)[0:-3]
            if 'FFT window:' in line:
                fft_size = int(line.rsplit('x')[-1])
            elif 'step:' in line:
                step = int(line.rsplit(':')[-1])
                break
    return fft_size, step


def main():
    """"main function
    """
    supported = ','.join(plt.figure().canvas.get_supported_filetypes())
    plt.close()

    parser = ArgumentParser(description='Analyze local cristallinity of data')
    parser.add_argument('-f', '--file', metavar='FILE',
                        type=str, required=True,
                        help='Base name of data file (.tif). Remove outliers (bright and dark) before.')
    parser.add_argument('-t', '--type', metavar='TYPE',
                        type=str, default='pdf',
                        help='Graphics type of output files [pdf]. Supported formats: ' + supported)
    args = parser.parse_args()

    image = imread(args.file + '.tif')

    window, inc = read_header(args.file + '_direction.dat.gz')

    for res in ('direction', 'coherence', 'periode', 'spread'):
        mat = np.loadtxt(args.file + '_' + res + '.dat.gz')
        fig = plt.figure()
        subplot1 = plt.subplot(111)
        subplot1.imshow(image, extent=None, aspect='equal', cmap='gray')
        x = np.arange(window // 2, image.shape[1] - window // 2, inc)
        y = np.arange(window // 2, image.shape[0] - window // 2, inc)
        cax = subplot1.contourf(x, y, mat, alpha=0.5, cmap='jet')
        if res == 'direction':
            ticks = np.linspace(0, np.pi, num=9, endpoint=True)
            labels = ['W', '', 'NW', '', 'N', '', 'NE', '', 'E']
            cbar = fig.colorbar(cax, ticks=ticks, shrink=0.7)
            cbar.ax.set_yticklabels(labels)
        else:
            fig.colorbar(cax, shrink=0.7)

        subplot1.set_title(res)

        plt.savefig(args.file + '_' + res + '.' + args.type)


if __name__ == '__main__':
    main()
