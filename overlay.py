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
                        help='Base name of data file')
    parser.add_argument('-t', '--type', metavar='TYPE',
                        type=str, default='pdf',
                        help='Graphics type of output files [pdf]. Supported formats: ' + supported)
    args = parser.parse_args()

    image = imread(args.file + '.tif')

    window, inc = read_header(args.file + '_direction.dat.gz')

    for res in ('direction', 'coherence', 'periode', 'spread'):
        mat = np.loadtxt(args.file + '_' + res + '.dat.gz')
        plt.figure()
        plt.subplots_adjust(left=0.10, right=1.00, top=0.90, bottom=0.06, hspace=0.30)
        subplot1 = plt.subplot(111)
        subplot1.imshow(image, extent=None, aspect='equal', cmap='gray')
        x = np.arange(window // 2, image.shape[1] - window // 2, inc)
        y = np.arange(window // 2, image.shape[0] - window // 2, inc)
        subplot1.contourf(x, y, mat, alpha=0.5, cmap='jet')
        subplot1.set_title(res)
        plt.savefig(args.file + '_' + res + '.' + args.type)


if __name__ == '__main__':
    main()
