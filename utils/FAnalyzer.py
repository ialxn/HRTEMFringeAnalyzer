#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2017-2018 Ivo Alxneit, Paul Scherrer Institut
#
# This file is part of  HRTEMFringeAnalyzer.
#
# HRTEMFringeAnalyzer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HRTEMFringeAnalyzer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HRTEMFringeAnalyzer. If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
from HRTEM import HRTEMFringeAnalyzer

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
        parser.add_argument('-o', '--output', metavar='FILE', type=str, default=None,
                            help='Summary output to file. Supported formats: ' + supported)
        parser.add_argument('-O', '--Overlays', metavar='FILE', type=str, default=None,
                            help='Plot results overlayed on micrograph. Supported formats: '
                            + supported)
        return parser.parse_args()


    args = parse_command_line()
    FA = HRTEMFringeAnalyzer(fft_size=args.FFT_size,
                             step=args.step,
                             jobs=args.jobs,
                             fname=args.file)

    FA.analyze()

    if args.save:
        FA.save_data(compressed=True)

    FA.summarize_data(args.output)

    if args.Overlays:
        for result in (FA.d, FA.sigma_d, FA.phi, FA.sigma_phi):
            FA.plot_overlayed(result, outfname=args.Overlays)

if __name__ == '__main__':
    main()
