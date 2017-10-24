#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import HRTEMFringeAnalyzer as FringeAnalyzer

FA = FringeAnalyzer(fft_size=64, step=32, jobs=4, fname='../test.tif')
print(FA.step)