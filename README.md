# HRTEMFringeAnalyzer

A python module to evaluate the local crystallinity of samples from high resolution TEM images in a mostly automated fashion. The user only selects the size of a square analyzer window and a step size which translates the window in the micrograph. Together they define the resolution of the results obtained. Regions where fringe patterns are visible are identified and their lattice spacing $d$ and direction `phi` as well as the corresponding mean errors `sigma` determined. `1/sigma_d` is proportional to the coherence length of the structure while `sigma_phi` is a measure of how well the direction of the fringes is defined. Maps of these four indicators are computed.
For details see [1].


## Features


## Installation

### Requirements

```
python3.6+
numpy
scipy
matplotlib
imageio
numba
joblib
```
Download zip and extract or clone repository. From the resulting folder run

```bash
$ python setup.py install
```

## Usage

### From a Python/IPython session or in a script

```python
>>> from HRTEM import HRTEMFringeAnalyzer
>>> FA = HRTEMFringeAnalyzer(fft_size=64, step=32, jobs=4, fname='test.tif')
>>> print(FA.jobs)
    4
>>> print(FA.step)
    32
>>> print(FA.fft_size)
    64
>>> FA.analyze()
>>> FA.d
    array([[ nan,  nan,  nan, ...,  nan,  nan,  nan],
           [ nan,  nan,  nan, ...,  nan,  nan,  nan],
           [ nan,  nan,  nan, ...,  nan,  nan,  nan],
           ...,
           [ nan,  nan,  nan, ...,  nan,  nan,  nan],
           [ nan,  nan,  nan, ...,  nan,  nan,  nan],
           [ nan,  nan,  nan, ...,  nan,  nan,  nan]])

    no finite values to plot
    no finite values to plot
    >>> FA.fft_size = 128
    >>> FA.d    # Results have been invalidated
    >>> print(FA.d)
    None
    >>> FA.analyze()    # Reevaluate
    >>> FA.d
    array([[ 25.58833813,  26.13036064,          nan, ...,          nan,
                     nan,          nan],
           [         nan,          nan,          nan, ...,          nan,
                     nan,          nan],
           [ 28.12283891,  28.76717216,  27.56026979, ...,          nan,
                     nan,          nan],
           ...,
           [         nan,          nan,          nan, ...,          nan,
                     nan,          nan],
           [         nan,          nan,          nan, ...,          nan,
                     nan,          nan],
           [         nan,          nan,          nan, ...,          nan,
                     nan,          nan]])

    >>> FA.summarize_data('test128_b.pdf', limits_d=(15.0, 20.0))
    >>> FA.save_data()    # Data saved as gzipped ASCII
    >>> for result in (FA.d, FA.sigma_d, FA.phi, FA.sigma_phi):
    ...:     FA.plot_overlayed(result)
```
### FAnalyzer.py

FAnalyzer is a small stand-alone program provided (utils) to perform the analysis

```bash
$ FAnalyzer.py -h
GLib-GIO-Message: Using the 'memory' GSettings backend.  Your settings will not be saved or shared with other applications.
usage: FAnalyzer.py [-h] -f FILE [-F N] [-j N] [-s S] [-S] [-o FILE] [-O FILE]

Analyze local cristallinity of data

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Name of data file. Remove outliers (bright and dark)
                        before.
  -F N, --FFT_size N    Size of moving window (NxN) [128].
  -j N, --jobs N        Number of threads to be started [1].
  -s S, --step S        Step size (x and y) in pixels of moving window [32]
  -S, --save            Store result in gzipped text files
  -o FILE, --output FILE
                        Summary output to file. Supported formats:
                        pgf,tiff,ps,svg,eps,jpeg,svgz,png,pdf,jpg,tif,raw,rgba
  -O FILE, --Overlays FILE
                        Plot results overlayed on micrograph. Supported
                        formats:
                        pgf,tiff,ps,svg,eps,jpeg,svgz,png,pdf,jpg,tif,raw,rgba
```

## Reference

[1] I. Alxneit, HRTEMFringeAnalyzer a free python module for a automated analysis of fringe pattern in transmission electron micrographs, Journal of Microscopy (submitted)
