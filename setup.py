#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup

with open('HRTEM/crystalline.py') as file:
    for line in file:
         if '__version__' in line:
             tokens = line.rsplit("'", 2)
             if tokens[1].strip() is not '':
                 this_version = tokens[1]
             else:
                 this_version = 'no version defined'
             break

setup(name='HRTEMCrystallinity',
      version=this_version,
      description='Analyze cristallinity from HRTEM Micrographs',
      author='Ivo Alxneit',
      author_email='ivo.alxneit@psi.ch',
      packages=['HRTEM'],
     )