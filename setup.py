#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE
#
from __future__ import absolute_import, division, print_function
#
# Standard imports
#
import sys
import os

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='pymusepipe',
      version='1.0',
      description='python module to reduce MUSE LP data associated with PHANGS',
      long_description=readme,
      keywords='MUSE, PHANGS',
      url='http://',
      author='Eric Emsellem',
      author_email='eric.emsellem@eso.org',
      license=license,
      packages=find_packages(exclude=('tests', 'docs')),
      install_requires=[],
      include_package_data=True,
      zip_safe=False)
