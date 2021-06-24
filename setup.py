# -*- coding: utf-8 -*-"""
"""
    Setup file for pymusepipe.
    Use setup.cfg to configure your project.
"""
# Licensed under a MIT style license - see LICENSE.txt

from __future__ import absolute_import, division, print_function
from setuptools import setup, find_packages

version = {}
with open("src/pymusepipe/version.py") as fp:
    exec(fp.read(), version)

with open('README.md', 'r') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(name='pymusepipe',
      version = version['__version__'],
      description='python module to reduce MUSE Raw data and combine them',
      long_description=readme,
      long_description_content_type="text/markdown",
      keywords=['MUSE', 'DATAREDUCTION'],
      url="https://github.com/emsellem/pymusepipe",
      download_url="https://github.com/emsellem/pymusepipe/archive/v2.9.6.beta.tar.gz",
      author='Eric Emsellem',
      author_email='eric.emsellem@eso.org',
      license="MIT",
      packages=find_packages(exclude=('tests', 'docs')),
      install_requires=['mpdaf', 'numpy', 'scipy', 'astropy'],
      include_package_data=True,
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
       ],
      )
