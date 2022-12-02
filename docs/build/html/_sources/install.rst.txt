======
Install
======

Prerequisites
""""""""""""""""""""""""

Pymusepipe assumes you have a working installation of ESOREX (i.e. that you have a working
MUSE data reduction pipeline installation) and likwid-pin. The installation of these components is not covered here but can be found in the `MUSE Pipeline User Manual <https://ftp.eso.org/pub/dfs/pipelines/instruments/muse/muse-pipeline-manual-2.8.7.pdf>`_.

Pymusepipe uses `Python 3 <https://www.python.org>`_ and is not compatible with Python 2.
It requires a number of standard python packages including:

* **numpy**
* **scipy**
* **matplotlib**
* **astropy**
* **mpdaf** a utility package to process and analyse datacubes, and more specifically
  MUSE cubes, images and spectra developed by the MUSE GTO-CRAL Team.

In addition some packages are needed to access specific functionality:

* **pypher** to use the convolution package of pymusepipe
* **spacepylot** to use the automatic alignment module

Installation
"""""""""""""""""""""""
You can install this package via pypi via a simple::

   pip install pymusepipe


You can obviously also install it by cloning it from github, or 
downloading the source (from github) and do something like::

   python setup.py develop

The "develop" option is recommended as it actually does not copy 
the files in your system but just creates a link. 
In that way you can easily update the source software without 
reinstalling it. The link will directly use the source which has been udpated.

The other option is to use the standard "install" option::

   python setup.py install

