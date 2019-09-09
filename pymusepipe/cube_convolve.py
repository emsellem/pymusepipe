"""MUSE-PHANGS convolve module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2019, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# This module uses some mpdaf routines and pypher to 
# convolve a given datacube to a calibrated PSF
# This uses Moffat functions as reference

# Importing modules
import numpy as np

# Standard modules
import os
from os.path import join as joinpath

# Importing mpdaf
try :
    import mpdaf
except ImportError :
    raise Exception("mpdaf is required for this - MUSE related - module")

from mpdaf.obj import Cube, Image
from mpdaf.obj import Spectrum, WaveCoord

# Astropy
from astropy.io import fits as pyfits

