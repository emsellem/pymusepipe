# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS core module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# Importing modules
import numpy as np
import os

try :
    import astropy as apy
    from astropy.io import fits as pyfits
except ImportError :
    raise Exception("astropy is required for this module")

__version__ = '0.0.1 (21 November 2017)'

class muse_combine(object) :
    def __init__(self) :
        """Initialisation of class muse_expo
        """
        pass

    def method1(self) :
        """Method 1
        """
        pass


