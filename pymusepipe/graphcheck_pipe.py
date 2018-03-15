# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS core module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# This module is based on a code written by Rebecca McElroy
#
# Eric Emsellem adapted a March 2018 version, provided by Rebecca
# and adapted it to the pymusepipe package

# Importing modules
import numpy as np

# Standard modules
import os
import time
import copy

# Importing mpdaf
try :
    import mpdaf
except ImportError :
    raise Exception("mpdaf is required for this - MUSE related - module")

from mpdaf.obj import Cube
from mpdaf.drs import PixTable
from mpdaf.drs import RawFile
from mpdaf.obj import Image, WCS
from mpdaf.obj import Spectrum, WaveCoord

# Plotting routines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Astropy
from astropy.io import fits

__version__ = '0.0.1 (15 March 2018)'

############################################################
#                      BEGIN
# The following parameters can be adjusted for the need of
# the specific pipeline to be used
############################################################

############################################################
#                      END
############################################################


#########################################################################
# Main class
#                           check_plot()
#########################################################################

class check_plot(object): 
    """
    """
    
    def __init__(self, cube_folder='./', cube_name=None, verbose=False) :
        """
        """
        self.open_cube(cube_folder, cube_name)
        self.verbose = verbose
        self.get_set_spectra()
        self.get_set_images()

    def open_cube(self, cube_folder"./", cube_name=None) :
        """Open the cube
        """
        self.cube_folder = cube_folder
        self._isCube = True
        self.cube_name = cube_name
        if cubein is None | not os.isfile(cube_folder + cubein):
            self._isCube = False
        else :
            self._isCube = True
        
        if self_isCube :
            self.cube_galaxy = Cube(cube_folder + cube_name)
        else :
            print("WARNING: No Cube is defined, yet")

    def open_image(self, image_folder"./", image_name=None) :
        """Open the image
        """
        self.image_folder = image_folder
        self._isImage = True
        self.image_name = image_name
        if imagein is None | not os.isfile(image_folder + imagein):
            self._isImage = False
        else :
            self._isImage = True
        self.image_galaxy = Image(image_folder + image_name)

    def get_spectrum(self, nx=None, ny=None, width=0) :
        if not self._isCube : 
            print("WARNING: No Cube is defined, yet")
            return

        if nx == None : nx = self.spaxel_centralcoord[2] // 2
        if ny == None : ny = self.spaxel_centralcoord[1] // 2
        width2 = width // 2
        return (self.cube_galaxy[:, ny - width2: ny + width2 + 1, 
                    nx - width2: nx + width2 + 1]).sum(axis=1,2)

    def get_image(self, nlambda=None, width=0) :
        if not self._isCube : 
            print("WARNING: No Cube is defined, yet")
            return

        if nlambda == None : nlambda = 1472
        width2 = width // 2
        return (self.cube_galaxy[nlambda - width2: nlambda + width2 + 1, 
            :, :).sum(axis=0)

    def get_set_images(self) :
        """Get a set of standard images from the Cube
        """
        self.

    def get_set_spectra(self) :
        """Get a set of standard spectra from the Cube
        """
        if self._isCube :
            self.spec_fullgalaxy = self.cube_galaxy.sum(axis=(1,2))
            self.spec_4quad = get_quadrant_spectrum()
            self.spec_central_aper = [get_spectrum(width=0), 
                   get_spectrum(width=20), get_spectrum(width=40))

    def get_quadrant_spectra(self, width=0) :
        """Get quadrant spectra from the Cube

        Input
        ----
        width : width of integration
        """
        if not self._isCube : 
            print("WARNING: No Cube is defined, yet")
            return

        ny4 = self.cube_galaxy.shape[1] // 4
        nx4 = self.cube_galaxy.shape[2] // 4
        nx34, ny34 = 3 * nx4, 3 * ny4

        spec1 = self.get_spectrum( nx4,  ny4, width) 
        spec2 = self.get_spectrum( nx4, ny34, width) 
        spec3 = self.get_spectrum(nx34,  ny4, width) 
        spec4 = self.get_spectrum(nx34, ny34, width) 
        return spec1, spec2, spec3, spec4

    def check_bias(self) :
        pass
    def check_flat(self) :
        pass

    def check_finalcube(self) :
        pass

