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

from musepipe import joinpath

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
#                           check_musepipe
#########################################################################

class check_musepipe(object): 
    """Graphic output to check MUSE data reduction products
    """
    
    def __init__(self, cube_folder='./', cube_name=None, verbose=False) :
        """
        """
        self.verbose = verbose
        self.cube_folder = cube_folder
        self.cube_name = cube_name

        # Open the cube and extract spectra if cube name is given
        self.open_cube(cube_folder, cube_name)
        if self._isCube :
            self.get_set_spectra()
            self.get_set_images()

    def open_cube(self, cube_folder=None, cube_name=None) :
        """Open the cube
        """
        # Check if cube folder and name are given 
        if cube_folder is not None : 
            self.cube_folder = cube_folder
        if cube_name is not None : 
            self.cube_name = cube_name

        # Check if cube exists
        if (self.cube_folder is None) | (self.cube_name is None) :
            self._isCube = False
            print("WARNING: No appropriate Cube name and folder defined")
            return

        cubepath = joinpath(self.cube_folder, self.cube_name)
        if os.path.isfile(cubepath) :
            self._isCube = True
            self.cube_galaxy = Cube(cubepath)
        else :
            self._isCube = False
            print("WARNING: Cube {0} not found".format(cubepath))

    def open_image(self, image_folder="./", image_name=None) :
        """Open the image
        """
        self.image_folder = image_folder
        self._isImage = True
        self.image_name = image_name
        if (imagein is None) | (not os.path.isfile(joinpath(image_folder, imagein))):
            self._isImage = False
        else :
            self._isImage = True
        self.image_galaxy = Image(joinpath(image_folder, image_name))

    def get_spectrum_from_cube(self, nx=None, ny=None, width=0) :
        if not self._isCube : 
            print("WARNING: No Cube is defined, yet")
            return

        if nx == None : nx = self.cube_galaxy.shape[2] // 2
        if ny == None : ny = self.cube_galaxy.shape[1] // 2
        width2 = width // 2
        return (self.cube_galaxy[:, ny - width2: ny + width2 + 1, 
                    nx - width2: nx + width2 + 1]).sum(axis=(1,2))

    def get_image_from_cube(self, nlambda=None, width=0) :
        if not self._isCube : 
            print("WARNING: No Cube is defined, yet")
            return

        if nlambda == None : nlambda = self.cube_galaxy.shape[0] // 2
        width2 = width // 2
        return (self.cube_galaxy[nlambda - width2: nlambda + width2 + 1, :, :].sum(axis=0))

    def get_set_images(self) :
        """Get a set of standard images from the Cube
        """
        pass

    def get_set_spectra(self) :
        """Get a set of standard spectra from the Cube
        """
        if self._isCube :
            self.spec_fullgalaxy = self.cube_galaxy.sum(axis=(1,2))
            self.spec_4quad = self.get_quadrant_spectra_from_cube()
            self.spec_central_aper = [self.get_spectrum_from_cube(width=0), 
                   self.get_spectrum_from_cube(width=20), self.get_spectrum_from_cube(width=40)]

    def get_quadrant_spectra_from_cube(self, width=0) :
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

        spec1 = self.get_spectrum_from_cube( nx4,  ny4, width) 
        spec2 = self.get_spectrum_from_cube( nx4, ny34, width) 
        spec3 = self.get_spectrum_from_cube(nx34,  ny4, width) 
        spec4 = self.get_spectrum_from_cube(nx34, ny34, width) 
        return spec1, spec2, spec3, spec4

    def check_bias(self) :
        pass
        # bias = RawFile(self. ..)

    def check_flat(self) :
        pass

    def check_finalcube(self) :
        pass

