# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS core module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# This module uses some mpdaf routines and wrap some of their
# functionalities to help the muse_pipe checkups

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

from mpdaf.obj import Cube
from mpdaf.obj import Image
from mpdaf.obj import Spectrum

# Astropy
from astropy.io import fits
from astropy import constants as const

from emission_lines import list_emission_lines

__version__ = '0.0.1 (15 March 2018)'

############################################################
#                      BEGIN
# The following parameters can be adjusted for the need of
# the specific pipeline to be used
############################################################

############################################################
#                      END
############################################################

def doppler_shift(wavelength, velocity=0.):
    """Return the redshifted wavelength
    """
    doppler_factor = np.sqrt((1. + velocity / const.c) / (1. - velocity / const.c))
    return wavelength * doppler_factor

#########################################################################
# Main class
#                           check_musepipe
#########################################################################

class muse_cube(Cube): 
    """Wrapper around the mpdaf Cube functionalities
    """
    
#    def __init__(self, cube_folder="./", cube_name=None, source=None, verbose=False, **kwargs) :
    def __init__(self, source=None, verbose=False, **kwargs) :
        """Initialisation of the opening of a Cube
        """
        if source is not None:
            self.__dict__.update(source.__dict__)
        else :
#            filename = joinpath(cube_folder, cube_name)
#            if os.path.isfile(filename) :
#                Cube.__init__(self, filename=filename, **kwargs)
                Cube.__init__(self, **kwargs)
 
        self.verbose = verbose

    def get_spectrum_from_cube(self, nx=None, ny=None, width=0, title="Spectrum") :
        """Get a spectrum from the cube with centre defined in pixels
        with nx, ny and a window of 'width'
        """
        if nx == None : nx = self.shape[2] // 2
        if ny == None : ny = self.shape[1] // 2
        width2 = width // 2
        subcube = self[:, ny - width2: ny + width2 + 1, 
                    nx - width2: nx + width2 + 1]
        return muse_spectrum(source=subcube.sum(axis=(1,2)), title=title)

    def get_whiteimage_from_cube(self) :
        return muse_image(source=self.sum(axis=0))

    def get_image_from_cube(self, nlambda=None, width=0) :
        """Get image from integrated cube, with spectral pixel
        centred at nlambda and with a width of width
        """
        if nlambda == None : nlambda = self.shape[0] // 2
        width2 = width // 2
        return muse_image(source=self[nlambda - width2: nlambda + width2 + 1, :, :].sum(axis=0))

    def get_set_spectra(self) :
        """Get a set of standard spectra from the Cube
        """
        self.spec_fullgalaxy = muse_spectrum(source=self.sum(axis=(1,2)), title="Full galaxy Spectrum")
        self.spec_4quad = self.get_quadrant_spectra_from_cube()
        self.spec_central_aper = museset_spectra(
               self.get_spectrum_from_cube(width=0, title="Central aperture"), 
               self.get_spectrum_from_cube(width=20, title="Central Aperture, w=20"), 
               self.get_spectrum_from_cube(width=40, title="Central Aperture, w=40"),
               subtitle="central_spectra")

    def get_quadrant_spectra_from_cube(self, width=0) :
        """Get quadrant spectra from the Cube

        Input
        ----
        width : width of integration
        """
        ny4 = self.shape[1] // 4
        nx4 = self.shape[2] // 4
        nx34, ny34 = 3 * nx4, 3 * ny4

        spec1 = self.get_spectrum_from_cube( nx4,  ny4, width, title="Quadrant 1")
        spec2 = self.get_spectrum_from_cube( nx4, ny34, width, title="Quadrant 2") 
        spec3 = self.get_spectrum_from_cube(nx34,  ny4, width, title="Quadrant 3") 
        spec4 = self.get_spectrum_from_cube(nx34, ny34, width, title="Quadrant 4") 
        return museset_spectra(spec1, spec2, spec3, spec4, subtitle="4 Quadrants")

    def get_emissionline_image(self, line=None, velocity=0., redshift=None, width=10, medium='vacuum') :
        """Get a narrow band image around Ha

        Input
        -----
        width: in Angstroems (10 by default). Width of the window of integration
        medium: vacuum or air (string, 'vacuum' by default)
        velocity: default is 0. (km/s)
        redshift: default is None. Overwrite velocity if provided.
        line: name of the emission line (see emission_lines dictionary)
        """
        index_line = {'vacuum': 0, 'air': 1}
        # Get the velocity
        if redshift is not None : velocity = redshift * const.c

        if line not in emission_lines :
            print("ERROR: could not guess the emission line you wish to use")
            print("ERROR: please review the 'emission_line' dictionary")
            return

        if medium not in index_line.keys() :
            print("ERROR: please choose between one of these media: {0}".format(index_line.key()))
            return

        wavel = emission_lines[line][index_line[medium]]
        red_wavel = doppler_shift[wavel, velocity]
        
        try :
            self.images[line] = self.select_lambda(red_wavel - width/2., red_wavel + width/2.)
        except AttributeError :
            self.images = {}
            self.images[line] = self.select_lambda(red_wavel - width/2., red_wavel + width/2.)

class muse_image(Image): 
    """Wrapper around the mpdaf Image functionalities
    """
    def __init__(self, source=None, title="Frame", scale='log', vmin=0, colorbar='v', verbose=False, **kwargs) :
        """Initialisation of the opening of an Image
        """
        if source is not None:
            self.__dict__.update(source.__dict__)
        else :
            Image.__init__(self, **kwargs)
 
        self.verbose = verbose
        # Arguments for the plots
        self.title = title
        self.scale = scale
        self.vmin = vmin
        self.colorbar = colorbar
        self.get_fwhm_startend()

    def get_fwhm_startend(self) :
        """Get range of FWHM
        """
        if self._isImage :
            self.fwhm_startobs = self.image.primary_header['HIERARCH ESO TEL AMBI FWHM START']
            self.fwhm_endobs = self.image.primary_header['HIERARCH ESO TEL AMBI FWHM END']
        else :
            if self.verbose :
                print("WARNING: image not yet opened, hence no header to be read")

class museset_images(list) :
    """Set of images
    """
    def __new__(self, add_number=True, *args, **kwargs):
        return super(museset_images, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)
        self.nimages = len(self)
        if add_number :
            for i in range(len(self)) :
                self[0].title += " {0:2d}".format(i+1)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

#    def add_image(self, image, add_number=True) :
#        """Add an image to the set
#        """
#        self.nimages += 1
#        image.id = self.nimages
#        if add_number :
#            image.title += " {0:2d}".format(image.id)
#        self.images.append(image)

class muse_spectrum(Spectrum): 
    """Wrapper around the mpdaf Spectrum functionalities
    """
    def __init__(self, source=None, title="Spectrum", verbose=False, **kwargs) :
        """Initialisation of the opening of spectra
        """
        if source is not None:
            self.__dict__.update(source.__dict__)
        else :
            Spectrum.__init__(self, **kwargs)

        self.verbose = verbose
        # Arguments for the plots
        self.title = title

class museset_spectra(list) :
    """Set of spectra
    """
    def __new__(self, *args, **kwargs):
        return super(museset_spectra, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)
        self.nspectra = len(self)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

#    def __init__(self, spectra_list=None, subtitle="", add_number=True) :
#        """Initialise the list
#        """
#        self.nspectra = 0
#        self.spectra = []
#        self.subtitle = subtitle
#
#        for spectrum in spectra_list :
#            self.add_spectrum(spectrum, add_number=add_number)
#
#    def add_spectrum(self, spectrum, add_number=True) :
#        """Add a spectrum to the set
#        """
#        spectrum.id = self.nspectra
#        if isinstance(spectrum, Spectrum) :
#            spectrum = muse_spectrum(source=spectrum)
#        if add_number :
#            spectrum.title += " {0:2d}".format(spectrum.id)
#        self.spectra.append(spectrum)
