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

from mpdaf.obj import Cube, Image
from mpdaf.obj import Spectrum, WaveCoord

# Astropy
from astropy.io import fits as pyfits
from astropy import constants as const
from astropy import units as units

from emission_lines import list_emission_lines
import musepipe as mpipe

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
    doppler_factor = np.sqrt((1. + velocity / const.c.value) / (1. - velocity / const.c.value))
    return wavelength * doppler_factor

#########################################################################
# Main class
#                           check_musepipe
#########################################################################

class MuseCube(Cube): 
    """Wrapper around the mpdaf Cube functionalities
    """
    
    def __init__(self, source=None, verbose=False, **kwargs) :
        """Initialisation of the opening of a Cube
        """
        if source is not None:
            self.__dict__.update(source.__dict__)
        else :
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
        return MuseSpectrum(source=subcube.sum(axis=(1,2)), title=title)

    def get_whiteimage_from_cube(self) :
        return MuseImage(source=self.sum(axis=0), title="White Image")

    def get_image_from_cube(self, nlambda=None, width=0) :
        """Get image from integrated cube, with spectral pixel
        centred at nlambda and with a width of width
        """
        if nlambda == None : nlambda = self.shape[0] // 2
        width2 = width // 2
        return MuseImage(source=self[nlambda - width2: nlambda + width2 + 1, :, :].sum(axis=0))

    def get_set_spectra(self) :
        """Get a set of standard spectra from the Cube
        """
        self.spec_fullgalaxy = MuseSpectrum(source=self.sum(axis=(1,2)), title="Full galaxy Spectrum")
        self.spec_4quad = self.get_quadrant_spectra_from_cube()
        self.spec_central_aper = MuseSetSpectra(
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
        return MuseSetSpectra(spec1, spec2, spec3, spec4, subtitle="4 Quadrants")

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

        if line not in list_emission_lines :
            mpipe.print_error("Could not guess the emission line you wish to use")
            mpipe.print_error("Please review the 'emission_line' dictionary")
            return

        if medium not in index_line.keys() :
            mpipe.print_error("Please choose between one of these media: {0}".format(index_line.key()))
            return

        wavel = list_emission_lines[line][index_line[medium]]
        red_wavel = doppler_shift(wavel, velocity)
        
        return MuseImage(self.select_lambda(red_wavel - width/2., red_wavel + width/2.).sum(axis=0), 
                      title="{0} map".format(line))

def get_sky_spectrum(filename) :
    """Read sky spectrum from MUSE data reduction
    """
    sky = pyfits.getdata(filename)
    crval = sky['lambda'][0]
    cdelt = sky['lambda'][1] - crval
    wavein = WaveCoord(cdelt=cdelt, crval=crval, cunit= units.angstrom)
    spec = Spectrum(wave=wavein, data=sky['data'], var=sky['stat'])
    return spec

class MuseImage(Image): 
    """Wrapper around the mpdaf Image functionalities
    """
    def __init__(self, source=None, **kwargs) :
        """Initialisation of the opening of an Image
        """
        self.verbose = kwargs.pop('verbose', False)
        # Arguments for the plots
        self.title = kwargs.pop('title', "Frame")
        self.scale = kwargs.pop('scale', "log")
        self.vmin = np.int(kwargs.pop('vmin', 0))
        self.colorbar = kwargs.pop('colorbar', "v")

        if source is not None:
            self.__dict__.update(source.__dict__)
        else :
            Image.__init__(self, **kwargs)
 
        self.get_fwhm_startend()

    def get_fwhm_startend(self) :
        """Get range of FWHM
        """
        self.fwhm_startobs = self.primary_header['HIERARCH ESO TEL AMBI FWHM START']
        self.fwhm_endobs = self.primary_header['HIERARCH ESO TEL AMBI FWHM END']

class MuseSetImages(list) :
    """Set of images
    """
    def __new__(self, *args, **kwargs):
        return super(MuseSetImages, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)
        self.update(**kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.update(**kwargs)
        return self

    def update(self, **kwargs) :
        # 1 required attribute
        if not hasattr(self, 'subtitle') :
            if 'subtitle' in kwargs :
                mpipe.print_warning("Overiding subtitle")
            self.subtitle = kwargs.get('subtitle', "")

class MuseSpectrum(Spectrum): 
    """Wrapper around the mpdaf Spectrum functionalities
    """
    def __init__(self, source=None, **kwargs) :
        """Initialisation of the opening of spectra
        """
        self.verbose = kwargs.pop('verbose', False)
        # Arguments for the plots
        self.title = kwargs.pop('title', "Spectrum")
        self.add_sky_lines = kwargs.pop("add_sky_lines", False)
        if self.add_sky_lines :
            self.color_sky = kwargs.pop("color_sky", "red")
            self.linestyle_sky = kwargs.pop("linestyle_sky", "--")
            self.alpha_sky = kwargs.pop("alpha_sky", 0.3)

        if source is not None:
            self.__dict__.update(source.__dict__)
        else :
            Spectrum.__init__(self, **kwargs)

class MuseSetSpectra(list) :
    """Set of spectra
    """
    def __new__(self, *args, **kwargs):
        return super(MuseSetSpectra, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)
        self.update(**kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.update(**kwargs)
        return self

    def update(self, **kwargs) :
        ## 1 required attribute
        if not hasattr(self, 'subtitle') :
            if 'subtitle' in kwargs :
                mpipe.print_warning("Overiding subtitle")
            self.subtitle = kwargs.get('subtitle', "")

