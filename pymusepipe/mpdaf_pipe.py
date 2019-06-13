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
from mpdaf.tools import add_mpdaf_method_keywords

# Astropy
from astropy.io import fits as pyfits
from astropy import units as units

from pymusepipe import util_pipe as upipe

# Versioning
__version__ = '0.1.0 (31 May 2019)'

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

    def get_spectrum_from_cube(self, nx=None, ny=None, pixel_window=0, title="Spectrum") :
        """Get a spectrum from the cube with centre defined in pixels
        with nx, ny and a window of 'pixel_window'
        """
        if nx == None : nx = self.shape[2] // 2
        if ny == None : ny = self.shape[1] // 2
        pixel_halfwindow = pixel_window // 2
        subcube = self[:, ny - pixel_halfwindow: ny + pixel_halfwindow + 1, 
                    nx - pixel_halfwindow: nx + pixel_halfwindow + 1]
        return MuseSpectrum(source=subcube.sum(axis=(1,2)), title=title)

    def get_whiteimage_from_cube(self) :
        return MuseImage(source=self.sum(axis=0), title="White Image")

    def get_image_from_cube(self, central_lambda=None, lambda_window=0) :
        """Get image from integrated cube, with spectral pixel
        centred at central_lambda and with a lambda_window of lambda_window
        """
        if central_lambda == None : central_lambda = self.shape[0] // 2
        lambda_halfwindow = lambda_window // 2
        return MuseImage(source=self[central_lambda - lambda_halfwindow: 
            central_lambda + lambda_halfwindow + 1, :, :].sum(axis=0))

    def get_set_spectra(self) :
        """Get a set of standard spectra from the Cube
        """
        self.spec_fullgalaxy = MuseSpectrum(source=self.sum(axis=(1,2)), title="Full galaxy Spectrum")
        self.spec_4quad = self.get_quadrant_spectra_from_cube()
        self.spec_central_aper = MuseSetSpectra(
               self.get_spectrum_from_cube(pixel_window=0, title="Central aperture"), 
               self.get_spectrum_from_cube(pixel_window=20, title="Central Aperture, w=20"), 
               self.get_spectrum_from_cube(pixel_window=40, title="Central Aperture, w=40"),
               subtitle="central_spectra")

    def get_quadrant_spectra_from_cube(self, pixel_window=0) :
        """Get quadrant spectra from the Cube

        Input
        ----
        pixel_window : pixel_window of integration
        """
        ny4 = self.shape[1] // 4
        nx4 = self.shape[2] // 4
        nx34, ny34 = 3 * nx4, 3 * ny4

        spec1 = self.get_spectrum_from_cube( nx4,  ny4, pixel_window, title="Quadrant 1")
        spec2 = self.get_spectrum_from_cube( nx4, ny34, pixel_window, title="Quadrant 2") 
        spec3 = self.get_spectrum_from_cube(nx34,  ny4, pixel_window, title="Quadrant 3") 
        spec4 = self.get_spectrum_from_cube(nx34, ny34, pixel_window, title="Quadrant 4") 
        return MuseSetSpectra(spec1, spec2, spec3, spec4, subtitle="4 Quadrants")

    def get_emissionline_image(self, line=None, velocity=0., redshift=None, lambda_window=10., medium='vacuum') :
        """Get a narrow band image around Ha

        Input
        -----
        lambda_window: in Angstroems (10 by default). Width of the window of integration
        medium: vacuum or air (string, 'vacuum' by default)
        velocity: default is 0. (km/s)
        redshift: default is None. Overwrite velocity if provided.
        line: name of the emission line (see emission_lines dictionary)
        """

        [lmin, lmax] = upipe.get_emissionline_band(line=line, velocity=velocity, 
                redshift=redshift, medium=medium, lambda_window=lambda_window)
        
        return MuseImage(self.select_lambda(lmin, lmax).sum(axis=0), 
                title="{0} map".format(line))

    def get_filter_image(self, filter_name=None, own_filter_file=None, filter_folder="",
            dic_extra_filters=None):
        """Get an image given by a filter. If the filter belongs to
        the filter list, then use that, otherwise use the given file
        """
        try:
            upipe.print_info("Reading MUSE filter {0}".format(filter_name))
            refimage = self.get_band_image(filter_name)
        except ValueError:
            # initialise the filter file
            upipe.print_info("Reading private reference filter {0}".format(filter_name))
            if dic_extra_filters is None:
                upipe,print_error("No extra filter directory given for private filter "
                        "[get_filter_image in mpdaf_pipe]")
            if filter_name in dic_extra_filters.keys():
                filter_file = dic_extra_filters[filter_name]
            else:
                if own_filter_file is None:
                    upipe.print_error("Private filter file is not set")
                    return
                else:
                    filter_file = own_filter_file
            # Now reading the filter data
            path_filter = joinpath(filter_folder, filter_file)
            filter_wave, filter_sensitivity = np.loadtxt(path_filter, unpack=True)
            refimage = self.bandpass_image(filter_wave, filter_sensitivity, 
                                           interpolation='linear')
            key = 'HIERARCH ESO DRS MUSE FILTER NAME'
            refimage.primary_header[key] = (filter_name, 'filter name used')
            add_mpdaf_method_keywords(refimage.primary_header, 
                    "cube.bandpass_image", ['name'], 
                    [filter_name], ['filter name used'])
            # Only selecting the primary + data HDU (removing DQ, STAT)
            data_refimage = pyfits.HDUList([refimage[0], refimage[1]])
        return data_refimage

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
                upipe.print_warning("Overiding subtitle")
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
                upipe.print_warning("Overiding subtitle")
            self.subtitle = kwargs.get('subtitle', "")

