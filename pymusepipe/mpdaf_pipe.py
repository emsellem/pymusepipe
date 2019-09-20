# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS mpdaf-functions module
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
from mpdaf.drs import PixTable

# Astropy
from astropy.io import fits as pyfits
from astropy import units as units

from pymusepipe import util_pipe as upipe

# Versioning
__version__ = '0.2.0 (9 Sep 2019)' # Adding PixTableToMask
#__version__ = '0.1.0 (31 May 2019)'

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
            if dic_extra_filters is not None:
                if filter_name in dic_extra_filters.keys():
                    filter_file = dic_extra_filters[filter_name]
                else:
                    upipe.print_error("Filter name not in private dictionary"
                            "[get_filter_image in mpdaf_pipe]")
                    upipe.print_error("Aborting"
                    return
            else:
                if own_filter_file is None:
                    upipe.print_error("No extra filter dictionary and...")
                    upipe.print_error("the private filter file is not set")
                    upipe.print_error("Aborting"
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
#        # Only selecting the primary + data HDU (removing DQ, STAT)
#        data_refimage = pyfits.HDUList([refimage[0], refimage[1]])
        return refimage

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

    def mask_trail(self, pq1=[0,0], pq2=[10,10], width=0.0, reset=False, extent=None):
        """Build an image mask from 2 points measured from a trail
    
        Input
        -----
        pq1: array or tuple (float)
            p and q coordinates of point 1 along the trail
        pq2: array or tuple (float) 
            p and q coordinates of point 2 along the trail
        width: float
            Value (in pixel) of the full slit width to exclude
        extent: float
            Value (in pixel) to extend the slit beyond the 2 extrema
            If 0, this means limiting it to the extrema themselves.
            Default is None, which mean infinitely long slit
        """
        # If width = 0 just don't do anything
        if width <= 0.:
            upipe.print_warning("Trail width is 0, hence no doing anything")
            return

        # Create an index grid using the Image shape
        # Note that p is the Y axis, while q is the X axis
        ind = np.indices(self.data.shape)

        # X2 - X1 -> note that X is q, hence the second value
        # Y2 - Y1 -> note that Y is p, hence the first value
        y21, x21 = np.array(pq2) - np.array(pq1)

        # Mask based on the distance to the line defined by the 2 points
        mask_image = np.abs(y21 * (ind[1] - pq1[1]) - x21 * (ind[0] - pq1[0])) \
                       / np.sqrt(x21**2 + y21**2) < (width / 2.)
        # Mask based on the extent of the slit
        if extent is not None:
            y01, x01 = ind[0] - pq1[0], ind[1] - pq1[1]
            y02, x02 = ind[0] - pq2[0], ind[1] - pq2[1]
            angle21 = np.arctan2(y01, x01) - np.arctan2(y21, x21)
            xcoord1 = np.hypot(x01, y01) * np.cos(angle21)
            ext_mask = (xcoord1 > -extent) & (xcoord1 - np.hypot(x21, y21) < extent)  
            mask_image = mask_image & ext_mask

        # Reset mask if requested
        if reset:
            self.reset_mask()

        self.mask[mask_image] = True

    def reset_mask(self):
        """Resetting the Image mask
        """
        # just resetting the mask of the image
        self.mask[:,:] = False

    def save_mask(self, mask_name="dummy_mask.fits"):
        """Save the mask into a 0-1 image
        """
        newimage = self.copy()
        newimage.data = np.where(self.mask, 0, 1).astype(np.int)
        newimage.mask[:,:] = False
        newimage.write(mask_name)

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

class PixTableToMask(object):
    """This class is meant to just be a simple tool to
    mask out some regions from the PixTable using Image masks
    """
    def __init__(self, pixtable_name, image_name, suffix_out="tmask"):
        """Class made to associate a PixTable and an Image
        Used to select all pixels from a pixtable outside the trail 
        defined by the trail mask of an image
        Then flushes the pixtable into a new one.

        Input
        -----
        pixtable_name: str
            Name of the Pixel Table
        image_name: str
            Name of Image which should include a mask
        suffix_out: str
            Suffix to be used for the new PixTable
        """
        self.suffix_out = suffix_out
        if not os.path.isfile(pixtable_name):
            upipe.print_error("Input PixTable does not exist")
            return
        if not os.path.isfile(image_name):
            upipe.print_error("Input Image does not exist")
            return
        self.image_name = image_name
        self.pixtable_name = pixtable_name

        self.pixtable_folder, self.pixtable_name = os.path.split(pixtable_name)
        self.image_folder, self.image_name = os.path.split(self.image_name)

        self.image = MuseImage(filename=image_name)

    def imshow(self, **kwargs):
        """ Just showing the image
        """
        self.image.plot(**kwargs)

    def create_mask(self, pq1=[0,0], pq2=[10,10], width=0.0, reset=False,
                        mask_name="dummy_mask.fits", extent=None, **kwargs):
        """Create the mask and save it in one go
   
        Input
        -----
        pq1: array or tuple (float)
            p and q coordinates of point 1 along the trail
        pq2: array or tuple (float) 
            p and q coordinates of point 2 along the trail
        width: float
            Value (in pixel) of the full slit width to exclude
        reset: bool
            By default False, so the mask goes on top of the existing one
            If True, will reset the mask before building it.
        extent: float
            Value (in pixel) to extend the slit beyond the 2 extrema
            If 0, this means limiting it to the extrema themselves.
            Default is None, which mean infinitely long slit
        """
        self.image.mask_trail(pq1=pq1, pq2=pq2, width=width, reset=reset, extent=extent)
        self.save_mask(mask_name, **kwargs)

    def save_mask(self, mask_name="dummy_mask.fits", use_folder=True):
        """Saving the mask from the Image into a fits file

        Input
        -----
        mask_name: str
            Name of the fits file for the mask
        use_folder: bool
            If True (default) will look for the mask in the image_folder.
            If False, will just look for it where the command is run.

        Creates
        -------
        A fits file with the mask as 0 and 1
        """
        if use_folder:
            self.mask_name = joinpath(self.image_folder, mask_name)
        else:
            self.mask_name = mask_name

        # Using the image folder to save the mask
        self.image.save_mask(self.mask_name)

    def mask_pixtable(self, mask_name=None, **kwargs):
        """Use the Image Mask and create a new Pixtable

        Input
        -----
        mask_name: str
            Name of the mask to be used (FITS file)
        use_folder: bool
            If True, use the same folder as the Pixtable
            Otherwise just write where you stand
        suffix_out: str
            Suffix for the name of the output Pixtable
            If provided, will overwrite the one in self.suffix_out
        """
        # Open the PixTable
        upipe.print_info("Opening the Pixtable {0}".format(
                          self.pixtable_name))
        pixtable = PixTable(self.pixtable_name)

        # Use the Image mask and create a pixtable mask
        if mask_name is not None:
            self.mask_name = mask_name
        else:
            if not hasattr(self, "mask_name"):
                upipe.print_error("Please provide a mask name (FITS file)")
                return

        upipe.print_info("Creating a column Mask from file {0}".format(
                          self.mask_name))
        mask_col = pixtable.mask_column(self.mask_name)

        # extract the right data using the pixtable mask
        upipe.print_info("Extracting the Mask")
        newpixtable = pixtable.extract_from_mask(mask_col.maskcol)

        # Rewrite a new pixtable
        self.suffix_out = kwargs.pop("suffix_out", self.suffix_out)
        use_folder = kwargs.pop("use_folder", True)
        if use_folder:
            self.newpixtable_name = joinpath(self.pixtable_folder, "{0}{1}".format(
                                    self.suffix_out, self.pixtable_name))
        else :
            self.newpixtable_name = "{0}{1}".format(self.suffix_out, self.pixtable_name)

        upipe.print_info("Writing the new PixTable in {0}".format(
                          self.newpixtable_name))
        newpixtable.write(self.newpixtable_name)

        # Now transfer the flat field if it exists
        ext_name = 'PIXTABLE_FLAT_FIELD'
        try:
            # Test if Extension exists by reading header
            # If it exists then do nothing
            test_data = pyfits.getheader(self.newpixtable_name, ext_name)
            upipe.print_warning("Flat field extension already exists in masked PixTable - all good")
        # If it does not exist test if it exists in the original PixTable
        except KeyError:
            try:
                # Read data and header
                ff_ext_data = pyfits.getdata(self.pixtable_name, ext_name)
                ff_ext_h = pyfits.getheader(self.pixtable_name, ext_name)
                upipe.print_warning("Flat field extension will be transferred from PixTable")
                # Append it to the new pixtable
                pyfits.append(self.newpixtable_name, ff_ext_data, ff_ext_h)
            except KeyError:
                upipe.print_warning("No Flat field extension to transfer - all good")
            except:
                pass
        except:
            pass

        # Patch to fix the extension names of the PixTable
        # We have to put a number of extension in lowercase to make sure 
        # the MUSE recipes understand them
        descl = ['xpos', 'ypos', 'lambda', 'data', 'dq', 'stat', 'origin']
        for d in descl:
            try:
                pyfits.setval(self.newpixtable_name, keyword='EXTNAME', 
                        value=d, extname=d.upper())
                upipe.print_warning("Rewriting extension name {0} as lowercase".format(
                                       d.upper()))
            except:
                upipe.print_warning("Extension {0} not present - patch ignored".format(
                                       d.upper()))

