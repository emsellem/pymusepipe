# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS alignement module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# Importing general modules
import os
import glob
import copy

# Matplotlib
import matplotlib.pyplot as plt

# Numpy Scipy
import numpy as np
import scipy.ndimage as nd
from scipy.signal import correlate

# Astropy
import astropy.wcs as wcs
from astropy.io import fits as pyfits
from astropy.modeling import models, fitting

# Montage
try :
    import montage_wrapper as montage
except ImportError :
    raise Exception("montage_wrapper is required for this module")

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))

import pymusepipe.util_pipe as upipe

# Main alignment Class
class AlignMusePointing(object):
    """Class to align MUSE images onto a reference image
    """
    def __init__(self, name_reference,
                 name_muse_image,
                 median_window=25,
                 subim_window=10,
                 dynamic_range=10,
                 border=50, hdu_ext=[0,1],
                 **kwargs):
        """Initialise the AlignMuseImages class

        Keywords
        --------
        median_window: int
            Size of window used in median filter to extract features in
            cross-correlation.  Should be an odd integer

        subim_window: int
            Size of window for fitting peak of cross-correlation function

        dynamic_range: float
            Apply an arctan transform to data to suppress values more than
            DynamicRange times the median of the image

        border: int
            Ignore pixels this close to the border in the cross correlation

        xoffset: float
            Offset in arcseconds to be added to the cross-correlation offset (X)
        yoffset: float
            Offset in arcseconds to be added to the cross-correlation offset (Y)

        """

        # Some input variables for the cross-correlation
        self.verbose = kwargs.pop("verbose", True)
        self.plot = kwargs.pop("plot", True)
        self.border = border
        self.subim_window = subim_window
        self.median_window = median_window
        self.dynamic_range = dynamic_range
        self.name_reference = name_reference
        self.name_muse_image = name_muse_image
        self.hdu_ext = hdu_ext

        # Open the images
        self.open_hdu()

        # Do the cross-correlation
        self.find_cross_peak(**kwargs)

        # Run the offset and comparison
        self.run()

    def run(self, **kwargs):
        """Run the offset and comparison
        """
        # Add the offset from user
        extra_xyoffset =kwargs.pop("extra_xyoffset", [0., 0.])
        self.shift(extra_xyoffset=extra_xyoffset)

        # Compare contours if plot is set to True
        if self.plot:
            self.compare_contours(**kwargs)

    def _add_user_offset(self, **kwargs):
        """Add user offset
        """
        self.extra_xyoffset = kwargs.pop("extra_xyoffset", [0., 0.])
        # Adding the user offset
        self.xarc_offset = self.xarc_cross + self.extra_xyoffset[0]
        self.yarc_offset = self.yarc_cross + self.extra_xyoffset[1]
        self.xpix_offset, self.ypix_offset = self.arcsec_to_pixel(self.xarc_offset, self.yarc_offset)
            
    def open_hdu(self, **kwargs):
        """OPen the HDU of the MUSE and reference images
        """
        self.name_muse_image = kwargs.pop("name_muse_image", self.name_muse_image)
        self.name_reference = kwargs.pop("name_reference", self.name_reference)
        self.name_refhdr = kwargs.pop("name_refhdr", "reference.hdr")
        self.name_musehdr = kwargs.pop("name_musehdr", "muse.hdr")
        self.name_offmusehdr = kwargs.pop("name_offmusehdr", "offsetmuse.hdr")
        # Open the images
        hdulist_image = pyfits.open(self.name_muse_image)
        hdulist_reference = pyfits.open(self.name_reference)
        self.reference_hdu = hdulist_reference[self.hdu_ext[0]]
        self.muse_hdu = hdulist_image[self.hdu_ext[1]]

    def find_cross_peak(self, **kwargs):
        """Aligns the MUSE HDU to a reference HDU

        Returns
        -------
        new_hdu : fits.PrimaryHDU
            HDU with header astrometry updated

        """
        # Create a new HDU
        self.muse_hdu = kwargs.pop("muse_hdu", self.muse_hdu)
        self.reference_hdu = kwargs.pop("reference_hdu", self.reference_hdu)

        tmphdr = self.muse_hdu.header.totextfile(self.name_musehdr,
                                            overwrite=True)
        tmphdu = self._project_reference_hdu(self.name_musehdr)
        im1 = self._prepare_image(tmphdu.data)
        im2 = self._prepare_image(self.muse_hdu.data)

        # Cross-correlate the images
        ccor = correlate(im1, im2, mode='full', method='auto')

        # Find peak of cross-correlation
        maxy, maxx = np.unravel_index(np.argmax(ccor),
                                      ccor.shape)

        # Extract a window around it
        window = self.subim_window
        y, x = np.ix_(np.arange(-window + maxy, window + 1 + maxy),
                      np.arange(-window + maxx, window + 1 + maxx))
        subim = ccor[y % ccor.shape[0], x % ccor.shape[1]]
        subim -= subim.min()
        mx = np.max(subim)
        smaxy, smaxx = np.unravel_index(np.argmax(subim),
                                        subim.shape)

        # Fit a 2D Gaussian to that peak
        gauss_init = models.Gaussian2D(amplitude=mx,
                                       x_mean=x[0, smaxx],
                                       y_mean=y[smaxy, 0],
                                       x_stddev=2,
                                       y_stddev=2,
                                       theta=0)
        fitter = fitting.LevMarLSQFitter()
        params = fitter(gauss_init, x * np.ones_like(y),
                        y * np.ones_like(x),
                        subim)

        # Update Astrometry
        self.xpix_cross = params.x_mean - ccor.shape[1]//2
        self.ypix_cross = params.y_mean - ccor.shape[0]//2

        self.xarc_cross, self.yarc_cross = self.pixel_to_arcsec(self.xpix_cross, 
                self.ypix_cross)

    def save_newhdu(self, name=None):
        """Save the newly determined hdu
        """
        if hasattr(self, "offset_hdu"):
            if name is None:
                self.newhdu_name = self.name_muse_image.replace(".fits", "_shift.fits")
            self.offset_hdu.writeto(self.newhdu_name, overwrite=True)
        else:
            upipe.print_error("There are not yet any new hdu to save")

    def _prepare_image(self, data):
        """Process image and return it
        """
        # Squish bright pixels down
        data = np.arctan(data / np.nanmedian(data) / self.dynamic_range)

        # Omit the border pixels
        data = data[self.border:-self.border, self.border:-self.border]

        medim1 = nd.filters.median_filter(data, self.median_window)
        data -= medim1
        data[data < 0] = 0.

        # Clean up the NaNs
        data = np.nan_to_num(data)

        return data

    def _project_reference_hdu(self, name_hdr=None):
        """Project the reference image onto the MUSE field
        """
        if name_hdr is None: 
            name_hdr = self.name_musehdr
        return montage.reproject_hdu(self.reference_hdu,
                     header=name_hdr, exact_size=True)

    def shift(self, **kwargs):
        """Create New HDU and send it back
        """
        self._add_user_offset(**kwargs)
        # Create a new HDU
        newhdr = copy.deepcopy(self.muse_hdu.header)

        # Shift the HDU in X and Y
        if self.verbose:
            print('Shifting CRPIX1 by {0:5.4f} pixels / {1:5.4f} arcsec'.format(-self.xpix_offset,
                -self.xarc_offset))
        newhdr['CRPIX1'] = newhdr['CRPIX1'] - self.xpix_offset
        if self.verbose:
            print('Shifting CRPIX2 by {0:5.4f} pixels / {1:5.4f} arcsec'.format(-self.ypix_offset,
                -self.yarc_offset))
        newhdr['CRPIX2'] = newhdr['CRPIX2'] - self.ypix_offset
        self.offset_hdu = pyfits.PrimaryHDU(self.muse_hdu.data, header=newhdr)

        tmphdr = newhdr.totextfile(self.name_offmusehdr, overwrite=True)
        self.proj_reference_hdu = self._project_reference_hdu(self.name_offmusehdr)

    def arcsec_to_pixel(self, xarc=0., yarc=0.):
        """Transform from arcsec to pixel for the muse image
        """
        # Importing theWCS from the muse image
        w = wcs.WCS(self.muse_hdu.header)
        scale_matrix = np.linalg.inv(w.pixel_scale_matrix * 3600.)
        # Transformation in Pixels
        dels = np.array([xarc, yarc])
        xpix = np.sum(dels * scale_matrix[0, :])
        ypix = np.sum(dels * scale_matrix[1, :])
        return xpix, ypix

    def pixel_to_arcsec(self, xpix=0., ypix=0.):
        """ Transform pixel to arcsec for the muse image
        """
        # Importing theWCS from the muse image
        w = wcs.WCS(self.muse_hdu.header)
        # Transformation in arcsecond
        dels = np.array([xpix, ypix])
        xarc = np.sum(dels * w.pixel_scale_matrix[0, :] * 3600.)
        yarc = np.sum(dels * w.pixel_scale_matrix[1, :] * 3600.)
        return xarc, yarc

    def compare_contours(self, muse_hdu=None, ref_hdu=None, factor=1.0, 
            nfig=1, nlevels=20, muse_smooth=0., ref_smooth=0.):
        """Plot the contours of the projected reference and MUSE image
        """
        if muse_hdu is None:
            muse_hdu = self.offset_hdu
        if ref_hdu is None:
            ref_hdu = self.proj_reference_hdu

        # Getting the data
        musedata = self._prepare_image(muse_hdu.data * factor)
        refdata = self._prepare_image(ref_hdu.data)
        # Smoothing out the result in case it is needed
        if muse_smooth > 0 :
           musedata = nd.filters.gaussian_filter(musedata, muse_smooth)
        if ref_smooth > 0 :
           refdata = nd.filters.gaussian_filter(refdata, ref_smooth)

        # Get the WCS
        musewcs = wcs.WCS(muse_hdu.header)
        refwcs = wcs.WCS(ref_hdu.header)

        # PReparing the figure
        fig = plt.figure(nfig)
        plt.clf()
        # Addinb an axes with WCS
        ax = fig.add_subplot(1, 1, 1, projection=musewcs)

        # First contours - MUSE
        cmuseset = ax.contour(musedata, nlevels, colors='k', origin='lower')
        # Second contours - Ref
        crefset = ax.contour(refdata, levels=cmuseset.levels, colors='r', origin='lower', alpha=0.5)
        ax.set_aspect('equal')
        h1,_ = cmuseset.legend_elements()
        h2,_ = crefset.legend_elements()
        ax.legend([h1[0], h2[0]], ['MUSE', 'REF'])
