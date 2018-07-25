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
                 name_muse_images,
                 median_window=10,
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
        run: boolean
            If True will use extra_xyoffset and plot the result
        """

        # Some input variables for the cross-correlation
        self.verbose = kwargs.pop("verbose", True)
        self.plot = kwargs.pop("plot", True)
        self.border = border
        self.subim_window = subim_window
        self.median_window = median_window
        self.dynamic_range = dynamic_range
        self.name_reference = name_reference
        self.name_muse_images = name_muse_images

        self.name_musehdr = kwargs.pop("name_musehdr", "muse")
        self.name_offmusehdr = kwargs.pop("name_offmusehdr", "offsetmuse")
        self.name_refhdr = kwargs.pop("name_refhdr", "reference.hdr")

        self._get_list_muse_images(**kwargs)
        if self.nimages == 0: 
            upipe.print_warning("0 MUSE images detected as input")
            return
        self.list_offmuse_hdu = [None] * self.nimages
        self.list_proj_refhdu = [None] * self.nimages

        # Initiallise the output coordinates
        self.cross_arcsec = np.zeros((self.nimages, 2), dtype=np.float16)
        self.cross_pixel = np.zeros_like(self.cross_arcsec)
        self.offset_arcsec = np.zeros_like(self.cross_arcsec)
        self.offset_pixel = np.zeros_like(self.cross_arcsec)
        self.cross_offset_arcsec = np.zeros_like(self.cross_arcsec)
        self.cross_offset_pixel = np.zeros_like(self.cross_arcsec)

        # Which extension to be used for the ref and muse images
        self.hdu_ext = hdu_ext

        # Open the Ref and MUSE image
        self.open_hdu()

        # find the cross correlation peaks for each image
        self.find_ncross_peak()

    def run(self, nima=0, plot=True, **kwargs):
        """Run the offset and comparison
        """
        if nima not in range(self.nimages-1) :
            upipe.print_error("nima not within the range allowed by self.nimages ({0})".format(self.nimages))
        # Overwrite the plot option if given
        self.plot = plot

        if "offset_pixel" in kwargs:
            offset_pixel = kwargs.pop("offset_pixel", [0., 0.])
            offset_arcsec = self.pixel_to_arcsec(self.list_muse_hdu[nima], offset_pixel)
        else:
            offset_arcsec = kwargs.pop("offset_arcsec", [0., 0.])

        # Add the offset from user
        self.shift_arcsecond(offset_arcsec, nima)

        # Compare contours if plot is set to True
        if self.plot:
            self.compare_contours(self.list_offmuse_hdu[nima], 
                    self.list_proj_refhdu[nima], **kwargs)
            
    def _get_list_muse_images(self):
        # test if 1 or several images
        if isinstance(self.name_muse_images, str):
            self.list_muse_images = [self.name_muse_images]
        elif isinstance(self.name_muse_images, list):
            self.list_muse_images = self.name_muse_images
        else:
            upipe.print_warning("Name of images is not a string or a list, "
                    "please check input name_muse_images")
            self.nimages = 0
            return

        # Number of images to deal with
        self.nimages = len(self.list_muse_images)

    def open_hdu(self):
        """OPen the HDU of the MUSE and reference images
        """
        self._open_ref_hdu()
        self._open_muse_nhdu()

    def _open_muse_nhdu(self):
        """Open the MUSE images hdu
        """
        self.list_name_musehdr = ["{0}{1:02d}.hdr".format(self.name_musehdr, i+1) for i in range(self.nimages)]
        self.list_name_offmusehdr = ["{0}{1:02d}.hdr".format(self.name_offmusehdr, i+1) for i in range(self.nimages)]
        list_hdulist_muse = [pyfits.open(self.list_muse_images[i]) for i in range(self.nimages)]
        self.list_muse_hdu = [list_hdulist_muse[i][self.hdu_ext[1]]  for i in range(self.nimages)]

    def _open_ref_hdu(self):
        """Open the reference image hdu
        """
        # Open the images
        hdulist_reference = pyfits.open(self.name_reference)
        self.reference_hdu = hdulist_reference[self.hdu_ext[0]]

    def find_ncross_peak(self):
        """Run the cross correlation peaks on all MUSE images
        """
        for nima in range(self.nimages):
            self.cross_pixel[nima] = self.find_cross_peak(self.list_muse_hdu[nima], 
                    self.list_name_musehdr[nima])
            self.cross_arcsec[nima] = self.pixel_to_arcsec(self.list_muse_hdu[nima],
                    self.cross_pixel[nima])

    def find_cross_peak(self, muse_hdu, name_musehdr):
        """Aligns the MUSE HDU to a reference HDU

        Returns
        -------
        new_hdu : fits.PrimaryHDU
            HDU with header astrometry updated

        """
        # Projecting the reference image onto the MUSE field
        tmphdr = muse_hdu.header.totextfile(name_musehdr,
                                            overwrite=True)
        proj_ref_hdu = self._project_reference_hdu(name_musehdr)

        # Cleaning the images
        ima_ref = self._prepare_image(proj_ref_hdu.data)
        ima_muse = self._prepare_image(muse_hdu.data)

        # Cross-correlate the images
        ccor = correlate(ima_ref, ima_muse, mode='full', method='auto')

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
        xpix_cross = params.x_mean - ccor.shape[1]//2
        ypix_cross = params.y_mean - ccor.shape[0]//2

        return xpix_cross, ypix_cross

    def save_newhdu(self, newhdu_name=None, nima=0):
        """Save the newly determined hdu
        """
        if hasattr(self, "list_offmuse_hdu"):
            if newhdu_name is None:
                newhdu_name = self.list_name_museimages[nima].replace(".fits", "_shift.fits")
            self.list_offmuse_hdu[nima].writeto(newhdu_name, overwrite=True)
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

    def _filtermed_image(self, data):
        """Process image and return it
        """
        # Omit the border pixels
        data = data[self.border:-self.border, self.border:-self.border]

        meddata = nd.filters.median_filter(data, 2)

        return meddata

    def _get_flux_range(self, data, low=10, high=90):
        """Process image and return it
        """
        # Omit the border pixels
        data = data[self.border:-self.border, self.border:-self.border]

        # Clean up the NaNs
        data = np.nan_to_num(data)
        lperc = np.percentile(data[data > 0.], 10)
        hperc = np.percentile(data[data > 0.], 90)

        return lperc, hperc

    def _project_reference_hdu(self, name_hdr):
        """Project the reference image onto the MUSE field
        """
        return montage.reproject_hdu(self.reference_hdu,
                     header=name_hdr, exact_size=True)

    def _add_user_arc_offset(self, offset_arcsec=[0., 0.], nima=0):
        """Add user offset in arcseconds
        """
        # Transforming the arc into pix
        self.offset_arcsec[nima] = offset_arcsec

        # Adding the user offset
        self.cross_offset_arcsec[nima] = self.cross_arcsec[nima] + self.offset_arcsec[nima]

        # Transforming into pixels - would be better with setter
        self.offset_pixel[nima] = self.arcsec_to_pixel(self.list_muse_hdu[nima], 
                self.offset_arcsec[nima])
        self.cross_offset_pixel[nima] = self.arcsec_to_pixel(self.list_muse_hdu[nima], 
                self.cross_offset_arcsec[nima])

    def shift_arcsecond(self, offset_arcsec=[0., 0.], nima=0):
        """Apply shift in arcseconds
        """
        self._add_user_arc_offset(offset_arcsec, nima)
        self.shift(nima)

    def shift(self, nima=0):
        """Create New HDU and send it back
        """
        # Create a new HDU
        newhdr = copy.deepcopy(self.list_muse_hdu[nima].header)

        # Shift the HDU in X and Y
        if self.verbose:
            print("Shifting CRPIX1 by {0:5.4f} pixels "
                  "/ {1:5.4f} arcsec".format(-self.cross_offset_pixel[nima][0],
                -self.cross_offset_arcsec[nima][0]))
        newhdr['CRPIX1'] = newhdr['CRPIX1'] - self.cross_offset_pixel[nima][0]
        if self.verbose:
            print("Shifting CRPIX2 by {0:5.4f} pixels "
                  "/ {1:5.4f} arcsec".format(-self.cross_offset_pixel[nima][1],
                -self.cross_offset_arcsec[nima][1]))
        newhdr['CRPIX2'] = newhdr['CRPIX2'] - self.cross_offset_pixel[nima][1]
        self.list_offmuse_hdu[nima] = pyfits.PrimaryHDU(self.list_muse_hdu[nima].data, header=newhdr)

        tmphdr = self.list_offmuse_hdu[nima].header.totextfile(self.list_name_offmusehdr[nima], overwrite=True)
        self.list_proj_refhdu[nima] = self._project_reference_hdu(self.list_name_offmusehdr[nima])

    def arcsec_to_pixel(self, hdu, xy_arcsec=[0., 0.]):
        """Transform from arcsec to pixel for the muse image
        """
        # Importing theWCS from the muse image
        w = wcs.WCS(hdu.header)
        scale_matrix = np.linalg.inv(w.pixel_scale_matrix * 3600.)
        # Transformation in Pixels
        dels = np.array(xy_arcsec)
        xpix = np.sum(dels * scale_matrix[0, :])
        ypix = np.sum(dels * scale_matrix[1, :])
        return xpix, ypix

    def pixel_to_arcsec(self, hdu, xy_pixel=[0.,0.]):
        """ Transform pixel to arcsec for the muse image
        """
        # Importing theWCS from the muse image
        w = wcs.WCS(hdu.header)
        # Transformation in arcsecond
        dels = np.array(xy_pixel)
        xarc = np.sum(dels * w.pixel_scale_matrix[0, :] * 3600.)
        yarc = np.sum(dels * w.pixel_scale_matrix[1, :] * 3600.)
        return xarc, yarc

    def compare_contours(self, muse_hdu=None, ref_hdu=None, factor=1.0, 
            nfig=1, nlevels=7, muse_smooth=1., ref_smooth=1., samecontour=False):
        """Plot the contours of the projected reference and MUSE image
        """
        if muse_hdu is None:
            muse_hdu = self.list_offmuse_hdu[0]
        if ref_hdu is None:
            ref_hdu = self.list_proj_refhdu[0]

        # Getting the data
        lowlevel_muse, highlevel_muse = self._get_flux_range(muse_hdu.data * factor)
        lowlevel_ref, highlevel_ref = self._get_flux_range(ref_hdu.data)
        if self.verbose:
            print("Low / High level MUSE flux: {0} {1}".format(lowlevel_muse, highlevel_muse))
            print("Low / High level REF  flux: {0} {1}".format(lowlevel_ref, highlevel_ref))
        musedata = self._filtermed_image(muse_hdu.data * factor)
        refdata = self._filtermed_image(ref_hdu.data)
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
        # Adding axes with WCS
        ax = fig.add_subplot(1, 1, 1, projection=musewcs)

        # First contours - MUSE
        levels_muse = np.linspace(np.log10(lowlevel_muse), np.log10(highlevel_muse), nlevels)
        levels_ref = np.linspace(np.log10(lowlevel_ref), np.log10(highlevel_ref), nlevels)
        cmuseset = ax.contour(np.log10(musedata), levels_muse, colors='k', origin='lower')
        # Second contours - Ref
        if samecontour: 
            crefset = ax.contour(np.log10(refdata), levels=cmuseset.levels, colors='r', origin='lower', alpha=0.5)
        else :
            crefset = ax.contour(np.log10(refdata), levels=levels_ref, colors='r', origin='lower', alpha=0.5)
        ax.set_aspect('equal')
        h1,_ = cmuseset.legend_elements()
        h2,_ = crefset.legend_elements()
        ax.legend([h1[0], h2[0]], ['MUSE', 'REF'])
