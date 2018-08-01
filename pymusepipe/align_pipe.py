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
from scipy.odr import ODR, Model, RealData

# Astropy
import astropy.wcs as wcs
from astropy.io import fits as pyfits
from astropy.modeling import models, fitting
from astropy.stats import mad_std
from astropy.table import Table

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

# ================== Useful function ====================== #
def open_new_wcs_figure(nfig, mywcs=None):
    """Open a new figure with wcs projection
    """
    fig = plt.figure(nfig)
    plt.clf()
    # Adding axes with WCS
    if mywcs is None:
        return fig, fig.add_subplot(1, 1, 1)
    else:
        return fig, fig.add_subplot(1, 1, 1, projection=mywcs)

def chunk_stats(list_data, chunk_size=15):
    """Cut the datasets in chunks and take the median
    Return the set of medians
    """
    ndatasets = len(list_data)

    nchunk_x = np.int(list_data[0].shape[0] // chunk_size - 1)
    nchunk_y = np.int(list_data[0].shape[1] // chunk_size - 1)
    # Check that all datasets have the same size
    med_data = np.zeros((ndatasets, nchunk_x * nchunk_y), dtype=np.float32)
    std_data = np.zeros_like(med_data)

    if not all([d.size for d in list_data]):
        upipe.print_error("Datasets are not of the same size in median_compare")
    else:
        for i in range(0, nchunk_x):
            for j in range(0, nchunk_y):
                for k in range(ndatasets):
                    # Taking the median
                    med_data[k, i*nchunk_y + j] = np.median(list_data[k][i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size])
                    std_data[k, i*nchunk_y + j] = mad_std(list_data[k][i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size])
   
    return med_data, std_data

def my_linear(B, x):
    """Linear function for the regression
    """
    return B[1] * (x + B[0])

def get_image_norm_poly(data1, data2, chunk_size=15):
    """Find the normalisation factor between two datasets
    Including the background and slope
    """
    med, std = chunk_stats([data1, data2], chunk_size=chunk_size)
    pos = (med[0] > 0.) & (std[0] > 0.) & (std[1] > 0.) & (med[1] > 0.)
    guess_slope = np.median(med[1][pos] / med[0][pos])
    result = regress_odr(x=med[0][pos], y=med[1][pos], sx=std[0][pos], sy=std[1][pos], beta0=[0., guess_slope])
    return result

def regress_odr(x, y, sx, sy, beta0=[0., 1.]):
    """Return an ODR linear fit
    """
    linear = Model(my_linear)
    mydata = RealData(x.ravel(), y.ravel(), sx=sx.ravel(), sy=sy.ravel())
    myodr = ODR(mydata, linear, beta0=beta0)
    return myodr.run()

def arcsec_to_pixel(hdu, xy_arcsec=[0., 0.]):
    """Transform from arcsec to pixel for the muse image
    """
    # Matrix
    input_wcs = wcs.WCS(hdu.header)
    scale_matrix = np.linalg.inv(input_wcs.pixel_scale_matrix * 3600.)

    # Transformation in Pixels
    dels = np.array(xy_arcsec)
    xpix = np.sum(dels * scale_matrix[0, :])
    ypix = np.sum(dels * scale_matrix[1, :])
    return xpix, ypix

def pixel_to_arcsec(hdu, xy_pixel=[0.,0.]):
    """ Transform pixel to arcsec for the muse image
    """
    # Matrix
    input_wcs = wcs.WCS(hdu.header)

    # Transformation in arcsecond
    dels = np.array(xy_pixel)
    xarc = np.sum(dels * input_wcs.pixel_scale_matrix[0, :] * 3600.)
    yarc = np.sum(dels * input_wcs.pixel_scale_matrix[1, :] * 3600.)
    return xarc, yarc

#################################################################
# ================== END Useful function ====================== #
#################################################################

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

        self.flag = kwargs.pop("flag", "")

        self._get_list_muse_images(**kwargs)
        if self.nimages == 0: 
            upipe.print_warning("0 MUSE images detected as input")
            return
        self.list_offmuse_hdu = [None] * self.nimages
        self.list_proj_refhdu = [None] * self.nimages

        self.cross_off_pixel = np.zeros((self.nimages, 2), dtype=np.float32)
        self.extra_off_pixel = np.zeros_like(self.cross_off_pixel)
        self.init_off_pixel = np.zeros_like(self.cross_off_pixel)
        self.total_off_pixel = np.zeros_like(self.cross_off_pixel)
        self.cross_off_arcsec = np.zeros_like(self.cross_off_pixel)
        self.extra_off_arcsec = np.zeros_like(self.cross_off_pixel)
        self.init_off_arcsec = np.zeros_like(self.cross_off_pixel)
        self.total_off_arcsec = np.zeros_like(self.cross_off_pixel)

        # Cross normalisation for the images
        # This contains the 2 parameters for a linear transformation
        self.ima_polynorm = np.zeros((self.nimages, 2), dtype=np.float32)

        # Which extension to be used for the ref and muse images
        self.hdu_ext = hdu_ext

        # Open the Ref and MUSE image
        self.open_hdu()

        # find the cross correlation peaks for each image
        self.find_ncross_peak()

        # Initialise the offset
        firstguess = kwargs.pop("firstguess", "crosscorr")
        self.init_guess_offset(firstguess, **kwargs)

        self.total_off_arcsec = self.init_off_arcsec + self.extra_off_arcsec
        self.total_off_pixel = self.init_off_pixel + self.extra_off_pixel

    def init_guess_offset(self, firstguess="crosscorr", name_input_table="OFFSET_LIST.fits"):
        """Initialise first guess, either from cross-correlation (default)
        or from an Offset FITS Table
        """
        # Implement the guess
        if firstguess not in ["crosscorr", "fits"]:
            firstguess = "crosscorr"
            upipe.print_warning("Keyword 'firstguess' not recognised")
            upipe.print_warning("Using Cross-Correlation as a first guess of the alignment")

        if firstguess == "crosscorr":
            self.init_off_arcsec = self.cross_off_arcsec
            self.init_off_pixel = self.cross_off_pixel
        elif firstguess == "fits":
            self.name_input_table = name_input_table
            exist_table, self.offset_table = self.open_offset_table(self.name_input_table)
            if exist_table is not True:
                upipe.print_warning("Fits initialisation table not found, setting init value to 0")
                self.init_off_pixel *= 0.
                self.init_off_arcsec *= 0.
            else:
                self.init_off_arcsec = np.vstack((self.offset_table['RA_OFFSET'], 
                    self.offset_table['DEC_OFFSET'])).T * 3600.
                for i in range(self.nimages):
                    self.init_off_pixel[nima] = arcsec_to_pixel(self.list_muse_hdu[nima],
                            self.init_off_arcsec[nima])

    def open_offset_table(self, name_table=None):
        """Read offset table from fits file
        """
        if name_table is None:
            if not hasattr(self, name_table):
                upipe.print_error("No FITS table name provided, Aborting Open")
                return None, Table()

        if not os.path.isfile(name_table):
            upipe.print_warning("FITS Table does not exist yet"
                "({0})".format(name_table))
            return False, Table()

        return True, Table.read(name_table)

    def print_offset_fromfits(self, name_table=None):
        """Print out the offset
        """
        exist_table, fits_table = self.open_offset_table(name_table)
        if exist_table is None:
            return

        if ('RA_OFFSET' not in fits_table.columns) or ('DEC_OFFSET' not in fits_table.columns):
            upipe.print_error("Table does not contain 'RA/DEC_OFFSET' columns, Aborting")
            return

        upipe.print_info("Offset recorded in OFFSET_LIST Table")
        upipe.print_info("Total in ARCSEC")
        for i in self.nimages:
            upipe.print_info("Image {0}: "
                "{1:8.4f} {1:8.4f}".format(fits_table['RA_OFFSET']*3600,
                    fits_table['DEC_OFFSET']*3600.))

    def print_offset(self):
        """Print out the offset
        """
        upipe.print_info("#---- Offset recorded so far ----#")
        upipe.print_info("Total in ARCSEC")
        for nima in range(self.nimages):
            upipe.print_info("    Image {0:02d}: "
                "{1:8.4f} {2:8.4f}".format(nima, self.total_off_arcsec[nima][0],
                    self.total_off_arcsec[nima][1]))
        upipe.print_info("Total in PIXEL")
        for nima in range(self.nimages):
            upipe.print_info("    Image {0:02d}: "
                "{1:8.4f} {2:8.4f}".format(nima, self.total_off_pixel[nima][0],
                    self.total_off_pixel[nima][1]))

    def save_fits_offset_table(self, name_output_table=None, overwrite=False, suffix=""):
        """Save the Offsets into a fits Table
        """
        if name_out_table is None: name_out_table = self.name_input_table
        self.suffix = suffix
        name_output_table = name_output_table.replace(".fits", "{0}.fits".format(self.suffix))

        exist_table, fits_table = self.open_offset_table(self.name_out_table)
        if exist_table is None:
            upipe.print_error("Save is aborted")
            return

        # Check if RA_OFFSET is there
        if 'RA_OFFSET' in fits_table.columns:
            # if yes, then check if the ORIG column is there
            if 'RA_OFFSET_ORIG' not in fits_table.columns:
                fits_table['RA_OFFSET_ORIG'] = fits_table['RA_OFFSET']
                fits_table['DEC_OFFSET_ORIG'] = fits_table['DEC_OFFSET']
            # if not, just continue
            # as it means the ORIG columns were already done

        # Saving the final values
        fits_table['RA_OFFSET'] = self.total_off_arcsec[:,0] / 3600.
        fits_table['DEC_OFFSET'] = self.total_off_arcsec[:,1] / 3600.
        fits_table['RA_CROSS_OFFSET'] = self.cross_off_arcsec[:,0] / 3600.
        fits_table['DEC_CROSS_OFFSET'] = self.cross_off_arcsec[:,1] / 3600.

        # Writing it up
        if exist_table and not overwrite:
            upipe.print_warning("Table already exists, but overwrite is set to False")
            upipe.print_warning("If you wish to overwrite the table {0}, "
                    "please set overwrite to True".format(name_output_table))
            return

        fits_table.write(self.name_output_table, overwrite=overwrite)
        self.name_output_table = name_output_table

    def run(self, nima=0, **kwargs):
        """Run the offset and comparison
        """
        if nima not in range(self.nimages) :
            upipe.print_error("nima not within the range allowed by self.nimages ({0})".format(self.nimages))
            return

        # Overwrite the plot option if given
        self.plot = kwargs.pop("plot", self.plot)

        if "extra_pixel" in kwargs:
            extra_pixel = kwargs.pop("extra_pixel", [0., 0.])
            extra_arcsec = pixel_to_arcsec(self.list_muse_hdu[nima], extra_pixel)
        else:
            extra_arcsec = kwargs.pop("extra_arcsec", [0., 0.])

        # Add the offset from user
        self.shift_arcsecond(extra_arcsec, nima)

        # Compare contours if plot is set to True
        if self.plot:
            self.compare(self.list_offmuse_hdu[nima], 
                    self.list_proj_refhdu[nima], nima=nima, **kwargs)
            
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
        self.list_muse_wcs = [wcs.WCS(hdu.header) for hdu in self.list_muse_hdu]

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
            self.cross_off_pixel[nima] = self.find_cross_peak(self.list_muse_hdu[nima], 
                    self.list_name_musehdr[nima])
            self.cross_off_arcsec[nima] = pixel_to_arcsec(self.list_muse_hdu[nima],
                    self.cross_off_pixel[nima])

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

    def save_image(self, newfits_name=None, nima=0):
        """Save the newly determined hdu
        """
        if hasattr(self, "list_offmuse_hdu"):
            if newfits_name is None:
                newfits_name = self.list_name_museimages[nima].replace(".fits", "_shift.fits")
            self.list_offmuse_hdu[nima].writeto(newfits_name, overwrite=True)
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

    def _filtermed_image(self, data, filter_size=2):
        """Process image and return it
        """
        # Omit the border pixels
        data = data[self.border:-self.border, self.border:-self.border]

        meddata = nd.filters.median_filter(data, filter_size)

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

    def _add_user_arc_offset(self, extra_arcsec=[0., 0.], nima=0):
        """Add user offset in arcseconds
        """
        # Transforming the arc into pix
        self.extra_off_arcsec[nima] = extra_arcsec

        # Adding the user offset
        self.total_off_arcsec[nima] = self.init_off_arcsec[nima] + self.extra_off_arcsec[nima]

        # Transforming into pixels - would be better with setter
        self.extra_off_pixel[nima] = arcsec_to_pixel(self.list_muse_hdu[nima], 
                self.extra_off_arcsec[nima])
        self.total_off_pixel[nima] = arcsec_to_pixel(self.list_muse_hdu[nima], 
                self.total_off_arcsec[nima])

    def shift_arcsecond(self, extra_arcsec=[0., 0.], nima=0):
        """Apply shift in arcseconds
        """
        self._add_user_arc_offset(extra_arcsec, nima)
        self.shift(nima)

    def shift(self, nima=0):
        """Create New HDU and send it back
        """
        # Create a new HDU
        newhdr = copy.deepcopy(self.list_muse_hdu[nima].header)

        # Shift the HDU in X and Y
        if self.verbose:
            print("Shifting CRPIX1 by {0:8.4f} pixels "
                  "/ {1:8.4f} arcsec".format(-self.total_off_pixel[nima][0],
                -self.total_off_arcsec[nima][0]))
        newhdr['CRPIX1'] = newhdr['CRPIX1'] - self.total_off_pixel[nima][0]
        if self.verbose:
            print("Shifting CRPIX2 by {0:8.4f} pixels "
                  "/ {1:8.4f} arcsec".format(-self.total_off_pixel[nima][1],
                -self.total_off_arcsec[nima][1]))
        newhdr['CRPIX2'] = newhdr['CRPIX2'] - self.total_off_pixel[nima][1]
        self.list_offmuse_hdu[nima] = pyfits.PrimaryHDU(self.list_muse_hdu[nima].data, header=newhdr)

        tmphdr = self.list_offmuse_hdu[nima].header.totextfile(self.list_name_offmusehdr[nima], overwrite=True)
        self.list_proj_refhdu[nima] = self._project_reference_hdu(self.list_name_offmusehdr[nima])


    def compare(self, muse_hdu=None, ref_hdu=None, factor=1.0, 
            start_nfig=1, nlevels=7, levels=None, muse_smooth=1., 
            ref_smooth=1., samecontour=True, nima=0,
            showcontours=True, showcuts=True, difference=True,
            normalise=True, median_filter=True, ncuts=5, showdiff=True,
            percentage=10.):
        """Plot the contours of the projected reference and MUSE image
        """
        # Getting the data
        if muse_hdu is None:
            muse_hdu = self.list_offmuse_hdu[nima]
        if ref_hdu is None:
            ref_hdu = self.list_proj_refhdu[nima]

        # Filtering the data if median_filter is True
        if median_filter :
            musedata = self._filtermed_image(muse_hdu.data)
            refdata = self._filtermed_image(ref_hdu.data)
        else:
            musedata = muse_hdu.data
            refdata = ref_hdu.data

        # If normalising, using the median ratio fit
        if normalise :
            polypar = get_image_norm_poly(musedata, refdata, chunk_size=15)
            if self.verbose:
                upipe.print_info("Renormalising the MUSE data as NewMUSE = "
                        "{1:8.4f} * ({0:8.4f} + MUSE)".format(polypar.beta[0], polypar.beta[1]))

            musedata = (polypar.beta[0] + musedata) * polypar.beta[1]
            self.ima_polynorm[nima] = polypar.beta

        # Getting the range of relevant fluxes
        lowlevel_muse, highlevel_muse = self._get_flux_range(musedata)
        lowlevel_ref, highlevel_ref = self._get_flux_range(refdata)
        if self.verbose:
            print("Low / High level MUSE flux: "
                    "{0:8.4f} {1:8.4f}".format(lowlevel_muse, highlevel_muse))
            print("Low / High level REF  flux: "
                    "{0:8.4f} {1:8.4f}".format(lowlevel_ref, highlevel_ref))

        # Smoothing out the result in case it is needed
        if muse_smooth > 0 :
           musedata = nd.filters.gaussian_filter(musedata, muse_smooth)
        if ref_smooth > 0 :
           refdata = nd.filters.gaussian_filter(refdata, ref_smooth)

        # Get the WCS
        musewcs = wcs.WCS(muse_hdu.header)
        refwcs = wcs.WCS(ref_hdu.header)

        # Preparing the figure
        current_fig = start_nfig
        self.list_figures = []

        # Starting the plotting
        if showcontours:
            fig, ax = open_new_wcs_figure(current_fig, musewcs)
            if levels is not None:
                mylevels = levels
                samecontour = True
            else :
                # First contours - MUSE
                levels_muse = np.linspace(np.log10(lowlevel_muse), 
                        np.log10(highlevel_muse), nlevels)
                levels_ref = np.linspace(np.log10(lowlevel_ref), 
                        np.log10(highlevel_ref), nlevels)
                mylevels = levels_muse
            cmuseset = ax.contour(np.log10(musedata), mylevels, colors='k', 
                    origin='lower', linestyles='solid')
            # Second contours - Ref
            if samecontour: 
                crefset = ax.contour(np.log10(refdata), levels=cmuseset.levels, 
                        colors='r', origin='lower', alpha=0.5, linestyles='solid')
            else :
                crefset = ax.contour(np.log10(refdata), levels=levels_ref, 
                        colors='r', origin='lower', alpha=0.5,
                        linestyles='solid')
            ax.set_aspect('equal')
            h1,_ = cmuseset.legend_elements()
            h2,_ = crefset.legend_elements()
            ax.legend([h1[0], h2[0]], ['MUSE', 'REF'])
            if nima is not None:
                plt.title("Image #{0:02d}".format(nima))

            self.list_figures.append(current_fig)
            current_fig += 1

        if showcuts:
            fig, ax = open_new_wcs_figure(current_fig)
            diffima = (refdata - musedata) * 200. / (lowlevel_muse + highlevel_muse)
            chunk_x = musedata.shape[0] // (ncuts + 1)
            chunk_y = musedata.shape[1] // (ncuts + 1)
            ax.plot(diffima[np.arange(ncuts)*chunk_x,:].T, 'k-')
            ax.plot(diffima[:,np.arange(ncuts)*chunk_x], 'r-')
            ax.set_ylim(-20,20)
            ax.set_xlabel("[pixels]", fontsize=20)
            ax.set_ylabel("[%]", fontsize=20)
            self.list_figures.append(current_fig)
            current_fig += 1

        if showdiff:
            fig, ax = open_new_wcs_figure(current_fig, musewcs)
            ratio = 100. * (refdata - musedata) / (musedata + 1.e-12)
            im = ax.imshow(ratio, vmin=-percentage, vmax=percentage)
            cbar = fig.colorbar(im)
            self.list_figures.append(current_fig)
            current_fig += 1


