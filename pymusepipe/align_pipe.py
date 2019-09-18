# Licensed under a MIT license - see LICENSE

"""MUSE-PHANGS alignement module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"

# Import general modules
import os
from os.path import join as joinpath

import glob
import copy

# Import Matplotlib
import matplotlib.pyplot as plt

# Import Numpy Scipy
import numpy as np
import scipy.ndimage as nd
from scipy.signal import correlate
from scipy.odr import ODR, Model, RealData

# Astropy
import astropy.wcs as awcs
from astropy.io import fits as pyfits
from astropy.modeling import models, fitting
from astropy.stats import mad_std
from astropy.table import Table, Column
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve

# Import mpdaf
import mpdaf
from mpdaf.obj import Image, Cube, WCS

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))

# Import needed modules from pymusepipe
import pymusepipe.util_pipe as upipe

# ================== Default units ======================== #
# Define useful units
default_muse_unit = u.erg / (u.cm * u.cm * u.second * u.AA) * 1.e-20
default_reference_unit = u.microJansky

# ================== Default keywords ===================== #
# Define useful keywords for fits table and images
mjd_names = {'table': "MJD_OBS", 'image': "MJD-OBS"}
date_names = {'table': "DATE_OBS", 'image': "DATE-OBS"}

default_offset_table = {'date': [date_names['table'], 'S23', ""],
                        'mjd' : [mjd_names['table'], 'f8', 0.0],
                        'ora':["RA_OFFSET", 'f8', 0.0],
                        'odec':["DEC_OFFSET", 'f8', 0.0],
                        'scale':["FLUX_SCALE", 'f8', 1.0]}

# ================== Useful function ====================== #
def create_offset_table(image_names=[], table_folder="", 
        table_name="dummy_offset_table.fits",
        overwrite=False):
    """Create an offet list table from a set of images. It will use
    the MJD and DATE as read from the descriptors of the images.
    The names for these keywords is stored in the dictionary
    default_offset_table from align_pipe.py.

    Input
    -----
    image_names: list of str
        List of image names to be considered.
    table_folder: str
        Name of the folder where to create the fits table
    table_name: str
        Name of the table to spit out
    overwrite: bool
        If True, will overwrite any existing table

    Creates
    -------
        A fits table with the output given name.
    """

    # Check if table exists and see if overwrite is set up
    table_fullname = joinpath(table_folder, table_name)
    if not overwrite and os.path.isfile(table_fullname):
        upipe.print_warning("[create_offset_table] Table {0} "
                            "already exists".format(table_fullname))
        upipe.print_warning("Use overwrite=True if you wish to proceed")
        return 

    nimages = len(image_names)
    if nimages == 0:
        upipe.print_warning("No image names provided for create_offset_table")
        return

    # Gather the values of DATE and MJD from the images
    date, mjd = [], []
    for ima in image_names:
        if not os.path.isfile(ima):
            upipe.print_warning("[create_offset] Image {0} does not exists".format(ima))
            continue

        head = pyfits.getheader(ima)
        date.append(head[date_names['image']])
        mjd.append(head[mjd_names['image']])

    nlines = len(date)

    # Create and fill the table
    offset_table = Table()
    for col in default_offset_table.keys():
        [name, form, default] = default_offset_table[col]
        offset_table[name] = Column([default for i in range(nlines)], 
                                    dtype=form)

    offset_table[date_names['table']] = date
    offset_table[mjd_names['table']] = mjd

    # Write the table
    offset_table.write(table_fullname, overwrite=overwrite)

def open_new_wcs_figure(nfig, mywcs=None):
    """Open a new figure with wcs projection.
    
    Input
    -----
    mywcs : astropy.wcs.WCS
         Input wcs to open a new figure
         
    Creates
    -------
    figure, subplots
        Returns the figure itself and the subplot with the wcs projection
    """
    fig = plt.figure(nfig)
    plt.clf()
    # Adding axes with WCS
    if mywcs is None:
        return fig, fig.add_subplot(1, 1, 1)
    else:
        return fig, fig.add_subplot(1, 1, 1, projection=mywcs)

def chunk_stats(list_data, chunk_size=15):
    """Cut the datasets in 2d chunks and take the median
    Return the set of medians for all chunks.
    
    Input
    -----
    list_data: list of numpy arrays
        List of arrays with the same sizes/shapes.
    chunk_size: int
        Size of the chunk to be binned (pixels)
    
    Returns
    -------
    median, standard: 2 arrays of the medians and 
        standard deviations for the given datasets analysed
        in chunks.
    
    """
    ndatasets = len(list_data)

    nchunk_x = np.int(list_data[0].shape[0] // chunk_size - 1)
    nchunk_y = np.int(list_data[0].shape[1] // chunk_size - 1)
    # Check that all datasets have the same size
    med_data = np.zeros((ndatasets, nchunk_x * nchunk_y), dtype=np.float64)
    std_data = np.zeros_like(med_data)

    if not all([d.size for d in list_data]):
        upipe.print_error("Datasets are not of the same "
                          "size in median_compare")
    else:
        for i in range(0, nchunk_x):
            for j in range(0, nchunk_y):
                for k in range(ndatasets):
                    # Taking the median
                    med_data[k, i*nchunk_y + j] = np.nanmedian(
                            list_data[k][i*chunk_size:(i+1)*chunk_size, 
                            j*chunk_size:(j+1)*chunk_size])
                    std_data[k, i*nchunk_y + j] = mad_std(
                            list_data[k][i*chunk_size:(i+1)*chunk_size, 
                            j*chunk_size:(j+1)*chunk_size], ignore_nan=True)

    # Cleaning in case of Nan
    med_data = np.nan_to_num(med_data)
    std_data = np.nan_to_num(std_data)
    return med_data, std_data

def my_linear_model(B, x):
    """Linear function for the regression
     
    Input
    -----
    B: 1D array (length is 2)
        Input 1D polynomial parameters (0=constant, 1=slope)
    x: array
        Array which will be multiplied by the polynomial
    
    Returns
    -------
    An array = B[1] * (x + B[0])
    """
    return B[1] * (x + B[0])

def get_image_norm_poly(data1, data2, chunk_size=15, 
        threshold1=0., threshold2=0):
    """Find the normalisation factor between two datasets
    Including the background and slope. This uses the function
    regress_odr which is included in align_pipe.py and itself
    makes use of ODR in scipy.odr.ODR.
     
    Input
    -----
    data1: array
    data2: array
        2 arrays (2D) of identical shapes
    chunk_size: int
        Size of the chunk to bin the images
    threshold1: float
    threshold2: float
        2 floats defining the lower threshold for filtering
    
    Returns
    -------
    result: python structure
        Result of the regression (ODR)
    """
    med, std = chunk_stats([data1, data2], chunk_size=chunk_size)
    pos = (med[0] > threshold1) & (std[0] > 0.) & (std[1] > 0.) & (med[1] > threshold2)
    guess_slope = np.nanmedian(med[1][pos] / med[0][pos])

    result = regress_odr(x=med[0][pos], y=med[1][pos], sx=std[0][pos], 
                         sy=std[1][pos], beta0=[0., guess_slope])
    result.med = med
    result.std = std
    result.selection = pos
    return result

def regress_odr(x, y, sx, sy, beta0=[0., 1.]):
    """Return an ODR linear regression using scipy.odr.ODR
     
    Input
    -----
    x, y: arrays
        Input nD arrays with signal
    sx, sy: arrays
        Input nD arrays (as x,y) with standard deviations
    beta0: list of 2 floats
        Initial guess for the constant and slope
    
    Returns
    -------
    result: result of the ODR analysis
    """
    linear = Model(my_linear_model)
    mydata = RealData(x.ravel(), y.ravel(), sx=sx.ravel(), sy=sy.ravel())
    result = ODR(mydata, linear, beta0=beta0)
    return result.run()

def get_conversion_factor(input_unit, output_unit, 
                          equivalencies=u.spectral_density(6483.58 * u.AA)):
    """ Conversion of units from an input one
    to an output one
     
    Input
    -----
    input_unit: astropy unit
        Input astropy unit to analyse
    output_unit: astropy unit
        Astropy unit to compare to input unit.
    equivalencies: astropy equivalency
        Used in case there is an existing equivalency
        to help the conversion
    
    Returns
    -------
    conversion: astropy unit conversion
    """

    # First testing if the quantities are Quantity
    # If not, transform them 
    if not isinstance(input_unit, u.quantity.Quantity):
        if not isinstance(input_unit, (u.core.Unit, u.core.CompositeUnit)):
            upipe.print_warning("Input provided unit could not be converted")
            upipe.print_warning("Using 1.0 as a conversion factor")
            return 1.0
        else :
            input_unit = input_unit * 1.0
    if not isinstance(output_unit, u.quantity.Quantity):
        if not isinstance(output_unit, (u.core.Unit, u.core.CompositeUnit)):
            upipe.print_warning("Output provided unit could not be converted")
            upipe.print_warning("Using 1.0 as a conversion factor")
            return 1.0
        else:
            output_unit = output_unit * 1.0

    if not input_unit.unit.is_equivalent(output_unit):
        # if not equivalent we try a spectral density equivalence
        if not input_unit.unit.is_equivalent(output_unit, 
                                             equivalencies=equivalencies):
            upipe.print_warning("Provided units for reference "
                                "and MUSE images are not equivalent")
            upipe.print_warning("A conversion factor of 1.0 will thus be used")
            return 1.0
        else :
            return input_unit.unit.to(
                    output_unit, equivalencies=equivalencies) * input_unit.value 
    else :
        return input_unit.unit.to(output_unit) * input_unit.value

def arcsec_to_pixel(hdu, xy_arcsec=[0., 0.]):
    """Transform from arcsec to pixel for the muse image
    using the hdu to extract the WCS, hence the scaling.
     
    Input
    -----
    hdu: astropy hdu (fits)
        Input hdu which includes a WCS
    xy_arcsec: list of 2 floats ([0,0])
        Coordinates to transform from arcsec to pixel.
    
    Returns
    -------
    xpix, ypix: 2 floats
        Pixel coordinates

    See also: pixel_to_arcsec (align_pipe.py)
    """
    # Matrix
    input_wcs = awcs.WCS(hdu.header)
    scale_matrix = np.linalg.inv(input_wcs.pixel_scale_matrix * 3600.)

    # Transformation in Pixels
    dels = np.array(xy_arcsec)
    xpix = np.sum(dels * scale_matrix[0, :])
    ypix = np.sum(dels * scale_matrix[1, :])
    return xpix, ypix

def pixel_to_arcsec(hdu, xy_pixel=[0.,0.]):
    """Transform from arcsec to pixel for the muse image
    using the hdu to extract the WCS, hence the scaling.
     
    Input
    -----
    hdu: astropy hdu (fits)
        Input hdu which includes a WCS
    xy_pixel: list of 2 floats ([0,0])
        Coordinates to transform from pixel to arcsec
    
    Returns
    -------
    xarc, yarc: 2 floats
        Arcseconds coordinates

    See also: arcsec_to_pixel (align_pipe.py)
    """
    # Matrix
    input_wcs = awcs.WCS(hdu.header)

    # Transformation in arcsecond
    dels = np.array(xy_pixel, dtype=np.float64)
    xarc = np.sum(dels * input_wcs.pixel_scale_matrix[0, :] * 3600.)
    yarc = np.sum(dels * input_wcs.pixel_scale_matrix[1, :] * 3600.)
    return xarc, yarc

def crop_data(data, border=10):
    """Crop a 2D data and return it cropped after a border
    has been removed (number of pixels) from each edge
    (borderx2 pixels are removed from each dimension)
    
    Input
    -----
    data: 2d array
        Array which has the signal to be cropped
    border: int
        Number of pixels to be cropped at each edge
    
    Returns
    -------
    cdata: 2d array
        Cropped data array
    """
    if border <= 0 :
        return data

    if data.ndim != 2:
        upipe.print_warning("Input data to crop is not 2, "
                      "returning the original data")
        return data

    if (data.shape[0] > 2 * border) & (data.shape[1] > 2 * border):
        return data[border:-border, border:-border]
    else:
        upipe.print_warning("Data is not being cropped, as shape is {0} "
             " while border is {1}".format(data.shape, border))
        return data

def filtermed_image(data, border=10, filter_size=2):
    """Process image by removing the borders
    and filtering it via a median filter
     
    Input
    -----
    data: 2d array
        Array to be processed
    border: int
        Number of pixels to remove at each edge
    filter_size: float
        Size of the filtering (median)
    
    Returns
    -------
    cdata: 2d array
        Processed array
    """
    # Omit the border pixels
    if border > 0:
        data = crop_data(data)
    meddata = nd.filters.median_filter(data, filter_size)

    return meddata

def prepare_image(data, border=10, dynamic_range=10, 
                  median_window=10, minflux=0.0):
    """Process image by squeezing the range, removing 
    the borders and filtering it. The image is first filtered, 
    then it is cropped. All values below a given minimum are 
    set to 0 and all Nan set to 0 or infinity accordingly.
     
    Input
    -----
    data: 2d array
        Input array to process
    dynamic_range: float [10]
        Dynamic range used to squash the bright pixels down
    median_window: int [10]
        Size of the window used for the median filtering.
    minflux: float [0]
        Value of the minimum flux allowed.
    
    Returns
    -------
    """
    # Squish bright pixels down
    data = np.arctan(data / np.nanmedian(data) / dynamic_range)

    # Omit the border pixels
    data -= filtermed_image(data, 0, median_window)
    cdata = crop_data(data, border)

    # Removing the zeros
    with np.errstate(invalid='ignore'):
        cdata[cdata < minflux] = 0.

    # Clean up the NaNs
    cdata = np.nan_to_num(cdata)

    return cdata

#################################################################
# ================== END Useful functions ===================== #
#################################################################

# Main alignment Class
class AlignMusePointing(object):
    """Class to align MUSE images onto a reference image.
    """
    def __init__(self, name_reference,
                 folder_reference="",
                 folder_muse_images="",
                 name_muse_images=None,
                 sel_indices_images=None,
                 median_window=10,
                 subim_window=10,
                 dynamic_range=10,
                 border=10, hdu_ext=[0,1],
                 chunk_size=15,
                 threshold_muse=0.,
                 **kwargs):
        """Initialise the AlignMuseImages class.
        
        Input
        -----
        name_reference: str
            Name of the reference image (fits file)
        folder_reference: str [""]
            Folder name of the reference image
        folder_muse_images: str [""]
            Folder name for the input images to compare
        name_muse_images: str or list
            List of names for the MUSE images (or str if only 1 image)
        firstguess: str
            If "crosscorr", will use cross-correlation to guess
            the alignment offsets.
            If "fits", will use input fits OFFSET table to start with
        name_offset_table: str
            Name of the fits OFFSET table. Only used as input
            as guess if "firstguess" is set to "fits". This name will be the
            default name when saving the offsets in an OFFSET fits table.
        sel_indices_images: list [None]
            List of images to select from the given set
        median_window: int [10]
            Size of window used in median filter to extract features in
            cross-correlation.  Should be an odd integer
        subim_window: int [10]
            Size of window for fitting peak of cross-correlation function
        dynamic_range: float [10]
            Apply an arctan transform to data to suppress values more than
            DynamicRange times the median of the image
        border: int [10]
            Ignore pixels this close to the border in the cross correlation
        hdu_ext: list of 2 floats [0,1]
            Number of the extension for the input reference image 
            and images to align, respectively. This is used to know
            where the data lies in the fits file.
        chunk_size: int [15]
            Size in pixels of the chunk used to bin and compute the 
            normalisation factors
        threshold_muse: float [0]
            Minimum threshold value to consider for the input 
            images to align

        Other keywords
        --------------
        verbose: bool [True]
            If True, spits out more verbose output
        plot: bool [True]
            If True, will provide plots
        debug: bool [False]
            If True, will provide some info to debug
            which will be stored in the python class
            as new attributes. Look for self._temp...
        use_polynorm: bool [True]
            Save the polynomial fitted slope and use as normalisation
            factors
        convert_units: bool [True]
            Use the given units to convert fluxes
        ref_unit: astropy unit
            Reference image unit
        muse_unit: astropy unit
            Input MUSE flux unit
        minflux_crosscorr: float [0]
            Minimum flux to consider when doing the cross-correlation.
        """

        # Some input variables for the cross-correlation
        self.verbose = kwargs.pop("verbose", True)
        self.plot = kwargs.pop("plot", True)
        self.border = np.int(border)
        self.chunk_size = np.int(chunk_size)
        self.subim_window = subim_window
        self.median_window = median_window
        self.dynamic_range = dynamic_range
        self.name_reference = name_reference
        self.folder_reference = folder_reference

        # Debug option
        self.debug = kwargs.pop("debug", False)

        # Check if folder reference exists
        if not os.path.isdir(self.folder_reference):
            upipe.print_error("Provided folder_reference is "
                        "not an existing folder")
            return

        self.name_muse_images = name_muse_images
        self.sel_indices_images = sel_indices_images
        self.folder_muse_images = folder_muse_images
        # Check if folder muse images exists
        if not os.path.isdir(self.folder_muse_images):
            upipe.print_error("Provided folder_muse_images is "
                        "not an existing folder")
            return

        # Creating the Header-Align folder
        header_folder_name = kwargs.pop("header_folder_name", "AlignHeaders")
        self.header_folder_name = joinpath(self.folder_muse_images, header_folder_name)
        upipe.safely_create_folder(self.header_folder_name, verbose=self.verbose)

        # Getting the names
        self.name_musehdr = kwargs.pop("name_musehdr", "muse")
        self.name_offmusehdr = kwargs.pop("name_offmusehdr", "offsetmuse")
        self.name_refhdr = kwargs.pop("name_refhdr", "reference.hdr")
        self.suffix_images = kwargs.pop("suffix_muse_images", "IMAGE_FOV")
        self.name_filter = kwargs.pop("name_filter", "WFI_ESO844")

        # Use polynorm or not
        self.use_polynorm = kwargs.pop("use_polynorm", True)

        # Getting the unit conversions
        self.convert_units = kwargs.pop("convert_units", True)
        if self.convert_units :
            self.ref_unit = kwargs.pop("ref_unit", default_reference_unit)
            self.muse_unit = kwargs.pop("muse_unit", default_muse_unit)
            self.conversion_factor = get_conversion_factor(self.ref_unit, 
                                                           self.muse_unit)
        else :
            self.conversion_factor = 1.0

        # Initialise the parameters for the first guess
        self.firstguess = kwargs.pop("firstguess", "crosscorr")
        self.name_offset_table = kwargs.pop("name_offset_table", None)
        self.minflux_crosscorr = kwargs.pop("minflux_crosscorr", 0.)

        # Get the MUSE images
        self._get_list_muse_images(**kwargs)
        upipe.print_info("{0} MUSE images detected as input".format(
                            self.nimages))
        if self.nimages == 0:
            upipe.print_error("No MUSE images detected. Aborted")
            return
        self.list_offmuse_hdu = [None] * self.nimages
        self.list_wcs_offmuse_hdu = [None] * self.nimages
        self.list_proj_refhdu = [None] * self.nimages
        self.list_wcs_proj_refhdu = [None] * self.nimages

        # Initialise the needed arrays for the offsets
        self.cross_off_pixel = np.zeros((self.nimages, 2), dtype=np.float64)
        self.extra_off_pixel = np.zeros_like(self.cross_off_pixel)
        self.total_off_pixel = np.zeros_like(self.cross_off_pixel)
        self.cross_off_arcsec = np.zeros_like(self.cross_off_pixel)
        self.extra_off_arcsec = np.zeros_like(self.cross_off_pixel)
        self.total_off_arcsec = np.zeros_like(self.cross_off_pixel)
        self._reset_init_guess_values()

        # Cross normalisation for the images
        # This contains the parameters of the linear fit
        self.ima_polypar = [None] * self.nimages
        # Normalisation factor to be saved or used
        self.ima_norm_factors = np.zeros((self.nimages), dtype=np.float64)
        self.ima_background = np.zeros((self.nimages), dtype=np.float64)
        self.threshold_muse = np.zeros_like(self.ima_norm_factors) + threshold_muse
        # Default lists for date and mjd of the MUSE images
        self.ima_dateobs = [None] * self.nimages
        self.ima_mjdobs = [None] * self.nimages

        # Which extension to be used for the ref and muse images
        self.hdu_ext = hdu_ext

        # Open the Ref and MUSE image
        status_open = self.open_hdu()
        if not status_open:
            upipe.print_error("Problem in opening frames, please check your input")
            return

        # find the cross correlation peaks for each image
        self.find_ncross_peak()

        # Initialise the offset
        self.init_guess_offset(self.firstguess, **kwargs)

        self.total_off_arcsec = self.init_off_arcsec + self.extra_off_arcsec
        self.total_off_pixel = self.init_off_pixel + self.extra_off_pixel

        # Now doing the shifts and projections with the guess/input values
        for nima in range(self.nimages):
            self.shift(nima)

    def show_norm_factors(self):
        """Print some information about the normalisation factors.
        """
        print("Normalisation factors")
        print("Image # : InitFluxScale     SlopeFit       NormFactor")
        for nima in range(self.nimages):
            print("Image {0:02d}:  {1:10.6e}   {2:10.6e}     {3:10.6e}".format(
                    nima,
                    self.init_flux_scale[nima], 
                    self.ima_polypar[nima].beta[1],
                    self.ima_norm_factors[nima]))

    def show_linearfit_values(self):
        """Print some information about the linearly fitted parameters
        pertaining to the normalisation.
        """
        print("Normalisation factors")
        print("Image # : BackGround        Slope")
        for nima in self.nimages:
            print("Image {0:02d}:  {1:10.6e}   {2:10.6e}".format(
                    nima,
                    self.ima_polypar[nima].beta[0], 
                    self.ima_polypar[nima].beta[1]))

    def init_guess_offset(self, firstguess="crosscorr"):
        """Initialise first guess, either from cross-correlation (default)
        or from an Offset FITS Table
         
        Input
        -----
        firstguess: str
            If "crosscorr" uses cross-correlation to get the first guess
            of the offsets. If "fits" uses the input fits table.
        """
        # Implement the guess
        self.firstguess = firstguess
        if firstguess not in ["crosscorr", "fits"]:
            firstguess = "crosscorr"
            upipe.print_warning("Keyword 'firstguess' not recognised")
            upipe.print_warning("Using Cross-Correlation as "
                                "a first guess of the alignment")

        if firstguess == "crosscorr":
            upipe.print_info("Using cross-correlation as the initial guess")
            self.init_off_arcsec = self.cross_off_arcsec * 1.0
            self.init_off_pixel = self.cross_off_pixel * 1.0
        elif firstguess == "fits":
            exist_table, self.offset_table = self.open_offset_table(
                    self.folder_muse_images + self.name_offset_table)
            if exist_table is not True :
                upipe.print_warning("Fits initialisation table not found, "
                                    "setting init value to 0")
                self._reset_init_guess_values()
                return

            upipe.print_info("Using input FITS table as initial guess: {0}".format(
                                 self.name_offset_table))
            # First get the right indices for the table by comparing MJD_OBS
            if mjd_names['table'] not in self.offset_table.columns:
                upipe.print_warning("Input table does not "
                                    "contain MJD_OBS column")
                self._reset_init_guess_values()
                return

            self.table_mjdobs = self.offset_table[mjd_names['table']]
            # Now finding the right match with the Images
            # Warning, needs > numpy 1.15.0
            mjd_values, ind_ima, ind_table = np.intersect1d(
                    self.ima_mjdobs, self.table_mjdobs,
                    return_indices=True, assume_unique=True)
            # Extracting the flux scale from table
            nonan_flux_scale_table = np.where(
                    np.isnan(self.offset_table['FLUX_SCALE']), 
                    1., self.offset_table['FLUX_SCALE'])
            # Extracting the rotangle, but only if there
            rotangle_exist = False
            if ('ROTANGLE' in self.offset_table.columns):
                nonan_rotangle_table = np.where(
                        np.isnan(self.offset_table['ROTANGLE']), 
                        0., self.offset_table['ROTANGLE'])
                rotangle_exist = True

            # Loop over the images, using MJD
            for nima, mjd in enumerate(self.ima_mjdobs):
                # Test if mjd is in the set of mjd_values
                if mjd in mjd_values:
                    # Find the index of the value array where mjd
                    ind = np.nonzero(mjd_values == mjd)[0][0]
                    self.init_off_arcsec[nima] = [
                            self.offset_table['RA_OFFSET'][ind_table[ind]] * 3600. \
                                    * np.cos(np.deg2rad(self.list_dec_muse[nima])),
                            self.offset_table['DEC_OFFSET'][ind_table[ind]] * 3600.]
                    self.init_flux_scale[nima] = nonan_flux_scale_table[ind_table[ind]]
                    if rotangle_exist:
                        self.muse_rotangles[nima] = nonan_rotangle_table[ind_table[ind]]
                    else:
                        self.muse_rotangles[nima] = 0.0
                # Otherwise use default values
                else :
                    self.init_flux_scale[nima] = 1.0
                    self.init_off_arcsec[nima] = [0., 0.]
                    self.muse_rotangles[nima] = 0.0

                # Transform into pixel values
                self.init_off_pixel[nima] = arcsec_to_pixel(
                        self.list_muse_hdu[nima],
                        self.init_off_arcsec[nima])

    def _reset_init_guess_values(self):
        """Reset the initial guess to 0. Hidden function as this is only
        used internally
        """
        self.table_mjdobs = [None] * self.nimages
        self.init_off_pixel = np.zeros((self.nimages, 2), dtype=np.float64)
        self.init_off_arcsec = np.zeros((self.nimages, 2), dtype=np.float64)
        self.init_flux_scale = np.ones(self.nimages, dtype=np.float64)
        self.muse_rotangles = np.zeros(self.nimages, dtype=np.float64)

    def open_offset_table(self, name_table=None):
        """Read offset table from fits file
         
        Input
        -----
        name_table: str
            Name of the input OFFSET table

        Returns
        -------
        status: None if no table name is given, False if file does not
            exist, True if it does
        Table: the result of a astropy.Table.read of the fits table
        """
        if name_table is None:
            if not hasattr(self, "name_table"):
                upipe.print_error("No FITS table name provided, "
                                  "Aborting Open")
                return None, Table()

        if not os.path.isfile(name_table):
            upipe.print_warning("FITS Table ({0}) does not "
                " exist yet".format(name_table))
            return False, Table()

        return True, Table.read(name_table)

    def print_offset_fromfits(self, name_table=None):
        """Print offset table from fits file
         
        Input
        -----
        name_table: str
            Name of the input OFFSET table
        """
        exist_table, fits_table = self.open_offset_table(name_table)
        if exist_table is None:
            return

        if (('RA_OFFSET' not in fits_table.columns) 
                or ('DEC_OFFSET' not in fits_table.columns)):
            upipe.print_error("Table does not contain 'RA/DEC_OFFSET' "
                              "columns, Aborting")
            return

        upipe.print_info("Offset recorded in OFFSET_LIST Table")
        upipe.print_info("Total in ARCSEC")
        for nima in range(self.nimages):
            upipe.print_info("Image {0}: {1:8.4f} {2:8.4f}".format(
                    nima,
                    fits_table['RA_OFFSET'][nima]*3600 \
                            * np.cos(np.deg2rad(self.list_dec_muse[nima])),
                    fits_table['DEC_OFFSET'][nima]*3600.))

    def print_offset(self):
        """Print out the offset from the Alignment class
        """
        upipe.print_info("#---- Offset recorded so far ----#")
        upipe.print_info("Total in ARCSEC")
        for nima in range(self.nimages):
            upipe.print_info("    Image {0:02d}: {1:8.4f} {2:8.4f}".format(
                    nima, self.total_off_arcsec[nima][0],
                    self.total_off_arcsec[nima][1]))
        upipe.print_info("Total in PIXEL")
        for nima in range(self.nimages):
            upipe.print_info("    Image {0:02d}: {1:8.4f} {2:8.4f}".format(
                    nima, self.total_off_pixel[nima][0],
                    self.total_off_pixel[nima][1]))

    def save_fits_offset_table(self, name_output_table=None, 
            overwrite=False, suffix="", save_flux_scale=True,
            save_other_params=True):
        """Save the Offsets into a fits Table
         
        Input
        -----
        name_outpout_table: str
            Name of the output fits table. If None (default) it will
            use the one given in self.name_output_table
        overwrite: bool [False]
            If True, overwrite if the file exists
        suffix: str [""]
            Suffix to be used to add to the input name. This is handy
            to just modify the given fits name with a suffix 
            (e.g., version number).
        save_flux_scale: bool
            If True (default), saving the flux in FLUX_SCALE
            If False, do not save the flux conversion
        save_other_params: bool
            If True (default), saving the background + rotation
            If False, do not save these 2 parameters.
        
        Creates
        -------
        A fits table with the given name (using the suffix if any)
        """
        if name_output_table is None: 
            if self.name_offset_table is None:
                name_output_table = "DUMMY_OFFSET_TABLE.fits"
            else :
                name_output_table = self.name_offset_table
        self.suffix = suffix
        self.name_output_table = name_output_table.replace(".fits", 
                "{0}.fits".format(self.suffix))

        exist_table, fits_table = self.open_offset_table(
                self.name_output_table)
        if exist_table is None:
            upipe.print_error("Save is aborted")
            return

        # Checking if overwrite and exist_table do go together
        if exist_table and not overwrite:
            upipe.print_warning("Table already exists, "
                                "but overwrite is set to False")
            upipe.print_warning("If you wish to overwrite the table {0}, "
                    "please set overwrite to True".format(name_output_table))
            return

        # Check if RA_OFFSET is there
        exist_ra_offset =  ('RA_OFFSET' in fits_table.columns)

        # First save the DATA and MJD references
        fits_table['DATE_OBS'] = self.ima_dateobs
        fits_table['MJD_OBS'] = self.ima_mjdobs

        # Saving the final values
        fits_table['RA_OFFSET'] = self.total_off_arcsec[:,0] / 3600. \
                / np.cos(np.deg2rad(self.list_dec_muse))
        fits_table['DEC_OFFSET'] = self.total_off_arcsec[:,1] / 3600.
        if save_flux_scale:
            fits_table['FLUX_SCALE'] = self.ima_norm_factors
        else:
            fits_table['FLUX_SCALE'] = 1.0
        if save_other_params:
            fits_table['BACKGROUND'] = self.ima_background
            fits_table['ROTANGLE'] = self.muse_rotangles

        # Deal with RA_OFFSET_ORIG if needed
        if exist_ra_offset:
            # if RA_OFFSET exists, then check if the ORIG column is there
            if 'RA_OFFSET_ORIG' not in fits_table.columns:
                fits_table['RA_OFFSET_ORIG'] = fits_table['RA_OFFSET']
                fits_table['DEC_OFFSET_ORIG'] = fits_table['DEC_OFFSET']
                fits_table['FLUX_SCALE_ORIG'] = fits_table['FLUX_SCALE']

        # Finally add the cross-correlation offsets
        fits_table['RA_CROSS_OFFSET'] = self.cross_off_arcsec[:,0] / 3600.  \
                / np.cos(np.deg2rad(self.list_dec_muse))
        fits_table['DEC_CROSS_OFFSET'] = self.cross_off_arcsec[:,1] / 3600.

        # Writing up
        fits_table.write(self.name_output_table, overwrite=overwrite)
        self.name_output_table = name_output_table

    def run(self, nima=0, **kwargs):
        """Run the offset and comparison
         
        Input
        -----
        nima: int
            Index of the image to consider
        extra_pixel: list of 2 floats [0,0]
            Offsets in X and Y in pixels to add to the existing
            guessed offsets
            IMPORTANT NOTE: extra_pixel will be considered first
            (before extra_arcsec).
        extra_arcsec: list of 2 floats [0,0]
            Offsets in X and Y in arcsec to add to the existing
            guessed offsets. Ignored if extra_pixel is given.
        rotation: float [0]
            Angle to rotate the image (in degrees)
        threshold_muse: float [0]
            Threshold to consider when plotting the comparison
        
        Plots (if self.plot is True)
        ----------------------------
        This plots a set of comparison maps:
           * flux comparison (1 to 1)
           * Map of the flux ratio
           * Contours of the two scaled maps
           * Cuts of the division between the 2 maps
        """
        if nima not in range(self.nimages) :
            upipe.print_error("nima not within the range "
                              "allowed by self.nimages ({0})".format(self.nimages))
            return

        # Overwrite the plot option if given
        self.plot = kwargs.pop("plot", self.plot)

        if "extra_pixel" in kwargs:
            extra_pixel = kwargs.pop("extra_pixel", [0., 0.])
            extra_arcsec = pixel_to_arcsec(self.list_muse_hdu[nima], 
                                           extra_pixel)
        else:
            extra_arcsec = kwargs.pop("extra_arcsec", [0., 0.])

        self.muse_rotangles[nima] = kwargs.pop("rotation", 0.0)
        self.threshold_muse[nima] = kwargs.pop("threshold_muse", 0.0)

        # Add the offset from user
        self.shift_arcsecond(extra_arcsec, nima)

        # Compare contours if plot is set to True
        if self.plot:
            self.compare(nima=nima, **kwargs)

    def _get_list_muse_images(self):
        """Extract the name of the muse images
        and build the list
        """
        from pathlib import Path

        if self.name_muse_images is None:
            set_of_paths = glob.glob("{0}{1}*{2}*".format(
                    self.folder_muse_images,
                    self.suffix_images, self.name_filter))
            self.list_muse_images = [Path(muse_path).name 
                                     for muse_path in set_of_paths]
            # Sort alphabetically
            self.list_muse_images.sort()
            # Subselection if sel_indices_images is given
            if self.sel_indices_images is not None:
                if not all([i in np.arange(len(self.list_muse_images)) 
                        for i in self.sel_indices_images]): 
                    upipe.print_warning("Selection list - sel_indices_images "
                                        "- does not match image list")
                    upipe.print_warning("Ignoring that input sel_indices_images")
                else :
                    newlist = [self.list_muse_images[nima] 
                            for nima in self.sel_indices_images]
                    self.list_muse_images = newlist

        # test if 1 or several images
        elif isinstance(self.name_muse_images, str):
            self.list_muse_images = [self.name_muse_images]
        elif isinstance(self.name_muse_images, list):
            self.list_muse_images = self.name_muse_images
        else:
            upipe.print_warning("Name of images is not a string or a list, "
                    "please check input name_muse_images")
            self.list_muse_images = []


        # Number of images to deal with
        self.nimages = len(self.list_muse_images)

    def open_hdu(self):
        """Open the HDU of the MUSE and reference images
        """
        status_ref = self._open_ref_hdu()
        if not status_ref:
            upipe.print_error("Problem in opening Reference frame, please check input")
            return 0

        status_muse = self._open_muse_nhdu()
        if not status_muse:
            upipe.print_error("Problem in opening MUSE frame, please check input")
            return 0

        return 1

    def _open_muse_nhdu(self):
        """Open the MUSE images hdu
        """
        self.list_name_musehdr = ["{0}{1:02d}.hdr".format(
                self.name_musehdr, i+1) for i in range(self.nimages)]
        self.list_name_offmusehdr = ["{0}{1:02d}.hdr".format(
                self.name_offmusehdr, i+1) for i in range(self.nimages)]
        self.list_hdulist_muse = [pyfits.open(
                self.folder_muse_images + self.list_muse_images[i]) 
                for i in range(self.nimages)]
        self.list_muse_hdu = [hdu[self.hdu_ext[1]] 
                              for hdu in self.list_hdulist_muse]
        # CHANGE to mpdaf WCS
        self.list_wcs_muse = [WCS(hdu[1].header) 
                              for hdu in self.list_hdulist_muse]
        self.list_dec_muse = np.array([muse_wcs.get_crval2()
                              for muse_wcs in self.list_wcs_muse])
        # Getting the orientation angles
        self.list_orig_rotangles = [musewcs.get_rot() 
                                    for musewcs in self.list_wcs_muse]

        # Filling in the MJD and DATE OBS keywords for the MUSE images
        # If not there, will be filled with "None"
        for nima, hdu in enumerate(self.list_hdulist_muse):
            if date_names['image'] not in hdu[0].header:
                self.ima_dateobs[nima] = None
            else :
                self.ima_dateobs[nima] = hdu[0].header[date_names['image']]
            if mjd_names['image'] not in hdu[0].header:
                self.ima_mjdobs[nima] = None
            else :
                self.ima_mjdobs[nima] = hdu[0].header[mjd_names['image']]

            if self.list_muse_hdu[nima].data is None:
                return 0

        return 1

    def _open_ref_hdu(self):
        """Open the reference image hdu
        """
        # Open the images
        hdulist_reference = pyfits.open(self.folder_reference 
                                        + self.name_reference)
        self.reference_hdu = hdulist_reference[self.hdu_ext[0]]
        if self.reference_hdu.data is None:
            upipe.print_error("No data found in extension of reference frame")
            upipe.print_error("Check your input, "
                    "or change the extention number in input hdu_ext[0]")
            return 0

        return 1

    def find_ncross_peak(self, list_nima=None, minflux=None):
        """Run the cross correlation peaks on all MUSE images
        Derive the self.cross_off_pixel/arcsec parameters
         
        Input
        -----
        list_nima: list of indices for images to process
            Should be a list. Default is None
            and all images are processed

        minflux: float [None]
            minimum flux to be used in the cross-correlation
            Flux below that value will be set to 0.
        """
        if list_nima is None:
            list_nima = range(self.nimages)

        for nima in list_nima:
            self.cross_off_pixel[nima] = self.find_cross_peak(
                    self.list_muse_hdu[nima],
                    self.list_name_musehdr[nima], 
                    rotation=self.muse_rotangles[nima], 
                    minflux=minflux)
            self.cross_off_arcsec[nima] = pixel_to_arcsec(
                    self.list_muse_hdu[nima],
                    self.cross_off_pixel[nima])

    def find_cross_peak(self, muse_hdu, name_musehdr, rotation=0.0, minflux=None):
        """Aligns the MUSE HDU to a reference HDU
         
        Input
        -----
        muse_hdu: MUSE hdu file
        name_musehdr: name of the muse hdr to save
        rotation: Angle in degrees (0). 
        minflux: minimum flux to be used in the cross-correlation
                Flux below that value will be set to 0.
                Default is 0.
        
        Returns
        -------
        xpix_cross
        ypix_cross: x and y pixel coordinates of the cross-correlation peak
        """
        # Projecting the reference image onto the MUSE field
        tmphdr = muse_hdu.header.totextfile(joinpath(self.header_folder_name,
                                            name_musehdr), overwrite=True)
        proj_ref_hdu = self._project_reference_hdu(muse_hdu, rotation=rotation)

        # Cleaning the images
        if minflux is None:
            minflux = self.minflux_crosscorr

        minflux_ref = minflux / self.conversion_factor
        ima_ref = prepare_image(proj_ref_hdu.data, self.border, 
                                self.dynamic_range,
                                self.median_window,
                                minflux=minflux_ref) * self.conversion_factor
        ima_muse = prepare_image(muse_hdu.data, self.border, 
                self.dynamic_range, self.median_window,
                minflux=minflux)
        if self.debug:
            self._temp_input_origmuse_cc = muse_hdu.data * 1.0
            self._temp_input_origref_cc = proj_ref_hdu.data * 1.0

        # Cross-correlate the images
        ccor = correlate(ima_ref, ima_muse, mode='full', method='auto')
        if self.debug:
            self._temp_ima_muse_tocc = ima_muse * 1.0
            self._temp_ima_ref_tocc = ima_ref * 1.0
            self._temp_cc = ccor * 1.0

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
        # Beware, the sign was changed here and is now ok
        xpix_cross = ccor.shape[1]//2 - params.x_mean
        ypix_cross = ccor.shape[0]//2 - params.y_mean

        return xpix_cross, ypix_cross

    def save_image(self, newfits_name=None, nima=0):
        """Save the newly determined hdu
         
        Input
        -----
        newfits_name: str
            Name of the fits file to be used
        nima: int [0]
            Index of the image to save

        Creates
        -------
        A new fits file
        """
        if hasattr(self, "list_offmuse_hdu"):
            if newfits_name is None:
                newfits_name = self.list_name_museimages[nima].replace(
                        ".fits", "_shift.fits")
            self.list_offmuse_hdu[nima].writeto(newfits_name, overwrite=True)
        else:
            upipe.print_error("There are not yet any new hdu to save")

    def _get_flux_range(self, data, low=10, high=99):
        """Get the range of fluxes within the array
        by looking at percentiles.
         
        Input
        -----
        data: 2d array
            Input array with signal to process
        low, high: two floats (10, 99)
            Percentiles to consider to filter
        
        Returns
        -------
        lperc, hperc: 2 floats
            Low and high percentiles
        """
        # Omit the border pixels
        data = crop_data(data, self.border)

        # Clean up the NaNs
        data = np.nan_to_num(data)
        lperc = np.percentile(data[data > 0.], low)
        hperc = np.percentile(data[data > 0.], high)

        return lperc, hperc

    def _project_reference_hdu(self, muse_hdu=None, rotation=0.0):
        """Project the reference image onto the MUSE field
        Hidden function, as only used internally
         
        Input
        -----
        muse_hdu: HDU [None]
            Input hdu
        rotation: float [0]
            Rotation angle in degrees
        
        Returns
        -------
        hdu_repr: HDU
            Reprojected HDU. None if nothing is provided
        """
        # The mpdaf way to project an image onto an other one
        # WARNING: the reference image will be converted in flux
        if muse_hdu is not None:
            wcs_ref = WCS(hdr=self.reference_hdu.header)
            ima_ref = Image(
                    data=self.reference_hdu.data * self.conversion_factor, 
                    wcs=wcs_ref)

            wcs_muse = WCS(hdr=muse_hdu.header)
            if rotation != 0.:
                wcs_muse.rotate(-rotation)
            ima_muse = Image(data=np.nan_to_num(muse_hdu.data), wcs=wcs_muse)

            ima_ref_proj = ima_ref.align_with_image(ima_muse, flux=True)
            hdu_repr = ima_ref_proj.get_data_hdu()

        else:
            hdu_repr = None
            print("Warning: please provide target HDU to allow reprojection")

        return hdu_repr

    def _add_user_arc_offset(self, extra_arcsec=[0., 0.], nima=0):
        """Add user offset in arcseconds
        and update total_off_pixel and arcsec
         
        Input
        -----
        extra_arcsec: list of 2 floats [0,0]
            Extra offsets (x,y) in arcseconds
        nima: int
            Index of image to consider
        """
        # Transforming the arc into pix
        self.extra_off_arcsec[nima] = extra_arcsec

        # Adding the user offset
        self.total_off_arcsec[nima] = (self.init_off_arcsec[nima]
                             + self.extra_off_arcsec[nima])

        # Transforming into pixels - would be better with setter
        self.extra_off_pixel[nima] = arcsec_to_pixel(self.list_muse_hdu[nima],
                self.extra_off_arcsec[nima])
        self.total_off_pixel[nima] = arcsec_to_pixel(self.list_muse_hdu[nima],
                self.total_off_arcsec[nima])

    def shift_arcsecond(self, extra_arcsec=[0., 0.], nima=0):
        """Shift image with index nima with the total offset
        after adding any extra given offset
        This does not return anything but could in principle
        if using the output of the self.shift
         
        Input
        -----
        extra_arcsec: list of 2 floats [0,0]
            Extra offsets (x,y) in arcseconds
        nima: int
            Index of image to consider
        """
        self._add_user_arc_offset(extra_arcsec, nima)
        self.shift(nima)

    def shift(self, nima=0):
        """Create New HDU after shifting it with the right offset
        (only considering image with index nima)
         
        Input
        -----
        nima: int
            Index of image to consider
        
        Does not return anything, but could in principle
        """
        # Create a new Header
        newhdr = copy.deepcopy(self.list_muse_hdu[nima].header)

        # Shift the HDU in X and Y
        if self.verbose:
            print("Image {0:03d} - Shifting CRPIX1 by {1:8.4f} pixels "
                  "/ {2:8.4f} arcsec".format(nima, 
                      self.total_off_pixel[nima][0], 
                      self.total_off_arcsec[nima][0]))
        newhdr['CRPIX1'] = newhdr['CRPIX1'] + self.total_off_pixel[nima][0]
        if self.verbose:
            print("Image {0:03d} - Shifting CRPIX2 by {1:8.4f} pixels "
                  "/ {2:8.4f} arcsec".format(nima, 
                      self.total_off_pixel[nima][1], 
                      self.total_off_arcsec[nima][1]))
        newhdr['CRPIX2'] = newhdr['CRPIX2'] + self.total_off_pixel[nima][1]

        # Creating a new Primary HDU with the input data, and the new Header
        self.list_offmuse_hdu[nima] = pyfits.PrimaryHDU(
                self.list_muse_hdu[nima].data, header=newhdr)
        # Now reading the WCS of that new HDU and saving it in the list
        self.list_wcs_offmuse_hdu[nima] = WCS(
                self.list_offmuse_hdu[nima].header)

        # Writing this up in an ascii file for record purposes
        tmphdr = self.list_offmuse_hdu[nima].header.totextfile(
                self.header_folder_name + self.list_name_offmusehdr[nima], 
                overwrite=True)

        # Reprojecting the Reference image onto the new MUSE frame
        self.list_proj_refhdu[nima] = self._project_reference_hdu(
                muse_hdu=self.list_offmuse_hdu[nima], 
                rotation=self.muse_rotangles[nima])
        # Now reading the WCS and saving it in the list
        self.list_wcs_proj_refhdu[nima] = WCS(
                self.list_proj_refhdu[nima].header)

        # Getting the normalisation factors again
        musedata, refdata = self.get_image_normfactor(nima)

    def get_image_normfactor(self, nima=0, median_filter=True, 
            convolve_muse=0., convolve_reference=0.,
            threshold_muse=None):
        """Get the normalisation factor for image nima
         
        Input
        -----
        nima: int
            Index of image to consider
        median_filter: bool
            If True, will median filter
        convolve_muse: float [0]
            Will convolve the image with index nima
            with a gaussian with that sigma. 0 means no convolution
        convolve_reference: float [0]
            Will convolve the reference image
            with a gaussian with that sigma. 0 means no convolution
        threshold_muse: float [None]
            Threshold for the input image flux to consider
        
        Returns
        -------
        data: 2d array
        refdata: 2d array
            The 2 arrays (input, reference) after processing
        """
        # If median filter do the filtermed_image process including the border
        # Both for the muse data and the reference data
        if median_filter:
            musedata = filtermed_image(self.list_offmuse_hdu[nima].data, 
                                       self.border)
            refdata = filtermed_image(
                    self.list_proj_refhdu[nima].data,
                    self.border)
        # Otherwise just copy the data
        else:
            musedata = copy.copy(self.list_offmuse_hdu[nima].data)
            refdata = self.list_proj_refhdu[nima].data

        # Smoothing out the result in case it is needed
        if convolve_muse > 0 :
            kernel = Gaussian2DKernel(x_stddev=convolve_muse)
            musedata = convolve(musedata, kernel)
        if convolve_reference > 0 :
            kernel = Gaussian2DKernel(x_stddev=convolve_reference)
            refdata = convolve(refdata, kernel)

        # Getting the result of the normalisation
        if threshold_muse is not None:
            self.threshold_muse[nima] = threshold_muse

        self.ima_polypar[nima] = get_image_norm_poly(musedata, 
                        refdata, chunk_size=self.chunk_size,
                        threshold1=self.threshold_muse[nima])
        if self.use_polynorm:
            self.ima_norm_factors[nima] = self.ima_polypar[nima].beta[1]
            self.ima_background[nima] = self.ima_polypar[nima].beta[0]
        return musedata, refdata

    def compare(self, start_nfig=1, nlevels=10, levels=None, convolve_muse=0.,
            convolve_reference=0., samecontour=True, nima=0,
            showcontours=True, showcuts=True, 
            shownormalise=True, showdiff=True,
            normalise=True, median_filter=True, 
            ncuts=5, percentage=5.,
            rotation=0.0,
            threshold_muse=None):
        """Compare the projected reference and MUSE image
        by plotting the contours, the difference and vertical/horizontal cuts.
         
        Input
        -----
        nima: int
            Index of image to consider
        showcontours: bool [True]
        showcuts: bool [True]
        shownormalise: bool [True]
        showdiff: bool [True]
            All options corresponding to 1 specific plot. By default
            show them all (all True)
        ncuts: int [5]
            Number of vertical / horizontal cuts along the ratio
            between the 2 maps to be shown ("cuts")
        percentage: float [5]
            Used to compute which percentile to show
        rotation: float [0]
            Adding a rotation (in degrees) in case it is needed
        start_nfig: int [1]
            Number of the matplotlib Figure to start with
        nlevels: int [10]
            Number of levels for the contour plots
        levels: list of float [None]
            Specific list of levels if any (default is None)
        convolve_muse: float [0]
            If not 0, will convolve with a gaussian of that sigma
        convolve_reference: float [0]
            If not 0, will convolve the reference image
            with a gaussian of that sigma
        samecontour: bool [True]
            If True, will use the same levels for both images
            (this is recommended). Otherwise levels can be
            automatically derived from percentiles, but will
            not necessarily be the same (which can bring
            confusion but may sometimes be useful).
        
        Makes a maximum of 4 figures
        """
        # Getting the data
        musedata, refdata = self.get_image_normfactor(nima=nima, 
                median_filter=median_filter, 
                convolve_muse=convolve_muse,
                convolve_reference=convolve_reference,
                threshold_muse=threshold_muse)

        # If normalising, using the median ratio fit
        if normalise or shownormalise :
            polypar = self.ima_polypar[nima]

        # If normalising, use the polypar slope and background
        if normalise :
            if self.verbose:
                upipe.print_info("Renormalising the MUSE data as NewMUSE = "
                        "{0:8.4e} * ({1:8.4e} + MUSE)".format(polypar.beta[1], 
                         polypar.beta[0]))

            musedata = (polypar.beta[0] + musedata) * polypar.beta[1]

        # Getting the range of relevant fluxes
        lowlevel_muse, highlevel_muse = self._get_flux_range(musedata)
        lowlevel_ref, highlevel_ref = self._get_flux_range(refdata)
        if self.verbose:
            print("Low / High level MUSE flux: "
                    "{0:8.4e} {1:8.4e}".format(lowlevel_muse, highlevel_muse))
            print("Low / High level REF  flux: "
                    "{0:8.4e} {1:8.4e}".format(lowlevel_ref, highlevel_ref))

        # Save the frames if in debug mode
        if self.debug:
            self._temp_refdata = refdata
            self._temp_musedata = musedata

        # Get the WCS from mpdaf to allow rotation if needed
        refwcs = self.list_wcs_proj_refhdu[nima]

        # WCS for plotting using astropy
        plotwcs = awcs.WCS(self.list_offmuse_hdu[nima].header)

        # Apply rotation in degrees
        # Apply it to the rerence image not to couple it with the offset
        if rotation != 0:
            if self.verbose:
                upipe.print_warning("Apply a rotation of "
                                    "{0} degrees".format(rotation))
            refwcs.rotate(rotation)

        # Preparing the figure
        current_fig = start_nfig
        self.list_figures = []

        # Starting the plotting
        if shownormalise:
            #plotting the normalization
            fig, ax = open_new_wcs_figure(current_fig)
            (x, y) = (polypar.med[0][polypar.selection], 
                      polypar.med[1][polypar.selection])
            ax.plot(x, y, '.')
            ax.set_xlabel("MuseData")
            ax.set_ylabel("RefData")
            ax.plot(x, my_linear_model(polypar.beta, x), 'k')
            # ax.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], 'r')

            self.list_figures.append(current_fig)
            current_fig += 1
            
        if showcontours:
            np.seterr(divide = 'ignore', invalid='ignore') 
            fig, ax = open_new_wcs_figure(current_fig, plotwcs)
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
                samecontour = False
            cmuseset = ax.contour(np.log10(musedata), mylevels, colors='k',
                    origin='lower', linestyles='solid')
            # Second contours - Ref
            if samecontour:
                crefset = ax.contour(np.log10(refdata), 
                                     levels=cmuseset.levels, 
                                     colors='r', origin='lower', 
                                     alpha=0.5, linestyles='solid')
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
            np.seterr(divide = 'warn', invalid='warn') 

        if showcuts:
            fig, ax = open_new_wcs_figure(current_fig)
            diffima = (refdata - musedata) * 200. / (lowlevel_muse 
                      + highlevel_muse)
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
            fig, ax = open_new_wcs_figure(current_fig, plotwcs)
            ratio = 100. * (refdata - musedata) / (musedata + 1.e-12)
            im = ax.imshow(ratio, vmin=-percentage, vmax=percentage)
            cbar = fig.colorbar(im)
            self.list_figures.append(current_fig)
            current_fig += 1
