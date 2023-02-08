"""Utility functions for images in pymusepipe
"""

__authors__ = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__ = "MIT License"
__contact__ = " <eric.emsellem@eso.org>"

# Importing modules
import os
from os.path import join as joinpath
import glob

# Numpy
import numpy as np
from scipy.odr import ODR, Model, RealData
from scipy import ndimage as nd

from astropy.io import fits as pyfits
from astropy import units as u
from astropy.wcs import WCS
from astropy.table import QTable, Column

from astropy.stats import mad_std, sigma_clip, sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import concatenate as concat_skycoords
from astropy.coordinates import SkyCoord

# Import package modules
from . import util_pipe as upipe
from .util_pipe import get_dataset_tpl_nexpo
from .config_pipe import (default_ndigits, default_str_dataset, default_offset_table)
from .config_pipe import mjd_names, date_names, tpl_names, iexpo_names, dataset_names
from .mpdaf_pipe import get_centre_from_pixtable

try:
    from photutils.detection import IRAFStarFinder
    _photutils = True
except ImportError:
    upipe.print_warning("The python packages photutils is not available. "
                        "If you wish to use star masking. Please install it.")
    _photutils = False


def select_spaxels(maskdict, maskname, x, y):
    """Selecting spaxels defined by their coordinates
    using the masks defined by Circle or Rectangle Zones
    """
    # All spaxels are set to GOOD (True) first
    selgood = (x**2 >= 0)

    # If no Mask is provided, we just return the full set of input X, Y
    if maskdict is None:
        return selgood

    # We first check if the maskName is in the list of the defined Masks
    # If the galaxy is not in the list, then the selection is all True
    if maskname in maskdict:
        # The mask is defined, so Get the list of Regions
        # From the defined dictionary
        listregions = maskdict[maskname]
        # For each region, select the good spaxels
        for region in listregions:
            selgood = selgood & region.select(x, y)

    return selgood


class Selection_Zone(object):
    """
    Parent class for Rectangle_Zone and Circle_Zone

    Input
    -----
    params: list of floats
        List of parameters for the selection zone
    """
    def __init__(self, params=None):
        self.params = params
        if self.params is None:
            upipe.print_error("Warning: no parameters given for Selection Zone")


class Rectangle_Zone(Selection_Zone):
    """Define a rectangular zone, given by
    a center, a length, a width and an angle
    """
    def __init__(self):
        self.geometry = "Rectangle"
        self.nparams = 5
        Selection_Zone.__init__(self)

    def select(self, xin, yin):
        """ Define a selection within a rectangle
            It can be rotated by an angle theta (in degrees)
        Input
        -----
        xin, yin: 2d arrays
            Input positions for the spaxels
        """
        if self.params is None:
            return xin**2 >= 0
        [x0, y0, length, width, theta] = self.params
        dx = xin - x0
        dy = yin - y0
        thetarad = np.deg2rad(theta)
        nx = dx * np.cos(thetarad) + dy * np.sin(thetarad)
        ny = -dx * np.sin(thetarad) + dy * np.cos(thetarad)
        selgood = (np.abs(ny) > width / 2.) | (np.abs(nx) > length / 2.)
        return selgood


class Circle_Zone(Selection_Zone):
    """Define a Circular zone, defined by
    a center and a radius
    """
    def __init__(self):
        self.geometry = "Circle"
        self.nparams = 5
        Selection_Zone.__init__(self)

    def select(self, xin, yin):
        """ Define a selection within a circle

        Input
        -----
        xin, yin: 2d arrays
            Input positions for the spaxels
        """
        if self.params is None:
            return xin**2 >= 0
        [x0, y0, radius] = self.params
        selgood = (np.sqrt((xin - x0)**2 + (yin - y0)**2) > radius)
        return selgood


class Trail_Zone(Selection_Zone):
    """Define a Trail zone, defined by
    two points and a width
    """
    def __init__(self):
        self.geometry = "Trail"
        self.nparams = 5
        Selection_Zone.__init__(self)

    def select(self, xin, yin):
        """ Define a selection within trail

        Input
        -----
        xin, yin: 2d arrays
            Input positions for the spaxels

        """
        if self.params is None:
            return xin**2 >= 0
        [x0, y0, radius] = self.params
        selgood = (np.sqrt((xin - x0)**2 + (yin - y0)**2) > radius)
        return selgood


def my_linear_model(b, x):
    """Linear function for the regression.

    Input
    -----
    b : 1D np.array of 2 floats
        Input 1D polynomial parameters (0=constant, 1=slope)
    x : np.array
        Array which will be multiplied by the polynomial

    Returns
    -------
        An array = b[1] * (x + b[0])
    """
    return b[1] * (x + b[0])


def get_polynorm(array1, array2, chunk_size=15, threshold1=0.,
                 threshold2=0, percentiles=(0., 100.), sigclip=0):
    """Find the normalisation factor between two arrays.

    Including the background and slope. This uses the function
    regress_odr which is included in align_pipe.py and itself
    makes use of ODR in scipy.odr.ODR.

    Parameters
    ----------
    array1 : 2D np.array
    array2 : 2D np.array
        2 arrays (2D) of identical shapes
    chunk_size : int
        Default value = 15
    threshold1 : float
        Lower threshold for array1 (Default value = 0.)
    threshold2 : float
        Lower threshold for array2 (Default value = 0)
    percentiles : list of 2 floats
        Percentiles (Default value = [0., 100.])
    sigclip : float
        Sigma clipping factor (Default value = 0)

    Returns
    -------
    result: python structure
        Result of the regression (ODR)
    """

    # proceeds by splitting the data arrays in chunks of chunk_size
    med, std = chunk_stats([array1, array2], chunk_size=chunk_size)

    # Selecting where data is supposed to be good
    if threshold1 is None:
        threshold1 = 0.
    if threshold2 is None:
        threshold2 = 0.
    pos = (med[0] > threshold1) & (std[0] > 0.) & (std[1] > 0.) & (med[1] > threshold2)
    # Guess the slope from this selection
    guess_slope = 1.0

    # Doing the regression itself
    result = regress_odr(x=med[0][pos], y=med[1][pos], sx=std[0][pos],
                         sy=std[1][pos], beta0=[0., guess_slope],
                         percentiles=percentiles, sigclip=sigclip)
    result.med = med
    result.std = std
    result.selection = pos
    return result


def regress_odr(x, y, sx, sy, beta0=(0., 1.),
                percentiles=(0., 100.), sigclip=0.0):
    """Return an ODR linear regression using scipy.odr.ODR

    Args:
        x : numpy.array
        y : numpy.array
            Input array with signal
        sx : numpy.array
        sy : numpy.array
            Input array (as x,y) with standard deviations
        beta0 : list or tuple of 2 floats
            Initial guess for the constant and slope
        percentiles: tuple or list of 2 floats
            Two numbers providing the min and max percentiles
        sigclip: float
            sigma factor for sigma clipping. If 0, no sigma clipping
            is performed

    Returns:
        result: result of the ODR analysis

    """
    # Percentiles
    xrav = x.ravel()
    if len(xrav) > 0:
        percentiles = np.percentile(xrav, percentiles)
        sel = (xrav >= percentiles[0]) & (xrav <= percentiles[1])
    else:
        sel = np.abs(xrav) > 0

    xsel, ysel = xrav[sel], y.ravel()[sel]
    sxsel, sysel = sx.ravel()[sel], sy.ravel()[sel]
    linear = Model(my_linear_model)

    # We introduce the minimum of x to avoid negative values
    minx = np.min(xsel)
    mydata = RealData(xsel - minx, ysel, sx=sxsel, sy=sysel)
    result = ODR(mydata, linear, beta0=beta0)

    if sigclip > 0:
        diff = ysel - my_linear_model([result.beta[0], result.beta[1]], xsel)
        filtered = sigma_clip(diff, sigma=sigclip)
        xnsel, ynsel = xsel[~filtered.mask], ysel[~filtered.mask]
        sxnsel, synsel = sxsel[~filtered.mask], sysel[~filtered.mask]
        clipdata = RealData(xnsel, ynsel, sx=sxnsel, sy=synsel)
        result = ODR(clipdata, linear, beta0=beta0)

    # Running the ODR
    r = result.run()
    # Offset from the min of x
    r.beta[0] -= minx

    return r


def chunk_stats(list_arrays, chunk_size=15):
    """Cut the datasets in 2d chunks and take the median
    Return the set of medians for all chunks.

    Parameters
    ----------
    list_arrays : list of np.arrays
        List of arrays with the same sizes/shapes
    chunk_size : int
        number of pixel (one D of a 2D chunk)
        of the chunk to consider (Default value = 15)

    Returns
    -------
    median, standard: 2 arrays of the medians and standard deviations
        for the given datasets analysed in chunks.

    """

    narrays = len(list_arrays)

    nchunk_x = int(list_arrays[0].shape[0] // chunk_size - 1)
    nchunk_y = int(list_arrays[0].shape[1] // chunk_size - 1)
    # Check that all arrays have the same size
    med_array = np.zeros((narrays, nchunk_x * nchunk_y), dtype=np.float64)
    std_array = np.zeros_like(med_array)

    if not all([d.size for d in list_arrays]):
        upipe.print_error("Datasets are not of the same "
                          "size in median_compare")
    else:
        for i in range(0, nchunk_x):
            for j in range(0, nchunk_y):
                for k in range(narrays):
                    # Taking the median of all arrays
                    med_array[k, i * nchunk_y + j] = np.nanmedian(
                        list_arrays[k][i * chunk_size:(i + 1) * chunk_size,
                        j * chunk_size:(j + 1) * chunk_size])
                    # Taking the std deviation of all arrays
                    std_array[k, i * nchunk_y + j] = mad_std(
                        list_arrays[k][i * chunk_size:(i + 1) * chunk_size,
                        j * chunk_size:(j + 1) * chunk_size], ignore_nan=True)

    # Cleaning in case of Nan
    med_array = np.nan_to_num(med_array)
    std_array = np.nan_to_num(std_array)
    return med_array, std_array


def get_flux_range(data, border=15, low=2, high=98):
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
    data = crop_data(data, border)

    # Clean up the NaNs
    data = np.nan_to_num(data)
    if data.size > 0:
        lperc = np.percentile(data[data > 0.], low)
        hperc = np.percentile(data[data > 0.], high)
    else:
        lperc, hperc = 0., 1.

    return lperc, hperc


def get_normfactor(array1, array2, median_filter=True, border=0,
                   convolve_data1=0., convolve_data2=0., chunk_size=10,
                   threshold=0.):
    """Get the normalisation factor for shifted and projected images. This function
    only consider the input images given by their data (numpy) arrays.

    Input
    -----
    array1: 2d np.array
    array2: 2d np.array
        Input arrays. Should be the same size
    median_filter: bool
        If True, will median filter
    convolve_muse: float [0]
        Will convolve the image with index nima
        with a gaussian with that sigma. 0 means no convolution
    convolve_reference: float [0]
        Will convolve the reference image
        with a gaussian with that sigma. 0 means no convolution
    border: int
        Number of pixels to crop
    threshold: float [None]
        Threshold for the input image flux to consider

    Returns
    -------
    data: 2d array
    refdata: 2d array
        The 2 arrays (input, reference) after processing
    polypar: the result of an ODR regression
    """
    # Retrieving the data and preparing it
    d1 = prepare_image(array1, median_filter=median_filter, sigma=convolve_data1, border=border)
    d2 = prepare_image(array2, median_filter=median_filter, sigma=convolve_data2, border=border)
    polypar = get_polynorm(d1, d2, chunk_size=chunk_size, threshold1=threshold)

    # Returning the processed data
    return d1, d2, polypar


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
    if border <= 0:
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


def filtermed_image(data, border=0, filter_size=2, keepnan=False):
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
        data = crop_data(data, border=border)

    # Masking the Nan with 0
    mynan = np.isnan(data)
    meddata = nd.filters.median_filter(np.nan_to_num(data), filter_size)
    # Putting the Nan back if needed
    if keepnan:
        meddata[mynan] = np.nan

    return meddata


def prepare_image(data, median_filter=True, sigma=0., border=0):
    """Median filter plus convolve the input image

    Input
    -----
    data: 2D np.array
        Data to process
    median_filter: bool
        If True, will median filter
    convolve float [0]
        Will convolve the data with this gaussian width (sigma)
        0 means no convolution

    Returns
    -------
    data: 2d array
    """
    # If median filter do the filtermed_image process including the border
    # No cropping here
    if median_filter:
        data = filtermed_image(data)

    # Smoothing out the result in case it is needed
    if sigma > 0:
        kernel = Gaussian2DKernel(x_stddev=sigma)
        data = convolve(data, kernel)

    # Cropping the data
    data = crop_data(data, border)

    # Returning the processed data
    return data


def flatclean_image(data, border=10, dynamic_range=10,
                  median_window=10, threshold=0.0, squeeze=True,
                  remove_bkg=True):
    """Process image by squeezing the range, removing
    the borders and filtering it. The image is first filtered,
    then it is cropped. All values below a given minimum are
    set to 0 and all Nan set to 0 or infinity accordingly.

    Input
    -----
    data: 2d ndarray
        Input array to process
    dynamic_range: float [10]
        Dynamic range used to squash the bright pixels down
    median_window: int [10]
        Size of the window used for the median filtering.
    threshold: float [0]
        Value of the minimum value allowed.
    squeeze: bool
        Squeeze the dynamic range by using the dynamic_range variable
    crop: bool
        Crop the borders using border as the variable
    remove_bkg: remove the filter_medianed background

    Returns
    -------
    flatcleaned_array: 2d ndarray
    """
    if squeeze:
        # Squish bright pixels down
        data = np.arctan(data / np.nanmedian(data) / dynamic_range)

    if remove_bkg:
        # Remove a median filtered value from the data
        data -= filtermed_image(data, 0, median_window)

    # Crop the border
    cdata = crop_data(data, border)

    # Removing all values below threshold
    with np.errstate(invalid='ignore'):
        if threshold is None:
            threshold = 0.
        cdata = np.clip(cdata, threshold, None)

    # Clean up the NaNs
    cdata = np.nan_to_num(cdata)

    return cdata


def mask_point_sources(ima, fwhm=3, mask_radius=30., brightest=5, sigma=3., verbose=False):
    """Find and mask point sources in an image by adding NaN

    Input
    -----
    ima: ndarray
        Image to mask
    fwhm: float
        guess for the FWHM in pixel of the PSF. Defaults to 3.
    mask_radius: float
        Radius in pixels to mask around sources
    brightest: int
        Maximum number of bright stars to mask. Defaults to 5.
    sigma: float
        Sigma to clip the image
    verbose: bool

    Returns
    -------
    ima: np.array
        with NaN where the mask applied
    """

    if not _photutils:
        print("Warning: photutils is not available - no option to detect and mask stars")
        return ima

    # Define a threshold to look for stars
    mean, median, std = sigma_clipped_stats(ima, sigma=sigma)
    thresh = mean + 10. * std

    # Initializing and starting the starfinder
    starfinder = IRAFStarFinder(threshold=thresh, fwhm=fwhm, brightest=brightest)
    sources = starfinder(ima)

    yy, xx = np.ogrid[:ima.shape[0], : ima.shape[1]]

    mima = ima.copy()

    # iterate over the identified sources and apply circular masks over them.
    if sources is not None:
        if verbose:
            print(f'{len(sources)} detected')

        for source in sources:
            dist_from_source = np.sqrt((xx - source['xcentroid'])**2 +
                                       (yy - source['ycentroid'])**2)
            mask = dist_from_source < mask_radius
            mima[mask] = np.nan

    return mima


def create_offset_table(image_names, table_folder="", table_name="dummy_offset_table.fits",
                        overwrite=False):
    """Create an offset list table from a given set of images. It will use
    the MJD and DATE as read from the descriptors of the images. The names for
    these keywords is stored in the dictionary default_offset_table from
    config_pipe.py

    Parameters
    ----------
    image_names : list of str
        List of image names to be considered. (Default value = [])
    table_folder : str
        folder of the table (Default value = "")
    table_name : str
        name of the table to save ['dummy_offset_table.fits']
        (Default value = "dummy_offset_table.fits")
    overwrite : bool
        if the table exists, it will be overwritten if set
        to True only. (Default value = False)
    overwrite : bool
        if the table exists, it will be overwritten if set
        to True only. (Default value = False)

    Returns
    -------
        A fits table with the output given name. (Default value = False)
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
    date, mjd, tpls, iexpo, dataset = [], [], [], [], []
    for ima in image_names:
        if not os.path.isfile(ima):
            upipe.print_warning("[create_offset] Image {0} does not exists".format(ima))
            continue

        head = pyfits.getheader(ima)
        date.append(head[date_names['image']])
        mjd.append(head[mjd_names['image']])
        tpls.append(head[tpl_names['image']])
        iexpo.append(head[iexpo_names['image']])
        dataset.append(head[dataset_names['image']])

    nlines = len(date)

    # Create and fill the table
    offset_table = QTable()
    for col in default_offset_table:
        [name, form, default] = default_offset_table[col]
        offset_table[name] = Column([default for i in range(nlines)],
                                    dtype=form)

    offset_table[date_names['table']] = date
    offset_table[mjd_names['table']] = mjd
    offset_table[tpl_names['table']] = tpls
    offset_table[iexpo_names['table']] = iexpo
    offset_table[dataset_names['table']] = dataset

    # Write the table
    offset_table.write(table_fullname, overwrite=overwrite)


def group_xy_per_fieldofview(center_dict, limit=10*u.arcsec):
    """Group exposures in pointings based on their proximity.

    Input
    -----
    center_dict: dict
        Dictionary containing a list of filenames and their coordinates.
    limit: Quantity default=10*.u.arcsec, optional
        maximum separation for files to belong to the same pointing. Defaults to 10*u.arcsec.

    Returns:
    Dict: [int, list]
        Dictionary grouping the input files by pointing. The keys of the dictionary are the
        pointing numbers, and to each pointing a list of filenames is associated.
    """

    # transform from lists to array, better masking
    # and concatenate single coordinates in a single SkyCoord object
    allnames = np.array(list(center_dict.keys()))
    allcoords = concat_skycoords(list(center_dict.values()))

    # empty list. Here I will put the lists of files belonging to pointings
    pointing_lists = []
    files_pointing_dict = {}

    for coord in allcoords:
        # measuring the distance between the center of this file and all the files
        distance = coord.separation(allcoords)

        # finding all the files closer than the limit
        mask = distance < limit

        # saving these files. Transforming each list in tuple to allow set to work later
        selnames = allnames[mask]
        selnames.sort()
        pointing_lists.append(tuple(selnames))

    # Remove duplicates if any and sort
    pointing_lists = list(set(pointing_lists))
    pointing_lists.sort()

    # generating output dictionary
    pointing_dict = {}
    for i, list_files in enumerate(pointing_lists):
        pointing = i+1
        pointing_dict[pointing] = list(list_files)  # requirements said list
        # Also a reverse dictionary to have a direct access to the pointing associated with a file
        for f in list_files:
            files_pointing_dict[f] = pointing

    return pointing_dict, files_pointing_dict


def compute_diagnostics(pointing_dict, center_dict):
    """
    Compute the average and std of the distance between the exposures belonging
    to the same pointing.

    Input
    -----
    pointing_dict: dict
        Dictionary for the pointings
    center_dict: dict
        dictionary of the files to be used

    Returns
    -------
    Diagnostic: dict
        Each pointing key has its [mean, std] as value of the distionary
    """

    # Creating the containers
    diags = {}

    for i, pointing in pointing_dict.items():
        # selecting only the interesting files
        selected_coords = []
        for name in pointing:
            selected_coords.append(center_dict[name])
        # putting all the other coords in the same SkyCoord object
        if len(selected_coords) > 1:
            selected_coords = concat_skycoords(selected_coords)
        else:
            selected_coords = selected_coords[0]

        ra_cen = selected_coords.ra.mean()
        dec_cen = selected_coords.dec.mean()

        ref = SkyCoord(ra_cen, dec_cen, unit=(u.deg, u.deg))

        distance = ref.separation(selected_coords)
        mean = np.round(distance.mean().to(u.arcsec).value, 3)
        std = np.round(distance.std().to(u.arcsec).value, 3)
        diags[i] = [mean, std]

    return diags


def group_exposures_per_pointing(list_files, target_path='', limit=10., unit=u.arcsec, ext=1,
                                 dtype='image'):
    """Separate a list of files in pointings based on their proximity.

    This function assign each file to a pointing. Pointings are defined as groups of exposures
    whose distance between the centers falls within a certain limit. Once the groups of exposures
    have been defined, they are sorted, and then a pointing number starting from 1 is assigned to
    all of them. Some info on the average std of the eparation between exposures can be optionally
    computed.

    Input
    ------
    list_files: list
        list of files to be reorganized in pointings
    target_path: str default=''
        path of the target files
    limit: float default=10
        maximum separation for files to belong to the same pointing
    unit: astropy unit default=u.arcsec
        Unit of spatial distance (e.g., astropy.unit.arcsec)
    ext: int default=1, optional
        header extension where the WCS information is located.
    dtype: str default 'image', optional
        type of file to be analyzed. It can be pixtable, image or cube. Defaults to image.

    Returns
    -------
    Pointing dictionary: Dict of [int, list]
        Dictionary grouping the input files by pointing. The keys of the dictionary are the
        pointing numbers, and to each pointing a list of filenames is associated.
    Diagnostic dictionary: Dict of [int, list], only if diagnostics
        Dictionary containing basic information on the distance between exposures belonging to
        the same pointing. For each pointing, the mean and std of the distance with respect to
        a reference exposure is reported.
    """

    # open all the files and recover the RA and dec informations
    center_dict = {}

    dict_dtype = ['pixtable', 'image', 'cube', 'guess']
    if dtype not in dict_dtype:
        print(f'{dtype} is not a supported data type.')
        return None, None

    for name in list_files :
        # open file and recover the primary header
        fullname = joinpath(target_path, name)
        if dtype == "guess":
            ldtype = [dtype for dtype in dict_dtype if dtype in name]
            if len(ldtype) == 0:
                upipe.print_warning(f"Could not guess type of file {name} - Skipping")
                continue
            thistype = ldtype[0]
        else:
            thistype = dtype

        if thistype == 'pixtable':
            coord_center = get_centre_from_pixtable(fullname)
        else:
            coord_center = get_centre_from_image_or_cube(fullname, ext=ext, dtype=dtype)

        # save in a dictionary the file names and the coordinates
        # Note that we only store the name, not the full name with folder
        center_dict[name] = coord_center

    pointing_dict, files_pointing_dict = group_xy_per_fieldofview(center_dict, limit=limit * unit)

    return center_dict, pointing_dict, files_pointing_dict


def get_centre_from_image_or_cube(filename, ext=1, dtype='image'):
    """Compute the coordinate of the center of the FOV from an image. Only pixels with
    actual signal are considered.

    Input
    -----
    file_name: st
        name of the file to analyse
    ext: int default=1, optional
        extension where the data and WCS info are located. Defaults to 1.
    dtype: str default='image', optional
        type of file to be analyzed. It can be either image or cube. Defaults to image.

    Returns
    -------
    SkyCoord: astropy.coordinates.SkyCoord
        Coordinate of the center of the FOV
    """

    # open the file and extract data and header
    with pyfits.open(filename) as hdu:
        data = hdu[ext].data
        head = hdu[ext].header

    # recover the WCS
    wcs = WCS(head)

    if dtype == 'cube':
        nz, ny, nx = data.shape
        # considering only central slice
        slice_id = nz // 2
        data = data[slice_id, :, :]
    elif dtype == 'image':
        ny, nx = data.shape
    else:
        return None

    # selecting only part of the image with signal
    mask = ~np.isnan(data)
    xx, yy = np.mgrid[0: ny, 0: nx]

    xx = xx[mask]
    yy = yy[mask]

    # computing the barycenter
    xcen = xx.sum() / len(xx)
    ycen = yy.sum() / len(yy)

    # from pixels to sky coordinates
    if dtype == 'image':
        coord = wcs.pixel_to_world(xcen, ycen)
    elif dtype == 'cube':
        coord = wcs.pixel_to_world(xcen, ycen, slice_id)[0]

    return coord


def get_pointing_table_from_folder(folder="", prefix="", suffix="", ext="fits", **kwargs):
    """Scan a given folder and look for a set of filenames that could enter a pointing
    table. Those names are decrypted following a given scheme.

    Input
    ------
    folder: str default=''
        Name of the folder to scan
    prefix: str default=''
        Prefix to be used to filter the filenames
    suffix: str default=''
        End of the word before the stem (extension)
    ext: str default='fits'
        Extension

    Create the self.pointing_table attribute
    """
    # First check that the folder exists
    realfolder = os.path.relpath(os.path.realpath(folder))
    if not os.path.isdir(realfolder):
        upipe.print_error(f"Folder {folder} does not exist - Aborting folder scan [pointing_table]")
        return None

    # initialise the table
    column_formats = kwargs.pop("column_formats", ('S50', 'S30', 'i4', 'i4'))
    pointing_table = QTable(names=('filename', 'tpls', 'dataset', 'expo'), dtype=column_formats)
    str_dataset = kwargs.pop("str_dataset", default_str_dataset)
    ndigits = kwargs.pop("ndigits", default_ndigits)
    filtername = kwargs.pop("filtername", None)

    # List the files in the folder, using the prefix
    this_folder = os.path
    list_existing_files = glob.glob(f"{folder}{prefix}*{suffix}.{ext}")
    upipe.print_info(f"Building the pointing table from the list of {len(list_existing_files)} "
                     "files")

    for filename in list_existing_files:
        fdataset, ftpls, fexpo = get_dataset_tpl_nexpo(filename, str_dataset=str_dataset,
                                                       ndigits=ndigits, filtername=filtername)
        # Exclude file which didn't have a proper dataset
        if int(fdataset) > 0:
            pointing_table.add_row((filename, ftpls, fdataset, fexpo))

    pointing_table.sort("filename")
    return pointing_table


class PointingTable(object):
    list_colnames_ptable = ['filename', 'dataset', 'tpls', 'expo']

    def __init__(self, tablename=None, folder='', **kwargs):
        """Set up the Pointing Table which contains information about datasets and
        pointings. This is initialised by a given filename.

        Input
        -----
        tablename: str
        folder: str

        """
        self.tablename = tablename
        realfolder = os.path.relpath(os.path.realpath(folder))
        if not os.path.isdir(realfolder):
            upipe.print_error(f"Folder {folder} does not exist. Using local folder")
            folder = ""

        self.folder = folder
        self.folderout = kwargs.pop("folderout", self.folder)

        self.format_read = kwargs.pop("format", "ascii")
        self.guess = kwargs.pop("guess", False)
        self.verbose = kwargs.pop("verbose", False)
        # Init an empty table
        self.pointing_table = QTable()

        # if filename exists is not None we read
        if tablename is not None:
            # If path name does not exist abort
            if not os.path.exists(self.fullname):
                upipe.print_error(f"Filename {tablename} does not exist in folder {folder}")
                return

            self.read(filename=tablename, folder=folder, format=self.format_read, guess=self.guess)
        # else we just scan the folder
        else:
            self.scan_folder(folder=folder, **kwargs)

    @property
    def fulltablename(self):
        return joinpath(self.folder, self.tablename)

    @property
    def fullnameout(self):
        if self.tablenameout is None:
            tablenameout = self.tablename
        else:
            tablenameout = self.tablenameout
        return joinpath(self.folderout, tablenameout)

    def _exist(self):
        if not hasattr(self, 'pointing_table'):
            upipe.print_error("Object does not have pointing_table attribute")
            return False

        return True

    def _check(self):
        """Check if the pointing table has the right minimum entries. Namely:
        filename, dataset, tpls, expo, pointing. If pointing does not exist,
        will create it following the logic : pointing=dataset.
        Returns boolean: True means it's all good. False will follow a specific message.

        """
        if not self._exist():
            return False

        missing_cols = [colname for colname in self.list_colnames_ptable if colname not in
                        self.pointing_table.colnames]
        if len(missing_cols) > 0:
            upipe.print_error(f"Missing columns in input file: {missing_cols}")
            return False
        else:
            return True

    def scan_folder(self, folder=None, **kwargs):
        """Scan a folder to create a full pointing table

        Input
        -----
        folder: str default to None
           If not provided, will use the default self.folder
        **kwargs:
            Other keywords are passed to create_pointing_table_from_folder

        Creates
        -------
        Attribute pointing_table
        """
        if folder is not None:
            self.folder = folder
        upipe.print_info(f"Scanning folder {self.folder}")

        self.pointing_table = get_pointing_table_from_folder(folder=self.folder, **kwargs)
        self._reset_select()
        self._reset_pointing()

    def write(self, overwrite=False, **kwargs):
        """Write out the pointing_table on disk, using the nameout and provided folder.

        Parameters
        ----------
        overwrite: bool default=False
        **kwargs:
            Valid keywords are
            folder: str
            nameout: str
            Extra keywords are passed to the astropy QTable.write() function

        Writes the pointing table on disk
        """
        # Reading the input
        folder = kwargs.pop("folder", self.folder)
        nameout = kwargs.pop("nameout", self.tablename)
        if nameout is None:
            upipe.print_error("No provided output filename")

        # Writing up using the astropy QTable write
        fullnameout = joinpath(folder, nameout)
        self.pointing_table.write(fullnameout, overwrite=overwrite, **kwargs)

    def _reset_select(self, value=1):
        """Reset the select column with 1 by default

        Input
        -----
        value: int default=1
        """
        # If no selection is done, just select all by default
        if "select" not in self.pointing_table.colnames:
            self.pointing_table['select'] = int(value)

    def _reset_pointing(self, overwrite=False):
        """Reset the pointing column in the pointing table
        """
        if 'pointing' in self.pointing_table.colnames:
            if overwrite==True:
                self.pointing_table.replace_column(int(0), name='pointing')
        else:
            self.pointing_table.add_column(int(0), name='pointing')

    def read(self, **kwargs):
        """Read the input filename in given folder assuming a given format.

        Input
        -----
        filename: str, optional
            Name of the filename
        folder: str default='', optional
            Name of the folder where to find the filename
        format: str default='ascii'

        Returns
        -------
        self.pointing_table with the content of the file
        """
        self.filename = kwargs.pop("filename", self.filename)
        self.folder = kwargs.pop("folder", self.folder)
        fullname = joinpath(self.folder, self.filename)
        self.format_read = kwargs.pop("format", 'ascii')
        self.guess = kwargs.pop("guess", False)
        if not os.path.exists(fullname):
            upipe.print_error(f"Pointing Table {fullname} does not exist. Cannot open")
            return

        self.pointing_table = QTable.read(fullname, format=self.format_read, guess=self.guess,
                                         **kwargs)
        if not self._check():
            upipe.print_warning("Please consider updating the pointing table")

        self._reset_select()
        self._reset_pointing()

        # After reading, save the selection to an original one to backup
        self.pointing_table['select_orig'] = self.pointing_table('select')

        self._get_centres()

    def _get_centres(self, dtype="image", center_dict=None, **kwargs):
        """Get the centre of each exposure, assuming a given type

        Input
        ------
        dtype: str
            Must be 'image', 'cube' or 'pixeltable'
        center_dict: dictionary, optional
            Dictionary including the filenames and their centres
            If not provided, it will just use the filenames and rederive the centres

        **kwargs:
            Additional keywords. Valid ones are
               ext: int
                   Number of the extension to look at for thw WCS

        Add column centre in the pointing_table

        """
        if dtype not in ["image", "cube", "pixtable", "guess"]:
            upipe.print_error(f"Dtype {dtype} not recognised ['image', 'cube', 'pixtable', 'guess']")
            return

        dict_dtype = {'IMAGE': 'image', 'CUBE': 'cube', 'PIXTABLE': 'pixtable'}

        # initialise the centre column
        if 'centre' not in self.pointing_table.colnames:
            dummycoord = [SkyCoord(0, 0, unit='deg')] * len(self.pointing_table)
            self.pointing_table.add_column(dummycoord, name="centre")

        # Loop over the pointing table and find the centre
        for i in range(len(self.pointing_table)):
            filename = self.pointing_table['filename'][i]

            centre = None
            if center_dict is None:
                fullname = joinpath(self.folder, filename)
                if dtype == "guess":
                    ldtype = [dtype for dtype in dict_dtype if dtype in filename]
                    if len(ldtype) == 0:
                        upipe.print_warning(f"Could not guess type of file {filename} - Skipping")
                        continue
                    thistype = ldtype[0]
                else:
                    thistype = dtype

                if thistype in ["image", "cube"]:
                    centre = get_centre_from_image_or_cube(fullname, dtype=thistype, **kwargs)
                else:
                    centre = get_centre_from_pixtable(fullname, **kwargs)
            else:
                if filename in center_dict:
                    centre = center_dict[filename]

            # Assigning the centre to the right row
            self.pointing_table['centre'][i] = centre

    def assign_pointings(self, **kwargs):
        """Assign pointing according to distance rules. Will also update the centre values.

        Returns
        -------

        """
        verbose = kwargs.pop("verbose", self.verbose)
        overwrite = kwargs.pop("overwrite", False)
        self._reset_pointing(overwrite=overwrite)

        self.center_dict, self.pointing_dict, self.file_pointing_dict = \
            group_exposures_per_pointing(self.selected_filenames, target_path=self.folder, **kwargs)

        # Assign centres
        self._get_centres(center_dict=self.center_dict)

        # Now writing the pointing in the pointing table
        upipe.print_info(f"Assigning Pointings ---")
        for filename in self.file_pointing_dict:
            pointing = self.file_pointing_dict[filename]
            self.pointing_table['pointing'][self.pointing_table['filename'] == filename] = pointing
            if verbose:
                upipe.print_info(f"File: {filename} = Pointing {pointing:02d}")

    def select_from_list(self, **kwargs):
        """Select all filenames with that pointing number.

        Input
        -----
        **kwargs: default lists are empty ones
            Valid keywords are:
                list_datasets: list of int
                list_pointings: list of int

        Returns
        -------
        An astropy table selected according to the list of datasets and pointings
        """
        list_pointings = kwargs.pop("list_pointings", [])
        list_datasets = kwargs.pop("list_datasets", [])

        # Now getting the selections
        for i in range(len(self.pointing_table)):
            p = self.pointing_table['pointing'][i]
            d = self.pointing_table['dataset'][i]
            if (p not in list_pointings) or (d not in list_datasets):
                self.pointing_table['select'] = 0

    @property
    def selected_filenames(self):
        """Return the list of filenames following the selection

        Returns
        -------
        list_filename

        """
        if not hasattr(self, 'pointing_table'):
            upipe.print_warning("Missing a pointing table - returning an empty filename list")
            return []

        self.pointing_table.add_index('select')
        inds = self.pointing_table.loc_indices['select', 1]

        lfiles = list(self.pointing_table['filename'][inds])
        lfiles.sort()

        return lfiles
