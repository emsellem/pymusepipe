"""Utility functions for images in pymusepipe
"""

__authors__ = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__ = "MIT License"
__contact__ = " <eric.emsellem@eso.org>"

# Importing modules

# Numpy
import numpy as np
from scipy.odr import ODR, Model, RealData
from scipy import ndimage as nd

from astropy.stats import mad_std, sigma_clip, sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, convolve

# Import package modules
from . import util_pipe as upipe


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
                  median_window=10, minflux=0.0, squeeze=True,
                  remove_bkg=True):
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
    squeeze: bool
        Squeeze the dynamic range by using the dynamic_range variable
    crop: bool
        Crop the borders using border as the variable
    remove_bkg: remove the filter_medianed background

    Returns
    -------
    """
    if squeeze:
        # Squish bright pixels down
        data = np.arctan(data / np.nanmedian(data) / dynamic_range)

    if remove_bkg:
        # Omit the border pixels
        data -= filtermed_image(data, 0, median_window)

    cdata = crop_data(data, border)

    # Removing the zeros
    with np.errstate(invalid='ignore'):
        cdata[cdata < minflux] = 0.

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
            dist_from_source = np.sqrt((xx - source['xcentroid'])**2 + (yy - source['ycentroid'])**2)
            mask = dist_from_source < mask_radius
            mima[mask] = np.nan

    return mima
