"""MUSE-PHANGS convolve module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2019, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"

# This module uses some mpdaf routines and pypher to 
# convolve a given datacube to a calibrated PSF
# This uses Moffat functions as reference

# Importing modules
import numpy as np

# Astropy
from astropy.convolution import (Moffat2DKernel, Gaussian2DKernel, convolve,
                                 convolve_fft)
from astropy.stats import gaussian_fwhm_to_sigma

# pypher
try:
    import pypher.pypher as ph
except ImportError:
    print("IMPORT WARNING: pypher is needed for cube_convolve")

# Importing mpdaf
try :
    import mpdaf
except ImportError :
    print("IMPORT ERROR: mpdaf is needed for cube_convolve")

def pypher_script(psf_source, psf_target, pixscale_source=0.2,
                  pixscale_target=0.2, angle_source=0., angle_target=0.,
                  reg_fact=1e-4, verbose=False):
    """calculate the convolution kernel to move from one PSF to a target one.
    This is an adaptation of the main pypher script that it is meant to be used
    from the terminal.

    Parameters:
        psf_source (ndarray): 2D PSF of the source image.
        psf_target (ndarray): target 2D PSF
        pixscale_source (float): pixel scale of the source PSF [0.2]
        pixscale_target (float): pixel scale of the target PSF [0.2]
        angle_source (float): position angle of the source PSF. [0]
        angle_target (float): position angle of the target PSF. [0]
        reg_fact (float): Regularisation parameter for the Wiener filter [1.e-4]
        verbose (bool): If True it prints more info on screen [False]

    Returns:
        kernel: a 2D kernel that convolved with the source PSF
            returns the target PSF

    """

    # Set NaNs to 0.0 for both input PSFs
    psf_source = np.nan_to_num(psf_source)
    psf_target = np.nan_to_num(psf_target)

    if verbose:
        print('Source PSF pixel scale: %.2f arcsec', pixscale_source)
        print('Target PSF pixel scale: %.2f arcsec', pixscale_target)

    # Rotate images (if necessary)
    if angle_source != 0.0:
        psf_source = ph.imrotate(psf_source, angle_source)
        if verbose:
            print('Source PSF rotated by %.2f degrees', angle_source)
    if angle_target != 0.0:
        psf_target = ph.imrotate(psf_target, angle_target)
        if verbose:
            print('Target PSF rotated by %.2f degrees', angle_target)

    # Normalize the PSFs if needed
    normalize_s = psf_source.sum()
    normalize_t = psf_target.sum()
    if np.allclose([normalize_s], [1.0], atol=1.e-3):
        psf_source /= normalize_s
        if verbose:
            print('Source PSF normalized with normalization '
                  'constant {:0.2f}'.format(normalize_s))
    if np.allclose([normalize_t], [1.0], atol=1.e-3):
        psf_target /= normalize_t
        if verbose:
            print('Target PSF normalized with normalization '
                  'constant {:0.2f}'.format(normalize_t))

    # Resample high resolution image to the low one
    if pixscale_source != pixscale_target:
        try:
            psf_source = ph.imresample(psf_source,
                                       pixscale_source,
                                       pixscale_target)
        except MemoryError:
            print('- COMPUTATION ABORTED -')
            print('The size of the resampled PSF would have '
                  'exceeded 10K x 10K')
            print('Please resize your image and try again')
            return None
        if verbose:
            print('Source PSF resampled to the target pixel scale')

    # check the new size of the source vs. the target
    if psf_source.shape > psf_target.shape:
        psf_source = ph.trim(psf_source, psf_target.shape)
    else:
        psf_source = ph.zero_pad(psf_source, psf_target.shape,
                                 position='center')

    kernel, _ = ph.homogenization_kernel(psf_target, psf_source,
                                         reg_fact=reg_fact)

    if verbose:
        print('Kernel computed using Wiener filtering and a regularisation '
              'parameter r = {:0.2e}'.format(reg_fact))

    return kernel

def moffat_kernel(fwhm, size, n=1.0, scale=0.2):
    """ Moffat kernel. Returns a Moffat function array according to given
    input parameters. Using astropy Moffat2DKernel.

    Parameters:
        fwhm (float): fwhm of the Moffat kernel, in arcsec.
        n (float): power index of the Moffat
        size (int numpy array): size of the requested kernel along each axis.
            If ``size'' is a scalar number the final kernel will be a square of
            side ``size''. If ``size'' has two element they must be in
            (y_size, x_size) order. In each case size must be an integer
            number of pixels.
        scale (float): pixel scale of the image [optional]
    """
    if n is None:
        print("ERROR[moffat_kernel]: n cannot be None for Moffat")
        return None

    # Check the size nature (scalar and array or not)
    if np.isscalar(size):
        size = np.repeat(size, 2)

    if len(size) > 2:
        print('ERROR[moffat_kernel]: size must have at most two elements.')
        return None

    # Calculate the gamma power for the Moffat function
    gamma = fwhm / (2.0 * scale * np.sqrt(2.0**(1. / n) - 1.0))
    moffat_k = Moffat2DKernel(gamma, n, x_size=size[1], y_size=size[0])

    return moffat_k.array

def gaussian_kernel(fwhm, size, scale=0.2, **kwargs):
    """
    Gaussian kernel.
    Input:
        fwhm (float): fwhm of the Gaussian kernel, in arcsec.
        size (int, ndarray): size of the requested kernel along each axis.
            If ``size'' is a scalar number the final kernel will be a square of
            side ``size''. If ``size'' has two element they must be in
            (y_size, x_size) order. In each case size must be an integer number
            of pixels.
        scale (float): pixel scale of the image
        **kwargs: is there to absorb any additional parameter which could be
           provided (but won't be used).

    """
    if np.isscalar(size):
        size = np.repeat(size, 2)

    if len(size) > 2:
        print('size must have at most two elements.')

    # Compute sigma for the gaussian in pixels
    std = fwhm * gaussian_fwhm_to_sigma / scale
    gaussian_k = Gaussian2DKernel(std, x_size=size[1], y_size=size[0])

    return gaussian_k.array

def psf3d(wave, size, fwhm0, lambda0=6483.58, b=-3e-5, scale=0.2, nmoffat=None,
          function="moffat"):
    """
    Function to create the cube with the  lambda dependent PSF, following
    a given slope and nominal wavelength.

    Parameters
        wave: np.ndarray
            array with the wavelength axis of the datacube
        size: int, array-like
            the size of the 2D PSF.  If ``size'' is a scalar number the 2D PSF
            kernel will be a square of side ``size''. If ``size'' has two element
            they must be in (y_size, x_size) order.
        fwhm0: float
            the fwhm at the reference wavelength in arcseconds.
        n: float
            Power index of the Moffat profile. It is usually 2.8 for NOAO cubes
            and 2.3 for AO cubes.
        lambda0: float
            reference wavelength at which fwhm0 is measured. Default: 6483.58.
            (It's the average wavelength for the WFI_BB filter)
        b: float, optional
            the steepness of the relation between wavelength and FWHM.
            Default: -3e-5 (arcsec/A) (From MUSE team)
        scale: float, optional
            spatial scale of the new datacube in arcsec. Default: 0.2 (MUSE
            spatial resolution).
        function (str): 'moffat' or 'gaussian'
    Returns
        psf_cube: np.array
            Datacube containing MUSE PSF as a function of wavelength.

    """
    dict_kernel = {'gaussian': gaussian_kernel, 'moffat': moffat_kernel}

    if np.isscalar(size):
        size = np.repeat(size, 2)

    if len(size) > 2:
        print('size must have at most two elements.')
        return None, None

    # computing the fwhm as a function of wavelength
    fwhm_wave = b * (wave - lambda0) + fwhm0

    # check the maximum size of the PSF.
    print('Max. FWHM {0:0.3f} at wavelength {1:.2f}'.format(
            np.max(fwhm_wave), wave[np.argmax(fwhm_wave)]))

    # creating the 3D PSF.
    psf_cube = np.zeros((len(wave), *size))
    if function in dict_kernel:
        function_kernel = dict_kernel[function]
        for i, fwhm in enumerate(fwhm_wave):
            kernel = function_kernel(fwhm, size, scale=scale, n=nmoffat)
            psf_cube[i, :, :] = kernel
        return psf_cube, fwhm_wave
    else:
        print("ERROR[psf3d]: input function not part of the available ones"
              "({})".format(list(dict_kernel.keys())))
        return None, None

def psf2d(size, fwhm, function='gaussian', nmoffat=None, scale=0.2):
    """
    Create a model of the target PSF of the convolution. The target PSF does
    not vary as a function of wavelenght, therefore the output is a 2D array.

    Parameters
        size: int, array-like
            the size of the final array. If ``size'' is a scalar number the
            kernel will be a square of side ``size''. If ``size'' has two
            elements they must be in (y_size, x_size) order.
        fwhm: float
            the FWHM of the psf
        function: str, optional
            the function to model the target PSF. Only 'gaussian' or 'moffat'
            are accepted. Default: 'gaussian'
        nmoffat (float): Moffat power index. It must be defined if
        function = 'moffat'.
            Default: None
        scale: float, optional
            the spatial scale of the final kernel

    Returns
        target: np.ndarray
            a 2D array with the model of the target PSF.
    """
    dict_kernel = {'gaussian': gaussian_kernel, 'moffat': moffat_kernel}

    if np.isscalar(size):
        size = np.repeat(size, 2)

    if len(size) > 2:
        print('size must have at most two elements.')
        return None

    if function in dict_kernel:
        function_kernel = dict_kernel[function]
        kernel = function_kernel(fwhm, size, scale=scale, n=nmoffat)
        return kernel
    else:
        print("ERROR[psf2d]: input function not part of the available ones"
              "({})".format(list(dict_kernel.keys())))
        return None


def convolution_kernel(input_psf, target_psf, scale=0.2):
    """Create the 3D convolution kernel starting from a 3D model of the original
    PSF and a 2D model of the target PSF using pypher.

    Parameters
        input_psf (np.ndarray): 3D array with the model of the original PSF
        target_psf (np.ndarray): 2D array with a model of the target PSF
        scale (float): spatial scale of both PSF in arcsec/pix

    Returns
        conv_kernel (np.ndarray): 3D array with a convolution kernel
            that varies as a function of wavelength.
    """

    assert len(target_psf.shape) == 2, 'the target_psf must be a 2d array'

    n_lam = input_psf.shape[0]

    conv_kernel = np.zeros_like(input_psf, dtype=np.float32)

    for i in range(n_lam):
        conv_kernel[i, :, :] = pypher_script(input_psf[i, :, :], target_psf,
                                             pixscale_source=scale,
                                             pixscale_target=scale,
                                             angle_source=0, angle_target=0, )

    return conv_kernel


def convolution_kernel_gaussian(fwhm_wave, target_fwhm, target_psf,
                                scale=0.2):
    """Create the 3D convolution kernel starting from a 3D model of the original
    PSF and a 2D model of the target PSF using both gaussian functions.

    Args:
        fwhm_wave (array): FWHM of the original PSF as a function of
            wavelength
        target_fwhm (float): fwhm of the target PSF
        target_psf (np.ndarray): target psf2d
        scale (float): spatial scale of both PSF in arcsec/pix

    Returns:
        conv_kernel: np.ndarray
            3D array with a convolution kernel that varies as a function of
            wavelength.
    """

    assert len(target_psf.shape) == 2, 'the target_psf must be a 2d array'

    # getting the first dimension (lambda) of the input PSF
    n_lam = target_psf.shape[0]
    conv_kernel = np.zeros_like(target_psf)

    if fwhm <= np.max(fwhm_wave):
        raise ValueError('The new PSF is smaller than the old one')

    # Loop over the wavelengths
    for i in range(n_lam):
        new_fwhm = np.sqrt(fwhm**2 - input_fwhm[i]**2)
        ker = gaussian_kernel(new_fwhm, target_psf.shape,
                              scale=0.2)
        conv_kernel[i, :, :] = ker / ker.sum()

    return conv_kernel

def cube_convolve(data, kernel, variance=None, fft=True):
    """Convolve a 3D datacube

    Args:
        datacube:
        kernel:

    Returns:
        the convolved 3D data and its variance

    """
    dict_func = {True: convolve_fft, False: convolve}
    conv_function = dict_func[fft]
    # if fft:
    #     print("Starting the convolution with fft")
    #     conv_function = convolve_fft
    # else:
    #     print("Starting the linear convolution")
    #     conv_function = convolve

    norm_kernel = np.divide(kernel.T, kernel.sum(axis=(1, 2)).T).T
    var_kernel = norm_kernel**2

    # Removed the perslice option as it MUST be done per slice
    # Or it provides a 3D FFT which is something different
    # if perslice:
    print("Convolution using per slice-2D convolve in astropy")
    for i in range(data.shape[0]):
        # Signal
        data[i, :, :] = conv_function(data[i, :, :],
                                      norm_kernel[i, :, :],
                                      allow_huge=True,
                                      psf_pad=True,
                                      fft_pad=False,
                                      boundary='fill',
                                      fill_value=0,
                                      normalize_kernel=False)

    if variance is not None:
        for i in range(variance.shape[0]):
            # Variance
            variance[i, :, :] = conv_function(variance[i, :, :],
                                              var_kernel[i, :, :],
                                              allow_huge=True,
                                              psf_pad=True,
                                              fft_pad=False,
                                              boundary='fill',
                                              fill_value=0,
                                              normalize_kernel=False)
    # else:
    #     print("Convolution using 3D convolve in astropy")
    #     data = conv_function(data, norm_kernel, allow_huge=True,
    #                          psf_pad=True, fft_pad=False,
    #                          boundary='fill', fill_value=0,
    #                          normalize_kernel=False)
    #     if variance is not None:
    #         variance = conv_function(variance, var_kernel, allow_huge=True,
    #                                  psf_pad=True, fft_pad=False,
    #                                  boundary='fill', fill_value=0,
    #                                  normalize_kernel=False)

    return data, variance

def cube_kernel(shape, wave, input_fwhm,  target_fwhm,
                input_function, target_function, lambda0=6483.58,
                input_nmoffat=None, target_nmoffat=None, b=-3e-5,
                scale=0.2, compute_kernel='pypher'):
    """Main function to create the convolution kernel for the datacube

    Args:
        shape (array): the shape of the datacube that is going to be convolved.
            It must be in the form (z, y, x).
        wave (float array): wavelengths for the datacube
        target_fwhm (float): fwhm of the target PSF.
        input_fwhm (float): fwhm of the original PSF at the reference
            wavelength lambda0
        input_function (str): function to be used to describe the input PSF
        target_function (str): function to be used to describe the target PSF
        lambda0: float, optional
            the wavelength at which the original_fwhm has been measured.
            Default: 6483.58 (central wavelenght of WFI_BB filter)
        input_nmoffat (float): power index of the original PSF if Moffat [None]
        target_nmoffat (float): power index for the target PSF if Moffat [None]
        b (float): steepness of the fwhm vs wavelength relation. Default: -3e-5
        step (float): wavelength dispersion in AA/px
        scale (float): spatial pixel scale of the PSFs in arcsec/pix
        compute_kernel (str): method to compute the convolution kernel.
            It can be 'pypher' or 'gaussian'

    Returns:
        Kernel: np.ndarray
            3D array to be used in the convolution

    """

    # Size of the 2d PSF (x,y)
    size = shape[1:]

    print('\nCreating the cube with the original PSF')
    original_psf, fwhm_wave = psf3d(wave, size, input_fwhm, nmoffat=input_nmoffat,
                                    function=input_function, lambda0=lambda0,
                                    b=b, scale=scale)

    print('Creating the image with the target PSF')
    target_psf = psf2d(size, target_fwhm, function=target_function,
                       nmoffat=target_nmoffat, scale=scale)

    print('Creating the convolution kernel')
    if compute_kernel == 'pypher':
        print('Building the convolution kernel via pypher')
        kernel = convolution_kernel(original_psf, target_psf, scale=0.2)
    elif compute_kernel == 'gaussian':
        print('Building Gaussian Kernel')
        kernel = convolution_kernel_gaussian(target_psf, target_fwhm,
                                             fwhm_wave, scale=0.2)
    else:
        kernel = None

    return kernel
