# Licensed under a MIT license - see LICENSE

"""MUSE-PHANGS mpdaf-functions module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"

# This module uses some mpdaf routines and wrap some of their
# functionalities to help the muse_pipe checkups

# Importing modules
import numpy as np
from numpy import ma

# Standard modules
import os
from os.path import join as joinpath
import glob
import copy

# Importing mpdaf
try :
    import mpdaf
except ImportError :
    raise Exception("mpdaf is required for this - MUSE related - module")

from mpdaf.obj import Cube, Image, CubeMosaic, DataArray
from mpdaf.obj import Spectrum, WaveCoord
from mpdaf.tools import add_mpdaf_method_keywords
from mpdaf.drs import PixTable

# Astropy
from astropy.io import fits as pyfits
from astropy import units as u

# Scipy erosion
from scipy import ndimage as ndi

import pymusepipe
from .config_pipe import default_wave_wcs, ao_mask_lambda, dict_extra_filters
from .util_pipe import (filter_list_with_suffix_list, add_string, default_str_dataset, default_ndigits,
                        get_dataset_name, check_filter_list)
from .util_pipe import (print_error, print_info, print_warning, print_debug)
from .util_image import filter_list_with_pointingtable
from .cube_convolve import cube_kernel, cube_convolve
from .emission_lines import get_emissionline_band


def get_sky_spectrum(specname):
    """Read sky spectrum from MUSE data reduction
    """
    if not os.path.isfile(specname):
        print_error("{0} not found".format(specname))
        return None

    sky = pyfits.getdata(specname)
    crval = sky['lambda'][0]
    cdelt = sky['lambda'][1] - crval
    wavein = WaveCoord(cdelt=cdelt, crval=crval, cunit=u.angstrom)
    spec = Spectrum(wave=wavein, data=sky['data'], var=sky['stat'])
    return spec


def integrate_spectrum(spectrum, wave_filter, throughput_filter, ao_mask=False):
    """Integrate a spectrum using a certain Muse Filter file.

    Input
    -----
    spectrum: Spectrum
        Input spectrum given as an mpdaf Spectrum
    wave_filter: float array
        Array of wavelength for the filter
    throughput_filter: float array
        Array of throughput (between 0 and 1) for the filter. Should be the 
        same dimension (1D, N floats) as wave_filter
    """
    # interpolation linearly the filter throughput onto
    # the spectrum wavelength
    specwave = spectrum.wave.coord()
    specdata = spectrum.data

    # If we have an AO mask, we interpolate the spectrum within that range
    if ao_mask:
        outside_AO = (specwave < ao_mask_lambda[0]) | (specwave > ao_mask_lambda[1])
        good_spectrum = np.interp(specwave, specwave[outside_AO],
                                  specdata[outside_AO])
        # Replacing the interval with interpolated spectrum
        specdata[~outside_AO] = good_spectrum[~outside_AO]

    effS = np.interp(specwave, wave_filter, throughput_filter)
    # Retaining only positive values
    goodpix = (effS > 0)

    filtwave = np.sum(effS[goodpix])
    if filtwave > 0:
        flux_cont = np.sum(spectrum.data[goodpix] * effS[goodpix]) / filtwave
    else:
        flux_cont = 0.0

    return flux_cont


def rotate_image_wcs(ima_name, ima_folder="", outwcs_folder=None, rotangle=0.,
                     **kwargs):
    """Routine to remove potential Nan around an image and reconstruct
    an optimal WCS reference image. The rotation angle is provided as a way
    to optimise the extent of the output image, removing Nan along X and Y
    at that angle.

    Input
    -----
    ima_name: str
        input image name. No default.
    ima_folder: str default='', optional
        input image folder
    outwcs_folder: str, optional
        folder where to write the output frame. Default is
        None which means that it will use the folder of the input image.
    rotangle: float default=0, optional
        rotation angle in degrees
    in_suffix: str default='prealign'
        in suffix to remove from name
    out_suffix: str default='rotwcs'
        out suffix to add to name
    margin_factor: float
        factor to extend the image [1.1]

    Returns
    -------

    """

    # Reading the input names and setting output folder
    fullname = joinpath(ima_folder, ima_name)
    ima_folder, ima_name = os.path.split(fullname)
    if outwcs_folder is None:
        outwcs_folder = ima_folder

    # Suffix
    in_suffix = kwargs.pop("in_suffix", "prealign")
    out_suffix = kwargs.pop("out_suffix", "rotwcs")

    # Get margin if needed
    margin_factor = kwargs.pop("margin_factor", 1.1)
    extend_fraction = np.maximum(0., (margin_factor - 1.))
    print_info("Will use a {:5.2f}% extra margin".format(
                     extend_fraction*100.))

    # Opening the image via mpdaf
    imawcs = Image(fullname)
    extra_pixels = (np.array(imawcs.shape) * extend_fraction).astype(int)

    # New dimensions and extend current image
    new_dim = tuple(np.array(imawcs.shape).astype(int) + extra_pixels)
    ima_ext = imawcs.regrid(newdim=new_dim, refpos=imawcs.get_start(),
                            refpix=tuple(extra_pixels / 2.),
                            newinc=imawcs.get_step()[0]*3600.)

    # Copy and rotate WCS
    new_wcs = copy.deepcopy(ima_ext.wcs)
    print_info("Rotating WCS by {} degrees".format(rotangle))
    new_wcs.rotate(rotangle)

    # New rotated image
    ima_rot = Image(data=np.nan_to_num(ima_ext.data), wcs=new_wcs)

    # Then resample the image using the initial one as your reference
    ima_rot_resampled = ima_rot.align_with_image(ima_ext, flux=True)

    # Crop NaN
    ima_rot_resampled.crop()

    # get the new header with wcs and rotate back
    finalwcs = ima_rot_resampled.wcs
    finalwcs.rotate(-rotangle)

    # create the final image
    final_rot_image = Image(data=ima_rot_resampled.data, wcs=finalwcs)

    # Save image
    if isinstance(in_suffix, str) and in_suffix != "" and in_suffix in ima_name:
            out_name = ima_name.replace(in_suffix, out_suffix)
    else:
        name, extension = os.path.splitext(ima_name)
        out_suffix = add_string(out_suffix)
        out_name = "{0}{1}{2}".format(name, out_suffix, extension)

    # write output
    final_rot_image.write(joinpath(outwcs_folder, out_name))
    return outwcs_folder, out_name


def rotate_cube_wcs(cube_name, cube_folder="", outwcs_folder=None, rotangle=0.,
                     **kwargs):
    """Routine to remove potential Nan around an image and reconstruct
    an optimal WCS reference image. The rotation angle is provided as a way
    to optimise the extent of the output image, removing Nan along X and Y
    at that angle.

    Args:
        cube_name (str): input image name. No default.
        cube_folder (str): input image folder ['']
        outwcs_folder (str): folder where to write the output frame. Default is
            None which means that it will use the folder of the input image.
        rotangle (float): rotation angle in degrees [0]
        **kwargs:
            in_suffix (str): in suffix to remove from name ['prealign']
            out_suffix (str): out suffix to add to name ['rotwcs']
            margin_factor (float): factor to extend the image [1.1]

    Returns:

    """

    # Reading the input names and setting output folder
    fullname = joinpath(cube_folder, cube_name)
    cube_folder, cube_name = os.path.split(fullname)
    if outwcs_folder is None:
        outwcs_folder = cube_folder

    # Suffix
    in_suffix = kwargs.pop("in_suffix", "prealign")
    out_suffix = kwargs.pop("out_suffix", "rotwcs")

    # Get margin if needed
    margin_factor = kwargs.pop("margin_factor", 1.1)
    extend_fraction = np.maximum(0., (margin_factor - 1.))
    print_info("Will use a {:5.2f}% extra margin".format(
                     extend_fraction*100.))

    # Opening the image via mpdaf
    cubewcs = Cube(fullname)
    imawcs = cubewcs.sum(axis=0)
    extra_pixels = (np.array(imawcs.shape) * extend_fraction).astype(int)

    # New dimensions and extend current image
    new_dim = tuple(np.array(imawcs.shape).astype(int) + extra_pixels)
    ima_ext = imawcs.regrid(newdim=new_dim, refpos=imawcs.get_start(),
                            refpix=tuple(extra_pixels / 2.),
                            newinc=imawcs.get_step()[0]*3600.)

    # Copy and rotate WCS
    new_wcs = copy.deepcopy(ima_ext.wcs)
    print_info("Rotating spatial WCS of Cube by {} degrees".format(rotangle))
    new_wcs.rotate(rotangle)

    # New rotated image
    ima_rot = Image(data=np.nan_to_num(ima_ext.data), wcs=new_wcs)

    # Then resample the image using the initial one as your reference
    ima_rot_resampled = ima_rot.align_with_image(ima_ext, flux=True)

    # Crop NaN
    ima_rot_resampled.crop()

    # get the new header with wcs and rotate back
    finalwcs = ima_rot_resampled.wcs
    finalwcs.rotate(-rotangle)

    # create the final image
    data_cube_rot = np.repeat(ima_rot_resampled[np.newaxis,:,:].data,
                              cubewcs.shape[0], axis=0)
    final_rot_cube = Cube(data=data_cube_rot, wave=cubewcs.wave, wcs=finalwcs)

    # Save image
    if isinstance(in_suffix, str) and in_suffix != "" and in_suffix in cube_name:
            out_name = cube_name.replace(in_suffix, out_suffix)
    else:
        name, extension = os.path.splitext(cube_name)
        if out_suffix != "":
            out_suffix = add_string(out_suffix)
        out_name = "{0}{1}{2}".format(name, out_suffix, extension)

    # write output
    final_rot_cube.write(joinpath(outwcs_folder, out_name))
    return outwcs_folder, out_name


class BasicPSF(object):
    """Basic PSF function and parameters
    """
    def __init__(self, function="gaussian", fwhm0=0., nmoffat=2.8, b=0.,
                 l0=6483.58, psf_array=None):

        if psf_array is not None:
            self.function = psf_array[0].lower()
            self.fwhm0 = psf_array[1]
            self.nmoffat = psf_array[2]
            self.b = psf_array[3]
            self.l0 = psf_array[4]
        else:
            self.function = function.lower()
            self.fwhm0 = fwhm0
            self.nmoffat = nmoffat
            self.b = b
            self.l0 = l0

    @property
    def psf_array(self):
        return [self.function, self.fwhm0, self.nmoffat, self.l0, self.b]

class BasicFile(object):
    """Basic file with just the name and some properties
    to attach to that Cube
    """
    def __init__(self, filename, **kwargs):
        self.filename = filename
        for key in kwargs:
            val = kwargs.get(key)
            setattr(self, key, val)


class MuseCubeMosaic(CubeMosaic):
    def __init__(self, ref_wcs, folder_ref_wcs="", folder_cubes="",
                 prefix_cubes="DATACUBE_FINAL_WCS",
                 list_suffix=[], use_fixed_cubes=True,
                 excluded_suffix=[],
                 included_suffix=[],
                 prefix_fixed_cubes="tmask", verbose=False,
                 pointing_table=None, list_pointings=None,
                 dict_psf={}, list_cubes=None, **kwargs):

        self.verbose = verbose
        self.folder_cubes = folder_cubes
        self.folder_ref_wcs = folder_ref_wcs
        self.ref_wcs = ref_wcs
        if not self._check_folder():
            return
        self.full_wcs_name = joinpath(self.folder_ref_wcs, self.ref_wcs)

        self.prefix_cubes = prefix_cubes
        self.list_suffix = list_suffix
        self.excluded_suffix = excluded_suffix
        self.included_suffix = included_suffix
        self.prefix_fixed_cubes = prefix_fixed_cubes
        self.use_fixed_cubes = use_fixed_cubes
        self.dict_psf = dict_psf
        self.pointing_table = pointing_table
        self.list_pointings = list_pointings

        self.str_dataset = kwargs.pop("str_dataset", default_str_dataset)
        self.ndigits = kwargs.pop("ndigits", default_ndigits)

        # Building the list of cubes
        self.build_list(list_cubes=list_cubes)

        # Initialise the super class
        super(MuseCubeMosaic, self).__init__(self.cube_names, self.full_wcs_name)

        # Check unit
        if self.unit == u.dimensionless_unscaled:
            self._get_unit()

    @property
    def cube_names(self):
        return [c.filename for c in self.list_cubes]

    def print_cube_names(self):
        lnames = copy.copy(self.cube_names)
        lnames.sort()
        print("Mosaic Cubes ==========================")
        for i, name in enumerate(lnames):
            print(f"#{i+1:02d} - {name}")
        print("=======================================")

    @property
    def ncubes(self):
        return len(self.list_cubes)

    @property
    def list_cubes(self):
        if not hasattr(self, "_list_cubes"):
            self._list_cubes = []
        return self._list_cubes

    @list_cubes.setter
    def list_cubes(self, val):
        self._list_cubes = val

    def _check_folder(self):
        if not os.path.isdir(self.folder_cubes):
            print_error("Cube Folder {} does not exists \n"
                              "- Aborting".format(self.folder_cubes))
            return False
        if not os.path.isdir(self.folder_ref_wcs):
            print_error("WCS Folder {} does not exists \n"
                              "- Aborting".format(self.folder_ref_wcs))
            return False
        return True

    def _get_unit(self):
        list_units = []
        for name in self.cube_names:
            try:
                list_units.append(Cube(name).unit)
            except:
                pass

        u_units = np.unique(list_units)
        if len(u_units) == 0:
            print("Warning: no unit found in input cubes, will set dimensionless one")
            self.unit = u.dimensionless_unscaled
        else:
            self.unit = u_units[0]
            if len(u_units) != 1:
                print("Warning: units are not all the same for all input Cubes. "
                      "  Selecting the first encountered")
            print(f"Unit of mosaiced Cube = {self.unit.to_string()}")

    def build_list(self, folder_cubes=None, prefix_cubes=None, list_cubes=None,
                   **kwargs):
        """Building the list of cubes to process

        Args:
            folder_cubes (str): folder for the cubes
            prefix_cubes (str): prefix to be used

        """

        print_info('Building the list of appropriate Cubes for the mosaick')
        self.list_suffix = kwargs.pop("list_suffix", self.list_suffix)
        # Get the folder if needed
        if folder_cubes is not None:
            self.folder_cubes = folder_cubes
            if not self._check_folder():
                return

        # get the prefix if provided
        if prefix_cubes is not None:
            self.prefix_cubes = prefix_cubes

        if list_cubes is None:
            # get the list of cubes and return if 0 found
            list_existing_cubes = glob.glob(f"{self.folder_cubes}{self.prefix_cubes}*.fits")

            print_info(f"Found {len(list_existing_cubes)} existing Cubes "
                             f"in this folder with prefix {self.prefix_cubes}")

            # if the list of exclusion suffix is empty, just use all cubes
            list_existing_cubes = filter_list_with_suffix_list(list_existing_cubes,
                                                               self.included_suffix,
                                                               self.excluded_suffix,
                                                               name_list="Existing Cubes")
            print_info("Found {} Cubes after suffix filtering".format(
                len(list_existing_cubes)))

            # Filtering the list using the input pointing table
            if self.pointing_table is not None:
                list_cubes = filter_list_with_pointingtable(list_existing_cubes,
                                                            pointing_table=self.pointing_table,
                                                            str_dataset=self.str_dataset,
                                                            ndigits=self.ndigits)

            # OLD WAY TO BE REMOVED
            # # Filter the list with the pointing dictionary if given
            # if self.dict_exposures is not None:
            #     print_info("Will be using a dictionary for "
            #                      "further selecting the appropriate cubes")
            # list_cubes, dict_exposures_per_pointing, dict_tplexpo_per_pointing,\
            #     dict_tplexpo_per_dataset = filter_list_with_pdict(list_existing_cubes,
            #                                                       list_datasets=None,
            #                                                       dict_files=self.dict_exposures)

            # Take (or not) the fixed Cubes
            if self.use_fixed_cubes:
                print_warning("Using Corrected cubes with prefix {} "
                              "when relevant".format(self.prefix_fixed_cubes))
                prefix_to_consider = "{0}{1}".format(self.prefix_fixed_cubes,
                                                     self.prefix_cubes)
                list_fixed_cubes = glob.glob("{0}{1}*fits".format(self.folder_cubes,
                                                                  prefix_to_consider))
                print_info("Initial set of {:02d} Corrected "
                           "cubes found".format(len(list_fixed_cubes)))

                # if the list of exclusion suffix is empty, just use all cubes
                list_fixed_cubes = filter_list_with_suffix_list(list_fixed_cubes,
                                                                 self.included_suffix,
                                                                 self.excluded_suffix,
                                                                 name_list="Fixed Cubes")

                # Looping over the existing fixed pixtables
                for fixed_cube in list_fixed_cubes:
                    # Finding the name of the original one
                    orig_cube = fixed_cube.replace(prefix_to_consider,
                                                   self.prefix_cubes)
                    temp_list = copy.copy(list_cubes)
                    if orig_cube in temp_list:
                        # If it exists, remove it
                        list_cubes.remove(orig_cube)
                        print_warning(f"Cube {orig_cube} was removed from "
                                            f"the list (fixed one will be used)")
                    else:
                        print_warning(f"Original Cube {orig_cube} not "
                                            f"found (but fixed cube will "
                                            f"be used nevertheless)")
                    # and add the fixed one
                    list_cubes.append(fixed_cube)
                    print_warning(f"Fixed Cube {fixed_cube} has been included "
                                        f"in the list")

            # if the list of suffix is empty, just use all cubes
            if len(self.list_suffix) > 0:
                # Filtering out the ones that don't have any of the suffixes
                temp_list_cubes = []
                for l in list_cubes:
                    if any([suff in l for suff in self.list_suffix]):
                        temp_list_cubes.append(l)
                list_cubes = temp_list_cubes

        # Attach the other properties to the list of cubes (e.g. PSF)
        self.list_cubes = []
        # Names for the OBs
        str_dataset = kwargs.pop("str_dataset", default_str_dataset)
        ndigits = kwargs.pop("ndigits", default_ndigits)
        for name in list_cubes:
            self.list_cubes.append(BasicFile(name))
            if len(self.dict_psf) > 0:
                found = False
                for key in self.dict_psf:
                    if found:
                        break
                    keyword = f"{self.prefix_cubes}_" \
                              f"{get_dataset_name(int(key), str_dataset, ndigits)}"
                    if keyword in name:
                        psf = self.dict_psf[key]
                        self.list_cubes[-1].psf = BasicPSF(psf_array=psf)
                        found = True
                # If none correspond, set to the 0 FWHM Gaussian
                if not found:
                    print_warning(f"No PSF found for cube {name}. Using default")
                    self.list_cubes[-1].psf = BasicPSF()
            else:
                self.list_cubes[-1].psf = BasicPSF()

        if self.verbose:
            for i, c in enumerate(self.list_cubes):
                print_info("Cube {0:03d}: {1}".format(i+1, c.filename))

        if self.ncubes == 0:
            print_error(f"Found 0 cubes in this folder with suffix {self.prefix_cubes}: "
                        "please change suffix")
        else:
            print_info(f"Found {self.ncubes} cubes to be processed")

    def convolve_cubes(self, target_fwhm, target_nmoffat=None,
                        target_function="gaussian", suffix="conv", **kwargs):
        """

        Args:
            target_fwhm:
            target_nmoffat:
            input_function:
            target_function:
            suffix:
            **kwargs:

        Returns:

        """
        # Convolving cube per cube
        for i, c in enumerate(self.list_cubes):
            # Removing the input folder
            folder, filename = os.path.split(c.filename)
            # Getting just the name and the extension
            name, extension = os.path.splitext(filename)
            outcube_name = f"{name}_{suffix}{extension}"
            cube = MuseCube(filename=c.filename, psf_array=c.psf.psf_array)
            cube_folder, outcube_name = cube.convolve_cube_to_psf(target_fwhm,
                                      target_nmoffat=target_nmoffat,
                                      target_function=target_function,
                                      outcube_name=outcube_name,
                                      outcube_folder=folder,
                                      **kwargs)

            # updating the convolved cube name
            psf = BasicPSF(function=target_function, fwhm0=target_fwhm,
                           nmoffat=target_nmoffat, l0=c.psf.l0, b=0.)
            self.list_cubes[i] = BasicFile(joinpath(cube_folder,outcube_name),
                                           psf=psf)

    def madcombine(self, folder_cubes=None, outcube_name="dummy.fits",
                   fakemode=False, mad=True):
        """Combine the CubeMosaic and write it out.

        Args:
            folder_cubes (str): name of the folder for the cube [None]
            outcube_name (str): name of the outcube
            mad (bool): using mad or not [True]

        Creates:
            A new cube, combination of all input cubes listes in CubeMosaic
        """

        # Combine
        print_info("Starting the combine of "
                         "all {} input cubes".format(self.ncubes))
        if not fakemode:
            cube, expmap, statpix, rejmap = self.pycombine(mad=mad)

        # Saving
        if folder_cubes is not None:
            self.folder_cubes = folder_cubes
            if not self._check_folder():
                return

        full_cube_name = joinpath(self.folder_cubes, outcube_name)
        self.mosaic_cube_name = full_cube_name
        if not fakemode:
            print_info("Writing the new Cube {}".format(full_cube_name))
            cube.write(full_cube_name)

class MuseCube(Cube):
    """Wrapper around the mpdaf Cube functionalities
    """
    
    def __init__(self, source=None, verbose=False, **kwargs) :
        """Initialisation of the opening of a Cube
        """
        # PSF for that Cube
        psf_array = kwargs.pop("psf_array", None)
        self.psf = BasicPSF(psf_array=psf_array)

        if source is not None:
            self.__dict__.update(source.__dict__)
        else :
            Cube.__init__(self, **kwargs)

        self.verbose = verbose
        self._debug = kwargs.pop("debug", False)

    def get_spectrum_from_cube(self, nx=None, ny=None, pixel_window=0, title="Spectrum"):
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

    def extract_onespectral_cube(self, wave1=default_wave_wcs, outcube_name=None, **kwargs):
        """Create a single pixel cube extracted from this one.

        Parameters
        ----------
        wave1 : float
            Value of the wavelength to extract. In Angstroems.
        outcube_name : str
            Name of the output cube
        prefix : str
            If outcube_name is None (default), use that prefix to append
            in front of the input cube name (same folder)

        Returns
        -------
        A new cube with only 2 lambda. To be used as a WCS reference for
        masks.
        """

        # Find the wavelength
        k1 = self.wave.pixel([wave1], nearest=True)[0]

        # extracting the cube
        subcube = self[k1:k1+2, :, :]

        cube_folder, cube_name = os.path.split(self.filename)

        if outcube_name is None:
            prefix = kwargs.pop("prefix", "l{0:.0f}_".format(wave1))
            outcube_name = "{0}{1}".format(prefix, cube_name)

        print_info("Writing up single wave-cube {0}\n"
                         "in folder {1}".format(outcube_name, cube_folder))
        subcube.write(joinpath(cube_folder, outcube_name))
        return cube_folder, outcube_name

    def rebin_spatial(self, factor, mean=False, inplace=False, full_covariance=False, **kwargs):
        """Combine neighboring pixels to reduce the size of a cube by integer factors along each axis.

        Each output pixel is the mean of n pixels, where n is the product of the 
        reduction factors in the factor argument.
        Uses mpdaf rebin function, but add a normalisation factor if mean=False (sum).
        It also updates the unit by just copying the old one.

        Input
        -----
        factor :  (int or (int,int))
            Factor by which the spatial dimensions are reduced
        mean : bool
            If True, taking the mean, if False (default) summing
        inplace : bool
            If False (default) making a copy. Otherwise using the present cube.
        full_covariance: bool
            If True, will assume that spaxels are fully covariant. This means that
            the variance will be normalised by sqrt(N) where N is the number of 
            summed spaxels. Default is False

        Returns
        -------
        Cube: rebinned cube
        """
        # We copy it to keep the unit :-(
        res = self if inplace else self.copy()

        # Use the same reduction factor for all dimensions?
        # Copy from the mpdaf _rebin (in data.py)
        nfacarr = np.ones((res.ndim), dtype=int)
        if isinstance(factor, int):
            nfacarr[1:] *= factor
        elif len(factor) == 2:
            nfacarr[1:] = np.asarray(factor)
        elif len(factor) == 3:
            nfacarr = np.asarray(factor)
        else:
            print("Factor should be an integer or list/array of 2 or 3 integers")
        spafactor = nfacarr[1] * nfacarr[2]
       
        # Do the rebin using the rebin method from mpdaf Cube
        res = res.rebin(nfacarr, inplace=True, **kwargs)

        # Mean or Sum
        if mean:
            norm_factor = 1.
        else:
            norm_factor = spafactor

        # Now scaling the data and variances
        res._data *= norm_factor
        res._var *= norm_factor**2

        # If full covariance, the S/N does not change.
        # Hence by summing ns spaxels, the S/N should in principle increase by sqrt(ns)
        # 1- If not mean, it recovers the right factor for the surface
        #    Namely : data * norm_factor, var * norm_factor**2
        # 2- If full_covariance, the S/N does not change so we need to multiply the variance
        #    to compensate. Since the S/N increased by sqrt(ns), we multiply var * ns
        if full_covariance:
            print("Warning: Full Covariance option is ON")
            res._var *= spafactor

        return res

    def astropy_convolve(self, other, fft=True, inplace=False):
        """Convolve a DataArray with an array of the same number of dimensions
        using a specified convolution function.

        Copy of _convolve for a cube, but doing it per slice or not

        Masked values in self.data and self.var are replaced with
        zeros before the convolution is performed. However masked
        pixels in the input data remain masked in the output.

        Any variances in self.var are propagated correctly.

        If self.var exists, the variances are propagated using the equation::

            result.var = self.var (*) other**2

        where (*) indicates convolution. This equation can be derived
        by applying the usual rules of error-propagation to the
        discrete convolution equation.

        Uses `astropy.convolution.convolve_fft' or 'astropy.convolution.convolve'

        Parameters
        ----------
        fft : boolean
            The convolution function to use, chosen from:

            - `astropy.convolution.convolve_fft'
            - `astropy.convolution.convolve'

            In general convolve_fft() is faster than convolve() except when
            other.data only contains a few pixels. However convolve_fft uses
            a lot more memory than convolve(), so convolve() is sometimes the
            only reasonable choice. In particular, convolve_fft allocates two
            arrays whose dimensions are the sum of self.shape and other.shape,
            rounded up to a power of two. These arrays can be impractically
            large for some input data-sets.
        other : DataArray or numpy.ndarray
          The array with which to convolve the contents of self.  This must
          have the same number of dimensions as self, but it can have fewer
          elements. When this array contains a symmetric filtering function,
          the center of the function should be placed at the center of pixel,
          ``(other.shape - 1)//2``.

          Note that passing a DataArray object is equivalent to just
          passing its DataArray.data member. If it has any variances,
          these are ignored.
        inplace : bool
            If False (the default), return a new object containing the
            convolved array.
            If True, record the convolved array in self and return self.

        Returns
        -------
        `~mpdaf.obj.DataArray`

        """
        out = self if inplace else self.copy()

        if out.ndim != other.ndim:
            raise IOError('The other array must have the same rank as self')

        if np.any(np.asarray(other.shape) > np.asarray(self.shape)):
            raise IOError('The other array must be no larger than self')

        kernel = other.data if isinstance(other, DataArray) else other

        # Replace any masked pixels in the convolution kernel with zeros.
        if isinstance(kernel, ma.MaskedArray) and ma.count_masked(kernel) > 0:
            kernel = kernel.filled(0.0)

        # Replace any masked pixels in out._data with zeros
        masked = self._mask is not None and self._mask.sum() > 0
        if masked:
            out._data = out.data.filled(0.0)
        elif out._mask is None and ~np.isfinite(out._data.sum()):
            out._data = out._data.copy()
            out._data[~np.isfinite(out._data)] = 0.0

        # Are there any variances to be propagated?
        if out._var is not None:
            # Replace any masked pixels in out._var with zeros
            if masked:
                out._var = out.var.filled(0.0)
            elif out._mask is None and ~np.isfinite(out._var.sum()):
                out._var = out._var.copy()
                out._var[~np.isfinite(out._var)] = 0.0

        # Calling the external function now
        out._data, out._var = cube_convolve(out._data, kernel,
                                            variance=out._var, fft=fft)
        # Put back nan in the data and var
        if masked:
            out._data[out._mask] = np.nan
            out._var[out._mask] = np.nan

        return out

    def convolve_cube_to_psf(self, target_fwhm, target_nmoffat=None,
                             target_function="gaussian",
                             outcube_folder=None,
                             outcube_name=None, factor_fwhm=3,
                             fft=True, erode_edges=True, npixels_erosion=2):
        """Convolve the cube for a target function 'gaussian' or 'moffat'

        Args:
            target_fwhm (float): target FWHM in arcsecond
            target_nmoffat: target n if Moffat function
            target_function (str): 'gaussian' or 'moffat' ['gaussian']
            factor_fwhm (float): number of FWHM for size of Kernel
            fft (bool): use FFT to convolve or not [False]
            perslice (bool): doing it per slice, or not [True]
                If doing it per slice, using a direct astropy fft. If
                doing it with the cube, it uses much more memory but is
                more efficient as the convolution is done via mpdaf directly.

        Creates:
            Folder and convolved cube names
        """
        # Separate folder and name of file
        cube_folder, cube_name = os.path.split(self.filename)
        if outcube_folder is None:
            outcube_folder = cube_folder

        # Creating the outcube filename
        if outcube_name is None:
            outcube_name = "conv{0}_{1:.2f}{2}".format(target_function.lower()[0],
                                                       target_fwhm, cube_name)
        else:
            _, outcube_name = os.path.split(outcube_name)
        print_info(f"The new cube will be named: {outcube_name}")
        print_info(f"Products will be written in {outcube_folder}")

        # Getting the shape of the Kernel
        scale_spaxel = self.get_step(unit_wcs=u.arcsec)[1]
        nspaxel = int(factor_fwhm * target_fwhm / scale_spaxel)
        # Make nspaxel odd to have a PSF centred at the centre of the frame
        if nspaxel % 2 == 0:
            nspaxel += 1
        shape = [self.shape[0], nspaxel, nspaxel]

        # Computing the kernel
        kernel3d = cube_kernel(shape, self.wave.coord(), self.psf.fwhm0,
                               target_fwhm, self.psf.function, target_function,
                               lambda0=self.psf.l0,
                               input_nmoffat=self.psf.nmoffat,
                               target_nmoffat=target_nmoffat, b=self.psf.b,
                               scale=scale_spaxel, compute_kernel='pypher')

        # Calling the local method using astropy convolution
        conv_cube = self.astropy_convolve(other=kernel3d, fft=fft)
        # Copying the unit from the original to the convolved cube
        conv_cube.unit = copy.copy(self.unit)

        # Erode by npixels in case erode is True
        if erode_edges:
            nmask = ~ndi.binary_erosion(~np.isnan(conv_cube._data), 
                                        mask=~conv_cube._mask, 
                                        iterations=npixels_erosion)
            conv_cube._data[nmask] = np.nan 
            conv_cube._var[nmask] = np.nan 
            conv_cube._mask = nmask

        # Write the output
        print_info("Writing up the derived cube")
        conv_cube.write(joinpath(outcube_folder, outcube_name))

        # Write the kernel3D
        print_info("Writing up the used kernel")
        kercube = Cube(data=kernel3d)
        kercube.write(joinpath(outcube_folder, "ker3d_{}".format(outcube_name)))

        # just provide the output name by folder+name
        return outcube_folder, outcube_name

    def create_reference_cube(self, lambdamin=4700, lambdamax=9400,
            step=1.25, outcube_name=None, filter_for_nan=False, **kwargs):
        """Create a reference cube using an input one, and overiding
        the lambda part, to get a new WCS

        Args:
            lambdamin:
            lambdamax:
            step:
            outcube_name:
            filter_for_nan:
            **kwargs:

        Returns:
            cube_folder, outcube_name (str, str): the name of the folder where
               the output cube is, and its name

        """

        # Separate folder and name of file
        cube_folder, cube_name = os.path.split(self.filename)

        # Creating the outcube filename
        if outcube_name is None:
            prefix = kwargs.pop("prefix", "l{0:4d}l{1:4d}_".format(
                int(lambdamin), int(lambdamax)))
            outcube_name = "{0}{1}".format(prefix, cube_name)

        # Range of lambd and number of spectral pixels
        range_lambda = lambdamax - lambdamin
        npix_spec = int(range_lambda // step + 1.0)

        # if filter_nan remove the Nan
        if filter_for_nan:
            ind = np.indices(self.data[0].shape)
            selgood = np.any(~np.isnan(self.data), axis=0)
            if self._debug:
                print_debug("Xmin={0} Xmax={1} / Ymin={2} Ymax={3}".format(
                                  np.min(ind[0][selgood]), np.max(ind[0][selgood]),
                                  np.min(ind[1][selgood]), np.max(ind[1][selgood])))
            subcube = self[:,np.min(ind[0][selgood]): np.max(ind[0][selgood]),
                               np.min(ind[1][selgood]): np.max(ind[1][selgood])]
        else:
            subcube = self

        # Create the WCS which we need for the output cube
        wcs_header = subcube.get_wcs_header()
        wcs1 = subcube.wcs
        wave1 = WaveCoord(cdelt=step, crval=lambdamin, 
                ctype=wcs_header['CTYPE3'], crpix=1.0, shape=npix_spec)
        # Create a fake dataset with int to be faster
        cube_data = np.ones((npix_spec, wcs1.naxis2, wcs1.naxis1), dtype=np.uint8)
        cube = Cube(data=cube_data, wcs=wcs1, wave=wave1)
        # Write the output
        cube.write(joinpath(cube_folder, outcube_name))

        # just provide the output name by folder+name
        return cube_folder, outcube_name

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

        [lmin, lmax] = get_emissionline_band(line=line, velocity=velocity,
                redshift=redshift, medium=medium, lambda_window=lambda_window)
        
        return MuseImage(self.select_lambda(lmin, lmax).sum(axis=0), 
                title="{0} map".format(line))

    def build_filterlist_images(self, filter_list, prefix="IMAGE_FOV",
                              suffix="", folder=None, **kwargs):
        """

        Args:
            filter_list:
            prefix:
            suffix:
            folder:
            **kwargs:

        Returns:

        """
        print_info("Building images for each filter in given list")
        cube_folder, cube_name = os.path.split(self.filename)
        if folder is None:
            folder = cube_folder

        suffix = add_string(suffix)

        filter_list = check_filter_list(filter_list)
        for filtername in filter_list:
            print_info(f"Filter = {filtername}")
            ima = self.get_filter_image(filter_name=filtername, **kwargs)
            if ima is None:
                print_error(f"Could not reconstruct Image with Filter {filtername}")
                continue

            ima_name = f"{prefix}_{filtername}{suffix}.fits"
            print_info(f"Writing image {ima_name}")
            ima.write(joinpath(folder, ima_name))

    def get_filter_image(self, filter_name=None, own_filter_file=None, filter_folder="",
            dict_filters=None):
        """Get an image given by a filter. If the filter belongs to
        the filter list, then use that, otherwise use the given file
        """
        try:
            print_info("Building Image with MUSE "
                             "filter {0}".format(filter_name))
            if filter_name.lower() == "white":
                # filter_wave = np.array([4000.,5000.,6000.,7000.,
                #                         8000.,9000.,10000.])
                # filter_sensitivity = np.ones_like(filter_wave)
                # refimage = self.bandpass_image(filter_wave, filter_sensitivity,
                #                            interpolation='linear')
                refimage = self.sum(axis=0)
            else:
                refimage = self.get_band_image(filter_name)
        except ValueError:
            # initialise the filter file
            print_info("Reading private reference filter {0}".format(filter_name))

            # First we check the extra dictionary if provided
            if dict_filters is not None:
                if filter_name in dict_filters:
                    filter_file = dict_filters[filter_name]
            # then we check the package internal filter list
            elif filter_name in dict_extra_filters:
                print_info("Found filter in pymusepipe internal dictionary "
                                 "(see data/Filters)")
                filter_folder = pymusepipe.__path__[0]
                filter_file = dict_extra_filters[filter_name]
            else:
                print_warning("[mpdaf_pipe / get_filter_image] "
                                  "Filter name not in private dictionary - Aborting")
                if own_filter_file is None:
                    print_error("[mpdaf_pipe / get_filter_image] "
                                      "No extra filter dictionary and "
                                      "the private filter file is not set - Aborting")
                    return None
                else:
                    filter_file = own_filter_file
            # Now reading the filter data
            path_filter = joinpath(filter_folder, filter_file)
            filter_wave, filter_sensitivity = np.loadtxt(path_filter, unpack=True)
            print_info("Building Image with filter {}".format(filter_name))
            # using mpdaf bandpass_image to build the image
            refimage = self.bandpass_image(filter_wave, filter_sensitivity,
                                           interpolation='linear')

            # Adding FILTER NAME to the primary header
            key = 'HIERARCH ESO DRS MUSE FILTER NAME'
            refimage.primary_header[key] = (filter_name, 'filter name used')

            # Copying the unit from the cube
            refimage.unit = copy.copy(self.unit)

            add_mpdaf_method_keywords(refimage.primary_header, 
                    "cube.bandpass_image", ['name'], 
                    [filter_name], ['filter name used'])

        return refimage

    def mask_trail(self, pq1=[0, 0], pq2=[10, 10], width=1.0,
                   margins=0., reset=False, save=True, **kwargs):
        """Build a cube mask from 2 points measured from a trail on an image

        Input
        -----
        pq1: array or tuple (float)
            p and q coordinates of point 1 along the trail
        pq2: array or tuple (float)
            p and q coordinates of point 2 along the trail
        width: float
            Value (in pixel) of the full slit width to exclude
        margins: float
            Value (in pixel) to extend the slit beyond the 2 extrema
            If 0, this means limiting it to the extrema themselves.
            Default is None, which mean infinitely long slit
        reset (bool): if True, reset the mask before masking the slit
        save (bool): if True, save the masked cube
        """
        # If width = 0 just don't do anything
        if width <= 0.:
            print_warning("Trail width is 0, hence no doing anything")
            return
        w2 = width / 2.

        # Calculating the coordinates of the p and q's
        # X2 - X1 -> note that X is q, hence the second value
        # Y2 - Y1 -> note that Y is p, hence the first value
        pq1 = np.asarray(pq1)
        pq2 = np.asarray(pq2)
        vect = pq2 - pq1
        nvect = vect / np.hypot(vect[0], vect[1])
        ovect = np.array([nvect[1], -nvect[0]])

        # New corners depending on width and margins
        corner1 = pq1 - margins * nvect + w2 * ovect
        corner2 = pq1 - margins * nvect - w2 * ovect
        corner3 = pq2 + margins * nvect - w2 * ovect
        corner4 = pq2 + margins * nvect + w2 * ovect
        # This is the new tuple to feed mask_polygon
        pol = [tuple(corner1), tuple(corner2), tuple(corner3), tuple(corner4)]

        out = self.copy()
        # Reset mask if requested
        if reset:
            out.unmask()

        out.mask_polygon(pol, unit_poly=None, inside=True)
        out.data[out.mask] = np.nan
        out.var[out.mask] = np.nan

        # Rewrite a new cube
        if save:
            prefix = kwargs.pop("prefix", "tmask")
            use_folder = kwargs.pop("use_folder", True)

            cube_folder, cube_name = os.path.split(self.filename)
            trailcube_name = "{0}{1}".format(prefix, cube_name)
            if use_folder:
                trailcube_name = joinpath(cube_folder, trailcube_name)

            print_info("Writing the new Cube {}".format(trailcube_name))
            out.write(trailcube_name)

    def save_mask(self, mask_name="dummy_mask.fits"):
        """Save the mask into a 0-1 image
        """
        newimage = self.copy()
        newimage.data = np.where(self.mask, 0, 1).astype(int)
        newimage.mask[:, :] = False
        newimage.write(mask_name)


class MuseSkyContinuum(object):
    def __init__(self, filename):
        self.filename = filename
        self.read()

    def read(self):
        """Read sky continuum spectrum from MUSE data reduction
        """
        if not os.path.isfile(self.filename):
            print_error("{0} not found".format(self.filename))
            crval = 0.0
            data = np.zeros(0)
            cdelt = 1.0
        else:
            sky = pyfits.getdata(self.filename)
            crval = sky['lambda'][0]
            cdelt = sky['lambda'][1] - crval
            data = sky['flux']
        wavein = WaveCoord(cdelt=cdelt, crval=crval, cunit=u.angstrom)
        self.spec = Spectrum(wave=wavein, data=data)

    def integrate(self, muse_filter, ao_mask=False):
        """Integrate a sky continuum spectrum using a certain filter file.
        If the file is a fits file, use it as the MUSE filter list.
        Otherwise use it as an ascii file
    
        Input
        -----
        muse_filter: MuseFilter
        """
        # interpolation linearly the filter throughput onto
        # the spectrum wavelength
        muse_filter.flux_cont = integrate_spectrum(self.spec, 
                                                   muse_filter.wave, 
                                                   muse_filter.throughput,
                                                   ao_mask=ao_mask)
        setattr(self, muse_filter.filter_name, muse_filter)

    def set_normfactor(self, background, filter_name="Cousins_R"):
        """Get the normalisation factor given a background value
        Takes the background value and the sky continuuum spectrum
        and convert this to the scaling Ks needed for this sky continuum
        The principle relies on having the background measured as:
        MUSE_calib = ((MUSE - Sky_cont) + Background) * Norm

        as measured from the alignment procedure.

        Since we want:
        MUSE_calib = ((MUSE - Ks * Sky_cont) + 0) * Norm

        This means that: Ks * Sky_cont = Sky_cont - Background
        ==> Ks = 1 - Background / Sky_cont

        So we integrate the Sky_cont to get the corresponding S value
        and then provide Ks as 1-B/S

        Input
        -----
        background: float
            Value of the background to consider
        filter_name: str
            Name of the filter to consider
        """
        if not hasattr(self, filter_name):
            print_error("No integration value for filter {0}".format(
                                  filter_name))
            self._norm = 1.
        else :
            muse_filter = getattr(self, filter_name)
            if muse_filter.flux_cont != 0.:
                norm = 1. - background / muse_filter.flux_cont
            else:
                norm = 1.

            # Saving the norm value for that filter
            muse_filter.norm = norm
            muse_filter.background = background

    def save_normalised(self, norm_factor=1.0, prefix="norm", overwrite=False):
        """Normalises a sky continuum spectrum and save it
        within a new fits file
    
        Input
        -----
        norm_factor: float
            Scale factor to multiply the input continuum
        prefix: str
            Prefix for the new continuum fits name. Default
            is 'norm', so that the new file is 'norm_oldname.fits'
        overwrite: bool
            If True, existing file will be overwritten.
            Default is False.
        """
        if prefix == "":
            print_error("[mpdaf_pipe / save_normalised] The new and old sky "
                              "continuum fits files will share the same name")
            print_error("This is not recommended - Aborting")
            return
    
        folder_spec, filename = os.path.split(self.filename)
        newfilename = "{0}{1}".format(prefix, filename)
        norm_filename = joinpath(folder_spec, newfilename)
    
        # Opening the fits file
        skycont = pyfits.open(self.filename)
    
        # getting the data
        dcont = skycont['CONTINUUM'].data
    
        # Create new continuum
        # ------------------------------
        new_cont = dcont['flux'] * norm_factor
        skycont['CONTINUUM'].data['flux'] = new_cont
    
        # Writing to the new file
        skycont.writeto(norm_filename, overwrite=overwrite)
        print_info('Normalised Factor used = {0:8.4f}'.format(norm_factor))
        print_info('Normalised Sky Continuum {} has been created'.format(norm_filename))

# Routine to read filters
class MuseFilter(object):
    def __init__(self, filter_name="Cousins_R", filter_fits_file="filter_list.fits",
                 filter_ascii_file=None):
        """Routine to read the throughput of a filter

        Input
        -----
        filter_name: str
            Name of the filter, required if filter_file is a fits file.
        filter_fits_file: str
            Name of the fits file for the filter list
        filter_ascii_file: str
            Name of the ascii file with the lambda, throughout values
            By default, it is None and ignored. If not, this is taken
            as the input for the filter throughput
        """
        self.filter_fits_file = filter_fits_file
        self.filter_name = filter_name
        self.filter_ascii_file = filter_ascii_file
        self.read()

    def read(self):
        """Reading the data in the file
        """
        if self.filter_ascii_file is None:
            try:
                print_info("Using the fits file {0} as input".format(self.filter_fits_file))
                filter_data = pyfits.getdata(self.filter_fits_file, extname=self.filter_name)
                self.wave = filter_data['lambda']
                self.throughput = filter_data['throughput']
            except:
                print_error("Problem opening the filter fits file {0}".format(self.filter_fits_file))
                print_error("Did not manage to get the filter {0} throughput".format(self.filter_name))
                self.wave = np.zeros(0)
                self.throughput = np.zeros(0)
        else:
            print_info("Using the ascii file {0} as input".format(self.filter_ascii_file))
            self.wave, self.throughput = np.loadtxt(self.filter_ascii_file, unpack=True)

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
        self.vmin = int(kwargs.pop('vmin', 0))
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
            print_warning("Trail width is 0, hence no doing anything")
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
        newimage.data = np.where(self.mask, 0, 1).astype(int)
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
                print_warning("Overiding subtitle")
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
                print_warning("Overiding subtitle")
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
            print_error("Input PixTable does not exist")
            return
        if not os.path.isfile(image_name):
            print_error("Input Image does not exist")
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
        print_info("Opening the Pixtable {0}".format(
                          self.pixtable_name))
        pixtable = PixTable(self.pixtable_name)

        # Use the Image mask and create a pixtable mask
        if mask_name is not None:
            self.mask_name = mask_name
        else:
            if not hasattr(self, "mask_name"):
                print_error("Please provide a mask name (FITS file)")
                return

        print_info("Creating a column Mask from file {0}".format(
                          self.mask_name))
        mask_col = pixtable.mask_column(self.mask_name)

        # extract the right data using the pixtable mask
        print_info("Extracting the Mask")
        newpixtable = pixtable.extract_from_mask(mask_col.maskcol)

        # Rewrite a new pixtable
        self.suffix_out = kwargs.pop("suffix_out", self.suffix_out)
        use_folder = kwargs.pop("use_folder", True)
        if use_folder:
            self.newpixtable_name = joinpath(self.pixtable_folder, "{0}{1}".format(
                                    self.suffix_out, self.pixtable_name))
        else :
            self.newpixtable_name = "{0}{1}".format(self.suffix_out, self.pixtable_name)

        print_info("Writing the new PixTable in {0}".format(
                          self.newpixtable_name))
        newpixtable.write(self.newpixtable_name)

        # Now transfer the flat field if it exists
        ext_name = 'PIXTABLE_FLAT_FIELD'
        try:
            # Test if Extension exists by reading header
            # If it exists then do nothing
            test_data = pyfits.getheader(self.newpixtable_name, ext_name)
            print_warning("Flat field extension already exists in masked PixTable - all good")
        # If it does not exist test if it exists in the original PixTable
        except KeyError:
            try:
                # Read data and header
                ff_ext_data = pyfits.getdata(self.pixtable_name, ext_name)
                ff_ext_h = pyfits.getheader(self.pixtable_name, ext_name)
                print_warning("Flat field extension will be transferred from PixTable")
                # Append it to the new pixtable
                pyfits.append(self.newpixtable_name, ff_ext_data, ff_ext_h)
            except KeyError:
                print_warning("No Flat field extension to transfer - all good")
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
                print_warning("Rewriting extension name {0} as lowercase".format(
                                       d.upper()))
            except:
                print_warning("Extension {0} not present - patch ignored".format(
                                       d.upper()))
