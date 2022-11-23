# Licensed under a MIT license - see LICENSE

"""MUSE-PHANGS utility functions for pymusepipe
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"

# Importing modules
import os
import time
from os.path import join as joinpath
import copy
from collections import OrderedDict
import re

# Numpy
import numpy as np
from scipy.odr import ODR, Model, RealData
from scipy import ndimage as nd

from astropy import constants as const
from astropy.io import fits as pyfits
from astropy.stats import mad_std, sigma_clip
from astropy.convolution import Gaussian2DKernel, convolve

# Import package modules
from .emission_lines import list_emission_lines, full_muse_wavelength_range
from .config_pipe import (default_filter_list, dict_musemodes,
                          default_ndigits, default_str_dataset)
from . import util_pipe as upipe

# MPDAF
from mpdaf.obj import Image, Cube


#  PRINTING FUNCTIONS #
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[0;31;20m'
INFO = '\033[0;32;20m'
ERROR = '\033[1;91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
DEBUG = '\033[1m'

def print_endline(text, **kwargs):
    print(INFO + text + ENDC, **kwargs)

def print_warning(text, **kwargs):
    toprint = "# MusePipeWarning " + text
    mypipe = kwargs.pop("pipe", None)
    try:
        mypipe.write_logfile(toprint)
    except:
        pass
    try:
        verbose = mypipe.verbose
    except:
        verbose = kwargs.pop("verbose", True)
    
    if verbose:
        print(WARNING + "# MusePipeWarning " + ENDC + text, **kwargs)

def print_info(text, **kwargs):
    """Print processing information

    Input
    -----
    text: str
    pipe: musepipe [None]
        If provided, will print the text in the logfile
    """
    toprint = "# MusePipeInfo " + text
    mypipe = kwargs.pop("pipe", None)
    try:
        mypipe.write_logfile(toprint)
    except:
        pass
    try:
        verbose = mypipe.verbose
    except:
        verbose = kwargs.pop("verbose", True)
    
    if verbose:
        print(INFO + "# MusePipeInfo " + ENDC + text, **kwargs)

def print_debug(text, **kwargs) :
    """Print debugging information

    Input
    -----
    text: str
    pipe: musepipe [None]
        If provided, will print the text in the logfile
    """
    mypipe = kwargs.pop("pipe", None)
    try:
        verbose = mypipe.verbose
    except:
        verbose = kwargs.pop("verbose", True)
    
    if verbose:
        print(DEBUG + "# DebugInfo " + ENDC + text, **kwargs)

def print_error(text, **kwargs):
    """Print error information

    Input
    -----
    text: str
    pipe: musepipe [None]
        If provided, will print the text in the logfile
    """
    toprint = "# MusePipeError " + text
    mypipe = kwargs.pop("pipe", None)
    try:
        mypipe.write_logfile(toprint)
    except:
        pass
    try:
        verbose = mypipe.verbose
    except:
        verbose = kwargs.pop("verbose", True)
    
    if verbose:
        print(ERROR + "# MusePipeError " + ENDC + text, **kwargs)

#-----------  END PRINTING FUNCTIONS -----------------------

def analyse_musemode(musemode, field, delimiter='-'):
    """Extract the named field from the musemode

    Input
    -----
    musemode: str
        Mode of the MUSE data to be analysed
    field: str
        Field to analyse ('ao', 'field', 'lambda_range')
    delimiter: str
        Character to delimit the fields to analyse

    Returns
    -------
    val: str
        Value of the field which was analysed (e.g., 'AO' or 'NOAO')
    """
    if field not in dict_musemodes:
        upipe.print_error(f"Cannot find such a field ({field}) in the dict_musemodes")
        return ""

    index = dict_musemodes[field]
    sval = musemode.split(delimiter)

    if len(sval) < index+1:
        upipe.print_error(f"Error in analyse_musemode. Cannot access field {index} "
                          f"After splitting the musemode {musemode} = sval")
        val = ""
    else:
        val = musemode.split(delimiter)[index]
    return val.lower()

def add_string(text, word="_", loc=0):
    """Adding string at given location
    Default is underscore for string which are not empty.

    Input
    ----
    text (str): input text
    word (str): input word to be added
    loc (int): location in 'text'. [Default is 0=start]
               If None, will be added at the end.

    Returns
    ------
    Updated text
    """
    if len(text) > 0:
        if loc is None:
            text = f"{text}{word}"
        else:
            try:
                if text[loc] != "_":
                    text = f"{text[:loc]}{word}{text[loc:]}"

            except:
                print(f"String index [{loc}] out of range [{len(text)}] in add_string")

    return text

def get_tpl_nexpo(filename):
    """Get the tpl and nexpo from a filename assuming it is at the end
    of the filename

    Input
    -----
    filename: str
       Input filename

    Returns
    -------
    tpl, nexpo: str, int
    """
    basestr, ext = os.path.splitext(filename)
    try:
        [(tpl, nexpo)] = re.findall(r'\_(\S{19})\_(\d{4})', basestr)
        if len(nexpo) > 0:
            return tpl, int(nexpo)
        else:
            return "", -1
    except ValueError:
        return "", -1

def get_dataset_name(dataset=1, str_dataset=default_str_dataset, ndigits=default_ndigits):
    """Formatting for the dataset/pointing names using the number and
    the number of digits and prefix string

    Input
    -----
    dataset: int
       Dataset (or Pointing) number
    str_dataset: str
        Prefix representing the dataset (or pointing)
    ndigits: int
        Number of digits to be used for formatting

    Returns
    -------
    string for the dataset/pointing name prefix
    """
    return f"{str_dataset}{int(dataset):0{int(ndigits)}}"

def lower_rep(text):
    """Lower the text and return it after removing all underscores

    Args:
        text (str): text to treat

    Returns:
        updated text (with removed underscores and lower-cased)

    """
    return text.replace("_", "").lower()

def lower_allbutfirst_letter(mystring):
    """Lowercase all letters except the first one
    """
    return mystring[0].upper() + mystring[1:].lower()

class TimeStampDict(OrderedDict):
    """Class which builds a time stamp driven
    dictionary of objects
    """
    def __init__(self, description="", myobject=None):
        """Initialise an empty dictionary
        with a given name
        """
        OrderedDict.__init__(self)
        self.description = description
        self.create_new_timestamp(myobject)

    def create_new_timestamp(self, myobject=None):
        """Create a new item in dictionary
        using a time stamp
        """
        if myobject is not None:
            self.present_tstamp = create_time_name()
            self[self.present_tstamp] = myobject
        else:
            self.present_stamp = None

    def delete_timestamp(self, tstamp=None):
        """Delete a key in the dictionary
        """
        _ = self.pop(tstamp)

def merge_dict(dict1, dict2):
    """Merging two dictionaries by appending
    keys which are duplicated

    Input
    -----
    dict1: dict
    dict2: dict

    Returns
    -------
    dict1 : dict
        merged dictionary
    """
    for key, value in dict2.items():
        if key in dict1:
            dict1[key].extend(value)
        else:
            dict1[key] = value
    return dict1

def create_time_name() :
    """Create a time-link name for file saving purposes

    Return: a string including the YearMonthDay_HourMinSec
    """
    return str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))

def formatted_time() :
    """ Return: a string including the formatted time
    """
    return str(time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()))

def safely_create_folder(path, verbose=True):
    """Create a folder given by the input path
    This small function tries to create it and if it fails
    it checks whether the reason is because it is not a path
    and then warn the user
    and then warn the user
    """
    if path is None :
        if verbose : print_info("Input path is None, not doing anything")
        return
    if verbose : 
        print_info("Trying to create {folder} folder".format(folder=path), end='')
    try: 
        os.makedirs(path)
        if verbose:
            print_endline("... Done", end='\n')
    except OSError:
        if not os.path.isdir(path):
            print_error("Failed to create folder! Please check the path")
            return
        if os.path.isdir(path):
            if verbose:
                print_endline("... Folder already exists, doing nothing.")

def append_file(filename, content):
    """Append in ascii file
    """
    with open(filename, "a") as myfile:
        myfile.write(content)
        
def abspath(path) :
    """Normalise the path to get it short but absolute
    """
    return os.path.abspath(os.path.realpath(path))

def normpath(path) :
    """Normalise the path to get it short
    """
    return os.path.normpath(os.path.realpath(path))

def doppler_shift(wavelength, velocity=0.):
    """Return the redshifted wavelength
    """
    doppler_factor = np.sqrt((1. + velocity / const.c.value) / (1. - velocity / const.c.value))
    return wavelength * doppler_factor

def get_emissionline_wavelength(line="Ha", velocity=0., redshift=None, medium='air'):
    """Get the wavelength of an emission line, including a correction
    for the redshift (or velocity)
    """
    index_line = {'vacuum': 0, 'air': 1}
    # Get the velocity
    if redshift is not None : velocity = redshift * const.c

    if line is None:
        return -1.
    elif line not in list_emission_lines:
        upipe.print_error("Could not guess the emission line you wish to use")
        upipe.print_error("Please review the 'emission_line' dictionary")
        return -1.

    if medium not in index_line:
        upipe.print_error(f"Please choose between one of these media: {list(index_line.keys())}")
        return -1.

    wavel = list_emission_lines[line][index_line[medium]]
    return doppler_shift(wavel, velocity)

def get_emissionline_band(line="Ha", velocity=0., redshift=None, medium='air', lambda_window=10.0):
    """Get the wavelengths of an emission line, including a correction
    for the redshift (or velocity) and a lambda_window around that line (in Angstroems)

    Parameters
    ----------
    line: name of the line (string). Default is 'Ha'
    velocity: shift in velocity (km/s)
    medium: 'air' or 'vacuum'
    lambda_window: lambda_window in Angstroem
    """
    red_wavel = get_emissionline_wavelength(line=line, velocity=velocity, redshift=redshift, medium=medium)
    # In case the line is not in the list, just return the full lambda Range
    if red_wavel < 0 :
        return full_muse_wavelength_range
    else:
        return [red_wavel - lambda_window/2., red_wavel + lambda_window/2.]

    
def select_spaxels(maskDic, maskName, X, Y) :
    """Selecting spaxels defined by their coordinates
    using the masks defined by Circle or Rectangle Zones
    """
    ## All spaxels are set to GOOD (True) first
    selgood = (X**2 >= 0)

    ## If no Mask is provided, we just return the full set of input X, Y
    if maskDic == None :
        return selgood

    ## We first check if the maskName is in the list of the defined Masks
    ## If the galaxy is not in the list, then the selection is all True
    if maskName in maskDic:
        ## The mask is defined, so Get the list of Regions
        ## From the defined dictionary
        listRegions = maskDic[maskName]
        ## For each region, select the good spaxels
        for region in  listRegions :
            selgood = selgood & region.select(X, Y)

    return selgood


class Selection_Zone :
    """
    Parent class for Rectangle_Zone and Circle_Zone

    Input
    -----
    params: list of floats
        List of parameters for the selection zone
    """
    def __init__(self, params=None) :
        self.params = params
        if self.params is None:
            print_error("Error: {0} Zone needs {1} input parameters - {2} given".format(
                            self.type, self.nparams, len(params)))


class Rectangle_Zone(Selection_Zone) :
    """Define a rectangular zone, given by 
    a center, a length, a width and an angle
    """
    def __init__(self):
        self.type = "Rectangle"
        self.nparams = 5
        Selection_Zone.__init__(self)

    def select(self, xin, yin) :
        """ Define a selection within a rectangle
            It can be rotated by an angle theta (in degrees) 
        Input
        -----
        xin, yin: 2d arrays
            Input positions for the spaxels
        """
        if self.params is None :
           return (xin**2 >=0)
        [x0, y0, length, width, theta] = self.params
        dx = xin - x0
        dy = yin - y0
        thetarad = np.deg2rad(theta)
        nx =   dx * np.cos(thetarad) + dy * np.sin(thetarad)
        ny = - dx * np.sin(thetarad) + dy * np.cos(thetarad)
        selgood = (np.abs(ny) > width / 2.) | (np.abs(nx) > length / 2.)
        return selgood

class Circle_Zone(Selection_Zone) :
    """Define a Circular zone, defined by 
    a center and a radius
    """
    def __init__(self):
        self.type = "Circle"
        self.nparams = 5
        Selection_Zone.__init__(self)

    def select(self, xin, yin) :
        """ Define a selection within a circle 

        Input
        -----
        xin, yin: 2d arrays
            Input positions for the spaxels
        """
        if self.params is None :
           return (xin**2 >=0)
        [x0, y0, radius] = self.params
        selgood = (np.sqrt((xin - x0)**2 + (yin - y0)**2) > radius)
        return selgood

class Trail_Zone(Selection_Zone) :
    """Define a Trail zone, defined by
    two points and a width
    """
    def __init__(self):
        self.type = "Trail"
        self.nparams = 5
        Selection_Zone.__init__(self)

    def select(self, xin, yin) :
        """ Define a selection within trail

        Input
        -----
        xin, yin: 2d arrays
            Input positions for the spaxels

        """
        if self.params is None :
           return (xin**2 >=0)
        [x0, y0, radius] = self.params
        selgood = (np.sqrt((xin - x0)**2 + (yin - y0)**2) > radius)
        return selgood

def reconstruct_filter_images(cubename, filter_list=default_filter_list,
        filter_fits_file="filter_list.fits"):
    """ Reconstruct all images in a list of Filters
    cubename: str
        Name of the cube
    filter_list: str
        List of filters, e.g., "Cousins_R,Johnson_I"
        By default, the default_filter_list from pymusepipe.config_pipe

    filter_fits_file: str
        Name of the fits file containing all the filter characteristics
        Usually in filter_list.fits (MUSE default)
    """
    
    command = "muse_cube_filter -f {0} {1} {2}".format(
                  filter_list, cubename, filter_fits_file)
    os.system(command)

def add_key_dataset_expo(imaname, iexpo, dataset):
    """Add dataset and expo number to image

    Input
    -----
    imaname: str
    iexpo: int
    dataset: int
    """
    # Writing the dataset and iexpo in the IMAGE_FOV
    this_image = pyfits.open(imaname, mode='update')
    this_image[0].header['MUSEPIPE_DATASET'] = (dataset, "Dataset number")
    this_image[0].header['MUSEPIPE_IEXPO'] = (iexpo, "Exposure number")
    this_image.flush()
    print_info("Keywords MUSEPIPE_DATASET/EXPO updated for image {}".format(
        imaname))

def rotate_image_wcs(ima_name, ima_folder="", outwcs_folder=None, rotangle=0.,
                     **kwargs):
    """Routine to remove potential Nan around an image and reconstruct
    an optimal WCS reference image. The rotation angle is provided as a way
    to optimise the extent of the output image, removing Nan along X and Y
    at that angle.

    Args:
        ima_name (str): input image name. No default.
        ima_folder (str): input image folder ['']
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
    upipe.print_info("Will use a {:5.2f}% extra margin".format(
                     extend_fraction*100.))

    # Opening the image via mpdaf
    imawcs = Image(fullname)
    extra_pixels = (np.array(imawcs.shape) * extend_fraction).astype(np.int)

    # New dimensions and extend current image
    new_dim = tuple(np.array(imawcs.shape).astype(np.int) + extra_pixels)
    ima_ext = imawcs.regrid(newdim=new_dim, refpos=imawcs.get_start(),
                            refpix=tuple(extra_pixels / 2.),
                            newinc=imawcs.get_step()[0]*3600.)

    # Copy and rotate WCS
    new_wcs = copy.deepcopy(ima_ext.wcs)
    upipe.print_info("Rotating WCS by {} degrees".format(rotangle))
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
    upipe.print_info("Will use a {:5.2f}% extra margin".format(
                     extend_fraction*100.))

    # Opening the image via mpdaf
    cubewcs = Cube(fullname)
    imawcs = cubewcs.sum(axis=0)
    extra_pixels = (np.array(imawcs.shape) * extend_fraction).astype(np.int)

    # New dimensions and extend current image
    new_dim = tuple(np.array(imawcs.shape).astype(np.int) + extra_pixels)
    ima_ext = imawcs.regrid(newdim=new_dim, refpos=imawcs.get_start(),
                            refpix=tuple(extra_pixels / 2.),
                            newinc=imawcs.get_step()[0]*3600.)

    # Copy and rotate WCS
    new_wcs = copy.deepcopy(ima_ext.wcs)
    upipe.print_info("Rotating spatial WCS of Cube by {} degrees".format(rotangle))
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


class ExposureInfo(object):
    def __init__(self, targetname, dataset, tpl, nexpo):
        """A dummy class to just store temporarily
        the various basic info about a Muse exposure
        """
        self.targetname = targetname
        self.dataset = dataset
        self.tpl = tpl
        self.nexpo = nexpo


def filter_list_with_pdict(input_list, list_datasets=None,
                           dict_files=None, verbose=True):
    """Filter out exposures (pixtab or cube namelist) using a dictionary which
    has a list of datasets and for each dataset a list of exposure number.

    Args:
        input_list (list of str):  input list to filter
        dict_files (dict):  dictionary used to filter

    Returns:
        selected_filename_list: selected list of files
        exposure_list_per_pointing: selected list of files for each pointing

    """
    nfiles_input_list = len(input_list)
    if dict_files is None:
        # Returning the default input list
        selected_filename_list = input_list
        # Just one dummy pointing with all files
        dict_exposures_per_pointing = {0: input_list}
        # Building the dummy list of tpl and nexpo for
        # this input list, decrypting with get_tpl_nexpo
        list_tplexpo = []
        for filename in input_list:
            ftpl, fnexpo = get_tpl_nexpo(filename)
            list_tplexpo.append([ftpl, fnexpo])
        dict_tplexpo_per_pointing = {0: list_tplexpo}

    # Otherwise use the ones which are given via their expo numbers
    else:
        selected_filename_list = []
        dict_exposures_per_pointing = {}
        dict_tplexpo_per_pointing = {}
        # this is the list of exposures to consider

        if list_datasets is None:
            list_datasets = dict_files.keys()
        elif not isinstance(list_datasets, list):
            upipe.print_error("Cannot recognise input dataset(s)")
        else:
            for dataset in list_datasets:
                if dataset not in dict_files:
                    upipe.print_warning("Dataset {} not in dictionary "
                                        "- skipping".format(dataset))
                else:
                    list_tpltuple = dict_files[dataset]
                    # We loop on that list which should contain 
                    # the list of tpl associated
                    # with a list of exposure numbers
                    for expotuple in list_tpltuple:
                        # We get the tpl, and then the list of expo numbers
                        tpl, list_expo = expotuple[0], expotuple[1]
                        # For each list of expo numbers, check 
                        # if this is just a number
                        # or also a pointing association
                        for expo in list_expo:
                            # By default we assign the dataset as 
                            # pointing number
                            if type(expo) in [str, int]:
                                nexpo = int(expo)
                                pointing = int(dataset)
                            elif len(expo) == 2:
                                nexpo = int(expo[0])
                                pointing = int(expo[1])
                            else:
                                upipe.print_warning(f"Dictionary entry {expotuple} "
                                                    f"ignored (type of expo - {expo} - "
                                                    f"is {type(expo)}")
                                break

                            # Check whether this exists in the our cube list
#                            suffix_expo = "_{0:04d}".format(nexpo)
                            for filename in input_list:
                                ftpl, fnexpo = get_tpl_nexpo(filename)
#                                if (suffix_expo in filename) and (tpl in filename):
                                if (nexpo == int(fnexpo)) & (ftpl == tpl):
                                    # We select the file
                                    selected_filename_list.append(filename)
                                    if pointing not in dict_exposures_per_pointing:
                                        dict_exposures_per_pointing[pointing] = []
                                        dict_tplexpo_per_pointing[pointing] = []
                                    dict_exposures_per_pointing[pointing].append(filename)
                                    dict_tplexpo_per_pointing[pointing].append([tpl, nexpo])
                                    # And remove it from the list
                                    input_list.remove(filename)
                                    # We break out of the cube for loop
                                    break

    if verbose:
        upipe.print_info("Datasets {0} - Selected {1}/{2} exposures after "
                         "dictionary filtering".format(list_datasets,
                                                len(selected_filename_list),
                                                nfiles_input_list))

        for pointing in dict_tplexpo_per_pointing:
            upipe.print_info(f"Pointing {pointing} - Detected exposures [TPL / NEXPO]:")
            for tplexpo in dict_tplexpo_per_pointing[pointing]:
                upipe.print_info(f"     {tplexpo[0]} / {tplexpo[1]}")

    return selected_filename_list, dict_exposures_per_pointing

def filter_list_with_suffix_list(list_names, included_suffix_list=[],
                                 excluded_suffix_list=[], name_list=""):
    """

    Args:
        list_names (list of str):
        included_suffix_list (list of str):
        excluded_suffix_list (list of str):

    Returns:

    """
    if name_list is not None:
        add_message = f"for list {name_list}"
    else:
        add_message = ""

    # if the list of inclusion suffix is empty, just use all cubes
    if len(included_suffix_list) > 0:
        upipe.print_info(f"Using suffixes {included_suffix_list} "
                         f"as an inclusive condition {add_message}")
        # Filtering out the ones that don't have any of the suffixes
        temp_list = copy.copy(list_names)
        for l in temp_list:
            if any([suff not in l for suff in included_suffix_list]):
                _ = list_names.remove(l)

    # if the list of exclusion suffix is empty, just use all cubes
    if len(excluded_suffix_list) > 0:
        upipe.print_info(f"Using suffixes {excluded_suffix_list} "
                         f"as an exclusive condition {add_message}")
        # Filtering out the ones that have any of the suffixes
        temp_list = copy.copy(list_names)
        for l in temp_list:
            if any([suff in l for suff in excluded_suffix_list]):
                _ = list_names.remove(l)

    return list_names


def my_linear_model(B, x):
    """Linear function for the regression.

    Parameters
    ----------
    B : 1D np.array of 2 floats
        Input 1D polynomial parameters (0=constant, 1=slope)
    x : np.array
        Array which will be multiplied by the polynomial

    Returns
    -------
        An array = B[1] * (x + B[0])
    """
    return B[1] * (x + B[0])


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

    nchunk_x = np.int(list_arrays[0].shape[0] // chunk_size - 1)
    nchunk_y = np.int(list_arrays[0].shape[1] // chunk_size - 1)
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


def filtermed_image(data, border=0, filter_size=2):
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
    meddata = nd.filters.median_filter(data, filter_size)

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