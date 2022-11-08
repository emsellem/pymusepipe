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

# Numpy
import numpy as np

from astropy import constants as const
from astropy.io import fits as pyfits

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
        upipe.error(f"Cannot find such a field ({field}) in the dict_musemodes")
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
        upipe.print_error("Please choose between one of these media: {0}".format(index_line.key()))
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
        if len(params) != self.nparams:
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
        if self.params == None :
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
        if self.params == None :
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
        if self.params == None :
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
            out_suffix = "_{}".format(out_suffix)
        out_name = "{0}{1}{2}".format(name, out_suffix, extension)

    # write output
    final_rot_cube.write(joinpath(outwcs_folder, out_name))
    return outwcs_folder, out_name

def filter_list_with_pdict(input_list, list_datasets=None,
                           dict_files=None,
                           verbose=True):
    """Filter out exposures (pixtab or cube namelist) using a dictionary which
    has a list of datasets and for each dataset a list of exposure number.

    Args:
        input_list (list of str):  input list to filter
        dict_files (dict):  dictionary used to filter

    Returns:
        selected_list: selected list of files

    """
    nfiles_input_list = len(input_list)
    if dict_files is None:
          selected_list = input_list

    # Otherwise use the ones which are given via their expo numbers
    else:
        selected_list = []
        # this is the list of exposures to consider

        if list_datasets is None:
            list_datasets = dict_files.keys()
        elif not isinstance(list_datasets, list):
            upipe.print_error("Cannot recognise input dataset(s)")
            return selected_list

        for dataset in list_datasets:
            if dataset not in dict_files:
                upipe.print_warning("Dataset {} not in dictionary "
                                    "- skipping".format(dataset))
            else:
                list_expo = dict_files[dataset]
                # We loop on that list
                for expotuple in list_expo:
                    tpl, nexpo = expotuple[0], expotuple[1]
                    for expo in nexpo:
                        # Check whether this exists in the our cube list
                        suffix_expo = "_{0:04d}".format(np.int(expo))
                        for filename in input_list:
                            if (suffix_expo in filename) and (tpl in filename):
                                # We select the file
                                selected_list.append(filename)
                                # And remove it from the list
                                input_list.remove(filename)
                                # We break out of the cube for loop
                                break

    if verbose:
        upipe.print_info("Datasets {0} - Selected {1}/{2} files after "
                         "dictionary filtering".format(list_datasets,
                                                len(selected_list),
                                                nfiles_input_list))
    return selected_list

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
