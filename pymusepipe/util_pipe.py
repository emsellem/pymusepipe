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

# Numpy
import numpy as np

from astropy import constants as const
from astropy.io import fits as pyfits

# Import package modules
from pymusepipe.emission_lines import list_emission_lines
from pymusepipe.emission_lines import full_muse_wavelength_range
from pymusepipe.config_pipe import default_filter_list

from collections import OrderedDict

############    PRINTING FUNCTIONS #########################
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
        
def normpath(path) :
    """Normalise the path to get it short
    """
    return os.path.relpath(os.path.realpath(path))

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

def add_key_pointing_expo(name_image, iexpo, pointing):
    """Add pointing and expo number to image

    Input
    -----
    name_image: str
    iexpo: int
    pointing: int
    """
    # Writing the pointing and iexpo in the IMAGE_FOV
    this_image = pyfits.open(name_image, mode='update')
    this_image[0].header['MUSEPIPE_POINTING'] = (pointing, "Pointing number")
    this_image[0].header['MUSEPIPE_IEXPO'] = (iexpo, "Exposure number")
    this_image.flush()
    print_info("Keyworks MUSEPIPE_POINTING/EXPO updated for image {}".format(
        name_image))

