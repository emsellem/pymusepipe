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
import copy
from collections import OrderedDict
import re

from astropy.io import fits as pyfits

# Import package modules
from .config_pipe import (default_filter_list, dict_musemodes, default_short_filter_list,
                          default_ndigits, default_str_pointing, default_str_dataset)
from . import util_pipe as upipe


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

# -----------  END PRINTING FUNCTIONS -----------------------
def filter_list_to_str(filter_list):
    if filter_list is None:
        return default_short_filter_list

    if type(filter_list) is list:
        fl = str(filter_list[0])
        for f in filter_list[1:]:
            fl += ",f"
        return fl
    elif type(filter_list) is str:
        return filter_list
    else:
        upipe.print_warning(f"Could not recognise type of filter_list {filter_list}")
        return ""


def check_filter_list(filter_list):
    if filter_list is None:
        return []

    if type(filter_list) is list:
        return filter_list
    elif type(filter_list) is str:
        return filter_list.split(',')
    else:
        upipe.print_warning(f"Could not recognise type of filter_list {filter_list}")
        return []


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

def get_dataset_tpl_nexpo(filename, str_dataset=default_str_dataset, ndigits=default_ndigits):
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
        [(dataset, tpl, nexpo)] = re.findall("_" + str_dataset + r'(\d{' + str(ndigits)
                                             + r'})' + r'\_(\S{19})\_(\d{4})', basestr)
        if len(nexpo) > 0:
            return int(dataset), tpl, int(nexpo)
        else:
            return -1, "", -1
    except ValueError:
        return -1, "", -1


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


def get_pointing_name(pointing=1, str_pointing=default_str_pointing, ndigits=default_ndigits):
    """Formatting for the names using the number and
    the number of digits and prefix string

    Input
    -----
    pointing: int
       Pointing number
    str_pointing: str
        Prefix representing the pointing
    ndigits: int
        Number of digits to be used for formatting

    Returns
    -------
    string for the dataset/pointing name prefix
    """
    return f"{str_pointing}{int(pointing):0{int(ndigits)}}"


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
                           dict_files=None, verbose=True,
                           str_dataset=default_str_dataset, ndigits=default_ndigits):
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
    if list_datasets is None and dict_files is not None:
        list_datasets = dict_files.keys()
    elif not isinstance(list_datasets, list):
        upipe.print_error("Cannot recognise input dataset(s)")
        list_datasets = []

    # If not dictionary is provided, we try to build it
    if dict_files is None:
        # Returning the default input list
        selected_filename_list = input_list
        # Building the dummy list of tpl and nexpo for
        # this input list, decrypting with get_tpl_nexpo
        dict_files = {}
        dict_files_with_tpl = {}
        for filename in input_list:
            fdataset, ftpl, fnexpo = get_dataset_tpl_nexpo(filename, str_dataset=str_dataset,
                                                           ndigits=ndigits)
            # Did not find the string associated with dataset
            if fdataset == -1:
                continue
            # or found it, then record it
            else:
                # Record only if the input list
                if fdataset in list_datasets:
                    if fdataset not in dict_files:
                        dict_files_with_tpl[fdataset] = {ftpl: [fnexpo]}
                    else:
                        if ftpl not in dict_files_with_tpl:
                            dict_files_with_tpl[fdataset][ftpl] = [fnexpo]
                        else:
                            dict_files_with_tpl[fdataset][ftpl].append(fnexpo)

                for dataset in dict_files_with_tpl:
                    dict_files[dataset] = []
                    for tpl in dict_files_with_tpl[dataset]:
                        list_nexpo = []
                        for nexpo in dict_files_with_tpl[dataset][tpl]:
                            list_nexpo.append(nexpo)
                        dict_files[dataset].append((tpl, list_nexpo))


        # if len(dict_files) == 0:
        #     list_tplexpo = []
        #     for filename in input_list:
        #         ftpl, fnexpo = get_tpl_nexpo(filename)
        #         list_tplexpo.append([ftpl, fnexpo])
        #     # Just one dummy pointing with all files
        #     # Still use the dataset number if provided
        #     dict_tplexpo_per_dataset = {}
        #     if len(list_datasets) == 1:
        #         dict_exposures_per_pointing = {list_datasets[0]: input_list}
        #         dict_tplexpo_per_pointing = {list_datasets[0]: list_tplexpo}
        #         dict_tplexpo_per_dataset[list_datasets[0]] = {1: list_tplexpo}
        #     # if more than 1, then we need to pass them all
        #     # to a dummy dataset number
        #     else:
        #         dict_exposures_per_pointing = {1: input_list}
        #         dict_tplexpo_per_pointing = {1: list_tplexpo}
        #         for dataset in list_datasets:
        #             dict_tplexpo_per_dataset[dataset] = {1: list_tplexpo}

    # Otherwise use the ones which are given via their expo numbers
    selected_filename_list = []
    dict_exposures_per_pointing = {}
    dict_tplexpo_per_pointing = {}
    dict_tplexpo_per_dataset = {}
    # this is the list of exposures to consider

    for dataset in list_datasets:
        dict_tplexpo_per_dataset[dataset] = {}
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
                    for filename in input_list:
                        ftpl, fnexpo = get_tpl_nexpo(filename)
                        if (nexpo == int(fnexpo)) & (ftpl == tpl):
                            # We select the file
                            selected_filename_list.append(filename)
                            if pointing not in dict_exposures_per_pointing:
                                dict_exposures_per_pointing[pointing] = []
                                dict_tplexpo_per_pointing[pointing] = []
                            dict_exposures_per_pointing[pointing].append(filename)
                            dict_tplexpo_per_pointing[pointing].append([dataset, tpl,
                                                                        nexpo])
                            if pointing not in dict_tplexpo_per_dataset[dataset]:
                                dict_tplexpo_per_dataset[dataset][pointing] = []
                            dict_tplexpo_per_dataset[dataset][pointing].append([tpl, nexpo])
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
            upipe.print_info(f"Pointing {pointing} - Detected exposures [DATASET / TPL / NEXPO]:")
            for tplexpo in dict_tplexpo_per_pointing[pointing]:
                upipe.print_info(f"     {tplexpo[0]} / {tplexpo[1]} / {tplexpo[2]}")

    return selected_filename_list, dict_exposures_per_pointing, dict_tplexpo_per_pointing, \
        dict_tplexpo_per_dataset

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
