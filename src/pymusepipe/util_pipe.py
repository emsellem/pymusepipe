# Licensed under a MIT license - see LICENSE

"""MUSE-PHANGS utility functions for pymusepipe
"""

__authors__ = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__ = "MIT License"
__contact__ = " <eric.emsellem@eso.org>"

# Importing generic modules
import os
from os.path import join as joinpath
import glob
import time
import copy
from collections import OrderedDict
import re

# Numpy
import numpy as np

# Astropy
from astropy.io import fits as pyfits

# Import package modules
from .config_pipe import (default_filter_list, dict_musemodes, default_short_filter_list,
                          default_ndigits, default_str_pointing, default_str_dataset,
                          dict_folders, dict_products_scipost)

prefix_final_cube = dict_products_scipost['cube'][0]
default_object_folder = dict_folders['object']

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
    except AttributeError:
        pass
    try:
        verbose = mypipe.verbose
    except AttributeError:
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
    except AttributeError:
        pass
    try:
        verbose = mypipe.verbose
    except AttributeError:
        verbose = kwargs.pop("verbose", True)
    
    if verbose:
        print(INFO + "# MusePipeInfo " + ENDC + text, **kwargs)


def print_debug(text, **kwargs):
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
    except AttributeError:
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
    except AttributeError:
        pass
    try:
        verbose = mypipe.verbose
    except AttributeError:
        verbose = kwargs.pop("verbose", True)
    
    if verbose:
        print(ERROR + "# MusePipeError " + ENDC + text, **kwargs)
# -----------  END PRINTING FUNCTIONS -----------------------


def append_value_to_dict(mydict, key, value):
    """Append a value to key within a given dictionary. If the key does not exist it creates
    a list of 1 element for that key

    Input
    -----
    mydict: dict
    key:
    value:

    Returns
    -------
    Updated dictionary
    """
    if key in mydict:
        if not isinstance(mydict[key], list):
            # converting key to list type
            mydict[key] = [mydict[key]]

    # Otherwise create an empty list
    else:
        mydict[key] = []

    # Append the key's value in list
    mydict[key].append(value)

    return mydict


def filter_list_to_str(filter_list):
    if filter_list is None:
        return default_short_filter_list

    if type(filter_list) is list:
        fl = str(filter_list[0])
        for fi in filter_list[1:]:
            fl += f",{fi}"
        return fl
    elif type(filter_list) is str:
        return filter_list
    else:
        print_warning(f"Could not recognise type of filter_list {filter_list}")
        return ""


def check_filter_list(filter_list):
    if filter_list is None:
        return []

    if type(filter_list) is list:
        return filter_list
    elif type(filter_list) is str:
        return filter_list.split(',')
    else:
        print_warning(f"Could not recognise type of filter_list {filter_list}")
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
        print_error(f"Cannot find such a field ({field}) in the dict_musemodes")
        return ""

    index = dict_musemodes[field]
    sval = musemode.split(delimiter)

    if len(sval) < index+1:
        print_error(f"Error in analyse_musemode. Cannot access field {index} "
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

            except IndexError:
                print(f"String index [{loc}] out of range [{len(text)}] in add_string")

    return text


def get_dataset_tpl_nexpo(filename, str_dataset=default_str_dataset, ndigits=default_ndigits,
                          filtername=None):
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
    if filtername is None:
        filtername = ""
    else:
        filtername = f"_{filtername}"

    try:
        [(dataset, tpl, nexpo)] = re.findall("_" + str_dataset + r'(\d{' + str(ndigits) + r'})'
                                             + str(filtername) + r'_(\S{19})_(\d{4})', basestr)
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


def create_time_name():
    """Create a time-link name for file saving purposes

    Return: a string including the YearMonthDay_HourMinSec
    """
    return str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))


def formatted_time():
    """ Return: a string including the formatted time
    """
    return str(time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()))


def safely_create_folder(path, verbose=True):
    """Create a folder given by the input path This small function tries to create it
    and if it fails it checks whether the reason is that it is not a path and then warn the user

    Input
    -----
    path: str
    verbose: bool

    Creates
    -------
    A new folder if the folder does not yet exist
    """
    if path is None:
        if verbose:
            print_info("Input path is None, not doing anything")
        return
    if verbose:
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


def abspath(path):
    """Normalise the path to get it short but absolute
    """
    return os.path.abspath(os.path.realpath(path))


def normpath(path):
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
    
    command = "muse_cube_filter -f {0} {1} {2}".format(filter_list, cubename, filter_fits_file)
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


# def filter_list_with_pdict(input_list, list_datasets=None, dict_files=None, verbose=True,
#                            str_dataset=default_str_dataset, ndigits=default_ndigits,
#                            filtername=None):
#     """Filter out exposures (pixtab or cube namelist) using a dictionary which
#     has a list of datasets and for each dataset a list of exposure number.
#
#     Args:
#         input_list (list of str):  input list to filter
#         dict_files (dict):  dictionary used to filter
#         list_datasets: list of int
#         dict_files: dictionary
#         verbose: bool
#         str_dataset: str
#         ndigits: int
#         filtername: str
#
#     Returns:
#         selected_filename_list: selected list of files
#         exposure_list_per_pointing: selected list of files for each pointing
#
#     """
#     if list_datasets is None:
#         list_datasets = []
#     elif not isinstance(list_datasets, list):
#         print_error("Cannot recognise input dataset(s)")
#         list_datasets = []
#
#     nfiles_input_list = len(input_list)
#
#     # If not dictionary is provided, we try to build it
#     if dict_files is None:
#         # Building the dummy list of tpl and nexpo for
#         # this input list, decrypting with get_tpl_nexpo
#         dict_files = {}
#         dict_files_with_tpl = {}
#         for filename in input_list:
#             if verbose:
#                 print_info(f"Adressing File name: {filename}")
#             fdataset, ftpl, fnexpo = get_dataset_tpl_nexpo(filename, str_dataset=str_dataset,
#                                                            ndigits=ndigits, filtername=filtername)
#             if verbose:
#                 print_info(f"    Detected = Dataset/TPLS/Nexpo: {fdataset} / {ftpl} / {fnexpo}")
#             # Did not find the string associated with dataset
#             if fdataset == -1:
#                 continue
#             # or found it, then record it
#             else:
#                 # Record only if the input list
#                 if fdataset in list_datasets or len(list_datasets) == 0:
#                     if fdataset not in dict_files_with_tpl:
#                         dict_files_with_tpl[fdataset] = {ftpl: [fnexpo]}
#                     else:
#                         if ftpl not in dict_files_with_tpl[fdataset]:
#                             dict_files_with_tpl[fdataset][ftpl] = [fnexpo]
#                         else:
#                             dict_files_with_tpl[fdataset][ftpl].append(fnexpo)
#
#                 for dataset in dict_files_with_tpl:
#                     dict_files[dataset] = []
#                     for tpl in dict_files_with_tpl[dataset]:
#                         list_nexpo = []
#                         for nexpo in dict_files_with_tpl[dataset][tpl]:
#                             list_nexpo.append(nexpo)
#                         list_nexpo.sort()
#                         dict_files[dataset].append((tpl, list_nexpo))
#
#     if len(list_datasets) == 0:
#         list_datasets = list(dict_files.keys())
#
#     if verbose:
#         print_info(f"Check file = dict_files is : {dict_files}")
#         print_info(f"Check file = list_datasets is : {list_datasets}")
#
#     # Otherwise use the ones which are given via their expo numbers
#     selected_filename_list = []
#     dict_exposures_per_pointing = {}
#     dict_tplexpo_per_pointing = {}
#     dict_tplexpo_per_dataset = {}
#     # this is the list of exposures to consider
#
#     for dataset in list_datasets:
#         dict_tplexpo_per_dataset[dataset] = {}
#         if dataset not in dict_files:
#             print_warning(f"Dataset {dataset} not in dictionary - skipping")
#         else:
#             list_tpltuple = dict_files[dataset]
#             # We loop on that list which should contain
#             # the list of tpl associated
#             # with a list of exposure numbers
#             for expotuple in list_tpltuple:
#                 # We get the tpl, and then the list of expo numbers
#                 tpl, list_expo = expotuple[0], expotuple[1]
#                 # For each list of expo numbers, check
#                 # if this is just a number
#                 # or also a pointing association
#                 for expo in list_expo:
#                     # By default, we assign the dataset as
#                     # pointing number
#                     if type(expo) in [str, int]:
#                         nexpo = int(expo)
#                         pointing = int(dataset)
#                     elif len(expo) == 2:
#                         nexpo = int(expo[0])
#                         pointing = int(expo[1])
#                     else:
#                         print_warning(f"Dictionary entry {expotuple} "
#                                       f"ignored (type of expo - {expo} - "
#                                       f"is {type(expo)}")
#                         break
#
#                     # Check whether this exists in the cube list
#                     for filename in input_list:
#                         ftpl, fnexpo = get_tpl_nexpo(filename)
#                         if (nexpo == int(fnexpo)) & (ftpl == tpl):
#                             # We select the file
#                             selected_filename_list.append(filename)
#                             if pointing not in dict_exposures_per_pointing:
#                                 dict_exposures_per_pointing[pointing] = []
#                                 dict_tplexpo_per_pointing[pointing] = []
#                             dict_exposures_per_pointing[pointing].append(filename)
#                             dict_tplexpo_per_pointing[pointing].append([dataset, tpl,
#                                                                         nexpo])
#                             if pointing not in dict_tplexpo_per_dataset[dataset]:
#                                 dict_tplexpo_per_dataset[dataset][pointing] = []
#                             dict_tplexpo_per_dataset[dataset][pointing].append([tpl, nexpo])
#                             # And remove it from the list
#                             input_list.remove(filename)
#                             # We break out of the cube for loop
#                             break
#
#     selected_filename_list.sort()
#     if verbose:
#         print_info(f"Datasets {list_datasets} - "
#                    f"Selected {len(selected_filename_list)}/{nfiles_input_list} "
#                    f"exposures after dictionary filtering")
#
#         for pointing in dict_tplexpo_per_pointing:
#             print_info(f"Pointing {pointing} - Detected exposures [DATASET / TPL / NEXPO]:")
#             for tplexpo in dict_tplexpo_per_pointing[pointing]:
#                 print_info(f"     {tplexpo[0]} / {tplexpo[1]} / {tplexpo[2]}")
#
#     return selected_filename_list, dict_exposures_per_pointing, dict_tplexpo_per_pointing, \
#         dict_tplexpo_per_dataset


def filter_list_with_suffix_list(list_names, included_suffix_list=[],
                                 excluded_suffix_list=[], name_list=""):
    """Filter a list using suffixes (to exclude or include)

    Input
    -----
    list_names: list of str
    included_suffix_list: list of str
    excluded_suffix_list: list of str
    name_list: str default=""

    Returns
    -------

    """
    if name_list is not None:
        add_message = f"for list {name_list}"
    else:
        add_message = ""

    # if the list of inclusion suffix is empty, just use all cubes
    if len(included_suffix_list) > 0:
        print_info(f"Using suffixes {included_suffix_list} "
                   f"as an inclusive condition {add_message}")
        # Filtering out the ones that don't have any of the suffixes
        temp_list = copy.copy(list_names)
        for litem in temp_list:
            if any([suff not in litem for suff in included_suffix_list]):
                _ = list_names.remove(litem)

    # if the list of exclusion suffix is empty, just use all cubes
    if len(excluded_suffix_list) > 0:
        print_info(f"Using suffixes {excluded_suffix_list} "
                   f"as an exclusive condition {add_message}")
        # Filtering out the ones that have any of the suffixes
        temp_list = copy.copy(list_names)
        for litem in temp_list:
            if any([suff in litem for suff in excluded_suffix_list]):
                _ = list_names.remove(litem)

    return list_names


def get_list_targets(folder=""):
    """Getting a list of existing targets given path. This is done by simply listing the existing
    folders. This may need to be filtered.

    Input
    -----
    folder: str
        Folder name where the targets are

    Return
    ------
    list_targets: list of str
    """
    # Done by scanning the target path
    list_targets = [name for name in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, name))]

    list_targets.sort()
    print_info("Potential Targets -- list: {0}".format(str(list_targets)))
    return list_targets


def build_dict_datasets(data_path="", str_dataset=default_str_dataset, ndigits=default_ndigits):
    """Build a dictionary of datasets for each target in the sample

    Input
    ------
    data_path: str
       Path of the target data
    str_dataset: str default=default_str_dataset
        Prefix string for datasets (see config_pipe.py)
    ndigits: int default=default_ndigits
       Number of digits to format the name of the dataset (see config_pipe.py)

    Returns
    -------
    dict_dataset: dict

    """
    list_targets = get_list_targets(data_path)
    dict_dataset = {}

    npath = os.path.normpath(data_path)
    s_npath = npath.split(os.sep)

    for target in list_targets:
        target_path = f"{data_path}/{target}/"
        list_datasets = get_list_datasets(target_path, str_dataset, ndigits)
        dict_t = {}
        for ds in list_datasets:
            dict_t[ds] = 1

        dict_dataset[target] = [s_npath[-1], dict_t]

    return dict_dataset


def build_dict_exposures(target_path="", str_dataset=default_str_dataset,
                         ndigits=default_ndigits, show_pointings=False):
    """Build a dictionary of exposures using the list of datasets found for the
    given dataset path

    Input
    ------
    target_path: str
       Path of the target data
    str_dataset: str
        Prefix string for datasets
    ndigits: int
       Number of digits to format the name of the dataset

    Returns
    -------
    dict_expo: dict
        Dictionary of exposures in each dataset

    """
    list_datasets = get_list_datasets(target_path, str_dataset, ndigits)
    dict_expos = {}
    for dataset in list_datasets:
        # Get the name of the dataset
        name_dataset = get_dataset_name(dataset, str_dataset, ndigits)
        print_info(f"For dataset {dataset}")
        # Get the list of exposures for that dataset
        dict_p = get_list_exposures(joinpath(target_path, name_dataset))

        # If introducing the pointings, we set them by just showing the dataset
        # numbers as a default, to be changed lated by the user
        if show_pointings:
            dict_p_wpointings = {}
            for tpl in dict_p:
                dict_p_wpointings[tpl] = [[nexpo, dataset] for nexpo in dict_p[tpl]]
            # Now changing dict_p
            dict_p = copy.copy(dict_p_wpointings)

        # Now creating the dictionary entry for that dataset
        dict_expos[dataset] = [(tpl, dict_p[tpl]) for tpl in dict_p]

    return dict_expos

def get_list_datasets(target_path="", str_dataset=default_str_dataset,
                      ndigits=default_ndigits, verbose=False):
    """Getting the list of existing datasets for a given target path

    Input
    -----
    target_path: str
       Path of the target data
    str_dataset: str
        Prefix string for datasets
    ndigits: int
       Number of digits to format the name of the dataset

    Return
    ------
    list_datasets: list of int
    """
    # Done by scanning the target path
    if verbose:
        print_info(f"Searching datasets in {target_path} with {str_dataset} prefix")
    all_folders = glob.glob(f"{target_path}/{str_dataset}*")
    if verbose:
        print_info(f"All folder names  = {all_folders}")
    all_datasets = [os.path.split(s)[-1] for s in all_folders]
    # Sorting names
    all_datasets.sort()
    if verbose:
        print_info(f"All detected folder names  = {all_datasets}")

    # Now filtering with the right rule
    r = re.compile(f"{str_dataset}\d{{{ndigits}}}$")
    good_datasets = [f for f in all_datasets if r.match(f)]
    good_datasets.sort()
    if verbose:
        print_info(f"All good folder names  = {good_datasets}")

    # Creating the list of datasets which is a list of numbers (int)
    list_datasets = []
    for folder in good_datasets:
        list_datasets.append(int(folder[-int(ndigits):]))

    list_datasets.sort()
    if verbose:
        print_info(f"Dataset list: {str(list_datasets)}")
    return list_datasets

def get_list_exposures(dataset_path="", object_folder=default_object_folder):
    """Getting a list of exposures from a given path

    Input
    -----
    dataset_path: str
        Folder name where the dataset is

    Return
    ------
    list_expos: list of int
    """
    # Done by scanning the target path
    list_files = glob.glob(f"{dataset_path}/{object_folder}/{prefix_final_cube}*_????.fits")
    list_expos = []
    for name in list_files:
        tpl, lint = get_tpl_nexpo(name)
        if lint > 0:
            list_expos.append((tpl, int(lint)))

    # Making it unique and sort
    list_expos = np.unique(list_expos, axis=0)
    # Sorting by tpl and expo number
    sorted_list = sorted(list_expos, key=lambda e: (e[0], e[1]))

    # Building the final list
    dict_expos = {}
    for l in sorted_list:
        tpl = l[0]
        if tpl in dict_expos:
            dict_expos[tpl].append(int(l[1]))
        else:
            dict_expos[tpl] = [int(l[1])]

    # Finding the full list of tpl
    print_info("Exposures list:")
    for tpl in dict_expos:
        print_info("TPL= {0} : Exposures= {1}".format(tpl, dict_expos[tpl]))

    return dict_expos

def get_list_reduced_pixtables(target_path="", list_datasets=None,
                               suffix="", str_dataset=default_str_dataset,
                               ndigits=default_ndigits, **kwargs):
    """Provide a list of reduced pixtables

    Input
    -----
    target_path: str
        Path for the target folder
    list_datasets: list of int
        List of integers, providing the list of datasets to consider
    suffix: str
        Additional suffix, if needed, for the names of the PixTables.
    """
    # Getting the pieces of the names to be used for pixtabs
    pixtable_prefix = dict_products_scipost['individual'][0]
    print_info(f"Will be looking for PIXTABLES with suffix {pixtable_prefix}")

    # Object folder
    object_folder = kwargs.pop("object_folder", default_object_folder)

    # Initialise the dictionary of pixtabs to be found in each dataset
    dict_pixtables = {}

    # Defining the dataset list if not provided
    # Done by scanning the target path
    if list_datasets is None:
        list_datasets = get_list_datasets(target_path, ndigits=ndigits, str_dataset=str_dataset)

    # Looping over the datasets
    for dataset in list_datasets:
        # get the path of the dataset
        path_dataset = joinpath(target_path, get_dataset_name(dataset, str_dataset, ndigits))
        # List existing pixtabs, using the given suffix
        list_pixtabs = glob.glob(path_dataset + f"/{object_folder}/{pixtable_prefix}{suffix}*fits")

        # Reset the needed temporary dictionary
        dict_tpl = {}
        # Loop over the pixtables for that dataset
        for pixtab in list_pixtabs:
            # Split over the PIXTABLE_REDUCED string
            sl = pixtab.split(pixtable_prefix + "_")
            # Find the expo number
            nf = len(".fits")
            expo = sl[1][-int(nf+4):-int(nf)]
            # Find the tpl
            tpl = sl[1].split("_" + expo)[0]
            # If not already there, add it
            if tpl not in dict_tpl:
                dict_tpl[tpl] = [int(expo)]
            # if already accounted for, add the expo number
            else:
                dict_tpl[tpl].append(int(expo))

        # Creating the full list for that dataset
        full_list = []
        for tpl in dict_tpl:
            dict_tpl[tpl].sort()
            full_list.append((tpl, dict_tpl[tpl]))

        # And now filling in the dictionary for that dataset
        dict_pixtables[dataset] = full_list

    return dict_pixtables