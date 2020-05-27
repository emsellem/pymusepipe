# Licensed under a MIT license - see LICENSE

"""MUSE-PHANGS pipeline wrapper
   initialisation of folders
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "MIT"
__contact__   = " <eric.emsellem@eso.org>"

# Standard modules
import os
from os.path import join as joinpath

import re

import copy

from . import util_pipe as upipe
from .config_pipe import (dict_user_folders, default_rc_filename,
                          dict_extra_filters, dict_calib_tables, dict_input_folders,
                          dict_folders, dict_folders_target, )

############################################################
# Some fixed parameters for the structure
############################################################
def add_suffix_tokeys(dic, suffix="_folder") :
    newdic = {}
    for key in dic:
        setattr(newdic, key + suffix, dic[key])

############################################################
# Main class InitMuseParameters
############################################################

class InitMuseParameters(object) :
    def __init__(self, folder_config="Config/", 
                 rc_filename=None, cal_filename=None, 
                 verbose=True, **kwargs) :
        """Define the default parameters (folders/calibration files) 
        and name suffixes for the MUSE data reduction

        Parameters
        ----------
        folder_config: str
            Name of the input folder for the configurations files
        rc_filename: str
            Name of the configuration file 
            including root input folder names
        cal_filename: str
            Name of the configuration file including
            the calibration input folders 
        verbose: bool [True]
        """
        self.verbose = verbose
        # Will first test if there is an rc_file provided
        # If not, it will look for a default rc_filename, the name of which is provided
        # above. If not, the hardcoded default will be used.

        # First adding the suffix to the dictionaries
        # attributing the dictionaries
        self._dict_folders = dict_folders
        self._dict_input_folders = dict_input_folders
        self._dict_folders_target = dict_folders_target
        self._dict_extra_filters = dict_extra_filters

        if rc_filename is None :
            if not os.path.isfile(default_rc_filename):
                upipe.print_warning(("No filename or {default_rc} file "
                     "to initialise from. We will use the default hardcoded " 
                     "in the init_musepipe.py module").format(default_rc=default_rc_filename))
                self.init_default_param(dict_user_folders)

            else :
                self.read_param_file(default_rc_filename, dict_user_folders)
            self.rcfile = "default_values"
        else :
            rcfile = joinpath(folder_config, rc_filename)
            self.read_param_file(rcfile, dict_user_folders)
            self.rcfile = rcfile

        # Initialisation of fixed attributes for the structure
        self.init_default_param(dict_folders)
        self.init_default_param(dict_input_folders)

        # Same happens with the calibration files.
        # If filename is provided, will use that, otherwise use the hard coded values.
        if cal_filename is None :
            self.init_default_param(dict_calib_tables)
            self.calfile = "default_values"
        else :
            calfile = joinpath(folder_config, cal_filename)
            self.read_param_file(calfile, dict_calib_tables)
            self.calfile = calfile

    def init_default_param(self, dict_param) :
        """Initialise the parameters as defined in the input dictionary
        Hardcoded in init_musepipe.py
        """
        for key in dict_param:
            upipe.print_info("Default initialisation of attribute {0}".format(key), pipe=self)
            setattr(self, key, dict_param[key])

    def read_param_file(self, filename, dict_param) :
        """Reading an input parameter initialisation file 
        """
        # Testing existence of filename
        if not os.path.isfile(filename) :
            upipe.print_error(("Input parameter {inputname} cannot be found. "
                    "We will use the default hardcoded in the "
                    "init_musepipe.py module").format(inputname=filename))
            return

        # If it exists, open and read it
        f_param = open(filename)
        lines = f_param.readlines()

        # Dummy dictionary to see which items are not initialised
        noninit_dict_param = copy.copy(dict_param)
        for line in lines :
            if line[0] in ["#", "%"] : continue 

            sline = re.split(r'(\s+)', line)
            keyword_name = sline[0]
            keyword = ("".join(sline[2:])).rstrip()
            if keyword_name in dict_param:
                upipe.print_info("Initialisation of attribute {0}".format(keyword_name), 
                                 pipe=self)
                setattr(self, keyword_name, keyword) 
                # Here we drop the item which was initialised
                val = noninit_dict_param.pop(keyword_name)
            else :
                continue

        # Listing them as warning and using the hardcoded default
        for key in noninit_dict_param:
            upipe.print_warning(("Parameter {param} not initialised "
                   "We will use the default hardcoded value from "
                   "init_musepipe.py").format(param=key))
            setattr(self, key, dict_param[key])
