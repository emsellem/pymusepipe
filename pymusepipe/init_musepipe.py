# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS initialisation of folders
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# Standard modules
import os
from os.path import join as joinpath
import copy
from pymusepipe import util_pipe as upipe

############################################################
#                      BEGIN
# The following parameters can be adjusted for the need of
# the specific pipeline to be used
############################################################
# Default hard-coded folders
dic_user_folders = {
        # values provide the folder and whether or not this should be attempted to create
            # Muse calibration files (common to all)
            "musecalib": "/home/mcelroy/reflex/install/calib/muse-2.2/cal/'",
            # Calibration files (specific to OBs)
            "root" : "/mnt/fhgfs/PHANGS/MUSE/LP_131117/",
            }

# Default hard-coded fits files - Calibration Tables
# This should be replaced by an ascii file reading at some point
dic_calib_tables = {
            # Muse calibration files (common to all)
            "geo_table": "geometry_table_wfm.fits",
            # Calibration files (specific to OBs)
            "astro_table" : "astrometry_wcs_wfm.fits",
            # Raw Data files
            "badpix_table" : "badpix_table_2015-06-02.fits",
            # Reduced files
            "vignetting_mask" : "vignetting_mask.fits",
            # Pixel tables
            "std_flux_table" : "std_flux_table.fits",
            # Sky Flat files
            "extinct_table" : "extinct_table.fits",
            # Line Catalog
            "line_catalog" : "line_catalog.fits",
            # Sky lines
            "sky_lines" : "sky_lines.fits",
            # Filter List
            "filter_list" : "filter_list.fits",
            }

############################################################
#                      END
############################################################

######################################################mp######
# Some fixed parameters for the structure
############################################################
def add_suffix_tokeys(dic, suffix="_folder") :
    newdic = {}
    for key in dic.keys() :
        setattr(newdic, key + suffix, dic[key])

# Default initialisation file
default_rc_filename = "~/.musepiperc"

dic_input_folders = {
            # Raw Data files
            "rawfiles" : "Raw/",
            # Config files
            "config" : "Config/",
            # Tables
            "astro_tables" : "Astro_tables/",
            # esores log files
            "esorex_log" : "Esorex_log/",
            # Data Products - first writing
            "pipe_products": "Pipe_products/"
            }

dic_folders = {
        # values provide the folder and whether or not this should be attempted to create
            # Master Calibration files
            "master" : "Master/",
            # Object files
            "object" : "Object/",
            # Sky files
            "sky" : "Sky/",
            # Std files
            "std" : "Std/",
            # Cubes
            "cubes" : "Cubes/",
            # Reconstructed Maps
            "maps" : "Maps/",
            # SOF folder 
            "sof" : "Sof/", 
            # Figure
            "figures" : "Figures/",
            }

# This dictionary shows the folders to be created within the TARGET name
dic_folders_target = {
        "alignment" : "Alignment/"
        }

############################################################
# Main class InitMuseParameters
############################################################

class InitMuseParameters(object) :
    def __init__(self, dirname="Config/", rc_filename=None, cal_filename=None, verbose=True, **kwargs) :
        """Define the default parameters (folders/calibration files) 
        and name suffixes for the MUSE data reduction
        """
        self.verbose = verbose
        # Will first test if there is an rc_file provided
        # If not, it will look for a default rc_filename, the name of which is provided
        # above. If not, the hardcoded default will be used.

        # First adding the suffix to the dictionaries
        # attributing the dictionaries
        self._dic_folders = dic_folders
        self._dic_input_folders = dic_input_folders
        self._dic_folders_target = dic_folders_target

        if rc_filename is None :
            if not os.path.isfile(default_rc_filename):
                upipe.print_warning(("No filename or {default_rc} file "
                     "to initialise from. We will use the default hardcoded " 
                     "in the init_musepipe.py module").format(default_rc=default_rc_filename))
                self.init_default_param(dic_user_folders)

            else :
                self.read_param_file(default_rc_filename, dic_user_folders) 
            self.rcfile = "default_values"
        else :
            rcfile = joinpath(dirname, rc_filename)
            self.read_param_file(rcfile, dic_user_folders)
            self.rcfile = rcfile

        # Initialisation of fixed attributes for the structure
        self.init_default_param(dic_folders)
        self.init_default_param(dic_input_folders)

        # Same happens with the calibration files.
        # If filename is provided, will use that, otherwise use the hard coded values.
        if cal_filename is None :
            self.init_default_param(dic_calib_tables)
            self.calfile = "default_values"
        else :
            calfile = joinpath(dirname, cal_filename)
            self.read_param_file(calfile, dic_calib_tables)
            self.calfile = calfile

    def init_default_param(self, dic_param) :
        """Initialise the parameters as defined in the input dictionary
        Hardcoded in init_musepipe.py
        """
        for key in dic_param.keys() :
            if self.verbose :
                upipe.print_info("Default initialisation of attribute {0}".format(key))
            setattr(self, key, dic_param[key])

    def read_param_file(self, filename, dic_param) :
        """Reading an input parameter initialisation file 
        """
        # Testing existence of filename
        if not os.path.isfile(filename) :
            upipe.print_info(("ERROR: input parameter {inputname} cannot be found. "
                    "We will use the default hardcoded in the "
                    "init_musepipe.py module").format(inputname=filename))
            return

        # If it exists, open and read it
        f_param = open(filename)
        lines = f_param.readlines()

        # Dummy dictionary to see which items are not initialised
        dummy_dic_param = copy.copy(dic_param)
        for line in lines :
            if line[0] in ["#", "%"] : continue 

            sline = line.split()
            if sline[0] in dic_param.keys() :
                if self.verbose :
                    upipe.print_info("Initialisation of attribute {0}".format(sline[0]))
                setattr(self, sline[0], sline[1]) 
                # Here we drop the item which was initialised
                val = dummy_dic_param.pop(sline[0])
            else :
                continue

        # Set of non initialised folders
        not_initialised_param = dummy_dic_param.keys()
        # Listing them as warning and using the hardcoded default
        for key in not_initialised_param :
            upipe.print_info(("WARNING: parameter {param} not initialised "
                   "We will use the default hardcoded value from "
                   "init_musepipe.py").format(param=key))
            setattr(self, key, dic_param[key])
