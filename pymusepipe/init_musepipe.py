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
from pymusepipe.musepipe import MusePipe

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

# ----------------- Galaxies and Pointings ----------------#

# Sample of galaxies
# For each galaxy, we provide the pointings numbers and the run attached to that pointing
MUSEPIPE_sample = {
        "NGC628": {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
        "NGC1087": {1:1}, 
        "NGC1672": {1:1, 2:1, 3:1, 4:1, 5:1}
        }

# List of MUSEPIPE Observing runs
# For each Observing run, we provide the start and end dates
MUSEPIPE_runs = {
        'Run00' : ['P099','2017-10-01','2017-10-31'],
        'Run01' : ['P100','2017-10-01','2017-10-31'], 
        'Run02' : ['P101','2017-11-01','2017-11-30']
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
            # Reduced files
            "reduced" : "Reduced/",
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


####################################################
# Defining classes to get samples and objects
####################################################
class MusepipeSample(object) :
    def __init__(self) :
        self.sample = MUSEPIPE_sample
        self.targets = MUSEPIPE_sample.keys()

class MusepipeTarget(object) :
    def __init__(self, galaxyname=None, list_pointings=[1]) :
        if galaxyname not in MUSEPIPE_sample.keys() :
            upipe.print_error("ERROR: no Galaxy named {gal} in the defined sample".format(gal=galaxyname))
            return

        # Galaxy name
        upipe.print_info("Initialising Target {name}".format(name=galaxyname))
        self.targetname = galaxyname

        # Info of the pointings and extracting the observing run for each pointing
        self.info_pointings = MUSEPIPE_sample[galaxyname]
        if any([_ not in self.info_pointings.keys() for _ in list_pointings]) :
            upipe.print_error("ERROR: no pointing {0} for the Galaxy".format(list_pointings))
            return
        self.list_pointings = list_pointings
        self.observing_run = [self.info_pointing[_] for _ in self.list_pointings]

    def _get_file_name(self, suffix, pointing):
        return "{0}_P{1:02d}.txt".format(suffix, pointing)

    def run_pipeline(self, list_pointings=[1], fakemode=False, 
            suffix_logfile="logfile", suffix_rcfile="rcfile", suffix_calfile="calfile"):
        """Run the pipeline for all pointings in the list
        """
        if any([_ not in self.info_pointings.keys() for _ in self.list_pointings]) :
            upipe.print_error("ERROR: some pointing are not in "
                "the available list ({0})".format(self.list_pointings))
            return

        # Setting up the suffixes for the files
        self.suffix_logfile = suffix_logfile
        self.suffix_calfile = suffix_calfile
        self.suffix_rcfile = suffix_rcfile

        # Loop on the pointings
        self.pipelines = []
        upipe.print_info("---- Starting the Data Reduction ----")
        self.history = []
        for pointing in list_pointings:

            # Setting up the names of the output files
            logfile = self._get_logfile_name(self.suffix_logfile, pointing)
            calfile = self._get_calfile_name(self.suffix_calfile, pointing)
            rcfile = self._get_rcfile_name(self.suffix_rcfile, pointing)
            
            python_command = ("mypipe = musepipe.MusePipe(galaxyname={0}, "
                    "pointing={1}, rc_filename={2}, cal_filename={3}, "
                    "outlog=None, logfile={4}, fakemode={5}, "
                    "nocache=False)".format(galaxyname, pointing, rcfile, calfile, 
                            logfile, fakemode))
            upipe.print_info("====== START - POINTING {0:2d} ======".format(pointing))
            upipe.print_info(python_command)
            upipe.print_info("====== END   - POINTING {0:2d} ======".format(pointing))
            self.history.append(python_command)
            mypipe = MusePipe(galaxyname=galaxyname, pointing=pointing, 
                    rc_filename=rcfile, cal_filename=calfile, outlog=None, 
                    logfile=logfile, fakemode=fakemode, nocache=False)

            self.pipelines.append(mypipe)
            mypipe.run_all_recipes()

    def combine(self):
        pass
