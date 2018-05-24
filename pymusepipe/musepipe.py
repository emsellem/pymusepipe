# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS core module
"""
# For print - see compatibility with Python 3
from __future__ import print_function

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# This module has been largely inspired by one developed 
# by Kyriakos and Martina from the GTO MUSE MAD team
# and further rewritten by Mark van den Brok. Thanks to all three for this!
#
# Eric Emsellem adapted a version from early 2017, provided by Mark
# and adapted it for the needs of the PHANGS project (PI Schinnerer)

# Importing modules
import numpy as np

# Standard modules
import os
from os.path import join as joinpath
import time

# cpl module to link with esorex
try :
    import cpl
except ImportError :
    raise Exception("cpl is required for this - MUSE related - module")

# Pyfits from astropy
try :
    import astropy as apy
    from astropy.io import fits as pyfits
except ImportError :
    raise Exception("astropy is required for this module")

# ascii reading
try :
    from astropy.io import ascii
except ImportError :
    raise Exception("astropy.io.ascii is required for this module")

try :
    from astropy.table import Table
except ImportError :
    raise Exception("astropy.table.Table is required for this module")

import warnings
from astropy.utils.exceptions import AstropyWarning

# Importing pymusepipe modules
from init_musepipe import InitMuseParameters
from recipes_pipe import PipeRecipes
from create_sof import SofPipe

# Likwid command
likwid = "likwid-pin -c N:"

# Included an astropy table
__version__ = '0.2.0 (22 May 2018)'
#__version__ = '0.1.0 (03 April    2018)'
#__version__ = '0.0.2 (08 March    2018)'
#__version__ = '0.0.1 (21 November 2017)'

############################################################
#                      BEGIN
# The following parameters can be adjusted for the need of
# the specific pipeline to be used
############################################################

# NOTE: most of the parameters have now been migrated to
# init_musepipe.py for consistency.

listexpo_types = {'DARK': 'DARK', 'BIAS' : 'BIAS', 'FLAT,LAMP': 'FLAT',
        'FLAT,LAMP,ILLUM' : 'ILLUM', 'FLAT,SKY': 'TWILIGHT', 
        'WAVE': 'WAVE', 'STD': 'STD', 'AST': 'AST',
        'OBJECT': 'OBJECT', 'SKY': 'SKY'
        }

# This dictionary contains the types
dic_listMaster = {'DARK': 'MASTER_DARK', 
        'BIAS': 'MASTER_BIAS', 
        'FLAT': 'MASTER_FLAT',
        'TRACE': 'TRACE_TABLE',
        'TWILIGHT': 'TWILIGHT_CUBE', 
        'WAVE': 'WAVECAL_TABLE', 
        'LSF': 'LSF_PROFILE', 
        'STD': 'PIXTABLE_STD', 
        'OBJECT': 'DUMMY', 
        'SKY': 'DUMMY', 
        }

# listexpo_files = {"TYPE" : ['Type', 'ESO DPR TYPE', "raw_hdr_dprtype.txt"],
#         "DATE": ['MJD-OBS', "raw_hdr_mjd.txt"],
#         "MODE": ['HIERARCH ESO INS MODE', "raw_hdr_mode.txt"]
#         "TPLS": ['HIERARCH ESO TPL START', "raw_hdr_tpls.txt"]
#          }
listexpo_files = {
        "OBJECT" : ['object', 'OBJECT', str, '20A'],
        "TYPE" : ['type', 'ESO DPR TYPE', str, '20A'],
        "DATE":  ['mjd', 'MJD-OBS', np.float, 'E'],
        "MODE":  ['mode', 'ESO INS MODE', str, '10A'],
        "EXPTIME":  ['exptime', 'EXPTIME', float, 'E'],
        "TPLS":  ['tpls', 'ESO TPL START', str, '30A'],
        "TPLN":  ['tplnexp', 'ESO TPL NEXP', np.int, 'J'],
        "TPLNO":  ['tplno', 'ESO TPL EXPNO', np.int, 'J']
         }

exclude_list_checkmode = ['BIAS', 'DARK']

esorex_rc = "/home/soft/ESO/MUSE/muse-kit-2.2-5/esorex-3.12.3/etc/esorex.rc"
# Fits Raw table (default)
dic_files_tables = {'RAWFILES': 'RAWFILES_list_table.fits',
        'BIAS': 'MASTER_BIAS_list_table.fits',
        'FLAT': 'MASTER_FLAT_list_table.fits',
        'TRACE': 'MASTER_TRACE_list_table.fits',
        'WAVE': 'WAVE_list_table.fits',
        'TWILIGHT': 'TWILIGHT_list_table.fits',
        'STD': 'STD_list_table.fits',
        'LSF': 'LSF_list_table.fits',
        'SKY': 'SKY_list_table.fits',
        'OBJECT': 'OBJECT_list_table.fits'
        }

dic_files_products = {
        'STD': ['DATACUBE_STD', 'STD_FLUXES', 
            'STD_RESPONSE', 'STD_TELLURIC'],
        'TWILIGHT': ['DATACUBE_SKYFLAT', 'TWILIGHT_CUBE'],
        'SKY': ['SKY_SPECTRUM', 'PIXTABLE_REDUCED']
        }

list_scibasic = ['OBJECT', 'SKY', 'STD']
        
dic_geo_table = {
        '1900-01-01': "geometry_table_wfm.fits",
        '2000-01-01': "geometry_table_wfm.fits",
        '2014-12-01': "geometry_table_wfm.fits",
        '2015-04-16': "geometry_table_wfm.fits",
        '2015-09-08': "geometry_table_wfm.fits",
        }

dic_astro_table = {
        '1900-01-01': "astrometry_table_wfm.fits",
        '2000-01-01': "astrometry_table_wfm.fits",
        '2014-12-01': "astrometry_table_wfm.fits",
        '2015-04-16': "astrometry_table_wfm.fits",
        '2015-09-08': "astrometry_table_wfm.fits",
        }

future_date = '2099-01-01'

############################################################
#                      END
############################################################
def lower_allbutfirst_letter(mystring):
    """Lowercase all letters except the first one
    """
    return mystring[0].upper() + mystring[1:].lower()

def create_time_name() :
    """Create a time-link name for file saving purposes

    Return: a string including the YearMonthDay_HourMinSec
    """
    return str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))

def formatted_time() :
    """ Return: a string including the formatted time
    """
    return str(time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()))

def get_date_inD(indate) :
    """Transform date in Y-M-D
    """
    return np.datetime64(indate).astype('datetime64[D]')

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
        print_endline("... Done", end='\n')
    except OSError:
        if not os.path.isdir(path):
            print_error("Failed to create folder! Please check the path")
            raise
            return
        if os.path.isdir(path):
            print_endline("... Folder already exists, doing nothing.")

def append_file(filename, content):
    """Append in ascii file
    """
    with open(filename, "a") as myfile:
        myfile.write(content)
        
def overwrite_file(filename, content):
    """Overwite in ascii file
    """
    with open(filename, "w+") as myfile:
        myfile.write(content)
        
def add_listpath(suffix, paths) :
    """Add a suffix to a list of path
    and normalise them
    """
    newlist = []
    for mypath in paths:
        newlist.append(normpath(joinpath(suffix, mypath)))
    return newlist

def norm_listpath(paths) :
    """Normalise the path for a list of paths
    """
    newlist = []
    for mypath in paths:
        newlist.append(normpath(mypath))
    return newlist

def normpath(path) :
    """Normalise the path to get it short
    """
    return os.path.relpath(os.path.realpath(path))

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[1;31;20m'
INFO = '\033[1;32;20m'
ERROR = '\033[91m'
ENDC = '\033[0m'
BOLD = "\033[1m"
def print_endline(text, **kwargs) :
    print(INFO + text + ENDC, **kwargs)

def print_warning(text, **kwargs) :
    print(WARNING + "# MusePipeWarning " + ENDC + text, **kwargs)

def print_info(text, **kwargs) :
    print(INFO + "# MusePipeInfo " + ENDC + text, **kwargs)

def print_error(text, **kwargs) :
    print(ERROR + "# MusePipeError " + ENDC + text, **kwargs)

#########################################################################
# Useful Classes for the Musepipe
#########################################################################
class MyDict(dict) :
    """New Dictionary with extra attributes
    """
    def __init__(self) :
        dict.__init__(self)

class PipeObject(object) :
    """New class to store the tables
    """
    def __init__(self, info=None) :
        """Initialise the nearly empty class
        Add _info for a description if needed
        """
        self._info = info

def lower_rep(text) :
    return text.replace("_","").lower()

#########################################################################
# Main class
#                           MusePipe
#########################################################################
    
class MusePipe(PipeRecipes, SofPipe):
    """Main Class to define and run the MUSE pipeline, given a certain galaxy name
    
    musep = MusePipe(galaxyname='NGC1087', rc_filename="", cal_filename="", 
                      outlog"NGC1087_log1.log", objects=[''])
    musep.run()
    """

    def __init__(self, galaxyname=None, pointing=0, objectlist=[], rc_filename=None, 
            cal_filename=None, outlog=None, logfile="MusePipe.log", reset_log=False,
            verbose=True, mode="WFM-NOAO-N", checkmode=True, 
            strong_checkmode=False, **kwargs):
        """Initialise the file parameters to be used during the run

        Input
        -----
        galaxyname: string (e.g., 'NGC1208'). default is None. 
        objectlist= list of objects (string=filenames) to process. Default is empty

        rc_filename: filename to initialise folders
        cal_filename: filename to initiale FIXED calibration MUSE files
        outlog: string, output directory for the log files
        verbose: boolean. Give more information as output (default is True)
        mode: string (default is WFM_N) String to define the mode to be considered
        checkmode: boolean (default is True) Check the mode or not when reducing
        strong_checkmode: boolean (default is False) Enforce the checkmode for all if True, 
                         or exclude DARK/BIAS if False

        Other possible entries
        ----------------------
        overwrite_astropy_table: boolean (default is False). Overwrite the astropy table even when
            it exists.
        warnings: strong  ('ignore'by default. If set to ignore, will ignore the Astropy Warnings.
        time_geo_table: boolean (default is True). Use the time dependent geo_table
        time_astro_table: boolean (default is True). Use the time dependent astro_table

        """
        # Verbose option
        self.verbose = verbose

        # Warnings for astropy
        self.warnings = kwargs.pop("warnings", 'ignore')
        if self.warnings == 'ignore':
           warnings.simplefilter('ignore', category=AstropyWarning)

        # Overwriting option for the astropy table
        self._overwrite_astropy_table = kwargs.pop("overwrite_astropy_table", False)

        # Use time dependent geo_table
        self._time_geo_table = kwargs.pop("time_geo_table", True)
        self._time_astro_table = kwargs.pop("time_astro_table", True)

        PipeRecipes.__init__(self, **kwargs)
        SofPipe.__init__(self)

        # Setting the default attibutes #####################
        self.galaxyname = galaxyname
        self.pointing = pointing

        # Setting other default attributes
        if outlog is None : 
            outlog = "log_{timestamp}".format(timestamp=create_time_name())
            print_info("The Log folder will be {log}".format(outlog))
        self.outlog = outlog
        self.logfile = joinpath(self.outlog, logfile)

        # Mode of the observations
        self.mode = mode
        # Checking if mode is correct
        self.checkmode = checkmode
        # Checking if mode is correct also for BIAS & DARK
        self.strong_checkmode = strong_checkmode

        # Set of objects to reduced
        self.objectlist = objectlist
        # End of parameter settings #########################

        # =========================================================== #
        # Setting up the folders and names for the data reduction
        # Can be initialised by either an rc_file, 
        # or a default rc_file or harcoded defaults.
        self.my_params = InitMuseParameters(rc_filename=rc_filename, 
                            cal_filename=cal_filename)

        # Setting up the relative path for the data, using Galaxy Name + Pointing
        self.my_params.data = "{0}/P{1:02d}/".format(self.galaxyname, self.pointing)

        # Create full path folder 
        self.set_fullpath_names()

        # Go to the data directory
        # and Recording the folder where we start
        self.paths.orig = os.getcwd()

        # Making the output folders in a safe mode
        if self.verbose:
            print_info("Creating directory structure")
        self.goto_folder(self.paths.data)

        # ----------------------------------------------
        # Creating the Astropy file Table folder
        # ----------------------------------------------
        safely_create_folder(self.my_params.folder_filetables, verbose=verbose)

        # ==============================================
        # Creating the folder structure itself if needed
        for folder in self.my_params._dic_folders.keys() :
            safely_create_folder(self.my_params._dic_folders[folder], verbose=verbose)

        # Init the Master exposure flag dictionary
        self.Master = {}
        for mastertype in dic_listMaster.keys() :
            safely_create_folder(self._get_path_master(mastertype), verbose=self.verbose)
            self.Master[mastertype] = False
        # ==============================================

        # Going back to initial working directory
        self.goto_prevfolder()

        # ===========================================================
        # Now creating the raw table, and attribute containing the
        # astropy dataset probing the rawfiles folder
        self._reset_tables()
        if verbose :
            print_info("Creating the astropy fits raw data table")
        # When creating the table, if the table already exists
        # it will read the old one, except if an overwrite_astropy_table
        # is set to True.
        self.create_raw_table()
        # ===========================================================

    def goto_prevfolder(self, logfile=False) :
        """Go back to previous folder
        """
        print_info("Going back to the original folder {0}".format(self.paths._prev_folder))
        self.goto_folder(self.paths._prev_folder, logfile=logfile, verbose=False)
            
    def goto_folder(self, newpath, logfile=False, verbose=True) :
        """Changing directory and keeping memory of the old working one
        """
        try: 
            prev_folder = os.getcwd()
            newpath = os.path.normpath(newpath)
            os.chdir(newpath)
            if verbose :
                print_info("Going to folder {0}".format(newpath))
            if logfile :
                append_file(joinpath(self.paths.data, self.logfile), "cd {0}\n".format(newpath))
            self.paths._prev_folder = prev_folder 
        except OSError:
            if not os.path.isdir(newpath):
                raise
    
    def set_fullpath_names(self) :
        """Create full path names to be used
        """
        # initialisation of the full paths 
        self.paths = PipeObject("All Paths useful for the pipeline")
        self.paths.root = self.my_params.root
        self.paths.data = joinpath(self.paths.root, self.my_params.data)
        self.paths.filetables = joinpath(self.paths.data, self.my_params.folder_filetables)
        for name in self.my_params._dic_folders.keys() + self.my_params._dic_input_folders.keys():
            setattr(self.paths, name, joinpath(self.paths.data, getattr(self.my_params, name)))

        # Creating the filenames for Master files
        self.masterfiles = PipeObject("Information pertaining to the Masterfiles")
        for expotype in dic_listMaster.keys() :
            name_attr = self._get_attr_master(expotype)
            # Adding the path of the folder
            setattr(self.paths, name_attr, joinpath(self.paths.data, self._get_path_master(expotype)))
            # Adding the full path name for the master files
            setattr(self.masterfiles, expotype.lower(), self._get_suffix_master(expotype))

    def read_file_table(self, filetype=None, **kwargs):
        """Read an existing RAW data table to start the pipeline
        """

        l_filetype = lower_allbutfirst_letter(filetype)
        folder = self.paths.filetables
        name_table = kwargs.pop('name_table', dic_files_tables[filetype])

        # Check the raw folder
        self.goto_folder(folder)

        # Read the astropy table
        if not os.path.isfile(name_table):
            print_info("ERROR: file table {0} does not exist".format(name_table))
        else :
            if self.verbose : print_info("Reading fits Table {0}".format(name_table))
            setattr(self.Tables, l_filetype, Table.read(name_table, format="fits"))
        
        # Going back to the original folder
        self.goto_prevfolder()

    def create_raw_table(self, overwrite_astropy_table=None, verbose=None, **kwargs) :
        """ Create a fits table with all the information from
        the Raw files
        Also create an astropy table with the same info
        """
        if verbose is None: verbose = self.verbose
        if overwrite_astropy_table is not None: self._overwrite_astropy_table = overwrite_astropy_table
        name_table = kwargs.pop('name_table', dic_files_tables['RAWFILES'])
        name_table = joinpath(self.paths.filetables, name_table)

        rawfolder = self.paths.rawfiles

        # Testing if raw table exists
        if os.path.isfile(name_table) :
            if self._overwrite_astropy_table :
                print_warning("The raw-files table will be overwritten")
            else :
                print_warning("The raw files table already exists")
                print_warning("If you wish to overwrite it, "
                      " please turn on the 'overwrite_astropy_table' option to 'True'")
                print_warning("In the meantime, the existing table will be read and used")
                self.read_file_table(filetype='RAWFILES')
                self.sort_types()
                return

        # Check the raw folder
        self.goto_folder(self.paths.rawfiles)
        # Get the list of files from the Raw data folder
        files = os.listdir(".")

        smalldic = {"FILENAME" : ['filename', '', str, '100A']}
        fulldic = listexpo_files.copy()
        fulldic.update(smalldic)

        # Init the lists
        MUSE_infodic = {}
        for key in fulldic.keys() :
            MUSE_infodic[key] = []

        # Looping over the files
        for f in files:
            # Excluding the files without MUSE and fits.fz
            if ('MUSE' in f) and ('.fits.fz')  in f:
                MUSE_infodic['FILENAME'].append(f)
                header = pyfits.getheader(f, 0)
                for k in listexpo_files.keys() :
                    [namecol, keyword, func, form] = listexpo_files[k]
                    MUSE_infodic[k].append(func(header[keyword]))

        # Transforming into numpy arrayimport pymusepipe
        for k in fulldic.keys() :
            MUSE_infodic[k] = np.array(MUSE_infodic[k])

        # Getting a sorted array with indices
        idxsort = np.argsort(MUSE_infodic['FILENAME'])

        # Creating the astropy table
        self.Tables.Rawfiles = Table([MUSE_infodic['FILENAME'][idxsort]], names=['filename'], meta={'name': 'raw file table'})

        # Creating the columns
        for k in fulldic.keys() :
            [namecol, keyword, func, form] = fulldic[k]
            self.Tables.Rawfiles[namecol] = MUSE_infodic[k][idxsort]

        # Writing up the table
        self.Tables.Rawfiles.write(name_table, format="fits", overwrite=self._overwrite_astropy_table)

        # Sorting the type
        self.sort_types()
        # Going back to the original folder
        self.goto_prevfolder()

    def _reset_tables(self) :
        """Reseting the astropy Tables for expotypes
        """
        # Reseting the select_type item
        self.Tables = PipeObject("File astropy tables")
        for expotype in listexpo_types.values() :
            setattr(self.Tables, expotype, [])

    def sort_types(self, checkmode=None, strong_checkmode=None, reset=False) :
        """Provide lists of exposures with types defined in the dictionary
        """
        # Reseting the list if reset is True (default)
        if reset: self._reset_tables()

        if checkmode is not None : self.checkmode = checkmode
        else : checkmode = self.checkmode

        if strong_checkmode is not None : self.strong_checkmode = strong_checkmode
        else : strong_checkmode = self.strong_checkmode

        # Sorting alphabetically (thus by date)
        for keytype in listexpo_types.keys() :
            expotype = listexpo_types[keytype].upper()
            try :
                mask = (self.Tables.Rawfiles['type'] == keytype)
                setattr(self.Tables, expotype, self.Tables.Rawfiles[mask])
            except AttributeError:
                pass

    def select_expotype_fromraw(self, expotype=None):
        """ This will return the info Table of raw files corresponding 
        to the given expotype.
        """
        return getattr(self.Tables, expotype)

    def _get_attr_master(self, expotype):
        return "master{0}".format(expotype.lower())

    def _get_table_master(self, expotype) :
        return getattr(self.Tables, self._get_attr_master(expotype))

    def _get_name_master(self, expotype) :
        return normpath(getattr(self.masterfiles, expotype.lower()))

    def _get_suffix_master(self, expotype):
        return dic_listMaster[expotype]

    def _get_path_master(self, expotype):
        masterfolder = lower_allbutfirst_letter(expotype)
        return joinpath(self.my_params.master, masterfolder)
    
    def _get_fullpath_master(self, expotype) :
        return normpath(getattr(self.paths, self._get_attr_master(expotype)))

    def _get_path_files(self, expotype) :
        return normpath(getattr(self.paths, expotype.lower()))

    def select_closest_mjd(self, mjdin, group_table) :
        """Get the closest frame within the expotype
        If the attribute does not exist in Tables, it tries to read
        the table from the folder
        """
        # Get the closest tpl
        index = np.argmin((mjdin - group_table['mjd'])**2)
        closest_tpl = group_table[index]['tpls']
        return index, closest_tpl

    def add_list_tplmaster_to_sofdict(self, mean_mjd, list_expotype):
        """Add a list of masterfiles to the SOF
        """
        for expotype in list_expotype :
            self.add_tplmaster_to_sofdict(mean_mjd, expotype)

    def add_tplmaster_to_sofdict(self, mean_mjd, expotype, reset=False):
        """ Add item to dictionary for the sof writing
        """
        if reset: self._sofdict.clear()
        # Finding the best tpl for this master
        index, this_tpl = self.select_closest_mjd(mean_mjd, self._get_table_master(expotype)) 
        dir_master = self._get_fullpath_master(expotype)
        self._sofdict[self._get_suffix_master(expotype)] = [joinpath(dir_master, self._get_name_master(expotype) + "_" + this_tpl + ".fits")]

    def add_tplraw_to_sofdict(self, mean_mjd, expotype, reset=False):
        """ Add item to dictionary for the sof writing
        """
        if reset: self._sofdict.clear()
        # Finding the best tpl for this raw file type
        expo_table = getattr(self.Tables, expotype)
        index, this_tpl = self.select_closest_mjd(mean_mjd, expo_table) 
        self._sofdict[expotype] = [expo_table['filename'][index]]

    def add_skycalib_to_sofdict(self, tag, mean_mjd, expotype, reset=False):
        """ Add item to dictionary for the sof writing
        """
        if reset: self._sofdict.clear()
        # Finding the best tpl for this sky calib file type
        expo_table = getattr(self.Tables, expotype)
        index, this_tpl = self.select_closest_mjd(mean_mjd, expo_table) 
        dir_master = self._get_fullpath_master(expotype)
        self._sofdict[tag] = [joinpath(dir_master, "{0}_{1}.fits".format(tag, this_tpl))]

    def add_calib_to_sofdict(self, calibtype, reset=False):
        """Adding a calibration file for the SOF 
        """
        if reset: self._sofdict.clear()
        calibfile = getattr(self.my_params, calibtype.lower())
        self._sofdict[calibtype] = [joinpath(self.my_params.musecalib, calibfile)]

    def add_geometry_to_sofdict(self, tpls):
        """Extract the geometry table and add it to the dictionary
        for the SOF file
        """
        calfolder = self.my_params.musecalib
        if self._time_geo_table :
            listkeys = list(dic_geo_table.keys())
            listkeys.append(future_date)
            for ikey in range(len(listkeys) - 1):
                if tpls >= listkeys[ikey] and tpls < listkeys[ikey+1]:
                    geofile = dic_geo_table[listkeys[ikey]]
        else:
            geofile = self.my_params.geo_table

        self._sofdict['GEOMETRY_TABLE']=["{folder}{geo}".format(folder=calfolder, geo=geofile)]

    def add_astrometry_to_sofdict(self, tpls):
        """Extract the astrometry table and add it to the dictionary
        for the SOF file
        """
        calfolder = self.my_params.musecalib
        if self._time_geo_table :
            listkeys = list(dic_astro_table.keys())
            listkeys.append(future_date)
            for ikey in range(len(listkeys) - 1):
                if tpls >= listkeys[ikey] and tpls < listkeys[ikey+1]:
                    astrofile = dic_astro_table[listkeys[ikey]]
        else :
            astrofile = self.my_params.astro_table

        self._sofdict['ASTROMETRY_WCS']=["{folder}{astro}".format(folder=calfolder, astro=astrofile)]

    def save_master_table(self, expotype, tpl_gtable, fits_tablename=None):
        """Save the Master Table corresponding to the mastertype
        """
        mastertype = self._get_attr_master(expotype)
        if fits_tablename is None :
            fits_tablename = dic_files_tables[expotype]
        setattr(self.Tables, mastertype, tpl_gtable.groups.aggregate(np.mean)['tpls','mjd', 'tplnexp'])
        full_tablename = joinpath(getattr(self.paths, mastertype), fits_tablename)
        setattr(self.Tables, mastertype + "name", full_tablename)
        if (not self._overwrite_astropy_table) and os.path.isfile(full_tablename):
            print_warning("Astropy Table {0} already exists, "
                " use overwrite_astropy_table to overwrite it".format(mastertype.upper()))
        else :
            getattr(self.Tables, mastertype).write(full_tablename, 
                format="fits", overwrite=self._overwrite_astropy_table)

    def get_tpl_meanmjd(self, gtable):
        """Get tpl of the group and mean mjd of the group
        """
        tpl = gtable['tpls'][0]
        mean_mjd = gtable.groups.aggregate(np.mean)['mjd'].data[0]
        return tpl, mean_mjd

    def select_tpl_files(self, expotype=None, tpl="ALL"):
        """Selecting a subset of files from a certain type
        """
        if expotype not in listexpo_types.keys() :
            print_info("ERROR: input expotype is not in the list of possible values")
            return 

        MUSE_subtable = self.select_expotype_fromraw(listexpo_types[expotype])
        if len(MUSE_subtable) == 0:
            if self.verbose :
                print_warning("Empty file table of type {0}".format(expotype))
                print_warning("Returning an empty Table from the tpl -astropy- selection")
            return MUSE_subtable

        group_table = MUSE_subtable.group_by('tpls')
        if tpl == "ALL":
            return group_table
        else :
            return group_table.groups[group_table.groups.key['tpls'] == tpl]

    def run_bias(self, sof_filename='bias', tpl="ALL", fits_tablename=None):
        """Reducing the Bias files and creating a Master Bias
        Will run the esorex muse_bias command on all Biases

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='BIAS', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                print_warning("No BIAS recovered from the astropy Table - Aborting")
                return
        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the BIASES including
        # the list of files to be processed for one MASTER BIAS
        # Adding the bad pixel table
        self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['BIAS'] = add_listpath(self.paths.rawfiles,
                    gtable['filename'].data.astype(np.object))
            # extract the tpl (string)
            tpl = gtable['tpls'][0]
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Bias
            name_masterbias = self._get_suffix_master('BIAS')
            self._get_name_master('BIAS')
            # Run the recipe
            self.recipe_bias(self.current_sof, name_masterbias, tpl)

        # Write the MASTER BIAS Table and save it
        self.save_master_table('BIAS', tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_flat(self, sof_filename='flat', tpl="ALL", fits_tablename=None):
        """Reducing the Flat files and creating a Master Flat
        Will run the esorex muse_flat command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='FLAT,LAMP', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                print_warning("No FLAT recovered from the astropy Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the FLATs including
        # the list of files to be processed for one MASTER Flat
        # Adding the bad pixel table
        self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['FLAT'] = add_listpath(self.paths.rawfiles,
                    gtable['filename'].data.astype(np.object))
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self.get_tpl_meanmjd(gtable)
            # Adding the best tpc MASTER_BIAS
            self.add_tplmaster_to_sofdict(mean_mjd, 'BIAS')
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Flat and Trace Table
            name_masterflat =  self._get_suffix_master('FLAT')
            name_tracetable = self._get_suffix_master('TRACE')
            # Run the recipe
            self.recipe_flat(self.current_sof, name_masterflat, name_tracetable, tpl)

        # Write the MASTER FLAT Table and save it
        self.save_master_table('FLAT', tpl_gtable, fits_tablename)
        self.save_master_table('TRACE', tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_wave(self, sof_filename='wave', tpl="ALL", fits_tablename=None):
        """Reducing the WAVE-CAL files and creating the Master Wave
        Will run the esorex muse_wave command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='WAVE', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                print_warning("No WAVE recovered from the astropy file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the FLATs including
        # the list of files to be processed for one WAVECAL
        # Adding Badpix table and Line Catalog
        self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        self.add_calib_to_sofdict("LINE_CATALOG")
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['ARC'] = add_listpath(self.paths.rawfiles,
                    gtable['filename'].data.astype(np.object))
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self.get_tpl_meanmjd(gtable)
            # Finding the best tpl for BIAS + TRACE
            self.add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'TRACE'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Wave
            name_masterwave = self._get_suffix_master('WAVE')
            # Run the recipe
            self.recipe_wave(self.current_sof, name_masterwave, tpl)

        # Write the MASTER WAVE Table and save it
        self.save_master_table('WAVE', tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_lsf(self, sof_filename='lsf', tpl="ALL", fits_tablename=None):
        """Reducing the LSF files and creating the LSF PROFILE
        Will run the esorex muse_lsf command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='WAVE', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                print_warning("No WAVE recovered from the astropy file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        # Adding Badpix table and Line Catalog
        self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        self.add_calib_to_sofdict("LINE_CATALOG")
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['ARC'] = add_listpath(self._get_fullpath_master('WAVE'),
                    gtable['filename'].data.astype(np.object))
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self.get_tpl_meanmjd(gtable)
            # Finding the best tpl for BIAS, TRACE, WAVE
            self.add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'TRACE', 'WAVE'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Wave
            name_masterlsf = self._get_suffix_master('LSF')
            # Run the recipe
            self.recipe_lsf(self.current_sof, name_masterlsf, tpl)

        # Write the MASTER LSF PROFILE Table and save it
        self.save_master_table('LSF', tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_twilight(self, sof_filename='twilight', tpl="ALL", fits_tablename=None):
        """Reducing the  files and creating the TWILIGHT CUBE.
        Will run the esorex muse_twilight command on all TWILIGHT

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='FLAT,SKY', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                print_warning("No TWILIGHT recovered from the astropy file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        self.add_calib_to_sofdict("VIGNETTING_MASK")
        for gtable in tpl_gtable.groups:
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self.get_tpl_meanmjd(gtable)
            self.add_geometry_to_sofdict(tpl)
            # Provide the list of files to the dictionary
            self._sofdict['TWILIGHT'] = add_listpath(self._get_fullpath_master('TWILIGHT'),
                    gtable['filename'].data.astype(np.object))
            # Finding the best tpl for BIAS, ILLUM, FLAT, TRACE, WAVE
            self.add_tplraw_to_sofdict(mean_mjd, "ILLUM")
            self.add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'FLAT', 'TRACE', 'WAVE'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Wave
            name_mastertwilight = self._get_suffix_master('TWILIGHT')
            # Run the recipe
            name_products = dic_files_products['TWILIGHT']
            dir_products = self._get_fullpath_master('TWILIGHT')
            self.recipe_twilight(self.current_sof, name_mastertwilight, dir_products, name_products, tpl)

        # Write the MASTER TWILIGHT Table and save it
        self.save_master_table('TWILIGHT', tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_scibasic_all(self, list_object=list_scibasic, tpl="ALL"):
        """Running scibasic for all objects in list_object
        Making different sof for each category
        """
        for expotype in list_object:
            sof_filename = 'scibasic_{0}'.format(expotype.lower())
            self.run_scibasic(sof_filename=sof_filename, expotype=expotype, tpl=tpl)

    def run_scibasic(self, sof_filename='scibasic', expotype="OBJECT", tpl="ALL", fits_tablename=None):
        """Reducing the files of a certain category and creating the PIXTABLES
        Will run the esorex muse_scibasic 

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype=expotype, tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                print_warning("No {0} recovered from the astropy file Table - Aborting".format(expotype))
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        for gtable in tpl_gtable.groups:
            self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
            self.add_calib_to_sofdict("LINE_CATALOG")
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self.get_tpl_meanmjd(gtable)
            self.add_geometry_to_sofdict(tpl)
            # Provide the list of files to the dictionary
            self._sofdict[expotype] = add_listpath(self._get_path_files(expotype),
                    gtable['filename'].data.astype(np.object))
            # Finding the best tpl for BIAS
            self.add_tplraw_to_sofdict(mean_mjd, "ILLUM") 
            self.add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'FLAT', 
                'TRACE', 'WAVE', 'TWILIGHT'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Run the recipe to reduce the standard (muse_scibasic)
            if expotype == 'STD':
                dir_products = self._get_fullpath_master('STD')
                name_products = ['PIXTABLE_STD_0001-{0:02d}.fits'.format(j+1) for j in range(24)]
            else:
                dir_products = None
                name_products=None
            self.recipe_scibasic(self.current_sof, dir_products, name_products, tpl)

        # Write the MASTER files Table and save it
        self.save_master_table(expotype, tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_std(self, sof_filename='standard', tpl="ALL"):
        """Reducing the STD files after they have been obtained
        Running the muse_standard routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        std_table = self.Tables.masterstd
        if len(std_table) == 0:
            if self.verbose :
                print_warning("No STD recovered from the astropy file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the STD Sof
        for i in range(len(std_table)):
            mytpl = std_table['tpls'][i]
            if tpl != "ALL" and tpl != mytpl :
                continue
            # Now starting with the standard recipe
            self.add_calib_to_sofdict("EXTINCT_TABLE", reset=True)
            self.add_calib_to_sofdict("STD_FLUX_TABLE")
            dir_products = self._get_fullpath_master('STD')
            self._sofdict['PIXTABLE_STD'] = [joinpath(dir_products, 
                'PIXTABLE_STD_{0}-{1:02d}.fits'.format(mytpl, j+1)) for j in range(24)]
            self.write_sof(sof_filename=sof_filename + "_" + mytpl, new=True)
            name_products = dic_files_products['STD']
            self.recipe_std(self.current_sof, dir_products, name_products, mytpl)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_sky(self, sof_filename='sky', tpl="ALL", fraction=0.8):
        """Reducing the SKY after they have been scibasic reduced
        Will run the esorex muse_create_sky routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        sky_table = self.Tables.mastersky
        if len(sky_table) == 0:
            if self.verbose :
                print_warning("No SKY recovered from the astropy file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        for i in range(len(sky_table)):
            mytpl = sky_table['tpls'][i]
            mymjd = sky_table['mjd'][i]
            if tpl != "ALL" and tpl != mytpl :
                continue
            # Now starting with the standard recipe
            self.add_calib_to_sofdict("EXTINCT_TABLE", reset=True)
            self.add_calib_to_sofdict("SKY_LINES")
            self.add_skycalib_to_sofdict("STD_RESPONSE", mymjd, 'STD')
            self.add_skycalib_to_sofdict("STD_TELLURIC", mymjd, 'STD')
            self.add_tplmaster_to_sofdict(mymjd, 'LSF')
            dir_products = self._get_fullpath_master('SKY')
            self._sofdict['PIXTABLE_SKY'] = [joinpath(dir_products,
                'PIXTABLE_SKY_{0:04d}-{1:02d}.fits'.format(i+1,j+1)) for j in range(24)]
            self.write_sof(sof_filename=sof_filename + "{0:02d}".format(i+1) + "_" + mytpl, new=True)
            name_products = dic_files_products['SKY']
            self.recipe_sky(self.current_sof, dir_products, name_products, mytpl, fraction=fraction)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_all_recipes(self, fraction=0.8):
        """Running all recipes in one shot
        """
        self.run_bias()
        self.run_flat()
        self.run_wave()
        self.run_lsf()
        self.run_twilight()
        self.run_scibasic_all()
        self.run_std()
        self.run_sky(fraction=fraction)

#    def create_calibrations(self):
#        self.check_for_calibrations()
#        # MdB: this creates the standard calibrations, if needed
#        if not ((self.redo==0) and (self.Master['BIAS']==1)):
#            self.run_bias()
#        if not ((self.redo==0) and (self.Master['DARK']==1)):
#            self.run_dark()
#        if not ((self.redo==0) and (self.Master['FLAT']==1)):
#            self.run_flat()
#        if not ((self.redo==0) and (self.Master['WAVE']==1)):
#            self.run_wavecal()
#        if not ((self.redo==0) and (self.Master['TWILIGHT']==1)):
#            self.run_twilight()    
#        if not ((self.redo==0) and (self.Master['STD']==1)):
#            self.create_standard_star()
#            self.run_standard_int()
#
#    def check_for_calibrations(self, verbose=None):
#        """This function checks which calibration file are present, just in case
#        you do not wish to redo them. Variables are named: 
#        masterbias, masterdark, standard, masterflat, mastertwilight
#        """
#        
#        outcal = getattr(self.my_params, "master" + suffix_folder)
#        if verbose is None: verbose = self.verbose
#
#        for mastertype in dic_listMaster.keys() :
#            [fout, suffix] = dic_listMaster[mastertype]
#            # If TWILIGHT = only one cube
#            if mastertype == "TWILIGHT" :
#                if not os.path.isfile(joinpath(outcal, fout, "{suffix}.fits".format(suffix=suffix))):
#                    self.Master[mastertype] = False
#
#                if verbose :
#                    if self.Master[mastertype] :
#                        print("WARNING: Twilight flats already made")
#                    else :
#                        print("WARNING: Twilight flats NOT there yet")
#
#            # Others = 24 IFU exposures
#            else :
#                for ifu in range(1,25):
#                    if not os.path.isfile(joinpath(outcal, fout, "{0}-{1:02d}.fits".format(suffix, ifu))):
#                        self.Master[mastertype] = False
#                        break
#                if verbose :
#                    if self.Master[mastertype]:
#                        print("WARNING: Master {0} already in place".format(mastertype))
#                    else :
#                        print("WARNING: Master {0} NOT yet there".format(mastertype))
#
#    def select_stdfile(self):
#        """Minimise the time difference between the object and the illumination
#        """
#        
#        if len(self.stdlist) == 0:
#            self.stdfile=''
#            return
#        
#        # Initialisation of the time array
#        std_times = []
#        # Transforming the list into an array
#        stdlist = np.array(self.stdlist)
#        for std in stdlist:
#            std_times.append(np.datetime64(std[-28:-5]))
#        # Transforming the list into an array
#        std_times = np.array(std_times)
#
#        for object in self.objectlist: 
#            object_time = np.datetime64(object[-28:-5])
#            DeltaTime = np.abs(std_times - object_time)
#            mask = (DeltaTime == np.min(DeltaTime))
#            self.stdfile = stdlist[mask][0]
#
#    def select_run_tables(self, runname="Run01", verbose=None):
#        """
#        """
#        if len(self.objectlist) < 1 : 
#            print("WARNING: objectlist is empty, hence no CAL tables defined")
#            return
#
#        filename = self.objectlist[0]
#        hdu = pyfits.open(filename)
#        filetime = np.datetime64(hdu[0].header['DATE-OBS']).astype('datetime64[D]')
#        for runs in MUSEPIPE_runs.keys():
#            Pdates = MUSEPIPE_runs[runs]
#            date1 = np.datetime64(Pdates[1]).astype('datetime64[D]')
#            date2 = np.datetime64(Pdates[2]).astype('datetime64[D]')
#            if ((filetime >= date1) and (filetime <= date2)):
#                runname = runs
#                break
#        if verbose is None: verbose = self.verbose
#        if verbose:
#            print "Using geometry calibration data from MUSE runs %s\n"%finalkey
#        self.geo_table = joinpath(getattr(self.my_params, "musecalib" + suffix_folder), 
#                "geometry_table_wfm_{runname}.fits".format(runname=runname)) 
#        self.astro_table = joinpath(getattr(self.my_params, "musecalib" + suffix_folder), 
#                "astrometry_wcs_wfm_{runname}.fits".format(runname=runname))
#
#    def select_illumfiles(self):
#        """Minimise the difference between the OBJECT and ILLUM files
#        """
#        self.final_illumlist=[]
#        if len(self.illumlist) == 0:
#            return
#        illum_times=[]
#        illumlist=np.array(self.illumlist)
#        for illum in illumlist:
#            illum_times.append(np.datetime64(illum[-28:-5]))
#        illum_times=np.array(illum_times)
#        for object in self.objectlist: 
#            object_time=np.datetime64(object[-28:-5])
#            DeltaTime=np.abs(illum_times - object_time)
#            mask = (DeltaTime==np.min(DeltaTime))
#            self.final_illumlist.append(illumlist[mask][0])
#        
