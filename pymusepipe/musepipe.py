# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS core module
"""

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
import collections
from collections import OrderedDict
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

# Importing pymusepipe modules
from init_musepipe import InitMuseParameters
from recipes_pipe import PipeRecipes
from create_sof import SofPipe

# Likwid command
likwid = "likwid-pin -c N:"

# Included an astropy table
__version__ = '0.1.0 (03 April    2018)'

__version__ = '0.0.2 (08 March    2018)'
__version__ = '0.0.1 (21 November 2017)'

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
dic_listMaster = {'DARK': ['Dark', 'MASTER_DARK'], 
        'BIAS': ['Bias', 'MASTER_BIAS'], 
        'FLAT': ['Flat', 'MASTER_FLAT'],
        'TRACE': ['Trace', 'TRACE_TABLE'],
        'TWILIGHT': ['Twilight', 'TWILIGHT_CUBE'], 
        'WAVE': ['Wave', 'WAVECAL_TABLE'], 
        'LSF': ['Lsf', 'LSF_PROFILE'], 
        'STD': ['Std', 'PIXTABLE_STD'], 
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
dic_files_tables = {'rawfiles': 'rawfiles_list_table.fits',
        'masterbias': 'MASTER_BIAS_list_table.fits',
        'masterflat': 'MASTER_FLAT_list_table.fits',
        'masterwave': 'WAVE_list_table.fits'
        }

############################################################
#                      END
############################################################
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
        if verbose : print("Input path is None, not doing anything")
        return
    if verbose : 
        print("Trying to create {folder} folder".format(folder=path))
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

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

class MyDict(dict) :
    """New Dictionary with extra attributes
    """
    pass

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
            cal_filename=None, outlog=None, logfile="MusePipe.log", verbose=True, 
            redo=False, align_file=None, mode="WFM-NOAO-N", checkmode=True, 
            strong_checkmode=False, overwrite_table=False, create_raw_table=True, **kwargs):
        """Initialise the file parameters to be used during the run

        Input
        -----
        galaxyname: string (e.g., 'NGC1208'). default is None. 
        objectlist= list of objects (string=filenames) to process. Default is empty

        rc_filename: filename to initialise folders
        cal_filename: filename to initiale FIXED calibration MUSE files
        outlog: string, output directory for the log files
        verbose: boolean. Give more information as output (default is True)
        redo: boolean. Default is False = do not reduce existing calibrations by default
        align_file: (default is None). 
        mode: (default is WFM_N) String to define the mode to be considered
        checkmode: (default is True) Check the mode or not when reducing
        strong_checkmode: (default is False) Enforce the checkmode for all if True, 
                         or exclude DARK/BIAS if False
        """
        self.verbose = verbose
        PipeRecipes.__init__(self, **kwargs)

        # Setting the default attibutes #####################
        self.galaxyname = galaxyname
        self.pointing = pointing

        # Setting other default attributes
        if outlog is None : 
            outlog = "log_{timestamp}".format(timestamp=create_time_name())
            print("The Log folder will be {log}".format(outlog))
        self.outlog = outlog
        self.logfile = joinpath(self.outlog, logfile)

        self.redo = redo
        self.align_file = align_file
        self.overwrite_table = overwrite_table

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
            print("Creating directory structure")
        self.goto_folder(self.paths.data)

        # ==============================================
        # Creating the folder structure itself if needed
        for folder in self.my_params._dic_folders.keys() :
            safely_create_folder(self.my_params._dic_folders[folder], verbose=verbose)

        # Init the Master exposure flag dictionary
        self.Master = {}
        for mastertype in dic_listMaster.keys() :
            [masterfolder, mastername] = dic_listMaster[mastertype]
            safely_create_folder(joinpath(self.my_params.master, masterfolder), 
                    verbose=self.verbose)
            self.Master[mastertype] = False
        # ==============================================

        # Going back to initial working directory
        self.goto_prevfolder()

        # ===========================================================
        # Now creating the raw table, and attribute containing the
        # astropy dataset probing the rawfiles folder
        self.Tables = lambda:None
        if create_raw_table :
            if verbose :
                print("Creating the astropy fits raw data table")
            self.create_raw_table()
        else :
            if verbose :
                print("Reading the existing astropy fits raw data table")
            self.read_file_table(filetype='rawfiles')
            self.sort_types()
        # ===========================================================

    def goto_prevfolder(self, logfile=False) :
        """Go back to previous folder
        """
        print("Going back to the original folder {0}".format(self.paths._prev_folder))
        self.goto_folder(self.paths._prev_folder, logfile=logfile, verbose=False)
            
    def goto_folder(self, newpath, logfile=False, verbose=True) :
        """Changing directory and keeping memory of the old working one
        """
        try: 
            prev_folder = os.getcwd()
            newpath = os.path.normpath(newpath)
            os.chdir(newpath)
            if verbose :
                print("Going to folder {0}".format(newpath))
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
        self.paths = lambda:None
        self.paths.root = self.my_params.root
        self.paths.data = joinpath(self.paths.root, self.my_params.data)
        for name in self.my_params._dic_folders.keys() + self.my_params._dic_input_folders.keys():
            setattr(self.paths, name, joinpath(self.paths.data, getattr(self.my_params, name)))

        # Creating the filenames for Master files
        self.masterfiles = lambda:None
        self.dic_attr_master = {}
        for mastertype in dic_listMaster.keys() :
            name_attr = "master{0}".format(mastertype.lower())
            self.dic_attr_master[mastertype] = name_attr
            [masterfolder, mastername] = dic_listMaster[mastertype]
            # Adding the path of the folder
            setattr(self.paths, name_attr, joinpath(self.paths.master, masterfolder))
            # Adding the full path name for the master files
            setattr(self.masterfiles, name_attr, joinpath(self.paths.master, masterfolder, mastername))

#    def get_master(self, mastertype, **kwargs) :
#        """Getting the master type MuseImage
#        Return None if not found or mastertype does not match
#        """
#        if mastertype.upper() not in self.dic_attr_master.keys() :
#            print("ERROR: mastertype not in the list of predefined types")
#            return None
#
#        attr_master = self.dic_attr_master[mastertype]
#        if os.path.isfile(getattr(self.masterfiles, attr_master)) :
#            return MuseImage(filename=getattr(self.masterfiles, attr_master), title=mastertype, **kwargs)
#        else :
#            print("ERROR: file {0} not found".format(getattr(self.masterfiles, attr_master)))
#            return None

#    def read_master_table(self, expotype, name_table=None, verbose=None):
#        if verbose is None: 
#            verbose = self.verbose
#        if name_table is None: 
#            name_table = default_Master_table
#
#        setattr(self.Table, expotype, XX)

    def read_file_table(self, filetype=None, folder=None, **kwargs):
        """Read an existing RAW data table to start the pipeline
        """

        l_filetype = lower_rep(filetype)
        if folder is None :
            folder = getattr(self.paths, l_filetype)
        name_table = kwargs.pop('name_table', dic_files_tables[l_filetype])

        # Check the raw folder
        self.goto_folder(folder)

        # Read the astropy table
        if not os.path.isfile(name_table):
            print("ERROR: file table {0} does not exist".format(name_table))
        else :
            if self.verbose : print("Reading fits Table {0}".format(name_table))
            setattr(self.Tables, filetype, Table.read(name_table, format="fits"))
        
        # Going back to the original folder
        self.goto_prevfolder()

    def create_raw_table(self, overwrite_table=None, verbose=None, **kwargs) :
        """ Create a fits table with all the information from
        the Raw files
        Also create an astropy table with the same info
        """
        if verbose is None: verbose = self.verbose
        if overwrite_table is not None: self.overwrite_table = overwrite_table
        name_table = kwargs.pop('name_table', dic_files_tables['rawfiles'])

        rawfolder = self.paths.rawfiles

        # Check the raw folder
        self.goto_folder(self.paths.rawfiles)

        # Testing if raw table exists
        if os.path.isfile(name_table) :
            if self.overwrite_table :
                print("WARNING: the raw-files table will be overwritten")
            else :
                print("WARNING: the raw files table already exists")
                print("If you wish to overwrite it, "
                      " please turn on the 'overwrite_table' option to 'True'")
                print("In the meantime, the existing table will be read and used")
                self.goto_prevfolder()
                self.read_file_table(filetype='rawfiles', folder=self.paths.rawfiles)
                self.sort_types()
                return

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
        self.Tables.rawfiles = Table([MUSE_infodic['FILENAME'][idxsort]], names=['filename'], meta={'name': 'raw file table'})

        # Creating the columns
        for k in fulldic.keys() :
            [namecol, keyword, func, form] = fulldic[k]
            self.Tables.rawfiles[namecol] = MUSE_infodic[k][idxsort]

        # Writing up the table
        self.Tables.rawfiles.write(name_table, format="fits", overwrite=self.overwrite_table)

        # Sorting the type
        self.sort_types()
        # Going back to the original folder
        self.goto_prevfolder()

    def reset_expotypes(self) :
        """Reseting all lists of expotypes
        """
        # Reseting the select_type item
        self.Tables = lambda:None
        for expotype in listexpo_types.values() :
            setattr(self.Tables, expotype, [])

    def sort_types(self, checkmode=None, strong_checkmode=None) :
        """Provide lists of exposures with types defined in the dictionary
        """
        # Reseting the list if reset is True (default)
        if not hasattr(self, 'Tables'): self.reset_expotypes()

        if checkmode is not None : self.checkmode = checkmode
        else : checkmode = self.checkmode

        if strong_checkmode is not None : self.strong_checkmode = strong_checkmode
        else : strong_checkmode = self.strong_checkmode

        # Sorting alphabetically (thus by date)
        for keytype in listexpo_types.keys() :
            expotype = listexpo_types[keytype].upper()
            try :
                mask = (self.Tables.rawfiles['type'] == keytype)
                setattr(self.Tables, expotype, self.Tables.rawfiles[mask])
            except AttributeError:
                pass

    def select_expotype_fromraw(self, expotype=None):
        """ This will return the info Table of raw files corresponding 
        to the given expotype.
        """
        return getattr(self.Tables, expotype)

    def get_name_master(self, expotype) :
        return normpath(getattr(self.masterfiles, self.dic_attr_master[expotype]))

    def get_path_master(self, expotype) :
        return normpath(getattr(self.paths, self.dic_attr_master[expotype]))

    def select_closest_mjd(self, mjdin, group_table) :
        """Get the closest frame within the expotype
        If the attribute does not exist in Tables, it tries to read
        the table from the folder
        """
        # Get the closest tpl
        index = np.argmin((mjdin - group_table['mjd'])**2)
        closest_tpl = group_table[index]['tpls']
        return closest_tpl

    def select_tpl_files(self, expotype=None, tpl="ALL") :
        """Selecting a subset of files from a certain type
        """
        if expotype not in listexpo_types.keys() :
            print("ERROR: input expotype is not in the list of possible values")
            return 

        MUSE_subtable = self.select_expotype_fromraw(listexpo_types[expotype])
        if len(MUSE_subtable) == 0:
            if self.verbose :
                print("WARNING: empty file table of type {0}".format(expotype))
                print("WARNING: returning an empty Table from the tpl selection")
            return MUSE_subtable

        group_table = MUSE_subtable.group_by('tpls')
        if tpl == "ALL":
            return group_table
        else :
            return group_table.groups[group_table.groups.key['tpls'] == tpl]

    def run_bias(self, sof_filename='bias', tpl="ALL", **kwargs) :
        """Reducing the Bias files and creating a Master Bias
        Will run the esorex muse_bias command on all Biases

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        fits_tablename = kwargs.pop('fits_tablename', dic_files_tables['masterbias'])
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='BIAS', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                print("WARNING: no BIAS recovered from the file Table - Aborting")
                return
        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the BIASES including
        # the list of files to be processed for one MASTER BIAS
        dic_bias = OrderedDict()
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            dic_bias['BIAS'] = add_listpath(self.get_path_master('BIAS'),
                    + gtable['filename'].data.astype(np.object))
            # extract the tpl (string)
            tpl = gtable['tpls'][0]
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, 
                    dic_files=dic_bias, new=True)
            # Name of final Master Bias
            name_masterbias = self.get_name_master('BIAS')
            # Run the recipe
            self.recipe_bias(self.current_sof, name_masterbias, tpl)

        # Write the MASTER BIAS Table and save it
        self.Tables.masterbias = tpl_gtable.groups.aggregate(np.mean)['tpls','mjd', 'tplnexp']
        full_tablename = joinpath(self.paths.masterbias, fits_tablename)
        self.Tables.masterbias_tablename = full_tablename
        if (not self.overwrite_table) and os.path.isfile(full_tablename) :
            print("WARNING: Table MasterBias already exists, use overwrite_table to overwrite it")
        else :
            self.Tables.masterbias.write(full_tablename, format="fits", overwrite=self.overwrite_table)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_flat(self, sof_filename='flat', tpl="ALL", **kwargs) :
        """Reducing the Flat files and creating a Master Flat
        Will run the esorex muse_flat command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        fits_tablename = kwargs.pop('fits_tablename', dic_files_tables['masterflat'])
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='FLAT,LAMP', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                print("WARNING: no FLAT recovered from the file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the FLATs including
        # the list of files to be processed for one MASTER Flat
        dic_flat = OrderedDict()
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            dic_flat['FLAT'] = add_listpath(self.get_path_master('FLAT'),
                    + gtable['filename'].data.astype(np.object))
            # extract the tpl (string)
            # Add Bias & Badpix Table
            tpl = gtable['tpls'][0]
            mean_mjd = gtable.groups.aggregate(np.mean)['mjd'].data[0]
            # Finding the best tpl for BIAS
            bias_tpl = self.select_closest_mjd(mean_mjd, self.Tables.masterbias)
            # Adding the bad pixel table
            dic_flat["MASTER_BIAS"] = [self.get_name_master('BIAS') + "_" + bias_tpl + ".fits"]
            dic_flat["BADPIX_TABLE"] = [joinpath(self.my_params.musecalib, self.my_params.badpix_table)]
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, 
                    dic_files=dic_flat, new=True)
            # Name of final Master Flat
            name_masterflat = self.get_name_master('FLAT')
            # Name of final Trace Table
            name_tracetable = self.get_name_master('TRACE')
            # Run the recipe
            self.recipe_flat(self.current_sof, name_masterflat, name_tracetable, tpl)

        # Write the MASTER FLAT Table and save it
        self.Tables.masterflat = tpl_gtable.groups.aggregate(np.mean)['tpls','mjd', 'tplnexp']
        full_tablename = joinpath(self.paths.masterflat, fits_tablename)
        self.Tables.masterflatname = full_tablename
        if (not self.overwrite_table) and os.path.isfile(full_tablename) :
            print("WARNING: Table MasterBias already exists, use overwrite_table to overwrite it")
        else :
            self.Tables.masterflat.write(full_tablename, format="fits", overwrite=self.overwrite_table)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_wave(self, sof_filename='wave', tpl="ALL", **kwargs) :
        """Reducing the WAVE-CAL files and creating the Master Wave
        Will run the esorex muse_wave command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        fits_tablename = kwargs.pop('fits_tablename', dic_files_tables['masterwave'])
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='WAVE', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                print("WARNING: no WAVE recovered from the file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the FLATs including
        # the list of files to be processed for one WAVECAL
        dic_wave = OrderedDict()
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            dic_wave['ARC'] = add_listpath(self.get_path_master('WAVE'),
                    + gtable['filename'].data.astype(np.object))
            # extract the tpl (string)
            # Add Bias & Badpix Table
            tpl = gtable['tpls'][0]
            mean_mjd = gtable.groups.aggregate(np.mean)['mjd'].data[0]
            # Finding the best tpl for BIAS
            bias_tpl = self.select_closest_mjd(mean_mjd, self.Tables.masterbias)
            dic_wave["MASTER_BIAS"] = [self.get_name_master('BIAS') + "_" + bias_tpl + ".fits"]
            # Finding the best tpl for TRACE_TABLE
            trace_tpl = self.select_closest_mjd(mean_mjd, self.Tables.masterflat)
            dic_wave["TRACE_TABLE"] = [self.get_name_master('TRACE') + "_" + trace_tpl + ".fits"]
            dic_wave["BADPIX_TABLE"] = [joinpath(self.my_params.musecalib, self.my_params.badpix_table)]
            dic_wave["LINE_CATALOG"] = [joinpath(self.my_params.musecalib, self.my_params.line_catalog)]
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, 
                    dic_files=dic_wave, new=True)
            # Name of final Master Wave
            name_masterwave = self.get_name_master('WAVE')
            # Run the recipe
            self.recipe_wave(self.current_sof, name_masterwave, tpl)

        # Write the MASTER FLAT Table and save it
        self.Tables.masterwave = tpl_gtable.groups.aggregate(np.mean)['tpls','mjd', 'tplnexp']
        full_tablename = joinpath(self.paths.masterwave, fits_tablename)
        self.Tables.masterwavename = full_tablename
        if (not self.overwrite_table) and os.path.isfile(full_tablename) :
            print("WARNING: Table Wave already exists, use overwrite_table to overwrite it")
        else :
            self.Tables.masterwave.write(full_tablename, format="fits", overwrite=self.overwrite_table)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_lsf(self, sof_filename='lsf', tpl="ALL", **kwargs) :
        """Reducing the LSF files and creating the LSF PROFILE
        Will run the esorex muse_lsf command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        fits_tablename = kwargs.pop('fits_tablename', dic_files_tables['masterwave'])
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='WAVE', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                print("WARNING: no WAVE recovered from the file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        dic_lsf = OrderedDict()
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            dic_lsf['ARC'] = add_listpath(self.get_path_master('WAVE'),
                    + gtable['filename'].data.astype(np.object))
            # extract the tpl (string)
            # Add Bias & Badpix Table
            tpl = gtable['tpls'][0]
            mean_mjd = gtable.groups.aggregate(np.mean)['mjd'].data[0]
            # Finding the best tpl for BIAS
            bias_tpl = self.select_closest_mjd(mean_mjd, self.Tables.masterbias)
            dic_lsf["MASTER_BIAS"] = [self.get_name_master('BIAS') + "_" + bias_tpl + ".fits"]
            # Finding the best tpl for TRACE_TABLE
            trace_tpl = self.select_closest_mjd(mean_mjd, self.Tables.masterflat)
            dic_lsf["TRACE_TABLE"] = [self.get_name_master('TRACE') + "_" + trace_tpl + ".fits"]
            wave_tpl = self.select_closest_mjd(mean_mjd, self.Tables.masterwave)
            dic_lsf["WAVECAL_TABLE"] = [self.get_name_master('WAVE') + "_" + wave_tpl + ".fits"]
            dic_lsf["BADPIX_TABLE"] = [joinpath(self.my_params.musecalib, self.my_params.badpix_table)]
            dic_lsf["LINE_CATALOG"] = [joinpath(self.my_params.musecalib, self.my_params.line_catalog)]
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, 
                    dic_files=dic_lsf, new=True)
            # Name of final Master Wave
            name_masterlsf = self.get_name_master('LSF')
            # Run the recipe
            self.recipe_lsf(self.current_sof, name_masterlsf, tpl)

        # Write the MASTER LSF PROFILE Table and save it
        self.Tables.masterlsf = tpl_gtable.groups.aggregate(np.mean)['tpls','mjd', 'tplnexp']
        full_tablename = joinpath(self.paths.masterlsf, fits_tablename)
        self.Tables.masterlsfname = full_tablename
        if (not self.overwrite_table) and os.path.isfile(full_tablename) :
            print("WARNING: Table LSF already exists, use overwrite_table to overwrite it")
        else :
            self.Tables.masterlsf.write(full_tablename, format="fits", overwrite=self.overwrite_table)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def create_calibrations(self):
        self.check_for_calibrations()
        # MdB: this creates the standard calibrations, if needed
        if not ((self.redo==0) and (self.Master['BIAS']==1)):
            self.run_bias()
        if not ((self.redo==0) and (self.Master['DARK']==1)):
            self.run_dark()
        if not ((self.redo==0) and (self.Master['FLAT']==1)):
            self.run_flat()
        if not ((self.redo==0) and (self.Master['WAVE']==1)):
            self.run_wavecal()
        if not ((self.redo==0) and (self.Master['TWILIGHT']==1)):
            self.run_twilight()    
        if not ((self.redo==0) and (self.Master['STD']==1)):
            self.create_standard_star()
            self.run_standard_int()

    def check_for_calibrations(self, verbose=None):
        """This function checks which calibration file are present, just in case
        you do not wish to redo them. Variables are named: 
        masterbias, masterdark, standard, masterflat, mastertwilight
        """
        
        outcal = getattr(self.my_params, "master" + suffix_folder)
        if verbose is None: verbose = self.verbose

        for mastertype in dic_listMaster.keys() :
            [fout, suffix] = dic_listMaster[mastertype]
            # If TWILIGHT = only one cube
            if mastertype == "TWILIGHT" :
                if not os.path.isfile(joinpath(outcal, fout, "{suffix}.fits".format(suffix=suffix))):
                    self.Master[mastertype] = False

                if verbose :
                    if self.Master[mastertype] :
                        print("WARNING: Twilight flats already made")
                    else :
                        print("WARNING: Twilight flats NOT there yet")

            # Others = 24 IFU exposures
            else :
                for ifu in range(1,25):
                    if not os.path.isfile(joinpath(outcal, fout, "{0}-{1:02d}.fits".format(suffix, ifu))):
                        self.Master[mastertype] = False
                        break
                if verbose :
                    if self.Master[mastertype]:
                        print("WARNING: Master {0} already in place".format(mastertype))
                    else :
                        print("WARNING: Master {0} NOT yet there".format(mastertype))

    def select_stdfile(self):
        """Minimise the time difference between the object and the illumination
        """
        
        if len(self.stdlist) == 0:
            self.stdfile=''
            return
        
        # Initialisation of the time array
        std_times = []
        # Transforming the list into an array
        stdlist = np.array(self.stdlist)
        for std in stdlist:
            std_times.append(np.datetime64(std[-28:-5]))
        # Transforming the list into an array
        std_times = np.array(std_times)

        for object in self.objectlist: 
            object_time = np.datetime64(object[-28:-5])
            DeltaTime = np.abs(std_times - object_time)
            mask = (DeltaTime == np.min(DeltaTime))
            self.stdfile = stdlist[mask][0]

    def select_run_tables(self, runname="Run01", verbose=None):
        """
        """
        if len(self.objectlist) < 1 : 
            print("WARNING: objectlist is empty, hence no CAL tables defined")
            return

        filename = self.objectlist[0]
        hdu = pyfits.open(filename)
        filetime = np.datetime64(hdu[0].header['DATE-OBS']).astype('datetime64[D]')
        for runs in MUSEPIPE_runs.keys():
            Pdates = MUSEPIPE_runs[runs]
            date1 = np.datetime64(Pdates[1]).astype('datetime64[D]')
            date2 = np.datetime64(Pdates[2]).astype('datetime64[D]')
            if ((filetime >= date1) and (filetime <= date2)):
                runname = runs
                break
        if verbose is None: verbose = self.verbose
        if verbose:
            print "Using geometry calibration data from MUSE runs %s\n"%finalkey
        self.geo_table = joinpath(getattr(self.my_params, "musecalib" + suffix_folder), 
                "geometry_table_wfm_{runname}.fits".format(runname=runname)) 
        self.astro_table = joinpath(getattr(self.my_params, "musecalib" + suffix_folder), 
                "astrometry_wcs_wfm_{runname}.fits".format(runname=runname))

    def select_illumfiles(self):
        """Minimise the difference between the OBJECT and ILLUM files
        """
        self.final_illumlist=[]
        if len(self.illumlist) == 0:
            return
        illum_times=[]
        illumlist=np.array(self.illumlist)
        for illum in illumlist:
            illum_times.append(np.datetime64(illum[-28:-5]))
        illum_times=np.array(illum_times)
        for object in self.objectlist: 
            object_time=np.datetime64(object[-28:-5])
            DeltaTime=np.abs(illum_times - object_time)
            mask = (DeltaTime==np.min(DeltaTime))
            self.final_illumlist.append(illumlist[mask][0])
        
