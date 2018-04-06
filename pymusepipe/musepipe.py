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
        'TWILIGHT': ['Twilight', 'TWILIGHT_CUBE'], 
        'WAVE': ['Wave', 'WAVECAL_TABLE'], 
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
default_raw_table = "rawfiles_table.fits"

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

    def __init__(self, galaxyname=None, pointing=0, objectlist=[], rc_filename=None, cal_filename=None, 
            outlog=None, logfile="MusePipe.log", verbose=True, redo=False, align_file=None, mode="WFM-NOAO-N", 
            checkmode=True, strong_checkmode=False, create_raw_table=True):
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
        PipeRecipes.__init__(self)

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
        # astropy dataset probing the rawdata folder
        if create_raw_table :
            self.create_raw_table()
        # ===========================================================

    def goto_prevfolder(self, verbose=True) :
        """Go back to previous folder
        """
        print("Going back to the original folder {0}".format(self.paths._prev_folder))
        self.goto_folder(self.paths._prev_folder, verbose=False)
            
    def goto_folder(self, newpath, verbose=True) :
        """Changing directory and keeping memory of the old working one
        """
        try: 
            prev_folder = os.getcwd()
            os.chdir(newpath)
            if verbose :
                print("Going to folder {0}".format(newpath))
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

    def get_master(self, mastertype, **kwargs) :
        """Getting the master type
        Return None if not found or mastertype does not match
        """
        if mastertype.upper() not in self.dic_attr_master.keys() :
            print("ERROR: mastertype not in the list of predefined types")
            return None

        attr_master = self.dic_attr_master[mastertype]
        if os.path.isfile(getattr(self.masterfiles, attr_master)) :
            return MuseImage(filename=getattr(self.masterfiles, attr_master), title=mastertype, **kwargs)
        else :
            print("ERROR: file {0} not found".format(getattr(self.masterfiles, attr_master)))
            return None

    def create_raw_table(self, name_table=None, verbose=None) :
        """ Create a fits table with all the information from
        the Raw files
        Also create an astropy table with the same info
        """
        if verbose is None: verbose = self.verbose
        rawfolder = self.paths.rawdata

        if name_table is None : name_table = default_raw_table
        self.name_table = name_table

        # Check the raw folder
        self.goto_folder(self.paths.rawdata)

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
        self.rawfiles = Table([MUSE_infodic['FILENAME'][idxsort]], names=['filename'], meta={'name': 'raw file table'})

        # Creating the columns
        columns=[]
        for k in fulldic.keys() :
            [namecol, keyword, func, form] = fulldic[k]
            self.rawfiles[namecol] = MUSE_infodic[k][idxsort]
            columns.append(pyfits.Column(name=namecol, format=form, array=MUSE_infodic[k][idxsort]))

        # Writing up the table
        table = pyfits.BinTableHDU.from_columns(columns)
        table.writeto(self.name_table, overwrite=True)

        # Sorting the type
        self.sort_types()
        # Going back to the original folder
        self.goto_prevfolder()

    def reset_expotypes(self) :
        """Reseting all lists of expotypes
        """
        # Reseting the select_type item
        for expotype in listexpo_types.values() :
            setattr(self, expotype, [])

    def sort_types(self, reset=True, checkmode=None, strong_checkmode=None) :
        """Provide lists of exposures with types defined in the dictionary
        """
        # Reseting the list if reset is True (default)
        if reset: self.reset_expotypes()

        if checkmode is not None : self.checkmode = checkmode
        else : checkmode = self.checkmode

        if strong_checkmode is not None : self.strong_checkmode = strong_checkmode
        else : strong_checkmode = self.strong_checkmode

        self.files = lambda:None
        # Sorting alphabetically (thus by date)
        for expotype in listexpo_types.values() :
            expotype = expotype.upper()
            try :
                mask = self.rawfiles['type'] == expotype
                setattr(self.files, expotype, self.rawfiles[mask])
            except AttributeError:
                pass

    def select_tpl_files(self, expotype=None, tpl="ALL") :
        """Selecting a subset of files from a certain type
        """
        if expotype not in listexpo_types.values() :
            print("ERROR: input expotype is not in the list of possible values")
            return

        MUSE_subtable = getattr(self.files, expotype)
        MUSE_files = MUSE_subtable['filename']
        MUSE_tpls = MUSE_subtable['tpls']
        MUSE_tplnexp = MUSE_subtable['tplnexp']

        dicout = {}
        if tpl == "ALL" :
            all_tpl = np.unique(MUSE_tpls)
            for mytpl in MUSE_tpls :
                select_TPL = (MUSE_tpls == mytpl) 
                # Checking that we have all files
                if MUSE_tplnexp[select_TPL][0] == np.sum(select_TPL) :
                    dicout[mytpl] = MUSE_files[select_TPL]
        else :
            select_TPL = (MUSE_tpls == tpl) 
            # Checking that we have all files
            if MUSE_tplnexp[select_TPL][0] == np.sum(select_TPL) :
                dicout[tpl] = MUSE_files[select_TPL]

        return dicout

    def run_bias(self, sof_filename='bias', tpl="ALL") :
        """Reducing the Bias files and creating a Master Bias
        Will run the esorex muse_bias command on all Biases

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files
        dic_tpl = self.select_tpl_files(expotype='BIAS', tpl=tpl)

        # Go to the data folder
        self.goto_folder(self.paths.data)

        dic_bias = {}
        for tpl in dic_tpl.keys() :
            # Setting the dictionary
            dic_bias['BIAS'] = dic_tpl[tpl]
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, 
                    dic_files=dic_bias, new=True)
            # Run the recipe
            self.recipe_bias(self.current_sof, tpl)

        # Go back to original folder
        self.goto_prevfolder()

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
        
