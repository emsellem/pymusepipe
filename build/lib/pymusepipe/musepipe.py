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
import time
import glob
import subprocess
from subprocess import call
import copy

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


# Importing pymusepipe modules
from init_musepipe import init_muse_parameters

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
listMaster_dic = {'DARK': ['Dark', 'MASTER_DARK'], 
        'BIAS': ['Bias', 'MASTER_BIAS'], 
        'FLAT': ['Flat', 'MASTER_FLAT'],
        'TWILIGHT': ['Twilight', 'TWILIGHT_CUBE'], 
        'WAVE': ['Wave', 'WAVECAL_TABLE'], 
        'STD': ['Std', 'PIXTABLE_STD'], 
        }

listexpo_files = {"TYPE" : ['ESO DPR TYPE', "raw_hdr_dprtype.txt"],
        "DATE": ['MJD-OBS', "raw_hdr_mjd.txt"],
        "MODE": ['HIERARCH ESO INS MODE', "raw_hdr_mode.txt"]
         }

exclude_list_checkmode = ['BIAS', 'DARK']

esorex_rc = "/home/soft/ESO/MUSE/muse-kit-2.2-5/esorex-3.12.3/etc/esorex.rc"
############################################################
#                      END
############################################################
def run_oscommand(text, logfile=None, verbose=False, fakemode=True) :
    """Running an os.system shell command
    Fake mode will just spit out the command but not actually do it.
    """
    if verbose : 
        print(text)

    if logfile is not None :
       fout = open(logile, 'a')
       fout.write(text + "\n")
       fout.close()

    if not fakemode :
        os.system(command)

def create_time_name() :
    """Create a time-link name for file saving purposes

    Return: a string including the time, hence a priori unique
    """
    return str(time.time())

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

def get_date_inD(indate) :
    """Transform date in Y-M-D
    """
    return np.datetime64(indate).astype('datetime64[D]')

def feed_sof(sof_filename, folderin="", list_files=[], type_files="BIAS", verbose=True) :
    """Feeding an sof file with some filename (which can be a list)
    and a type (e.g., BIAS=default)
    """
    sof_file = open(sof_filename, "a")
    print("Writing in file {0}".format(joinpath(folderin,sof_filename)))

    for filename in list_files :
        text_to_write = "{0} {1}\n".format(joinpath(folderin, filename), type_files)
        sof_file.write(text_to_write)
        if verbose :
            print(text_to_write)

def changeto_dir(path) :
    """Changing directory and keeping memory of the old working one
    """
    prev_folder = os.getcwd()
    try: 
        os.chdir(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    return prev_folder

#########################################################################
# Main class
#                           muse_pipe
#########################################################################

    
class muse_pipe(object):
    """Main Class to define and run the MUSE pipeline, given a certain galaxy name
    
    musep = muse_pipe(galaxyname='NGC1087', rc_filename="", cal_filename="", 
                      outlog"NGC1087_log1.log", objects=[''])
    musep.run()
    """

    def __init__(self, galaxyname=None, pointing=0, objectlist=[], rc_filename=None, cal_filename=None, 
            outlog=None, verbose=True, redo=False, align_file=None, mode="WFM-NOAO-N", 
            checkmode=True, strong_checkmode=False, cpu0=0, cpu1=23):
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

        # Setting up the folders and names for the calibration files
        # Can be initialised by either an rc_file, a default rc_file or harcoded defaults.
        self.my_params = init_muse_parameters(rc_filename=rc_filename, 
                            cal_filename=cal_filename)

        # Set up the default parameters
        self.galaxyname = galaxyname
        self.pointing = pointing

        if outlog is None : 
            outlog = "log_{timestamp}".format(timestamp=create_time_name())
            print("The Log folder will be {log}".format(outlog))
        self.outlog = outlog

        self.verbose = verbose
        self.redo = redo
        self.align_file = align_file

        # Mode of the observations
        self.mode = mode
        # Checking if mode is correct
        self.checkmode = checkmode
        # Checking if mode is correct also for BIAS & DARK
        self.strong_checkmode = strong_checkmode
        # Addressing CPU by number (cpu0=start, cpu1=end)
        self.cpu0 = cpu0
        self.cpu1 = cpu1

        # Set of objects to reduced
        self.objectlist = objectlist
        
        # Create full path folder 
        self.set_fullpath_names()

        # Go to the data directory
        # and Recording the folder where we start
        self.orig_wd = changeto_dir(self.paths.fulldata)

        # Making the output folders in a safe mode
        if self.verbose:
            print("Creating directory structure")
            print("Going to the data folder {0}".format(self.paths.fulldata))

        # Init the Master folder
        safely_create_folder(self.my_params.mastercalib_folder)
        # Init the Master exposure flag dictionary
        self.Master = {}
        for mastertype in listMaster_dic.keys() :
            [masterfolder, mastername] = listMaster_dic[mastertype]
            safely_create_folder(joinpath(self.my_params.mastercalib_folder, masterfolder))
            self.Master[mastertype] = False
        self.check_for_calibrations(verbose=False)

        # Reduced folder
        safely_create_folder(self.my_params.reducedfiles_folder)
        # Sky folder
        safely_create_folder(self.my_params.sky_folder)
        # Cubes folder
        safely_create_folder(self.my_params.cubes_folder)
        # Log file folder
        safely_create_folder(outlog)

        # First, list all the files and find out which types they are
        self.probe_files()

        # Selecting the ones which 
        self.select_run_tables()
        # Going back to initial working directory
        old_folder = changeto_dir(self.orig_wd)

    def set_fullpath_names(self) :
        """Create full path names to be used
        """
        # initialisation of the full paths 
        self.my_params.data_folder = "{0}/P{1:02d}/".format(self.galaxyname, self.pointing)
        self.paths = lambda:None
        self.paths.root = self.my_params.root_folder
        self.paths.fulldata = joinpath(self.paths.root, self.my_params.data_folder)
        self.paths.master = joinpath(self.paths.fulldata, self.my_params.mastercalib_folder)
        self.paths.sky = joinpath(self.paths.fulldata, self.my_params.sky_folder)
        self.paths.reduced = joinpath(self.paths.fulldata, self.my_params.reducedfiles_folder)
        self.paths.raw = joinpath(self.paths.fulldata, self.my_params.rawdata_folder)
        self.paths.cubes = joinpath(self.paths.fulldata, self.my_params.cubes_folder)
        self.paths.maps = joinpath(self.paths.fulldata, self.my_params.maps_folder)

        self.filenames = lambda:None
        for mastertype in listMaster_dic.keys() :
            name_attr = "master{0}".format(mastertype.lower())
            [masterfolder, mastername] = listMaster_dic[mastertype]
            setattr(self.paths, name_attr, joinpath(self.paths.master, masterfolder))
            setattr(self.filenames, name_attr, joinpath(self.paths.master, masterfolder, mastername))

    def get_master(self, mastertype, **kwargs) :
        """Getting the master type
        Return None if not found or mastertype does not match
        """
        if mastertype.upper() not in listMaster_dic.keys() :
            print("ERROR: mastertype not in the list of predefined types")
            return None

        attribute_master = "master{0}".format(mastertype.lower())
        if os.path.isfile(getattr(self.filenames, attribute_master)) :
            return muse_image(filename=self.filenames.attribute_master, title=mastertype, **kwargs)
        else :
            print("ERROR: file {0} not found".format(self.filenames.attribute_master))
            return None

    def probe_files(self, rawfolder=None, verbose=None) :
        """Determine which files are in the raw folder
        """
        if verbose is None: verbose = self.verbose

        if rawfolder is None :
            rawfolder = self.my_params.rawdata_folder
        else :
            # Changing the raw folder default
            self.my_params.rawdata_folder = rawfolder

        # Check the raw folder
        prev_folder = changeto_dir(joinpath(self.paths.fulldata, rawfolder))

        if verbose : print(("Probing the files in the {rawfolder} folder").format(rawfolder=rawfolder))

        # Extracting the fits Headers and processing them
        command = ("fitsheader -e 0 -k '{0}' -t ascii.tab "
                "*fits.fz > {filename}".format(listexpo_files['TYPE'][0], 
                filename=listexpo_files['TYPE'][1]))
        run_oscommand(command)
        command = ("fitsheader -e 0 -k '{0}' -t ascii.tab "
                "*fits.fz > {filename}".format(listexpo_files['DATE'][0], 
                filename=listexpo_files['DATE'][1]))
        run_oscommand(command)
        command = ("fitsheader -e 0 -k '{0}' -t ascii.tab "
                "*fits.fz > {filename}".format(listexpo_files['MODE'][0], 
                filename=listexpo_files['MODE'][1]))
        run_oscommand(command)

        info_types = ascii.read(listexpo_files['TYPE'][1])
        info_dates = ascii.read(listexpo_files['DATE'][1])
        info_modes = ascii.read(listexpo_files['MODE'][1])
        # We need to copy the first one as we will scan and remove lines
        copy_info_types = copy.copy(info_types)

        # Reset the dictionary of exposures
        self.info_expo = {}
        info_dates.add_index('filename')
        nfiles_found = 0
        # Loop over the file names and process the types
        for i in range(info_types['filename'].size) :
            filename = info_types['filename'][i]
            if filename in info_dates['filename'] :
                index_date = info_dates.loc[filename].index
                index_mode = info_dates.loc[filename].index
                expotype = info_types['value'][i] 
                expodate = info_dates['value'][index_date]
                expomode = info_modes['value'][index_mode]
                # Values of this dictionary are the TYPE and DATE
                self.info_expo[filename] = [expotype, expodate, expomode]
                # removing line from the TYPE and DATE files
                copy_info_types.remove_row(i - nfiles_found)
                nfiles_found += 1
                info_dates.remove_row(index_date)
                info_modes.remove_row(index_mode)

        # Check that all types are within the default list
        for val in self.info_expo.values() :
            # Only checking expotype, not expodate
            if val[0] not in listexpo_types.keys() :
                print("WARNING: {0} not in default TYPE list".format(val[0]))

        # Printing out the ones which were not found in both files
        if verbose:
            for missedname in copy_info_types['filename'] :
                print("WARNING: file {filename} was missing a DATE keyword")
            for missedname in info_dates['filename'] :
                print("WARNING: file {filename} was missing a TYPE keyword")
            for missedname in info_dates['filename'] :
                print("WARNING: file {filename} was missing a MODE keyword")

        # Create the parameters to have the list of exposures for each type
        self.sort_types()
        os.chdir(self.orig_wd)

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

        for expo in self.info_expo.keys() :
            # first get the current list - and append the current exponame
            var = listexpo_types[self.info_expo[expo][0]]
            if checkmode & (self.info_expo[expo][2] != self.mode) :
                if (var not in exclude_list_checkmode) | self.strong_checkmode :
                    if self.verbose : 
                        print(("WARNING: excluding file {0},{1} "
                           "because of wrong mode").format(expo, var))
                    continue
            currentlist = getattr(self, var)
            currentlist.append(expo)
            # Then reset the attribute with the updated list
            setattr(self, var, currentlist)

        # Sorting alphabetically (thus by date)
        for expotype in listexpo_types.values() :
            try :
                setattr(self, expotype, sorted(getattr(self, expotype)))
            except AttributeError:
                pass

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
        
        outcal = self.my_params.mastercalib_folder
        if verbose is None: verbose = self.verbose

        for mastertype in listMaster_dic.keys() :
            [fout, suffix] = listMaster_dic[mastertype]
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
            Pdates = MUSEPIPE_runs[key]
            date1 = np.datetime64(Pdates[1]).astype('datetime64[D]')
            date2 = np.datetime64(Pdates[2]).astype('datetime64[D]')
            if ((filetime >= date1) and (filetime <= date2)):
                runname = key
        if verbose is None: verbose = self.verbose
        if verbose:
            print "Using geometry calibration data from MUSE runs %s\n"%finalkey
        self.geo_table = joinpath(self.my_params.musecalib_folderr, "geometry_table_wfm_{runname}.fits".format(runname=runname)) 
        self.astro_table = joinpath(self.my_params.musecalib_folder, "astrometry_wcs_wfm_{runname}.fits".format(runname=runname))

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
        
    def shellrun_bias(self, sof_filename='bias.sof', expoBIAS=None) :
        """
        """

        # Checking the list of files if None
        if expoBIAS is not None :
            if expoBIAS not in self.BIAS :
                print("ERROR: {0} not in given BIAS list".format(expoBIAS))
                return
        else : expoBIAS = self.BIAS

        # Go to the data folder
        prev_folder = changeto_dir(self.paths.fulldata)
        # Feeding the sof input file
        feed_sof(sof_filename, self.my_params.rawdata_folder, expoBIAS, "BIAS")

        command = ("likwid-pin -c N:{CPU0}-{CPU1} esorex --no-checksum "
                   "--log-file={outlog}/bias.log --output_dir muse_bias --nifu=-1 "
                   "--merge {sof}".format(CPU0=self.cpu0, CPU1=self.cpu1, 
                       outlog=self.outlog, sof=sof_filename))
        run_oscommand(command)
        # Go back to original folder
        os.chdir(prev_folder)

    def run_bias(self, expoBIAS=None):
       
        if expoBIAS is not None :
            if expoBIAS not in self.BIAS :
                print("ERROR: {0} not in given BIAS list".format(expoBIAS))
                return
        else : expoBIAS = self.BIAS

        cpl.esorex.init(esorex_rc)
        cpl.esorex.log.file = self.outlog + 'bias.log'
        muse_bias = cpl.Recipe('muse_bias')
        muse_bias.param.nifu=-1
        safely_create_folder(joinpath(self.my_params.mastercalib_folder, 'BIAS'))
        muse_bias.output_dir = joinpath(self.my_params.mastercalib_folder, 'BIAS')
        muse_bias.calib.BADPIX_TABLE = joinpath(self.my_params.musecalib_folder, self.badpix_table)
        muse_bias(self.BIAS)

    # MdB: Make master dark
    def run_dark(self):
        darklist=self.darklist
        if len(darklist) > 0:
            cpl.esorex.init(esorex_rc)
            cpl.esorex.log.file = self.outlog+'dark.log'
            muse_dark = cpl.Recipe('muse_dark')
            try: os.mkdir(joinpath(self.my_params.mastercalib_folder, 'muse_DARK_dir'))
            except: pass
            muse_dark.output_dir = joinpath(self.my_params.mastercalib_folder, 'muse_DARK_dir')
            muse_dark.param.nifu=-1 #.samples?
            muse_dark.calib.BADPIX_TABLE = self.badpix_table
            muse_dark.calib.MASTER_BIAS = [(self.my_params.mastercalib_folder \
                    +'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu) for ifu in range(1, 25)]
            muse_dark(darklist)
            self.masterdark=1
        else:
            self.masterdark=0
        
    # MdB: make master dome flat  
    def run_flat(self):
        cpl.esorex.init(esorex_rc)
        cpl.esorex.log.file = joinpath(self.outlog, 'flat.log')
        muse_flat = cpl.Recipe('muse_flat')
        
        try: os.mkdir(joinpath(self.my_params.mastercalib_folder, 'muse_FLAT_dir'))
        except: pass
        
        muse_flat.output_dir = joinpath(self.my_params.mastercalib_folder, 'muse_FLAT_dir')
        muse_flat.param.samples = True
        muse_flat.param.nifu=-1 
        # MdB: samples is advocated on page 20 of the software manual
        # MdB: but I'm not sure if it's actually useful.
        muse_flat.calib.BADPIX_TABLE = self.badpix_table
        muse_flat.calib.MASTER_BIAS = [(self.my_params.mastercalib_folder \
                +'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu) for ifu in range(1, 25)]
        if self.masterdark:
            muse_flat.calib.MASTER_DARK = [(self.my_params.mastercalib_folder \
                    +'muse_DARK_dir/MASTER_DARK-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_flat(self.flatlist)#CAREFUL HERE! is flat ok?
        # MdB: This is somewhat worrisome? 

        print 'finito master_flat'       
    
    # MdB: wavelength calibration
    def run_wavecal(self):
        # Initing ESOREX
        cpl.esorex.init(esorex_rc)
        # Setting up the Log File
        cpl.esorex.log.file = self.outlog+'wavecal.log'
        wavelist = self.wavelist
        outcal = self.my_params.mastercalib_folder
        muse_wavecal = cpl.Recipe('muse_wavecal')
        
        try: os.mkdir(outcal + 'muse_WAVECAL_dir')
        except: pass

        muse_wavecal.output_dir = joinpath(outcal, 'muse_WAVECAL_dir')
        muse_wavecal.calib.LINE_CATALOG = joinpath(self.calibrationfolder, 'line_catalog.fits')
        muse_wavecal.param.residuals = False
        muse_wavecal.param.resample = False
        muse_wavecal.param.wavemap=False
        # MdB: these are in fact the default settings
        muse_wavecal.calib.BADPIX_TABLE = self.badpix_table 
        muse_wavecal.param.nifu = -1
        muse_wavecal.calib.MASTER_BIAS = [(outcal + \
                'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_wavecal.calib.MASTER_FLAT = [(outcal + \
                'muse_FLAT_dir/MASTER_FLAT-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_wavecal.calib.TRACE_TABLE = [(outcal + \
                'muse_FLAT_dir/TRACE_TABLE-%02i.fits' % ifu) for ifu in range(1, 25)]
        if self.masterdark:
            muse_wavecal.calib.MASTER_DARK = [(outcal + \
                    'muse_DARK_dir/MASTER_DARK-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_wavecal(wavelist)

    def run_twilight(self):
        if len(self.skyflatlist) == 0:
            return

        outcal = self.my_params.mastercalib_folder
        # Initing ESOREX
        cpl.esorex.init(esorex_rc)
        # Setting up the Log File
        cpl.esorex.log.file = self.outlog+'twilight.log'
        muse_twilight = cpl.Recipe('muse_twilight')
        
        try: os.mkdir(outcal + 'muse_TWILIGHT_dir')
        except: pass
        
        muse_twilight.output_dir = outcal + 'muse_TWILIGHT_dir'
        muse_twilight.calib.MASTER_BIAS = [(outcal + \
                'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_twilight.calib.MASTER_FLAT = [(outcal + \
                'muse_FLAT_dir/MASTER_FLAT-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_twilight.calib.TRACE_TABLE = [(outcal + \
                'muse_FLAT_dir/TRACE_TABLE-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_twilight.calib.WAVECAL_TABLE = [(outcal + \
                'muse_WAVECAL_dir/WAVECAL_TABLE-%02i.fits' % ifu) for ifu in range(1, 25)]
        if self.masterdark:
            muse_twilight.calib.MASTER_DARK = [(outcal + \
                    'muse_DARK_dir/MASTER_DARK-%02i.fits' % ifu) for ifu in range(1, 25)]

        muse_twilight.calib.BADPIX_TABLE = self.badpix_table
        muse_twilight.calib.GEOMETRY_TABLE = self.geo_table
        muse_twilight.calib.VIGNETTING_MASK = self.vignetting_mask
        # # MdB: we pick a random illum correction. Not sure if this helps.
        # muse_twilight.raw.ILLUM=self.final_illumlist[0]
        muse_twilight(self.skyflatlist)
        self.mastertwilight=1
        print 'finito TWILIGHT' 

    def create_standard_star(self):
        stdfile=self.stdfile
        outcal = self.my_params.mastercalib_folder

        # Initing ESOREX
        cpl.esorex.init(esorex_rc)
        cpl.esorex.log.file = self.outlog+'scibasic.log'
        
        muse_scibasic = cpl.Recipe('muse_scibasic')

        try: os.mkdir(outcal +'muse_POST_dir')
        except: pass
        
        muse_scibasic.calib.BADPIX_TABLE = self.badpix_table
        muse_scibasic.output_dir = outcal + 'muse_POST_dir'
        muse_scibasic.calib.GEOMETRY_TABLE = self.geo_table
        muse_scibasic.param.saveimage = False

       
        for ifu in range(1, 25):
            calib = {'TRACE_TABLE': outcal + 'muse_FLAT_dir/TRACE_TABLE-%02i.fits' % ifu,
                     'MASTER_BIAS': outcal +'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu,
                     'MASTER_FLAT': outcal +'muse_FLAT_dir/MASTER_FLAT-%02i.fits' % ifu,
                     'WAVECAL_TABLE': outcal +'muse_WAVECAL_dir/WAVECAL_TABLE-%02i.fits' % ifu}
            if self.masterdark:
                calib['MASTER_DARK']= outcal + 'muse_DARK_dir/MASTER_DARK-%02i.fits' % ifu
            muse_scibasic(stdfile, tag='STD', param = {'nifu': ifu}, calib = calib)

    def run_standard_int(self): 
        # Initing ESOREX
        cpl.esorex.init(esorex_rc)
        cpl.esorex.log.file = self.outlog + 'std_moffat.log'
        muse_standard = cpl.Recipe('muse_standard')
        muse_standard.calib.STD_FLUX_TABLE = self.calibrationfolder + 'std_flux_table.fits'
        muse_standard.calib.EXTINCT_TABLE =  self.calibrationfolder + 'extinct_table.fits' #problem here?
  
        # run flux integration with Moffat profile fits:
        # MdB: the Moffat integration somehow takes the sky into account.
        # MdB: although it is not clear exactly how.

        outcal = self.my_params.mastercalib_folder
        try: os.mkdir(outcal + 'muse_POST_dir')
        except: pass
        
        try: os.mkdir(outcal + 'muse_POST_dir/moffat')
        except: pass
        
        muse_standard.output_dir = outcal + 'muse_POST_dir/moffat'
        # muse_standard.param.profile= 'circle'
        muse_standard([(outcal + 'muse_POST_dir/PIXTABLE_STD_0001-%02i.fits' % ifu) for ifu in range(1,25)])

    def run_basic(self, verbose=None):
        # Initing ESOREX
        cpl.esorex.init(esorex_rc)
        cpl.esorex.log.file = self.outlog + 'scibasic.log'
        muse_scibasic = cpl.Recipe('muse_scibasic')
  
        try: os.mkdir(self.reducedfiles_folder+'muse_SCIBASIC_dir')
        except: pass

        muse_scibasic.output_dir = self.reducedfiles_folder+'muse_SCIBASIC_dir'
        
        muse_scibasic.calib.BADPIX_TABLE = self.badpix_table  
        muse_scibasic.calib.GEOMETRY_TABLE = self.geo_table
        muse_scibasic.param.resample = True  
        muse_scibasic.param.crop = False

        if verbose is None: verbose = self.verbose
        if verbose:
            print "Running sci-basic on: ",self.objectlist
        objectlist = self.objectlist
        
        # MdB: Crop is True by default. 
        # MdB: I'm adding a list of bright sky lines here (standard pipeline feature).
        # MdB: Right now ILLUM is just random, should take the one
        # MdB: closest in time.
        outcal = self.my_params.mastercalib_folder
        for ifu in range(1, 25):
            calib = {'TRACE_TABLE': outcal + 'muse_FLAT_dir/TRACE_TABLE-%02i.fits' % ifu,
                     'MASTER_BIAS': outcal + 'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu,
                     'MASTER_FLAT': outcal + 'muse_FLAT_dir/MASTER_FLAT-%02i.fits' % ifu,
                     'WAVECAL_TABLE': outcal + 'muse_WAVECAL_dir/WAVECAL_TABLE-%02i.fits' % ifu}
            if self.masterdark:
                calib['MASTER_DARK'] = outcal +'muse_DARK_dir/MASTER_DARK-%02i.fits' % ifu
            if self.mastertwilight:
                calib['TWILIGHT_CUBE'] = outcal +'muse_TWILIGHT_dir/TWILIGHT_CUBE.fits'
            muse_scibasic({u'ILLUM': self.final_illumlist, u'OBJECT': objectlist}, param = {'nifu': ifu}, calib=calib)
                
        # MdB: we should add a twilight flat option here.

    def make_cube(self,align=None):
        # Initing ESOREX
        cpl.esorex.init(esorex_rc)
        cpl.esorex.log.file = self.outlog+'scipost1.log'
        try: os.mkdir(self.reducedfiles_folder+'muse_CUBE_dir/')
        except: pass

        muse_scipost = cpl.Recipe('muse_scipost')
        muse_scipost.output_dir = self.reducedfiles_folder+'muse_CUBE_dir/'

        muse_scipost.param.skymethod = 'none'
        # MdB: We subtract the sky externally at Cube level.
        muse_scipost.param.save = 'cube,individual'
        
        #muse_scipost.calib.LSF_CUBE = [('cal/LSF_CUBE-%02i.fits') % ifu
        #for ifu in range(1, 25)]
        # MdB: not sure why this is commented out.
        
        # MdB: We should check if this module is implemented. If it is
        # MdB: we should definitely use it.
        
        outcal = self.my_params.mastercalib_folder
        muse_scipost.calib.ASTROMETRY_WCS = self.astro_table
        muse_scipost.calib.SKY_LINES = self.calibrationfolder + 'sky_lines'
        if self.align_file is not None:
            muse_scipost.calib.OUTPUT_WCS = self.align_file
        muse_scipost.calib.EXTINCT_TABLE = self.calibrationfolder + 'extinct_table.fits'
        muse_scipost.calib.STD_RESPONSE = outcal+'muse_POST_dir/moffat/STD_RESPONSE_0001.fits'
        muse_scipost.calib.STD_TELLURIC = outcal+'muse_POST_dir/moffat/STD_TELLURIC_0001.fits'
        muse_scipost.calib.FILTER_LIST = self.calibrationfolder + 'filter_list.fits'
        muse_scipost.param.pixfrac = 1.
        muse_scipost.param.filter = 'white' #Johnson_V,Cousins_R,Cousins_I'
        muse_scipost.param.format = 'xCube'
        muse_scipost.param.crsigma = 15.0  #should this be 10?

        if (align==None):
            muse_scipost([(self.reducedfiles_folder + 'muse_SCIBASIC_dir/PIXTABLE_OBJECT_0001-%02i.fits' % ifu) for ifu in range(1, 25) ])
        else:
            objects=[(self.reducedfiles_folder + 'muse_SCIBASIC_dir/PIXTABLE_OBJECT_0001-%02i.fits' % ifu) for ifu in range(1, 25) ]
            muse_scipost({u'PIXTABLE_OBJECT':objects, u'OUTPUT_WCS':align})
            objects=[(self.reducedfiles_folder + 'muse_SCIBASIC_dir/PIXTABLE_OBJECT_0001-%02i.fits' % ifu) for ifu in range(1, 25) ]
            muse_scipost({u'PIXTABLE_OBJECT':objects, u'OUTPUT_WCS':align})

    def run(self, verbose=None):
        if verbose is None: verbose = self.verbose
        if verbose:
            print "Checking and reducing calibration\n"
        self.create_calibrations()
        self.run_basic()
        self.make_cube(align=self.align_file)
