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
import time
import glob
import subprocess
from subprocess import call

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


# Importing pymuselp modules
from pymuselp.init_muselp import init_muse_parameters

__version__ = '0.0.2 (08 March    2018)'
__version__ = '0.0.1 (21 November 2017)'

############################################################
#                      BEGIN
# The following parameters can be adjusted for the need of
# the specific pipeline to be used
############################################################

# NOTE: most of the parameters have now been migrated to
# init_muselp.py for consistency.

listexpo_types = {'DARK': 'DARK', 'BIAS' : 'BIAS', 'FLAT': 'FLAT',
        'ARC' : 'ARC', 'ILLUM' : 'ILLUM', 'TWILIGHT': 'TWILIGHT',
        'STD': 'STD', 'AST': 'AST'}

listexpo_files = {"TYPE" : ['ESO DPR TYPE', "raw_hdr_dprtype.txt"],
        "DATE": ['MJD-OBS', "raw_hdr_mjd.txt"]}

############################################################
#                      END
############################################################

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
    
class muse_pipe(object):
    """Main Class to define and run the MUSE pipeline, given a certain galaxy name
    
    OLD WAY
        mp = muse_pipe(galaxyname='NGC7162_I',rawdir='/scratch/denbrokm/reduction/raw/NGC7162_I/muse_raw_files/',outsci='/scratch/denbrokm/reduction/out/NGC7162/sci/A/',outcal='/scratch/denbrokm/reduction/out/NGC7162_I/cal/', outlog='/scratch/denbrokm/reduction/out/NGC7162_I/logs/', objects=['/scratch/denbrokm/reduction/raw/NGC7162_I/muse_raw_files/MUSE.2015-10-12T00:10:50.779.fits'])

    NEW WAY
        musep = muse_pipe(galaxyname='NGC1087', rc_filename="", cal_filename="", outlog"NGC1087_log1.log",
        objects=[''])
    mp.run()
    """

    def __init__(self, galaxyname=None, objectlist=[], rc_filename=None, cal_filename=None, outlog=None, verbose=True, redo=False, align_file=None):
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
        """

        # Setting up the folders and names for the calibration files
        # Can be initialised by either an rc_file, a default rc_file or harcoded defaults.
        self.my_params = init_muse_parameters(rc_filename=rc_filename, cal_filename=cal_filename)
    
        self.galaxyname = galaxyname
        if outlog is None : 
            outlog = "log_{timestamp}.log".format(timestamp=create_time_name())
        self.outlog = outlog
        self.verbose = verbose
        self.redo = redo
        self.align_file = align_file
        self.objectlist = objectlist
        
        # MdB: we assume that, as a standard, no dark is present
        self.masterdark=0
        
        # MdB: try to make the out directory for the calibration files
        # MdB: if it doesn't exist yet.
        if self.verbose:
            print "Creating directory structure"
        safely_create_folder(self.my_params.reducedfiles_folder)
        safely_create_folder(self.my_params.mastercalib_folder)
        safely_create_folder(self.my_params.pixtable_folder)
#        safely_create_folder(outlog)
        
        # First, list all the files and find out which types they are
        self.probe_files()
        # Making the lists of various types of exposures
        self.make_lists()
        # Selecting the ones which 
        self.select_run_tables()

    def probe_files(self, folder, verbose=None) :
        """Determine which files are in the raw folder
        """
        rawfolder = self.my_params.rawdata_folder
        command = "fitsheader --extension 0 --keyword {keyword} --table ascii.tab {rawfolder}/*fits.fz > {rawfolder}raw_hdr_dprtype.txt".format(rawfolder=rawfolder, filename=listexpo_files['TYPE'][1], keyword=listexpo_files['TYPE'][0])
        call(command)
        command = "fitsheader --extension 0 --keyword {keyword} --table ascii.tab {rawfolder}/*fits.fz > {rawfolder}raw_hdr_mjd.txt".format(rawfolder=rawfolder, filename=listexpo_files['DATE'][1], keyword=listexpo_files['DATE'][0])
        call(command)

        info_types = ascii.read(listexpo_files['TYPE'])
        info_dates = ascii.read(listexpo_files['DATE'])

        self.info_expo = {}
        for filename in info_types['filename'] :
            if filename in info_dates['filename'] :
                expotype = info_types['value'] 
                expodate = info_dates['value']
                self.info_expo[filename] = [expotype, expodate]
                # removing line from the TYPE and DATE files
                info_types.pop(filename)
                info_dates.pop(filename)

        # Printing out the ones which were not found in both files
        if verbose is None: verbose = self.verbose
        if verbose:
            for missedname in info_types['filename'] :
                print("WARNING: file {filename} was missing a DATE keyword")
            for missedname in info_dates['filename'] :
                print("WARNING: file {filename} was missing a TYPE keyword")

        # Reseting the select_type item
        self.select_type = {}
        for expotype in listexpo_types.keys() :
            setattr(self, expotype, [])

    def sort_types(self, expotype=None) :
        """Provide 
        """
        if expotype is None : 
            print("No expotype defined")

        self.select_type[expotype] = []
        for expo in self.info_expo.keys() :
            if (self.info_expo[expo] == expotype) :
                self.select_type[expotype].append(expo)
        
    def select_WFM_N(self, folder, category):
        """Check the fits file and whether they are WFM_N compatible
        """
        ghost_list = []
        ghost_list_all = glob.glob(folder + category + '/*.fits')
        for ii in range(len(ghost_list_all)):
            hdulist = pyfits.open(ghost_list_all[ii])
            mode = hdulist[0].header['HIERARCH ESO INS MODE']
            if mode == 'WFM-NOAO-N':
                ghost_list.append(ghost_list_all[ii])       
        return ghost_list

    def make_lists(self):
        """Create lists of files according to their chacteristics
        """
        rawfolder = self.my_params.rawdata_folder
        self.illumlist = self.select_WFM_N(rawfolder ,'ILLUM')
        self.select_illumfiles()
        #Kyriakos/Martina: leave illum like this and delete unneeded illums in respective folders 
        # MdB: this is really weird,
        # MdB: we should write a module that chooses the one closest in time!
        
        self.flatlist = self.select_WFM_N(rawfolder, 'FLAT')
        self.darklist = glob.glob(rawfolder + 'DARK/*.fits')
        self.wavelist = self.select_WFM_N(rawfolder, 'ARC') 
        self.skyflatlist = self.select_WFM_N(rawfolder, 'TWILIGHT') + self.select_WFM_N(rawfolder, 'SKYFLAT')
        self.biaslist = glob.glob(rawfolder + 'BIAS/*.fits')
        self.lsflist = self.wavelist
        self.stdlist = self.select_WFM_N(rawfolder, 'STD')
        self.select_stdfile()
        self.astrometryfile = rawfolder + 'AST/*.fits'
        
    def create_calibrations(self):
        self.check_for_calibrations()
        # MdB: this creates the standard calibrations, if needed
        if not ((self.redo==0) and (self.masterbias==1)):
            self.run_bias()
        if not ((self.redo==0) and (self.masterdark==1)):
            self.run_dark()
        if not ((self.redo==0) and (self.masterflat==1)):
            self.run_flat()
        if not ((self.redo==0) and (self.wavecal==1)):
            self.run_wavecal()
        if not ((self.redo==0) and (self.mastertwilight==1)):
            self.run_twilight()    
        if not ((self.redo==0) and (self.standard==1)):
            self.create_standard_star()
            self.run_standard_int()
        
    def check_for_calibrations(self, verbose=None):
        """This function checks which calibration file are present, just in case
        you do not wish to redo them. Variables are named: 
        masterbias, masterdark, standard, masterflat, mastertwilight
        """
        
        outcal = self.mastercalib_folder
        # master bias:
        self.masterbias = 1
        for ifu in range(1,25):
            if not os.path.isfile(outcal + 'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu):
                self.masterbias = 0

        # master dark
        self.masterdark = 1
        for ifu in range(1,25):
            if not os.path.isfile(outcal + 'muse_DARK_dir/MASTER_DARK-%02i.fits' % ifu):
                self.masterdark = 0

        # standard star
        self.standard = 1
        for ifu in range(1,25):
            if not os.path.isfile(outcal + 'muse_POST_dir/PIXTABLE_STD_0001-%02i.fits' % ifu):
                self.standard = 0

        # wavecal
        self.wavecal = 1
        for ifu in range(1,25):
            if not os.path.isfile(outcal + 'muse_WAVECAL_dir/WAVECAL_TABLE-%02i.fits' % ifu):
                self.wavecal = 0
        
        # flat
        self.masterflat = 1
        for ifu in range(1,25):
            if not os.path.isfile(outcal + 'muse_FLAT_dir/MASTER_FLAT-%02i.fits' % ifu):
                self.masterflat = 0
        
        # twilight
        self.mastertwilight = 1
        if not os.path.isfile(outcal + 'muse_TWILIGHT_dir/TWILIGHT_CUBE.fits'):
            self.mastertwilight=0

        if verbose is None: verbose = self.verbose
        if verbose:
            if (self.masterbias):
                print "Master bias already made"

            if (self.masterflat):
                print "Master flat already made"
                
            if (self.masterdark):
                print "Master dark already made"

            if (self.standard):
                print "Standard star already made"
                
            if (self.mastertwilight):
                print "Twilight flats already made"
                
            if (self.wavecal):
                print "Wavecal already made" 
            
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
        for runs in MUSELP_runs.keys():
            Pdates = MUSELP_runs[key]
            date1 = np.datetime64(Pdates[1]).astype('datetime64[D]')
            date2 = np.datetime64(Pdates[2]).astype('datetime64[D]')
            if ((filetime >= date1) and (filetime <= date2)):
                runname = key
        if verbose is None: verbose = self.verbose
        if verbose:
            print "Using geometry calibration data from MUSE runs %s\n"%finalkey
        self.geo_table = "{calfolder}geometry_table_wfm_{runname}.fits".format(calfolder=self.my_params.musecalib_folder, runname=runname) 
        self.astro_table = "{calfolder}astrometry_wcs_wfm_{runname}.fits".format(calfolder=self.my_params.musecalib_folder, runname=runname)

    def select_illumfiles(self):
        # MdB: adapted from Martina's script:
        # MdB: minimize the time difference between object and illum
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
        
    def run_bias(self):
        biaslist=self.biaslist
       
        cpl.esorex.init()
        cpl.esorex.log.file = self.outlog + 'bias.log'
        muse_bias = cpl.Recipe('muse_bias')
        muse_bias.param.nifu=-1
        try: os.mkdir(self.my_params.mastercalib_folder + 'muse_BIAS_dir')
        except: pass
        muse_bias.output_dir = self.my_params.mastercalib_folder + 'muse_BIAS_dir'
        muse_bias.calib.BADPIX_TABLE = self.badpix_table
        muse_bias(biaslist)

    # MdB: Make master dark
    def run_dark(self):
        darklist=self.darklist
        if len(darklist) > 0:
            cpl.esorex.init()
            cpl.esorex.log.file = self.outlog+'dark.log'
            muse_dark = cpl.Recipe('muse_dark')
            try: os.mkdir(self.my_params.mastercalib_folder +'muse_DARK_dir')
            except: pass
            muse_dark.output_dir = self.my_params.mastercalib_folder +'muse_DARK_dir'
            muse_dark.param.nifu=-1 #.samples?
            muse_dark.calib.BADPIX_TABLE = self.badpix_table
            muse_dark.calib.MASTER_BIAS = [(self.my_params.mastercalib_folder +'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu) for ifu in range(1, 25)]
            muse_dark(darklist)
            self.masterdark=1
        else:
            self.masterdark=0
        
    # MdB: make master dome flat  
    def run_flat(self):
        cpl.esorex.init()
        cpl.esorex.log.file = self.outlog+'flat.log'
        muse_flat = cpl.Recipe('muse_flat')
        
        try: os.mkdir(self.my_params.mastercalib_folder +'muse_FLAT_dir')
        except: pass
        
        muse_flat.output_dir = self.my_params.mastercalib_folder +'muse_FLAT_dir'
        muse_flat.param.samples = True
        muse_flat.param.nifu=-1 
        # MdB: samples is advocated on page 20 of the software manual
        # MdB: but I'm not sure if it's actually useful.
        muse_flat.calib.BADPIX_TABLE = self.badpix_table
        muse_flat.calib.MASTER_BIAS = [(self.my_params.mastercalib_folder +'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu) for ifu in range(1, 25)]
        if self.masterdark:
            muse_flat.calib.MASTER_DARK = [(self.my_params.mastercalib_folder +'muse_DARK_dir/MASTER_DARK-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_flat(self.flatlist)#CAREFUL HERE! is flat ok?
        # MdB: This is somewhat worrisome? 

        print 'finito master_flat'       
    
    # MdB: wavelength calibration
    def run_wavecal(self):
        # Initing ESOREX
        cpl.esorex.init()
        # Setting up the Log File
        cpl.esorex.log.file = self.outlog+'wavecal.log'
        wavelist = self.wavelist
        outcal = self.my_params.mastercalib_folder
        muse_wavecal = cpl.Recipe('muse_wavecal')
        
        try: os.mkdir(outcal + 'muse_WAVECAL_dir')
        except: pass

        muse_wavecal.output_dir = outcal + 'muse_WAVECAL_dir'
        muse_wavecal.calib.LINE_CATALOG = self.calibrationfolder + 'line_catalog.fits' #for theia+
        muse_wavecal.param.residuals = False
        muse_wavecal.param.resample = False
        muse_wavecal.param.wavemap=False
        # MdB: these are in fact the default settings
        muse_wavecal.calib.BADPIX_TABLE = self.badpix_table 
        muse_wavecal.param.nifu = -1
        muse_wavecal.calib.MASTER_BIAS = [(outcal + 'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_wavecal.calib.MASTER_FLAT = [(outcal + 'muse_FLAT_dir/MASTER_FLAT-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_wavecal.calib.TRACE_TABLE = [(outcal + 'muse_FLAT_dir/TRACE_TABLE-%02i.fits' % ifu) for ifu in range(1, 25)]
        if self.masterdark:
            muse_wavecal.calib.MASTER_DARK = [(outcal + 'muse_DARK_dir/MASTER_DARK-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_wavecal(wavelist)

    def run_twilight(self):
        if len(self.skyflatlist) == 0:
            return

        outcal = self.my_params.mastercalib_folder
        # Initing ESOREX
        cpl.esorex.init()
        # Setting up the Log File
        cpl.esorex.log.file = self.outlog+'twilight.log'
        muse_twilight = cpl.Recipe('muse_twilight')
        
        try: os.mkdir(outcal + 'muse_TWILIGHT_dir')
        except: pass
        
        muse_twilight.output_dir = outcal + 'muse_TWILIGHT_dir'
        muse_twilight.calib.MASTER_BIAS = [(outcal + 'muse_BIAS_dir/MASTER_BIAS-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_twilight.calib.MASTER_FLAT = [(outcal + 'muse_FLAT_dir/MASTER_FLAT-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_twilight.calib.TRACE_TABLE = [(outcal + 'muse_FLAT_dir/TRACE_TABLE-%02i.fits' % ifu) for ifu in range(1, 25)]
        muse_twilight.calib.WAVECAL_TABLE = [(outcal + 'muse_WAVECAL_dir/WAVECAL_TABLE-%02i.fits' % ifu) for ifu in range(1, 25)]
        if self.masterdark:
            muse_twilight.calib.MASTER_DARK = [(outcal + 'muse_DARK_dir/MASTER_DARK-%02i.fits' % ifu) for ifu in range(1, 25)]


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
        cpl.esorex.init()
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
        cpl.esorex.init()
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
        cpl.esorex.init()
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
        cpl.esorex.init()
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

    def run(self, verbose=None):
        if verbose is None: verbose = self.verbose
        if verbose:
            print "Checking and reducing calibration\n"
        self.create_calibrations()
        self.run_basic()
        self.make_cube(align=self.align_file)
