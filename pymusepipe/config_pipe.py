# Licensed under a MIT license - see LICENSE

"""MUSE-PHANGS configuration module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017-2019, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"

# import modules
import numpy as np

#======================================================================#
# BEGIN
#           FILTER LIST
#======================================================================#

default_filter_list = ("white,Johnson_B,Johnson_V,Cousins_R,"
                       "SDSS_g,SDSS_r,SDSS_i")
default_short_filter_list = ("white,Cousins_R")
default_PHANGS_filter_list = ("white,Johnson_B,Johnson_V,Cousins_R,"
                       "SDSS_g,SDSS_r,SDSS_i,WFI_BB,WFI_NB,DUPONT_R")
default_short_PHANGS_filter_list = ("white,WFI_BB,DUPONT_R")

#-- END ---------------------------------------------------------------#

#======================================================================#
# BEGIN
#           MUSE PIPELINE FITS KEYWORDS parameters
#======================================================================#
# For musepipe module

dict_expotypes = {'DARK': 'DARK', 'BIAS' : 'BIAS', 'FLAT': 'FLAT,LAMP',
        'ILLUM': 'FLAT,LAMP,ILLUM', 'TWILIGHT': 'FLAT,SKY', 
        'WAVE': 'WAVE', 'STD': 'STD', 'AST': 'AST',
        'OBJECT': 'OBJECT', 'SKY': 'SKY',
        'ASTROMETRY': 'ASTROMETRY',
        'GEOMETRY': 'GEOMETRY'
        }

dict_astrogeo = {'ASTROMETRY': 'Astrometric calibration (ASTROMETRY)',
                'GEOMETRY': 'WAVE,MASK'
                }

# This dictionary contains the types
dict_listMaster = {'DARK': 'MASTER_DARK',
        'BIAS': 'MASTER_BIAS', 
        'FLAT': 'MASTER_FLAT',
        'TRACE': 'TRACE_TABLE',
        'TWILIGHT': 'TWILIGHT_CUBE', 
        'WAVE': 'WAVECAL_TABLE', 
        'LSF': 'LSF_PROFILE', 
        'STD': 'PIXTABLE_STD' 
        }

dict_listObject = {'OBJECT': 'PIXTABLE_OBJECT',
        'SKY': 'PIXTABLE_SKY', 
        'STD': 'PIXTABLE_STD',
        'REDUCED': 'PIXTABLE_REDUCED'
        }

dict_listMasterObject = {**dict_listMaster, **dict_listObject}

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

# Suffix for the pre/post-alignment files. Will be part of the output names
suffix_prealign = "_prealign"
suffix_checkalign = "_checkalign"

# List of suffix you wish to have scanned
suffix_rawfiles = ['fits.fz', 'fits']

#-- END ---------------------------------------------------------------#
#======================================================================#
# BEGIN
#           DEFAULT PATHS for the Muse pipeline
# 
# Default hard-coded folders
# The setting of these folders can be overwritten 
# by a given rc file if provided
#======================================================================#
# for init_musepipe module

dict_user_folders = {
        # values provide the folder and whether or not this should be attempted to create
            # Muse calibration files (common to all)
            "musecalib": "/home/mcelroy/reflex/install/calib/muse-2.2/cal/'",
            # Time varying calibrations
            "musecalib_time": "/data/beegfs/astro-storage/groups/schinnerer/PHANGS/MUSE/LP_131117/astrocal/",
            # Calibration files (specific to OBs)
            "root" : "/mnt/fhgfs/PHANGS/MUSE/LP_131117/",
            }

# Default initialisation file
default_rc_filename = "~/.musepiperc"

PHANGS_reduc_config = {'fakemode':False, 'nocache':False, 'checkmode':False, 
                 'overwrite_astropy_table':True, 'filter_list':"white,WFI_BB,DUPONT_R",
                 'filter_for_alignment':"WFI_BB"}
PHANGS_combine_config = {'fakemode':False, 'nocache':False}

# Extra filters which may be used in the course of the reduction
dict_extra_filters = {
        # Narrow band filter 
        "WFI_BB": "data/Filters/LaSilla_WFI_ESO844p.txt",
        # Broad band filter
        "WFI_NB": "data/Filters/LaSilla_WFI_ESO856p.txt",
        # DUPONT R filter
	    "DUPONT_R": "data/Filters/LCO_SITe3_rp.txt"
        }

# Default hard-coded fits files - Calibration Tables
# These are also overwritten by the given calib input file (if provided)
dict_calib_tables = {
            # Muse geometric file
            "geo_table": "geometry_table_wfm.fits",
            # Astrometry
            "astro_table" : "astrometry_wcs_wfm.fits",
            # Bad pixel table
            "badpix_table" : "badpix_table_2015-06-02.fits",
            # Vignetting
            "vignetting_mask" : "vignetting_mask.fits",
            # Flux table
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

# Default structure folders
# If already existing, won't be created
# If not, will be created automatically
dict_input_folders = {
            # Raw Data files
            "rawfiles" : "Raw/",
            # Config files
            "config" : "Config/",
            # Tables
            "astro_tables" : "Astro_tables/",
            # esores log files
            "esorex_log" : "Esorex_log/",
            # Data Products - first writing
            "pipe_products": "Pipe_products/",
            # Log
            "log": "Log/"
            }

# Values provide the folder names for the file structure
# If already existing, won't be created
# If not, will be created automatically
dict_folders = {
            # Master Calibration files
            "master" : "Master/",
            # Object files
            "object" : "Object/",
            # Sky files
            "sky" : "Sky/",
            # Std files
            "std" : "Std/",
            # Reconstructed Maps
            "maps" : "Maps/",
            # SOF folder 
            "sof" : "Sof/", 
            # Figure
            "figures" : "Figures/",
            }

# This dictionary includes extra folders for certain specific task
# e.g., alignment - associated with the target
# Will be created automatically if not already existing
dict_folders_target = {
        "alignment" : "Alignment/",
        }

dict_combined_folders = {
        # Sof files
        "sof": "Sof/",
        # Combined products
        "cubes": "Cubes/",
         # esores log files
        "esorex_log" : "Esorex_log/",
        # Data Products - first writing
        "pipe_products": "Pipe_products/",
        # Log
        "log": "Log/"
        }

#-- END ---------------------------------------------------------------#
#======================================================================#
# BEGIN
#           DEFAULT GEOMETRY AND ASTROWCS FILES
# 
# These are provided if you need time varying astrometry files
#======================================================================#
# For musepipe module

dict_geo_astrowcs_table = {
    'comm1' :   ['2014-02-09', '2014-02-09'], # mean date
    'comm2a':   ['2014-04-27', '2014-05-06'],
    'comm2b':   ['2014-07-24', '2014-08-03'],
    'gto01' :   ['2014-09-13', '2014-09-26'],
    'gto02' :   ['2014-10-18', '2014-10-29'],
    'gto03' :   ['2014-11-16', '2014-11-27'],
    '2014d' :   ['2014-12-16', '2014-12-25'],
    'gto05' :   ['2015-04-14', '2015-04-25'],
    'gto06' :   ['2015-05-10', '2015-05-23'],
    'gto07' :   ['2015-08-17', '2015-08-24'],
    'gto08' :   ['2015-09-07', '2015-09-12'],
    'gto09' :   ['2015-10-09', '2015-10-16'],
    'gto10' :   ['2015-11-04', '2015-11-13'],
    'gto11' :   ['2016-01-31', '2016-02-06'],
    'gto12' :   ['2016-03-09', '2016-03-14'],
    'gto13' :   ['2016-04-05', '2016-04-10'],
    'gto14' :   ['2016-05-06', '2016-05-12'],
    'gto15' :   ['2016-08-29', '2016-09-06'],
    'gto16' :   ['2017-01-27', '2017-02-03'],
# Initial gto17 is actually extended with gto18
# which did not have a solution
#    'gto17' :   ['2017-04-22', '2017-04-25'],
#    'gto18' :   ['2017-05-20', '2017-05-23'],
    'gto17' :   ['2017-04-22', '2017-05-23'],
    'gto19' :   ['2017-09-19', '2017-09-25'],
    'gto20' :   ['2017-10-16', '2017-10-26'],
    'gto21' :   ['2017-11-15', '2017-11-20'],
    'gto22' :   ['2018-02-11', '2018-02-16'],
    'gto23' :   ['2018-03-14', '2018-03-19'],
    'gto24' :   ['2018-04-11', '2018-04-19'],
    'gto25' :   ['2018-05-10', '2018-05-13'],
    'gto26' :   ['2018-08-12', '2018-08-18'],
    'gto27' :   ['2018-09-05', '2018-09-15'],
    'gto28' :   ['2018-10-07', '2018-10-15'],
# gto30 did not provide a solution hence
# gto29 also includes gto30 run
#    'gto29' :   ['2018-07-14', '2018-11-14'], 
#    'gto30' :   ['2018-12-06', '2018-12-12'],
    'gto29' :   ['2018-07-14', '2018-12-12'],
    'gto31' :   ['2019-01-02', '2019-01-09'],
    'gto32' :   ['2019-03-04', '2019-03-07'],
    'gto33' :   ['2019-04-05', '2019-04-11'],
    'gto34' :   ['2019-05-03', '2019-05-05'] 
}
#-- END ---------------------------------------------------------------#

#======================================================================#
# BEGIN
#           DEFAULT KEYWORDS and COLUMNS for OFFSET_TABLES
# 
# Define useful keywords for fits table and images
#======================================================================#
mjd_names = {'table': "MJD_OBS", 'image': "MJD-OBS"}
date_names = {'table': "DATE_OBS", 'image': "DATE-OBS"}
tpl_names = {'table': "TPL_START", 'image': "HIERARCH ESO TPL START"}
iexpo_names = {'table': "IEXPO_OBS", 'image': "MUSEPIPE_IEXPO"}
pointing_names = {'table': "POINTING_OBS", 'image': "MUSEPIPE_POINTING"}

default_offset_table = {'date': [date_names['table'], 'S23', ""],
                        'mjd' : [mjd_names['table'], 'f8', 0.0],
                        'tpls' : [tpl_names['table'], 'S19', ""],
                        'iexpo' : [iexpo_names['table'], 'i4', 0],
                        'pointing' : [iexpo_names['table'], 'i4', 0],
                        'ora':["RA_OFFSET", 'f8', 0.0],
                        'odec':["DEC_OFFSET", 'f8', 0.0],
                        'scale':["FLUX_SCALE", 'f8', 1.0]}
#-- END ---------------------------------------------------------------#

#--- Some useful functions using the input configuration --------------#
def get_suffix_product(expotype):
    return dict_listMasterObject[expotype]

default_prefix_wcs = "refwcs_"
prefix_mosaic = "full"
default_prefix_wcs_mosaic = "{0}{1}".format(prefix_mosaic, default_prefix_wcs)
default_wave_wcs = 6500.0
default_prefix_mask = "mask_"

AO_mask_lambda = [5800, 5970]
lambdaminmax_for_wcs = [6800, 6805]
lambdaminmax_for_mosaic = [4700, 9400]

#===========================================
# Recipes for data reduction
dict_recipes_per_num = {1:'bias', 2:'flat', 3:'wave',
               4: 'lsf', 5:'twilight', 
               6:'scibasic_all', 7:'standard',
               8:'sky', 9:'prep_align', 10:'align_bypointing', 
               11:'align_bygroup', 12:'scipost_perexpo',
               13:'scipost_sky', 14:'combine_pointing'}

# and creating the inverse dictionary
dict_recipes_per_name = {}
for key in dict_recipes_per_num:
    dict_recipes_per_name[dict_recipes_per_num[key]] = key
