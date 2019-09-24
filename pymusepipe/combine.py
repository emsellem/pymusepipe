# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS core module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# Importing modules
import numpy as np
import os
from os.path import join as joinpath
import glob
import copy

try :
    import astropy as apy
    from astropy.io import fits as pyfits
except ImportError :
    raise Exception("astropy is required for this module")

from astropy.utils.exceptions import AstropyWarning
try :
    from astropy.table import Table
except ImportError :
    raise Exception("astropy.table.Table is required for this module")

import warnings

# Importing pymusepipe modules
from pymusepipe.recipes_pipe import PipeRecipes
from pymusepipe.create_sof import SofPipe
from pymusepipe.init_musepipe import InitMuseParameters
import pymusepipe.util_pipe as upipe
from pymusepipe import musepipe, prep_recipes_pipe 
from pymusepipe.config_pipe import default_filter_list

# Default keywords for MJD and DATE
from pymusepipe.align_pipe import mjd_names, date_names

__version__ = '0.0.3 (4 Sep 2019)'
# 0.0.2 28 Feb, 2019: trying to make it work
# 0.0.1 21 Nov, 2017 : just setting up the scene

dic_combined_folders = {
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

class MusePointings(SofPipe, PipeRecipes) :
    def __init__(self, targetname=None, list_pointings=[1], 
            dic_exposures_in_pointing=None,
            suffix_fixed_pixtables="tmask",
            use_fixed_pixtables=False,
            rc_filename=None, cal_filename=None, 
            combined_folder_name="Combined", suffix="",
            offset_table_name=None,
            logfile="MusePipeCombine.log", reset_log=False,
            verbose=True, debug=False, **kwargs):
        """Initialisation of class muse_expo

        Input
        -----
        targetname: string (e.g., 'NGC1208'). default is None. 

        rc_filename: str
            filename to initialise folders
        cal_filename: str
            filename to initiale FIXED calibration MUSE files
        verbose: bool 
            Give more information as output (default is True)
        debug: bool
            Allows to get more messages when needed
            Default is False
        vsystemic: float 
            Default is 0. Systemic velocity of the galaxy [in km/s]
        suffix_fixed_pixtables: str
            Default is "tmask". Suffix for fixed PixTables
        use_fixed_pixtables: bool
            Default is False. If True, will use suffix_fixed_pixtables to filter out
            Pixtables which have been fixed.

        Other possible entries
        ----------------------
        warnings: strong  ('ignore'by default. If set to ignore, will ignore the Astropy Warnings.

        """
        # Verbose option
        self.verbose = verbose
        self.debug = debug

        # Warnings for astropy
        self.warnings = kwargs.pop("warnings", 'ignore')
        if self.warnings == 'ignore':
           warnings.simplefilter('ignore', category=AstropyWarning)

        # Setting the default attibutes #####################
        self.targetname = targetname
        self.list_pointings = list_pointings
        self.filter_list = kwargs.pop("filter_list", default_filter_list)

        self.combined_folder_name = combined_folder_name
        self.vsystemic = np.float(kwargs.pop("vsystemic", 0.))

        # Setting other default attributes
        if logfile is None : 
            logfile = "log_{timestamp}.txt".format(timestamp = upipe.create_time_name())
            upipe.print_info("The Log file will be {0}".format(logfile))
        self.logfile = logfile
        self.suffix = suffix

        # End of parameter settings #########################

        # Init of the subclasses
        PipeRecipes.__init__(self, **kwargs)
        SofPipe.__init__(self)

        # ---------------------------------------------------------
        # Setting up the folders and names for the data reduction
        # Can be initialised by either an rc_file, 
        # or a default rc_file or harcoded defaults.
        self.pipe_params = InitMuseParameters(rc_filename=rc_filename, 
                            cal_filename=cal_filename)

        # Setting up the relative path for the data, using Galaxy Name + Pointing
        self.pipe_params.data = "{0}/{1}/".format(self.targetname, self.combined_folder_name)

        self.pipe_params.init_default_param(dic_combined_folders)
        self._dic_combined_folders = dic_combined_folders

        # Including or not the fixed Pixtables in place of the original ones
        self.use_fixed_pixtables = use_fixed_pixtables
        self.suffix_fixed_pixtables = suffix_fixed_pixtables

        # Setting all the useful paths
        self.set_fullpath_names()
        self.paths.logfile = joinpath(self.paths.log, logfile)

        # and Recording the folder where we start
        self.paths.orig = os.getcwd()

        # END Set up params =======================================

        # =========================================================== 
        # ---------------------------------------------------------
        # Create the Combined folder
        upipe.safely_create_folder(self.paths.data, verbose=verbose)

        # Go to the Combined Folder
        self.goto_folder(self.paths.data)

        # Now create full path folder 
        for folder in self._dic_combined_folders.keys() :
            upipe.safely_create_folder(self._dic_combined_folders[folder], verbose=verbose)

        # Checking input pointings and pixtables
        self._check_pointings(dic_exposures_in_pointing)

        # Checking input offset table and corresponding pixtables
        folder_offset_table = kwargs.pop("folder_offset_table", self.paths.alignment)
        self._check_offset_table(offset_table_name, folder_offset_table)
        # END CHECK UP ============================================

        # Making the output folders in a safe mode
        if self.verbose:
            upipe.print_info("Creating directory structure")

        # Going back to initial working directory
        self.goto_origfolder()

    def _check_pointings(self, dic_exposures_in_pointings=None):
        """Check if pointings and dictionary are compatible
        """
        # Dictionary of exposures to select per pointing
        self.dic_exposures_in_pointings = dic_exposures_in_pointings

        # Getting the pieces of the names to be used for pixtabs
        pixtable_suffix = prep_recipes_pipe.dic_products_scipost['individual'][0]

        # Initialise the dictionary of pixtabs to be found in each pointing
        self.dic_pixtabs_in_pointings = {}
        self.dic_allpixtabs_in_pointings = {}
        # Loop on Pointings
        for pointing in self.list_pointings:
            # get the path
            path_pointing = getattr(self.paths, self.dic_name_pointings[pointing])
            # List existing pixtabs, using the given suffix
            list_pixtabs = glob.glob(path_pointing + self.pipe_params.object + 
                    "{0}{1}*fits".format(pixtable_suffix, self.suffix))

            # Take (or not) the fixed pixtables
            if self.use_fixed_pixtables:
                suffix_to_consider = "{0}{1}".format(self.suffix_fixed_pixtables,
                        pixtable_suffix)
                list_fixed_pixtabs = glob.glob(path_pointing + self.pipe_params.object + 
                    "{0}{1}*fits".format(suffix_to_consider, self.suffix))

                # Looping over the existing fixed pixtables
                for fixed_pixtab in list_fixed_pixtabs:
                    # Finding the name of the original one
                    orig_pixtab = fixed_pixtab.replace(suffix_to_consider, 
                                            pixtable_suffix)
                    if orig_pixtab in list_pixtabs:
                        # If it exists, remove it
                        list_pixtabs.remove(orig_pixtab)
                        # and add the fixed one
                        list_pixtabs.append(fixed_pixtab)
                        upipe.print_warning("Fixed PixTable {0} was included".format(
                                fixed_pixtab))
                        upipe.print_warning("and Pixtable {0} was thus removed".format(
                                orig_pixtab))
                    else:
                        upipe.print_warning("Original Pixtable {0} not found".format(
                                orig_pixtab))
                        upipe.print_warning("Hence will not include fixed PixTable "
                                "{0}".format(fixed_pixtab))

            self.dic_allpixtabs_in_pointings[pointing] = (copy.copy(list_pixtabs)).sort()

            # if no selection on exposure names are given
            # Select all existing pixtabs
            if self.dic_exposures_in_pointings is None:
                select_list_pixtabs = list_pixtabs

            # Otherwise use the ones which are given via their expo numbers
            else:
                select_list_pixtabs = []
                # this is the list of exposures to consider
                list_expo = self.dic_exposures_in_pointings[pointing]
                # We loop on that list
                for expotuple in list_expo:
                    tpl, nexpo = expotuple[0], expotuple[1]
                    for expo in nexpo:
                        # Check whether this exists in the our cube list
                        suffix_expo = "{0}_{1:04d}".format(tpl, expo)
                        if self.debug:
                            upipe.print_debug("Checking which exposures are tested")
                            upipe.print_debug(suffix_expo)
                        for pixtab in list_pixtabs:
                            if suffix_expo in pixtab:
                                # We select the cube
                                select_list_pixtabs.append(pixtab)
                                # And remove it from the list
                                list_pixtabs.remove(pixtab)
                                # We break out of the cube for loop
                                break

            select_list_pixtabs.sort()
            self.dic_pixtabs_in_pointings[pointing] = select_list_pixtabs

    def _check_offset_table(self, offset_table_name=None, folder_offset_table=None):
        """Checking if DATE-OBS and MJD-OBS are in the OFFSET Table
        """
        self.offset_table_name = offset_table_name
        if self.offset_table_name is None:
            upipe.print_warning("No Offset table given")
            return
        
        # Using the given folder name, alignment one by default
        if folder_offset_table is None:
            self.folder_offset_table = self.paths.alignment
        else:
            self.folder_offset_table = folder_offset_table

        full_offset_table_name = joinpath(self.folder_offset_table,
                                    self.offset_table_name)
        if not os.path.isfile(full_offset_table_name):
            upipe.print_error("Offset table [{0}] not found".format(
                full_offset_table_name))
            return

        # Opening the offset table
        offset_table = Table.read(full_offset_table_name)

        # getting the MJD and DATE from the OFFSET table
        self.table_mjdobs = offset_table[mjd_names['table']]
        self.table_dateobs = offset_table[date_names['table']]

        # Checking existence of each pixel_table in the offset table
        nexcluded_pixtab = 0
        nincluded_pixtab = 0
        for pointing in self.list_pointings:
            pixtab_to_exclude = []
            for pixtab_name in self.dic_pixtabs_in_pointings[pointing]:
                pixtab_header = pyfits.getheader(pixtab_name)
                mjd_obs = pixtab_header['MJD-OBS']
                date_obs = pixtab_header['DATE-OBS']
                # First check MJD
                index = np.argwhere(self.table_mjdobs == mjd_obs)
                # Then check DATE
                if (index.size == 0) or (self.table_dateobs[index] != date_obs):
                    upipe.warning("PIXELTABLE {0} not found in OFFSET table: "
                            "please Check MJD-OBS and DATE-OBS".format(pixtab_name))
                    pixtab_to_exclude.append(pixtab_name)
                nincluded_pixtab += 1
                # Exclude the one which have not been found
            nexcluded_pixtab += len(pixtab_to_exclude)
            for pixtab in pixtab_to_exclude:
                self.dic_pixtabs_in_pointings[pointing].remove(pixtab)
                if self.verbose:
                    upipe.print_warning("PIXTABLE [not found in OffsetTable]: "
                                        "{0}".format(pixtab))

        # printing result
        upipe.print_info("Offset Table checked: #{0} PixTables included".format(
                          nincluded_pixtab))
        if nexcluded_pixtab == 0:
            upipe.print_info("All PixTables were found in Offset Table")
        else:
            upipe.print_warning("#{0} PixTables not found in Offset Table".format(
                                 nexcluded_pixtab))

    def goto_origfolder(self, addtolog=False) :
        """Go back to original folder
        """
        upipe.print_info("Going back to the original folder {0}".format(self.paths.orig))
        self.goto_folder(self.paths.orig, addtolog=addtolog, verbose=False)
            
    def goto_prevfolder(self, addtolog=False) :
        """Go back to previous folder
        """
        upipe.print_info("Going back to the previous folder {0}".format(self.paths._prev_folder))
        self.goto_folder(self.paths._prev_folder, addtolog=addtolog, verbose=False)
            
    def goto_folder(self, newpath, addtolog=False, verbose=True) :
        """Changing directory and keeping memory of the old working one
        """
        try: 
            prev_folder = os.getcwd()
            newpath = os.path.normpath(newpath)
            os.chdir(newpath)
            if verbose :
                upipe.print_info("Going to folder {0}".format(newpath))
            if addtolog :
                upipe.append_file(self.paths.logfile, "cd {0}\n".format(newpath))
            self.paths._prev_folder = prev_folder 
        except OSError:
            if not os.path.isdir(newpath):
                raise
    
    def set_fullpath_names(self) :
        """Create full path names to be used
        """
        # initialisation of the full paths 
        self.paths = musepipe.PipeObject("All Paths useful for the pipeline")
        self.paths.root = self.pipe_params.root
        self.paths.data = joinpath(self.paths.root, self.pipe_params.data)
        self.paths.target = joinpath(self.paths.root, self.targetname)

        self._dic_paths = {"combined": self.paths}

        for name in list(self._dic_combined_folders.keys()):
            setattr(self.paths, name, joinpath(self.paths.data, getattr(self.pipe_params, name)))

        # Creating the filenames for Master files
        self.dic_name_pointings = {}
        for pointing in self.list_pointings:
            name_pointing = "P{0:02d}".format(np.int(pointing))
            self.dic_name_pointings[pointing] = name_pointing
            # Adding the path of the folder
            setattr(self.paths, name_pointing,
                    joinpath(self.paths.root, "{0}/P{1:02d}/".format(self.targetname, pointing)))

        # Creating the attributes for the folders needed in the TARGET root folder, e.g., for alignments
        for name in list(self.pipe_params._dic_folders_target.keys()):
            setattr(self.paths, name, joinpath(self.paths.target, self.pipe_params._dic_folders_target[name]))

    def run_combine_all_single_pointings(self,
            add_suffix="",
            sof_filename='pointings_combine',
            lambdaminmax=[4000.,10000.], 
            **kwargs):
        """Run for all pointings individually, provided in the 
        list of pointings

        Input
        -----
        list_pointings: list of int
            By default to None (using the default self.list_pointings).
            Otherwise a list of pointings you wish to conduct
            a combine but for each individual pointing.
        add_suffix: str
            Additional suffix. 'PXX' where XX is the pointing number
            will be automatically added to that add_suffix for 
            each individual pointing.
        sof_filename: str
            Name (suffix only) of the sof file for this combine.
            By default, it is set to 'pointings_combine'.
        lambdaminmax: list of 2 floats [in Angstroems]
            Minimum and maximum lambda values to consider for the combine.
            Default is 4000 and 10000 for the lower and upper limits, resp.
        """
        # If list_pointings is None using the initially set up one
        list_pointings = kwargs.pop("list_pointings", self.list_pointings)

        # Additional suffix if needed
        for pointing in list_pointings:
            self.run_combine_single_pointing(pointing, add_suffix=add_suffix,
                    sof_filename=sof_filename, lambdaminmax=lambdaminmax,
                    **kwargs)

    def run_combine_single_pointing(self, pointing, add_suffix="", 
            sof_filename='pointings_combine',
            lambdaminmax=[4000.,10000.], 
            **kwargs):
        """Running the combine routine on just one single pointing

        Input
        =====
        pointing: int
            Pointing number. No default: must be provided.
        add_suffix: str
            Additional suffix. 'PXX' where XX is the pointing number
            will be automatically added to that add_suffix.
        sof_filename: str
            Name (suffix only) of the sof file for this combine.
            By default, it is set to 'pointings_combine'.
        lambdaminmax: list of 2 floats [in Angstroems]
            Minimum and maximum lambda values to consider for the combine.
            Default is 4000 and 10000 for the lower and upper limits, resp.
        """

        # getting the suffix with the additional PXX
        suffix = "{0}_P{1:02d}".format(add_suffix, np.int(pointing))

        # Running the combine for that single pointing
        self.run_combine(list_pointings=[np.int(pointing)], suffix=suffix, 
                        lambdaminmax=lambdaminmax,
                        sof_filename=sof_filename, **kwargs)

    def run_combine(self, sof_filename='pointings_combine', 
            lambdaminmax=[4000.,10000.], 
            suffix="", **kwargs):
        """MUSE Exp_combine treatment of the reduced pixtables
        Will run the esorex muse_exp_combine routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        lambdaminmax: list of 2 floats
            Minimum and maximum lambda values to consider for the combine
        suffix: str
            Suffix to be used for the output name
        """
        # Lambda min and max?
        [lambdamin, lambdamax] = lambdaminmax
        # Save options
        save = kwargs.pop("save", "cube,combined")
        # Filters
        filter_list = kwargs.pop("filter_list", self.filter_list)
        expotype = kwargs.pop("expotype", 'REDUCED')

        if "offset_table_name" in kwargs:
            offset_table_name = kwargs.pop("offset_table_name", None)
            folder_offset_table = kwargs.pop("offset_table_name", self.folder_offset_table)
            self._check_offset_table(offset_table_name, folder_offset_table)

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # If list_pointings is None using the initially set up one
        list_pointings = kwargs.pop("list_pointings", self.list_pointings)

        # Abort if only one exposure is available
        # exp_combine needs a minimum of 2
        nexpo_tocombine = sum(len(self.dic_pixtabs_in_pointings[pointing]) 
                              for pointing in list_pointings)
        if nexpo_tocombine <= 1:
            if self.verbose:
                upipe.print_warning("All considered pointings only "
                                    "have one exposure: process aborted", 
                                    pipe=self)
            return

        # Now creating the SOF file, first reseting it
        self._sofdict.clear()
        # Selecting only exposures to be treated
        # Producing the list of REDUCED PIXTABLES
        self._add_calib_to_sofdict("FILTER_LIST")

        # Adding a WCS if needed
        ref_wcs = kwargs.pop("ref_wcs", None)
        folder_ref_wcs = kwargs.pop("folder_ref_wcs", None)
        if ref_wcs is not None:
            if folder_ref_wcs is None:
                folder_ref_wcs = upipe.normpath(self.paths.cubes)
            self._sofdict['OUTPUT_WCS'] = [joinpath(folder_ref_wcs, ref_wcs)]

        # Setting the default option of offset_list
        if self.offset_table_name is not None:
            self._sofdict['OFFSET_LIST'] = [joinpath(self.folder_offset_table, 
                                                     self.offset_table_name)]

        pixtable_name = musepipe.dic_listObject[expotype]
        self._sofdict[pixtable_name] = []
        for pointing in list_pointings:
            self._sofdict[pixtable_name] += self.dic_pixtabs_in_pointings[pointing]

        self.write_sof(sof_filename="{0}_{1}{2}".format(sof_filename, self.targetname, suffix), new=True)

        # Product names
        dir_products = upipe.normpath(self.paths.cubes)
        name_products, suffix_products, suffix_prefinalnames = \
                prep_recipes_pipe._get_combine_products(filter_list) 

        # Combine the exposures 
        self.recipe_combine_pointings(self.current_sof, dir_products, name_products, 
                suffix_products=suffix_products,
                suffix_prefinalnames=suffix_prefinalnames,
                save=save, suffix=suffix, filter_list=filter_list, 
                lambdamin=lambdamin, lambdamax=lambdamax,
                **kwargs)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

## Useful function to change some pixtables
def rotate_pixtables(folder="", name_suffix="", list_ifu=None, 
                     angle=0., **kwargs):
    """Will update the derotator angle in each of the 24 pixtables
    Using a loop on rotate_pixtable

    Will thus update the HIERARCH ESO INS DROT POSANG keyword.

    Input
    -----
    folder: str
        name of the folder where the PIXTABLE are
    name_suffix: str
        name of the suffix to be used on top of PIXTABLE_OBJECT
    list_ifu: list[int]
        List of Pixtable numbers. If None, will do all 24
    angle: float
        Angle to rotate (in degrees)
    """

    if list_ifu is None:
        list_ifu = np.arange(24) + 1

    for nifu in list_ifu:
        rotate_pixtable(folder=folder, name_suffix=name_suffix, nifu=nifu, 
                        angle=angle, **kwargs)

def rotate_pixtable(folder="", name_suffix="", nifu=1, angle=0., **kwargs):
    """Rotate a single IFU PIXTABLE_OBJECT
    Will thus update the HIERARCH ESO INS DROT POSANG keyword.

    Input
    -----
    folder: str
        name of the folder where the PIXTABLE are
    name_suffix: str
        name of the suffix to be used on top of PIXTABLE_OBJECT
    nifu: int
        Pixtable number. Default is 1
    angle: float
        Angle to rotate (in degrees)
    """
    angle_keyword = "HIERARCH ESO INS DROT POSANG"
    angle_orig_keyword = "{0} ORIG".format(angle_keyword)

    pixtable_basename = kwargs.pop("pixtable_basename", 
                               musepipe.dic_listObject['OBJECT'])
    name_pixtable = "{0}_{1}-{2:02d}.fits".format(pixtable_basename, 
                                                 name_suffix, np.int(nifu))
    fullname_pixtable = joinpath(folder, name_pixtable)

    if os.path.isfile(fullname_pixtable):
        mypix = pyfits.open(fullname_pixtable, mode='update')
        if not angle_orig_keyword in mypix[0].header:
            mypix[0].header[angle_orig_keyword] = mypix[0].header[angle_keyword]
        mypix[0].header[angle_keyword] = mypix[0].header[angle_orig_keyword] + angle
        upipe.print_info("Updating INS DROT POSANG for {0}".format(name_pixtable))
        mypix.flush()
    else:
        upipe.print_error("Input Pixtable {0} does not exist - Aborting".format(
                              fullname_pixtable))
