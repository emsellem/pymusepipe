# Licensed under a MIT license - see LICENSE

"""MUSE-PHANGS combine module
"""

__authors__ = "Eric Emsellem"
__copyright__ = "(c) 2017-2019, ESO + CRAL"
__license__ = "MIT License"
__contact__ = " <eric.emsellem@eso.org>"

# Importing modules
import numpy as np
import os
from os.path import join as joinpath
import glob
import copy
import re

try:
    import astropy as apy
    from astropy.io import fits as pyfits
except ImportError:
    raise Exception("astropy is required for this module")

from astropy.utils.exceptions import AstropyWarning

try:
    from astropy.table import Table
except ImportError:
    raise Exception("astropy.table.Table is required for this module")

import warnings

# Importing pymusepipe modules
from pymusepipe.recipes_pipe import PipeRecipes
from pymusepipe.create_sof import SofPipe
from pymusepipe.init_musepipe import InitMuseParameters
import pymusepipe.util_pipe as upipe
from pymusepipe import musepipe, prep_recipes_pipe
from pymusepipe.config_pipe import default_filter_list, default_PHANGS_filter_list, dic_combined_folders
from pymusepipe.config_pipe import default_prefix_wcs, default_prefix_mask, default_prefix_wcs_mosaic
from pymusepipe.config_pipe import dic_listObject, lambdaminmax_for_wcs, lambdaminmax_for_mosaic
from pymusepipe.mpdaf_pipe import MuseCube

# Default keywords for MJD and DATE
from pymusepipe.align_pipe import mjd_names, date_names

__version__ = '0.0.3 (4 Sep 2019)'
# 0.0.2 28 Feb, 2019: trying to make it work
# 0.0.1 21 Nov, 2017 : just setting up the scene

def get_list_periods(root_path=""):
    """Getting a list of existing periods
    for a given path

    Input
    -----
    path: str

    Return
    ------
    list_targets: list of str
    """
    # Done by scanning the target path
    list_folders = glob.glob(root_path + "/P???")
    list_periods = []
    for folder in list_folders:
        lint = re.findall(r'(\d{3})', folder)
        if len(lint) > 0:
            list_periods.append(np.int(lint[-1]))

    list_periods.sort()
    upipe.print_info("Periods list: {0}".format(str(list_periods)))
    return list_periods


def get_list_targets(period_path=""):
    """Getting a list of existing periods
    for a given path

    Input
    -----
    period_path: str

    Return
    ------
    list_targets: list of str
    """
    # Done by scanning the target path
    list_targets = [name for name in os.listdir(period_path)
                    if os.path.isdir(os.path.join(period_path, name))]

    list_targets.sort()
    upipe.print_info("Potential Targets -- list: {0}".format(str(list_targets)))
    return list_targets

def build_dic_exposures(target_path=""):
    """

    Parameters
    ----------
    target_path

    Returns
    -------
    dic_expo: dict
        Dictionary of exposures in each pointing

    """
    list_pointings = get_list_pointings(target_path)
    dic_expos = {}
    for pointing in list_pointings:
        name_pointing = "P{:02d}".format(pointing)
        upipe.print_info("For pointing {0}".format(name_pointing))
        dic_p = get_list_exposures(joinpath(target_path, name_pointing))
        dic_expos[pointing] = [(tpl, dic_p[tpl]) for tpl in dic_p]

    return dic_expos

def get_list_pointings(target_path=""):
    """Getting a list of existing pointings
    for a given path

    Input
    -----
    target_path: str

    Return
    ------
    list_pointings: list of int
    """
    # Done by scanning the target path
    list_folders = glob.glob(target_path + "/P??")
    list_pointings = []
    for folder in list_folders:
        list_pointings.append(np.int(folder[-2:]))

    list_pointings.sort()
    upipe.print_info("Pointings list: {0}".format(str(list_pointings)))
    return list_pointings


def get_list_exposures(pointing_path=""):
    """Getting a list of exposures from a given path

    Input
    -----
    pointing_path: str

    Return
    ------
    list_expos: list of int
    """
    # Done by scanning the target path
    list_files = glob.glob(pointing_path + "/Object/DATACUBE_FINAL*_????.fits")
    list_expos = []
    for name in list_files:
        [(tpl, lint)] = re.findall(r'\_(\S{19})\_(\d{4}).fits', name)
        if len(lint) > 0:
            list_expos.append((tpl, np.int(lint)))

    # Making it unique and sort
    list_expos = np.unique(list_expos, axis=0)
    # Sorting by tpl and expo number
    sorted_list = sorted(list_expos, key=lambda e: (e[0], e[1]))

    # Building the final list
    dic_expos = {}
    for l in sorted_list:
        tpl = l[0]
        if tpl in dic_expos:
            dic_expos[tpl].append(l[1])
        else:
            dic_expos[tpl] = [l[1]]

    # Finding the full list of tpl
    upipe.print_info("Exposures list:")
    for tpl in dic_expos:
        upipe.print_info("TPL= {0} : Exposures= {1}".format(tpl, dic_expos[tpl]))

    return dic_expos

def get_pixtable_list(target_path="", list_pointings=None, suffix=""):
    """Provide a list of reduced pixtables

    Input
    -----
    target_path: str
        Path for the target folder
    list_pointings: list of int
        List of integers, providing the list of pointings to consider
    suffix: str
        Additional suffix, if needed, for the names of the PixTables.
    """
    # Getting the pieces of the names to be used for pixtabs
    pixtable_suffix = prep_recipes_pipe.dic_products_scipost['individual'][0]

    # Initialise the dictionary of pixtabs to be found in each pointing
    dic_pixtables = {}

    # Defining the pointing list if not provided
    # Done by scanning the target path
    if list_pointings is None:
        list_pointings = get_list_pointings(target_path)

    # Looping over the pointings
    for pointing in list_pointings:
        # get the path of the pointing
        path_pointing = joinpath(target_path, "P{0:02d}".format(np.int(pointing)))
        # List existing pixtabs, using the given suffix
        list_pixtabs = glob.glob(path_pointing + "/Object/" +
                                 "{0}{1}*fits".format(pixtable_suffix, suffix))

        # Reset the needed temporary dictionary
        dic_tpl = {}
        # Loop over the pixtables for that pointing
        for pixtab in list_pixtabs:
            # Split over the PIXTABLE_REDUCED string
            sl = pixtab.split(pixtable_suffix + "_")
            # Find the expo number
            expo = sl[1][-9:-5]
            # Find the tpl
            tpl = sl[1].split("_" + expo)[0]
            # If not already there, add it
            if tpl not in dic_tpl:
                dic_tpl[tpl] = [np.int(expo)]
            # if already accounted for, add the expo number
            else:
                dic_tpl[tpl].append(np.int(expo))

        # Creating the full list for that pointing
        full_list = []
        for tpl in dic_tpl:
            dic_tpl[tpl].sort()
            full_list.append((tpl, dic_tpl[tpl]))

        # And now filling in the dictionary for that pointing
        dic_pixtables[pointing] = full_list

    return dic_pixtables


class MusePointings(SofPipe, PipeRecipes):
    def __init__(self, targetname=None, list_pointings="all",
                 dic_exposures_in_pointings=None,
                 suffix_fixed_pixtables="tmask",
                 folder_config="",
                 rc_filename=None, cal_filename=None,
                 combined_folder_name="Combined", suffix="",
                 offset_table_name=None,
                 log_filename="MusePipeCombine.log",
                 verbose=True, debug=False, **kwargs):
        """Initialisation of class muse_expo

        Input
        -----
        targetname: string (e.g., 'NGC1208'). default is None. 

        rc_filename: str
            filename to initialise folders
        cal_filename: str
            filename to initial FIXED calibration MUSE files
        verbose: bool 
            Give more information as output (default is True)
        debug: bool
            Allows to get more messages when needed
            Default is False
        vsystemic: float 
            Default is 0. Systemic velocity of the galaxy [in km/s]
        suffix_fixed_pixtables: str
            Suffix for fixed PixTables. Default is ''.
        use_fixed_pixtables: bool
            Default is False. If True, will use suffix_fixed_pixtables to filter out
            Pixtables which have been fixed.

        Other possible entries
        ----------------------
        warnings: strong  ('ignore'by default. If set to ignore, will ignore the Astropy Warnings.

        """
        # Verbose option
        self.verbose = verbose
        self._debug = debug
        if self._debug:
            upipe.print_warning("In DEBUG Mode [more printing]")

        # Warnings for astropy
        self.warnings = kwargs.pop("warnings", 'ignore')
        if self.warnings == 'ignore':
            warnings.simplefilter('ignore', category=AstropyWarning)

        # Setting the default attibutes #####################
        self.targetname = targetname
        self.__phangs = kwargs.pop("PHANGS", False)
        if self.__phangs:
            self.filter_list = kwargs.pop("filter_list", default_PHANGS_filter_list)
        else:
            self.filter_list = kwargs.pop("filter_list", default_filter_list)

        self.combined_folder_name = combined_folder_name
        self.vsystemic = np.float(kwargs.pop("vsystemic", 0.))

        # Including or not the fixed Pixtables in place of the original ones
        self.use_fixed_pixtables = kwargs.pop("use_fixed_pixtables", False)
        self.suffix_fixed_pixtables = suffix_fixed_pixtables

        # Setting other default attributes
        if log_filename is None:
            log_filename = "log_{timestamp}.txt".format(timestamp=upipe.create_time_name())
            upipe.print_info("The Log file will be {0}".format(log_filename))
        self.log_filename = log_filename
        self.suffix = suffix
        folder_offset_table = kwargs.pop("folder_offset_table", None)

        # End of parameter settings #########################

        # Init of the subclasses
        PipeRecipes.__init__(self, **kwargs)
        SofPipe.__init__(self)

        # ---------------------------------------------------------
        # Setting up the folders and names for the data reduction
        # Can be initialised by either an rc_file, 
        # or a default rc_file or harcoded defaults.
        self.pipe_params = InitMuseParameters(folder_config=folder_config,
                                              rc_filename=rc_filename,
                                              cal_filename=cal_filename,
                                              verbose=verbose)

        # Setting up the relative path for the data, using Galaxy Name + Pointing
        self.pipe_params.data = "{0}/{1}/".format(self.targetname,
                                                  self.combined_folder_name)

        self.pipe_params.init_default_param(dic_combined_folders)
        self._dic_combined_folders = dic_combined_folders

        # Now the list of pointings
        if isinstance(list_pointings, str):
            if list_pointings.lower() == "all":
                self.list_pointings = get_list_pointings(
                    joinpath(self.pipe_params.root, self.targetname))
        else:
            self.list_pointings = list_pointings

        # Setting all the useful paths
        self.set_fullpath_names()
        self.paths.log_filename = joinpath(self.paths.log, log_filename)

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
        for folder in self._dic_combined_folders:
            upipe.safely_create_folder(self._dic_combined_folders[folder], verbose=verbose)

        # Checking input pointings and pixtables
        self._check_pointings(dic_exposures_in_pointings)

        # Checking input offset table and corresponding pixtables
        self._check_offset_table(offset_table_name, folder_offset_table)
        # END CHECK UP ============================================

        # Making the output folders in a safe mode
        if self.verbose:
            upipe.print_info("Creating directory structure")

        # Going back to initial working directory
        self.goto_origfolder()

    def _add_targetname(self, name):
        """Add targetname to input name and return it
        
        Input
        -----
        name: str

        Returns
        -------
        new name including targetname
        """
        if self.add_targetname:
            return "{0}_{1}".format(self.targetname, name)
        else:
            return name

    def run_combine_all_single_pointings_withmasks(self, combine=True, masks=True, 
            mosaic_wcs=True, perpointing_combine=True, **kwargs):
        """Run all combine recipes including WCS and masks

        combine: bool [True]
            Default is True. Will run the combine for all pointings.

        masks: bool [True]
            Will run the combined WCS and the individual pointing ones
            (and masks).

        mosaic_wcs (bool): [True]. Reference WCS for the full mosaic
            
        perpointing_combine: bool [True]
            Will run individual pointings using the WCS.
        """
        lambdaminmax = kwargs.pop("lambdaminmax", lambdaminmax_for_mosaic)
        if combine:
            upipe.print_info("Running the mosaic combine")
            offset_table_name = kwargs.get("offset_table_name", None)
            folder_offset_table = kwargs.get("folder_offset_table",
                                             self.folder_offset_table)
            if offset_table_name is not None:
                self._check_offset_table(offset_table_name, folder_offset_table)
            self.run_combine(lambdaminmax=lambdaminmax,
                             offset_table_name=offset_table_name,
                             folder_offset_table=folder_offset_table)

        if masks:
            # Creating the full mosaic WCS first with a narrow lambda range
            upipe.print_info("Start creating the narrow-lambda WCS and Masks")
            _ = self.create_combined_wcs()
            # Then creating the mask WCS for each pointing
            upipe.print_info("Start creating the individual Pointings Masks")
            self.create_all_pointings_mask_wcs(lambdaminmax_mosaic=lambdaminmax, 
                                               **kwargs)

        if mosaic_wcs:
            # Creating a reference WCS for the Full Mosaic with the right 
            # Spectral coverage for a full mosaic
            upipe.print_info("Start creating the full-lambda WCS")
            self._combined_wcs_name = self.create_combined_wcs(
                prefix_wcs=default_prefix_wcs_mosaic,
                lambdaminmax_wcs=lambdaminmax_for_mosaic)

        if perpointing_combine:
            upipe.print_info("Running the Individual Pointing combine")
            # Then merging each single pointing using the masks
            self.run_combine_all_single_pointings(**kwargs)

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
                        upipe.print_warning("Fixed PixTable {0} was included in the list".format(
                            fixed_pixtab))
                        upipe.print_warning("and Pixtable {0} was thus removed from the list".format(
                            orig_pixtab))
                    else:
                        upipe.print_warning("Original Pixtable {0} not found".format(
                            orig_pixtab))
                        upipe.print_warning("Hence will not include fixed PixTable in the list"
                                            "{0}".format(fixed_pixtab))

            full_list = copy.copy(list_pixtabs)
            full_list.sort()
            self.dic_allpixtabs_in_pointings[pointing] = full_list

            # if no selection on exposure names are given
            # Select all existing pixtabs
            if self.dic_exposures_in_pointings is None:
                select_list_pixtabs = list_pixtabs

            # Otherwise use the ones which are given via their expo numbers
            else:
                select_list_pixtabs = []
                # this is the list of exposures to consider
                if pointing not in self.dic_exposures_in_pointings:
                    upipe.print_warning("Pointing {} not in dictionary - skipping".format(pointing))
                else:
                    list_expo = self.dic_exposures_in_pointings[pointing]
                    # We loop on that list
                    for expotuple in list_expo:
                        tpl, nexpo = expotuple[0], expotuple[1]
                        for expo in nexpo:
                            # Check whether this exists in the our cube list
                            suffix_expo = "_{0:04d}".format(np.int(expo))
                            if self._debug:
                                upipe.print_debug("Checking which exposures are tested")
                                upipe.print_debug(suffix_expo)
                            for pixtab in list_pixtabs:
                                if (suffix_expo in pixtab) and (tpl in pixtab):
                                    # We select the cube
                                    select_list_pixtabs.append(pixtab)
                                    # And remove it from the list
                                    list_pixtabs.remove(pixtab)
                                    # We break out of the cube for loop
                                    break

            select_list_pixtabs.sort()
            self.dic_pixtabs_in_pointings[pointing] = select_list_pixtabs

    def _read_offset_table(self, offset_table_name=None, folder_offset_table=None):
        """Reading the Offset Table
        If readable, the table is read and set in the offset_table attribute.

        Input
        -----
        offset_table_name: str
            Name of the offset table
            Default is None
        folder_offset_table: str
            Name of the folder to find the offset table
            Default is None
        """
        self.offset_table_name = offset_table_name
        if self.offset_table_name is None:
            upipe.print_warning("No Offset table name given", pipe=self)
            self.offset_table = Table()
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
                full_offset_table_name), pipe=self)
            self.offset_table = Table()
            return

        # Opening the offset table
        self.offset_table = Table.read(full_offset_table_name)

    def _check_offset_table(self, offset_table_name=None, folder_offset_table=None):
        """Checking if DATE-OBS and MJD-OBS are in the OFFSET Table

        Input
        -----
        offset_table_name: str
            Name of the offset table
            Default is None
        folder_offset_table: str
            Name of the folder to find the offset table
            Default is None
        """
        if offset_table_name is None:
            return

        self._read_offset_table(offset_table_name=offset_table_name,
                                folder_offset_table=folder_offset_table)

        # getting the MJD and DATE from the OFFSET table
        if not set(self.offset_table.columns.keys()) \
            & {mjd_names['table'], date_names['table']}:
            upipe.print_warning("Could not find some keywords "
                                "in offset table")
            return

        self.table_mjdobs = self.offset_table[mjd_names['table']]
        self.table_dateobs = self.offset_table[date_names['table']]

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

    def goto_origfolder(self, addtolog=False):
        """Go back to original folder
        """
        upipe.print_info("Going back to the original folder {0}".format(self.paths.orig),
                             pipe=self)
        self.goto_folder(self.paths.orig, addtolog=addtolog, verbose=False)

    def goto_prevfolder(self, addtolog=False):
        """Go back to previous folder
        """
        upipe.print_info("Going back to the previous folder {0}".format(self.paths._prev_folder),
                             pipe=self)
        self.goto_folder(self.paths._prev_folder, addtolog=addtolog, verbose=False)

    def goto_folder(self, newpath, addtolog=False, verbose=True):
        """Changing directory and keeping memory of the old working one
        """
        try:
            prev_folder = os.getcwd()
            newpath = os.path.normpath(newpath)
            os.chdir(newpath)
            upipe.print_info("Going to folder {0}".format(newpath), pipe=self)
            if addtolog:
                upipe.append_file(self.paths.log_filename, "cd {0}\n".format(newpath))
            self.paths._prev_folder = prev_folder
        except OSError:
            if not os.path.isdir(newpath):
                raise

    def set_fullpath_names(self):
        """Create full path names to be used
        That includes: root, data, target, but also _dic_paths, paths
        """
        # initialisation of the full paths 
        self.paths = musepipe.PipeObject("All Paths useful for the pipeline")
        self.paths.root = self.pipe_params.root
        self.paths.data = joinpath(self.paths.root, self.pipe_params.data)
        self.paths.target = joinpath(self.paths.root, self.targetname)

        self._dic_paths = {"combined": self.paths}

        for name in self._dic_combined_folders:
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
        for name in self.pipe_params._dic_folders_target:
            setattr(self.paths, name, joinpath(self.paths.target, self.pipe_params._dic_folders_target[name]))

    def run_combine_all_single_pointings(self,
                                         add_suffix="",
                                         sof_filename='pointings_combine',
                                         **kwargs):
        """Run for all pointings individually, provided in the 
        list of pointings, by just looping over the pointings.

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
            upipe.print_info("Combining single pointings - Pointing {0:02d}".format(
                             pointing))
            self.run_combine_single_pointing(pointing, add_suffix=add_suffix,
                                             sof_filename=sof_filename,
                                             **kwargs)

    def run_combine_single_pointing(self, pointing, add_suffix="",
                                    sof_filename='pointing_combine',
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
        wcs_from_mosaic: bool
            True by default, meaning that the WCS of the mosaic will be used.
            If not there, will ignore it.
        """

        # getting the suffix with the additional PXX
        suffix = "{0}_P{1:02d}".format(add_suffix, np.int(pointing))

        ref_wcs = kwargs.pop("ref_wcs", None)
        wcs_from_mosaic = kwargs.pop("wcs_from_mosaic", True)

        # Wcs_from_mosaic
        # If true, use the reference wcs
        if wcs_from_mosaic:
            if ref_wcs is not None:
                upipe.print_warning("wcs_from_mosaic is set to True. "
                                    "Hence will overwrite ref_wcs given input")
            prefix_wcs = kwargs.pop("prefix_wcs", default_prefix_wcs)
            self.add_targetname = kwargs.pop("add_targetname", True)
            prefix_wcs = self._add_targetname(prefix_wcs)

            ref_wcs = "{0}P{1:02d}.fits".format(prefix_wcs,
                                                np.int(pointing))

        # Running the combine for that single pointing
        self.run_combine(list_pointings=[np.int(pointing)], suffix=suffix,
                         sof_filename=sof_filename,
                         ref_wcs=ref_wcs,
                         **kwargs)

    def create_all_pointings_mask_wcs(self, filter_list="white", **kwargs):
        """Create all pointing masks one by one
        as well as the wcs for each individual pointings. Using the grid
        from the global WCS of the mosaic but restricting it to the 
        range of non-NaN.

        Input
        -----
        filter_list = list of str
            List of filter names to be used. 
        """
        # If list_pointings is None using the initially set up one
        list_pointings = kwargs.pop("list_pointings", self.list_pointings)

        # Additional suffix if needed
        for pointing in list_pointings:
            upipe.print_info("Making WCS Mask for pointing {0:02d}".format(pointing))
            _ = self.create_pointing_mask_wcs(pointing=pointing,
                                          filter_list=filter_list,
                                          **kwargs)

    def create_pointing_mask_wcs(self, pointing, 
            lambdaminmax_mosaic=lambdaminmax_for_mosaic,
            filter_list="white", **kwargs):
        """Create the mask of a given pointing
        And also a WCS file which can then be used to compute individual pointings
        with a fixed WCS.

        Input
        -----
        pointing: int
            Number of the pointing
        filter_list = list of str
            List of filter names to be used.

        Creates:
            Pointing mask WCS cube

        Returns:
            Name of the created WCS cube
        """

        # Adding target name as prefix or not
        self.add_targetname = kwargs.pop("add_targetname", True)
        prefix_mask = kwargs.pop("prefix_mask", default_prefix_mask)
        prefix_wcs = kwargs.pop("prefix_wcs", default_prefix_wcs)

        # Running combine with the ref WCS with only 2 spectral pixels
        # Limit the maximum lambda to the wcs ones
        self.run_combine_single_pointing(pointing=pointing,
                                         filter_list=filter_list,
                                         sof_filename="pointing_mask",
                                         add_targetname=self.add_targetname,
                                         prefix_all=prefix_mask,
                                         lambdaminmax=lambdaminmax_for_wcs,
                                         wcs_auto=True, **kwargs)

        # Now creating the mask with 0's and 1's
        dir_mask = upipe.normpath(self.paths.cubes)

        # Adding targetname for the final names
        prefix_mask = self._add_targetname(prefix_mask)
        prefix_wcs = self._add_targetname(prefix_wcs)

        name_mask = "{0}DATACUBE_FINAL_P{1:02d}.fits".format(
            prefix_mask, np.int(pointing))
        finalname_mask = "{0}P{1:02d}.fits".format(prefix_mask,
                                                   np.int(pointing))
        finalname_wcs = "{0}P{1:02d}.fits".format(prefix_wcs,
                                                  np.int(pointing))

        # First create a subcube without all the Nan
        mask_cube = MuseCube(filename=joinpath(dir_mask, name_mask))

        # Creating the new cube
        upipe.print_info("Now creating the Reference WCS cube "
                         "for pointing {0}".format(np.int(pointing)))
        cfolder, cname = mask_cube.create_reference_cube(
                lambdamin=lambdaminmax_mosaic[0],
                lambdamax=lambdaminmax_mosaic[1], 
                filter_for_nan=True, prefix=prefix_wcs, 
                outcube_name=finalname_wcs, **kwargs)

        # Now transforming this into a bona fide 1 extension WCS file
        full_cname = joinpath(cfolder, cname)
        d = pyfits.getdata(full_cname, ext=1)
        h = pyfits.getheader(full_cname, ext=1)
        hdu = pyfits.PrimaryHDU(data=d, header=h)
        hdu.writeto(full_cname, overwrite=True)
        upipe.print_info("...Done")
        return full_cname

    def extract_combined_narrow_wcs(self, name_cube=None, **kwargs):
        """Create the reference WCS from the full mosaic with
        only 2 lambdas

        Input
        -----
        name_cube: str
            Name of the cube. Can be None, and then the final
            datacube from the combine folder will be used.
        wave1: float - optional
            Wavelength taken for the extraction. Should only
            be present in all spaxels you wish to get.
        prefix_wcs: str - optional
            Prefix to be added to the name of the input cube.
            By default, will use "refwcs".
        add_targetname: bool [True]
            Add the name of the target to the name of the output
            WCS reference cube. Default is True.

        Creates:
            Combined narrow band WCS cube

        Returns:
            name of the created cube
        """
        # Adding targetname in names or not
        self.add_targetname = kwargs.pop("add_targetname", True)

        if name_cube is None:
            # getting the name of the final datacube (mosaic)
            cube_suffix = prep_recipes_pipe.dic_products_scipost['cube'][0]
            cube_suffix = self._add_targetname(cube_suffix)
            name_cube = joinpath(self.paths.cubes, cube_suffix + ".fits")

        # test if cube exists
        if not os.path.isfile(name_cube):
            upipe.print_error("[combine/extract_combined_narrow_wcs] File {0} "
                              "does not exist. Aborting.".format(name_cube))
            return

        # Opening the cube via MuseCube
        refcube = MuseCube(filename=name_cube)

        # Creating the new cube
        prefix_wcs = kwargs.pop("prefix_wcs", default_prefix_wcs)
        upipe.print_info("Now creating the Reference WCS cube using prefix '{0}'".format(
            prefix_wcs))
        cfolder, cname = refcube.extract_onespectral_cube(prefix=prefix_wcs, **kwargs)

        # Now transforming this into a bona fide 1 extension WCS file
        full_cname = joinpath(cfolder, cname)
        d = pyfits.getdata(full_cname, ext=1)
        h = pyfits.getheader(full_cname, ext=1)
        hdu = pyfits.PrimaryHDU(data=d, header=h)
        hdu.writeto(full_cname, overwrite=True)
        upipe.print_info("...Done")
        return full_cname

    def create_combined_wcs(self, name_cube=None, 
            lambdaminmax_wcs=lambdaminmax_for_wcs,
            **kwargs):
        """Create the reference WCS from the full mosaic
        with a given range of lambda.

        Input
        -----
        name_cube: str
            Name of the cube. Can be None, and then the final
            datacube from the combine folder will be used.
        wave1: float - optional
            Wavelength taken for the extraction. Should only
            be present in all spaxels you wish to get.
        prefix_wcs: str - optional
            Prefix to be added to the name of the input cube.
            By default, will use "refwcs".
        add_targetname: bool [True]
            Add the name of the target to the name of the output
            WCS reference cube. Default is True.
        """
        # Adding targetname in names or not
        self.add_targetname = kwargs.pop("add_targetname", True)

        if name_cube is None:
            # getting the name of the final datacube (mosaic)
            cube_suffix = prep_recipes_pipe.dic_products_scipost['cube'][0]
            cube_suffix = self._add_targetname(cube_suffix)
            name_cube = joinpath(self.paths.cubes, cube_suffix + ".fits")

        # test if cube exists
        if not os.path.isfile(name_cube):
            upipe.print_error("[combine/create_combined_wcs] File {0} does not exist "
                              "- Aborting.".format(name_cube))
            return

        # Opening the cube via MuseCube
        refcube = MuseCube(filename=name_cube)

        # Creating the new cube
        prefix_wcs = kwargs.pop("prefix_wcs", default_prefix_wcs)
        upipe.print_info("Now creating the Reference WCS cube using prefix '{0}'".format(
            prefix_wcs))
        cfolder, cname = refcube.create_reference_cube(lambdamin=lambdaminmax_wcs[0],
                lambdamax=lambdaminmax_wcs[1], prefix=prefix_wcs, **kwargs)

        # Now transforming this into a bona fide 1 extension WCS file
        combined_wcs_name = joinpath(cfolder, cname)
        d = pyfits.getdata(full_cname, ext=1)
        h = pyfits.getheader(full_cname, ext=1)
        hdu = pyfits.PrimaryHDU(data=d, header=h)
        hdu.writeto(combined_wcs_name, overwrite=True)
        upipe.print_info("...Done")
        return combined_wcs_name

    def run_combine(self, sof_filename='pointings_combine',
                    lambdaminmax=[4000., 10000.],
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

        # Expotype
        expotype = kwargs.pop("expotype", 'REDUCED')

        # Adding target name as prefix or not
        self.add_targetname = kwargs.pop("add_targetname", True)
        prefix_wcs = kwargs.pop("prefix_wcs", default_prefix_wcs)
        prefix_all = kwargs.pop("prefix_all", "")
        prefix_all = self._add_targetname(prefix_all)

        if "offset_table_name" in kwargs:
            offset_table_name = kwargs.pop("offset_table_name")
            folder_offset_table = kwargs.pop("folder_offset_table", self.folder_offset_table)
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
        wcs_auto = kwargs.pop("wcs_auto", False)
        ref_wcs = kwargs.pop("ref_wcs", None)
        if wcs_auto:
            upipe.print_warning("wcs_auto is True, hence overwriting ref_wcs name")
            # getting the name of the final datacube (mosaic)
            cube_suffix = prep_recipes_pipe.dic_products_scipost['cube'][0]
            cube_suffix = self._add_targetname(cube_suffix)
            ref_wcs = "{0}{1}.fits".format(prefix_wcs, cube_suffix)
            upipe.print_warning("ref_wcs used is {0}".format(ref_wcs))

        folder_ref_wcs = kwargs.pop("folder_ref_wcs", upipe.normpath(self.paths.cubes))
        if ref_wcs is not None:
            full_ref_wcs = joinpath(folder_ref_wcs, ref_wcs)
            if not os.path.isfile(full_ref_wcs):
                upipe.print_error("Reference WCS file {0} does not exist".format(
                    full_ref_wcs))
                upipe.print_error("Consider using the create_combined_wcs recipe"
                                  " if you wish to create pointing masks. Else"
                                  " just check that the WCS reference file exists.")
                return

            self._sofdict['OUTPUT_WCS'] = [joinpath(folder_ref_wcs, ref_wcs)]

        # Setting the default option of offset_list
        if self.offset_table_name is not None:
            self._sofdict['OFFSET_LIST'] = [joinpath(self.folder_offset_table,
                                                     self.offset_table_name)]

        pixtable_name = dic_listObject[expotype]
        self._sofdict[pixtable_name] = []
        for pointing in list_pointings:
            self._sofdict[pixtable_name] += self.dic_pixtabs_in_pointings[pointing]

        self.write_sof(sof_filename="{0}_{1}{2}".format(sof_filename,
                                                        self.targetname,
                                                        suffix), new=True)

        # Product names
        dir_products = upipe.normpath(self.paths.cubes)
        name_products, suffix_products, suffix_prefinalnames, prefix_products = \
            prep_recipes_pipe._get_combine_products(filter_list,
                                                    prefix_all=prefix_all)

        # Combine the exposures 
        self.recipe_combine_pointings(self.current_sof, dir_products, name_products,
                                      suffix_products=suffix_products,
                                      suffix_prefinalnames=suffix_prefinalnames,
                                      prefix_products=prefix_products,
                                      save=save, suffix=suffix, filter_list=filter_list,
                                      lambdamin=lambdamin, lambdamax=lambdamax)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)
