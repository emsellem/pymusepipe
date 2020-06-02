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
from .recipes_pipe import PipeRecipes
from .create_sof import SofPipe
from .init_musepipe import InitMuseParameters
from . import util_pipe as upipe
from .util_pipe import filter_list_with_pdict
from . import musepipe, prep_recipes_pipe
from .config_pipe import (default_filter_list, default_PHANGS_filter_list,
                          dict_combined_folders, default_prefix_wcs,
                          default_prefix_mask, prefix_mosaic, dict_listObject,
                          lambdaminmax_for_wcs, lambdaminmax_for_mosaic)
from .mpdaf_pipe import MuseCube

# Default keywords for MJD and DATE
from .align_pipe import mjd_names, date_names
prefix_final_cube = prep_recipes_pipe.dict_products_scipost['cube'][0]

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

def build_dict_exposures(target_path=""):
    """

    Parameters
    ----------
    target_path

    Returns
    -------
    dict_expo: dict
        Dictionary of exposures in each pointing

    """
    list_pointings = get_list_pointings(target_path)
    dict_expos = {}
    for pointing in list_pointings:
        name_pointing = "P{:02d}".format(pointing)
        upipe.print_info("For pointing {0}".format(name_pointing))
        dict_p = get_list_exposures(joinpath(target_path, name_pointing))
        dict_expos[pointing] = [(tpl, dict_p[tpl]) for tpl in dict_p]

    return dict_expos

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
    list_files = glob.glob(pointing_path + "/Object/{}*_????.fits".format(
                              prefix_final_cube))
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
    dict_expos = {}
    for l in sorted_list:
        tpl = l[0]
        if tpl in dict_expos:
            dict_expos[tpl].append(l[1])
        else:
            dict_expos[tpl] = [l[1]]

    # Finding the full list of tpl
    upipe.print_info("Exposures list:")
    for tpl in dict_expos:
        upipe.print_info("TPL= {0} : Exposures= {1}".format(tpl, dict_expos[tpl]))

    return dict_expos

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
    pixtable_suffix = prep_recipes_pipe.dict_products_scipost['individual'][0]

    # Initialise the dictionary of pixtabs to be found in each pointing
    dict_pixtables = {}

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
        dict_tpl = {}
        # Loop over the pixtables for that pointing
        for pixtab in list_pixtabs:
            # Split over the PIXTABLE_REDUCED string
            sl = pixtab.split(pixtable_suffix + "_")
            # Find the expo number
            expo = sl[1][-9:-5]
            # Find the tpl
            tpl = sl[1].split("_" + expo)[0]
            # If not already there, add it
            if tpl not in dict_tpl:
                dict_tpl[tpl] = [np.int(expo)]
            # if already accounted for, add the expo number
            else:
                dict_tpl[tpl].append(np.int(expo))

        # Creating the full list for that pointing
        full_list = []
        for tpl in dict_tpl:
            dict_tpl[tpl].sort()
            full_list.append((tpl, dict_tpl[tpl]))

        # And now filling in the dictionary for that pointing
        dict_pixtables[pointing] = full_list

    return dict_pixtables

class MusePointings(SofPipe, PipeRecipes):
    def __init__(self, targetname=None, list_pointings=None,
                 dict_exposures=None,
                 prefix_fixed_pixtables="tmask",
                 folder_config="",
                 rc_filename=None, cal_filename=None,
                 combined_folder_name="Combined", suffix="",
                 name_offset_name=None,
                 folder_offset_table=None,
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
        prefix_fixed_pixtables: str
            Suffix for fixed PixTables. Default is ''.
        use_fixed_pixtables: bool
            Default is False. If True, will use prefix_fixed_pixtables to filter out
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
            self.filter_list = kwargs.pop("filter_list",
                                          default_PHANGS_filter_list)
        else:
            self.filter_list = kwargs.pop("filter_list",
                                          default_filter_list)

        self.combined_folder_name = combined_folder_name
        self.vsystemic = np.float(kwargs.pop("vsystemic", 0.))

        # Including or not the fixed Pixtables in place of the original ones
        self.use_fixed_pixtables = kwargs.pop("use_fixed_pixtables", False)
        self.prefix_fixed_pixtables = prefix_fixed_pixtables

        # Setting other default attributes
        if log_filename is None:
            log_filename = "log_{timestamp}.txt".format(timestamp=upipe.create_time_name())
            upipe.print_info("The Log file will be {0}".format(log_filename))
        self.log_filename = log_filename
        self.suffix = suffix
        self.add_targetname = kwargs.pop("add_targetname", True)

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

        self.pipe_params.init_default_param(dict_combined_folders)
        self._dict_combined_folders = dict_combined_folders

        self.list_pointings = self._check_pointings_list(list_pointings)
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
        for folder in self._dict_combined_folders:
            upipe.safely_create_folder(self._dict_combined_folders[folder], verbose=verbose)

        # Checking input pointings and pixtables
        self._pixtab_in_comb_folder = kwargs.pop("pixtab_in_comb_folder", True)
        self._get_pixtable_list(dict_exposures)

        # Checking input offset table and corresponding pixtables
        self._check_offset_table(name_offset_name, folder_offset_table)
        # END CHECK UP ============================================

        # Making the output folders in a safe mode
        if self.verbose:
            upipe.print_info("Creating directory structure")

        # Going back to initial working directory
        self.goto_origfolder()

    @property
    def full_pointings_list(self):
        return get_list_pointings( joinpath(self.pipe_params.root,
                                            self.targetname))

    def _check_pointings_list(self, list_pointings=None, default_list=None):
        """set the list pointings to self

        Args:
            list_pointings:

        Returns:
            list_pointings after decrypting
        """
        if default_list is None:
            default_list = self.full_pointings_list
        # Now the list of pointings
        if list_pointings is None:
            # This is using all the existing pointings
            return default_list
        else:
            checked_list_pointings = []
            for pointing in list_pointings:
                if pointing not in default_list:
                    upipe.print_warning("No pointing {} for the given "
                                        "target".format(pointing))
                else:
                    checked_list_pointings.append(pointing)

            return checked_list_pointings

    def _add_targetname(self, name, asprefix=True):
        """Add targetname to input name and return it
        
        Input
        -----
        name: str

        Returns
        -------
        new name including targetname
        """
        if self.add_targetname:
            if asprefix:
                return "{0}_{1}".format(self.targetname, name)
            else:
                return "{0}{1}_".format(name, self.targetname)
        else:
            return name

    def _get_pixtable_list(self, dict_exposures=None):
        """Check if pointings and dictionary are compatible
        """
        # Dictionary of exposures to select per pointing
        self.dict_exposures = dict_exposures

        # Getting the pieces of the names to be used for pixtabs
        pixtable_suffix = prep_recipes_pipe.dict_products_scipost['individual'][0]
        if self._pixtab_in_comb_folder and self.add_targetname:
            pixtable_suffix = self._add_targetname(pixtable_suffix)

        # Initialise the dictionary of pixtabs to be found in each pointing
        self.dict_pixtabs_in_pointings = {}
        self.dict_allpixtabs_in_pointings = {}
        # Loop on Pointings
        for pointing in self.list_pointings:
            # get the path
            if self._pixtab_in_comb_folder:
                path_pixtables = self.paths.cubes
                pointing_suffix = "_P{0:02d}".format(np.int(pointing))
            else:
                path_pointing = getattr(self.paths, self.dict_name_pointings[pointing])
                path_pixtables = path_pointing + self.pipe_params.object
                pointing_suffix = ""
            # List existing pixtabs, using the given suffix
            list_pixtabs = glob.glob(path_pixtables + "{0}{1}{2}*fits".format(
                                     pixtable_suffix, self.suffix,
                                     pointing_suffix))

            # Take (or not) the fixed pixtables
            if self.use_fixed_pixtables:
                prefix_to_consider = "{0}{1}".format(self.prefix_fixed_pixtables,
                                                     pixtable_suffix)
                list_fixed_pixtabs = glob.glob(path_pixtables +
                                               "{0}{1}{2}*fits".format(
                                                   prefix_to_consider,
                                                   self.suffix,
                                                   pointing_suffix))

                # Looping over the existing fixed pixtables
                for fixed_pixtab in list_fixed_pixtabs:
                    # Finding the name of the original one
                    orig_pixtab = fixed_pixtab.replace(prefix_to_consider,
                                                       pixtable_suffix)
                    if orig_pixtab in list_pixtabs:
                        # If it exists, replace it
                        list_pixtabs[list_pixtabs.index(orig_pixtab)] = fixed_pixtab
                        upipe.print_warning("Fixed PixTable {0} was included in "
                                            "the list and Pixtable {1} was thus "
                                            "removed from the list.".format(
                                               fixed_pixtab, orig_pixtab))
                    else:
                        upipe.print_warning("Original Pixtable {0} not found."
                                            "Hence will not include fixed "
                                            "PixTable in the list {0}".format(
                                               orig_pixtab, fixed_pixtab))

            full_list = copy.copy(list_pixtabs)
            full_list.sort()
            self.dict_allpixtabs_in_pointings[pointing] = full_list

            # Filter the list with the pointing dictionary if given
            select_list_pixtabs = filter_list_with_pdict(list_pixtabs,
                                                         [pointing],
                                                         self.dict_exposures,
                                                         verbose=self.verbose)

            select_list_pixtabs.sort()
            self.dict_pixtabs_in_pointings[pointing] = select_list_pixtabs

    def _read_offset_table(self, name_offset_name=None, folder_offset_table=None):
        """Reading the Offset Table
        If readable, the table is read and set in the offset_table attribute.

        Input
        -----
        name_offset_name: str
            Name of the offset table
            Default is None
        folder_offset_table: str
            Name of the folder to find the offset table
            Default is None
        """
        self.name_offset_name = name_offset_name
        if self.name_offset_name is None:
            upipe.print_warning("No Offset table name given", pipe=self)
            self.offset_table = Table()
            return

        # Using the given folder name, alignment one by default
        if folder_offset_table is None:
            self.folder_offset_table = self.paths.alignment
        else:
            self.folder_offset_table = folder_offset_table

        fullname_offset_name = joinpath(self.folder_offset_table,
                                          self.name_offset_name)
        if not os.path.isfile(fullname_offset_name):
            upipe.print_error("Offset table [{0}] not found".format(
                fullname_offset_name), pipe=self)
            self.offset_table = Table()
            return

        # Opening the offset table
        self.offset_table = Table.read(fullname_offset_name)

    def _check_offset_table(self, name_offset_name=None, folder_offset_table=None):
        """Checking if DATE-OBS and MJD-OBS are in the OFFSET Table

        Input
        -----
        name_offset_name: str
            Name of the offset table
            Default is None
        folder_offset_table: str
            Name of the folder to find the offset table
            Default is None
        """
        if name_offset_name is None:
            return

        self._read_offset_table(name_offset_name=name_offset_name,
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
            for pixtab_name in self.dict_pixtabs_in_pointings[pointing]:
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
                self.dict_pixtabs_in_pointings[pointing].remove(pixtab)
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
        That includes: root, data, target, but also _dict_paths, paths
        """
        # initialisation of the full paths 
        self.paths = musepipe.PipeObject("All Paths useful for the pipeline")
        self.paths.root = self.pipe_params.root
        self.paths.data = joinpath(self.paths.root, self.pipe_params.data)
        self.paths.target = joinpath(self.paths.root, self.targetname)

        self._dict_paths = {"combined": self.paths}

        for name in self._dict_combined_folders:
            setattr(self.paths, name, joinpath(self.paths.data, getattr(self.pipe_params, name)))

        # Creating the filenames for Master files
        self.dict_name_pointings = {}
        for pointing in self.list_pointings:
            name_pointing = "P{0:02d}".format(np.int(pointing))
            self.dict_name_pointings[pointing] = name_pointing
            # Adding the path of the folder
            setattr(self.paths, name_pointing,
                    joinpath(self.paths.root, "{0}/P{1:02d}/".format(self.targetname, pointing)))

        # Creating the attributes for the folders needed in the TARGET root folder, e.g., for alignments
        for name in self.pipe_params._dict_folders_target:
            setattr(self.paths, name, joinpath(self.paths.target, self.pipe_params._dict_folders_target[name]))

    def create_reference_wcs(self, pointings_wcs=True, mosaic_wcs=True,
                             reference_cube=True, refcube_name=None,
                             **kwargs):
        """Create the WCS reference files, for all individual pointings and for
        the mosaic.

        pointings_wcs: bool [True]
            Will run the individual pointing WCS
        mosaic_wcs: bool [True]
            Will run the combined WCS
        lambdaminmax: [float, float]

        """
        lambdaminmax = kwargs.pop("lambdaminmax", lambdaminmax_for_mosaic)

        # Creating the reference cube if not existing
        if reference_cube:
            if refcube_name is None:
                upipe.print_info("@@@@@@@ First creating a reference (mosaic) "
                                 "cube with short spectral range @@@@@@@")
                self.run_combine(lambdaminmax=lambdaminmax_for_wcs,
                                 filter_list="white",
                                 prefix_all=default_prefix_wcs,
                                 targetname_asprefix=False)
                wcs_refcube_name = self._combined_cube_name
            else:
                upipe.print_info("@@@@@@@@ Start creating the narrow-lambda "
                                 "WCS and Mask from reference Cube @@@@@@@@")
                wcs_refcube_name = self.create_combined_wcs(
                                       refcube_name=refcube_name)
        else:
            # getting the name of the final datacube (mosaic)
            cube_suffix = prep_recipes_pipe.dict_products_scipost['cube'][0]
            cube_name = "{0}{1}.fits".format(default_prefix_wcs,
                                          self._add_targetname(cube_suffix))
            wcs_refcube_name = joinpath(self.paths.cubes, cube_name)

        if pointings_wcs:
            # Creating the full mosaic WCS first with a narrow lambda range
            # Then creating the mask WCS for each pointing
            upipe.print_info("@@@@@@@@ Start creating the individual "
                             "Pointings Masks @@@@@@@@")
            self.create_all_pointings_wcs(lambdaminmax_mosaic=lambdaminmax,
                                          **kwargs)

        if mosaic_wcs:
            # Creating a reference WCS for the Full Mosaic with the right
            # Spectral coverage for a full mosaic
            upipe.print_info("@@@@@@@ Start creating the full-lambda WCS @@@@@@@")
            self._combined_wcs_name = self.create_combined_wcs(
                                        prefix_wcs=prefix_mosaic,
                                        lambdaminmax_wcs=lambdaminmax_for_mosaic,
                                        refcube_name=wcs_refcube_name)

    def run_combine_all_single_pointings(self,
                                         add_suffix="",
                                         sof_filename='pointings_combine',
                                         list_pointings=None,
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
        list_pointings = self._check_pointings_list(list_pointings,
                                                    self.list_pointings)

        # Additional suffix if needed
        for pointing in list_pointings:
            upipe.print_info("Combining single pointings - Pointing {0:02d}".format(
                             np.int(pointing)))
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
        wcs_from_pointing = kwargs.pop("wcs_from_pointing", False)
        print(ref_wcs, wcs_from_pointing)

        # Wcs_from_pointing
        # If true, use the reference pointing wcs
        if wcs_from_pointing:
            if ref_wcs is not None:
                upipe.print_warning("wcs_from_pointing is set to True. "
                                    "but will not overwrite ref_wcs as it was"
                                    "specifically provided")
            else:
                prefix_wcs = kwargs.pop("prefix_wcs", default_prefix_wcs)
                self.add_targetname = kwargs.pop("add_targetname", True)
                prefix_wcs = self._add_targetname(prefix_wcs, asprefix=False)
                ref_wcs = "{0}{1}_P{2:02d}.fits".format(prefix_wcs,
                                                        prefix_final_cube,
                                                        np.int(pointing))

        # Running the combine for that single pointing
        self.run_combine(list_pointings=[np.int(pointing)], suffix=suffix,
                         sof_filename=sof_filename,
                         ref_wcs=ref_wcs,
                         **kwargs)

    def create_all_pointings_wcs(self, filter_list="white",
                                 list_pointings=None, **kwargs):
        """Create all pointing masks one by one
        as well as the wcs for each individual pointings. Using the grid
        from the global WCS of the mosaic but restricting it to the 
        range of non-NaN.
        Hence this needs a global WCS mosaic as a reference to work.

        Input
        -----
        filter_list = list of str
            List of filter names to be used. 
        """
        # If list_pointings is None using the initially set up one
        list_pointings = self._check_pointings_list(list_pointings,
                                                    self.list_pointings)

        # Additional suffix if needed
        for pointing in list_pointings:
            upipe.print_info("Making WCS Mask for "
                             "pointing {0:02d}".format(np.int(pointing)))
            _ = self.create_pointing_wcs(pointing=pointing,
                                         filter_list=filter_list, **kwargs)

    def create_pointing_wcs(self, pointing,
            lambdaminmax_mosaic=lambdaminmax_for_mosaic,
            filter_list="white", **kwargs):
        """Create the mask of a given pointing
        And also a WCS file which can then be used to compute individual
        pointings with a fixed WCS.

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
        wcs_auto = kwargs.pop("wcs_auto", True)

        # Running combine with the ref WCS with only 2 spectral pixels
        # Limit the maximum lambda to the wcs ones
        self.run_combine_single_pointing(pointing=pointing,
                                         filter_list=filter_list,
                                         sof_filename="pointing_mask",
                                         add_targetname=self.add_targetname,
                                         prefix_all=prefix_mask,
                                         lambdaminmax=lambdaminmax_for_wcs,
                                         wcs_auto=wcs_auto,
                                         **kwargs)

        # Now creating the mask with 0's and 1's
        dir_mask = upipe.normpath(self.paths.cubes)

        # Adding targetname for the final names
        prefix_mask = self._add_targetname(prefix_mask)
        prefix_wcs = self._add_targetname(prefix_wcs, asprefix=False)

        name_mask = "{0}{1}_P{2:02d}.fits".format(prefix_mask,
                                                  prefix_final_cube,
                                                  np.int(pointing))
        finalname_wcs = "{0}{1}_P{2:02d}.fits".format(prefix_wcs,
                                               prefix_final_cube,
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
            cube_suffix = prep_recipes_pipe.dict_products_scipost['cube'][0]
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

    def create_combined_wcs(self, refcube_name=None,
            lambdaminmax_wcs=lambdaminmax_for_wcs,
            **kwargs):
        """Create the reference WCS from the full mosaic
        with a given range of lambda.

        Input
        -----
        refcube_name: str
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

        if refcube_name is None:
            # getting the name of the final datacube (mosaic)
            cube_suffix = prep_recipes_pipe.dict_products_scipost['cube'][0]
            cube_suffix = self._add_targetname(cube_suffix)
            refcube_name = joinpath(self.paths.cubes, cube_suffix + ".fits")

        # test if cube exists
        if not os.path.isfile(refcube_name):
            upipe.print_error("[combine/create_combined_wcs] File {0} does not exist "
                              "- Aborting.".format(refcube_name))
            return

        # Opening the cube via MuseCube
        refcube = MuseCube(filename=refcube_name)

        # Creating the new cube
        prefix_wcs = kwargs.pop("prefix_wcs", default_prefix_wcs)
        upipe.print_info("Now creating the Reference WCS cube using prefix '{0}'".format(
            prefix_wcs))
        cfolder, cname = refcube.create_reference_cube(lambdamin=lambdaminmax_wcs[0],
                lambdamax=lambdaminmax_wcs[1], prefix=prefix_wcs, **kwargs)

        # Now transforming this into a bona fide 1 extension WCS file
        combined_wcs_name = joinpath(cfolder, cname)
        d = pyfits.getdata(combined_wcs_name, ext=1)
        h = pyfits.getheader(combined_wcs_name, ext=1)
        hdu = pyfits.PrimaryHDU(data=d, header=h)
        hdu.writeto(combined_wcs_name, overwrite=True)
        upipe.print_info("...Done")
        return combined_wcs_name

    def run_combine(self, sof_filename='pointings_combine',
                    lambdaminmax=[4000., 10000.],
                    list_pointings=None,
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
        self.add_targetname = kwargs.pop("add_targetname", self.add_targetname)
        asprefix = kwargs.pop("targetname_asprefix", True)
        prefix_wcs = kwargs.pop("prefix_wcs", default_prefix_wcs)
        prefix_all = kwargs.pop("prefix_all", "")
        prefix_all = self._add_targetname(prefix_all, asprefix)

        if "name_offset_name" in kwargs:
            name_offset_name = kwargs.pop("name_offset_name")
            folder_offset_table = kwargs.pop("folder_offset_table", self.folder_offset_table)
            self._check_offset_table(name_offset_name, folder_offset_table)

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # If list_pointings is None using the initially set up one
        list_pointings = self._check_pointings_list(list_pointings,
                                                    self.list_pointings)

        # Abort if only one exposure is available
        # exp_combine needs a minimum of 2
        nexpo_tocombine = sum(len(self.dict_pixtabs_in_pointings[pointing])
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
            if ref_wcs is not None:
                 upipe.print_warning("wcs_auto is True, but ref_wcs was "
                                     "specifically provided, and "
                                     "will not be overwritten.")
                 upipe.print_warning("Provided ref_wcs is {}".format(ref_wcs))
            else:
                # getting the name of the final datacube (mosaic)
                cube_suffix = prep_recipes_pipe.dict_products_scipost['cube'][0]
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
        if self.name_offset_name is not None:
            self._sofdict['OFFSET_LIST'] = [joinpath(self.folder_offset_table,
                                                     self.name_offset_name)]

        pixtable_name = dict_listObject[expotype]
        self._sofdict[pixtable_name] = []
        for pointing in list_pointings:
            self._sofdict[pixtable_name] += self.dict_pixtabs_in_pointings[pointing]

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
