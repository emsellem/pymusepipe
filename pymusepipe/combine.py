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

try :
    import astropy as apy
    from astropy.io import fits as pyfits
except ImportError :
    raise Exception("astropy is required for this module")

from astropy.utils.exceptions import AstropyWarning
import warnings

# Importing pymusepipe modules
from pymusepipe.recipes_pipe import PipeRecipes
from pymusepipe.prep_recipes_pipe import PipePrep
from pymusepipe.init_musepipe import InitMuseParameters
import pymusepipe.util_pipe as upipe
from pymusepipe import musepipe 
# Default keywords for MJD and DATE
from pymusepipe.align_pipe import default_mjd_table, default_date_table
from pymusepipe.prep_recipes_pipe import dic_products_scipost

__version__ = '0.0.2 (28 Feb 2019)'
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
        # Alignment for various pointings
        "pointings": "Pointings/"
        }

class MusePointings(PipePrep, PipeRecipes) :
    def __init__(self, targetname=None, list_pointings=[1], 
            dic_exposures_in_pointing=None,
            rc_filename=None, cal_filename=None, 
            combined_folder_name="Combined", suffix="",
            offset_table=None,
            outlog=None, logfile="MusePipeCombine.log", reset_log=False,
            verbose=True, **kwargs):
        """Initialisation of class muse_expo

        Input
        -----
        targetname: string (e.g., 'NGC1208'). default is None. 

        rc_filename: filename to initialise folders
        cal_filename: filename to initiale FIXED calibration MUSE files
        outlog: string, output directory for the log files
        verbose: boolean. Give more information as output (default is True)
        vsystemic: float (default is 0), indicating the systemic velocity of the galaxy [in km/s]

        Other possible entries
        ----------------------
        warnings: strong  ('ignore'by default. If set to ignore, will ignore the Astropy Warnings.

        """
        # Verbose option
        self.verbose = verbose

        # Warnings for astropy
        self.warnings = kwargs.pop("warnings", 'ignore')
        if self.warnings == 'ignore':
           warnings.simplefilter('ignore', category=AstropyWarning)

        # Setting the default attibutes #####################
        self.targetname = targetname
        self.list_pointings = list_pointings

        self.combined_folder_name = combined_folder_name
        self.vsystemic = np.float(kwargs.pop("vsystemic", 0.))

        # Setting other default attributes
        if outlog is None : 
            outlog = "log_{timestamp}".format(timestamp = upipe.create_time_name())
            upipe.print_info("The Log folder will be {0}".format(outlog))
        self.outlog = outlog
        self.logfile = joinpath(self.outlog, logfile)
        self.suffix = suffix

        # End of parameter settings #########################

        # Init of the subclasses
        PipeRecipes.__init__(self, **kwargs)

        # =========================================================== #
        # Setting up the folders and names for the data reduction
        # Can be initialised by either an rc_file, 
        # or a default rc_file or harcoded defaults.
        self.pipe_params = InitMuseParameters(rc_filename=rc_filename, 
                            cal_filename=cal_filename)

        # Setting up the relative path for the data, using Galaxy Name + Pointing
        self.pipe_params.data = "{0}/{1}/".format(self.targetname, self.combined_folder_name)

        self.pipe_params.init_default_param(dic_combined_folders)
        self._dic_combined_folders = dic_combined_folders

        # Setting all the useful paths
        self.set_fullpath_names()

        # Checking input cubes
        self._check_pointings(dic_exposures_in_pointing)
        if offset_table is not None:
            self._check_offset_table(offset_table)

        # Making the output folders in a safe mode
        if self.verbose:
            upipe.print_info("Creating directory structure")

        # and Recording the folder where we start
        self.paths.orig = os.getcwd()

        # First create the data Combined folder
        upipe.safely_create_folder(self.paths.data, verbose=verbose)

        # Go to the Combined Folder
        self.goto_folder(self.paths.data)

        # =========================================================== #
        # Now create full path folder 
        for folder in self._dic_combined_folders.keys() :
            upipe.safely_create_folder(self._dic_combined_folders[folder], verbose=verbose)

        # Going back to initial working directory
        self.goto_prevfolder()

    def _check_pointings(self, dic_exposures_in_pointings=None):
        """Check if pointings and dictionary are compatible
        """
        # Dictionary of exposures to select per pointing
        self.dic_exposures_in_pointings = dic_exposures_in_pointings

        # Getting the pieces of the names to be used for pixtabs
        pixtable_suffix = dic_products_scipost['individual'][0]

        # Initialise the dictionary of pixtabs to be found in each pointing
        self.dic_pixtabs_in_pointings = {}
        # Loop on Pointings
        for pointing in self.list_pointings:
            # get the path
            path_pointing = getattr(self.paths, self.dic_name_pointings[pointing])
            # List existing pixtabs, using the given suffix
            list_pixtabs = glob.glob(path_pointing + self.pipe_params.object + 
                    "{0}{1}*fits".format(pixtable_suffix, self.suffix))

            # if no selection on exposure names are given
            # Select all existing pixtabs
            if self.dic_exposures_in_pointings is None:
                select_list_pixtabs = list_pixtabs

            # Otherwise use the ones which are given via their expo numbers
            else:
                try:
                    # this is the list of exposures to consider
                    list_expo = self.dic_exposures_in_pointings[pointing]
                    # We loop on that list
                    for expo in list_expo:
                        # Check whether this exists in the our cube list
                        for cube_name in list_pixtabs:
                            if expo in cube_name:
                                # We select the cube
                                select_list_pixtabs.append(cube_name)
                                # And remove it from the list
                                list_pixtabs.remove(cube_name)
                                # We break out of the cube for loop
                                break
                except:
                    select_list_pixtabs = []

            self.dic_pixtabs_in_pointings[pointing] = select_list_pixtabs.sort()

    def _check_offset_table(self, offset_table=None):
        """Checking if DATE-OBS and MJD-OBS are in the OFFSET Table
        """
        if offset_table is None:
            upipe.print_error("No Offset table given")
            return
        else:
            self.offset_table = offset_table
        # getting the MJD and DATE from the OFFSET table
        self.table_mjdobs = self.offset_table[default_mjd_table]
        self.table_dateobs = self.offset_table[default_date_table]

        # Checking existence of each pixel_table in the offset table
        for pointing in self.list_pointings:
            pixtab_to_exclude.append = []
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
                # Exclude the one which have not been found
                for pixtab in pixtab_to_exclude:
                    self.dic_pixtabs_in_pointings[pointing].remove(pixtab)

    def goto_prevfolder(self, logfile=False) :
        """Go back to previous folder
        """
        upipe.print_info("Going back to the original folder {0}".format(self.paths._prev_folder))
        self.goto_folder(self.paths._prev_folder, logfile=logfile, verbose=False)
            
    def goto_folder(self, newpath, logfile=False, verbose=True) :
        """Changing directory and keeping memory of the old working one
        """
        try: 
            prev_folder = os.getcwd()
            newpath = os.path.normpath(newpath)
            os.chdir(newpath)
            if verbose :
                upipe.print_info("Going to folder {0}".format(newpath))
            if logfile :
                upipe.append_file(joinpath(self.paths.data, self.logfile), "cd {0}\n".format(newpath))
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

    def run_combine(self, sof_filename='exp_combine', expotype="REDUCED", 
            tpl="ALL", stage="reduced", list_pointing=None, 
            lambdaminmax=[4000.,10000.], suffix="", **kwargs):
        """MUSE Exp_combine treatment of the reduced pixtables
        Will run the esorex muse_exp_combine routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time
        list_expo: list of integers providing the exposure numbers
        """
        # Lambda min and max?
        [lambdamin, lambdamax] = lambdaminmax
        # Save options
        save = kwargs.pop("save", "cube,combined")
        # Filters
        filter_list = kwargs.pop("filter_list", "white")
        filter_for_alignment = kwargs.pop("filter_for_alignment", "Cousins_R")
        offset_list = kwargs.pop("offset_list", "True")

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Abort if only one exposure is available
        # exp_combine needs a minimum of 2
        if len(combine_table) <= 1:
            if self.verbose:
                upipe.print_warning("The combined pointing has only one exposure: process aborted",
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Now creating the SOF file, first reseting it
        self._sofdict.clear()
        # Selecting only exposures to be treated
        # Producing the list of REDUCED PIXTABLES
        self._add_calib_to_sofdict("FILTER_LIST")
        pixtable_name = self._get_suffix_product('REDUCED')
        pixtable_name_thisone = dic_products_scipost['individual']

        # Setting the default option of offset_list
        if offset_list :
            offset_list_tablename = kwargs.pop("offset_list_tablename", None)
            if offset_list_tablename is None:
                offset_list_tablename = "{0}{1}_{2}_{3}_{4}.fits".format(
                        dic_files_products['ALIGN'][0], suffix, filter_for_alignment, 
                        expotype, tpl)
            if not os.path.isfile(joinpath(folder_expo, offset_list_tablename)):
                upipe.print_error("OFFSET_LIST table {0} not found in folder {1}".format(
                        offset_list_tablename, folder_expo), pipe=self)
                return

            self._sofdict['OFFSET_LIST'] = [joinpath(folder_expo, offset_list_tablename)]

        self._sofdict[pixtable_name] = []
        for prod in pixtable_name_thisone:
           self._sofdict[pixtable_name] += [joinpath(folder_expo,
               '{0}_{1}_{2:04d}.fits'.format(prod, row['tpls'], row['iexpo'])) for row in
               combine_table]
        self.write_sof(sof_filename="{0}_{1}{2}_{3}".format(sof_filename, expotype, 
            suffix, tpl), new=True)

        # Product names
        dir_products = self._get_fullpath_expo(expotype, stage)
        name_products, suffix_products, suffix_prefinalnames = self._get_combine_products(filter_list) 

        # Combine the exposures 
        self.recipe_combine(self.current_sof, dir_products, name_products, 
                tpl, expotype, suffix_products=suffix_products,
                suffix_prefinalnames=suffix_prefinalnames,
                save=save, suffix=suffix, filter_list=filter_list, **kwargs)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)
