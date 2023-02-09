# Licensed under a MIT style license - see LICENSE.rst

"""MUSE-PHANGS target sample module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"

# Standard modules
import os
from os.path import join as joinpath
import copy

import numpy as np

from . import util_pipe as upipe
from .musepipe import MusePipe
from .config_pipe import (PHANGS_reduc_config,
                          default_short_PHANGS_filter_list,
                          default_PHANGS_filter_list,
                          default_short_filter_list,
                          default_filter_list,
                          default_prefix_wcs,
                          default_prefix_wcs_mosaic,
                          dict_default_for_recipes)
from .init_musepipe import InitMuseParameters
from .combine import MusePointings
from .align_pipe import rotate_pixtables
from .mpdaf_pipe import MuseCubeMosaic, MuseCube
from .prep_recipes_pipe import dict_products_scipost
from .version import __version__ as version_pack
from .util_pipe import (add_string, get_pointing_name, get_dataset_name,
                        check_filter_list)

from astropy.table import Table

# ----------------- Galaxies and Datasets ----------------#
# Sample of galaxies
# For each galaxy, we provide the datasets numbers and the run attached to that dataset
dict_SAMPLE_example = {
        "NGC628": ['P100', {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}],
        "NGC1087": ['P101', {1:1}], 
        }
# ----------------- Galaxies and Datasets ----------------#

#=============== Useful function ==========================#
def insert_suffix(filename, suffix=""):
    """Create a new filename including the 
    suffix in the name

    Input
    -----
    filename: str
    suffix: str
    """
    # Create new name
    sfilename, extension = os.path.splitext(filename)
    if suffix != "":
        suffix = f"_{suffix}"
    return "{0}{1}{2}".format(sfilename, suffix, extension)


def update_calib_file(filename, subfolder="", folder_config=""):
    """Update the rcfile with a new root

    Input
    -----
    filename: str
        Name of the input filename
    folder_config: str
        Default is "". Name of folder for filename
    subfolder: str
        Name of subfolder to add in the path
    """
    full_filename = joinpath(folder_config, filename)
    if full_filename is None:
        upipe.print_error("ERROR: input filename is None")
        return ""

    # Testing existence of filename
    if not os.path.isfile(full_filename) :
        upipe.print_error("ERROR: input filename {inputname} cannot be found. ".format(
                            inputname=full_filename))
        return ""

    # If it exists, open and read it
    old_rc = open(full_filename)
    lines = old_rc.readlines()

    # Create new file
    new_filename = insert_suffix(full_filename, subfolder)
    new_rc = open(new_filename, 'w')

    # loop on lines
    for line in lines :
        sline = line.split()
        if sline[0] != "root":
            new_rc.write(line)
            continue
        if not os.path.isdir(sline[1]):
            upipe.print_warning("{} not an existing folder (from rcfile)".format(sline[1]))

        newline = line.replace(sline[1], joinpath(sline[1], subfolder))
        new_rc.write(newline)

    new_rc.close()
    old_rc.close()
    return new_filename
#------------ End of Useful functions -------------#

####################################################
# Defining Dictionary with running functions
####################################################
class PipeDict(dict):
    """Dictionary with extra attributes
    """
    def __init__(self, *args, **kwargs) :
        self.update(*args, **kwargs)
        self._initialised = False

    def __setitem__(self, key, value):
        """Setting the item by using the dictionary of the new values
        """
        for funcname in dir(value):
            if callable(getattr(value, funcname)) and ("run" in funcname):
                setattr(self, funcname, self.run_on_all_keys(funcname))

        super(PipeDict, self).__setitem__(key, value)

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, "
                                "got %d" % len(args))
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]

    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]

    def run_on_all_keys(self, funcname):
        """Runs the given function on all the keys
        """
        def _function(**kwargs):
            for key in self:
                getattr(self[key], funcname)(**kwargs)

        return _function

####################################################
# Defining classes to get samples and objects
####################################################
class MusePipeTarget(object):
    def __init__(self, targetname="",
                 subfolder='P100', list_datasets=None):
        self.targetname = targetname
        self.subfolder = subfolder
        self.list_datasets = list_datasets
        self.pipes = PipeDict()

class MusePipeSample(object):
    def __init__(self, TargetDic, rc_filename=None, cal_filename=None, 
            folder_config="", first_recipe=1, **kwargs) :
        """Using a given dictionary to initialise the sample
        That dictionary should include the names of the targets
        as keys and the subfolder plus datasets to consider

        Input
        -----
        TargetDic: dic
            Dictionary of targets. Keys are target names.
            Values for each target name should be a list of 2 parameters.
                - The first one is the name of the subfolder (e.g. 'P101')
                - The second one is the list of datasets, itself a dictionary
                  with a 0 or 1 for each pointing number depending on whether
                  this should be included in the reduction or not.
                  Results can be seen in self.dict_targets dictionary.
        rc_filename: str
            Default to None
        cal_filename: str
            Default to None

        PHANGS: bool
            Default to False. If True, will use default configuration dictionary
            from config_pipe.
        """
        self.sample = TargetDic
        self.targetnames = list(TargetDic.keys())

        self.first_recipe = first_recipe

        self.__phangs = kwargs.pop("PHANGS", False)
        self.verbose = kwargs.pop("verbose", False)

        # Reading configuration filenames
        if rc_filename is None or cal_filename is None:
            upipe.print_error("rc_filename and/or cal_filename is None. Please define both.")
            return
        self.cal_filename = cal_filename
        self.rc_filename = rc_filename
        self.folder_config = folder_config

        # Initialisation of rc and cal files
        self._init_calib_files()
        # Setting the filters
        if self.__phangs:
            self._short_filter_list = kwargs.pop("short_filter_list",
                                                 default_short_PHANGS_filter_list)
            self._filter_list = kwargs.pop("filter_list",
                                           default_PHANGS_filter_list)
        else:
            self._short_filter_list = kwargs.pop("short_filter_list",
                                                 default_short_filter_list)
            self._filter_list = kwargs.pop("filter_list",
                                           default_filter_list)

        # Initialisation of targets
        self.init_pipes = kwargs.pop("init_pipes", True)
        self.add_targetname = kwargs.pop("add_targetname", True)
        self._init_targets(**kwargs)

    def _init_calib_files(self):
        """Initialise the calibration files with the new
        name using the subfolders
        """
        # Getting the right input for rc and cal names
        folder_config, rc_filename, cal_filename = self._get_calib_filenames()
        # First read the root folder
        init_cal_params = InitMuseParameters(folder_config=folder_config,
                                             rc_filename=rc_filename,
                                             cal_filename=cal_filename,
                                             verbose=self.verbose)
        self.root_path = init_cal_params.root
        self._subfolders = np.unique([self.sample[targetname][0]
                                for targetname in self.targetnames])

        for subfolder in self._subfolders:
            update_calib_file(rc_filename, subfolder, folder_config=folder_config)
            update_calib_file(cal_filename, subfolder, folder_config=folder_config)

    def _get_calib_filenames(self, targetname=None):
        """Get calibration file names

        Input
        ----
        targetname: str

        Returns
        -------
        folder_name: str
        rcname: str
        calname: str
        """
        # using targetname or not
        if targetname is None:
            name_rc = self.rc_filename
            name_cal = self.cal_filename
        else:
            name_rc = insert_suffix(self.rc_filename, self.targets[targetname].subfolder)
            name_cal = insert_suffix(self.cal_filename, self.targets[targetname].subfolder)

        folder_config = self.folder_config

        # Checking the folders
        folder_rc, rc_filename_target = os.path.split(joinpath(folder_config, name_rc))
        folder_cal, cal_filename_target = os.path.split(joinpath(folder_config, name_cal))

        if rc_filename_target=="" or cal_filename_target=="":
            upipe.print_error("Missing a calibration name file")
            return

        if folder_rc == folder_cal:
            folder_config = folder_rc
        else:
            rc_filename_target = joinpath(folder_rc, rc_filename_target)
            cal_filename_target = joinpath(folder_cal, cal_filename_target)
            folder_config = ""

        return folder_config, rc_filename_target, cal_filename_target

    def _init_targets(self, **kwargs_init):
        """Initialise the targets using the dictionary
        Returning self.targets with the datasets to consider
        """
        self.targets = {}
        self.pipes = {}
        self.pipes_combine = {}
        self.pipes_mosaic = {}
        for targetname in self.targetnames:
            subfolder = self.sample[targetname][0]
            ldatasets = self.sample[targetname][1]
            # Finding out which datasets should be included
            list_datasets = []
            for lds in ldatasets:
                if ldatasets[lds] == 1:
                    list_datasets.append(lds)
            # Defining the MusePipe for that target - which datasets
            self.targets[targetname] = MusePipeTarget(targetname=targetname,
                                                      subfolder=subfolder,
                                                      list_datasets=list_datasets)
            # Shortcut to call the musepipe instance
            self.pipes[targetname] = self.targets[targetname].pipes

            folder_config, rc_filename, cal_filename = self._get_calib_filenames(targetname)
            self.targets[targetname].rc_filename = rc_filename
            self.targets[targetname].cal_filename = cal_filename
            self.targets[targetname].folder_config = folder_config

            init_params_target = InitMuseParameters(rc_filename=rc_filename,
                                                    cal_filename=cal_filename,
                                                    folder_config=folder_config,
                                                    verbose=self.verbose)
            self.targets[targetname].root_path = init_params_target.root
            self.targets[targetname].data_path = joinpath(init_params_target.root, targetname)
            self.pipes[targetname].root_path = init_params_target.root
            self.pipes[targetname].data_path = joinpath(init_params_target.root, targetname)

            init_comb_target = MusePointings(targetname=targetname,
                                             list_datasets=list_datasets,
                                             rc_filename=rc_filename,
                                             cal_filename=cal_filename,
                                             folder_config=folder_config,
                                             verbose=False, check=False)
            self.targets[targetname].combcubes_path = init_comb_target.paths.cubes

            if self.init_pipes:
                self.set_pipe_target(targetname, **kwargs_init)

    def _check_list_datasets_for_target(self, targetname, list_datasets):
        """Check if dataset is in the list of datasets
        Returns the list of datasets if ok. If not, return an empty list

        Input
        -----
        targetname: str
            name of the target
        list_datasets: list
            List of integer (datasets).

        Returns
        -------
        list_datasets: list
            Empty if input list of datasets is not fully in defined list.
        """
        # Info of the datasets and extracting the observing run for each dataset
        target_datasets = self.targets[targetname].list_datasets

        # Now the list of datasets
        if list_datasets is None:
            return target_datasets
        else:
            checked_datasets_list = []
            # Check they exist
            upipe.print_warning(f"Wished dataset list = {list_datasets}")
            upipe.print_warning(f"Existing Target dataset list = {target_datasets}")
            for dataset in list_datasets:
                if dataset not in target_datasets:
                    upipe.print_warning(f"No dataset [{dataset}] for the given target")
                else:
                    checked_datasets_list.append(dataset)
            return checked_datasets_list

    def _check_targetname(self, targetname):
        """Check if targetname is in list

        Input
        -----
        targetname: str

        Returns
        -------
        status: bool
            True if yes, False if no.
        """
        if targetname not in self.targetnames:
            upipe.print_error("ERROR: no Target named {name} in the defined sample".format(
                                name=targetname))
            return False
        else:
            return True

    def set_pipe_target(self, targetname=None, list_datasets=None, **kwargs):
        """Create the musepipe instance for that target and list of datasets

        Input
        -----
        targetname: str
            Name of the target
        list_datasets: list
            Dataset numbers. Default is None (meaning all datasets
            indicated in the dictonary will be reduced)
        config_args: dic
            Dictionary including extra configuration parameters to pass
            to MusePipe. This allows to define a global configuration.
            If self.__phangs is set to True, this is overwritten with the default
            PHANGS configuration parameters as provided in config_pipe.py.
        """
        verbose = kwargs.pop("verbose", self.verbose)
        # Check if targetname is valid
        if not self._check_targetname(targetname):
            return

        # Galaxy name
        upipe.print_info("=== Initialising MusePipe for Target {name} ===".format(name=targetname))

        # Check if datasets are valid
        list_datasets = self._check_list_datasets_for_target(targetname, list_datasets)
        if len(list_datasets) == 0:
            return

        # Get the filename and extension of log file
        log_filename, log_fileext = os.path.splitext(kwargs.pop("log_filename", 
                        "{0}_{1}.log".format(targetname, version_pack)))

        # Reading extra arguments from config dictionary
        if self.__phangs:
            config_args = PHANGS_reduc_config
            # Set overwrite to False to keep existing tables
            config_args['overwrite_astropy_tables'] = False
        else:
            config_args = kwargs.pop("config_args", None)

        first_recipe = kwargs.pop("first_recipe", 1)
        last_recipe = kwargs.pop("last_recipe", None)

        # Over-writing the arguments in kwargs from config dictionary
        if config_args is not None:
            for attr in config_args:
                if attr not in kwargs:
                    kwargs[attr] = config_args[attr]

        # extracting the kwargs
        list_kwargs = ', '.join(['{}={!r}'.format(k, v) for k, v in kwargs.items()])

        # Config files
        rc_filename = self.targets[targetname].rc_filename
        cal_filename = self.targets[targetname].cal_filename
        folder_config = self.targets[targetname].folder_config

        # Loop on the datasets
        for dataset in list_datasets:
            upipe.print_info("Initialise Pipe for Target = {0:10s} / Dataset {1:03d} ".format(
                                 targetname, dataset))
            # New log file name with dataset included
            log_filename_dataset = f"{log_filename}_{get_pointing_name(dataset)}{log_fileext}"
            # Setting up the names of the output files
            python_command = ("mypipe = musepipe.MusePipe(targetname='{0}', "
                              "dataset={1}, folder_config='{2}', " 
                              "rc_filename='{3}', " "cal_filename='{4}', "
                              "log_filename='{5}', verbose={6}, "
                              "{7})".format(targetname, dataset, folder_config, 
                                  rc_filename, cal_filename, log_filename_dataset, 
                                  verbose, list_kwargs))

            # Creating the musepipe instance, using the shortcut
            self.pipes[targetname][dataset] = MusePipe(
                targetname=targetname, dataset=dataset,
                folder_config=folder_config, rc_filename=rc_filename,
                cal_filename=cal_filename, log_filename=log_filename_dataset,
                first_recipe=first_recipe, last_recipe=last_recipe,
                init_raw_table=True, verbose=verbose, **kwargs)

            # Saving the command
            self.pipes[targetname][dataset].history = python_command
            # Setting back verbose to True to make sure we have a full account
            self.pipes[targetname][dataset].verbose = True
            upipe.print_info(python_command, pipe=self)

        upipe.print_info("End of Pipe initialisation")
        self.pipes[targetname]._initialised = True

    def  _get_path_data(self, targetname, dataset):
        """Get the path for the data
        Parameters
        ----------
        targetname: str
            Name of the target
        dataset: int
            Number for the dataset

        Returns
        -------
        path_data

        """
        return self.pipes[targetname][dataset].paths.data

    def  _get_path_files(self, targetname, dataset, expotype="OBJECT"):
        """Get the path for the files of a certain expotype
        Parameters
        ----------
        targetname: str
            Name of the target
        dataset: int
            Number for the dataset

        Returns
        -------
        path_files

        """
        return self.pipes[targetname][dataset]._get_path_files(expotype)

    def reduce_all_targets(self, **kwargs):
        """Reduce all targets already initialised

        Input
        -----
        first_recipe: int or str
            One of the recipe to start with
        last_recipe: int or str
            One of the recipe to end with
        """
        for target in self.targets:
            upipe.print_info("=== Start Reduction of Target {name} ===".format(name=target))
            self.reduce_target(targetname=target, **kwargs)
            upipe.print_info("===  End  Reduction of Target {name} ===".format(name=target))

    def reduce_target_prealign(self, targetname=None, list_datasets=None, **kwargs):
        """Reduce target for all steps before pre-alignment (included)

        Input
        -----
        targetname: str
            Name of the target
        list_datasets: list
            Dataset numbers. Default is None (meaning all datasets
            indicated in the dictionary will be reduced)
        """
        self.reduce_target(targetname=targetname, list_datasets=list_datasets,
                           last_recipe="prep_align", **kwargs)

    def reduce_target_postalign(self, targetname=None, list_datasets=None, **kwargs):
        """Reduce target for all steps after pre-alignment

        Input
        -----
        targetname: str
            Name of the target
        list_datasets: list
            Dataset numbers. Default is None (meaning all datasets
            indicated in the dictonary will be reduced)
        """
        self.reduce_target(targetname=targetname, list_datasets=list_datasets,
                           first_recipe="align_bydataset", **kwargs)

    def finalise_reduction(self, targetname=None, rot_pixtab=False, create_wcs=True,
                           create_expocubes=True, create_pixtables=True,
                           create_pointingcubes=True,
                           name_offset_table=None, folder_offset_table=None,
                           dict_exposures=None, list_datasets=None,
                           **kwargs):
        """Finalise the reduction steps by using the offset table, rotating the
        pixeltables, then reconstructing the PIXTABLE_REDUCED, produce reference
        WCS for each pointing, and then run the reconstruction of the final
        individual cubes

        Args:
            targetname (str):
            rot_pixtab (bool):
            create_wcs (bool):
            create_expocubes (bool):
            name_offset_table (str):
            folder_offset_table (str):
            **kwargs:

        Returns:

        """
        if rot_pixtab:
            # We first use the offset table to rotate the pixtables
            upipe.print_info("========== ROTATION OF PIXTABLES ===============")
            self.rotate_pixtables_target(targetname=targetname,
                                         folder_offset_table=folder_offset_table,
                                         name_offset_table=name_offset_table,
                                         fakemode=False)

        norm_skycontinuum = kwargs.pop("norm_skycontinuum", True)
        skymethod = kwargs.pop("skymethod", 'model')
        if create_pixtables:
            # We then reconstruct the pixtable reduced so we can
            # redo a muse_exp_combine if needed
            upipe.print_info("==== REDUCED PIXTABLES for REFERENCE MOSAIC ====")
            self.run_target_scipost_perexpo(targetname=targetname,
                                            folder_offset_table=folder_offset_table,
                                            name_offset_table=name_offset_table,
                                            save="individual",
                                            wcs_auto=False,
                                            norm_skycontinuum=norm_skycontinuum,
                                            list_datasets=list_datasets,
                                            pixtab_in_comb_folder=False,
                                            pixtable_type='OBJECT',
                                            skymethod=skymethod)

        if create_wcs:
            # Creating the WCS reference frames. Full mosaic and individual
            # Pointings.
            upipe.print_info("=========== CREATION OF WCS MASKS ==============")
            mosaic_wcs = kwargs.pop("mosaic_wcs", True)
            reference_cube = kwargs.pop("reference_cube", True)
            pointings_wcs = kwargs.pop("pointings_wcs", True)
            refcube_name = kwargs.pop("refcube_name", None)
            full_ref_wcs = kwargs.pop("full_ref_wcs", None)
            default_comb_folder = self.targets[targetname].combcubes_path
            folder_full_ref_wcs = kwargs.pop("folder_full_ref_wcs",
                                             default_comb_folder)
            self.create_reference_wcs(targetname=targetname,
                                      folder_offset_table=folder_offset_table,
                                      name_offset_table=name_offset_table,
                                      reference_cube=reference_cube,
                                      refcube_name=refcube_name,
                                      mosaic_wcs=mosaic_wcs,
                                      pointings_wcs=pointings_wcs,
                                      list_datasets=list_datasets,
                                      ref_wcs=full_ref_wcs,
                                      folder_ref_wcs=folder_full_ref_wcs,
                                      fakemode=False)

        if create_expocubes:
            # Running the individual cubes now with the same WCS reference
            upipe.print_info("=========== CREATION OF EXPO CUBES =============")
            self.run_target_scipost_perexpo(targetname=targetname,
                                            folder_offset_table=folder_offset_table,
                                            name_offset_table=name_offset_table,
                                            save="cube",
                                            norm_skycontinuum=norm_skycontinuum,
                                            list_datasets=list_datasets,
                                            skymethod=skymethod,
                                            **kwargs)

        if create_pointingcubes:
            # Running the pointing cubes now with the same WCS reference
            upipe.print_info("========= CREATION OF POINTING CUBES ===========")
            self.combine_target_per_pointing(targetname=targetname,
                                             name_offset_table=name_offset_table,
                                             folder_offset_table=folder_offset_table,
                                             list_datasets=list_datasets,
                                             filter_list=self._short_filter_list)

    def run_target_scipost_perexpo(self, targetname=None, list_datasets=None, list_pointings=None,
                                   folder_offset_table=None, name_offset_table=None,
                                   **kwargs):
        """Build the cube per exposure using a given WCS

        Args:
            targetname:
            list_datasets:
            **kwargs:

        Returns:

        """
        # Check if datasets are valid
        list_datasets = self._check_list_datasets_for_target(targetname, list_datasets)
        upipe.print_info(f"List of datasets to be reduced: {list_datasets}")
        if len(list_datasets) == 0:
            return

        # WCS imposed by setting the reference
        add_targetname = kwargs.pop("add_targetname", self.add_targetname)
        prefix_all = kwargs.pop("prefix_all", "")
        cube_suffix = dict_products_scipost['cube'][0]
        if add_targetname:
            prefix_all = "{0}_{1}".format(targetname, prefix_all)
            cube_suffix = "{0}_{1}".format(targetname, cube_suffix)
        save = kwargs.pop("save", 'cube')

        wcs_auto = kwargs.pop("wcs_auto", True)
        wcs_suffix = "{0}{1}".format(default_prefix_wcs, cube_suffix)
        if not wcs_auto:
            ref_wcs = kwargs.pop("ref_wcs", None)

        # Fetch the default folder for the WCS files which is the folder
        # of the Combined cubes
        default_comb_folder = self.targets[targetname].combcubes_path
        # Now fetch the value set by the user
        folder_ref_wcs = kwargs.pop("folder_ref_wcs", default_comb_folder)
        filter_list = kwargs.pop("filter_list", self._short_filter_list)

        # initialisation of the combination to make sure we get the pointings right
        # We will thus make use of the dict_tplexpo_in_pointing
        pointing_table = kwargs.pop("pointing_table", None)
        pointing_table_folder = kwargs.pop("pointing_table_folder", "")
        pointing_table_format = kwargs.pop("pointing_table_format", "ascii")
        pixtab_in_comb_folder = kwargs.pop("pixtab_in_comb_folder", False)
        pixtable_type = kwargs.pop("pixtable_type", 'REDUCED')
        self.init_combine(targetname=targetname, list_datasets=list_datasets,
                          list_pointings=list_pointings, pointing_table=pointing_table,
                          pointing_table_folder=pointing_table_folder,
                          pointing_table_format=pointing_table_format,
                          pixtab_in_comb_folder=pixtab_in_comb_folder,
                          pixtable_type=pixtable_type)

        # Running the scipost_perexpo for all datasets individually
        dict_tplexpo_per_dataset = self.pipes_combine[targetname].dict_tplexpo_per_dataset
        for dataset in list_datasets:
            obname = self.pipes[targetname][dataset]._get_dataset_name()
            # We now need to identify which tpls are there and which ref_wcs (pointing)
            # For each dataset, we go through all pointings one by one since each has a ref_wcs
            # We then use the list tpl and iexpo to pass to run_scipost_perexpo as list_tplexpo
            for pointing in dict_tplexpo_per_dataset[dataset]:
                pointing_name = get_pointing_name(pointing)
                list_tplexpo = dict_tplexpo_per_dataset[dataset][pointing]
                if wcs_auto:
                    ref_wcs = f"{wcs_suffix}_{pointing_name}.fits"
                if ref_wcs is not None:
                    suffix = f"_WCS_{obname}"
                else:
                    suffix = f"_{obname}"
                kwargs_dataset = {'ref_wcs': ref_wcs,
                                  'suffix': suffix,
                                  'folder_ref_wcs': folder_ref_wcs,
                                  'sof_filename': 'scipost_wcs',
                                  'dir_products': default_comb_folder,
                                  'name_offset_table': name_offset_table,
                                  'folder_offset_table': folder_offset_table,
                                  'offset_list': True,
                                  'filter_list': filter_list,
                                  'prefix_all': prefix_all,
                                  'list_tplexpo': list_tplexpo,
                                  'save': save}
                kwargs.update(kwargs_dataset)
                self.pipes[targetname][dataset].run_scipost_perexpo(**kwargs)

    def run_target_recipe(self, recipe_name, targetname=None,
                          list_datasets=None, **kwargs):
        """Run just one recipe on target

        Input
        -----
        recipe_name: str
        targetname: str
            Name of the target
        list_datasets: list
            Pointing numbers. Default is None (meaning all datasets
            indicated in the dictonary will be reduced)
        """
        # General print out
        upipe.print_info("---- Starting the Recipe {0} for Target={1} "
                         "----".format(recipe_name, targetname))

        # Sorting out the kwargs specific for the MUSE recipes
        kwargs_recipe = {}
        list_keys = list(kwargs.keys())
        for kw in list_keys:
            if kw in dict_default_for_recipes.keys():
                kwargs_recipe[kw] = kwargs.pop(kw, dict_default_for_recipes[kw])

        # Initialise the pipe if needed
        self.set_pipe_target(targetname=targetname, list_datasets=list_datasets,
                first_recipe=recipe_name, last_recipe=recipe_name, **kwargs)

        # Check if datasets are valid
        list_datasets = self._check_list_datasets_for_target(targetname, list_datasets)
        if len(list_datasets) == 0:
            return

        # some parameters which depend on the datasets for this recipe
        kwargs_per_dataset = kwargs.pop("kwargs_per_dataset", {})
        param_recipes = kwargs.pop("param_recipes", {})

        # Loop on the datasets
        for dataset in list_datasets:
            upipe.print_info("====== START - DATASET {0:2d} "
                             "======".format(dataset))

            this_param_recipes = copy.deepcopy(param_recipes)
            if dataset in kwargs_per_dataset:
                if recipe_name in kwargs_per_dataset[dataset]:
                    this_param_recipes[recipe_name].update(
                                     kwargs_per_dataset[dataset][recipe_name])

            # Initialise raw tables if not already done (takes some time)
            if not self.pipes[targetname][dataset]._raw_table_initialised:
                self.pipes[targetname][dataset].init_raw_table(overwrite=True)
            if self.__phangs:
                self.pipes[targetname][dataset].run_phangs_recipes(param_recipes=this_param_recipes,
                                                                    **kwargs_recipe)
            else:
                self.pipes[targetname][dataset].run_recipes(param_recipes=this_param_recipes,
                                                             **kwargs_recipe)
            upipe.print_info("====== END   - DATASET {0:2d} ======".format(dataset))

    def reduce_target(self, targetname=None, list_datasets=None, **kwargs):
        """Reduce one target for a list of datasets

        Input
        -----
        targetname: str
            Name of the target
        list_datasets: list
            Dataset numbers. Default is None (meaning all datasets
            indicated in the dictonary will be reduced)
        first_recipe: str or int [1]
        last_recipe: str or int [max of all recipes]
            Name or number of the first and last recipes to process
        """
        # General print out
        upipe.print_info("---- Starting the Data Reduction for Target={0} ----".format(
                            targetname))

        # Get the parameters for the recipes
        param_recipes = kwargs.pop("param_recipes", {})
        kwargs_recipe = {}
        list_keys = list(kwargs.keys())
        for kw in list_keys:
            if kw in dict_default_for_recipes.keys():
                kwargs_recipe[kw] = kwargs.pop(kw, dict_default_for_recipes[kw])

        for key in ['first_recipe', 'last_recipe']:
            if key in kwargs:
                kwargs_recipe[key] = kwargs.pop(key)

        # Initialise the pipe if needed
        if not self.pipes[targetname]._initialised :
            self.set_pipe_target(targetname=targetname, list_datasets=list_datasets, **kwargs)

        # Check if datasets are valid
        list_datasets = self._check_list_datasets_for_target(targetname, list_datasets)
        if len(list_datasets) == 0:
            return

        # Loop on the datasets
        for dataset in list_datasets:
            upipe.print_info("====== START - DATASET {0:2d} ======".format(dataset))
            # Initialise raw tables if not already done (takes some time)
            if not self.pipes[targetname][dataset]._raw_table_initialised:
                self.pipes[targetname][dataset].init_raw_table(overwrite=True)
            if self.__phangs:
                self.pipes[targetname][dataset].run_phangs_recipes(param_recipes=param_recipes,
                                                                    **kwargs_recipe)
            else:
                self.pipes[targetname][dataset].run_recipes(param_recipes=param_recipes,
                                                             **kwargs_recipe)
            upipe.print_info("====== END   - DATASET {0:2d} ======".format(dataset))

    def rotate_pixtables_target(self, targetname=None, list_datasets=None,
                                folder_offset_table=None, name_offset_table=None,
                                fakemode=False, **kwargs):
        """Rotate all pixel table of a certain targetname and datasets
        """
        # General print out
        upipe.print_info("---- Starting the PIXTABLE ROTATION "
                         "for Target={0} ----".format(targetname))

        # Initialise the pipe if needed
        if not self.pipes[targetname]._initialised \
            or "first_recipe" in kwargs or "last_recipe" in kwargs:
            self.set_pipe_target(targetname=targetname,
                                 list_datasets=list_datasets, **kwargs)

        # Check if datasets are valid
        list_datasets = self._check_list_datasets_for_target(targetname, list_datasets)

        if len(list_datasets) == 0:
            return

        prefix = kwargs.pop("prefix", "")
        if folder_offset_table is None:
            folder_offset_table = self.pipes[targetname][list_datasets[0]].paths.alignment
        offset_table = Table.read(joinpath(folder_offset_table, name_offset_table))
        offset_table.sort(["DATASET_OBS", "IEXPO_OBS"])
        # Loop on the datasets

        for row in offset_table:
            iexpo = row['IEXPO_OBS']
            dataset = row['DATASET_OBS']
            tpls = row['TPL_START']
            angle = row['ROTANGLE']
            ndigits = int(self.pipes[targetname][list_datasets[0]].pipe_params.ndigits)
            upipe.print_info(f"Rotation ={angle} Deg for "
                             f"Dataset={dataset:{ndigits}d}, "
                             f"TPLS={tpls} - Expo {iexpo:02d}")
            folder_expos = self._get_path_files(targetname, dataset)
            name_suffix = "{0}_{1:04d}".format(tpls, iexpo)
            rotate_pixtables(folder=folder_expos, name_suffix=name_suffix,
                             list_ifu=None, angle=angle, fakemode=fakemode,
                             prefix=prefix, **kwargs)

    def init_mosaic(self, targetname=None, list_datasets=None, prefix_cubes="DATACUBE_FINAL_WCS",
                    **kwargs):
        """Prepare the combination of targets

        Input
        -----
        targetname: str [None]
            Name of target
        list_datasets: list [or None=default meaning all datasets]
            List of datasets (e.g., [1,2,3])
        prefix_cubes: str default="DATACUBE_FINAL_WCS", optional
            Prefix to be used to list the cubes to consider
        """
        add_targetname = kwargs.pop("add_targetname", self.add_targetname)
        # Check if pointings are ok
        if list_datasets is None:
            list_datasets = copy.copy(self.targets[targetname].list_datasets)

        # Make a list for the masking of the cubes to take into account
        list_datasets_names = [f"{get_dataset_name(dataset)}"
                                for dataset in list_datasets]
        upipe.print_info(f"List of datasets names: {list_datasets_names}")

        default_comb_folder = self.targets[targetname].combcubes_path
        folder_ref_wcs = kwargs.pop("folder_ref_wcs", default_comb_folder)
        folder_cubes = kwargs.pop("folder_cubes", default_comb_folder)
        if add_targetname:
            wcs_prefix = "{}_".format(targetname)
            prefix_cubes = "{0}_{1}".format(targetname, prefix_cubes)
        else:
            wcs_prefix = ""
        ref_wcs = kwargs.pop("ref_wcs", f"{default_prefix_wcs_mosaic}{wcs_prefix}"
                                        f"DATACUBE_FINAL.fits")
        upipe.print_info(f"Check file: ref_wcs is {ref_wcs}")

        self.pipes_mosaic[targetname] = MuseCubeMosaic(ref_wcs=ref_wcs,
                                                       folder_ref_wcs=folder_ref_wcs,
                                                       folder_cubes=folder_cubes,
                                                       prefix_cubes=prefix_cubes,
                                                       list_suffix=list_datasets_names,
                                                       **kwargs)

    def convolve_mosaic_per_pointing(self, targetname=None, list_pointings=None,
                                     dict_psf={}, target_fwhm=0.,
                                     target_nmoffat=None,
                                     target_function="gaussian", suffix=None,
                                     best_psf=True, min_dfwhm=0.2, fakemode=False,
                                     **kwargs):
        """Convolve the datacubes listed in a mosaic with some target function
        and FWHM. It will try to homogeneise all individual cubes to that
        target PSF.

        Args:
            targetname (str): name of the target
            list_pointings (list): list of pointing numbers for the list of pointings
                to consider
            dict_psf (dict): dictionary providing individual PSFs per pointing
            target_fwhm (float): target FWHM for the convolution [arcsec]
            target_nmoffat (float): tail factor for the moffat function.
            target_function (str): 'moffat' or 'gaussian' ['gaussian']
            suffix (str): input string to be added
            best_psf (bool): if True use the minimum overall possible value. If
                True it will overwrite all the target parameters.
            min_dfwhm (float): minimum difference to be added in quadrature
                [in arcsec]
            filter_list (list): list of filters to be used for reconstructing
                images
            fakemode (bool): if True, will only initialise parameters but not
                proceed with the convolution.
            **kwargs:

        Returns:

        """
        # Filter list for the convolved exposures
        filter_list = kwargs.pop("filter_list", self._short_filter_list)
        filter_list = check_filter_list(filter_list)

        # Initialise and filter with list of datasets
        self.init_mosaic(targetname=targetname, list_pointings=list_pointings,
                         dict_psf=dict_psf, **kwargs)

        # Use the mosaic to determine the lambda range
        l_range = self.pipes_mosaic[targetname].wave.get_range()

        # Calculate the worst psf
        # Detect if there are larger values to account for
        # Using the wavelength dependent FWHM
        if len(dict_psf) > 0:
            best_fwhm = 0.
            for key in dict_psf:
                psf = dict_psf[key]
                fwhm_wave = np.max(psf[4] * (l_range - psf[3]) + psf[1])
                if fwhm_wave > best_fwhm:
                    best_fwhm = fwhm_wave

        if best_psf:
            target_function = "gaussian"
            target_fwhm = np.sqrt(best_fwhm**2 + min_dfwhm**2)
            upipe.print_info(f"Minimum overall FWHM = {best_fwhm:.2f} and "
                             f"target FWHM will be {target_fwhm:.2f}")
            suffix = f"copt_{target_fwhm:.2f}asec"
            upipe.print_warning(f"Overwriting options for the target PSF as"
                                f" best_psf is set to True. \n"
                                f"Suffix for convolved cubes = {suffix}")
            self.pipes_mosaic[targetname].copt_suffix = suffix
            self.pipes_mosaic[targetname].copt_fwhm = target_fwhm

        if suffix is None:
            suffix = "conv{0}_{1:.2f}".format(target_function.lower()[:4], target_fwhm)

        # Convolve
        if not fakemode:
            # Printing the names of the selected cubes
            self.pipes_mosaic[targetname].print_cube_names()
            # Convolving
            self.pipes_mosaic[targetname].convolve_cubes(target_fwhm=target_fwhm,
                                                         target_nmoffat=target_nmoffat,
                                                         target_function=target_function,
                                                         suffix=suffix)

            for name in self.pipes_mosaic[targetname].cube_names:
                # Building the images
                cube = MuseCube(filename=name)
                prefix = (name.replace("DATACUBE_FINAL", "IMAGE_FOV")).split(add_string(suffix))[0]
                cube.build_filterlist_images(filter_list=filter_list,
                                             prefix=prefix, suffix=suffix)

    def mosaic(self, targetname=None, list_pointings=None, init_mosaic=True,
               build_cube=True, build_images=True, **kwargs):
        """

        Args:
            targetname:
            list_pointings:
            **kwargs:

        Returns:

        """
        # Constructing the images for that mosaic
        filter_list = kwargs.pop("filter_list", self._filter_list)
        filter_list = check_filter_list(filter_list)

        # Doing the mosaic with mad
        default_comb_folder = self.targets[targetname].combcubes_path
        folder_cubes = kwargs.pop("folder_cubes", default_comb_folder)

        # defining the default cube name here to then define the output cube name
        suffixout = kwargs.pop("suffixout", "WCS_Pall_mad")
        suffixout = add_string(suffixout)
        default_cube_name = "{0}_DATACUBE_FINAL{1}.fits".format(targetname, suffixout)
        outcube_name = kwargs.pop("outcube_name", default_cube_name)
        outcube_name = joinpath(folder_cubes, outcube_name)

        # Initialise the mosaic or not
        if init_mosaic:
            self.init_mosaic(targetname=targetname,
                             list_pointings=list_pointings,
                             **kwargs)
        else:
            if targetname not in self.pipes_mosaic:
                upipe.print_error(f"Targetname {targetname} not in "
                                  f"self.pipes_mosaic. Please initialise the "
                                  f"mosaic first with self.init_mosaic() or "
                                  f"use init_mosaic=True when calling "
                                  f"self.mosaic().")
                return

        # Doing the MAD combination using mpdaf. Note the build_cube fakemode
        self.pipes_mosaic[targetname].print_cube_names()
        self.pipes_mosaic[targetname].madcombine(outcube_name=outcube_name,
                                                 fakemode=not build_cube)

        if build_images:
            mosaic_name = self.pipes_mosaic[targetname].mosaic_cube_name
            if not os.path.isfile(mosaic_name):
                upipe.print_error(f"Mosaic cube file does not exist = {mosaic_name} \n"
                                  f"Aborting Image reconstruction")
                return
            upipe.print_info(f"Using mosaic cube {mosaic_name}")
            cube = MuseCube(filename=mosaic_name)
            cube.build_filterlist_images(filter_list=filter_list,
                                         prefix=f"{targetname}_IMAGE_FOV",
                                         suffix=suffixout,
                                         folder=folder_cubes)

    def init_combine(self, targetname=None, list_pointings=None, list_datasets=None,
                     folder_offset_table=None, name_offset_table=None,
                     **kwargs):
        """Prepare the combination of targets. The use can provide a pointing table providing a
        given selection.

        Input
        -----
        targetname: str [None]
            Name of target
        list_pointings: list [or None=default= all pointings]
            List of pointings (e.g., [1,2,3])
        name_offset_table: str
            Name of Offset table

        **kwargs: additional keywords including
            pointing_table, pointing_table_folder, pointing_table_format
        """
        log_filename = kwargs.pop("log_filename", "{0}_combine_{1}.log".format(targetname, version_pack))
        self.pipes_combine[targetname] = MusePointings(targetname=targetname,
                                                       list_pointings=list_pointings,
                                                       list_datasets=list_datasets,
                                                       rc_filename=self.targets[targetname].rc_filename,
                                                       cal_filename=self.targets[targetname].cal_filename,
                                                       folder_config=self.targets[targetname].folder_config,
                                                       name_offset_table=name_offset_table,
                                                       folder_offset_table=folder_offset_table,
                                                       log_filename=log_filename, **kwargs)

    def combine_target_per_pointing(self, targetname=None, wcs_from_pointing=True,
                                 **kwargs):
        """Run the combine recipe. Shortcut for combine[targetname].run_combine()
        """
        self.init_combine(targetname=targetname, **kwargs)
        self.pipes_combine[targetname].run_combine_all_single_pointings(
                wcs_from_pointing=wcs_from_pointing, **kwargs)

    def combine_target(self, targetname=None, **kwargs):
        """Run the combine recipe. Shortcut for combine[targetname].run_combine()
        """
        self.init_combine(targetname=targetname, **kwargs)
        self.pipes_combine[targetname].run_combine(**kwargs)

    def create_reference_wcs(self, targetname=None, pointings_wcs=True,
                             mosaic_wcs=True, reference_cube=True,
                             ref_wcs=None,
                             refcube_name=None, **kwargs):
        """Run the combine for individual exposures first building up
        a mask.
        """
        default_comb_folder = self.targets[targetname].combcubes_path
        folder_ref_wcs = kwargs.pop("folder_ref_wcs", default_comb_folder)
        self.init_combine(targetname=targetname, **kwargs)
        self.pipes_combine[targetname].create_reference_wcs(pointings_wcs=pointings_wcs,
                                                            mosaic_wcs=mosaic_wcs,
                                                            reference_cube=reference_cube,
                                                            refcube_name=refcube_name,
                                                            ref_wcs=ref_wcs,
                                                            folder_ref_wcs=folder_ref_wcs,
                                                            **kwargs)
