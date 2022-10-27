
# Licensed under a MIT license - see LICENSE

"""MUSE-PHANGS preparation recipe module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"

# Importing modules
import os
from os.path import join as joinpath

from copy import deepcopy

# Numpy
import numpy as np

# pymusepipe modules
from . import util_pipe as upipe
from .create_sof import SofPipe
from .align_pipe import create_offset_table, AlignMuseField
from . import musepipe
from .mpdaf_pipe import MuseSkyContinuum, MuseFilter
from .config_pipe import (mjd_names, get_suffix_product, dict_default_for_recipes,
                          dict_recipes_per_num, dict_recipes_per_name,
                          dict_files_iexpo_products, dict_files_products,
                          dict_products_scipost)

try :
    import astropy as apy
    from astropy.io import fits as pyfits
except ImportError :
    raise Exception("astropy is required for this module")

# =======================================================
# Few useful functions
# =======================================================
def add_listpath(suffix, paths) :
    """Add a suffix to a list of path
    and normalise them
    """
    newlist = []
    for mypath in paths:
        newlist.append(upipe.normpath(joinpath(suffix, mypath)))
    return newlist

def norm_listpath(paths) :
    """Normalise the path for a list of paths
    """
    newlist = []
    for mypath in paths:
        newlist.append(upipe.normpath(mypath))
    return newlist

import functools
def print_my_function_name(f):
    """Function to provide a print of the name of the function
    Can be used as a decorator
    """
    @functools.wraps(f)
    def wrapped(*myargs, **mykwargs):
        upipe.print_info("################   " + f.__name__ + "   ################")
        return f(*myargs, **mykwargs)
    return wrapped

def _get_combine_products(filter_list='white', prefix_all=""):
    """Provide a set of key output products depending on the filters
    for combine
    """
    name_products = []
    suffix_products = []
    suffix_prefinalnames = []
    prefix_products = []
    for prod in dict_products_scipost['cube']:
        if prod == "IMAGE_FOV":
            for i, value in enumerate(filter_list.split(','), start=1):
                suffix_products.append(f"_{i:04d}")
                suffix_prefinalnames.append(f"_{value}")
                name_products.append(prod)
                prefix_products.append(prefix_all)
        else :
            suffix_products.append("")
            suffix_prefinalnames.append("")
            name_products.append(prod)
            prefix_products.append(prefix_all)

    return name_products, suffix_products, suffix_prefinalnames, prefix_products

###################################################################
# Class for preparing the launch of recipes
###################################################################
class PipePrep(SofPipe) :
    """PipePrep class prepare the SOF files and launch the recipes
    """
    def __init__(self, first_recipe=1, last_recipe=None):
        """Initialisation of PipePrep
        """
        SofPipe.__init__(self)
#        super(PipePrep, self).__init__()
        self.first_recipe = first_recipe
        if last_recipe is None:
            self.last_recipe = np.max(list(dict_recipes_per_num.keys()))
        else:
            self.last_recipe = last_recipe

    def _get_tpl_meanmjd(self, gtable):
        """Get tpl of the group and mean mjd of the group
        """
        # This returns the first tpl of the group table
        tpl = gtable['tpls'][0]
        # This returns the mean mjd, using aggregate
        mean_mjd = gtable.groups.aggregate(np.mean)['mjd'].data[0]
        return tpl, mean_mjd

    def select_tpl_files(self, expotype=None, tpl="ALL", stage="raw"):
        """Selecting a subset of files from a certain type
        """
        if expotype not in musepipe.dict_expotypes:
            upipe.print_info("ERROR: input {0} is not in the list of possible values".format(expotype),
                    pipe=self)
            return 

        MUSE_subtable = self._get_table_expo(expotype, stage)
        if len(MUSE_subtable) == 0:
            if self.verbose :
                upipe.print_warning("Empty file table of type {0}".format(expotype),
                        pipe=self)
                upipe.print_warning("Returning an empty Table from the tpl -astropy- selection",
                        pipe=self)
            return MUSE_subtable

        group_table = MUSE_subtable.group_by('tpls')
        if tpl == "ALL":
            return group_table
        else :
            return group_table.groups[group_table.groups.keys['tpls'] == tpl]
        
    @staticmethod
    def print_recipes():
        """Printing the list of recipes
        """
        upipe.print_info("=============================================")
        upipe.print_info("The dictionary of recipes which can be run is")
        for key in dict_recipes_per_num:
            print("{0}: {1}".format(key, dict_recipes_per_num[key]))
        upipe.print_info("=============================================")

    @print_my_function_name
    def run_recipes(self, **kwargs):
        """Running all recipes in one shot

        Input
        -----
        fraction: float
            Fraction of sky to consider in sky frames for the sky spectrum
            Default is 0.8.
        illum: bool
            Default is True (use illumination during twilight calibration)
        skymethod: str
            Default is "model".
        filter_for_alignment: str
            Default is defined in config_pipe
        line: str
            Default is None as defined in config_pipe
        lambda_window: float
            Default is 10.0 as defined in config_pipe
        """
        class DefVal(object):
            def __init__(self):
                pass

        defval = DefVal()
        for key in dict_default_for_recipes:
            setattr(defval, key, kwargs.get(key, dict_default_for_recipes[key]))

        # We overwrite filter_for_alignment as it can be already set
        defval.filter_for_alignment = kwargs.pop("filter_for_alignment", self.filter_for_alignment)

        # Dictionary of arguments for each recipe
        default_dict_kwargs_recipes = {'twilight': {'illum': defval.illum},
                             'scibasic_all': {'illum': defval.illum},
                             'sky': {'fraction': defval.fraction},
                             'prep_align': {'skymethod': defval.skymethod, 
                                            'filter_for_alignment': defval.filter_for_alignment,
                                            'line': defval.line,
                                            'lambda_window': defval.lambda_window
                                            },
                             'scipost_per_expo': {'skymethod': defval.skymethod},
                             'scipost': {'skymethod': defval.skymethod}
                             }

        dict_kwargs_recipes = kwargs.pop("param_recipes", {})
        for key in default_dict_kwargs_recipes:
            if key in dict_kwargs_recipes:
                dict_kwargs_recipes[key].update(default_dict_kwargs_recipes[key])
            else:
                dict_kwargs_recipes[key] = default_dict_kwargs_recipes[key]

        # First and last recipe to be used
        first_recipe = kwargs.pop("first_recipe", self.first_recipe)
        last_recipe = kwargs.pop("last_recipe", self.last_recipe)
        if last_recipe is None: last_recipe = np.max(list(dict_recipes_per_num.keys()))

        # Transforming first and last if they are strings and not numbers
        if isinstance(first_recipe, str):
            if first_recipe not in dict_recipes_per_name:
                upipe.print_error("First recipe {} not in list of recipes".format(first_recipe))
                return
            first_recipe = dict_recipes_per_name[first_recipe]
        if isinstance(last_recipe, str):
            if last_recipe not in dict_recipes_per_name:
                upipe.print_error("Last recipe {} not in list of recipes".format(last_recipe))
                return
            last_recipe = dict_recipes_per_name[last_recipe]

        # Printing the info about which recipes will be used
        upipe.print_info("Data reduction from recipe {0} to {1}".format(
                            dict_recipes_per_num[first_recipe],
                            dict_recipes_per_num[last_recipe]))
        upipe.print_info("               [steps {0} - {1}]".format(
                            first_recipe, last_recipe))

        # Now doing the recipes one by one in order
        for ind in range(first_recipe, last_recipe + 1):
            recipe = dict_recipes_per_num[ind]
            name_recipe = "run_{}".format(recipe)
            # Including the kwargs for each recipe
            if recipe in dict_kwargs_recipes:
                kdic = dict_kwargs_recipes[recipe]
            else:
                kdic = {}
            getattr(self, name_recipe)(**kdic)

    @print_my_function_name
    def run_phangs_recipes(self, fraction=0.8, illum=True, skymethod="model",
                                **kwargs):
        """Running all PHANGS recipes in one shot
        Using the basic set up for the general list of recipes

        Input
        -----
        fraction: float
            Fraction of sky to consider in sky frames for the sky spectrum
            Default is 0.8.
        illum: bool
            Default is True (use illumination during twilight calibration)
        skymethod: str
            Default is "model".
        """
        self.run_recipes(fraction=fraction, skymethod=skymethod, illum=illum, **kwargs)

    @print_my_function_name
    def run_bias(self, sof_filename='bias', tpl="ALL", update=None):
        """Reducing the Bias files and creating a Master Bias
        Will run the esorex muse_bias command on all Biases

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='BIAS', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                upipe.print_error("[run_bias] No BIAS recovered from the astropy Table - Aborting",
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Create the dictionary for the BIASES including
        # the list of files to be processed for one MASTER BIAS
        # Adding the bad pixel table
        self._add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['BIAS'] = add_listpath(self.paths.rawfiles,
                    list(gtable['filename']))
            # extract the tpl (string)
            tpl = gtable['tpls'][0]
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Bias
            name_bias = get_suffix_product('BIAS')
            dir_bias = self._get_fullpath_expo('BIAS', "master")
            # Run the recipe
            self.recipe_bias(self.current_sof, dir_bias, name_bias, tpl)

        # Write the MASTER BIAS Table and save it
        self.save_expo_table('BIAS', tpl_gtable, "master", update=update)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

    @print_my_function_name
    def run_flat(self, sof_filename='flat', tpl="ALL", update=None):
        """Reducing the Flat files and creating a Master Flat
        Will run the esorex muse_flat command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='FLAT', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                upipe.print_error("[run_flat] No FLAT recovered from the astropy Table - Aborting",
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Create the dictionary for the FLATs including
        # the list of files to be processed for one MASTER Flat
        # Adding the bad pixel table
        self._add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['FLAT'] = add_listpath(self.paths.rawfiles,
                    list(gtable['filename']))
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self._get_tpl_meanmjd(gtable)
            # Adding the best tpc MASTER_BIAS
            self._add_tplmaster_to_sofdict(mean_mjd, 'BIAS')
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Flat and Trace Table
            dir_flat = self._get_fullpath_expo('FLAT', "master")
            name_flat = get_suffix_product('FLAT')
            dir_trace = self._get_fullpath_expo('TRACE', "master")
            name_tracetable = get_suffix_product('TRACE')
            # Run the recipe
            self.recipe_flat(self.current_sof, dir_flat, name_flat, dir_trace, name_tracetable, tpl)

        # Write the MASTER FLAT Table and save it
        self.save_expo_table('FLAT', tpl_gtable, "master", update=update)
        self.save_expo_table('TRACE', tpl_gtable, "master", update=update)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

    @print_my_function_name
    def run_wave(self, sof_filename='wave', tpl="ALL", update=None):
        """Reducing the WAVE-CAL files and creating the Master Wave
        Will run the esorex muse_wave command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='WAVE', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                upipe.print_error("[run_wave] No WAVE recovered from the astropy file Table - Aborting", 
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Create the dictionary for the FLATs including
        # the list of files to be processed for one WAVECAL
        # Adding Badpix table and Line Catalog
        self._add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        self._add_calib_to_sofdict("LINE_CATALOG")
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['ARC'] = add_listpath(self.paths.rawfiles,
                    list(gtable['filename']))
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self._get_tpl_meanmjd(gtable)
            # Finding the best tpl for BIAS + TRACE
            self._add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'TRACE'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Wave
            dir_wave = self._get_fullpath_expo('WAVE', "master")
            name_wave = get_suffix_product('WAVE')
            # Run the recipe
            self.recipe_wave(self.current_sof, dir_wave, name_wave, tpl)

        # Write the MASTER WAVE Table and save it
        self.save_expo_table('WAVE', tpl_gtable, "master", update=update)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

    @print_my_function_name
    def run_lsf(self, sof_filename='lsf', tpl="ALL", update=None):
        """Reducing the LSF files and creating the LSF PROFILE
        Will run the esorex muse_lsf command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='WAVE', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                upipe.print_error("[run_lsf] No WAVE recovered from the astropy file Table - Aborting", 
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        # Adding Badpix table and Line Catalog
        self._add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        self._add_calib_to_sofdict("LINE_CATALOG")
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['ARC'] = add_listpath(self.paths.rawfiles,
                    list(gtable['filename']))
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self._get_tpl_meanmjd(gtable)
            # Finding the best tpl for BIAS, TRACE, WAVE
            self._add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'TRACE', 'WAVE'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Wave
            dir_lsf = self._get_fullpath_expo('LSF', "master")
            name_lsf = get_suffix_product('LSF')
            # Run the recipe
            self.recipe_lsf(self.current_sof, dir_lsf, name_lsf, tpl)

        # Write the MASTER LSF PROFILE Table and save it
        self.save_expo_table('LSF', tpl_gtable, "master", update=update)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

    @print_my_function_name
    def run_twilight(self, sof_filename='twilight', tpl="ALL", update=None, illum=True):
        """Reducing the  files and creating the TWILIGHT CUBE.
        Will run the esorex muse_twilight command on all TWILIGHT

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='TWILIGHT', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                upipe.print_error("[run_twilight] No TWILIGHT recovered from the astropy file Table - Aborting", 
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        self._add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        self._add_calib_to_sofdict("VIGNETTING_MASK")
        for gtable in tpl_gtable.groups:
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self._get_tpl_meanmjd(gtable)
            self._add_geometry_to_sofdict(tpl, mean_mjd)
            # Provide the list of files to the dictionary
            self._sofdict['SKYFLAT'] = add_listpath(self.paths.rawfiles,
                    list(gtable['filename']))
            # Finding the best tpl for BIAS, FLAT, ILLUM, TRACE, WAVE
            if illum:
                self._add_tplraw_to_sofdict(mean_mjd, "ILLUM")
            self._add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'FLAT', 'TRACE', 'WAVE'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Names and folder of final Master Wave
            dir_twilight = self._get_fullpath_expo('TWILIGHT', "master")
            name_twilight = deepcopy(dict_files_products['TWILIGHT'])
            self.recipe_twilight(self.current_sof, dir_twilight, name_twilight, tpl)

        # Write the MASTER TWILIGHT Table and save it
        self.save_expo_table('TWILIGHT', tpl_gtable, "master", update=update)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

    @print_my_function_name
    def run_scibasic_all(self, list_object=['OBJECT', 'SKY', 'STD'], tpl="ALL",
                         illum=True, **kwargs):
        """Running scibasic for all objects in list_object
        Making different sof for each category
        """
        for expotype in list_object:
            sof_filename = 'scibasic_{0}'.format(expotype.lower())
            self.run_scibasic(sof_filename=sof_filename, expotype=expotype,
                              tpl=tpl, illum=illum, **kwargs)

    @print_my_function_name
    def run_scibasic(self, sof_filename='scibasic', expotype="OBJECT", tpl="ALL", illum=True,
                     update=True, overwrite=True):
        """Reducing the files of a certain category and creating the PIXTABLES
        Will run the esorex muse_scibasic 

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype=expotype, tpl=tpl, stage="raw")
        if len(tpl_gtable) == 0:
            if self.verbose :
                upipe.print_error("[run_scibasic] No {0} recovered from the astropy file Table - Aborting".format(expotype), 
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)
        dir_products = self._get_fullpath_expo(expotype, "processed")

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        for ntable, gtable in enumerate(tpl_gtable.groups):
            self._add_calib_to_sofdict("BADPIX_TABLE", reset=True)
            self._add_calib_to_sofdict("LINE_CATALOG")
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self._get_tpl_meanmjd(gtable)
            self._add_geometry_to_sofdict(tpl, mean_mjd)
            # Provide the list of files to the dictionary
            self._sofdict[expotype] = add_listpath(self.paths.rawfiles,
                    list(gtable['filename']))
            # Number of objects
            Nexpo = len(self._sofdict[expotype])
            if self.verbose:
                upipe.print_info("Number of expo is {Nexpo} for {expotype}".format(
                                 Nexpo=Nexpo, expotype=expotype), pipe=self)

            # Finding the best tpl for BIAS
            if illum:
                self._add_tplraw_to_sofdict(mean_mjd, "ILLUM") 
            self._add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'FLAT', 
                'TRACE', 'WAVE', 'TWILIGHT'])

            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)

            # Run the recipe to reduce the standard (muse_scibasic)
            suffix = get_suffix_product(expotype)
            name_products = []
            list_expo = np.arange(Nexpo).astype(np.int) + 1
            for iexpo in list_expo:
                name_products += ['{0:04d}-{1:02d}.fits'.format(iexpo, j+1) for j in range(24)]
            self.recipe_scibasic(self.current_sof, tpl, expotype, dir_products, name_products, suffix)

            # Write the Processed files Table and save it
            # Note: always overwrite the table so that a fresh numbering is done
            gtable['iexpo'] = list_expo
            if ntable == 0:
                self.save_expo_table(expotype, gtable, "processed", aggregate=False,
                                     update=update, overwrite=overwrite)
            else:
                self.save_expo_table(expotype, gtable, "processed", aggregate=False,
                                     update=update, overwrite=False)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

    @print_my_function_name
    def run_standard(self, sof_filename='standard', tpl="ALL", update=None,
                     overwrite=True):
        """Reducing the STD files after they have been obtained
        Running the muse_standard routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        std_table = self.select_tpl_files("STD", tpl=tpl, stage="processed")
        if len(std_table) == 0:
            if self.verbose :
                upipe.print_error("[run_standard] No processed STD recovered from the astropy file Table - Aborting",
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Create the dictionary for the STD Sof
        for i in range(len(std_table)):
            mytpl = std_table['tpls'][i]
            iexpo = np.int(std_table['iexpo'][i])
            if tpl != "ALL" and tpl != mytpl :
                continue
            # Now starting with the standard recipe
            self._add_calib_to_sofdict("EXTINCT_TABLE", reset=True)
            self._add_calib_to_sofdict("STD_FLUX_TABLE")
            self._sofdict['PIXTABLE_STD'] = [joinpath(self._get_fullpath_expo("STD", "processed"),
                'PIXTABLE_STD_{0}_{1:04d}-{2:02d}.fits'.format(mytpl, iexpo, j+1)) for j in range(24)]
            self.write_sof(sof_filename=sof_filename + "_" + mytpl, new=True)
            name_std = deepcopy(dict_files_products['STD'])
            dir_std = self._get_fullpath_expo('STD', "master")
            self.recipe_std(self.current_sof, dir_std, name_std, mytpl)

        # Write the MASTER files Table and save it
        self.save_expo_table("STD", std_table, "master", aggregate=False,
                             update=update, overwrite=overwrite)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

    @print_my_function_name
    def run_sky(self, sof_filename='sky', tpl="ALL", fraction=0.8,
                update=None, overwrite=True):
        """Reducing the SKY after they have been scibasic reduced
        Will run the esorex muse_create_sky routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        sky_table = self._get_table_expo("SKY", "processed")
        if len(sky_table) == 0:
            if self.verbose :
                upipe.print_warning("[run_sky] No SKY recovered from the astropy file Table - Aborting",
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        for i in range(len(sky_table)):
            mytpl = sky_table['tpls'][i]
            mymjd = sky_table['mjd'][i]
            iexpo = np.int(sky_table['iexpo'][i])
            if tpl != "ALL" and tpl != mytpl :
                continue
            # Now starting with the standard recipe
            self._add_calib_to_sofdict("EXTINCT_TABLE", reset=True)
            self._add_calib_to_sofdict("SKY_LINES")
            self._add_skycalib_to_sofdict("STD_RESPONSE", mymjd, 'STD')
            self._add_skycalib_to_sofdict("STD_TELLURIC", mymjd, 'STD')
            self._add_tplmaster_to_sofdict(mymjd, 'LSF')
            self._sofdict['PIXTABLE_SKY'] = [joinpath(self._get_fullpath_expo("SKY", "processed"),
                'PIXTABLE_SKY_{0}_{1:04d}-{2:02d}.fits'.format(mytpl, iexpo, j+1)) for j in range(24)]
            self.write_sof(sof_filename="{0}_{1}_{2:02d}".format(sof_filename, mytpl, iexpo), new=True)
            dir_sky = self._get_fullpath_expo('SKY', "processed")
            name_sky = deepcopy(dict_files_products['SKY'])
            self.recipe_sky(self.current_sof, dir_sky, name_sky, mytpl, iexpo, fraction)

        # Write the MASTER files Table and save it
        self.save_expo_table("SKY", sky_table, "processed", aggregate=False,
                             update=update, overwrite=overwrite)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

    @print_my_function_name
    def run_autocal_sky(self, sof_filename='scipost', expotype="SKY", 
            AC_suffix="_AC", tpl="ALL", **extra_kwargs):
        """Launch the scipost command to get individual exposures in a narrow
        band filter
        """
        # First selecting the files via the grouped table
        object_table = self._get_table_expo(expotype, "processed")

        for i in range(len(object_table)):
            iexpo = np.int(object_table['iexpo'][i])
            mytpl = object_table['tpls'][i]
            if tpl != "ALL" and tpl != mytpl :
                continue
            self.run_scipost(sof_filename=sof_filename, expotype=expotype,
                    tpl=mytpl, list_expo=[iexpo], save='autocal', 
                    offset_list=False, suffix=AC_suffix, autocalib='deepfield', 
                    **extra_kwargs)

    @print_my_function_name
    def run_prep_align(self, sof_filename='scipost', expotype="OBJECT", tpl="ALL", 
            line=None, suffix="", **extra_kwargs):
        """Launch the scipost command to get individual exposures in a narrow
        band filter
        """
        # First selecting the files via the grouped table
        object_table = self._get_table_expo("OBJECT", "processed")

        # Filter used for the alignment
        filter_for_alignment = extra_kwargs.pop("filter_for_alignment", self.filter_for_alignment)
        if self.verbose:
            upipe.print_info("Filter for alignment is {0}".format(
                filter_for_alignment), pipe=self)

        # Getting the band corresponding to the line
        lambda_window = extra_kwargs.pop("lambda_window", 10.0)
        [lmin, lmax] = upipe.get_emissionline_band(line=line, 
                                                   velocity=self.vsystemic, 
                                                   lambda_window=lambda_window)

        # Tag the suffix with the prealign suffix
        suffix = "{0}{1}".format(suffix, self._suffix_prealign)

        if line is not None: 
            suffix = "{0}_{1}".format(suffix, line)

        # Processing individual exposures to get the full cube and image
        for i in range(len(object_table)):
            iexpo = np.int(object_table['iexpo'][i])
            mytpl = object_table['tpls'][i]
            if tpl != "ALL" and tpl != mytpl :
                continue
            # Running scipost now on the individual exposure
            self.run_scipost(sof_filename=sof_filename, expotype=expotype,
                    tpl=mytpl, list_expo=[iexpo], suffix=suffix, 
                    lambdaminmax=[lmin, lmax], save='cube', 
                    offset_list=False, filter_list=filter_for_alignment,
                    **extra_kwargs)

    @print_my_function_name
    def run_check_align(self, name_offset_table, sof_filename='scipost', expotype="OBJECT", tpl="ALL",
            line=None, suffix="", folder_offset_table=None, **extra_kwargs):
        """Launch the scipost command to get individual exposures in a narrow
        band filter to check if the alignments are ok (after rotation 
        and using a given offset_table)
        """
        # First selecting the files via the grouped table
        object_table = self._get_table_expo("OBJECT", "processed")

        # Filter used for the alignment
        filter_for_alignment = extra_kwargs.pop("filter_for_alignment", self.filter_for_alignment)
        if self.verbose:
            upipe.print_info("Filter for alignment is {0}".format(
                filter_for_alignment), pipe=self)

        # Getting the band corresponding to the line
        lambda_window = extra_kwargs.pop("lambda_window", 10.0)
        [lmin, lmax] = upipe.get_emissionline_band(line=line, 
                                                   velocity=self.vsystemic, 
                                                   lambda_window=lambda_window)

        # Tag the suffix with the prealign suffix
        suffix = "{0}{1}".format(suffix, self._suffix_checkalign)

        if line is not None: 
            suffix = "{0}_{1}".format(suffix, line)

        # Processing individual exposures to get the full cube and image
        for i in range(len(object_table)):
            iexpo = np.int(object_table['iexpo'][i])
            mytpl = object_table['tpls'][i]
            if tpl != "ALL" and tpl != mytpl :
                continue
            # Running scipost now on the individual exposure
            self.run_scipost(sof_filename=sof_filename, expotype=expotype,
                    tpl=mytpl, list_expo=[iexpo], suffix=suffix, 
                    lambdaminmax=[lmin, lmax], save='cube', 
                    offset_list=True, filter_list=filter_for_alignment,
                    name_offset_table=name_offset_table,
                    folder_offset_table=folder_offset_table,
                    **extra_kwargs)

    @print_my_function_name
    def run_scipost_perexpo(self, sof_filename='scipost', expotype="OBJECT", 
                            tpl="ALL", stage="processed", 
                            suffix="", offset_list=False, **kwargs):
        """Launch the scipost command exposure per exposure

        Input
        -----
        See run_scipost parameters
        """
        # First selecting the files via the grouped table
        object_table = self._get_table_expo(expotype, stage)

        # Processing individual exposures to get the full cube and image
        for i in range(len(object_table)):
            iexpo = np.int(object_table['iexpo'][i])
            mytpl = object_table['tpls'][i]
            # Skip this exposure if tpl does not match
            if tpl != "ALL" and tpl != mytpl :
                continue
            # Running scipost now on the individual exposure
            self.run_scipost(sof_filename=sof_filename, expotype=expotype,
                    tpl=mytpl, list_expo=[iexpo], suffix=suffix,
                    offset_list=offset_list,
                    **kwargs)

    def _get_scipost_products(self, save='cube,skymodel', list_expo=[], filter_list=None):
        """Provide a set of key output products depending on the save mode
        for scipost
        """
        name_products = []
        suffix_products = []
        suffix_prefinalnames = []
        suffix_postfinalnames = []
        extlist_expo = []
        list_options = save.split(',')
        if filter_list is None: filter_list = self.filter_list
        if self.verbose:
            upipe.print_info("Filter list is {0}".format(filter_list), pipe=self)
        for option in list_options:
            for prod in dict_products_scipost[option]:
                if prod == "IMAGE_FOV":
                    for i, value in enumerate(filter_list.split(','), 
                            start=1):
                        name_products.append(prod)
                        suffix_products.append("_{0:04d}".format(i))
                        suffix_prefinalnames.append("_{0}".format(value))
                        if len(list_expo) == 1 :
                            suffix_postfinalnames.append("_{0:04d}".format(list_expo[0]))
                        else :
                            suffix_postfinalnames.append("")
                        extlist_expo.append(list_expo[0])

                elif any(x in prod for x in ['PIXTABLE', 'RAMAN', 'SKY', 'AUTOCAL']):
                    for i in range(len(list_expo)):
                        name_products.append(prod)
                        suffix_products.append("_{0:04d}".format(i+1))
                        suffix_prefinalnames.append("")
                        suffix_postfinalnames.append("_{0:04d}".format(list_expo[i]))
                        extlist_expo.append(i+1)
                else :
                    if "DATACUBE" in prod:
                        if len(list_expo) == 1 :
                            suffix_postfinalnames.append("_{0:04d}".format(list_expo[0]))
                        else :
                            suffix_postfinalnames.append("")
                    extlist_expo.append(list_expo[0])
                    name_products.append(prod)
                    suffix_products.append("")
                    suffix_prefinalnames.append("")

        if self._debug:
            upipe.print_info("Products for scipost: \n"
                    "Name_products: [{0}] \n"
                    "suffix_products: [{1}] \n"
                    "suffix_pre: [{2}] \n"
                    "suffix_post: [{3}] \n"
                    "Expo number: [{4}]".format(
                        name_products, suffix_products,
                        suffix_prefinalnames, suffix_postfinalnames, extlist_expo))
        return name_products, suffix_products, suffix_prefinalnames, \
                   suffix_postfinalnames, extlist_expo

    def _select_list_expo(self, expotype, tpl, stage, list_expo=[]):
        """Select the expo numbers which exists for a certain expotype
        """
        # First selecting the files via the grouped table
        tpl_table = self.select_tpl_files(expotype=expotype, tpl=tpl, stage=stage)

        # Selecting the table with the right iexpo
        if len(list_expo) == 0: 
            list_expo = tpl_table['iexpo'].data
        # First we isolate the unique values of iexpo from the table
        list_expo = np.unique(list_expo)
        # Then we select those who exist in the table
        # And don't forget to group the table by tpls
        group_table = tpl_table[np.isin(tpl_table['iexpo'], list_expo)].group_by('tpls')
        group_list_expo = []
        for gtable in group_table:
            group_list_expo.append(gtable['iexpo'].data)

        found_expo = True
        if len(group_table) == 0:
            found_expo = False
            if self.verbose :
                upipe.print_warning("[prep_recipes/_select_list_expo] No {0} recovered from the {1} astropy file "
                    "Table - Aborting".format(expotype, stage), pipe=self)

        return found_expo, list_expo, group_list_expo, group_table

    def _normalise_skycontinuum(self, mjd_expo, tpl_expo, iexpo, 
            suffix="", name_offset_table=None, **kwargs):
        """Create a normalised continuum, to be included in the sof file
        """
        stage = "processed"
        expotype = "SKY"
        # Finding the best tpl for this sky calib file type
        expo_table = self._get_table_expo(expotype, stage)
        index, this_tpl = self._select_closest_mjd(mjd_expo, expo_table) 
        if index < 0:
            upipe.print_info("[prep_recipes/_normalise_skycontinum/scipost] Failed to find an "
                             "exposure in the table - Aborting")
            return

        dir_calib = self._get_fullpath_expo(expotype, stage)

        # Adding the expo number of the SKY continuum
        iexpo_cont = expo_table[index]['iexpo']
        suffix += "_{0:04d}".format(iexpo_cont)

        # Name for the sky calibration file
        name_skycalib = "SKY_CONTINUUM_{0}{1}.fits".format(this_tpl, suffix)
        mycont = MuseSkyContinuum(joinpath(dir_calib, name_skycalib))

        # Check if normalise factor or from offset table background value
        normalise_factor = kwargs.pop("normalise_factor", None)
        status = 0
        if normalise_factor is None:
            # Checking input offset table and corresponding pixtables
            folder_offset_table = kwargs.pop("folder_offset_table", None)
            self._read_offset_table(name_offset_table, folder_offset_table)
            # Get the background value
            if mjd_names['table'] not in self.offset_table.columns:
                upipe.print_warning("No MJD column in offset table {0}".format(
                                     name_offset_table))
                status = -1
            else:    
                table_mjdobs = self.offset_table[mjd_names['table']]
                if (mjd_expo in table_mjdobs) and ('BACKGROUND' in self.offset_table.columns):
                    ind_table = np.argwhere(table_mjdobs == mjd_expo)[0]
                    background = self.offset_table['BACKGROUND'][ind_table[0]]
                else:
                    status = -2

            if status < 0:
                dict_err = {-1: "MJD", -2: "BACKGROUND"}
                upipe.print_error("Table {0} - {1}".format(folder_offset_table,
                                  name_offset_table))
                upipe.print_error("Could not find {0} value in offset table".format(
                                    dict_err[status]))
                upipe.print_warning("A background of 0 will be assumed, and")
                upipe.print_warning("a normalisation of 1 will be used for the SKY_CONTINUUM")
                return ""

            # Getting the muse filter for sky continuum
            filter_for_alignment = kwargs.pop("filter_for_alignment", self.filter_for_alignment)
            filter_fits_file = self._get_name_calibfile("FILTER_LIST")
            mymusefilter = MuseFilter(filter_name=filter_for_alignment,
                                      filter_fits_file=filter_fits_file)
            # Find the background value from the offset table
            # Integration of the continuum in that filter
            thisfile_musemode = expo_table[0]['mode']
            upipe.print_info("For this sky continuum, we use MUSE Mode = {0}".format(thisfile_musemode))
            mycont.integrate(mymusefilter, ao_mask=("-AO-" in thisfile_musemode))
            mycont.get_normfactor(background, filter_for_alignment)
            normalise_factor = getattr(mycont, filter_for_alignment).norm

        # Returning the name of the normalised continuum
        prefix_skycont = "norm_{0}_{1:04d}_".format(tpl_expo, iexpo)
        mycont.save_normalised(normalise_factor, prefix=prefix_skycont, overwrite=True)

        return prefix_skycont

    @print_my_function_name
    def run_scipost_sky(self):
        """Run scipost for the SKY with no offset list and no skymethod
        """
        self.run_scipost(expotype="SKY", offset_list=False, skymethod='none')

    @print_my_function_name
    def run_scipost(self, sof_filename='scipost', expotype="OBJECT", tpl="ALL", 
            stage="processed", list_expo=[], 
            lambdaminmax=[4000.,10000.], suffix="", **kwargs):
        """Scipost treatment of the objects
        Will run the esorex muse_scipost routine

        Input
        -----
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time
        list_expo: list of integers
            Exposure numbers. By default, an empty list which means that all
            exposures will be used.
        lambdaminmax: tuple of 2 floats
            Minimum and Maximum wavelength to pass to the muse_scipost recipe
        suffix: str
            Suffix to add to the input pixtables.

        norm_skycontinuum: bool
            Normalise the skycontinuum or not. Default is False.
            If normalisation is to be done, it will use the offset_table
            and the tabulated background value to renormalise the sky 
            continuum.
        skymethod: str
            Type of skymethod. See MUSE manual.
        offset_list: bool
            If True, using an OFFSET list. Default is True.
        name_offset_table: str
            Name of the offset table table. If not provided, will use the 
            default name produced during the pipeline run.
        filter_for_alignment: str
            Name of the filter used for alignment. 
            Default is self.filter_for_alignment
        filter_list: str
            List of filters to be considered for reconstructed images.
            By Default will use the list in self.filter_list.
        """
        # Selecting the table with the right iexpo
        found_expo, list_expo, group_list_expo, scipost_table = self._select_list_expo(expotype, tpl, stage, list_expo) 
        if not found_expo:
            return

        # Lambda min and max?
        [lambdamin, lambdamax] = lambdaminmax
        # Skymethod
        skymethod = kwargs.pop("skymethod", "model")

        # Allow the use of None to work and be replaced by "none"
        if skymethod is None:
            skymethod = "none"
        # Save options
        if skymethod != "none":
            save = kwargs.pop("save", "cube,skymodel,individual")
        else :
            # If skymethod is none, no need to save the skymodel...
            save = kwargs.pop("save", "cube,individual")

        # Normalisation of the sky continuum
        norm_skycontinuum = kwargs.pop("norm_skycontinuum", False)

        # Filters
        filter_for_alignment = kwargs.pop("filter_for_alignment", self.filter_for_alignment)
        filter_list = kwargs.pop("filter_list", self.filter_list)

        # Offsets
        offset_list = kwargs.pop("offset_list", False)
        name_offset_table = kwargs.pop("name_offset_table", None)
        folder_offset_table = kwargs.pop("folder_offset_table", None)
        if offset_list:
            upipe.print_info("Will use offset table: {0} in {1}".format(
                name_offset_table, folder_offset_table))

        # Misc parameters - autocalib, barycentric correction, AC
        autocalib = kwargs.pop("autocalib", "none")
        rvcorr = kwargs.pop("rvcorr", "bary")
        if rvcorr != "bary":
            upipe.print_warning("Scipost will use '{0}' as barycentric "
                                "correction [Options are bary, helio, geo, none]".format(rvcorr),
                                pipe=self)
        AC_suffix = kwargs.pop("AC_suffix", "_AC")

        # WCS
        ref_wcs = kwargs.pop("ref_wcs", None)
        if ref_wcs is not None:
            folder_ref_wcs = kwargs.pop("folder_ref_wcs", "")
        dir_products = kwargs.pop("dir_products",
                                  self._get_fullpath_expo(expotype, "processed"))
        prefix_all = kwargs.pop("prefix_all", "")

        # Continuum correction
        default_suffix_skycontinuum = ""
        suffix_skycontinuum = kwargs.pop("suffix_skycont", default_suffix_skycontinuum)
        if skymethod != "none":
            if suffix_skycontinuum != default_suffix_skycontinuum:
                upipe.print_warning("Scipost will use '{0}' as suffix "
                        "for the SKY_CONTINUUM files".format(suffix_skycontinuum),
                        pipe=self)

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Create the dictionary for scipost
        # Selecting either one or all of the exposures
        for gtable in scipost_table.groups:
            # Getting the expo list
            list_group_expo = gtable['iexpo'].data
            # Adding the expo number if only 1 exposure is considered
            if len(list_group_expo) == 1:
                suffix_iexpo = "_{0:04d}".format(list_group_expo[0])
            else :
                suffix_iexpo = ""

            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self._get_tpl_meanmjd(gtable)
            # Now starting with the standard recipe
            self._add_calib_to_sofdict("EXTINCT_TABLE", reset=True)
            self._add_calib_to_sofdict("FILTER_LIST")
            if autocalib == "user":
                self._add_skycalib_to_sofdict("AUTOCAL_FACTORS", mean_mjd, 'SKY', 
                        "processed", suffix=AC_suffix)
            self._add_astrometry_to_sofdict(tpl, mean_mjd)
            self._add_skycalib_to_sofdict("STD_RESPONSE", mean_mjd, 'STD')
            self._add_skycalib_to_sofdict("STD_TELLURIC", mean_mjd, 'STD')
            if ref_wcs is not None:
                self._sofdict['OUTPUT_WCS'] = [joinpath(folder_ref_wcs, ref_wcs)]

            if name_offset_table is None:
                folder_offset_table = self._get_fullpath_expo(expotype, "processed")
                name_offset_table = '{0}{1}_{2}_{3}.fits'.format(
                                       dict_files_products['ALIGN'][0],
                                       suffix, filter_for_alignment, tpl)
            else:
                if folder_offset_table is None:
                    folder_offset_table = self.paths.alignment
            if offset_list :
                self._sofdict['OFFSET_LIST'] = [joinpath(folder_offset_table, name_offset_table)]

            # The sky subtraction method on the sky continuum to normalise it
            # But only if requested
            if norm_skycontinuum:
                if len(list_group_expo) > 1:
                    upipe.print_warning("More than 1 exposure in group table (scipost)")
                    upipe.print_warning("The sky continuum will be "
                                        "normalised according to the first exposure")
                prefix_skycontinuum = self._normalise_skycontinuum(mjd_expo=mean_mjd, 
                        tpl_expo=tpl, iexpo=list_group_expo[0], 
                        suffix=suffix_skycontinuum, 
                        folder_offset_table=folder_offset_table,
                        name_offset_table=name_offset_table)
            else:
                prefix_skycontinuum = ""
            if skymethod != "none":
                self._add_calib_to_sofdict("SKY_LINES")
                self._add_skycalib_to_sofdict("SKY_CONTINUUM", mean_mjd, 'SKY', 
                        "processed", suffix=suffix_skycontinuum, 
                        prefix=prefix_skycontinuum, perexpo=True)
            self._add_tplmaster_to_sofdict(mean_mjd, 'LSF')

            # Selecting only exposures to be treated
            # We need to force 'OBJECT' to make sure scipost will deal with the exposure
            # even if it is e.g., a SKY
            pixtable_name = get_suffix_product('OBJECT')
            pixtable_name_thisone = get_suffix_product(expotype)
            self._sofdict[pixtable_name] = []
            for iexpo in list_group_expo:
               self._sofdict[pixtable_name] += [joinpath(self._get_fullpath_expo(expotype, "processed"),
                   '{0}_{1}_{2:04d}-{3:02d}.fits'.format(
                       pixtable_name_thisone, tpl, iexpo, j+1)) for j in range(24)]

            # Adding the suffix_iexpo to the end of the name if needed
            # This is the expo number if only 1 exposure is present in the list
            self.write_sof(sof_filename="{0}_{1}{2}_{3}{4}".format(sof_filename, expotype, 
                suffix, tpl, suffix_iexpo), new=True)
            # products
            name_products, suffix_products, suffix_prefinalnames, suffix_postfinalnames, fl_expo = \
                self._get_scipost_products(save, list_group_expo, filter_list)
            self.recipe_scipost(self.current_sof, tpl, expotype, dir_products, 
                    name_products, suffix_products, suffix_prefinalnames, 
                    suffix_postfinalnames, suffix=suffix, 
                    lambdamin=lambdamin, lambdamax=lambdamax, save=save, 
                    filter_list=filter_list, autocalib=autocalib, rvcorr=rvcorr, 
                    skymethod=skymethod, filter_for_alignment=filter_for_alignment,
                    list_expo=fl_expo, prefix_all=prefix_all,
                    **kwargs)

            # Write the MASTER files Table and save it
            if len(list_expo) == 1: suffix_expo = "_{0:04d}".format(list_expo[0])
            else: suffix_expo = ""
            self.save_expo_table(expotype, scipost_table, "reduced", 
                    "IMAGES_FOV{0}_{1}{2}_{3}_list_table.fits".format(
                        suffix, expotype, suffix_expo, tpl), 
                        aggregate=False, overwrite=True)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

    @print_my_function_name
    def run_align_bygroup(self, sof_filename='exp_align_bygroup', expotype="OBJECT", 
            list_expo=[], stage="processed", line=None, suffix="",
            tpl="ALL", **kwargs):
        """Aligning the individual exposures from a dataset
        using the emission line region 
        With the muse exp_align routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # Selecting the table with the right iexpo
        found_expo, list_expo, group_list_expo, align_table = self._select_list_expo(
                expotype, tpl, stage, list_expo) 
        if not found_expo:
            return
        
        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Setting the default alignment filter
        filter_for_alignment = kwargs.pop("filter_for_alignment", self.filter_for_alignment)
        if line is not None:
            suffix += "_{0}".format(line)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        for gtable in align_table.groups:
            # extract the tpl (string) and mean mjd (float) 
            mytpl, mymjd = self._get_tpl_meanmjd(gtable)
            # Now starting with the standard recipe
            self._sofdict.clear()
            list_group_expo = gtable['iexpo'].data

            # Skip this group if only 1 or fewer exposures 
            # exp_align needs at least 2
            if len(list_group_expo) <= 1:
                if self.verbose:
                    upipe.print_warning("Group = {0}".format(mytpl),
                            pipe=self)
                    upipe.print_warning("No derived OFFSET LIST as only 1 exposure retrieved in this group",
                            pipe=self)
                continue
            long_suffix = "{0}_{1}_{2}".format(suffix, filter_for_alignment, mytpl)
            long_suffix_align = "{0}{1}_{2}_{3}".format(suffix, self._suffix_prealign,
                                 filter_for_alignment, mytpl)
            list_images = [joinpath(self._get_fullpath_expo("OBJECT", "processed"),
                                          'IMAGE_FOV{0}_{1:04d}.fits'.format(
                                              long_suffix_align, iexpo)) 
                                              for iexpo in list_group_expo]
            self._sofdict['IMAGE_FOV'] = list_images
            create_offset_table(list_images, table_folder=self.paths.pipe_products, 
                                table_name="{0}.fits".format(dict_files_products['ALIGN'][0]))
            upipe.print_info("Creating empty OFFSET_LIST.fits using images list", 
                    pipe=self)
            self.write_sof(sof_filename=sof_filename + long_suffix, new=True)
            dir_align = self._get_fullpath_expo('OBJECT', "processed")
            namein_align = deepcopy(dict_files_products['ALIGN'])
            nameout_align = [name + long_suffix for name in deepcopy(dict_files_products['ALIGN'])]
            for iter_file in dict_files_iexpo_products['ALIGN']:
                for iexpo in list_group_expo:
                    namein_align.append('{0}_{1:04d}'.format(iter_file, iexpo))
                    nameout_align.append('{0}_bygroup{1}_{2:04d}'.format(iter_file, long_suffix, iexpo))
            self.recipe_align(self.current_sof, dir_align, namein_align, nameout_align, mytpl, "group", **kwargs)

            # Write the MASTER files Table and save it
            self.save_expo_table(expotype, align_table, "reduced", 
                    "ALIGNED_IMAGES_BYGROUP_{0}{1}_list_table.fits".format(expotype, 
                        long_suffix), aggregate=False, update=True)
        
        # Go back to original folder
        self.goto_prevfolder(addtolog=True)
        
    @print_my_function_name
    def run_align_bydataset(self, sof_filename='exp_align_bydataset', expotype="OBJECT",
            list_expo=[], stage="processed", line=None, suffix="",
            tpl="ALL", **kwargs):
        """Aligning the individual exposures from a dataset
        using the emission line region 
        With the muse exp_align routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # Selecting the table with the right iexpo
        found_expo, list_expo, group_list_expo, align_table = self._select_list_expo(
                                      expotype, tpl, stage, list_expo) 
        if not found_expo:
            return
        # Stop the process if only 1 or fewer exposures are retrieved
        if len(align_table) <= 1:
            if self.verbose:
                upipe.print_warning("Dataset = {0}".format(self.dataset),
                        pipe=self)
                upipe.print_warning("No derived OFFSET LIST as only "
                                    "1 exposure retrieved in this Dataset",
                        pipe=self)
            return
        
        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Setting the default alignment filter
        filter_for_alignment = kwargs.pop("filter_for_alignment", self.filter_for_alignment)
        if line is not None:
            suffix += "_{0}".format(line)

        # Use the dataset as a suffix for the names
        obname = self._get_obname()

        # Now creating the SOF file, first reseting it
        self._sofdict.clear()
        # Producing the list of IMAGES and add them to the SOF
        long_suffix = "{0}_{1}".format(suffix, filter_for_alignment)
        long_suffix_align = "{0}{1}_{2}".format(suffix, self._suffix_prealign,
                             filter_for_alignment)
        list_images = [joinpath(self._get_fullpath_expo("OBJECT", "processed"),
                       'IMAGE_FOV{0}_{1}_{2:04d}.fits'.format(
                          long_suffix_align, row['tpls'], row['iexpo'])) 
                          for row in align_table]
        self._sofdict['IMAGE_FOV'] = list_images
        upipe.print_info("Creating empty OFFSET_LIST.fits using images list", pipe=self)
        create_offset_table(list_images, table_folder=self.paths.pipe_products, 
                            table_name="{0}.fits".format(dict_files_products['ALIGN'][0]))

        # Write the SOF
        self.write_sof(sof_filename=sof_filename + "{0}_{1}_{2}_{3}".format(
                       long_suffix, expotype, obname, tpl), new=True)
        dir_align = self._get_fullpath_expo('OBJECT', "processed")
        namein_align = deepcopy(dict_files_products['ALIGN'])
        nameout_align = [name + "{0}_{1}_{2}_{3}".format(
                         long_suffix, expotype, obname, tpl)
                             for name in deepcopy(dict_files_products['ALIGN'])]
        for iter_file in dict_files_iexpo_products['ALIGN']:
            for i, row in enumerate(align_table):
                namein_align.append('{0}_{1:04d}'.format(iter_file, i+1))
                nameout_align.append('{0}_bydataset{1}_{2}_{3:04d}'.format(
                    iter_file, long_suffix, row['tpls'], row['iexpo']))

        # Find the alignment - Note that Field is used and tpl reflects the given selection
        self.recipe_align(self.current_sof, dir_align, namein_align, nameout_align, 
                tpl, obname, **kwargs)

        # Write the MASTER files Table and save it
        self.save_expo_table(expotype, align_table, "reduced", 
                "ALIGNED_IMAGES_BYDATASET_{0}{1}_{2}_{3}_list_table.fits".format(expotype,
                long_suffix, obname, tpl), aggregate=False, update=True)
        
        # Go back to original folder
        self.goto_prevfolder(addtolog=True)

    @print_my_function_name
    def save_fine_alignment(self, name_offset_table=None):
        """ Save the fine dataset alignment
        """
        if name_offset_table is None:
            name_offset_table = "XX"

        # Use the save from the alignment module
        tstamp = self.dict_alignments.present_tstamp
        self.dict_alignments[tstamp].save(name_offset_table)

    @print_my_function_name
    def run_fine_alignment(self, name_ima_reference=None, nexpo=1, list_expo=[],
                           line=None, bygroup=False, reset=False, **kwargs):
        """Run the alignment on this dataset using or not a reference image
        """
        # If not yet initialised, build the dictionary
        if not hasattr(self, "dict_alignments"):
            # Create an empty dictionary with a None timestamp
            self.dict_alignments = upipe.TimeStampDict()
            # Reset to True to initialise the structure
            reset = True

        # Setting the default alignment filter
        filter_for_alignment = kwargs.pop("filter_for_alignment", self.filter_for_alignment)

        # If reset, check if list is not empty
        # If not empty, create new time stamp and proceed with initialisation
        if reset:
            # Create a new set of alignments
            self.get_align_group(name_ima_reference=name_ima_reference, 
                    list_expo=list_expo, line=line, 
                    filter_for_alignment=filter_for_alignment, bygroup=bygroup)
            # if dictionary is empty, it creates the first timestamp
            self.dict_alignments.create_new_timestamp(self.align_group)

        # Run the alignment
        tstamp = self.dict_alignments.present_tstamp
        self.dict_alignments[tstamp].run(nexpo)
        
    @print_my_function_name
    def get_align_group(self, name_ima_reference=None, list_expo=[], line=None, 
            suffix="", bygroup=False, **kwargs):
        """Extract the needed information for a set of exposures to be aligned
        """
        # Selecting the table with the right iexpo
        found_expo, list_expo, group_list_expo, align_table = self._select_list_expo("OBJECT", 
                "ALL", "processed", list_expo) 
        if not found_expo:
            if self.verbose:
                upipe.print_warning("No exposure recovered for the fine alignment",
                        pipe=self)
            return
            
        # Initialise the list of Groups to be aligned
        self.align_group = []
        # Setting the default alignment filter
        filter_for_alignment = kwargs.pop("filter_for_alignment", self.filter_for_alignment)
        if line is not None:
            suffix += "_{0}".format(line)

        # Option to do a per group alignment
        if bygroup:
            for gtable in align_table.groups:
                mytpl, mymjd = self._get_tpl_meanmjd(gtable)
                list_group_expo = gtable['iexpo'].data
                long_suffix = "{0}_{1}_{2}".format(suffix, filter_for_alignment, mytpl)
                list_names_muse = [joinpath(self._get_fullpath_expo("OBJECT", "processed"),
                    'IMAGE_FOV{0}_{1:04d}.fits'.format(long_suffix, iexpo)) 
                    for iexpo in list_group_expo]
                self.align_group.append(AlignMuseField(name_ima_reference, list_names_muse,
                                                   flag="mytpl"))
        # If not by group
        else:
            long_suffix = "{0}_{1}".format(suffix, filter_for_alignment)
            list_names_muse = [joinpath(self._get_fullpath_expo("OBJECT", "processed"),
                'IMAGE_FOV{0}_{1}_{2:04d}.fits'.format(long_suffix, row['tpls'], row['iexpo'])) 
                for row in align_table]
            # Giving a reference image name, doing the alignment w.r.t. to that specific image
            if name_ima_reference is None:
                upipe.print_warning("Using the first MUSE exposure as a reference")
                self.align_group.append(AlignMuseField(list_names_muse[0],
                    list_names_muse[1:], flag="mytpl"))
            else:
                self.align_group.append(AlignMuseField(name_ima_reference,
                    list_names_muse, flag="mytpl"))

    @print_my_function_name
    def run_combine_dataset(self, sof_filename='exp_combine', expotype="OBJECT",
            list_expo=[], stage="processed", tpl="ALL", 
            lambdaminmax=[4000.,10000.], suffix="", **kwargs):
        """Produce a cube from all frames in the dataset
        list_expo or tpl specific arguments can still reduce the selection if needed
        """
        # Selecting the table with the right iexpo
        found_expo, list_expo, group_list_expo, combine_table = self._select_list_expo(expotype, tpl, stage, list_expo) 
        if not found_expo:
            return

        # Lambda min and max?
        [lambdamin, lambdamax] = lambdaminmax

        # Abort if only one exposure is available
        # exp_combine needs a minimum of 2
        if len(list_expo) <= 1:
            if self.verbose:
                upipe.print_warning("The combined dataset has only one exposure: process aborted",
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, addtolog=True)

        # Setting the default alignment filter
        filter_for_alignment = kwargs.pop("filter_for_alignment", self.filter_for_alignment)
        filter_list = kwargs.pop("filter_list", self.filter_list)
        prefix_all = kwargs.pop("prefix_all", "")

        # Use the dataset as a suffix for the names
        obname = self._get_obname()
        # Save option
        save = kwargs.pop("save", "cube,combined")

        # Now creating the SOF file, first reseting it
        self._sofdict.clear()
        # Selecting only exposures to be treated
        # Producing the list of REDUCED PIXTABLES
        self._add_calib_to_sofdict("FILTER_LIST")
        pixtable_name = get_suffix_product('REDUCED')
        pixtable_name_thisone = dict_products_scipost['individual']

        # Setting the default option of offset_list
        # And looking for that table, adding it to the sof file
        offset_list = kwargs.pop("offset_list", True)
        folder_expo = self._get_fullpath_expo(expotype, stage)
        long_suffix = "{0}_{1}".format(suffix, filter_for_alignment)
        if offset_list:
            offset_list_tablename = kwargs.pop("offset_list_tablename", None)
            if offset_list_tablename is None:
                offset_list_tablename = "{0}{1}_{2}_{3}_{4}.fits".format(
                        dict_files_products['ALIGN'][0], long_suffix,
                        expotype, obname, tpl)
            if not os.path.isfile(joinpath(folder_expo, offset_list_tablename)):
                upipe.print_error("OFFSET_LIST table {0} not found in folder {1}".format(
                        offset_list_tablename, folder_expo), pipe=self)

            self._sofdict['OFFSET_LIST'] = [joinpath(folder_expo, offset_list_tablename)]

        self._sofdict[pixtable_name] = []
        for prod in pixtable_name_thisone:
            self._sofdict[pixtable_name] += [joinpath(folder_expo,
                '{0}_{1}_{2:04d}.fits'.format(prod, row['tpls'], row['iexpo'])) for row in
                combine_table]
        self.write_sof(sof_filename="{0}_{1}{2}_{3}".format(sof_filename, expotype, 
            long_suffix, tpl), new=True)

        # Product names
        dir_products = kwargs.pop("dir_products",
                                  self._get_fullpath_expo(expotype, stage))
        name_products, suffix_products, suffix_prefinalnames, prefix_products = \
                _get_combine_products(filter_list, prefix_all=prefix_all) 

        # Combine the exposures 
        self.recipe_combine(self.current_sof, dir_products, name_products, 
                tpl, expotype, suffix_products=suffix_products,
                suffix_prefinalnames=suffix_prefinalnames,
                prefix_products=prefix_products,
                lambdamin=lambdamin, lambdamax=lambdamax,
                save=save, suffix=suffix, filter_list=filter_list, **kwargs)

        # Go back to original folder
        self.goto_prevfolder(addtolog=True)
