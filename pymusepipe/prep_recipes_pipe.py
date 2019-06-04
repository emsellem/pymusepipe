# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS preparation recipe module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# Importing modules
import os
from os.path import join as joinpath

import copy
from copy import deepcopy

# Numpy
import numpy as np

# pymusepipe modules
from pymusepipe import util_pipe as upipe
from pymusepipe.create_sof import SofPipe
from pymusepipe import musepipe

# =======================================================
# List of recipes
# =======================================================
list_recipes = ['bias', 'flat', 'wave', 'lsf', 
        'twilight', 'scibasic_all', 'std', 'sky']
        
dic_files_iexpo_products = {
        'ALIGN': ['SOURCE_LIST']
        }

dic_files_products = {
        'STD': ['DATACUBE_STD', 'STD_FLUXES', 
            'STD_RESPONSE', 'STD_TELLURIC'],
        'TWILIGHT': ['DATACUBE_SKYFLAT', 'TWILIGHT_CUBE'],
        'SKY': ['SKY_MASK', 'SKY_IMAGE', 'SKY_LINES', 'SKY_SPECTRUM', 
            'SKY_CONTINUUM'],
        'ALIGN': ['OFFSET_LIST']
        }

dic_products_scipost = {
        'cube': ['DATACUBE_FINAL', 'IMAGE_FOV'],
        'individual': ['PIXTABLE_REDUCED'],
        'stacked': ['OBJECT_RESAMPLED'],
        'positioned': ['PIXTABLE_POSITIONED'],
        'combined': ['PIXTABLE_COMBINED'],
        'skymodel': ['SKY_MASK', 'SKY_SPECTRUM', 
            'SKY_LINES', 'SKY_IMAGE'],
        'raman': ['RAMAN_IMAGES'],
        'autocal': ['AUTOCAL_FACTORS']
        }
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
        upipe.print_info(f.__name__)
        return f(*myargs, **mykwargs)
    return wrapped

def _get_combine_products(filter_list='white,Cousins_R'):
    """Provide a set of key output products depending on the filters
    for combine
    """
    name_products = []
    suffix_products = []
    suffix_prefinalnames = []
    for prod in dic_products_scipost['cube']:
        if prod == "IMAGE_FOV":
            for i, value in enumerate(filter_list.split(','), start=1):
                suffix_products.append("_{0:04d}".format(i))
                suffix_prefinalnames.append("_{0}".format(value))
                name_products.append(prod)
        else :
            suffix_products.append("")
            suffix_prefinalnames.append("")
            name_products.append(prod)

    return name_products, suffix_products, suffix_prefinalnames


###################################################################
# Class for preparing the launch of recipes
###################################################################
class PipePrep(SofPipe) :
    """PipePrep class prepare the SOF files and launch the recipes
    """
    def __init__(self):
        """Initialisation of PipePrep
        """
        SofPipe.__init__(self)
#        super(PipePrep, self).__init__()
        self.list_recipes = deepcopy(list_recipes)

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
        if expotype not in musepipe.listexpo_types.keys() :
            upipe.print_info("ERROR: input {0} is not in the list of possible values".format(expotype))
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
        
    @print_my_function_name
    def run_all_recipes(self, fraction=0.8, illum=True, bypointing=True):
        """Running all recipes in one shot
        """
        #for recipe in self.list_recipes:
        self.run_bias()
        self.run_flat()
        self.run_wave()
        self.run_lsf()
        self.run_twilight(illum=illum)
        self.run_scibasic_all(illum=illum)
        self.run_standard()
        self.run_sky(fraction=fraction)
        self.run_prep_align()
        self.run_align_bypointing()
        self.run_align_bygroup()
        self.run_scipost()
        self.run_scipost(expotype="SKY", offset_list=False, skymethod='none')
        self.run_combine_pointing()

    @print_my_function_name
    def run_all_phangs_recipes(self, fraction=0.8, illum=True, bypointing=True):
        """Running all recipes in one shot
        """
        #for recipe in self.list_recipes:
        self.run_bias()
        self.run_flat()
        self.run_wave()
        self.run_lsf()
        self.run_twilight(illum=illum)
        self.run_scibasic_all(illum=illum)
        self.run_standard()
        self.run_sky(fraction=fraction)
        self.run_prep_align()
        self.run_align_bypointing()
        self.run_align_bygroup()
        self.run_scipost()
        self.run_scipost(expotype="SKY", offset_list=False, skymethod='none')
        self.run_combine_pointing()

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
                upipe.print_warning("No BIAS recovered from the astropy Table - Aborting",
                        pipe=self)
                return
        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

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
            name_bias = self._get_suffix_product('BIAS')
            dir_bias = self._get_fullpath_expo('BIAS', "master")
            # Run the recipe
            self.recipe_bias(self.current_sof, dir_bias, name_bias, tpl)

        # Write the MASTER BIAS Table and save it
        self.save_expo_table('BIAS', tpl_gtable, "master", update=update)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

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
                upipe.print_warning("No FLAT recovered from the astropy Table - Aborting",
                        pipe=self)
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

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
            name_flat =  self._get_suffix_product('FLAT')

            dir_trace = self._get_fullpath_expo('TRACE', "master")
            name_tracetable = self._get_suffix_product('TRACE')
            # Run the recipe
            self.recipe_flat(self.current_sof, dir_flat, name_flat, dir_trace, name_tracetable, tpl)

        # Write the MASTER FLAT Table and save it
        self.save_expo_table('FLAT', tpl_gtable, "master", update=update)
        self.save_expo_table('TRACE', tpl_gtable, "master", update=update)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

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
                upipe.print_warning("No WAVE recovered from the astropy file Table - Aborting", 
                        pipe=self)
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

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
            name_wave = self._get_suffix_product('WAVE')
            # Run the recipe
            self.recipe_wave(self.current_sof, dir_wave, name_wave, tpl)

        # Write the MASTER WAVE Table and save it
        self.save_expo_table('WAVE', tpl_gtable, "master", update=update)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

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
                upipe.print_warning("No WAVE recovered from the astropy file Table - Aborting", 
                        pipe=self)
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

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
            name_lsf = self._get_suffix_product('LSF')
            # Run the recipe
            self.recipe_lsf(self.current_sof, dir_lsf, name_lsf, tpl)

        # Write the MASTER LSF PROFILE Table and save it
        self.save_expo_table('LSF', tpl_gtable, "master", update=update)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

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
                upipe.print_warning("No TWILIGHT recovered from the astropy file Table - Aborting", 
                        pipe=self)
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        self._add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        self._add_calib_to_sofdict("VIGNETTING_MASK")
        for gtable in tpl_gtable.groups:
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self._get_tpl_meanmjd(gtable)
            self._add_geometry_to_sofdict(tpl)
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
            name_twilight = deepcopy(dic_files_products['TWILIGHT'])
            self.recipe_twilight(self.current_sof, dir_twilight, name_twilight, tpl)

        # Write the MASTER TWILIGHT Table and save it
        self.save_expo_table('TWILIGHT', tpl_gtable, "master", update=update)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    @print_my_function_name
    def run_scibasic_all(self, list_object=['OBJECT', 'SKY', 'STD'], tpl="ALL", illum=True):
        """Running scibasic for all objects in list_object
        Making different sof for each category
        """
        for expotype in list_object:
            sof_filename = 'scibasic_{0}'.format(expotype.lower())
            self.run_scibasic(sof_filename=sof_filename, expotype=expotype, tpl=tpl, illum=illum)

    @print_my_function_name
    def run_scibasic(self, sof_filename='scibasic', expotype="OBJECT", tpl="ALL", illum=True):
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
                upipe.print_warning("No {0} recovered from the astropy file Table - Aborting".format(expotype), 
                        pipe=self)
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        for gtable in tpl_gtable.groups:
            self._add_calib_to_sofdict("BADPIX_TABLE", reset=True)
            self._add_calib_to_sofdict("LINE_CATALOG")
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self._get_tpl_meanmjd(gtable)
            self._add_geometry_to_sofdict(tpl)
            # Provide the list of files to the dictionary
            self._sofdict[expotype] = add_listpath(self.paths.rawfiles,
                    list(gtable['filename']))
            # Number of objects
            Nexpo = len(self._sofdict[expotype])
            if self.verbose:
                upipe.print_info("Number of expo is {Nexpo} for {expotype}".format(Nexpo=Nexpo, expotype=expotype))
            # Finding the best tpl for BIAS
            if illum:
                self._add_tplraw_to_sofdict(mean_mjd, "ILLUM") 
            self._add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'FLAT', 
                'TRACE', 'WAVE', 'TWILIGHT'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Run the recipe to reduce the standard (muse_scibasic)

            dir_products = self._get_fullpath_expo(expotype, "processed")
            suffix = self._get_suffix_product(expotype)
            name_products = []
            list_expo = np.arange(Nexpo).astype(np.int) + 1
            for iexpo in list_expo:
                name_products += ['{0:04d}-{1:02d}.fits'.format(iexpo, j+1) for j in range(24)]
            self.recipe_scibasic(self.current_sof, tpl, expotype, dir_products, name_products, suffix)

            # Write the Processed files Table and save it
            gtable['iexpo'] = list_expo
            self.save_expo_table(expotype, gtable, "processed", aggregate=False, update=True)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    @print_my_function_name
    def run_standard(self, sof_filename='standard', tpl="ALL", update=None):
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
                upipe.print_warning("No processed STD recovered from the astropy file Table - Aborting",
                        pipe=self)
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

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
            name_std = deepcopy(dic_files_products['STD'])
            dir_std = self._get_fullpath_expo('STD', "master")
            self.recipe_std(self.current_sof, dir_std, name_std, mytpl)

        # Write the MASTER files Table and save it
        self.save_expo_table("STD", std_table, "master", aggregate=False, update=update)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    @print_my_function_name
    def run_sky(self, sof_filename='sky', tpl="ALL", fraction=0.8, update=None):
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
                upipe.print_warning("No SKY recovered from the astropy file Table - Aborting",
                        pipe=self)
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

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
            name_sky = deepcopy(dic_files_products['SKY'])
            self.recipe_sky(self.current_sof, dir_sky, name_sky, mytpl, iexpo, fraction)

        # Write the MASTER files Table and save it
        self.save_expo_table("SKY", sky_table, "processed", aggregate=False, update=update)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

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
            line=None, filter_list='white', **extra_kwargs):
        """Launch the scipost command to get individual exposures in a narrow
        band filter
        """
        # First selecting the files via the grouped table
        object_table = self._get_table_expo("OBJECT", "processed")

        # Getting the band corresponding to the line
        lambda_window = extra_kwargs.pop("lambda_window", 10.0)
        [lmin, lmax] = upipe.get_emissionline_band(line=line, 
                                                   velocity=self.vsystemic, 
                                                   lambda_window=lambda_window)

        suffix = extra_kwargs.pop("suffix", "")
        if line is not None: 
            suffix = "{0}_{1}".format(suffix, line)

        # Filter for alignment if set up
        filter_for_alignment = extra_kwargs.pop("filter_for_alignment", 
                                                self.filter_for_alignment)

        # Processing individual exposures to get the full cube and image
        for i in range(len(object_table)):
            iexpo = np.int(object_table['iexpo'][i])
            mytpl = object_table['tpls'][i]
            if tpl != "ALL" and tpl != mytpl :
                continue
            # Running scipost now on the individual exposure
            self.run_scipost(sof_filename=sof_filename, expotype=expotype,
                    tpl=mytpl, list_expo=[iexpo], suffix=suffix, 
                    filter_list=filter_list, filter_for_alignment=filter_for_alignment,
                    lambdaminmax=[lmin, lmax], save='cube', 
                    offset_list=False, **extra_kwargs)

    def _get_scipost_products(self, save='cube,skymodel', list_expo=[], filter_list='white'):
        """Provide a set of key output products depending on the save mode
        for scipost
        """
        name_products = []
        suffix_products = []
        suffix_prefinalnames = []
        suffix_postfinalnames = []
        list_options = save.split(',')
        for option in list_options:
            for prod in dic_products_scipost[option]:
                name_products.append(prod)
                if prod == "IMAGE_FOV":
                    for i, value in enumerate(filter_list.split(','), 
                            start=1):
                        suffix_products.append("_{0:04d}".format(i))
                        suffix_prefinalnames.append("_{0}".format(value))
                        if len(list_expo) == 1 :
                            suffix_postfinalnames.append("_{0:04d}".format(list_expo[0]))
                        else :
                            suffix_postfinalnames.append("")

                elif any(x in prod for x in ['PIXTABLE', 'RAMAN', 'SKY', 'AUTOCAL']):
                    for i in range(len(list_expo)):
                        suffix_products.append("_{0:04d}".format(i+1))
                        suffix_prefinalnames.append("")
                        suffix_postfinalnames.append("_{0:04d}".format(i+1))
                else :
                    if "DATACUBE" in prod:
                        if len(list_expo) == 1 :
                            suffix_postfinalnames.append("_{0:04d}".format(list_expo[0]))
                        else :
                            suffix_postfinalnames.append("")
                    suffix_products.append("")
                    suffix_prefinalnames.append("")

        return name_products, suffix_products, suffix_prefinalnames, suffix_postfinalnames

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
                upipe.print_warning("No {0} recovered from the {1} astropy file "
                    "Table - Aborting".format(expotype, stage), pipe=self)

        return found_expo, list_expo, group_list_expo, group_table

    @print_my_function_name
    def run_scipost(self, sof_filename='scipost', expotype="OBJECT", tpl="ALL", stage="processed", list_expo=[], 
            lambdaminmax=[4000.,10000.], suffix="", **kwargs):
        """Scipost treatment of the objects
        Will run the esorex muse_scipost routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time
        list_expo: list of integers providing the exposure numbers
        """
        # Backward compatibility
        old_naming_convention = kwargs.pop("old_naming_convention", False)

        # Selecting the table with the right iexpo
        found_expo, list_expo, group_list_expo, scipost_table = self._select_list_expo(expotype, tpl, stage, list_expo) 
        if not found_expo:
            return

        # Lambda min and max?
        [lambdamin, lambdamax] = lambdaminmax
        # Skymethod
        skymethod = kwargs.pop("skymethod", "model")
        # Save options
        if skymethod != "none":
            save = kwargs.pop("save", "cube,skymodel,individual")
        else :
            # If skymethod is none, no need to save the skymodel...
            save = kwargs.pop("save", "cube,individual")
        # Filters
        filter_list = kwargs.pop("filter_list", "white")
        filter_for_alignment = kwargs.pop("filter_for_alignment", self.filter_for_alignment)
        offset_list = kwargs.pop("offset_list", "True")
        autocalib = kwargs.pop("autocalib", "none")
        rvcorr = kwargs.pop("rvcorr", "bary")
        AC_suffix = kwargs.pop("AC_suffix", "_AC")
        default_suffix_skycontinuum = ""
        suffix_skycontinuum = kwargs.pop("suffix_skycont", default_suffix_skycontinuum)
        if rvcorr != "bary":
            upipe.print_warning("Scipost will use '{0}' as barycentric "
                    "correction [Options are bary, helio, geo, none]".format(rvcorr), 
                    pipe=self)

        if skymethod != "none" :
            if suffix_skycontinuum != default_suffix_skycontinuum:
                upipe.print_warning("Scipost will use '{0}' as suffix "
                        "for the SKY_CONTINUUM files".format(suffix_skycontinuum),
                        pipe=self)

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for scipost
        # Selecting either one or all of the exposures
        for gtable in scipost_table.groups:
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self._get_tpl_meanmjd(gtable)
            # Now starting with the standard recipe
            self._add_calib_to_sofdict("EXTINCT_TABLE", reset=True)
            self._add_calib_to_sofdict("FILTER_LIST")
            if autocalib == "user":
                self._add_skycalib_to_sofdict("AUTOCAL_FACTORS", mean_mjd, 'SKY', 
                        "processed", suffix=AC_suffix)
            self._add_astrometry_to_sofdict(tpl)
            self._add_skycalib_to_sofdict("STD_RESPONSE", mean_mjd, 'STD')
            self._add_skycalib_to_sofdict("STD_TELLURIC", mean_mjd, 'STD')
            if skymethod != "none" :
                self._add_calib_to_sofdict("SKY_LINES")
                self._add_skycalib_to_sofdict("SKY_CONTINUUM", mean_mjd, 'SKY', 
                        "processed", perexpo=True, suffix=suffix_skycontinuum)
            self._add_tplmaster_to_sofdict(mean_mjd, 'LSF')
            if offset_list :
                self._sofdict['OFFSET_LIST'] = [joinpath(self._get_fullpath_expo(expotype, "processed"),
                        '{0}_{1}_{2}.fits'.format(dic_files_products['ALIGN'][0], 
                            filter_for_alignment, tpl))]

            # Selecting only exposures to be treated
            # We need to force 'OBJECT' to make sure scipost will deal with the exposure
            # even if it is e.g., a SKY
            pixtable_name = self._get_suffix_product('OBJECT')
            pixtable_name_thisone = self._get_suffix_product(expotype)
            self._sofdict[pixtable_name] = []
            list_group_expo = gtable['iexpo'].data
            for iexpo in list_group_expo:
                if old_naming_convention:
                   self._sofdict[pixtable_name] += [joinpath(self._get_fullpath_expo(expotype, "processed"),
                       '{0}_{1:04d}-{2:02d}.fits'.format(pixtable_name_thisone, iexpo, j+1)) for j in range(24)]
                else:
                   self._sofdict[pixtable_name] += [joinpath(self._get_fullpath_expo(expotype, "processed"),
                       '{0}_{1}_{2:04d}-{3:02d}.fits'.format(pixtable_name_thisone, tpl, iexpo, j+1)) for j in range(24)]
            self.write_sof(sof_filename="{0}_{1}{2}_{3}".format(sof_filename, expotype, 
                suffix, tpl), new=True)
            # products
            dir_products = self._get_fullpath_expo(expotype, "processed")
            name_products, suffix_products, suffix_prefinalnames, suffix_postfinalnames = \
                self._get_scipost_products(save, list_group_expo, filter_list)
            self.recipe_scipost(self.current_sof, tpl, expotype, dir_products, 
                    name_products, suffix_products, suffix_prefinalnames, 
                    suffix_postfinalnames, suffix=suffix, 
                    lambdamin=lambdamin, lambdamax=lambdamax, save=save, 
                    filter_list=filter_list, autocalib=autocalib, rvcorr=rvcorr, 
                    skymethod=skymethod, filter_for_alignment=filter_for_alignment,
                    **kwargs)

            # Write the MASTER files Table and save it
            if len(list_expo) == 1: suffix_expo = "_{0:04d}".format(list_expo[0])
            else: suffix_expo = ""
            self.save_expo_table(expotype, scipost_table, "reduced", 
                    "IMAGES_FOV_{0}_{1}{2}_list_table.fits".format(expotype, 
                        suffix_expo, tpl), aggregate=False, update=True)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def _get_align_products(self, list_expo=[], filter_list='white,Cousins_R'):
        """Provide a set of key output products for exp_align
        """
        name_products = []
        suffix_products = []
        list_options = save.split(',')
        for option in list_options:
            for prod in dic_products_scipost[option]:
                if prod == "IMAGE_FOV":
                    for i in range(len(filter_list.split(','))):
                        suffix_products.append("_{0:04d}".format(i+1))
                        name_products.append(prod)
                elif any(x in prod for x in ['PIXTABLE', 'RAMAN', 'SKY']):
                    for i in range(len(list_expo)):
                        suffix_products.append("_{0:04d}".format(i+1))
                        name_products.append(prod)
                else :
                    name_products.append(prod)
                    suffix_products.append("")
        return name_products, suffix_products

    @print_my_function_name
    def run_align_bygroup(self, sof_filename='exp_align', expotype="OBJECT", 
            list_expo=[], stage="processed", line=None, 
            filter_name="Cousins_R", tpl="ALL", **kwargs):
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
        found_expo, list_expo, group_list_expo, align_table = self._select_list_expo(expotype, tpl, stage, list_expo) 
        if not found_expo:
            return
        
        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        suffix = "_{0}".format(filter_name)
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
            self._sofdict['IMAGE_FOV'] = [joinpath(self._get_fullpath_expo("OBJECT", "processed"),
                'IMAGE_FOV{0}_{1}_{2:04d}.fits'.format(suffix, mytpl, iexpo)) for iexpo in list_group_expo]
            self.write_sof(sof_filename=sof_filename + "{0}_{1}".format(suffix, mytpl), new=True)
            dir_align = self._get_fullpath_expo('OBJECT', "processed")
            namein_align = deepcopy(dic_files_products['ALIGN'])
            nameout_align = [name + "{0}_{1}".format(suffix, mytpl) for name in deepcopy(dic_files_products['ALIGN'])] 
            for iter_file in dic_files_iexpo_products['ALIGN']:
                for iexpo in list_group_expo:
                    namein_align.append('{0}_{1:04d}'.format(iter_file, iexpo))
                    nameout_align.append('{0}{1}_{2}_{3:04d}'.format(iter_file, suffix, mytpl, iexpo))
            self.recipe_align(self.current_sof, dir_align, namein_align, nameout_align, mytpl, "group", **kwargs)

            # Write the MASTER files Table and save it
            self.save_expo_table(expotype, align_table, "reduced", 
                    "ALIGNED_IMAGES_{0}{1}_{2}_list_table.fits".format(expotype, 
                        suffix, mytpl), aggregate=False, update=True)
        
        # Go back to original folder
        self.goto_prevfolder(logfile=True)
        
    @print_my_function_name
    def run_align_bypointing(self, sof_filename='exp_align', expotype="OBJECT", 
            list_expo=[], stage="processed", line=None, 
            filter_name="Cousins_R", tpl="ALL", **kwargs):
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
        found_expo, list_expo, group_list_expo, align_table = self._select_list_expo(expotype, tpl, stage, list_expo) 
        if not found_expo:
            return
        # Stop the process if only 1 or fewer exposures are retrieved
        if len(align_table) <= 1:
            if self.verbose:
                upipe.print_warning("Pointing = {0}".format(self.pointing),
                        pipe=self)
                upipe.print_warning("No derived OFFSET LIST as only 1 exposure retrieved in this Pointing",
                        pipe=self)
            return
        
        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        suffix = "_{0}".format(filter_name)
        if line is not None:
            suffix += "_{0}".format(line)

        # Use the pointing as a suffix for the names
        pointing = "P{0:02d}".format(self.pointing)

        # Now creating the SOF file, first reseting it
        self._sofdict.clear()
        # Producing the list of IMAGES and add them to the SOF
        self._sofdict['IMAGE_FOV'] = [joinpath(self._get_fullpath_expo("OBJECT", "processed"),
            'IMAGE_FOV{0}_{1}_{2:04d}.fits'.format(suffix, row['tpls'], row['iexpo'])) for row in align_table]

        # Write the SOF
        self.write_sof(sof_filename=sof_filename + "{0}_{1}_{2}_{3}".format(suffix, expotype, pointing, tpl), new=True)
        dir_align = self._get_fullpath_expo('OBJECT', "processed")
        namein_align = deepcopy(dic_files_products['ALIGN'])
        nameout_align = [name + "{0}_{1}_{2}_{3}".format(suffix, expotype, pointing, tpl) for name in deepcopy(dic_files_products['ALIGN'])] 
        for iter_file in dic_files_iexpo_products['ALIGN']:
            for i, row in enumerate(align_table):
                namein_align.append('{0}_{1:04d}'.format(iter_file, i+1))
                nameout_align.append('{0}{1}_{2}_{3:04d}'.format(iter_file, suffix, row['tpls'], row['iexpo']))

        # Find the alignment - Note that Pointing is used and tpl reflects the given selection
        self.recipe_align(self.current_sof, dir_align, namein_align, nameout_align, 
                tpl, pointing, **kwargs)

        # Write the MASTER files Table and save it
        self.save_expo_table(expotype, align_table, "reduced", 
                "ALIGNED_IMAGES_{0}{1}_{2}_list_table.fits".format(expotype, 
                suffix, tpl), aggregate=False, update=True)
        
        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    @print_my_function_name
    def save_fine_alignment(self, name_offset_table=None):
        """ Save the fine pointing alignment
        """
        if name_offset_table is None:
            name_offset_table = "XX"

        # Use the save from the alignment module
        tstamp = self.dic_alignments.present_tstamp
        self.dic_alignments[tstamp].save(name_offset_table)

    @print_my_function_name
    def run_fine_alignment(self, name_ima_reference=None, nexpo=1, list_expo=[], line=None, 
            filter_name="Cousins_R", bygroup=False, reset=False):
        """Run the alignment on this pointing using or not a reference image
        """
        # If not yet initialised, build the dictionary
        if not hasattr(self, "dic_alignments"):
            # Create an empty dictionary with a None timestamp
            self.dic_alignments = upipe.TimeStampDict()
            # Reset to True to initialise the structure
            reset = True

        # If reset, check if list is not empty
        # If not empty, create new time stamp and proceed with initialisation
        if reset:
            # Create a new set of alignments
            self.get_align_group(name_ima_reference=name_ima_reference, 
                    list_expo=list_expo, line=line, filter_name=filter_name, 
                    bygroup=bygroup)
            # if dictionary is empty, it creates the first timestamp
            self.dic_alignments.create_new_timestamp(self.align_group)

        # Run the alignment
        tstamp = self.dic_alignments.present_tstamp
        self.dic_alignments[tstamp].run(nexpo)
        
    @print_my_function_name
    def get_align_group(self, name_ima_reference=None, list_expo=[], line=None, 
            filter_name="Cousins_R", bygroup=False):
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
        suffix = "_{0}".format(filter_name)
        if line is not None:
            suffix += "_{0}".format(line)

        # Option to do a per group alignment
        if bygroup:
            for gtable in align_table.groups:
                mytpl, mymjd = self._get_tpl_meanmjd(gtable)
                list_group_expo = gtable['iexpo'].data
                list_names_muse = [joinpath(self._get_fullpath_expo("OBJECT", "processed"),
                    'IMAGE_FOV{0}_{1}_{2:04d}.fits'.format(suffix, mytpl, iexpo)) 
                    for iexpo in list_group_expo]
                self.align_group.append(AlignMusePointing(name_ima_reference, list_names_muse, flag="mytpl"))
        # If not by group
        else:
            list_names_muse = [joinpath(self._get_fullpath_expo("OBJECT", "processed"),
                'IMAGE_FOV{0}_{1}_{2:04d}.fits'.format(suffix, row['tpls'], row['iexpo'])) 
                for row in align_table]
            # Giving a reference image name, doing the alignment w.r.t. to that specific image
            if name_ima_reference is None:
                upipe.print_warning("Using the first MUSE exposure as a reference")
                self.align_group.append(AlignMusePointing(list_names_muse[0], 
                    list_names_muse[1:], flag="mytpl"))
            else:
                self.align_group.append(AlignMusePointing(name_ima_reference, 
                    list_names_muse, flag="mytpl"))

    @print_my_function_name
    def run_combine_pointing(self, sof_filename='exp_combine', expotype="OBJECT", 
            list_expo=[], stage="processed", tpl="ALL", filter_list="Cousins_R", 
            suffix="", **kwargs):
        """Produce a cube from all frames in the pointing
        list_expo or tpl specific arguments can still reduce the selection if needed
        """
        # Backward compatibility
        old_naming_convention = kwargs.pop("old_naming_convention", False)

        # Selecting the table with the right iexpo
        found_expo, list_expo, group_list_expo, combine_table = self._select_list_expo(expotype, tpl, stage, list_expo) 
        if not found_expo:
            return

        # Abort if only one exposure is available
        # exp_combine needs a minimum of 2
        if len(combine_table) <= 1:
            if self.verbose:
                upipe.print_warning("The combined pointing has only one exposure: process aborted",
                        pipe=self)
            return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Setting the default alignment filter
        filter_for_alignment = kwargs.pop("filter_for_alignment", self.filter_for_alignment)
        # Use the pointing as a suffix for the names
        pointing = "P{0:02d}".format(self.pointing)
        # Save option
        save = kwargs.pop("save", "cube,combined")

        # Now creating the SOF file, first reseting it
        self._sofdict.clear()
        # Selecting only exposures to be treated
        # Producing the list of REDUCED PIXTABLES
        self._add_calib_to_sofdict("FILTER_LIST")
        pixtable_name = self._get_suffix_product('REDUCED')
        pixtable_name_thisone = dic_products_scipost['individual']

        # Setting the default option of offset_list
        # And looking for that table, adding it to the sof file
        offset_list = kwargs.pop("offset_list", "True")
        folder_expo = self._get_fullpath_expo(expotype, stage)
        if offset_list :
            offset_list_tablename = kwargs.pop("offset_list_tablename", None)
            if offset_list_tablename is None:
                offset_list_tablename = "{0}{1}_{2}_{3}_{4}_{5}.fits".format(
                        dic_files_products['ALIGN'][0], suffix, filter_for_alignment, 
                        expotype, pointing, tpl)
            if not os.path.isfile(joinpath(folder_expo, offset_list_tablename)):
                upipe.print_error("OFFSET_LIST table {0} not found in folder {1}".format(
                        offset_list_tablename, folder_expo), pipe=self)
                return

            self._sofdict['OFFSET_LIST'] = [joinpath(folder_expo, offset_list_tablename)]

        self._sofdict[pixtable_name] = []
        for prod in pixtable_name_thisone:
            if old_naming_convention:
               self._sofdict[pixtable_name] += [joinpath(folder_expo,
                   '{0}_{1:04d}.fits'.format(prod, row['iexpo'])) for row in combine_table]
            else:
               self._sofdict[pixtable_name] += [joinpath(folder_expo,
                   '{0}_{1}_{2:04d}.fits'.format(prod, row['tpls'], row['iexpo'])) for row in
                   combine_table]
        self.write_sof(sof_filename="{0}_{1}{2}_{3}".format(sof_filename, expotype, 
            suffix, tpl), new=True)

        # Product names
        dir_products = self._get_fullpath_expo(expotype, stage)
        name_products, suffix_products, suffix_prefinalnames = _get_combine_products(filter_list) 

        # Combine the exposures 
        self.recipe_combine(self.current_sof, dir_products, name_products, 
                tpl, expotype, suffix_products=suffix_products,
                suffix_prefinalnames=suffix_prefinalnames,
                save=save, suffix=suffix, filter_list=filter_list, **kwargs)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)
