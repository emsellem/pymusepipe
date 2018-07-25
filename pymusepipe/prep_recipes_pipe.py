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
        'SKY': ['SKY_MASK', 'IMAGE_FOV', 'SKY_LINES', 'SKY_SPECTRUM', 
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
        'raman': ['RAMAN_IMAGES']
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
        self.list_recipes = list_recipes

    def _get_tpl_meanmjd(self, gtable):
        """Get tpl of the group and mean mjd of the group
        """
        tpl = gtable['tpls'][0]
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
                upipe.print_warning("Empty file table of type {0}".format(expotype))
                upipe.print_warning("Returning an empty Table from the tpl -astropy- selection")
            return MUSE_subtable

        group_table = MUSE_subtable.group_by('tpls')
        if tpl == "ALL":
            return group_table
        else :
            return group_table.groups[group_table.groups.key['tpls'] == tpl]
        
    def run_all_recipes(self, fraction=0.8):
        """Running all recipes in one shot
        """
        self.run_bias()
        self.run_flat()
        self.run_wave()
        self.run_lsf()
        self.run_twilight()
        self.run_scibasic_all()
        self.run_standard()
        self.run_sky(fraction=fraction)
        self.run_prep_align()
        self.run_align()
        self.run_scipost()

    def run_bias(self, sof_filename='bias', tpl="ALL"):
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
                upipe.print_warning("No BIAS recovered from the astropy Table - Aborting")
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
        self.save_expo_table('BIAS', tpl_gtable, "master")

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_flat(self, sof_filename='flat', tpl="ALL"):
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
                upipe.print_warning("No FLAT recovered from the astropy Table - Aborting")
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
        self.save_expo_table('FLAT', tpl_gtable, "master")
        self.save_expo_table('TRACE', tpl_gtable, "master")

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_wave(self, sof_filename='wave', tpl="ALL"):
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
                upipe.print_warning("No WAVE recovered from the astropy file Table - Aborting")
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
        self.save_expo_table('WAVE', tpl_gtable, "master")

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_lsf(self, sof_filename='lsf', tpl="ALL"):
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
                upipe.print_warning("No WAVE recovered from the astropy file Table - Aborting")
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
        self.save_expo_table('LSF', tpl_gtable, "master")

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_twilight(self, sof_filename='twilight', tpl="ALL"):
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
                upipe.print_warning("No TWILIGHT recovered from the astropy file Table - Aborting")
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
            self._add_tplraw_to_sofdict(mean_mjd, "ILLUM")
            self._add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'FLAT', 'TRACE', 'WAVE'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Names and folder of final Master Wave
            dir_twilight = self._get_fullpath_expo('TWILIGHT', "master")
            name_twilight = dic_files_products['TWILIGHT']
            self.recipe_twilight(self.current_sof, dir_twilight, name_twilight, tpl)

        # Write the MASTER TWILIGHT Table and save it
        self.save_expo_table('TWILIGHT', tpl_gtable, "master")

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_scibasic_all(self, list_object=['OBJECT', 'SKY', 'STD'], tpl="ALL"):
        """Running scibasic for all objects in list_object
        Making different sof for each category
        """
        for expotype in list_object:
            sof_filename = 'scibasic_{0}'.format(expotype.lower())
            self.run_scibasic(sof_filename=sof_filename, expotype=expotype, tpl=tpl)

    def run_scibasic(self, sof_filename='scibasic', expotype="OBJECT", tpl="ALL"):
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
                upipe.print_warning("No {0} recovered from the astropy file Table - Aborting".format(expotype))
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
                name_products += ['{0}_{1:04d}-{2:02d}.fits'.format(suffix, iexpo, j+1) for j in range(24)]
            self.recipe_scibasic(self.current_sof, tpl, expotype, dir_products, name_products)

            # Write the Processed files Table and save it
            gtable['iexpo'] = list_expo
            self.save_expo_table(expotype, gtable, "processed", aggregate=False)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_standard(self, sof_filename='standard', tpl="ALL"):
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
                upipe.print_warning("No processed STD recovered from the astropy file Table - Aborting")
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
                'PIXTABLE_STD_{0:04d}-{1:02d}.fits'.format(iexpo, j+1)) for j in range(24)]
            self.write_sof(sof_filename=sof_filename + "_" + mytpl, new=True)
            name_std = dic_files_products['STD']
            dir_std = self._get_fullpath_expo('STD', "master")
            self.recipe_std(self.current_sof, dir_std, name_std, mytpl)

        # Write the MASTER files Table and save it
        self.save_expo_table("STD", std_table, "master", aggregate=False)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_sky(self, sof_filename='sky', tpl="ALL", fraction=0.8):
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
                upipe.print_warning("No SKY recovered from the astropy file Table - Aborting")
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
                'PIXTABLE_SKY_{0:04d}-{1:02d}.fits'.format(iexpo,j+1)) for j in range(24)]
            self.write_sof(sof_filename=sof_filename + "{0:02d}".format(iexpo) + "_" + mytpl, new=True)
            dir_sky = self._get_fullpath_expo('SKY', "processed")
            name_sky = dic_files_products['SKY']
            self.recipe_sky(self.current_sof, dir_sky, name_sky, mytpl, iexpo, fraction)

        # Write the MASTER files Table and save it
        self.save_expo_table("SKY", sky_table, "processed", aggregate=False)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_prep_align(self, sof_filename='scipost', expotype="OBJECT", tpl="ALL", 
            line='CentreMuse', window=6000.0, filter_list='white,Cousins_R', **kwargs):
        """Launch the scipost command to get individual exposures in a narrow
        band filter
        """
        # First selecting the files via the grouped table
        object_table = self._get_table_expo("OBJECT", "processed")

        # Getting the band corresponding to the line
        [lmin, lmax] = upipe.get_emissionline_band(line=line, velocity=self.vsystemic, window=window)

        for i in range(len(object_table)):
            iexpo = np.int(object_table['iexpo'][i])
            suffix = "_{0}".format(line)
            self.run_scipost(sof_filename='scipost', expotype="OBJECT", 
                    tpl="ALL", list_expo=[iexpo], suffix=suffix, 
                    filter_list=filter_list,
                    lambdaminmax=[lmin, lmax], save='cube', **kwargs)

    def _get_scipost_products(self, save='cube,skymodel', list_expo=[], filter_list='white,Cousins_R'):
        """Provide a set of key output products depending on the save mode
        for scipost
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

    def _select_list_expo(self, expotype, tpl, stage, list_expo=None):
        """Select the expo numbers which exists for a certain expotype
        """
        # First selecting the files via the grouped table
        tpl_table = self.select_tpl_files(expotype=expotype, tpl=tpl, stage=stage)

        # Selecting the table with the right iexpo
        if list_expo is None: 
            list_expo = tpl_table['iexpo'].data
        expo_table = tpl_table[np.isin(tpl_table['iexpo'], list_expo)]

        found_expo = True
        if len(expo_table) == 0:
            found_expo = False
            if self.verbose :
                upipe.print_warning("No {0} recovered from the {1} astropy file "
                    "Table - Aborting".format(expotype, stage))

        return found_expo, list_expo, expo_table

    def run_scipost(self, sof_filename='scipost', expotype="OBJECT", tpl="ALL", list_expo=None, 
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
        # Selecting the table with the right iexpo
        found_expo, list_expo, scipost_table = self._select_list_expo(expotype, tpl, stage, list_expo) 
        if not found_expo:
            return

        if len(list_expo) == 1: 
            suffix += "_{0:04d}".format(list_expo[0])
        
        # Lambda min and max?
        [lambdamin, lambdamax] = lambdaminmax
        # Save options
        save = kwargs.pop("save", "cube,skymodel")
        # Filters
        filter_list = kwargs.pop("filter_list", "white,Cousins_R")

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for scipost
        # Selecting either one or all of the exposures
        for gtable in scipost_table.groups:
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self._get_tpl_meanmjd(gtable)
            # Now starting with the standard recipe
            self._add_calib_to_sofdict("EXTINCT_TABLE", reset=True)
            self._add_calib_to_sofdict("SKY_LINES")
            self._add_calib_to_sofdict("FILTER_LIST")
#            self._add_calib_to_sofdict("ASTRO_TABLE")
            self._add_astrometry_to_sofdict(tpl)
            self._add_skycalib_to_sofdict("STD_RESPONSE", mean_mjd, 'STD')
            self._add_skycalib_to_sofdict("STD_TELLURIC", mean_mjd, 'STD')
            self._add_skycalib_to_sofdict("SKY_CONTINUUM", mean_mjd, 'SKY', "processed")
            self._add_tplmaster_to_sofdict(mean_mjd, 'LSF')
            # Selecting only exposures to be treated
            pixtable_name = self._get_suffix_product(expotype)
            self._sofdict[pixtable_name] = []
            for iexpo in list_expo:
                self._sofdict[pixtable_name] += [joinpath(self._get_fullpath_expo(expotype, "processed"),
                    '{0}_{1:04d}-{2:02d}.fits'.format(pixtable_name, iexpo, j+1)) for j in range(24)]
            self.write_sof(sof_filename="{0}_{1}{2}_{3}".format(sof_filename, expotype, 
                suffix, tpl), new=True)
            # products
            dir_products = self._get_fullpath_expo(expotype, "processed")
            name_products, suffix_products = self._get_scipost_products(save, list_expo, filter_list) 
            self.recipe_scipost(self.current_sof, tpl, expotype, dir_products, 
                    name_products, suffix_products, lambdamin=lambdamin, lambdamax=lambdamax, 
                    save=save, list_expo=list_expo, suffix=suffix, filter_list=filter_list, **kwargs)

        # Write the MASTER files Table and save it
        self.save_expo_table(expotype, scipost_table, "reduced", 
                "IMAGES_FOV_{0}{1}_{2}_list_table.fits".format(expotype, suffix, tpl), aggregate=False)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def _get_align_products(self, list_expo=[], filter_list='white,Cousins_R'):
        """Provide a set of key output products for exp_align
        """
        name_align = dic_files_products['ALIGN']
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

    def run_align(self, sof_filename='exp_align', expotype="OBJECT", list_expo=None, line="CousinsR", tpl="ALL"):
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
        found_expo, list_expo, align_table = self._select_list_expo(expotype, tpl, stage, list_expo) 
        if not found_expo:
            return
        
        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        for gtable in align_table.groups:
            # extract the tpl (string) and mean mjd (float) 
            mytpl, mymjd = self._get_tpl_meanmjd(gtable)
            # Now starting with the standard recipe
            self._sofdict.clear()
            self._sofdict['IMAGE_FOV'] = [joinpath(self._get_fullpath_expo("OBJECT", "processed"),
                'IMAGE_FOV_{0}_{1:04d}_{2}.fits'.format(line, iexpo, mytpl)) for iexpo in list_expo]
            self.write_sof(sof_filename=sof_filename + "_{0}_{1}".format(line, mytpl), new=True)
            dir_align = self._get_fullpath_expo('OBJECT', "processed")
            name_align = dic_files_products['ALIGN']
            for iter_file in dic_files_iexpo_products['ALIGN']:
                for iexpo in list_expo:
                    name_align.append('{0}_{1:04d}'.format(iter_file, iexpo))
            self.recipe_align(self.current_sof, dir_align, name_align, mytpl, suffix="_"+line)

        # Write the MASTER files Table and save it
        self.save_expo_table(expotype, align_table, "reduced", 
                "ALIGNED_IMAGES_{0}_{1}_{2}_list_table.fits".format(expotype, line, mytpl), aggregate=False)
        
        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def adjust_alignment(self, name_reference, list_expo=None, line="CousinsR"):
        """Adjust the alignment using a background image
        """
        # Selecting the table with the right iexpo
        found_expo, list_expo, align_table = self._select_list_expo("OBJECT", "ALL", "processed", list_expo) 
        if not found_expo:
            if self.verbose:
                upipe.print_warning("No exposure recovered for the fine alignment")
            return
            
        list_names_muse = []
        for gtable in align_table.groups:
            mytpl, mymjd = self._get_tpl_meanmjd(gtable)
            list_names_muse = [joinpath(self._get_fullpath_expo("OBJECT", "processed"),
                    'IMAGE_FOV_{0}_{1:04d}_{2}.fits'.format(line, iexpo, mytpl)) for iexpo in list_expo]
            aligning_muse = AlignMusePointing(name_reference, list_names_muse)
            
################### OLD ############################################
#    def create_calibrations(self):
#        self.check_for_calibrations()
#        # MdB: this creates the standard calibrations, if needed
#        if not ((self.redo==0) and (self.Master['BIAS']==1)):
#            self.run_bias()
#        if not ((self.redo==0) and (self.Master['DARK']==1)):
#            self.run_dark()
#        if not ((self.redo==0) and (self.Master['FLAT']==1)):
#            self.run_flat()
#        if not ((self.redo==0) and (self.Master['WAVE']==1)):
#            self.run_wavecal()
#        if not ((self.redo==0) and (self.Master['TWILIGHT']==1)):
#            self.run_twilight()    
#        if not ((self.redo==0) and (self.Master['STD']==1)):
#            self.create_standard_star()
#            self.run_standard_int()
#
#    def check_for_calibrations(self, verbose=None):
#        """This function checks which calibration file are present, just in case
#        you do not wish to redo them. Variables are named: 
#        masterbias, masterdark, standard, masterflat, mastertwilight
#        """
#        
#        outcal = getattr(self.my_params, "master" + suffix_folder)
#        if verbose is None: verbose = self.verbose
#
#        for mastertype in dic_listMaster.keys() :
#            [fout, suffix] = dic_listMaster[mastertype]
#            # If TWILIGHT = only one cube
#            if mastertype == "TWILIGHT" :
#                if not os.path.isfile(joinpath(outcal, fout, "{suffix}.fits".format(suffix=suffix))):
#                    self.Master[mastertype] = False
#
#                if verbose :
#                    if self.Master[mastertype] :
#                        print("WARNING: Twilight flats already made")
#                    else :
#                        print("WARNING: Twilight flats NOT there yet")
#
#            # Others = 24 IFU exposures
#            else :
#                for ifu in range(1,25):
#                    if not os.path.isfile(joinpath(outcal, fout, "{0}-{1:02d}.fits".format(suffix, ifu))):
#                        self.Master[mastertype] = False
#                        break
#                if verbose :
#                    if self.Master[mastertype]:
#                        print("WARNING: Master {0} already in place".format(mastertype))
#                    else :
#                        print("WARNING: Master {0} NOT yet there".format(mastertype))
#

