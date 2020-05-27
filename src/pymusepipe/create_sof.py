# Licensed under a MIT license - see LICENSE

"""MUSE-PHANGS creating sof file module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"

# This module has been largely inspired by work of
# Bern Husemann, Dimitri Gadotti, Kyriakos and Martina from the GTO MUSE MAD team
# and further rewritten by Mark van den Brok. 
# Thanks to all !

# Standard modules
import os
from os.path import join as joinpath

from collections import OrderedDict

from . import util_pipe as upipe
from .config_pipe import get_suffix_product

class SofDict(OrderedDict) :
    """New Dictionary for the SOF writing
    Inheriting from ordered Dictionary
    """
    def __init__(self) :
        OrderedDict.__init__(self)

class SofPipe(object) :
    """SofPipe class containing all the SOF writing modules
    """
    def __init__(self) :
        """Initialisation of SofPipe
        """
        # Creating an empty dictionary for the SOF writing
        self._sofdict = SofDict()

    def write_sof(self, sof_filename, new=False, verbose=None) :
        """Feeding an sof file with input filenames from a dictionary
        """

        # Removing the extension of the file if already set
        if not sof_filename.lower().endswith(".sof") :
            sof_filename = sof_filename + ".sof"

        sof = joinpath(self.paths.sof, sof_filename)
        # If new file, start from scratch (overwrite)
        if new :
            sof_file = open(sof, "w+")
            if verbose :
                upipe.print_info("Writing in file {0}".format(sof))
        # if not new, then append
        else :
            sof_file = open(sof, "a")
            if verbose :
                upipe.print_info("Appending in file {0}".format(sof))
    
        # Use dictionary to write up the lines
        for key in self._sofdict:
            for item in self._sofdict[key] :
                text_to_write = "{0} {1}\n".format(item, key)
                sof_file.write(text_to_write)
                if verbose :
                    upipe.print_info(text_to_write)

        sof_file.close()
        # Returning the current sof as relative path
        self.current_sof = upipe.normpath(os.path.relpath(sof))

    def _add_list_tplmaster_to_sofdict(self, mean_mjd, list_expotype):
        """Add a list of masterfiles to the SOF
        """
        for expotype in list_expotype :
            self._add_tplmaster_to_sofdict(mean_mjd, expotype)

    def _add_tplmaster_to_sofdict(self, mean_mjd, expotype, reset=False):
        """ Add item to dictionary for the sof writing
        """
        if reset: self._sofdict.clear()
        # Finding the best tpl for this master
        index, this_tpl = self._select_closest_mjd(mean_mjd, self._get_table_expo(expotype)) 
        if self._debug:
            upipe.print_debug("Index = {0}, Tpl = {1}".format(index, this_tpl))
        if index >= 0:
            dir_master = self._get_fullpath_expo(expotype)
            self._sofdict[get_suffix_product(expotype)] = [upipe.normpath(joinpath(dir_master, 
                get_suffix_product(expotype) + "_" + this_tpl + ".fits"))]
        else:
            upipe.print_error("Failed to find a master exposure of type {} "
                              "in this table".format(expotype))

    def _add_tplraw_to_sofdict(self, mean_mjd, expotype, reset=False):
        """ Add item to dictionary for the sof writing
        """
        if reset: self._sofdict.clear()
        # Finding the best tpl for this raw file type
        expo_table = self._get_table_expo(expotype, "raw")
        index, this_tpl = self._select_closest_mjd(mean_mjd, expo_table) 
        if index >= 0:
            self._sofdict[expotype] = [upipe.normpath(joinpath(self.paths.rawfiles, 
                expo_table['filename'][index]))]
        else:
            upipe.print_error("Failed to find a raw exposure of type {} "
                              "in this table".format(expotype))

    def _add_skycalib_to_sofdict(self, tag, mean_mjd, expotype, stage="master", 
            suffix="", prefix="", reset=False, perexpo=False):
        """ Add item to dictionary for the sof writing
        """
        if reset: self._sofdict.clear()
        # Finding the best tpl for this sky calib file type
        expo_table = self._get_table_expo(expotype, stage)
        index, this_tpl = self._select_closest_mjd(mean_mjd, expo_table) 
        dir_calib = self._get_fullpath_expo(expotype, stage)
        if perexpo:
            iexpo = expo_table[index]['iexpo']
            suffix += "_{0:04d}".format(iexpo)

        # Name for the sky calibration file
        name_skycalib = "{0}{1}_{2}{3}.fits".format(prefix, tag, this_tpl, suffix)

        self._sofdict[tag] = [joinpath(dir_calib, name_skycalib)]

    def _add_calib_to_sofdict(self, calibtype, reset=False):
        """Adding a calibration file for the SOF 
        """
        if reset: self._sofdict.clear()
        self._sofdict[calibtype] = [self._get_name_calibfile(calibtype)]

    def _get_name_calibfile(self, calibtype):
        """Get the name of the calibration file
        """
        calibfile = getattr(self.pipe_params, calibtype.lower())
        return joinpath(self.pipe_params.musecalib, calibfile)    

    def _add_geometry_to_sofdict(self, tpls, mean_mjd):
        """Extract the geometry table and add it to the dictionary
        for the SOF file
        """
        if self._time_astrometry :
            calfolder = self.pipe_params.musecalib_time
            geofile = self.retrieve_geoastro_name(tpls, filetype='geo')
        else:
            expo_table = self._get_table_expo("GEOMETRY", "raw")
            if len(expo_table) > 0:
                index, this_tpl = self._select_closest_mjd(mean_mjd, expo_table) 
                calfolder = self.paths.rawfiles
                geofile = expo_table['filename'][index]
            else:
                calfolder = self.pipe_params.musecalib
                geofile = self.pipe_params.geo_table

        self._sofdict['GEOMETRY_TABLE']=["{folder}{geo}".format(folder=calfolder, geo=geofile)]

    def _add_astrometry_to_sofdict(self, tpls, mean_mjd):
        """Extract the astrometry table and add it to the dictionary
        for the SOF file
        """
        if self._time_astrometry :
            calfolder = self.pipe_params.musecalib_time
            astrofile = self.retrieve_geoastro_name(tpls, filetype='astro')
        else :
            expo_table = self._get_table_expo("ASTROMETRY", "raw")
            if len(expo_table) > 0:
                index, this_tpl = self._select_closest_mjd(mean_mjd, expo_table) 
                calfolder = self.paths.rawfiles
                astrofile = expo_table['filename'][index]
            else:
                calfolder = self.pipe_params.musecalib
                astrofile = self.pipe_params.astro_table

        self._sofdict['ASTROMETRY_WCS']=["{folder}{astro}".format(folder=calfolder, astro=astrofile)]

