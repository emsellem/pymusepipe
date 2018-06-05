# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS creating sof file module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# This module has been largely inspired by work of
# Bern Husemann, Dimitri Gadotti, Kyriakos and Martina from the GTO MUSE MAD team
# and further rewritten by Mark van den Brok. 
# Thanks to all !

# Numpy
import numpy as np

# Standard modules
import os
from os.path import join as joinpath

import collections
from collections import OrderedDict

from pymusepipe import util_pipe as upipe
from pymusepipe import musepipe

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
        for key in self._sofdict.keys() :
            for item in self._sofdict[key] :
                text_to_write = "{0} {1}\n".format(item, key)
                sof_file.write(text_to_write)
                if verbose :
                    upipe.print_info(text_to_write)

        sof_file.close()
        # Returning the current sof as relative path
        self.current_sof = upipe.normpath(os.path.relpath(sof))

    def _select_closest_mjd(self, mjdin, group_table) :
        """Get the closest frame within the expotype
        If the attribute does not exist in Tables, it tries to read
        the table from the folder
        """
        # Get the closest tpl
        index = np.argmin((mjdin - group_table['mjd'])**2)
        closest_tpl = group_table[index]['tpls']
        return index, closest_tpl
    
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
        dir_master = self._get_fullpath_expo(expotype)
        self._sofdict[self._get_suffix_product(expotype)] = [upipe.normpath(joinpath(dir_master, 
            self._get_suffix_product(expotype) + "_" + this_tpl + ".fits"))]

    def _add_tplraw_to_sofdict(self, mean_mjd, expotype, reset=False):
        """ Add item to dictionary for the sof writing
        """
        if reset: self._sofdict.clear()
        # Finding the best tpl for this raw file type
        expo_table = self._get_table_expo(expotype, "raw")
        index, this_tpl = self._select_closest_mjd(mean_mjd, expo_table) 
        self._sofdict[expotype] = [upipe.normpath(joinpath(self.paths.rawfiles, 
            expo_table['filename'][index]))]

    def _add_skycalib_to_sofdict(self, tag, mean_mjd, expotype, stage="master", reset=False):
        """ Add item to dictionary for the sof writing
        """
        if reset: self._sofdict.clear()
        # Finding the best tpl for this sky calib file type
        expo_table = self._get_table_expo(expotype, stage)
        index, this_tpl = self._select_closest_mjd(mean_mjd, expo_table) 
        dir_calib = self._get_fullpath_expo(expotype, stage)
        self._sofdict[tag] = [joinpath(dir_calib, "{0}_{1}.fits".format(tag, this_tpl))]

    def _add_calib_to_sofdict(self, calibtype, reset=False):
        """Adding a calibration file for the SOF 
        """
        if reset: self._sofdict.clear()
        calibfile = getattr(self.my_params, calibtype.lower())
        self._sofdict[calibtype] = [joinpath(self.my_params.musecalib, calibfile)]

    def _add_geometry_to_sofdict(self, tpls):
        """Extract the geometry table and add it to the dictionary
        for the SOF file
        """
        calfolder = self.my_params.musecalib
        if self._time_geo_table :
            listkeys = list(musepipe.dic_geo_table.keys())
            listkeys.append(musepipe.future_date)
            for ikey in range(len(listkeys) - 1):
                if tpls >= listkeys[ikey] and tpls < listkeys[ikey+1]:
                    geofile = musepipe.dic_geo_table[listkeys[ikey]]
        else:
            geofile = self.my_params.geo_table

        self._sofdict['GEOMETRY_TABLE']=["{folder}{geo}".format(folder=calfolder, geo=geofile)]

    def _add_astrometry_to_sofdict(self, tpls):
        """Extract the astrometry table and add it to the dictionary
        for the SOF file
        """
        calfolder = self.my_params.musecalib
        if self._time_geo_table :
            listkeys = list(dic_astro_table.keys())
            listkeys.append(musepipe.future_date)
            for ikey in range(len(listkeys) - 1):
                if tpls >= listkeys[ikey] and tpls < listkeys[ikey+1]:
                    astrofile = dic_astro_table[listkeys[ikey]]
        else :
            astrofile = self.my_params.astro_table

        self._sofdict['ASTROMETRY_WCS']=["{folder}{astro}".format(folder=calfolder, astro=astrofile)]

