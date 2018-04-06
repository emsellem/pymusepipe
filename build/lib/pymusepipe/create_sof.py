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

# Standard modules
import os
from os.path import join as joinpath
import musepipe as mpipe

class SofPipe(object) :
    """SofPipe class containing all the SOF writing modules
    """
    def __init__(self, verbose=True) :
        """Initialisation of SofPipe
        """
        self.verbose = verbose

    def write_sof(self, sof_filename, sof_folder=None, dic_files={}, new=False, verbose=None) :
        """Feeding an sof file with input filenames from a dictionary
        """
        # Setting the default SOF folder if not provided
        if sof_folder is None : sof_folder = self.paths.sof

        # Removing the extension of the file if already set
        if not sof_filename.lower().endswith(".sof") :
            sof_filename = sof_filename + ".sof"

        sof = joinpath(sof_folder, sof_filename)
        # If new file, start from scratch (overwrite)
        if new :
            sof_file = open(sof, "w+")
            if verbose :
                print("Writing in file {0}".format(sof))
        # if not new, then append
        else :
            sof_file = open(sof, "a")
            if verbose :
                print("Appending in file {0}".format(sof))
    
        # Use dictionary to write up the lines
        for key in dic_files.keys() :
            for item in dic_files[key] :
                text_to_write = "{0} {1}\n".format(item, key)
                sof_file.write(text_to_write)
                if verbose :
                    print(text_to_write)

        sof_file.close()
        # Returning the current sof as relative path
        self.current_sof = mpipe.normpath(os.path.relpath(sof))

    def add_XX_tosof(self) :
        pass
