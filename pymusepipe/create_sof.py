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

def feed_sof(sof_filename, dic_files={}, new=False, verbose=True) :
    """Feeding an sof file with input filenames from a dictionary
    """
    sof = joinpath(sof_folder, sof_filename)
    if new :
        sof_file = open(sof, "w+")
        if verbose :
            print("Writing in file {0}".format(sof))
    else :
        sof_file = open(sof, "a")
        if verbose :
            print("Appending in file {0}".format(sof))

    for key in dic_file.keys() :
        for item in key :
            text_to_write = "{0} {1}\n".format(key, dic_file[key])
        sof_file.write(text_to_write)
        if verbose :
            print(text_to_write)

