# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS SOF files writing module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# This module has been largely inspired by work of
# Bern Husemann, Dimitri Gadotti, Kyriakos and Martina from the GTO MUSE MAD team
# and further rewritten by Mark van den Brok. 
# Thanks to all !


class SofPipe(object) :
    """SofPipe class containing all the SOF writing modules
    """
    def __init__(self, verbose=True) :
        """Initialisation of SofPipe
        """

        self.verbose = verbose

