# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS check pipeline module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"
# This module will take a muse_pipe object and do the plot check ups

# Importing modules
import numpy as np

# Standard modules
import os
from os.path import join as joinpath

from graph_pipe import graph_muse

__version__ = '0.0.1 (15 March 2018)'

from graph_pipe import graph_muse
from mpdaf_pipe import *
from musepipe import *

name_final_datacube = "DATACUBE_FINAL.fits"

class check_pipe(muse_pipe) :
    """Checking the outcome of the data reduction
    """
    def __init__(self, pdf_name="check_pipe.pdf", pipe=None, **kwargs) :
        if pipe is not None:
            self.__dict__.update(pipe.__dict__)
        else :
            muse_pipe.__init__(self, **kwargs)

        self.cube = muse_cube(filename=joinpath(self.paths.reduced, name_final_datacube))
        self.pdf = graph_muse(pdf_name=pdf_name)

        # getting standard spectra
        self.cube.get_set_spectra()

        # plotting all standard data
        self.check_quadrants()
        # plotting the white image and Ha image
        self.check_white_Ha_image()
        # closing the pdf
        self.pdf.close()

    def check_quadrants(self) :
        self.pdf.plot_page(self.cube.spec_4quad)

    def check_master_bias_flat(self) :
        """Checking the Master bias and Master flat
        """
        bias = muse_image(filename=joinpath(self.paths.masterbias, listMaster_dic['BIAS'][1]))
        flat = muse_image(filename=joinpath(self.paths.masterflat, listMaster_dic['FLAT'][1]))
        tocheck = museset_images(bias, flat, subtitle="Master Bias and Flat")
        self.pdf.plot_page(tocheck)

    def check_white_Ha_image(self, velocity=0.) :
        """Building the White and Ha images and 
        Adding them on the page
        """
        white = self.cube.get_whiteimage_from_cube()
        Ha = self.cube.get_emissionline_image(line="Ha", velocity=velocity)
        tocheck = museset_images(white, Ha, subtitle="White and Halpha images")
        self.pdf.plot_page(tocheck)
