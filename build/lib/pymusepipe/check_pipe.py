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
    def __init__(self, pdf_name="check_pipe.pdf", pipe=None, standard_set=True, **kwargs) :
        if pipe is not None:
            self.__dict__.update(pipe.__dict__)
        else :
            muse_pipe.__init__(self, **kwargs)

        self.cube = muse_cube(filename=joinpath(self.paths.reduced, name_final_datacube))
        self.pdf = graph_muse(pdf_name=pdf_name)

        if standard_set :
            # getting standard spectra
            self.cube.get_set_spectra()

            # plotting all standard data
            # Page 1
            self.check_quadrants()
            # plotting the white image and Ha image
            # Page 2
            self.check_white_Ha_image()
            # plotting the sky spectra
            # Page 3
            self.check_sky_spectra()
            # Checking the Ha reconstructed images
            self.check_Ha_images()

            # closing the pdf
            self.pdf.close()

    def check_quadrants(self) :
        """Checking spectra from the 4 quadrants
        """
        self.pdf.plot_page(self.cube.spec_4quad)

    def check_master_bias_flat(self) :
        """Checking the Master bias and Master flat
        """
        bias = self.get_master(mastertype="Bias", scale='arcsinh', title="Master Bias")
        flat = self.get_master(mastertype="Flat", scale='arcsing', title="Master Flat")
        tocheck = museset_images(bias, flat, subtitle="Master Bias - Master Flat")
        self.pdf.plot_page(tocheck)

    def check_white_Ha_image(self, velocity=0.) :
        """Building the White and Ha images and 
        Adding them on the page
        """
        white = self.cube.get_whiteimage_from_cube()
        Ha = self.cube.get_emissionline_image(line="Ha", velocity=velocity)
        tocheck = museset_images(white, Ha, subtitle="White and Halpha images")
        self.pdf.plot_page(tocheck)

    def check_sky_spectra(self) :
        """Check all sky spectra from the exposures
        """
        sky_spectra_names = glob.glob(self.paths.sky + "./SKY_SPECTRUM_????.fits")
        tocheck = museset_spectra(subtitle="Sky Spectra")
        counter = 1
        for specname in sky_spectra_names :
            tocheck.append(muse_spectrum(source=get_sky_spectrum(filename=specname), title="Sky {0:2d}".format(counter),
                add_sky_lines=True))
            counter += 1

        self.pdf.plot_page(tocheck)

    def check_Ha_images(self) :
        """Check all Ha images
        """
        Ha_image_names = glob.glob(self.paths.maps + "./img_ha?.fits")
        tocheck = museset_images(subtitle="Ha Images")
        counter = 1
        for imaname in Ha_image_names :
            tocheck.append(muse_image(filename=imaname, title="Ha {0:2d}".format(counter)))
            counter += 1

        self.pdf.plot_page(tocheck)

