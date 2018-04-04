# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS check pipeline module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"
# This module will take a MusePipe object and do the plot check ups

# Standard modules
import os
from os.path import join as joinpath
import glob

__version__ = '0.0.1 (15 March 2018)'

from graph_pipe import GraphMuse
from mpdaf_pipe import *
from musepipe import *

name_final_datacube = "DATACUBE_FINAL.fits"

class CheckPipe(MusePipe) :
    """Checking the outcome of the data reduction
    """
    def __init__(self, mycube=name_final_datacube, pdf_name="check_pipe.pdf", 
            pipe=None, standard_set=True, **kwargs) :
        """Init of the CheckPipe class. Using a default datacube to run some checks
        and create some plots
        """
        if pipe is not None:
            self.__dict__.update(pipe.__dict__)
        else :
            MusePipe.__init__(self, **kwargs)

        self.cube = MuseCube(filename=joinpath(self.paths.cubes, mycube))
        self.pdf = GraphMuse(pdf_name=joinpath(self.paths.figures, pdf_name))

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
        tocheck = MuseSetImages(bias, flat, subtitle="Master Bias - Master Flat")
        self.pdf.plot_page(tocheck)

    def check_white_Ha_image(self, velocity=0.) :
        """Building the White and Ha images and 
        Adding them on the page
        """
        white = self.cube.get_whiteimage_from_cube()
        Ha = self.cube.get_emissionline_image(line="Ha", velocity=velocity)
        tocheck = MuseSetImages(white, Ha, subtitle="White and Halpha images")
        self.pdf.plot_page(tocheck)

    def check_sky_spectra(self) :
        """Check all sky spectra from the exposures
        """
        sky_spectra_names = glob.glob(self.paths.sky + "./SKY_SPECTRUM_????.fits")
        tocheck = MuseSetSpectra(subtitle="Sky Spectra")
        counter = 1
        for specname in sky_spectra_names :
            tocheck.append(MuseSpectrum(source=get_sky_spectrum(filename=specname), title="Sky {0:2d}".format(counter),
                add_sky_lines=True))
            counter += 1

        self.pdf.plot_page(tocheck)

    def check_Ha_images(self) :
        """Check all Ha images
        """
        Ha_image_names = glob.glob(self.paths.maps + "./img_ha?.fits")
        tocheck = MuseSetImages(subtitle="Ha Images")
        counter = 1
        for imaname in Ha_image_names :
            tocheck.append(MuseImage(filename=imaname, title="Ha {0:2d}".format(counter)))
            counter += 1

        self.pdf.plot_page(tocheck)

