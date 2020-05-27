# Licensed under a MIT style license - see LICENSE.txt

"""MUSE-PHANGS check pipeline module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"
# This module will take a MusePipe object and do the plot check ups

# Standard modules
from os.path import join as joinpath
import glob

__version__ = '0.0.4 (21 Feb 2019)'
# v0.0.4: Debugged a bit more with the new MusePipe structure
# v0.0.3: Debugged a bit the sequence
# v0.0.2: Added some import after moving MuseCube, MuseImage, etc
# v0.0.1: initial

from .graph_pipe import GraphMuse
from .musepipe import MusePipe
from .mpdaf_pipe import MuseCube, MuseSpectrum, MuseSetSpectra
from .mpdaf_pipe import MuseImage, MuseSetImages, get_sky_spectrum

name_final_datacube = "DATACUBE_FINAL.fits"
PLOT = '\033[1;34;20m'
ENDC = '\033[0m'

def print_plot(text) :
    print(PLOT + "# CheckPipeInfo " + ENDC + text)


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

        self.cube = MuseCube(filename=joinpath(self.paths.object, mycube))
        self.pdf = GraphMuse(pdf_name=joinpath(self.paths.figures, pdf_name))

        # Input parameters useful to define a set of spectra and images
        suffix_skyspectra = kwargs.pop("suffix_skyspectra", "")
        suffix_images = kwargs.pop("suffix_images", None)

        if standard_set :
            # getting standard spectra
            self.cube.get_set_spectra()

            # plotting all standard data
            # Page 1
            self.check_quadrants()
            # plotting the white image and Ha image
            # Page 2
            self.check_white_line_images(line="Ha")
            # plotting the sky spectra
            # Page 3
            self.check_sky_spectra(suffix_skyspectra)

            # Checking some images only if suffix_images is provided
            if suffix_images is not None:
                self.check_given_images(suffix_images)

            # closing the pdf
            self.pdf.close()

    def check_quadrants(self) :
        """Checking spectra from the 4 quadrants
        """
        print_plot("Plotting the 4 quadrants-spectra")
        self.pdf.plot_page(self.cube.spec_4quad)

    def check_master_bias_flat(self) :
        """Checking the Master bias and Master flat
        """
        bias = self.get_master(mastertype="Bias", scale='arcsinh', title="Master Bias")
        flat = self.get_master(mastertype="Flat", scale='arcsing', title="Master Flat")
        tocheck = MuseSetImages(bias, flat, subtitle="Master Bias - Master Flat")
        print_plot("Plotting the Master Bias and Flat")
        self.pdf.plot_page(tocheck)

    def check_white_line_images(self, line="Ha", velocity=0.) :
        """Building the White and Ha images and 
        Adding them on the page
        """
        white = self.cube.get_whiteimage_from_cube()
        linemap = self.cube.get_emissionline_image(line=line, velocity=velocity)
        tocheck = MuseSetImages(white, linemap, subtitle="White and emission line {0} images".format(line))
        print_plot("Plotting the White and {0} images".format(line))
        self.pdf.plot_page(tocheck)

    def check_sky_spectra(self, suffix) :
        """Check all sky spectra from the exposures
        """
        sky_spectra_names = glob.glob(self.paths.sky + "./SKY_SPECTRUM_*{suffix}.fits".format(suffix=suffix))
        tocheck = MuseSetSpectra(subtitle="Sky Spectra")
        counter = 1
        for specname in sky_spectra_names :
            tocheck.append(MuseSpectrum(source=get_sky_spectrum(filename=specname), title="Sky {0:2d}".format(counter),
                add_sky_lines=True))
            counter += 1

        print_plot("Plotting the sky spectra")
        self.pdf.plot_page(tocheck)

    def check_given_images(self, suffix=None) :
        """Check all images with given suffix
        """
        if suffix is None: suffix = ""
        image_names = glob.glob(self.paths.maps + "./*{0}*.fits".format(suffix))
        tocheck = MuseSetImages(subtitle="Given Images - {0}".format(suffix))
        counter = 1
        for imaname in image_names :
            tocheck.append(MuseImage(filename=imaname, title="Image {0:2d}".format(counter)))
            counter += 1

        print_plot("Plotting the set of given images")
        self.pdf.plot_page(tocheck)

