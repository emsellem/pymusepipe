# Licensed under a MIT style license - see LICENSE.txt

"""MUSE-PHANGS plotting routines
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"

# This module provides some functions to plot and check the data reduction
#
# Eric Emsellem adapted a March 2018 version, provided by Rebecca
# and adapted it to the pymusepipe package

# Importing modules

# Standard modules

# Plotting routines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

from .mpdaf_pipe import MuseSetImages, MuseSetSpectra

__version__ = '0.0.1 (23 March 2018)'

############################################################
#                      BEGIN
# The following parameters can be adjusted for the need of
# the specific pipeline to be used
############################################################

############################################################
#                      END
############################################################
PLOT = '\033[1;35;20m'
ENDC = '\033[0m'

def print_fig(text) :
    print(PLOT + "# GraphPipeInfo " + ENDC + text)

#########################################################################
# Main class
#                           GraphMuse
#########################################################################
class GraphMuse(object): 
    """Graphic output to check MUSE data reduction products
    """
    
    def __init__(self, pdf_name='drs_check.pdf', 
            figsize=(10,14), rect_layout=[0, 0.03, 1, 0.95], verbose=True) :
        """Initialise the class for plotting the outcome results
        """
        self.verbose = verbose
        self.pdf_name = pdf_name
        self.pp = PdfPages(pdf_name)
        self.figsize = figsize
        self.rect_layout = rect_layout
        self.npages = 0

    def close(self) :
        self.pp.close()

    def savepage(self) :
        self.pp.savefig()
        plt.close()
        self.npages += 1

    def start_page(self) :
        """Start the page
        """
        if self.verbose :
            print_fig("Starting page {0}".format(self.npages+1))
        plt.figure(figsize=self.figsize)

    def plot_page(self, list_data) :
        """Plot a set of blocks, each made of a set of spectra or
        images. This is for 1 page
        It first counts the number of lines needed according to the
        separation for images (default is 2 per line, each image taking 2 lines)
        and spectra (1 spectrum per line over 2 columns)
        """
        if len(list_data) == 0 :
            print_fig("WARNING: datalist is empty, no plots will be created")
            return

        self.start_page()
        nspectra_blocks, nimages_blocks = 0, 0
        nlines = 0
        if isinstance(list_data, MuseSetSpectra) | isinstance(list_data, MuseSetImages) :
            plt.suptitle(list_data.subtitle)
            list_data = [list_data]

        for data in list_data :
            if isinstance(data, MuseSetSpectra) :
                nspectra_blocks += 1
                nlines += data.__len__()
            elif isinstance(data, MuseSetImages) :
                nimages_blocks += 1
                nlines += 2 * data.__len__()

        self.gs = gridspec.GridSpec(nlines, 2)
        self.list_ax = []
        self.count_lines = 0
        for data in list_data :
            if isinstance(data, MuseSetSpectra) :
                self.plot_set_spectra(data) 
            elif isinstance(data, MuseSetImages) :
                self.plot_set_images(data)

        self.savepage()

    def plot_set_spectra(self, set_of_spectra=None, ) :
        """Plotting a set of spectra
        Skipping the ones that are 'None'
        """

        # Set of sky lines to plot when relevant - add_sky_lines to True
        sky_lines = [5577., 6300., 6864., 7914., 8344., 8827.]

        if set_of_spectra is None :
            print_fig("ERROR: list of spectra is empty")
            return

        for spec in set_of_spectra :
            self.list_ax.append(plt.subplot(self.gs[self.count_lines,:]))
            self.count_lines += 1
            if spec is not None :
                spec.plot(title=spec.title, ax=self.list_ax[-1])
                if spec.add_sky_lines :
                    for line in sky_lines :
                        plt.axvline(x=line, color=spec.color_sky, linestyle=spec.linestyle_sky, alpha=spec.alpha_sky)

        plt.tight_layout(rect=self.rect_layout)

    def plot_set_images(self, set_of_images=None) :
        """Plotting a set of images
        Skipping the ones that are 'None'
        """
        if set_of_images is None :
            print_fig("ERROR: list of images is empty")
            return

        for i in range(set_of_images.__len__()) :
            image = set_of_images[i]
            count_cols = i%2
            if image is not None :
                self.list_ax.append(plt.subplot(self.gs[self.count_lines: self.count_lines + 2, count_cols]))
                image.plot(scale=image.scale, vmin=image.vmin, colorbar=image.colorbar, title=image.title, ax=self.list_ax[-1])
            self.count_lines += count_cols * 2

        plt.tight_layout(rect=self.rect_layout)
