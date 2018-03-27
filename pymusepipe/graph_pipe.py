# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS plotting routines
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# This module provides some functions to plot and check the data reduction
#
# Eric Emsellem adapted a March 2018 version, provided by Rebecca
# and adapted it to the pymusepipe package

# Importing modules
import numpy as np

# Standard modules

# Plotting routines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

from mpdaf_pipe import muse_image, muse_spectrum, museset_images, museset_spectra

__version__ = '0.0.1 (23 March 2018)'

############################################################
#                      BEGIN
# The following parameters can be adjusted for the need of
# the specific pipeline to be used
############################################################

############################################################
#                      END
############################################################


#########################################################################
# Main class
#                           muse_data()
#########################################################################
class graph_muse(object): 
    """Graphic output to check MUSE data reduction products
    """
    
    def __init__(self, folder_fig='./', name_of_figure='drs_check.pdf', 
            figsize=(10,14), rect_layout=[0, 0.03, 1, 0.95], verbose=True) :
        """Initialise the class for plotting the outcome results
        """
        self.verbose = verbose
        self.figure_name = name_of_figure
        self.pp = PdfPages(name_of_figure)
        self.figsize = figsize
        self.rect_layout = rect_layout
        self.npage = 0

    def savepage_and_close(self) :
        self.pp.savefig()
        self.pp.close()
        self.npage += 1

    def start_page(self) :
        """Start the page
        """
        if self.verbose :
            print("Starting page {0}".format(self.npage))
        plt.figure(figsize=self.figsize)

    def plot_page(self, list_data) :
        """Plot a set of blocks, each made of a set of spectra or
        images. This is for 1 page
        It first counts the number of lines needed according to the
        separation for images (default is 2 per line, each image taking 2 lines)
        and spectra (1 spectrum per line over 2 columns)
        """
        self.start_page()
        nblocks = len(list_data)
        nspectra_blocks, nimages_blocks = 0, 0
        nlines = 0
        if isinstance(list_data, museset_spectra) | isinstance(list_data, museset_images) :
            plt.suptitle(list_data.subtitle)
            list_data = [list_data]

        for data in list_data :
            if isinstance(data, museset_spectra) :
                nspectra_blocks += 1
                nlines += data.nspectra
            elif isinstance(data, museset_images) :
                nimages_blocks += 1
                nlines += 2 * data.nimages

        self.gs = gridspec.GridSpec(nlines, 2)
        self.list_ax = []
        self.count_lines = 0
        for data in list_data :
            if isinstance(data, museset_spectra) :
                self.plot_set_spectra(data) 
            elif isinstance(data, museset_images) :
                self.plot_set_images(data)

        self.savepage_and_close()

    def plot_set_spectra(self, set_of_spectra=None, add_sky_lines=False,
            color='red', ls='--', alpha=0.3) :
        """Plotting a set of spectra
        """

        # Set of sky lines to plot when relevant - add_sky_lines to True
        sky_lines = [5577., 6300., 6864., 7914., 8344., 8827.]

        if set_of_spectra is None :
            print("ERROR: list of spectra is empty")
            return

        nspec = len(set_of_spectra)

        for i in range(nspec) :
            self.list_ax.append(plt.subplot(self.gs[self.count_lines,:]))
            self.count_lines += 1
            set_of_spectra[i].plot(title=set_of_spectra[i].title, ax=self.list_ax[-1])
            if add_sky_lines :
                for line in sky_lines :
                    plt.axvline(x=line, color=color, linestyle=ls, alpha=alpha)

        plt.tight_layout(rect=self.rect_layout)

    def plot_set_images(self, list_of_images=None, scales=None, titles=None) :
        """Plotting a set of images
        """
        if self.verbose :
            print("Starting Page {0}".format(self.npage + 1))

        if list_of_images is None :
            print("ERROR: list of images is empty")
            return

        nima = len(list_of_images)
        if scales is None : myscales = ['log'] * nima
        else : myscales = scales

        if len(scales) != nima :
            print("ERROR: scales should have the same number of items than images")
            return

        if titles is None : mytitles = ['Frame {0}'.format(i+1) for i in range(nima)]
        else :
            if len(titles) != nima :
                print("ERROR: titles should have the same number of items than images")
                return
            mytitles = ['Frame {0} - {1}'.format(i+1, titles[i]) for i in range(nima)]

        for i in range(nima) :
            self.count_cols = i%2
            self.list_ax.append(plt.subplot(self.gs[self.count_lines:self.count_lines+2,self.counts_cols:1-self.count_cols]))
            self.count_lines += 2
            image = list_of_images[i]
            image.plot(scale=image.scale, vmin=image.vmin, colorbar=image.colorbar, title=image.title, ax=self.list_ax[-1])

        plt.tight_layout(rect=self.rect_layout)
