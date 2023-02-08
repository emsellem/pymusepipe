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
import numpy as np

# Standard modules
from os.path import join as joinpath

# Plotting routines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

from .mpdaf_pipe import MuseSetImages, MuseSetSpectra
from .util_image import my_linear_model, get_flux_range

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


def open_new_wcs_figure(nfig, mywcs=None):
    """Open a new figure (with number nfig) with given wcs.
    If not WCS is provided, just opens a subplot in that figure.

    Input
    -----
    nfig : int
        Figure number to consider
    mywcs : astropy.wcs.WCS
        Input WCS to open a new figure (Default value = None)

    Returns
    -------
    fig, subplot
        Figure itself with the subplots using the wcs projection

    """

    # get the figure
    fig = plt.figure(nfig)

    # Clean the figure
    plt.clf()

    # Adding axes with WCS
    if mywcs is None:
        return fig, fig.add_subplot(1, 1, 1)
    else:
        return fig, fig.add_subplot(1, 1, 1, projection=mywcs)

def plot_compare_contours(data1, data2, plotwcs=None, labels=('Data1', 'Data2'), levels=None,
                          nlevels=10, fignum=1, namefig='dummy_contours.png', figfolder="",
                          savefig=False, **kwargs):
    """Creates a plot with the contours of two input datasets for comparison

    Input
    -----
    data1
    data2: 2d np.arrays
        Input arrays to compare
    plotwcs: WCS
        WCS used to set the plot if provided
    labels: tuple/list of 2 str
        Labels for the plot
    levels: list of floats
        Levels to be used for the contours. Calculated if None.
    fignum: int
        Number for the figure
    namefig: str
        Name of the figure to be saved (if savefig is True)
    figfolder: str
        Name of the folder for the figure
    savefig: bool
        If True, will save the figure as namefig

    Creates
    -------
    Plot with contours of the two input dataset
    """
    np.seterr(divide='ignore', invalid='ignore')

    # Getting the range of relevant fluxes
    lowlevel_d1, highlevel_d1 = get_flux_range(data1)

    fig, ax = open_new_wcs_figure(fignum, plotwcs)

    # Defining the levels for MUSE
    if levels is not None:
        levels_d1 = levels
    else:
        levels_d1 = np.linspace(np.log10(lowlevel_d1),
                                  np.log10(highlevel_d1),
                                  nlevels)
    # Plot contours for MUSE
    cdata1 = ax.contour(np.log10(data1), levels_d1, colors='k',
                          origin='lower', linestyles='solid')

    levels_d2 = cdata1.levels
    # Plot contours for Ref
    cdata2 = ax.contour(np.log10(data2), levels=levels_d2, colors='r',
                        origin='lower', alpha=0.5, linestyles='solid')

    ax.set_aspect('equal')
    h1, _ = cdata1.legend_elements()
    h2, _ = cdata2.legend_elements()
    ax.legend([h1[0], h2[0]], [labels[0], labels[1]])

    if "title" in kwargs:
        plt.title(kwargs.pop('title'))
    plt.tight_layout()
    if savefig:
        plt.savefig(joinpath(figfolder, namefig))
    np.seterr(divide='warn', invalid='warn')

def plot_compare_diff(data1, data2, plotwcs=None, figfolder="", percentage=5, fignum=1,
                      namefig="dummy_diff.ong", savefig=False, **kwargs):
    """

    Parameters
    ----------
    data1
    data2
    figfolder
    fignum
    namefig
    savefig
    kwargs

    Returns
    -------
    """
    fig, ax = open_new_wcs_figure(fignum, plotwcs)
    ratio = 100. * (data2 - data1) / (data1 + 1.e-12)
    im = ax.imshow(ratio, vmin=-percentage, vmax=percentage)
    cbar = fig.colorbar(im, shrink=0.8)

    if "title" in kwargs:
        plt.title(kwargs.pop('title'))
    plt.tight_layout()
    if savefig:
        plt.savefig(joinpath(figfolder, namefig))


def plot_compare_cuts(data1, data2, labels=('X', 'Y'), figfolder="", fignum=1,
                 namefig="dummy_polypar.png", ncuts=11, savefig=False, **kwargs):
    """

    Input
    -----
    data1
    data2
    label1
    label2
    figfolder
    fignum
    namefig
    savefig
    kwargs

    Creates
    -------
    Plot with a comparison of the two data arrays using regular X and Y cuts
    """

    fig, ax = open_new_wcs_figure(fignum)

    # Getting the range of relevant fluxes
    lowlevel_d1, highlevel_d1 = get_flux_range(data1)

    diffima = (data2 - data1) * 200. / (lowlevel_d1 + highlevel_d1)
    chunk_x = data1.shape[0] // (ncuts + 1)
    chunk_y = data1.shape[1] // (ncuts + 1)

    c1 = ax.plot(diffima[np.arange(ncuts) * chunk_x, :].T, 'k-', label=labels[0])
    c2 = ax.plot(diffima[:, np.arange(ncuts) * chunk_y], 'r-', label=labels[1])
    ax.legend(handles=[c1[0], c2[0]], loc=0)
    ax.set_ylim(-20, 20)
    ax.set_xlabel("[pixels]", fontsize=20)
    ax.set_ylabel("[%]", fontsize=20)

    if "title" in kwargs:
        plt.title(kwargs.pop('title'))
    plt.tight_layout()
    if savefig:
        plt.savefig(joinpath(figfolder, namefig))

def plot_polypar(polypar, labels=("Data 1","Data 2"), figfolder="", fignum=1,
                 namefig="dummy_polypar.png", savefig=False, **kwargs):
    """Creating a plot showing the normalisation arising from a polypar object
    
    Parameters
    ----------
    polypar
    label1
    label2
    foldfig
    namefig

    Returns
    -------

    """
    # Opening the figure
    fig, ax = open_new_wcs_figure(fignum)

    # Getting the x, y to plot
    (x, y) = (polypar.med[0][polypar.selection],
              polypar.med[1][polypar.selection])
    ax.plot(x, y, '.')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.plot(x, my_linear_model(polypar.beta, x), 'k')

    if "title" in kwargs:
        plt.title(kwargs.pop('title'))
    plt.tight_layout()
    if savefig:
        plt.savefig(joinpath(figfolder, namefig))

            
#########################################################################
# Main class
#                           GraphMuse
#########################################################################
class GraphMuse(object): 
    """Graphic output to check MUSE data reduction products
    """
    
    def __init__(self, pdf_name='drs_check.pdf', 
            figsize=(10,14), rect_layout=[0, 0.03, 1, 0.95], verbose=True):
        """Initialise the class for plotting the outcome results
        """
        self.verbose = verbose
        self.pdf_name = pdf_name
        self.pp = PdfPages(pdf_name)
        self.figsize = figsize
        self.rect_layout = rect_layout
        self.npages = 0

    def close(self):
        self.pp.close()

    def savepage(self):
        self.pp.savefig()
        plt.close()
        self.npages += 1

    def start_page(self):
        """Start the page
        """
        if self.verbose :
            print_fig("Starting page {0}".format(self.npages+1))
        plt.figure(figsize=self.figsize)

    def plot_page(self, list_data):
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
        if isinstance(list_data, MuseSetSpectra) | isinstance(list_data, MuseSetImages):
            plt.suptitle(list_data.subtitle)
            list_data = [list_data]

        for data in list_data :
            if isinstance(data, MuseSetSpectra):
                nspectra_blocks += 1
                nlines += data.__len__()
            elif isinstance(data, MuseSetImages):
                nimages_blocks += 1
                nlines += 2 * data.__len__()

        self.gs = gridspec.GridSpec(nlines, 2)
        self.list_ax = []
        self.count_lines = 0
        for data in list_data :
            if isinstance(data, MuseSetSpectra):
                self.plot_set_spectra(data) 
            elif isinstance(data, MuseSetImages):
                self.plot_set_images(data)

        self.savepage()

    def plot_set_spectra(self, set_of_spectra=None):
        """Plotting a set of spectra
        Skipping the ones that are 'None'
        """

        # Set of sky lines to plot when relevant - add_sky_lines to True
        sky_lines = [5577., 6300., 6864., 7914., 8344., 8827.]

        if set_of_spectra is None:
            print_fig("ERROR: list of spectra is empty")
            return

        for spec in set_of_spectra:
            self.list_ax.append(plt.subplot(self.gs[self.count_lines,:]))
            self.count_lines += 1
            if spec is not None:
                spec.plot(title=spec.title, ax=self.list_ax[-1])
                if spec.add_sky_lines:
                    for line in sky_lines:
                        plt.axvline(x=line, color=spec.color_sky, linestyle=spec.linestyle_sky, alpha=spec.alpha_sky)

        plt.tight_layout(rect=self.rect_layout)

    def plot_set_images(self, set_of_images=None):
        """Plotting a set of images
        Skipping the ones that are 'None'
        """
        if set_of_images is None :
            print_fig("ERROR: list of images is empty")
            return

        for i in range(set_of_images.__len__()):
            image = set_of_images[i]
            count_cols = i%2
            if image is not None:
                self.list_ax.append(plt.subplot(self.gs[self.count_lines: self.count_lines + 2, count_cols]))
                image.plot(scale=image.scale, vmin=image.vmin, colorbar=image.colorbar, title=image.title, ax=self.list_ax[-1])
            self.count_lines += count_cols * 2

        plt.tight_layout(rect=self.rect_layout)
