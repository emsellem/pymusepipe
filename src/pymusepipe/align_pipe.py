#Creating a plot showing the normalisation arising from a polypar object Licensed under a MIT license - see LICENSE

"""MUSE-PHANGS alignement module. This module can be used to align MUSE
reconstructed images either with each others or using a reference background
image. It spits the results out in a Fits table which can then be used
to process and mosaic Muse PIXTABLES via the MUSE ESO pipeline. 
It includes a normalisation factor, an estimate of the background, 
as well as any potential rotation. Fine tuning
can be done by hand by the user, using a set of reference plots.
"""

__authors__ = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__ = "MIT License"
__contact__ = " <eric.emsellem@eso.org>"

# Import general modules
import os
from os.path import join as joinpath

import glob
import copy

# Import Matplotlib
import matplotlib.pyplot as plt

# Import Numpy Scipy
import numpy as np
from scipy.signal import correlate

# Astropy
from astropy import wcs as awcs
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.modeling import models, fitting
from astropy.table import Table, Column
from astropy import units as u

# Import mpdaf
from mpdaf.obj import Image, WCS

# Import needed modules from pymusepipe
from . import util_pipe as upipe  # noqua: E402
from .config_pipe import mjd_names, date_names, tpl_names
from .config_pipe import dataset_names, iexpo_names
from .config_pipe import default_offset_table, dict_listObject
from .graph_pipe import (plot_polypar, plot_compare_contours,
                         plot_compare_cuts, plot_compare_diff)
from .util_image import my_linear_model, flatclean_image, get_normfactor, mask_point_sources


try:
    import reproject
    from reproject import reproject_interp as repro_interp
    from reproject import reproject_exact as repro_exact
except ImportError:
    upipe.print_warning("If you wish to use reproject, please install "
                        "it via: pip install reproject.")


# Skimage if you have it
try:
    import skimage
    from skimage.registration import phase_cross_correlation
    from skimage import transform
except ImportError:
    upipe.print_warning("If you wish to use skimage for image registration, "
                        "please install it via "
                        "    pip install scikit-image "
                        "or  conda install scikit-image")


# Spacepylot
try:
    import spacepylot as spp
    import spacepylot.alignment as sppalign
    import spacepylot.plotting as spppl
    from spacepylot.alignment_utilities import TranslationTransform
except ImportError:
    upipe.print_warning("If you wish to use spacepylot please install it via"
                        "pip install or conda install or cloning the github version")

def is_sequence(arg):
    """Test if sequence and return the boolean result

    Parameters
    ----------
    arg : input argument


    Returns
    -------
    result: boolean

    """

    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))


# ================== Default units ======================== #
# Define useful units
default_muse_unit = u.erg / (u.cm * u.cm * u.second * u.AA) * 1.e-20
default_reference_unit = u.microJansky

dict_equivalencies = {"WFI_BB": u.spectral_density(6483.58 * u.AA),
                      "DUPONT_R": u.spectral_density(6483.58 * u.AA)}
# ================== Useful function ====================== #


def create_offset_table(image_names, table_folder="",
                        table_name="dummy_offset_table.fits", overwrite=False):
    """Create an offset list table from a given set of images. It will use
    the MJD and DATE as read from the descriptors of the images. The names for
    these keywords is stored in the dictionary default_offset_table from
    config_pipe.py

    Parameters
    ----------
    image_names : list of str
        List of image names to be considered. (Default value = [])
    table_folder : str
        folder of the table (Default value = "")
    table_name : str
        name of the table to save ['dummy_offset_table.fits']
        (Default value = "dummy_offset_table.fits")
    overwrite : bool
        if the table exists, it will be overwritten if set
        to True only. (Default value = False)
    overwrite : bool
        if the table exists, it will be overwritten if set
        to True only. (Default value = False)

    Returns
    -------
        A fits table with the output given name. (Default value = False)
    """

    # Check if table exists and see if overwrite is set up
    table_fullname = joinpath(table_folder, table_name)
    if not overwrite and os.path.isfile(table_fullname):
        upipe.print_warning("[create_offset_table] Table {0} "
                            "already exists".format(table_fullname))
        upipe.print_warning("Use overwrite=True if you wish to proceed")
        return

    nimages = len(image_names)
    if nimages == 0:
        upipe.print_warning("No image names provided for create_offset_table")
        return

    # Gather the values of DATE and MJD from the images
    date, mjd, tpls, iexpo, dataset = [], [], [], [], []
    for ima in image_names:
        if not os.path.isfile(ima):
            upipe.print_warning("[create_offset] Image {0} does not exists".format(ima))
            continue

        head = pyfits.getheader(ima)
        date.append(head[date_names['image']])
        mjd.append(head[mjd_names['image']])
        tpls.append(head[tpl_names['image']])
        iexpo.append(head[iexpo_names['image']])
        dataset.append(head[dataset_names['image']])

    nlines = len(date)

    # Create and fill the table
    offset_table = Table()
    for col in default_offset_table:
        [name, form, default] = default_offset_table[col]
        offset_table[name] = Column([default for i in range(nlines)],
                                    dtype=form)

    offset_table[date_names['table']] = date
    offset_table[mjd_names['table']] = mjd
    offset_table[tpl_names['table']] = tpls
    offset_table[iexpo_names['table']] = iexpo
    offset_table[dataset_names['table']] = dataset

    # Write the table
    offset_table.write(table_fullname, overwrite=overwrite)


def get_conversion_factor(input_unit, output_unit, filter_name="WFI"):
    """ Conversion of units from an input one
    to an output one
     
    Input
    -----
    input_unit: astropy unit
        Input astropy unit to analyse
    output_unit: astropy unit
        Astropy unit to compare to input unit.
    equivalencies: astropy equivalency
        Used in case there is an existing equivalency
        to help the conversion
    
    Returns
    -------
    conversion: astropy unit conversion
    """

    if filter_name not in dict_equivalencies:
        upipe.print_warning("Didn't find conversion for this filter. "
                            "Using 1.0 as a conversion factor")
        return 1.0

    equivalencies = dict_equivalencies[filter_name]
    # First testing if the quantities are Quantity
    # If not, transform them 
    if not isinstance(input_unit, u.quantity.Quantity):
        if not isinstance(input_unit, (u.core.Unit, u.core.CompositeUnit)):
            upipe.print_warning("Input provided unit could not be converted")
            upipe.print_warning("Using 1.0 as a conversion factor")
            return 1.0
        else:
            input_unit = input_unit * 1.0
    if not isinstance(output_unit, u.quantity.Quantity):
        if not isinstance(output_unit, (u.core.Unit, u.core.CompositeUnit)):
            upipe.print_warning("Output provided unit could not be converted")
            upipe.print_warning("Using 1.0 as a conversion factor")
            return 1.0
        else:
            output_unit = output_unit * 1.0

    if not input_unit.unit.is_equivalent(output_unit):
        # if not equivalent we try a spectral density equivalence
        if not input_unit.unit.is_equivalent(output_unit,
                                             equivalencies=equivalencies):
            upipe.print_warning("Provided units for reference "
                                "and MUSE images are not equivalent")
            upipe.print_warning("A conversion factor of 1.0 will thus be used")
            return 1.0
        else:
            return input_unit.unit.to(
                output_unit, equivalencies=equivalencies) * input_unit.value
    else:
        return input_unit.unit.to(output_unit) * input_unit.value


def arcsec_to_pixel(hdu, xy_arcsec=(0., 0.)):
    """Transform from arcsec to pixel for the muse image
    using the hdu to extract the WCS, hence the scaling.
     
    Input
    -----
    hdu: astropy hdu (fits)
        Input hdu which includes a WCS
    xy_arcsec: list of 2 floats ([0,0])
        Coordinates to transform from arcsec to pixel.
    
    Returns
    -------
    xpix, ypix: tuple or list of 2 floats
        Pixel coordinates

    See also: pixel_to_arcsec (align_pipe.py)
    """
    # Matrix
    input_wcs = awcs.WCS(hdu.header)
    scale_matrix = np.linalg.inv(input_wcs.pixel_scale_matrix * 3600.)

    # Transformation in Pixels
    dels = np.asarray(xy_arcsec)
    xpix = np.sum(dels * scale_matrix[0, :])
    ypix = np.sum(dels * scale_matrix[1, :])
    return xpix, ypix


def pixel_to_arcsec(hdu, xy_pixel=(0., 0.)):
    """Transform from arcsec to pixel for the muse image
    using the hdu to extract the WCS, hence the scaling.
     
    Input
    -----
    hdu: astropy hdu (fits)
        Input hdu which includes a WCS
    xy_pixel: tuple or list of 2 floats ((0,0))
        Coordinates to transform from pixel to arcsec
    
    Returns
    -------
    xarc, yarc: 2 floats
        Arcseconds coordinates

    See also: arcsec_to_pixel (align_pipe.py)
    """
    # Matrix
    input_wcs = awcs.WCS(hdu.header)

    # Transformation in arcsecond
    dels = np.asarray(xy_pixel, dtype=np.float64)
    xarc = np.sum(dels * input_wcs.pixel_scale_matrix[0, :] * 3600.)
    yarc = np.sum(dels * input_wcs.pixel_scale_matrix[1, :] * 3600.)
    return xarc, yarc


def rotate_pixtables(folder="", name_suffix="", list_ifu=None,
                     angle=0., **kwargs):
    """Will update the derotator angle in each of the 24 pixtables
    Using a loop on rotate_pixtable

    Will thus update the HIERARCH ESO INS DROT POSANG keyword.

    Input
    -----
    folder: str
        name of the folder where the PIXTABLE are
    name_suffix: str
        name of the suffix to be used on top of PIXTABLE_OBJECT
    list_ifu: list[int]
        List of Pixtable numbers. If None, will do all 24
    angle: float
        Angle to rotate (in degrees)
    """

    if list_ifu is None:
        list_ifu = np.arange(24) + 1

    for nifu in list_ifu:
        rotate_pixtable(folder=folder, name_suffix=name_suffix, nifu=nifu,
                        angle=angle, **kwargs)


def rotate_pixtable(folder="", name_suffix="", nifu=1, angle=0., **kwargs):
    """Rotate a single IFU PIXTABLE_OBJECT
    Will thus update the HIERARCH ESO INS DROT POSANG keyword.

    Input
    -----
    folder: str
        name of the folder where the PIXTABLE are
    name_suffix: str
        name of the suffix to be used on top of PIXTABLE_OBJECT
    nifu: int
        Pixtable number. Default is 1
    angle: float
        Angle to rotate (in degrees)
    """
    angle_keyword = "HIERARCH ESO INS DROT POSANG"
    angle_orig_keyword = "{0} ORIG".format(angle_keyword)

    pixtable_basename = kwargs.pop("pixtable_basename",
                                   dict_listObject['OBJECT'])
    prefix = kwargs.pop("prefix", "")
    name_pixtable = "{0}{1}_{2}-{3:02d}.fits".format(prefix, pixtable_basename,
                                                     name_suffix, int(nifu))
    fullname_pixtable = joinpath(folder, name_pixtable)
    fakemode = kwargs.pop("fakemode", False)

    # Check if table is there
    if not os.path.isfile(fullname_pixtable):
        upipe.print_error("Input Pixtable {0} does not exist - Aborting".format(
            fullname_pixtable))
        return

    # Continue with updating the table
    mypix = pyfits.open(fullname_pixtable, mode='update')
    hd = mypix[0].header
    if not fakemode and angle != 0.0:
        if not angle_orig_keyword in hd:
            hd[angle_orig_keyword] = hd[angle_keyword]
        hd[angle_keyword] = hd[angle_orig_keyword] + angle
        upipe.print_info("Updating INS DROT POSANG for {0}".format(name_pixtable))
        mypix.flush()

    # Reading the result and printing
    print("=== {} === ".format(name_pixtable), end="")
    if not angle_orig_keyword in hd:
        print("Present Angle [No Change] (deg): {}".format(hd[angle_keyword]))
    else:
        print("Orig / New / Rotation Angle (deg): {0:8.4f} / {1:8.4f} / {2:8.4f}".format(
            hd[angle_orig_keyword],
            hd[angle_keyword],
            float(hd[angle_keyword]) - hd[angle_orig_keyword]))


def align_hdu(hdu_target=None, hdu_to_align=None, target_rotation=0.0, to_align_rotation=0.0,
              conversion_factor=1.0, use_mpdaf=False):
    """Project the reference image onto the MUSE dataset
    Hidden function, as only used internally

    Input
    -----
    hdu_target: HDU [None]
        Target hdu (on to which to project)
    hdu_to_align: HDU [None]
         Hdu to be aligned
    target_rotation: float [0]
        Rotation angle in degrees of the target hdu
    to_align_rotation: float [0]
        Rotation angle in degrees of the to be aligned hdu
    conversion_factor: float
        Factor to be applied to the to_align hdu
    use_mpdaf: bool
        If True, use mpdaf to project. This is not recommended.
        If False, use reproject. This is the recommended option (default)

    Returns
    -------
    hdu_repr: HDU
        Reprojected HDU. None if nothing is provided
    """
    if hdu_target is not None and hdu_to_align is not None:
        # Getting the reference image data and WCS
        wcs_to_align = WCS(hdr=hdu_to_align.header)
        # If there is a rotation of the WCS we remove it
        if to_align_rotation != 0.:
            wcs_to_align.rotate(-to_align_rotation)
        ima_to_align = Image(data=hdu_to_align.data * conversion_factor,
                             wcs=wcs_to_align)

        # Apply differential RA if using MPDAF to fix the reference
        # Problem existing when using align_with_image
        if use_mpdaf:
            ra_target = WCS(hdu_target.header).to_header()['CRVAL1']
            ra_to_align = ima_to_align.wcs.to_header()['CRVAL1']
            dec_to_align = ima_to_align.wcs.to_header()['CRVAL2']
            dra = ra_target - ra_to_align
            diffang = np.rad2deg(np.arccos(np.cos(np.deg2rad(dra))
                                           * (np.sin(np.deg2rad(dec_to_align))) ** 2
                                           + (np.cos(np.deg2rad(dec_to_align))) ** 2))
            upipe.print_warning(f"Differential angle for this comparison is "
                                f"{diffang:.4f} (using mpdaf needs that fix)")
        else:
            diffang = 0.

        # Getting the MUSE image data and WCS
        wcs_target = WCS(hdr=hdu_target.header)

        # Fixing the differential angle when using mpdaf
        # For repro, the initial value is correct. For mpdaf it needs
        # the correction as the reference RA is different when projecting
        # - namely = keeping the original RA as a reference -
        fixed_target_rotation = target_rotation - diffang

        # Doing the rotation now on the target WCS
        if fixed_target_rotation != 0.:
            wcs_target.rotate(-fixed_target_rotation)
        ima_target = Image(data=np.nan_to_num(hdu_target.data),
                           wcs=wcs_target)
        hdu_rot_target = ima_target.get_data_hdu()

        # Aligning the reference image with the MUSE image using mpdaf
        # align_with_image

        if use_mpdaf:
            ima_aligned = ima_to_align.align_with_image(ima_target, flux=True)
            hdu_aligned = ima_aligned.get_data_hdu()
        else:
            # Change of area
            newinc = ima_target.wcs.get_axis_increments(unit=u.deg)
            oldinc = ima_to_align.wcs.get_axis_increments(unit=u.deg)
            change_area = np.abs(newinc[0] / oldinc[0]) \
                          * np.abs(newinc[1] / oldinc[1])
            daligned = repro_interp(ima_to_align.get_data_hdu(),
                                    ima_target.get_data_hdu().header,
                                    return_footprint=False)
            hdu_aligned = pyfits.PrimaryHDU(daligned * change_area)

    else:
        hdu_aligned = None
        diffang = 0.
        hdu_rot_target = copy.copy(hdu_target)
        print("Warning: please provide target HDU to allow reprojection")

    return hdu_rot_target, hdu_aligned, diffang


def init_plot_optical_flow(opflow):
    """Initialise the optical flow plot using the AlignmentPlotting

    Input
    -----
    opflow: optical flow instance (see spacepylot)

    Returns
    -------
    An optical flow plot instance
    """
    # Initialise the plot
    return spppl.AlignmentPlotting.from_align_object(opflow)


#################################################################
# ================== END Useful functions ===================== #
#################################################################
# Main alignment Class
class AlignMuseDataset(object):
    """Class to align MUSE images onto a reference image.
    """

    def __init__(self, name_reference,
                 folder_reference=None,
                 folder_muse_images=None,
                 name_muse_images=None,
                 sel_indices_images=None,
                 median_window=10,
                 subim_window=10,
                 dynamic_range=10,
                 border=10, hdu_ext=(0, 1),
                 chunk_size=15,
                 firstguess="crosscorr",
                 **kwargs):
        """Initialise the AlignMuseImages class.
        
        Input
        -----
        name_reference: str
            Name of the reference image (fits file)
        folder_reference: str [""]
            Folder name of the reference image
        folder_muse_images: str [""]
            Folder name for the input images to compare
        name_muse_images: str or list
            List of names for the MUSE images (or str if only 1 image)
        suffix_muse_images: str
            Suffix to be used for muse image names if only a subset should be selected.
        filter_name: str
            Name of filter to consider when filtering the muse image names
        firstguess: str
            If "crosscorr", will use cross-correlation to guess the alignment offsets.
            If "fits", will use input fits OFFSET table to start with
            If set to None explicitly, will put 0's as start offsets.
            Default is "crosscorr".
        name_offset_table: str
            Name of the fits OFFSET table. Only used as input as guess if "firstguess"
            is set to "fits". This name will be the default name when saving
            the offsets in an OFFSET fits table.
        sel_indices_images: list [None]
            List of images to select from the given set
        median_window: int [10]
            Size of window used in median filter to extract features in
            cross-correlation.  Should be an odd integer
        subim_window: int [10]
            Size of window for fitting peak of cross-correlation function
        dynamic_range: float [10]
            Apply an arctan transform to data to suppress values more than
            DynamicRange times the median of the image
        border: int [10]
            Ignore pixels this close to the border in the cross correlation
        hdu_ext: tuple or list of 2 floats (0,1)
            Number of the extension for the input reference image 
            and images to align, respectively. This is used to know
            where the data lies in the fits file.
        chunk_size: int [15]
            Size in pixels of the chunk used to bin and compute the 
            normalisation factors

        Other keywords
        --------------
        threshold_muse: float [0]
            Minimum threshold value to consider for the input
            images to align
        verbose: bool [True]
            If True, spits out more verbose output
        plot: bool [True]
            If True, will provide plots
        debug: bool [False]
            If True, will provide some info to debug
            which will be stored in the python class
            as new attributes. Look for self._temp...
        use_polynorm: bool [True]
            Save the polynomial fitted slope and use as normalisation
            factors
        convert_units: bool [True]
            Use the given units to convert fluxes
        ref_unit: astropy unit
            Reference image unit
        muse_unit: astropy unit
            Input MUSE flux unit
        minflux_crosscorr: float [0]
            Minimum flux to consider when doing the cross-correlation.
        """

        # Some input variables for the cross-correlation
        self.verbose = kwargs.pop("verbose", True)
        self.plot = kwargs.pop("plot", True)

        # Using mpdaf or image_registration - Default is False
        # as mpdaf does not define the output WCS homogeneously
        self.use_mpdaf = kwargs.pop("use_mpdaf", False)
        if self.use_mpdaf:
            upipe.print_info("Will use mpdaf for image regridding.")
            upipe.print_info("WARNING: when using mpdaf, a potential extra rotation \n"
                             "has to be applied to redefine the reference grid.")
        else:
            upipe.print_info("Will use image_registration for image regridding.")

        # Set of input parameters for the image processing
        self.border = int(border)
        self.chunk_size = int(chunk_size)
        self.subim_window = subim_window
        self.median_window = median_window
        self.dynamic_range = dynamic_range

        self.name_reference = name_reference
        default_folder = os.getcwd()
        if folder_reference is None:
            folder_reference = default_folder
        self.folder_reference = folder_reference

        # Debug option
        self._debug = kwargs.pop("debug", False)
        if self._debug:
            upipe.print_warning("In DEBUG Mode [more printing]")
        # Backward compatibility - TO BE REMOVED when fixed
        self._backward_comp = kwargs.pop("backward_comp", True)

        # Check if folder reference exists
        if not os.path.isdir(self.folder_reference):
            upipe.print_error("Provided folder_reference is "
                              "not an existing folder")
            return

        self.name_muse_images = name_muse_images
        self.sel_indices_images = sel_indices_images
        if folder_muse_images is None:
            folder_muse_images = default_folder
        self.folder_muse_images = folder_muse_images
        # Check if folder muse images exists
        if not os.path.isdir(self.folder_muse_images):
            upipe.print_error("Provided folder_muse_images is "
                              "not an existing folder")
            return

        # Creating the Header-Align folder
        header_folder_name = kwargs.pop("header_folder_name", "AlignHeaders")
        self.header_folder_name = joinpath(self.folder_muse_images, header_folder_name)
        upipe.safely_create_folder(self.header_folder_name, verbose=self.verbose)

        # Creating the Figure folder if needed
        figures_folder_name = kwargs.pop("figures_folder_name", "AlignFigures")
        self.figures_folder_name = joinpath(self.folder_muse_images, figures_folder_name)
        upipe.safely_create_folder(self.figures_folder_name, verbose=self.verbose)

        # Getting the names
        self.save_hdr = kwargs.pop("save_hdr", False)
        self.name_musehdr = kwargs.pop("name_musehdr", "muse")
        self.name_offmusehdr = kwargs.pop("name_offmusehdr", "offsetmuse")
        self.suffix_images = kwargs.pop("suffix_muse_images", "IMAGE_FOV")
        self.filter_name = kwargs.pop("filter_name", "Cousins_R")

        # Use polynorm or not
        self.use_polynorm = kwargs.pop("use_polynorm", True)

        # Use rotation angles from the offset table if they exist
        self.use_rotangles = kwargs.pop("use_rotangles", True)
        if self.use_rotangles:
            upipe.print_warning("By default, rotation angles given in initial "
                                "offset table will be used if they exist. If "
                                "not, all initial rotation angles will be set to 0.")

        # Getting the unit conversions
        self.convert_units = kwargs.pop("convert_units", True)
        if self.convert_units:
            self.ref_unit = kwargs.pop("ref_unit", default_reference_unit)
            self.muse_unit = kwargs.pop("muse_unit", default_muse_unit)
            self.conversion_factor = get_conversion_factor(self.ref_unit,
                                                           self.muse_unit,
                                                           self.filter_name)
        else:
            self.conversion_factor = 1.0

        # Initialise the parameters for the first guess
        self.phase_corr = kwargs.pop("phase_corr", True)
        self.phase_subsamp = kwargs.pop("phase_subsamp", 10)
        self.folder_offset_table = kwargs.pop("folder_offset_table",
                                              self.folder_muse_images)
        self.folder_output_table = kwargs.pop("folder_output_table",
                                              self.folder_offset_table)
        self.name_offset_table = kwargs.pop("name_offset_table", None)
        self.minflux_crosscorr = kwargs.pop("minflux_crosscorr", 0.)

        # Get the MUSE images
        self._get_list_muse_images()
        upipe.print_info("{0} MUSE images detected as input".format(
            self.nimages))
        if self.nimages == 0:
            upipe.print_error("No MUSE images detected. Aborted")
            return
        self.default_threshold_muse = kwargs.pop("threshold_muse", 0)

        # Reset! all parameters
        self._reset_init_guess_values()

        # Initialise the arrays
        self._init_alignment_arrays()

        # Which extension to be used for the ref and muse images
        self.hdu_ext = hdu_ext

        # Open the Ref and MUSE image
        status_open = self.open_hdu()
        if not status_open:
            upipe.print_error("Problem in opening frames, please check your input")
            return

        # Initialise the offsets using the cross-correlation or FITS table
        self.init_guess_offset(firstguess=firstguess)

        # Now doing the shifts and projections with the guess/input values
        for nima in range(self.nimages):
            self._apply_alignment_ima(nima)

    def show_norm_factors(self):
        """Print some information about the normalisation factors.
        """
        upipe.print_info("Normalisation factors")
        upipe.print_info("Image # : InitFluxScale     SlopeFit       "
                         "NormFactor       Background")
        for nima in range(self.nimages):
            upipe.print_info("Image {0:03d}:  {1:10.6e}   {2:10.6e}     "
                             "{3:10.6e}     {4:10.6e}".format(
                nima,
                self.init_flux_scale[nima],
                self.ima_polypar[nima].beta[1],
                self.ima_norm_factors[nima],
                self.ima_background[nima]))

    def show_linearfit_values(self):
        """Print some information about the linearly fitted parameters
        pertaining to the normalisation.
        """
        upipe.print_info("Normalisation factors")
        upipe.print_info("Image # : BackGround        Slope")
        for nima in range(self.nimages):
            upipe.print_info("Image {0:03d}:  {1:10.6e}   {2:10.6e}".format(
                nima,
                self.ima_polypar[nima].beta[0],
                self.ima_polypar[nima].beta[1]))

    def _init_alignment_arrays(self):
        """Initialise all list and image arrays which are needed
        for processing the alignment
        """

        self.list_offmuse_hdu = [None] * self.nimages
        self.list_wcs_offmuse_hdu = [None] * self.nimages
        self.list_proj_refhdu = [None] * self.nimages
        self.list_wcs_proj_refhdu = [None] * self.nimages

        # Initialise the needed arrays for the PCC offsets
        self.pcc_off_pixel = np.zeros((self.nimages, 2), dtype=np.float64)
        self.pcc_off_arcsec = np.zeros_like(self.pcc_off_pixel)
        self.pcc_off_init = [False] * self.nimages

        # Initialise the list of optical flows
        self.optical_flows = [None] * self.nimages
        self.op_plots = [None] * self.nimages

        # Initialise the needed arrays for the cross-correlation offsets
        self.cross_off_pixel = np.zeros((self.nimages, 2), dtype=np.float64)
        self.extra_off_pixel = np.zeros_like(self.cross_off_pixel)

        self.cross_off_arcsec = np.zeros_like(self.cross_off_pixel)
        self.extra_off_arcsec = np.zeros_like(self.cross_off_pixel)

        self.extra_rotangles = np.zeros((self.nimages), dtype=np.float64)
        self._diffra_angle = np.zeros_like(self.extra_rotangles)

        # Cross normalisation for the images
        # This contains the parameters of the linear fit
        self.ima_polypar = [None] * self.nimages

        # Normalisation factor to be saved or used
        self.ima_norm_factors = np.zeros(self.nimages, dtype=np.float64)
        self.ima_background = np.zeros_like(self.ima_norm_factors)
        self.ima_threshold = np.full_like(self.ima_norm_factors, self.default_threshold_muse)
        self._convolve_muse = np.zeros_like(self.ima_norm_factors)
        self._convolve_reference = np.zeros_like(self.ima_norm_factors)

        # Default lists for date, mjd, tpls of the MUSE images
        self.ima_dateobs = [None] * self.nimages
        self.ima_mjdobs = [None] * self.nimages
        self.ima_tplstart = [None] * self.nimages
        self.ima_iexpo = [None] * self.nimages
        self.ima_dataset = [None] * self.nimages

    def init_guess_offset(self, **kwargs):
        """Initialise first guess, either from cross-correlation (default)
        or from an Offset FITS Table
         
        Input
        -----
        firstguess: str
            If "crosscorr" uses cross-correlation to get the first guess
            of the offsets. If "fits" uses the input fits table.
        """
        self.firstguess = kwargs.pop("firstguess", "crosscorr")

        # Implement the guess
        if self.firstguess is None:
            upipe.print_info("No initial guess shift: all set to 0")
            self.init_off_arcsec = self.cross_off_arcsec * 0.0
            self.init_off_pixel = self.cross_off_pixel * 0.0
            return
        else:
            # Forcing crosscorr if not recognised
            if self.firstguess not in ["pcc", "crosscorr", "fits"]:
                self.firstguess = "crosscorr"
                upipe.print_warning("Keyword 'firstguess' not recognised")
                upipe.print_warning("Using Cross-Correlation as a first guess for the alignment")

        if self.firstguess == "crosscorr":
            upipe.print_info("Using cross-correlation as the initial guess")
            # find the cross correlation peaks for each image
            # This is using zero offset and zero rotation
            # as all parameters have been reset above
            # New values will be taken out from the cross-correlation or fits table
            # just below with the init_guess_offset
            self.find_cross_peak_listima()
            # Transfer the cross-correlation offsets to the initialisation offsets
            self.init_off_arcsec = self.cross_off_arcsec * 1.0
            self.init_off_pixel = self.cross_off_pixel * 1.0

        elif self.firstguess == "pcc":
            # We use PCC from spacepylot to get the first guess here
            self.get_shift_from_pcc_listima()
            # Transfer the PCC offsets to the initialisation offsets
            self.init_off_arcsec = self.pcc_off_arcsec * 1.0
            self.init_off_pixel = self.pcc_off_pixel * 1.0

        elif self.firstguess == "fits":
            # Look for the offset table
            exist_table, self.offset_table = self.open_offset_table(
                joinpath(self.folder_offset_table, self.name_offset_table))
            if exist_table is not True:
                upipe.print_warning("Fits initialisation table not found, "
                                    "setting init value to 0")
                self._reset_init_guess_values()
                return

            upipe.print_info("Using input FITS table as initial guess: {0}".format(
                self.name_offset_table))
            # First get the right indices for the table by comparing MJD_OBS
            if mjd_names['table'] not in self.offset_table.columns:
                upipe.print_warning("Input table does not contain MJD_OBS column")
                self._reset_init_guess_values()
                return

            self.table_mjdobs = self.offset_table[mjd_names['table']]
            # Now finding the right match with the Images
            # Warning, needs > numpy 1.15.0
            mjd_values, ind_ima, ind_table = np.intersect1d(
                self.ima_mjdobs, self.table_mjdobs,
                return_indices=True, assume_unique=True)
            # Extracting the flux scale from table
            nonan_flux_scale_table = np.where(
                np.isnan(self.offset_table['FLUX_SCALE']),
                1., self.offset_table['FLUX_SCALE'])
            # Extracting the rotangle, but only if there
            rotangle_exist = ('ROTANGLE' in self.offset_table.columns)
            if rotangle_exist:
                nonan_rotangle_table = np.where(
                    np.isnan(self.offset_table['ROTANGLE']),
                    0., self.offset_table['ROTANGLE'])
            else:
                upipe.print_warning("Rotation angles not present in offset table. "
                                    "Please use argument 'extra_rotation' "
                                    "in 'run' to force a non zero value.")

            # Loop over the images, using MJD
            for nima, mjd in enumerate(self.ima_mjdobs):
                # Test if mjd is in the set of mjd_values
                if mjd in mjd_values:
                    # Find the index of the value array where mjd
                    ind = np.nonzero(mjd_values == mjd)[0][0]
                    self.init_off_arcsec[nima] = [
                        self.offset_table['RA_OFFSET'][ind_table[ind]] * 3600. \
                        * np.cos(np.deg2rad(self.list_dec_muse[nima])),
                        self.offset_table['DEC_OFFSET'][ind_table[ind]] * 3600.]
                    self.init_flux_scale[nima] = nonan_flux_scale_table[ind_table[ind]]
                    if rotangle_exist:
                        self.init_rotangles[nima] = nonan_rotangle_table[ind_table[ind]]
                    else:
                        self.init_rotangles[nima] = 0.0
                # Otherwise use default values
                else:
                    self.init_flux_scale[nima] = 1.0
                    self.init_off_arcsec[nima] = [0., 0.]
                    self.init_rotangles[nima] = 0.0

                # Transform into pixel values
                self.init_off_pixel[nima] = arcsec_to_pixel(
                    self.list_muse_hdu[nima],
                    self.init_off_arcsec[nima])

    def _reset_init_guess_values(self):
        """Reset the initial guess to 0. Hidden function as this is only
        used internally
        """
        self.table_mjdobs = [None] * self.nimages
        self.init_off_pixel = np.zeros((self.nimages, 2), dtype=np.float64)
        self.init_off_arcsec = np.zeros((self.nimages, 2), dtype=np.float64)
        self.init_flux_scale = np.ones(self.nimages, dtype=np.float64)
        self.init_rotangles = np.zeros(self.nimages, dtype=np.float64)

    def open_offset_table(self, name_table=None):
        """Read offset table from fits file
         
        Input
        -----
        name_table: str
            Name of the input OFFSET table

        Returns
        -------
        status: None if no table name is given, False if file does not
            exist, True if it does
        Table: the result of a astropy.Table.read of the fits table
        """
        if name_table is None:
            if not hasattr(self, "name_table"):
                upipe.print_error("No FITS table name provided, Aborting Open")
                return None, Table()

        if not os.path.isfile(name_table):
            upipe.print_warning("FITS Table ({0}) does not "
                                " exist yet".format(name_table))
            return False, Table()

        return True, Table.read(name_table)

    def show_offset_fromfits(self, name_table=None):
        """Print offset table from fits file
         
        Input
        -----
        name_table: str
            Name of the input OFFSET table
        """
        exist_table, fits_table = self.open_offset_table(name_table)
        if exist_table is None:
            return

        if (('RA_OFFSET' not in fits_table.columns)
                or ('DEC_OFFSET' not in fits_table.columns)):
            upipe.print_error("Table does not contain 'RA/DEC_OFFSET' "
                              "columns, Aborting")
            return

        upipe.print_info("Offset recorded in OFFSET_LIST Table")
        upipe.print_info("Total in ARCSEC")
        for nima in range(self.nimages):
            upipe.print_info("Image {0:03d} - {1}".format(nima, self.list_name_museimages[nima]))
            upipe.print_info("          - {0:8.4f} {1:8.4f}".format(
                fits_table['RA_OFFSET'][nima] * 3600
                * np.cos(np.deg2rad(self.list_dec_muse[nima])),
                fits_table['DEC_OFFSET'][nima] * 3600.))

    def print_images_names(self):
        """Print out the names of the images being considered for alignment
        """
        upipe.print_info("Image names")
        for nima in range(self.nimages):
            upipe.print_info("{0:03d} - {1}".format(nima, self.list_name_museimages[nima]))

    def print_offsets_and_norms(self, filename="_temp.txt",
                                folder_output_file=None, overwrite=True):
        """Save all offsets and norms into filename. By default, file will
        be overwritten.

        Input
        -----
        filename: str
            Name of file where the output will be written
        folder_output_file: str
            Name of output folder where the file will be written
        overwrite: bool
            Default is True

        Creates
        -------
            Ascii file named via the filename input argument

        """
        if folder_output_file is None:
            folder_output_file = self.folder_output_table
        fullname_file = joinpath(folder_output_file, filename)

        if os.path.isfile(fullname_file):
            if not overwrite:
                upipe.print_warning("File exists and overwrite is False. "
                                    "Aborting.")
                return
            else:
                upipe.print_warning("File exists but will be overwritten as"
                                    "overwrite is True.")

        toff_arc = self._total_off_arcsec
        toff_pix = self._total_off_pixel
        trot = self._total_rotangles
        data = {'nb': np.arange(self.nimages) + 1,
                'name': [name[-29:-5] for name in self.list_name_museimages],
                'xarc': toff_arc[:, 0],
                'yarc': toff_arc[:, 1],
                'xpix': toff_pix[:, 0],
                'ypix': toff_pix[:, 1],
                'rot': trot,
                'norm': self.ima_norm_factors,
                'backg': self.ima_background
                }

        ascii.write(data, fullname_file,
                    formats={'nb': '%03d', 'name': '%s>26',
                             'xarc': '%8.4f', 'yarc': '%8.4f',
                             'xpix': '%8.4f', 'ypix': '%8.4f',
                             'rot': '%8.4f', 'norm': '%10.6e',
                             'backg': '%10.6e'
                             },
                    format='fixed_width', overwrite=True)

    def show_offsets(self):
        """Print out the offset from the Alignment class
        """
        upipe.print_info("#---- Offset recorded so far ----#")
        upipe.print_info("#    Name               OFFSETS  |ARCSEC|   "
                         "X        Y     |PIXEL|    X        Y      |ROT| (DEG)")
        for nima in range(self.nimages):
            upipe.print_info("{0:03d} -{1:>26}  |ARCSEC|{2:8.4f} {3:8.4f} "
                             " |PIXEL|{4:8.4f} {5:8.4f}  |ROT|{6:8.4f}".format(
                             nima, self.list_name_museimages[nima][-29:-5],
                             self._total_off_arcsec[nima][0],
                             self._total_off_arcsec[nima][1],
                             self._total_off_pixel[nima][0],
                             self._total_off_pixel[nima][1],
                             self._total_rotangles[nima]))

    def save_fits_offset_table(self, name_output_table=None,
                               folder_output_table=None,
                               overwrite=False, suffix="", save_flux_scale=True,
                               save_other_params=True):
        """Save the Offsets into a fits Table
         
        Input
        -----
        folder_output_table: str [None]
            Folder of the output table. If None (default) the folder
            for the input offset table will be used or alternatively
            the folder of the MUSE images.
        name_output_table: str [None]
            Name of the output fits table. If None (default) it will
            use the one given in self.name_output_table
        overwrite: bool [False]
            If True, overwrite if the file exists
        suffix: str [""]
            Suffix to be used to add to the input name. This is handy
            to just modify the given fits name with a suffix 
            (e.g., version number).
        save_flux_scale: bool [True]
            If True, saving the flux in FLUX_SCALE
            If False, do not save the flux conversion
        save_other_params: bool [True]
            If True, saving the background + rotation
            If False, do not save these 2 parameters.
        
        Creates
        -------
        A fits table with the given name (using the suffix if any)
        """
        if name_output_table is None:
            if self.name_offset_table is None:
                name_output_table = "DUMMY_OFFSET_TABLE.fits"
            else:
                name_output_table = self.name_offset_table
        self.suffix = suffix
        self.name_output_table = name_output_table.replace(".fits",
                                                           "{0}.fits".format(self.suffix))

        if folder_output_table is not None:
            self.folder_output_table = folder_output_table
        exist_table, fits_table = self.open_offset_table(
            joinpath(self.folder_output_table, self.name_output_table))
        if exist_table is None:
            upipe.print_error("Save is aborted")
            return

        # Checking if overwrite and exist_table do go together
        if exist_table and not overwrite:
            upipe.print_warning("Table already exists, "
                                "but overwrite is set to False")
            upipe.print_warning("If you wish to overwrite the table {0}, "
                                "please set overwrite to True".format(
                                 name_output_table))
            return

        # Check if RA_OFFSET is there
        exist_ra_offset = ('RA_OFFSET' in fits_table.columns)

        # First save the DATA and MJD references
        fits_table[date_names['table']] = self.ima_dateobs
        fits_table[mjd_names['table']] = self.ima_mjdobs
        fits_table[tpl_names['table']] = self.ima_tplstart
        fits_table[iexpo_names['table']] = self.ima_iexpo
        fits_table[dataset_names['table']] = self.ima_dataset

        # Saving the final values
        fits_table['RA_OFFSET'] = self._total_off_arcsec[:, 0] / 3600. \
                                  / np.cos(np.deg2rad(self.list_dec_muse))
        fits_table['DEC_OFFSET'] = self._total_off_arcsec[:, 1] / 3600.
        if save_flux_scale:
            fits_table['FLUX_SCALE'] = self.ima_norm_factors
        else:
            fits_table['FLUX_SCALE'] = 1.0
        if save_other_params:
            fits_table['BACKGROUND'] = self.ima_background
            fits_table['ROTANGLE'] = self._total_rotangles

        # Deal with RA_OFFSET_ORIG if needed
        if exist_ra_offset:
            # if RA_OFFSET exists, then check if the ORIG column is there
            if 'RA_OFFSET_ORIG' not in fits_table.columns:
                fits_table['RA_OFFSET_ORIG'] = fits_table['RA_OFFSET']
                fits_table['DEC_OFFSET_ORIG'] = fits_table['DEC_OFFSET']
                fits_table['FLUX_SCALE_ORIG'] = fits_table['FLUX_SCALE']

        # Finally add the cross-correlation offsets
        fits_table['RA_CROSS_OFFSET'] = self.cross_off_arcsec[:, 0] / 3600. \
                                        / np.cos(np.deg2rad(self.list_dec_muse))
        fits_table['DEC_CROSS_OFFSET'] = self.cross_off_arcsec[:, 1] / 3600.

        # Writing up
        fits_table.write(joinpath(self.folder_output_table, self.name_output_table),
                         overwrite=overwrite)
        self.name_output_table = name_output_table


    def _check_nima(self, nima):
        test_nima = nima in range(self.nimages)
        if self.verbose:
            if not test_nima:
                upipe.print_error(f"nima={nima} not within the range "
                                  f"allowed by self.nimages ({self.nimages})")
        return test_nima


    def offset_and_compare(self, nima=0, extra_pixel=None, extra_arcsec=None,
                           extra_rotation=None, **kwargs):
        """Run the offset and comparison for a given image number
         
        Input
        -----
        nima: int
            Index of the image to consider
        extra_pixel: list of 2 floats
            Offsets in X and Y in pixels to add to the existing
            guessed offsets
            IMPORTANT NOTE: extra_pixel will be considered first
            (before extra_arcsec).
        extra_arcsec: list of 2 floats
            Offsets in X and Y in arcsec to add to the existing
            guessed offsets. Ignored if extra_pixel is given or None
        extra_rotation: rotation in degrees
            Angle to rotate the image (in degrees). Ignore if None
        threshold_muse: float [0]
            Threshold to consider when plotting the comparison

        Additional arguments
        --------------------
        plot (bool): if True, will plot the comparison
            If not used, will use the default self.plot
               * flux comparison (1 to 1)
               * Map of the flux ratio
               * Contours of the two scaled maps
               * Cuts of the division between the 2 maps

        See also all arguments from self.compare
        """
        if not self._check_nima(nima):
            return

        # Add the offset from user
        border = kwargs.get("border", self.border)
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        self.apply_extra_offset_ima(nima=nima, extra_arcsec=extra_arcsec,
                                    extra_pixel=extra_pixel,
                                    extra_rotation=extra_rotation,
                                    border=border, chunk_size=chunk_size)

        # Compare contours if plot is set to True
        self.compare_ima(nima=nima, **kwargs)

    def _get_list_muse_images(self):
        """Extract the name of the muse images
        and build the list
        """
        from pathlib import Path

        if self.name_muse_images is None:
            set_of_paths = glob.glob("{0}*{1}*.fits".format(
                joinpath(self.folder_muse_images,
                         self.suffix_images), self.filter_name))
            self.list_name_museimages = [Path(muse_path).name
                                         for muse_path in set_of_paths]
            # Sort alphabetically
            self.list_name_museimages.sort()
            # Subselection if sel_indices_images is given
            if self.sel_indices_images is not None:
                if not all([i in np.arange(len(self.list_name_museimages))
                            for i in self.sel_indices_images]):
                    upipe.print_warning("Selection list - sel_indices_images "
                                        "- does not match image list")
                    upipe.print_warning("Ignoring that input sel_indices_images")
                else:
                    newlist = [self.list_name_museimages[nima]
                               for nima in self.sel_indices_images]
                    self.list_name_museimages = newlist

        # test if 1 or several images
        elif isinstance(self.name_muse_images, str):
            self.list_name_museimages = [self.name_muse_images]
        elif isinstance(self.name_muse_images, list):
            self.list_name_museimages = self.name_muse_images
        else:
            upipe.print_warning("Name of images is not a string or a list, "
                                "please check input name_muse_images")
            self.list_name_museimages = []

        # Number of images to deal with
        self.nimages = len(self.list_name_museimages)

    def open_hdu(self):
        """Open the HDU of the MUSE and reference images
        """
        status_ref = self._open_ref_hdu()
        if not status_ref:
            upipe.print_error("Problem in opening Reference frame, please check input")
            return 0

        status_muse = self._open_muse_nhdu()
        if not status_muse:
            upipe.print_error("Problem in opening MUSE frame, please check input")
            return 0

        return 1

    def _open_muse_nhdu(self):
        """Open the MUSE images hdu
        """
        self.list_name_musehdr = ["{0}{1:03d}.hdr".format(
            self.name_musehdr, i + 1) for i in range(self.nimages)]
        self.list_name_offmusehdr = ["{0}{1:03d}.hdr".format(
            self.name_offmusehdr, i + 1) for i in range(self.nimages)]
        self.list_hdulist_muse = [pyfits.open(
            joinpath(self.folder_muse_images, self.list_name_museimages[i]))
            for i in range(self.nimages)]
        self.list_muse_hdu = [hdu[self.hdu_ext[1]]
                              for hdu in self.list_hdulist_muse]
        # CHANGE to mpdaf WCS
        self.list_wcs_muse = [WCS(hdu[self.hdu_ext[1]].header)
                              for hdu in self.list_hdulist_muse]
        self.list_dec_muse = np.array([muse_wcs.get_crval2()
                                       for muse_wcs in self.list_wcs_muse])
        # Getting the orientation angles
        self.list_wcs_rotangles = [musewcs.get_rot()
                                   for musewcs in self.list_wcs_muse]

        # Filling in the MJD and DATE OBS keywords for the MUSE images
        # If not there, will be filled with "None"
        for nima, hdu in enumerate(self.list_hdulist_muse):
            if date_names['image'] not in hdu[0].header:
                self.ima_dateobs[nima] = None
            else:
                self.ima_dateobs[nima] = hdu[0].header[date_names['image']]
            if mjd_names['image'] not in hdu[0].header:
                self.ima_mjdobs[nima] = None
            else:
                self.ima_mjdobs[nima] = hdu[0].header[mjd_names['image']]
            if tpl_names['image'] not in hdu[0].header:
                self.ima_tplstart[nima] = None
            else:
                self.ima_tplstart[nima] = hdu[0].header[tpl_names['image']]
            if iexpo_names['image'] not in hdu[0].header:
                self.ima_iexpo[nima] = None
            else:
                self.ima_iexpo[nima] = hdu[0].header[iexpo_names['image']]
            if dataset_names['image'] in hdu[0].header:
                self.ima_dataset[nima] = hdu[0].header[dataset_names['image']]
            else:
                if self._backward_comp and dataset_names['oldimage'] in hdu[0].header:
                    self.ima_dataset[nima] = hdu[0].header[dataset_names['oldimage']]
                else:
                    self.ima_dataset[nima] = None

            if self.list_muse_hdu[nima].data is None:
                return 0

        return 1

    def _open_ref_hdu(self):
        """Open the reference image hdu
        """
        # Open the images
        hdulist_reference = pyfits.open(joinpath(self.folder_reference,
                                                 self.name_reference))
        self.reference_hdu = hdulist_reference[self.hdu_ext[0]]
        if self.reference_hdu.data is None:
            upipe.print_error("No data found in extension of reference frame")
            upipe.print_error("Check your input, "
                              "or change the extention number in input hdu_ext[0]")
            return 0

        return 1

    def get_imaref_muse(self, muse_hdu, rotation=0.0, minflux=0.0, **kwargs):
        """Returns the ref and input images on the same grid as the given
        input hdu assuming a given rotation

        Input
        -----
        muse_hdu: HDU
            MUSE hdu file
        name_musehdr: str
            name of the muse hdr to save
        rotation: float
            Angle in degrees (0).
        minflux: float
            Minimum flux to prepare the image (0).

        Returns
        -------
        ima_ref, ima_muse: arrays
            Reprojected images

        Note that the original images are saved in self._temp_input_origmuse and
        self._temp_input_origref when debug mode is on (self._debug)
        """
        remove_bkg = kwargs.pop("remove_bkg", True)
        squeeze = kwargs.pop("squeeze", True)
        border = kwargs.pop("border", self.border)
        mask_stars = kwargs.pop("mask_stars", False)

        # Save hdr if save_hdr is True
        if self.save_hdr:
            name_musehdr = kwargs.pop("name_musehdr", "dummy.hdr")
            _ = muse_hdu.header.totextfile(joinpath(self.header_folder_name,
                                                    name_musehdr), overwrite=True)

        # Projecting the reference image onto the MUSE dataset
        hdu_target, proj_ref_hdu, diffra_angle = self._align_reference_hdu(muse_hdu,
                                                                           target_rotation=rotation,
                                                                           conversion_factor=None)

        # Cleaning the images
        if minflux is None:
            minflux = self.minflux_crosscorr

        minflux_ref = minflux / self.conversion_factor
        ima_ref = flatclean_image(proj_ref_hdu.data, border, self.dynamic_range, self.median_window,
                                minflux=minflux_ref, squeeze=squeeze, remove_bkg=remove_bkg)
        ima_muse = flatclean_image(muse_hdu.data, border, self.dynamic_range, self.median_window,
                                 minflux=minflux, squeeze=squeeze, remove_bkg=remove_bkg)

        if mask_stars:
            ima_ref = mask_point_sources(ima_ref)
            ima_muse = mask_point_sources(ima_muse)

        if self._debug:
            self._temp_input_origmuse = muse_hdu.data * 1.0
            self._temp_input_origref = proj_ref_hdu.data * 1.0

        return ima_ref, ima_muse


    def get_shift_from_pcc_listima(self, list_nima=None, minflux=None, verbose=False):
        """Run the PCC shift guess on a list of images given by a list
        of indices

        Input
        -----
        list_nima: list of indices for images to process
            Should be a list. Default is None
            and all images are processed

        minflux: float [None]
            minimum flux to be used in the cross-correlation
            Flux below that value will be set to 0.
        """
        upipe.print_info("Starting the PCC shift guess for all images")
        if list_nima is None:
            list_nima = range(self.nimages)

        for nima in list_nima:
            self.get_shift_from_pcc_ima(nima=nima, minflux=minflux, verbose=verbose)

    def get_shift_from_pcc_ima(self, nima=None, minflux=None, rotation=None, verbose=False):
        """Run the PCC shift guess for image nima

        Input
        -----
        nima: int
            Index of image
        minflux: float [None]
           minimum flux to be used in the cross-correlation
           Flux below that value will be set to 0.
        rotation: float
           If None, will take the init_rotangle. Otherwise it will take the input value
        """
        upipe.print_info(f"PCC for image {nima}")

        if rotation is None:
            rotation = self.init_rotangles[nima]

        # Running shift_from_pcc
        self.pcc_off_pixel[nima] = self.get_shift_from_pcc(self.list_muse_hdu[nima],
                                                           rotation=rotation,
                                                           minflux=minflux,
                                                           name_musehdr=self.list_name_musehdr[
                                                               nima],
                                                           verbose=verbose)
        self.pcc_off_arcsec[nima] = pixel_to_arcsec(self.list_muse_hdu[nima],
                                                    self.pcc_off_pixel[nima])
        self.pcc_off_init[nima] = True

    def get_shift_from_pcc(self, muse_hdu, rotation=0.0, minflux=0.0, verbose=False, **kwargs):
        """Find a guess translation using PCC

        Input
        -----
        muse_hdu: HDU
            MUSE hdu file
        rotation: float
            Angle in degrees (0).
        minflux: float
            Minimum flux to prepare the image (0).
        name_musehdr: str
            Name of the muse hdr to save. Optional. Only operational if self.save_hdr is True

        Returns
        -------
        xpix_pcc
        ypix_pcc x and y pixel coordinates of the cross-correlation peak
        """
        ima_ref, ima_muse = self.get_imaref_muse(muse_hdu, rotation=rotation, minflux=minflux,
                                                 border=10, remove_bkg=False, squeeze=False,
                                                 **kwargs)
        if self._debug:
            self._temp_input_pccmuse = ima_muse
            self._temp_input_pccref = ima_ref

        gt = sppalign.AlignTranslationPCC(ima_muse, ima_ref, verbose=verbose)
        # Beware that get_translation from spacepylot return y, x
        # Hence the ::-1 to reverse it and the minus sign as it is the reverse than done with PCC
        return -gt.get_translation(split_image=2)[::-1]

    def run_optical_flow(self, list_nima=None, save_plot=True, use_rotation=True,
                         verbose=False, **kwargs):
        """Run Optical flow, first with a guess offset and then iterating. The solution
        is saved as extra offset in the class, and a op_plot instance is created.
        If save_plot is True, it will save a set of default plots

        Input
        -----
        list_nima: list
            List of indices. If None, will use the default list of all images
        save_plot : bool
            Whether to save the optical flow diagnostic plots or not.
        use_rotation: bool
            True if you wish to have rotation. False otherwise
        verbose: bool
        """
        upipe.print_info("Run the optical flow for all images in list (indices)")
        if list_nima is None:
            list_nima = range(self.nimages)

        # Use Translation and Rotation or only rotation?
        for nima in list_nima:
            upipe.print_info(f"------- Optical Flow for Image {nima} -------")
            self.run_optical_flow_ima(nima=nima, save_plot=save_plot,
                                      use_rotation=use_rotation,
                                      verbose=verbose, **kwargs)

    def run_optical_flow_ima(self, nima=0, save_plot=True, use_rotation=True, verbose=False, **kwargs):
        """Run Optical flow on image with index nima,
        first with a guess offset and then iterating. The solution
        is saved as extra offset in the class, and a op_plot instance is created.
        If save_plot is True, it will save a set of default plots

        Input
        -----
        nima: int
            Image index.
        save_plot : bool
            Whether to save the optical flow diagnostic plots or not.
        """
        self.iterate_on_optical_flow_ima(nima, verbose=verbose, use_rotation=use_rotation, **kwargs)
        self.apply_optical_flow_offset_ima(nima)
        if save_plot:
            plt.ioff()
            self.op_plots[nima].red_blue_before_after()
            plt.savefig(joinpath(self.figures_folder_name, f"opflow_redblue_{nima:03d}.png"))
            self.op_plots[nima].before_after()
            plt.savefig(joinpath(self.figures_folder_name, f"opflow_beforeafter_{nima:03d}.png"))
            self.op_plots[nima].illustrate_vector_fields()
            plt.savefig(joinpath(self.figures_folder_name, f"opflow_vectorfield_{nima:03d}.png"))
            self.op_plots[nima].before_after_diff_frac()
            plt.savefig(joinpath(self.figures_folder_name, f"opflow_beforeafter_frac_"
                                                           f"{nima:03d}.png"))
            plt.close('all')
            plt.ion()

    def apply_optical_flow_offset_listima(self, list_nima=None):
        """Apply the optical flow offset as extra pixels offsets and rotation

        Input
        -----
        list_nima: list
            If None, will be initiliased to the default list of indices
        """
        upipe.print_info("Apply the optical flow offset as extra user offset - and rotation")
        if list_nima is None:
            list_nima = range(self.nimages)

        for nima in list_nima:
            self.apply_optical_flow_offset_ima(nima=nima)


    def apply_optical_flow_offset_ima(self, nima=0):
        """Transfer the value of the optical flow into the extra pixel
        """
        if self.optical_flows[nima] is None:
            upipe.print_warning(f"No optical flow class for image {nima} -"
                                 "- run it before applying it")
            return

        if self.verbose:
            upipe.print_info(f"Apply optical flow offset solution to Image #{nima:03d}")
        # Now set up the extra needed offsets as the solution from optical flow
        # We need to invert Y, X to X, Y (the [::-1]) and use the minus sign
        # considering optical flow derives the motion of ref to MUSE
        self.apply_extra_offset_ima(nima=nima,
                                    extra_pixel=- self.optical_flows[nima].translation[::-1],
                                    extra_rotation=-self.optical_flows[nima].rotation_deg)
        # Initialise the plot
        self.op_plots[nima] = init_plot_optical_flow(self.optical_flows[nima])


    def iterate_on_optical_flow_ima(self, nima=0, niter=5, verbose=False,
                                    use_rotation=True, **kwargs):
        """Iterate solution using the optical flow guess

        Input
        -----
        nima: int
            Index of image to consider
        niter: int
            Number of iterations
        """
        if self.verbose:
            upipe.print_info(f"Optical flow with {niter} iterations for Image {nima}")

        list_guess_key = ["guess_rotation", "guess_offset_pixel", "guess_offset_arcsec"]
        given_guess = any(guess_key in kwargs for guess_key in list_guess_key)
        reset_opf = kwargs.pop("reset_optical_flow", False)

        if self.optical_flows[nima] is None or reset_opf or given_guess:
            self.init_optical_flow_ima(nima, verbose=verbose, **kwargs)

        if "homography_method" in kwargs:
            homography_method = kwargs.pop("homography_method", transform.EuclideanTransform)
        else:
            if use_rotation:
                homography_method = transform.EuclideanTransform
            else:
                homography_method = TranslationTransform
        self.optical_flows[nima].get_iterate_translation_rotation(niter,
                                                                  homography_method=homography_method)


    def iterate_on_optical_flow_listima(self, list_nima=None, use_rotation=True, **kwargs):
        """Run the iteration for the optical flow on a list of images
        given by a list of indices

        Input
        -----
        list_nima: list of indices for images to process
            Should be a list. Default is None
            and all images are processed

        niter: int
            Number of iterations. Optional. If  not provided, will use the
            default in self.iterate_on_optical_flow_ima
        """
        upipe.print_info("Iteration the Optical Flow solution on all images")
        if list_nima is None:
            list_nima = range(self.nimages)

        for nima in list_nima:
            self.iterate_on_optical_flow_ima(nima=nima, use_rotation=use_rotation, **kwargs)


    def init_optical_flow_listima(self, list_nima=None, **kwargs):
        """Initialise the optical flow on a list of images
        given by a list of indices

        Input
        -----
        list_nima: list of indices for images to process
            Should be a list. Default is None
            and all images are processed

        """
        upipe.print_info("Initiliase the Optical Flow on all images")
        if list_nima is None:
            list_nima = range(self.nimages)

        for nima in list_nima:
            self.init_optical_flow_ima(nima=nima, **kwargs)


    def init_optical_flow_ima(self, nima=0, minflux=None, guess_offset_pixel=None,
                              guess_offset_arcsec=None, guess_rotation=None,
                              force_pcc_guess=False, verbose=False, provide_header=True,
                              **kwargs):
        """Initialise the optical flow using the current image with index nima

        Input
        -----
        nima: int
            Index of image
        minflux: float
            Minimum flux to consider
        """
        # Forcing a pcc guess if all guess are None
        # WARNING: we need to set the offset as Y, X to pass it on optical flow
        if guess_offset_pixel is None and guess_offset_arcsec is None:
            # Force a PCC guess
            if force_pcc_guess:
                self.get_shift_from_pcc_ima(nima, minflux=minflux, verbose=verbose)
                guess_offset_pixel = self.pcc_off_pixel[nima][::-1]
            # Or use the already defined one
            else:
                guess_offset_pixel = self.init_off_pixel[nima]

        # Make sure guess_offset_pixel is there if arcsec are given
        guess_offset_pixel, guess_offset_arcsec = self._sort_offset_pixel_arcsec(
            self.list_muse_hdu[nima], guess_offset_pixel, guess_offset_arcsec)

        # Retrofitting the guessed values into the init values
        self.init_off_pixel[nima] = guess_offset_pixel
        self.init_off_arcsec[nima] = guess_offset_arcsec

        if guess_rotation is None:
            guess_rotation = self.init_rotangles[nima]

        # Do the alignment and get the off_muse and proj_ref using those guess offset
        hdu_off_muse, hdu_proj_ref, diffra = self._apply_alignment(self.list_muse_hdu[nima],
                                                                   total_off_pixel=guess_offset_pixel,
                                                                   total_off_arcsec=guess_offset_arcsec,
                                                                   total_rotangle=guess_rotation,
                                                                   verbose=False)
        upipe.print_info(f"Used grid with initial Offset / Rotation = "
                         f"{guess_offset_pixel[0]:8.4f} {guess_offset_pixel[1]:8.4f} [PIX] "
                         f"/ {guess_rotation} [DEG]")

        # Calling optical flow initialisation
        if provide_header:
            header = copy.copy(hdu_off_muse.header)
        else:
            header = None
        # Initialise optical flow with the corresponding HDU
        # All guesses should not be 0, since they have been processed above
        self.optical_flows[nima] = self.init_optical_flow_hdu(hdu_off_muse,
                                                              rotation=0., minflux=minflux,
                                                              guess_translation=[0., 0.],
                                                              header=header, verbose=verbose,
                                                              **kwargs)


    def init_optical_flow_hdu(self, muse_hdu, rotation=0., minflux=None, guess_translation=(0.,0.),
                              header=None, verbose=False, **kwargs):
        """Get the optical flow for this hdu

        Input
        -----
        muse_hdu: HDU
            Muse HDU input
        rotation: float
            Input rotation
        minflux: float
            Minimum flux to consider in the image
        guess_translation: tuple of 2 floats
            Guess offset in X and Y, e.g., (0., 0.)
        name_musehdr: str
            Name of hdr in case those are saved (self.save_hdr is True)
        """
        # Getting the images. Note that border must be 0 as otherwise you will need to change
        # the WCS (header passed to AlignOpticalFlow
        ima_ref, ima_muse = self.get_imaref_muse(muse_hdu, rotation, minflux, border=0,
                                                 remove_bkg=False, squeeze=False, **kwargs)
        if self._debug:
            self._temp_input_opflow_muse = ima_muse * 1.0
            self._temp_input_opflow_ref = ima_ref * 1.0

        # WARNING: we pass on Y, X hence the [::-1] in the guess_translation parameter
        upipe.print_info(f"Initialising Optical Flow, with guess_translation (x,y): "
                         f"{guess_translation[::-1]}")
        return sppalign.AlignOpticalFlow(ima_muse, ima_ref,
                                         guess_translation=guess_translation[::-1],
                                         header=header,
                                         verbose=verbose)


    def find_cross_peak_ima(self, nima=0, minflux=None):
        """Find the cross correlation peak and get the x and y shifts
        for a given image, given its index nima

        Input
        -----
        nima: int
           Index of the image
        minflux: float
           Minimum flux for the cross-correlation
        """
        upipe.print_info(f"Cross-correlation for image {nima}")

        self.cross_off_pixel[nima] = self.find_cross_peak(self.list_muse_hdu[nima],
                                                          rotation=self.init_rotangles[nima],
                                                          minflux=minflux,
                                                          name_musehdr=self.list_name_musehdr[nima])
        self.cross_off_arcsec[nima] = pixel_to_arcsec(self.list_muse_hdu[nima],
                                                      self.cross_off_pixel[nima])

    def find_cross_peak_listima(self, list_nima=None, minflux=None):
        """Run the cross correlation peaks on all MUSE images
        Derive the self.cross_off_pixel/arcsec parameters

        Input
        -----
        list_nima: list
            list of indices for images to process Should be a list. Default is None
            and all images are processed
        minflux: float [None]
            minimum flux to be used in the cross-correlation
            Flux below that value will be set to 0.
        """
        upipe.print_info("Starting the cross-correlation for all images")

        if list_nima is None:
            list_nima = range(self.nimages)

        for nima in list_nima:
            self.find_cross_peak_ima(nima, minflux=minflux)

    def find_cross_peak(self, muse_hdu, rotation=0.0, minflux=None, **kwargs):
        """Aligns the MUSE HDU to a reference HDU
         
        Input
        -----
        muse_hdu: astropy.io.fits hdu
            MUSE hdu file
        name_musehdr: str
            name of the muse hdr to save
        rotation: float
            Angle in degrees (0).
        minflux: minimum flux to be used in the cross-correlation
            Flux below that value will be set to 0.
            Default is 0.
        
        Returns
        -------
        xpix_cross
        ypix_cross: x and y pixel coordinates of the cross-correlation peak
        """
        ima_ref, ima_muse = self.get_imaref_muse(muse_hdu, rotation, minflux, **kwargs)

        if self.phase_corr:
            shifts, shift_errors, phasediff = phase_cross_correlation(ima_ref, ima_muse,
                                                                      upsample_factor=self.phase_subsamp)
            xpix_cross = shifts[1]
            ypix_cross = shifts[0]
        else:
            # Cross-correlate the images
            ccor = correlate(ima_ref, ima_muse, mode='full', method='auto')
            if self._debug:
                self._temp_ima_muse_tocc = ima_muse * 1.0
                self._temp_ima_ref_tocc = ima_ref * 1.0
                self._temp_cc = ccor * 1.0

            # Find peak of cross-correlation
            maxy, maxx = np.unravel_index(np.argmax(ccor),
                                          ccor.shape)

            # Extract a window around it
            window = self.subim_window
            y, x = np.ix_(np.arange(-window + maxy, window + 1 + maxy),
                          np.arange(-window + maxx, window + 1 + maxx))
            subim = ccor[y % ccor.shape[0], x % ccor.shape[1]]
            subim -= subim.min()
            mx = np.max(subim)
            smaxy, smaxx = np.unravel_index(np.argmax(subim),
                                            subim.shape)

            # Fit a 2D Gaussian to that peak
            gauss_init = models.Gaussian2D(amplitude=mx,
                                           x_mean=x[0, smaxx],
                                           y_mean=y[smaxy, 0],
                                           x_stddev=2,
                                           y_stddev=2,
                                           theta=0)
            fitter = fitting.LevMarLSQFitter()
            params = fitter(gauss_init, x * np.ones_like(y),
                            y * np.ones_like(x),
                            subim)

            # Update Astrometry
            # Beware, the sign was changed here and is now ok
            xpix_cross = ccor.shape[1] // 2 - params.x_mean
            ypix_cross = ccor.shape[0] // 2 - params.y_mean

        return xpix_cross, ypix_cross

    def save_image(self, newfits_name=None, nima=0):
        """Save the newly determined hdu
         
        Input
        -----
        newfits_name: str
            Name of the fits file to be used
        nima: int [0]
            Index of the image to save

        Creates
        -------
        A new fits file
        """
        if hasattr(self, "list_offmuse_hdu"):
            if newfits_name is None:
                newfits_name = self.list_name_museimages[nima].replace(
                    ".fits", "_shift.fits")
            self.list_offmuse_hdu[nima].writeto(newfits_name, overwrite=True)
        else:
            upipe.print_error("There are not yet any new hdu to save")


    def _align_hdu(self, hdu_target=None, hdu_to_align=None,
              target_rotation=0.0, to_align_rotation=0.0,
              conversion_factor=None):
        """Project an to be aligned hdu onto the input hdu
        Hidden function, as only used internally

        Input
        -----
        hdu_target: HDU [None]
            Target hdu (on to which to project)
        hdu_to_align: HDU [None]
             Hdu to be aligned
        target_rotation: float [0]
            Rotation angle in degrees of the target hdu
        to_align_rotation: float [0]
            Rotation angle in degrees of the to be aligned hdu
        conversion_factor: float
            If None, will use the self.conversion_factor parameter

        Returns
        -------
        hdu_repr: HDU
            Reprojected HDU. None if nothing is provided
        """
        if conversion_factor is None:
            conversion_factor = self.conversion_factor

        return align_hdu(hdu_target=hdu_target, hdu_to_align=hdu_to_align,
                         target_rotation=target_rotation, to_align_rotation=to_align_rotation,
                         conversion_factor=conversion_factor, use_mpdaf=self.use_mpdaf)


    def _align_reference_hdu(self, hdu_target=None, target_rotation=0.0,
                             ref_rotation=0.0, conversion_factor=None, **kwargs):
        """Project the reference image onto the target hdu
        Hidden function, as only used internally
         
        Input
        -----
        hdu_target: HDU [None]
            Input hdu
        target_rotation: float [0]
            Target rotation angle in degrees
        ref_rotation: float [0]
            Rotation of the reference image
        
        Returns
        -------
        hdu_repr: HDU
            Reprojected HDU. None if nothing is provided
        """

        return self._align_hdu(hdu_target=hdu_target, hdu_to_align=self.reference_hdu,
                               target_rotation=target_rotation, to_align_rotation=ref_rotation,
                               conversion_factor=conversion_factor, **kwargs)

    @property
    def _total_rotangles(self):
        return self.init_rotangles + self.extra_rotangles

    @property
    def _total_off_pixel(self):
        return self.init_off_pixel + self.extra_off_pixel

    @property
    def _total_off_arcsec(self):
        return self.init_off_arcsec + self.extra_off_arcsec

    def _set_extra_offset_ima(self, nima=0, extra_pixel=None, extra_arcsec=None,
                              extra_rotation=None):
        """Add user offset in pixel and transform into arcseconds

        Input
        -----
        extra_pixel: list of 2 floats [0,0]
            Extra offsets (x,y) in pixels
        extra_arsec: list of 2 floats [0,0]
            Extra offsets (x,y) in arcsec if extra_pixel is not provided
        extra_rotation: rotation in degrees [0]
        nima: int
            Index of image to consider
        """
        # Add this to the extra_off arrays
        if extra_pixel is None:
            if extra_arcsec is not None:
                # Transforming the arc into pix
                self.extra_off_arcsec[nima] = extra_arcsec
                # Transforming into pixels - would be better with setter
                self.extra_off_pixel[nima] = arcsec_to_pixel(self.list_muse_hdu[nima], extra_arcsec)
        else:
            self.extra_off_pixel[nima] = extra_pixel
            # Transforming into arcsec - would be better with setter
            self.extra_off_arcsec[nima] = pixel_to_arcsec(self.list_muse_hdu[nima], extra_pixel)

        # And the rotation angle in degrees
        if extra_rotation is not None:
            self.extra_rotangles[nima] = extra_rotation


    def apply_extra_offset_ima(self, nima=0, extra_pixel=None, extra_arcsec=None,
                               extra_rotation=None, **kwargs):
        """Shift image with index nima with the total offset
        after adding any extra given offset
        This does not return anything but could in principle
        if using the output of the self.shift

        Input
        -----
        nima: int
            Index of image to consider
        extra_pixel: list of 2 floats
            Extra offsets (x,y) in pixels. If None, nothing is applied
        extra_arcsec: list of 2 floats
            Extra offsets (x,y) in arcsec if extra_pixel is not provided
            If None, nothing is applied
        extra_rotation: float
            Rotation in degrees. If None, no new extra offset is applied
        """
        # Add this to the extra_off arrays
        self._set_extra_offset_ima(nima=nima, extra_arcsec=extra_arcsec, extra_pixel=extra_pixel,
                                    extra_rotation=extra_rotation)

        # Actually apply the alignment to the image
        self._apply_alignment_ima(nima, **kwargs)


    def _apply_alignment_ima(self, nima=0, **kwargs):
        """Apply alignment for image with index nima

        Input
        ----
        nima: int
            Index of image. Default is 0

        """
        # Call the alignment given the input nima image
        upipe.print_info(f"[#{nima:03d}] --- Regridding Image "
                         f" {self.list_name_museimages[nima]} ---")
        hdu_offmuse, hdu_projref, diffra  = self._apply_alignment(self.list_muse_hdu[nima],
            total_off_pixel=self._total_off_pixel[nima], total_rotangle=self._total_rotangles[nima])

        # Creating a new Primary HDU and its WCS with the input data, and the new Header
        # This HDU has now been offset using the total offsets
        self.list_offmuse_hdu[nima] = hdu_offmuse
        self.list_wcs_offmuse_hdu[nima] = WCS(hdu_offmuse.header)
        # Saving the projected reference hdu and its WCS
        self.list_proj_refhdu[nima] = hdu_projref
        self.list_wcs_proj_refhdu[nima] = WCS(hdu_projref.header)

        # Writing this up in an ascii file for record purposes
        if self.save_hdr:
             _ = self.list_offmuse_hdu[nima].header.totextfile(joinpath(self.header_folder_name,
                                                                        self.list_name_offmusehdr[nima]),
                                                               overwrite=True)
        # Getting the normalisation factors using those last projections
        _, _, self.ima_polypar[nima] = self.get_normfactor_ima(nima=nima, **kwargs)
        # Saving the normalisation
        self.save_polypar_ima(nima, self.ima_polypar[nima].beta)

    def _sort_offset_pixel_arcsec(self, hdu, offset_pixel=None, offset_arcsec=None):
        """

        Input
        -----
        hdu: HDU
        offset_pixel: tuple of float
        offset_arcsec: tuple of float

        Returns
        -------
        offset_pixel, offset_arsec

        """
        # Using input offset or total
        if offset_pixel is None:
            if offset_arcsec is None:
                offset_pixel = [0., 0.]
                offset_arcsec = pixel_to_arcsec(hdu, offset_pixel)
            else:
                offset_pixel = arcsec_to_pixel(hdu, offset_arcsec)
        else:
            offset_arcsec = pixel_to_arcsec(hdu, offset_pixel)

        return offset_pixel, offset_arcsec

    def _apply_alignment(self, hdu, total_off_pixel=None, total_off_arcsec=None,
                         total_rotangle=0., **kwargs):
        """Create New HDU after shifting it with the right offset
        (only considering image with index nima)
         
        Input
        -----
        hdu : HDU
            Input HDU of image to offset
        nima: int
            Index of image to consider
        
        Does not return anything, but could in principle
        """
        # Create a new Header
        newhdr = copy.copy(hdu.header)

        # Using input offset or total
        total_off_pixel, total_off_arcsec = self._sort_offset_pixel_arcsec(hdu, total_off_pixel,
                                                                           total_off_arcsec)

        # Shift the HDU in X and Y
        verbose = kwargs.pop("verbose", self.verbose)
        if verbose:
            upipe.print_info(f"       Offset   [PIXELS]: {total_off_pixel[0]:8.4f}"
                           f" {total_off_pixel[1]:8.4f}  /  "
                           f"[ARCSEC]: {total_off_arcsec[0]:8.4f}"
                           f"{total_off_arcsec[1]:8.4f}")
            upipe.print_info(f"       Rotation [DEGREE]: {total_rotangle:8.4f}")

        # Shifting the CRPIX values in the header
        newhdr['CRPIX1'] += total_off_pixel[0]
        newhdr['CRPIX2'] += total_off_pixel[1]

        # Creating a new Primary HDU with the input data, and the new Header
        # This HDU has now been offset using the total offsets
        hdu_offmuse = pyfits.PrimaryHDU(hdu.data, header=newhdr)

        # Reprojecting the Reference image onto the new MUSE frame
        hdu_target, hdu_projref, diffra = \
            self._align_reference_hdu(hdu_target=hdu_offmuse, target_rotation=total_rotangle)

        return hdu_offmuse, hdu_projref, diffra

    def get_normfactor_ima(self, nima=0, median_filter=True, border=0,
                               convolve_muse=0., convolve_reference=0., chunk_size=10):
        """Get the normalisation factor for shifted and projected images. This function only 
        consider the input image given by index nima and the reference image (after
        projection).
         
        Input
        -----
        nima: int
            Index of image to consider
        median_filter: bool
            If True, will median filter
        convolve_muse: float [0]
            Will convolve the image with index nima
            with a gaussian with that sigma. 0 means no convolution
        convolve_reference: float [0]
            Will convolve the reference image
            with a gaussian with that sigma. 0 means no convolution
        border: int
            Number of pixels to crop
        threshold_muse: float [None]
            Threshold for the input image flux to consider
        chunk_size: int
            Size of chunks to consider for chunk statistics (polynomial normalisation)
        
        Returns
        -------
        data: 2d array
        refdata: 2d array
            The 2 arrays (input, reference) after processing
        """
        return get_normfactor(self.list_offmuse_hdu[nima].data, self.list_proj_refhdu[nima].data,
                              convolve_data1=convolve_muse, convolve_data2=convolve_reference,
                              median_filter=median_filter, border=border,
                              chunk_size=chunk_size, threshold=self.ima_threshold[nima])


    def save_polypar_ima(self, nima=0, beta=None):
        """Saving the input values into the fixed arrays for the polynomial

        Input
        -----
        beta: list/array of 2 floats
        """
        if beta is not None:
            self.ima_background[nima] = beta[0]
            self.ima_norm_factors[nima] = beta[1]


    def compare_ima(self, nima=0, nima_museref=None,
                    convolve_muse=0, convolve_reference=0., **kwargs):
        """
        
        Input
        -----
        nima: int
            Index of input image
        nima_museref: int
            Index of second input image for the reference. Default is None, hence ignored
            and the default reference image will be used.

        Create
        ------
        Plots which compare the two input datasets as defined by the indices

        """
        # Getting the first MUSE image (aligned)
        musehdu = self.list_offmuse_hdu[nima]
        if self._debug:
            self._temp_prealigned = self.list_muse_hdu[nima].data

        # Getting data from the MUSE ref image if one is given
        museref = nima_museref is not None
        if museref:
            refhdu = self.list_offmuse_hdu[nima_museref]
        else:
            refhdu = self.list_proj_refhdu[nima]

        threshold_muse = kwargs.pop("threshold_muse", self.ima_threshold[nima])

        musedata, refdata, polypar = get_normfactor(musehdu.data, refhdu.data,
                                                    convolve_data1=convolve_muse,
                                                    convolve_data2=convolve_reference,
                                                    threshold=threshold_muse)
        self.compare(data1=musedata, data2=refdata, header=musehdu.header,
                     suffix_fig=f"{nima:03d}", **kwargs)

    def compare(self, data1, data2, header=None,
                start_nfig=1, nlevels=10, levels=None, convolve_data1=0., convolve_data2=0.,
                showcontours=True, showcuts=True, shownormalise=True, showdiff=True,
                normalise=True, median_filter=True, ncuts=5, percentage=5.,
                suffix_fig="", **kwargs):
        """Compare the projected reference and MUSE image
        by plotting the contours, the difference and vertical/horizontal cuts.
         
        Parameters
        ----------
        data1:
        data2: 2d np.arrays
            Array to compare
        header: Header
            If provided, will be use to get the WCS in the plots. Default is None (ignored).
        polypar: ODR result
            If None, it will be recalculated
        showcontours: bool [True]
        showcuts: bool [True]
        shownormalise: bool [True]
        showdiff: bool [True]
            All options corresponding to 1 specific plot. By default
            show them all (all True)
        ncuts: int [5]
            Number of vertical / horizontal cuts along the ratio
            between the 2 maps to be shown ("cuts")
        percentage: float [5]
            Used to compute which percentile to show
        start_nfig: int [1]
            Number of the matplotlib Figure to start with
        nlevels: int [10]
            Number of levels for the contour plots
        levels: list of float [None]
            Specific list of levels if any (default is None)
        convolve_data1: float [0]
            If not 0, will convolve with a gaussian of that sigma
        convolve_data2: float [0]
            If not 0, will convolve the reference image
            with a gaussian of that sigma
        savefig (bool): False
            If True, will save the figure into a png
        suffix_fig: str
            Suffix name to add to the figure filenames

        Makes a maximum of 4 figures
        """
        border = kwargs.pop("border", self.border)
        chunk_size = kwargs.pop("chunk_size", self.chunk_size)
        savefig = kwargs.pop("savefig", True)
        threshold = kwargs.pop("threshold", 0.)

        # Getting the data
        _, _, polypar = get_normfactor(data1, data2,
                                       convolve_data1=convolve_data1,
                                       convolve_data2=convolve_data2,
                                       median_filter=median_filter, border=border,
                                       chunk_size=chunk_size,
                                       threshold=threshold)

        # If normalising, use the polypar slope and background
        if normalise:
            if self.verbose:
                upipe.print_info(f"Renormalising the data as: Normalised = "
                                 f"{polypar.beta[1]:8.4e} * ({polypar.beta[0]:8.4e} + MUSE)")

            data1 = my_linear_model(polypar.beta, data1)

        # Save the frames in case this is needed
        if self._debug:
            self._temp_aligned = data1
            self._temp_reference = data2

        # WCS for plotting using astropy
        if header is not None:
            plotwcs = awcs.WCS(header)
        else:
            plotwcs = None

        # Preparing the figure
        current_fig = start_nfig
        self.list_figures = []
        foldfig = self.figures_folder_name

        if suffix_fig != "":
            suffix_fig = f"_{suffix_fig}"

        # Starting the plotting
        if shownormalise:
            plot_polypar(polypar, labels=["MuseData", "RefData"], figfolder=foldfig,
                         fignum=current_fig, namefig=f"align_norm_scatter{suffix_fig}.png",
                         savefig=savefig)
            self.list_figures.append(current_fig)
            current_fig += 1

        if showcontours:
            plot_compare_contours(data1, data2, plotwcs=plotwcs, fignum=current_fig,
                                  labels=['MUSE', 'REF'], nlevels=nlevels, levels=levels,
                                  figfolder=foldfig, namefig=f"align_contours{suffix_fig}.png",
                                  title=f'Image {suffix_fig}')
            self.list_figures.append(current_fig)
            current_fig += 1

        if showcuts:
            plot_compare_cuts(data1, data2, ncuts=ncuts, fignum=current_fig, figfolder=foldfig,
                              namefig=f"align_cuts{suffix_fig}.png", savefig=savefig)
            self.list_figures.append(current_fig)
            current_fig += 1

        if showdiff:
            plot_compare_diff(data1, data2, plotwcs=plotwcs, fignum=current_fig,
                              figfolder=foldfig, percentage=percentage,
                              namefig=f"align_diff{suffix_fig}.png", savefig=savefig)
            self.list_figures.append(current_fig)
            current_fig += 1
