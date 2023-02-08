"""Utility files and functions for wavelengths
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "MIT License"
__contact__   = " <eric.emsellem@eso.org>"

# Modules
from astropy import constants as const

from . import util_pipe as upipe

full_muse_wavelength_range = [4000., 10000.]

list_emission_lines = {
        "CaIIK":          [3933.663,   3934.777],
        "CaII":           [3968.468,   3969.591],
        "[OII]3726":      [3726.03,    3727.09],
        "[OII]3729":      [3728.82,    3729.88],
        "[NeIII]3869":    [3868.71,    3869.81],
        "[NeIII]3967":    [3967.41,    3968.53],
        "H12":	          [3750.15,    3751.22],
        "H11":	          [3770.63,    3771.70],
        "H10":            [3797.90,    3798.98],
        "H9":             [3835.39,    3836.48],
        "H8":             [3889.05,    3890.15],
        "He":             [3970.07,    3971.19],
        "Hd":             [4101.76,    4102.92],
        "Hg":             [4340.47,    4341.69],
        "Hb":             [4861.33,    4862.69],
        "Ha":             [6562.80,    6564.61],
        "[OIII]5007":     [5006.84,    5008.24],
        "[OIII]4959":     [4958.92,    4960.30],
        "[OIII]4363":     [4363.21,    4364.44],
        "[NII]6549":      [6548.03,    6549.84],
        "[NII]6583":      [6583.41,    6585.23],
        "[NII]5755":      [5754.64,    5756.24],
        "[SII]6717":      [6716.47,    6718.32],
        "[SII]6731":      [6730.85,    6732.71],
        "NaI5890":        [5889.951,   5891.583],
        "NaI5895":        [5895.924,   5897.558],
        "HeI3889":        [3888.65,    3889.75],
        "HeI5875":        [5875.67,    5877.30],
        "HeI6678":        [6678.152,   6679.996],
        "[OI]6300":       [6300.30,    6302.04],
        "[SIII]6312":     [6312.1,     6313.8],
        "[NeII]6583":     [6583.41,    6585.23],
        "[NeII]6548":     [6548.03,    6549.84],
        "[NeII]5754":     [5754.64,    5756.24],
        "[OI]5577":       [5577.3387,  5578.8874],
        "Mgb5167":        [5167.321,   5168.761],
        "Mgb5172":        [5172.684,   5174.125],
        "Mgb5183":        [5183.604,   5185.048],
        "[ArIII]7135":    [7135.8,     7137.8],
        "[ArIII]7751":    [7751.1,     7753.2],
        "CaII8498":       [8498.03,    8500.36],
        "CaII8542":       [8542.09,    8544.44],
        "CaII8662":       [8662.14,    8664.52],
        "[SIII]9068":     [9068.6,     9071.1],
        "[SIII]9530":     [9530.6,     9533.2],
        "P11":            [8862.783,   8865.217],
        "P10":            [9014.910,   9017.385],
        "P9":             [9229.014,   9231.547],
        "P8":             [9545.972,   9548.590],
        "P7":             [10049.373, 10052.128],
        "Pg":             [10938.095, 10941.091],
        "Pb":             [12818.08,  12821.59],
        "Pa":             [18751.01,  18756.13],
        "Brg":	          [21655.29,  21661.20],
            }


def print_emission_lines() :
    """Printing the names of the various emission lines
    """
    print(list_emission_lines.keys())


def doppler_shift(wavelength, velocity=0.):
    """Return the redshifted wavelength
    """
    doppler_factor = np.sqrt((1. + velocity / const.c.value) / (1. - velocity / const.c.value))
    return wavelength * doppler_factor


def get_emissionline_wavelength(line="Ha", velocity=0., redshift=None, medium='air'):
    """Get the wavelength of an emission line, including a correction
    for the redshift (or velocity)
    """
    index_line = {'vacuum': 0, 'air': 1}
    # Get the velocity
    if redshift is not None : velocity = redshift * const.c

    if line is None:
        return -1.
    elif line not in list_emission_lines:
        upipe.print_error("Could not guess the emission line you wish to use")
        upipe.print_error("Please review the 'emission_line' dictionary")
        return -1.

    if medium not in index_line:
        upipe.print_error(f"Please choose between one of these media: {list(index_line.keys())}")
        return -1.

    wavel = list_emission_lines[line][index_line[medium]]
    return doppler_shift(wavel, velocity)


def get_emissionline_band(line="Ha", velocity=0., redshift=None, medium='air', lambda_window=10.0):
    """Get the wavelengths of an emission line, including a correction
    for the redshift (or velocity) and a lambda_window around that line (in Angstroems)

    Parameters
    ----------
    line: name of the line (string). Default is 'Ha'
    velocity: shift in velocity (km/s)
    medium: 'air' or 'vacuum'
    lambda_window: lambda_window in Angstroem
    """
    red_wavel = get_emissionline_wavelength(line=line, velocity=velocity, redshift=redshift, medium=medium)
    # In case the line is not in the list, just return the full lambda Range
    if red_wavel < 0 :
        return full_muse_wavelength_range
    else:
        return [red_wavel - lambda_window/2., red_wavel + lambda_window/2.]
