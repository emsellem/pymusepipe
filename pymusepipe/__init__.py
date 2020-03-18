""" Copyright (C) 2017 ESO/Centre de Recherche Astronomique de Lyon (CRAL)
    print pymusepipe.__LICENSE__  for the terms of use

    This package is a wrapper around MUSE pipeline commands to reduce
    muse raw data frames. It includes modules for aligning and convolving the frames.
    It also has some basic routines wrapped around mpdaf.
"""
from .version import __date__, __version__

from pymusepipe import (align_pipe, check_pipe, combine, config_pipe, create_sof,
                        cube_convolve, emission_lines, graph_pipe, init_musepipe,
                        mpdaf_pipe, musepipe, prep_recipes_pipe, recipes_pipe,
                        target_sample, util_pipe)

