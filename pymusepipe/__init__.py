""" Copyright (C) 2017 ESO/Centre de Recherche Astronomique de Lyon (CRAL)
    print pygme.__LICENSE__  for the terms of use

    This package is a wrapper around MUSE pipeline commands to reduce
    muse dataframes.
"""
from .version import __init__, __date__

from pymusepipe import (init_musepipe, musepipe, target_sample, graph_pipe, mpdaf_pipe,
        check_pipe, emission_lines, prep_recipes_pipe, recipes_pipe, create_sof, util_pipe,
        align_pipe, combine)

#__all__ = ['init_musepipe', 'musepipe', 'graph_pipe', 'mpdaf_pipe', 'check_pipe',
#       'emission_lines', 'prep_recipes_pipe', 'recipes_pipe', 'create_sof']

