""" Copyright (C) 2017 ESO/Centre de Recherche Astronomique de Lyon (CRAL)
    print pygme.__LICENSE__  for the terms of use

    This package is a wrapper around MUSE pipeline commands to reduce
    muse dataframes.
"""
from pymusepipe import * 
import musepipe as mpipe
from musepipe import *
__all__ = ['init_musepipe', 'musepipe', 'graph_pipe', 'mpdaf_pipe', 'check_pipe',
        'emission_lines', 'recipes_pipe', 'create_sof']
