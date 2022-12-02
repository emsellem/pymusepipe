==================
Alignment
==================

Limitations of run_align recipes
""""""""""""""""""""""""""""""""""""""""""""""
The esorex implementation of the alignment procedure (implemented in
the :py:func:`pymusepipe.prep_recipes_pipe.PipePrep.run_align_bydataset()`)
for multiple object exposures suffers from some severe limitations:

* It does not perform absolute astrometry, but merely fixes the astrometry of
  subsequent exposures to the WCS of the first one. This is problematic for comparison of MUSE
  data with external datasets.
* It works by finding and matching point sources across white light images from
  multiple exposures. This requires the images to contain a sufficient number of point sources.
  Moreover, in case of mosaics, is requires the *overlap region* between different MUSE pointings
  to contain a sufficient number of point sources. In practice, this requirement is very limiting.

Pymusepipe provides the :py:func:`pymusepipe.align_pipe` module to overcome both these limitations. 