========
Welcome
========
Pymusepipe is a Python package which serves as a wrapper around the main processing steps of the MUSE data reduction pipeline (Weilbacher et al. `2019 <https://ui.adsabs.harvard.edu/abs/2020A%26A...641A..28W/abstract>`_). Pymusepipe includes a simple data organiser and prescriptions for the structure of the data files (but no database per se), a wrapper around the main functionalities of MUSE data reduction pipeline, accessed via `EsoRex <https://www.eso.org/sci/software/pipelines/muse/>`_ command-line recipes, to remove the instrumental signatures. Pymusepipe additionally provides a set of modules supporting the alignment, mosaicking, two-dimensional and three-dimensional convolution.

pymusepipe is also made for multi-pointing mosaics and multi-targets surveys
as it will process targets automatically when provided a specific dictionary 
of which target and which pointings to consider.

A description of the pipeline and its usage to reduce data from the PHANGS-MUSE survey are presented in Emsellem et al. (`2022 <https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.191E/abstract/>`_)

.. admonition:: Contact

   The pymusepipe module is maintained by Eric Emsellem. 
   contact via :email:`eric.emsellem@eso.org`

.. attention:: 
   Please do not forget to cite Emsellem et al. (`2022 <https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.191E/abstract/>`_)
   and the MUSE data reduction pipeline paper (Weilbacher et al. `2019 <https://ui.adsabs.harvard.edu/abs/2020A%26A...641A..28W/abstract>`_) if you make use of pymusepipe in your work. In particular, we suggest you add the following text (or equivalent) to the data reduction section of your work. 
   
   *The dataset was reduced using recipes the MUSE data processing pipeline software (Weilbacher et al. 2019). All recipes were executed with ESOREX commands, wrapped around using the dedicated python package pymusepipe (Emsellem et al. 2022).*

GitHub Repository
-----------------------------

You can access the source code of pymusepipe and its previous releases directly in its official GitHub repository `https://github.com/emsellem/pymusepipe <https://github.com/emsellem/pymusepipe>`_.

|


