==============================
Getting Started
==============================

Basic Usage - Dealing with OBs, individually
""""""""""""""""""""""""""""""""""""""""""""""
The pymusepipe wrapper is meant to provide the user with a simple way to 
run the MUSE pipeline.

Only three steps are needed:

 #. preparing the data (download), 
 #. preparing the configuration files (templates are provided), 
 #. using the code (a few lines at most). 

I recommend to use ``Ipython`` as an environment, possibly via 
a ``screen`` or ``tmux`` which would allow you to disconnect from the server that 
actually runs the commands. 

Step 1: Preparing your data
""""""""""""""""""""""""""""""""""""""""""""""
The first thing to do is to prepare the folder structure to welcome your 
MUSE datasets. 

Imagine you have:

* a target or field named e.g. ``NGC1000``.
* several **datasets**. In the simplest cases this corresponds to data from one
  MUSE Observing Block (OB), including all the calibration and science object raw data files, as
  downloaded from the ESO archive. In practice pymusepipe will also reduce several OBs, provided
  all the necessary calibrations are avaiable. Some functionality requires a distinction between
  **pointing**,  **dataset**, and **group**. This distinction is described in :doc:`mosaicking`, but in the simplest terms a pointing is a set of exposures covering the same FoV on the sky, a dataset is a set of exposures obtained within the same OB, which therefore share the same calibration data, and a group is a set od datasets which the user wishes to mosaic together, usually because they cover the same science target.

Then under your data root folder <my_data_folder> create the following folder structure::

   <my_data_folder>/NGC1000
      ./OB001
         ./Raw
      ./OB002
        ./Raw
      ./OB003
        ./Raw

Each dataset, or OB for short, has a ``Raw`` folder.

The next step is to download your MUSE data (including raw calibrations) from the
ESO web site, and put all the raw files (in fits or fits.gz format) into each individual
``Raw`` folders, associated with the right dataset.

Step 2: Preparing your configuration files
""""""""""""""""""""""""""""""""""""""""""""""

pymusepipe only needs two configurations ascii files: 

#. ``calib_tables.dic``, which contains a series of file names associated
   with muse static calibration files and other configuration files (e.g. fitler lists). Most
   names are self-explantory. These include:

   * *geo_table* and *astro_table*: static files, time dependent geometry files can be specified
     (see ``rc.dic``).
   * *badpix_table*, *vignetting_mask*, *std_flux_table*, *extinct_table*, *line_catalog*,
     statical calibrations provided with the MUSE pipeline, no need to change these.
   * *filter_list* : used in case you wish to provide your own. Note that the file it needs
     to follow the MUSE standard for such a table.

#. ``rc.dic``, which provides the root folders for the static calibration files and for your datasets.

   * *root* provides the root folder for your data. For Target NGC1000, and OB 1,
     the Raw data will be looked for in *root*/NGC1000/OB001/Raw.
   * *musecalib* should contain the standard MUSE calibration files.
     These are distributed in the MUSE pipeline installation in a "muse-calib-x.x.x/cal" folder.
   * *musecalib_time*: time dependent geometry and astrometry files (the correspondence
     between observing run dates and specific files are hard-coded into pymusepipe).

Examples of such files are provided in the ``config_templates`` folder of the pymusepipe package.

Step 3: Running the pipeline
""""""""""""""""""""""""""""""""""""""""""""""

Here is an example of how to run the pipeline to reduce a single OB (dataset)::

   # Import the modules
   import pymusepipe as pmp
   from pymusepipe import musepipe
   
   # define the paths to the two configuration files
   rcfile = "my_data/MUSE/Config/rc.dic"
   calfile = "my_data_MUSE/Config/calib_tables.dic"
   
   # Initialisation of the python - MusePipe Class - structure
   mypipe = musepipe.MusePipe(targetname="NGC1000", dataset=1, rc_filename=rcfile,
                           cal_filename=calfile, log_filename="NGC1000_version01.log",
                           fakemode=False, overwrite_astropy_table=True, 
                           filter_list="white,Cousins_R",
                           filter_for_alignment="Cousins_R")
                         
   # Launching the pipeline
   mypipe.run_recipes()

That's it! Your data has now been reduced!

Some explanation may be needed to understand what is happening:

   * ``targetname``: is just the name of the target, used to decided where the data will be
   * ``dataset``: the number of the OB that will be used, namely "OB001" etc.
   * ``logfile``: name of the logging file. 
   * ``fakemode``: you can set this to True if you just wish to initialise things without
     actually running any recipes. The pipeline will only set things up but if you run any recipes
     will only "fake" them (not launch any esorex command, only spitting the log out)
   * ``filter_list``: list of filter names to use to reconstruct images when building
     up cubes. This should be part of the filter_list fits table provided (see ``calib_tables``
     config file).
   * ``filter_for_alignment``: specific filter name used for alignment between exposures.

Other options can be useful:

   * ``musemode``: this is by default ``WFM_NOAO_N`` which is the most often used MUSE mode. This will filter out exposures not compatible with the given mode.
   * ``reset_log``: will reset the log file. By default it is False, hence new runs will be appended.
   * ``overwrite_astropy_table``: by default this is False. If True, new runs will rewrite the Astropy output tables.
   * ``time_astrometry```: by default it is False, meaning the pipeline will try to detect a GEOMETRY and ASTROMETRY Files delivered with the Rawfiles by ESO. If set to True, it will use the time dependent astro/geo files provided by the GTO Team but you would need to make these available on your system.Hence I would recommend to keep the default (False).


.. attention:: 
   This pipeline flow closely mirrors the standard data reduction for MUSE data implemented
   by the e.g. EsoReflex workflow. Pymusepipe offers alternative recipes to perform alignment
   (:doc:`alignment`) and mosaicking (:doc:`mosaicking`). For best results, therefore, we do not
   recommend running the above workflow. Examples workflows are presented in :doc:`phangs_example`.

Structure of the output
""""""""""""""""""""""""""""""""""""""""""""""

Folders
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The structure of the output is driven by a set of folder names described in
:py:func:`pymusepipe.init_musepipe` in a few dictionaries (:py:func:`dic_input_folders`,
:py:func:`dic_folders`, :py:func:`dic_folders_target`). You can in principle change the names
of the folders themselves, although it is not advisable.

The pipeline will create the folder structure automatically, checking whether the folders exist or not.

Log files
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Two basic log files are produced: one is the Esorex output which will be stored in the
"Esorex_log" folder. The other one will be in the "Log" folder with the name provided at start:
that one is like a shell script which can be used to rerun things directly via the command line.
In the "Log" folder, there will also be, for each log file, a file ".out" and one with ".err"
extensions, respectively including all the stdout and stderr messages. This may be useful to trace
details in the data reduction and problems.

Astropy Tables
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each recipe will trigger the creation of a astropy Table. 
These are stored under "Astro_Tables". You can use these to monitor which files have been
processed or used.

Sof files
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sof files are saved under the "Sof" directory for each esorex recipes used in the pipeline.
These are useful to see exactly which files are processed by each esorex recipe.

Python structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Most of the information you may need is actually stored in the python
:py:class:`pymusepipe.musepipe.MusePipe` class structure. More details to come.