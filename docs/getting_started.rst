======
Getting Started
======

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
   * several **datasets**. In the simplest cases this corresponds to data from one MUSE Observing Block (OB), including all the calibration and science object raw data files, as downloaded from the ESO archive. In practice pymusepipe will also reduce several OBs, provided all the necessary calibrations are avaiable. Some functionality requires a distinction between **dataset** and **pointing**. This distinction is described in :doc:`mosaicking`.

Then under your data root folder <my_data_folder> create the following folder structure::

   <my_data_folder>/NGC1000
      ./OB001
         ./Raw
      ./OB002
        ./Raw
      ./OB003
        ./Raw

Each dataset, or OB for short, has a ``Raw`` folder.

The next step is to download your MUSE data (including raw calibrations) from the ESO web site, and put all the raw files (in fitsor fits.gz format) into each individual ``Raw`` folders, associated with the right dataset.

Step 2: Preparing your configuration files
""""""""""""""""""""""""""""""""""""""""""""""

pymusepipe only needs two configurations ascii files: 

   #. ``calib_tables.dic``, which contains a series of given fits files which will be used by the pipeline. Most names are self-explantory. That includes:
      * *geo_table* and *astro_table*: only used if you don't rely on the default time dependent geometry files (see ``rc.dic`` file).
      * *badpix_table*, *vignetting_mask*, *std_flux_table*, *extinct_table*, *line_catalog* all usually provided with the MUSE pipeline.
      * *filter_list* : name of the fits filter list. This is used in case you wish to provide your own. Note that it needs to follow the MUSE standard for such a table.
   #. ``rc.dic``, which provides the root folders for the static calibration files and for your datasets.
      * *root* provides the root folder for your data. For Target NGC1000, and OB 1, the Raw data will be looked for in *root*/Target1000/OB001/Raw.
      * musecalib should contain the standard MUSE calibration files. These are distributed in the MUSE pipeline installation in a "muse-calib-x.x.x/cal" folder.
      * musecalib_time: time dependent geometry and astrometry files (the correspondence between observing run dates and specific files are given in the dic_geo_astrowcs_table in musepipe.py).

#### rc configuration file
It contains 3 lines, with: *musecalib*, *musecalib_time* and *root*.

   

#### calib_tables configuration file
It contains a series of given fits files which will be used by the pipeline. Most names are self-explantory. That includes:
   * *geo_table* and *astro_table*: only used if you don't rely on the default time dependent geometry files (see rc file).
   * *badpix_table*, *vignetting_mask*, *std_flux_table*, *extinct_table*, *line_catalog* all usually provided with the MUSE pipeline.
   * *filter_list* : name of the fits filter list. This is used in case you wish to provide your own. Note that it needs to follow the MUSE standard for such a table.

Examples of such files are provided in the ``config_templates`` folder of the pymusepipe package.

Step 3: Running the pipeline
""""""""""""""""""""""""""""""""""""""""""""""

The pipeline is meant to be run automatically from a given python  structure. This should thus take the user only a few lines of codes, including one to import pymusepipe, a couple to define the names of the configuration files, one to initialise the musepipe python structure, 
and one to launch the recipes. Here is an example of how this should look::

   # Import the modules
   import pymusepipe as pmp
   from pymusepipe import musepipe
   
   # define the names of the configuration files
   rcfile = "/my_data/MUSE/Config/rc.dic"
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
   * ``logfile``: name of the logging file. This logfile is actually a shell-like file which can be used to re-run the pipeline one step at a time. Two more files will also be created, namely <logfile_name>.out and <logfile_name>.err which will contain the full output of the commands, and the error messages (stdout, stderr, respectively).
   * ``fakemode``: you can set this to True if you just wish to initialise things without actually running any recipes. The pipeline will only set things up but if you run any recipes will only "fake" them (not launch any esorex command, only spitting the log out)
   * ``filter_list``: list of filter names to use to reconstruct images when building up cubes. This should be part of the filter_list fits table provided (see ``calib_tables`` config file).
   * ``filter_for_alignment``: specific filter name used for alignment between exposures.

Other options can be useful:

   * ``musemode``: this is by default ``WFM_NOAO_N`` which is the most often used MUSE mode. This will filter out exposures not compatible with the given mode.
   * ``reset_log``: will reset the log file. By default it is False, hence new runs will be appended.
   * ``overwrite_astropy_table``: by default this is False. If True, new runs will rewrite the Astropy output tables.
   * ``time_astrometry```: by default it is False, meaning the pipeline will try to detect a GEOMETRY and ASTROMETRY Files delivered with the Rawfiles by ESO. If set to True, it will use the time dependent astro/geo files provided by the GTO Team but you would need to make these available on your system.Hence I would recommend to keep the default (False).


""""""""""""""""""""""""""""""""""""""""""""""