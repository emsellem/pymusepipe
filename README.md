# pymusepipe

WARNING: Doc need to be updated since many major changes. Coming asap.

This is a Python wrapper for the MUSE/VLT pipeline recipes. This runs 
using the esorex command lines (in parallel, using by default likwid, although 
other options are available). This wrapper is meant to reduce MUSE OBs automatically, 
after dumping the fits files in a given folder.

This package includes an alignment module which may be useful, independently given 
a set of images to align and assuming a reference image is provided. It can spit out 
an "OFFSET_LIST" MUSE compatible fits table including the flux scaling factors.

pymusepipe is also made for multi-pointing mosaics and multi-targets surveys
as it will process targets automatically when provided a specific dictionary 
of which target and which pointings to consider.

## 1 - Getting Started
Here are some basic instructions to get you going with pymusepipe.

### Prerequisites
pymusepipe uses Python 3. It is not compatible with python 2.7. If you are 
still using python 2.7 I would recommend you to switch asap as it won't be 
supported much longer anyway.

pymusepipe needs a number of standard modules/packages including:
   * **numpy**
   * **scipy** for the alignment module
   * **matplotlib** if you wish to make some plots (e.g. via the check_pipe module)
   * **astropy** as a general provider of many useful functions and structures, most importantly the astropy Tables.

The convolution package of pymusepipe (allowing full convolution of a datacube) uses
**pypher** which is thus a requirement for that module to work.

This also includes a heavy-based usage of **mpdaf**, which is a super useful package
developed by the MUSE GTO-CRAL Team, to process and analyse datacubes, and more specifically
MUSE cubes, images and spectra.

### Installing

You can install this package via Pypi via a simple:
```buildoutcfg
pip install pymusepipe
```

You can obviously also install it by cloning it from github, or 
downloading the source (from github) and do something like:
```
python setup.py develop
```
I advise to use the "develop" option as it actually does not copy 
the files in your system but just create a link. 
In that way you can easily update the source software without 
reinstalling it. The link will directly use the source which has been udpated.

The other option is to use the standard "install" option:
```
python setup.py install
```

## 2 - Basic Usage - Dealing with OBs, individually
The pymusepipe wrapper is meant to provide the user with a simple way to 
run the MUSE pipeline.

Only 3 steps are needed: preparing the data (download), preparing the 
configuration files (templates are provided) which should take a couple of minutes, 
and using the code (a few lines at most). The package has many other hidden functionalities
which I'll document as soon as I can.

I recommend to use ```Ipython``` as an environment, possibly via 
a ```screen``` which would allow you to disconnect from the server that 
actually runs the commands. Jupyter notebooks may also be very handy 
although I would not necessarily recommend them for this specific purpose as 
the running of the pipeline may take hours.

### Step 1 = Preparing your data
The first thing to do is to prepare the folder structure to welcome your 
MUSE datasets. This is actually rather simple.

Imagine you have
   * a target or field named "NGC1000" (just for the sake of using an example).
   * several pointings (each pointing can represent a set of OBs which have 
   roughly the same coordinates, best is that they have a very significant 
   overlap): 1, 2, and 3. Note that this 'pointing' structure may not reflect your
   own observational strategy. There are cases where one OB includes different pointings.
   What is important here is that each 'pointing' is either a full single OB, or several
   OBs with the same pointing.

Then under your data root folder <my_data_folder> create the following folder structure:
```
<my_data_folder>/NGC1000
    ./P01
       ./Raw
   ./P02
       ./Raw
   ./P03
       ./Raw
```
Each pointing/OB has a "Raw" folder.

The next step is to download your MUSE data from the ESO web site, 
and put all the raw files (fits/fits.gz) into each individual "Raw" folder, 
associated with the right pointing.

### Step 2 = Preparing your configuration files
pymusepipe only needs 2 configurations ASCII files: 1 for the calibration 
(calib_tables.dic) and one for the root folders (rc.dic). 
**Examples** of such files are provided in the "config_templates" 
folder of the pymusepipe package.

### Step 3 = Running the pipeline
The pipeline is meant to be run automatically from a given python 
structure. This should thus take the user only a few lines of codes, 
including one to import pymusepipe, a couple to define the names of 
the configuration files, one to initialise the musepipe python structure, 
and one to launch the recipes. Here is an example of how this should look:

```buildoutcfg
# Import the modules
import pymusepipe as pmp
from pymusepipe import musepipe

# define the names of the configuration files
rcfile = "/my_data/MUSE/Config/rc.dic"
calfile = "my_data_MUSE/Config/calib_tables.dic"

# Initialisation of the python - MusePipe Class - structure
mypipe = musepipe.MusePipe(targetname="NGCXXXX", pointing=1, rc_filename=rcfile,
                          cal_filename=calfile, log_filename="NGCXXXX_version01.log",
                          fakemode=False, overwrite_astropy_table=True, 
                          filter_list="white,Cousins_R",
                          filter_for_alignment="Cousins_R")

# Launching the pipeline
mypipe.run_recipes()
```
Some explanation may be needed to understand what is happening:
   * "targetname": is just the name of the target, used to decided where the data will be
   * "pointing": will be used as "P01" for pointing=1, etc.
   * "logfile": name of the logging file. Note that this logfile is actually 
   a shell-like file which can be used to re-run the pipeline one step at a time. Note that
   there will be also 2 more files created using that name <logfile_name>, namely:
   a file named <logfile_name>.out and <logfile_name>.err which will
   contain the full output of the commands, and the error messages (stdout, stderr, resp.).
   * "fakemode": you can set this to True if you just wish to initialise 
   things without actually running any recipes. The pipeline will only set 
   things up but if you run any recipes will only "fake" them (not launch any 
   esorex command, only spitting the code out)
   * "filter_list": list of filter names to use to reconstruct images 
   when building up cubes. This should be part of the filter_list fits 
   table provided (see calib_tables config file).
   * "filter_for_alignment": specific filter name used for alignment between exposures.

Other options can be useful:
   * "musemode": this is by default "WFM_NOAO_N" which is the most often 
   used MUSE mode. This will filter out exposures not compatible with the 
   given mode. So please beware.
   * "reset_log": will reset the log file. By default it is False, hence 
   new runs will be appended.
   * "overwrite_astropy_table": by default this is False. If True, 
   new runs will rewrite the Astropy output tables.
   * "time_astrometry": by default it is False, meaning the pipeline 
   will try to detect a GEOMETRY and ASTROMETRY Files delivered with the Raw
   files by ESO. If set to True, it will use the time dependent astro/geo files
   provided by the GTO Team but you would need to make these available on your system.
   Hence I would recommend to keep the default (False).

## 3- Advanced Usage - Targets and Mosaics

## 4- Alignment module
Details to come.

## 5- Further details
Here we provide a bit more details about a few issues regarding 
the configuration files or set up parameters.

### Configuration files

#### rc configuration file
It contains 3 lines, with: *musecalib*, *musecalib_time* and *root*.

   * *root* provides the root folder for your data. For Target NGCXXXX, 
   and Pointing 1, the Raw data will be looked for 
   in *root*/TargetXXXX/P01/Raw (see *Preparing your data* above).
   * musecalib should contain the standard MUSE calibration files. 
   These are usually distributed in the MUSE kit within a 
   "muse-calib-x.x.x/cal" folder.
   * musecalib_time: time dependent geometry and astrometry files 
   (the correspondence between observing run dates and specific 
   files are given in the dic_geo_astrowcs_table in musepipe.py).

#### calib_tables configuration file
It contains a series of given fits files which will be used by the pipeline. Most names are self-explantory. That includes:
   * *geo_table* and *astro_table*: only used if you don't rely on the default time dependent geometry files (see rc file).
   * *badpix_table*, *vignetting_mask*, *std_flux_table*, *extinct_table*, *line_catalog* all usually provided with the MUSE pipeline.
   * *filter_list* : name of the fits filter list. This is used in case you wish to provide your own. Note that it needs to follow the MUSE standard for such a table.

### Recipes
Most MUSE pipeline recipes are run while run_all_recipes is launched. This can be changed in the prep_recipes_pipe.py or just scripted. Actually run_all_recipes() is just a function which launches (assuming 'mypipe' is your MusePipe structure, see above):
```
        mypipe.run_bias()
        mypipe.run_flat()
        mypipe.run_wave()
        mypipe.run_lsf()
        mypipe.run_twilight(illum=illum)
        mypipe.run_scibasic_all(illum=illum)
        mypipe.run_standard()
        mypipe.run_sky(fraction=fraction)
        mypipe.run_prep_align()
        mypipe.run_align_bypointing()
        mypipe.run_align_bygroup()
        mypipe.run_scipost()
        mypipe.run_scipost(expotype="SKY", offset_list=False, skymethod='none')
        mypipe.run_combine_pointing()
```
where "illum" is a boolean (default is True), fraction is 0.8.

Feel free to launch these steps one by one (the order is important as in any data reduction process).

### Structure of the output

#### Folders
The structure of the output is driven by a set of folder names described in init_musepipe.py in a few dictionaries (dic_input_folders, dic_folders, dic_folders_target). You can in principle change the names of the folders themselves, but I would advise against that.

The pipeline will create the folder structure automatically, checking whether the folders exist or not.

#### Log files
2 basic log files are spitted out: one is the Esorex output which will 
be stored in the "Esorex_log" folder. The other one will be in the "Log" 
folder with the name provided at start: that one is like a shell script 
which can be used to rerun things directly via the command line.
In the "Log" folder, there will also be, for each log file, a file ".out" and
one with ".err" extensions, respectively including all the stdout and stderr
messages. This may be useful to trace details in the data reduction and problems.

#### Astropy Tables
Each recipe will trigger the creation of a astropy Table. 
These are stored under "Astro_Tables". You can use these to monitor 
which files have been processed or used.

#### Python structure
Most of the information you may need is actually stored in the 
python "MusePipe" class structure. More details to come.

## Authors
* **Eric Emsellem** [2017-2020], at ESO and CRAL

## License

This project is licensed under the MIT License - see the 
[LICENSE](LICENSE) file for details

## Acknowledgments
I would like to thank people who have initially sent me 
their code-samples, including Bernd Husemann, Dimitri Gadotti, 
Lodovico Coccato, Mark den Brok. I would also like to specifically 
and warmly thank Rebecca McElroy who supported me with the MUSE
data reduction at the early stages of the development of this package,
and Francesco Santoro who has significantly contributed 
in the testing, debugging of the code (version 1), and proposed dedicated python lines 
to be integrated in pymusepipe (e.g., alignment module).
