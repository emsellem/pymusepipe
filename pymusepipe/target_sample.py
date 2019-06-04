# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS initialisation of folders
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# Standard modules
import os
from os.path import join as joinpath
from pymusepipe import util_pipe as upipe
from pymusepipe.musepipe import MusePipe

# ----------------- Galaxies and Pointings ----------------#

# Sample of galaxies
# For each galaxy, we provide the pointings numbers and the run attached to that pointing
MUSEPIPE_sample = {
        "NGC628": {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
        "NGC1087": {1:1}, 
        "NGC1672": {1:1, 2:1, 3:1, 4:1, 5:1}
        }

# List of MUSEPIPE Observing runs
# For each Observing run, we provide the start and end dates
MUSEPIPE_runs = {
        'Run00' : ['P099','2017-10-01','2017-10-31'],
        'Run01' : ['P100','2017-10-01','2017-10-31'], 
        'Run02' : ['P101','2017-11-01','2017-11-30']
        }

####################################################
# Defining classes to get samples and objects
####################################################
class MusepipeSample(object) :
    def __init__(self) :
        self.sample = MUSEPIPE_sample
        self.targets = MUSEPIPE_sample.keys()

class MusepipeTarget(object) :
    def __init__(self, targetname=None, list_pointings=[1]) :
        if targetname not in MUSEPIPE_sample.keys() :
            upipe.print_error("ERROR: no Galaxy named {gal} in the defined sample".format(gal=targetname))
            return

        # Galaxy name
        upipe.print_info("Initialising Target {name}".format(name=targetname))
        self.targetname = targetname

        # Info of the pointings and extracting the observing run for each pointing
        self.info_pointings = MUSEPIPE_sample[targetname]
        if any([_ not in self.info_pointings.keys() for _ in list_pointings]) :
            upipe.print_error("ERROR: no pointing {0} for the Galaxy".format(list_pointings))
            return
        self.list_pointings = list_pointings
        self.observing_run = [self.info_pointing[_] for _ in self.list_pointings]

    def _get_file_name(self, suffix, pointing):
        return "{0}_P{1:02d}.txt".format(suffix, pointing)

    def run_pipeline(self, list_pointings=[1], fakemode=False, 
            suffix_logfile="logfile", suffix_rcfile="rcfile", suffix_calfile="calfile"):
        """Run the pipeline for all pointings in the list
        """
        if any([_ not in self.info_pointings.keys() for _ in self.list_pointings]) :
            upipe.print_error("ERROR: some pointing are not in "
                "the available list ({0})".format(self.list_pointings))
            return

        # Setting up the suffixes for the files
        self.suffix_logfile = suffix_logfile
        self.suffix_calfile = suffix_calfile
        self.suffix_rcfile = suffix_rcfile

        # Loop on the pointings
        self.pipelines = []
        upipe.print_info("---- Starting the Data Reduction ----")
        self.history = []
        for pointing in list_pointings:

            # Setting up the names of the output files
            logfile = self._get_logfile_name(self.suffix_logfile, pointing)
            calfile = self._get_calfile_name(self.suffix_calfile, pointing)
            rcfile = self._get_rcfile_name(self.suffix_rcfile, pointing)
            
            python_command = ("mypipe = musepipe.MusePipe(targetname={0}, "
                    "pointing={1}, rc_filename={2}, cal_filename={3}, "
                    "logfile={4}, fakemode={5}, "
                    "nocache=False)".format(targetname, pointing, rcfile, calfile, 
                            logfile, fakemode))
            upipe.print_info("====== START - POINTING {0:2d} ======".format(pointing))
            upipe.print_info(python_command)
            upipe.print_info("====== END   - POINTING {0:2d} ======".format(pointing))
            self.history.append(python_command)
            mypipe = MusePipe(targetname=targetname, pointing=pointing, 
                    rc_filename=rcfile, cal_filename=calfile, 
                    logfile=logfile, fakemode=fakemode, nocache=False)

            self.pipelines.append(mypipe)
            mypipe.run_all_recipes()

    def combine(self):
        pass
