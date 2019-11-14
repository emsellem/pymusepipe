# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS target sample module
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
from pymusepipe.config_pipe import dic_user_folders, PHANGS_config
from .version import __version__ as version_pack

# ----------------- Galaxies and Pointings ----------------#

# Sample of galaxies
# For each galaxy, we provide the pointings numbers and the run attached to that pointing
dic_SAMPLE_example = {
        "NGC628": ['P100', {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}],
        "NGC1087": ['P101', {1:1}], 
        }

# Update the rc file with a subfolder name
def update_rcfile(filename, subfolder=""):
    """Update the rcfile with a new root
    """
    if filename is None:
        upipe.print_error("ERROR: input filename is None")
        return

    # Testing existence of filename
    if not os.path.isfile(filename) :
        upipe.print_error("ERROR: input filename {inputname} cannot be found. ".format(
                            inputname=filename))
        return

    # If it exists, open and read it
    old_rc = open(filename)
    lines = old_rc.readlines()

    # Create new file
    sfilename, extension = os.path.splitext(filename)
    new_filename = "{0}_{1}{2}".format(sfilename, subfolder, extension)
    new_rc = open(new_filename, 'w')

    # loop on lines
    for line in lines :
        sline = line.split()
        if sline[0] != "root":
            new_rc.write(line)
            continue
        if not os.path.isdir(sline[1]):
            upipe.print_warning("{} not an existing folder (from rcfile)".format(sline[1]))

        newline = line.replace(sline[1], joinpath(sline[1], subfolder))
        new_rc.write(newline)

    new_rc.close()
    old_rc.close()
    return new_filename

####################################################
# Defining classes to get samples and objects
####################################################
class MusePipeTarget(object):
    def __init__(self, subfolder='P100', list_pointings=None):
        self.subfolder = subfolder
        self.list_pointings = list_pointings

class MusePipeSample(object):
    def __init__(self, TargetDic, rc_filename=None, cal_filename=None, **kwargs) :
        """Using a given dictionary to initialise the sample
        That dictionary should include the names of the targets
        as keys and the subfolder plus pointings to consider

        Input
        -----
        TargetDic: dic
            Dictionary of targets. Keys are target names.
            Values for each target name should be a list of 2 parameters.
                - The first one is the name of the subfolder (e.g. 'P101')
                - The second one is the list of pointings, itself a dictionary
                  with a 0 or 1 for each pointing number depending on whether
                  this should be included in the reduction or not.
                  Results can be seen in self.dic_targets dictionary.
        rc_filename: str
            Default to None
        cal_filename: str
            Default to None

        PHANGS: bool
            Default to False. If True, will use default configuration dictionary
            from config_pipe.
        """
        self.sample = TargetDic
        self.targetnames = list(TargetDic.keys())
        self.rc_filename = rc_filename
        self.cal_filename = cal_filename

        self.__phangs = kwargs.pop("PHANGS", False)
        self._init_targets()

    def _init_targets(self):
        """Initialise the targets using the dictionary
        Returning self.dic_targets with the pointings to consider
        """
        self.dic_targets = {}
        for target in self.targetnames:
            subfolder = self.sample[target][0]
            lpoints = self.sample[target][1]
            list_pointings = []
            for lp in lpoints.keys():
                if lpoints[lp] == 1:
                    list_pointings.append(lp)
            self.dic_targets[target] = MusePipeTarget(subfolder=subfolder, 
                                                      list_pointings=list_pointings)

    def reduce_all_targets(self):
        """Reduce all targets already initialised
        """
        for target in self.dic_targets.keys():
            upipe.print_info("Starting the reduction of target {name}".format(
                                target))
            self.reduce_target(targetname=target)

    def reduce_target(self, targetname=None, list_pointings=None, **kwargs):
        """Reduce one target for a list of pointings

        Input
        -----
        targetname: str
            Name of the target
        list_pointings: list
            Pointing numbers. Default is None (meaning all pointings
            indicated in the dictonary will be reduced)
        config_args: dic
            Dictionary including extra configuration parameters to pass
            to MusePipe. This allows to define a global configuration.
            If self.__phangs is set to True, this is overwritten with the default
            PHANGS configuration parameters as provided in config_pipe.py.
        """
        if targetname not in self.targetnames:
            upipe.print_error("ERROR: no Target named {name} in the defined sample".format(
                                name=targetname))
            return

        # General print out
        upipe.print_info("---- Starting the Data Reduction ----")
        # Galaxy name
        upipe.print_info("Initialising Target {name}".format(name=targetname))

        # Info of the pointings and extracting the observing run for each pointing
        target_pointings = self.dic_targets[targetname].list_pointings

        # if no list_pointings we just do them all
        if list_pointings == None:
            list_pointings = target_pointings
        else:
            if any([_ not in target_pointings for _ in list_pointings]) :
                upipe.print_error("ERROR: no pointing {0} for the given target".format(
                                    list_pointings))
                return

        # Reading configuration filenames
        rc_filename = kwargs.pop("rc_filename", self.rc_filename)
        cal_filename = kwargs.pop("cal_filename", self.cal_filename)
        if rc_filename is None or cal_filename is None:
            upipe.print_error("rc_filename and/or cal_filename is None. Please define both.")
            return

        rc_filename = update_rcfile(rc_filename, 
                                    self.dic_targets[targetname].subfolder)
        cal_filename = update_rcfile(cal_filename, 
                                     self.dic_targets[targetname].subfolder)

        def_log_filename = "{0}_{1}.log".format(targetname, version_pack)
        log_filename = kwargs.pop("log_filename", def_log_filename)

        # Reading extra arguments from config dictionary
        if self.__phangs:
            config_args = PHANGS_config
        else:
            config_args = kwargs.pop("config_args", None)

        # Over-writing the arguments in kwargs from config dictionary
        if config_args is not None:
            for attr in config_args.keys():
                kwargs[attr] = config_args[attr]

        # extracting the kwargs
        list_kwargs = ', '.join(['{}={!r}'.format(k, v) for k, v in kwargs.items()])
#        list_kwargs = ""
#        for k in kwargs.keys():
#            list_kwargs += ", {0}={1}".format(k, kwargs[k])

        # Loop on the pointings
        self.pipelines = []
        self.history = []
        for pointing in list_pointings:
            # Setting up the names of the output files
            python_command = ("mypipe = musepipe.MusePipe(targetname='{0}', "
                              "pointing={1}, rc_filename='{2}', "
                              "cal_filename='{3}', log_filename='{4}', "
                              "{5})".format(targetname, pointing, rc_filename,
                                  cal_filename, log_filename, list_kwargs))

            upipe.print_info("====== START - POINTING {0:2d} ======".format(pointing))
            upipe.print_info(python_command)
            upipe.print_info("====== END   - POINTING {0:2d} ======".format(pointing))
            self.history.append(python_command)
            mypipe = MusePipe(targetname=targetname, pointing=pointing, 
                              rc_filename=rc_filename, cal_filename=cal_filename,
                              log_filename=log_filename, **kwargs)

            self.pipelines.append(mypipe)
            mypipe.run_all_phangs_recipes()

    def combine_all_targets(self):
        pass

    def combine_target(self):
        pass
