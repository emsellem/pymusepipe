# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS recipe module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# This module has been largely inspired by work of
# Bern Husemann, Dimitri Gadotti, Kyriakos and Martina from the GTO MUSE MAD team
# and further rewritten by Mark van den Brok. 
# Thanks to all !

# Importing modules
import os
from os.path import join as joinpath

# pymusepipe modules
from pymusepipe import util_pipe as upipe

# Likwid command
default_likwid = "likwid-pin -c N:"

class PipeRecipes(object) :
    """PipeRecipes class containing all the esorex recipes for MUSE data reduction
    """
    def __init__(self, nifu=-1, first_cpu=0, ncpu=24, list_cpu=[], likwid=default_likwid,
            fakemode=True, domerge=True, nocache=True, nochecksum=True) :
        """Initialisation of PipeRecipes
        """
        # Fake mode
        self.fakemode = fakemode
        if self.verbose :
            if fakemode : upipe.print_warning("WARNING: running in FAKE mode")
            else : upipe.print_warning("WARNING: running actual recipes")

        if nocache : self.nocache = "nocache"
        else : self.nocache = ""

        # Addressing CPU by number (cpu0=start, cpu1=end)
        self.first_cpu = first_cpu
        self.ncpu = ncpu

        self.likwid = likwid
        self.nifu = nifu
        self._set_cpu(first_cpu, ncpu, list_cpu)
        self._domerge = domerge

        self.nochecksum = nochecksum

    @property
    def esorex(self):
        return ("{likwid}{list_cpu} {nocache} esorex --output-dir={outputdir} {checksum}" 
                    " --log-dir={logdir}").format(likwid=self.likwid, 
                    list_cpu=self.list_cpu, nocache=self.nocache, outputdir=self.paths.pipe_products, 
                    checksum=self.checksum, logdir=self.paths.esorex_log)

    @property
    def checksum(self):
        if self.nochecksum:
            return "--no-checksum"
        else : 
            return ""
    @property
    def merge(self):
        if self._domerge: 
            return "--merge"
        else : 
            return ""

    def _set_cpu(self, first_cpu=0, ncpu=24, list_cpu=None) :
        """Setting the cpu format for calling the esorex command
        """
        if (list_cpu is None) | (len(list_cpu) < 1):
            self.list_cpu = "{0}-{1}".format(first_cpu, first_cpu + ncpu - 1)
        else :
            self.list_cpu = "{0}".format(list_cpu[0])
            for i in range(1, len(list_cpu)) :
                self.list_cpu += ":{0}".format(list_cpu[i])
        if self.verbose:
            upipe.print_info("LIST_CPU: {0}".format(self.list_cpu))

    def run_oscommand(self, command, log=True) :
        """Running an os.system shell command
        Fake mode will just spit out the command but not actually do it.
        """
        if self.verbose : 
            print(command)
    
        if log :
            fout = open(self.logfile, 'a')
            text = "# At : " + upipe.formatted_time()
            if self.fakemode : 
                text += " FAKEMODE\n"
            else :
                text += "\n"
            fout.write(text)
            fout.write(command + "\n")
            fout.close()

        if not self.fakemode :
            os.system(command)

    def joinprod(self, name):
        return joinpath(self.paths.pipe_products, name)

    def recipe_bias(self, sof, dir_bias, name_bias, tpl):
        """Running the esorex muse_bias recipe
        """
        # Runing the recipe
        self.run_oscommand(("{esorex} --log-file=bias_{tpl}.log muse_bias " 
                "--nifu={nifu} {merge} {sof}").format(esorex=self.esorex, 
                    nifu=self.nifu, merge=self.merge, sof=sof, tpl=tpl))
        # Moving the MASTER BIAS
        self.run_oscommand("{nocache} mv {namein}.fits {nameout}_{tpl}.fits".format(nocache=self.nocache, 
            namein=self.joinprod(name_bias), nameout=joinpath(dir_bias, name_bias), tpl=tpl))

    def recipe_flat(self, sof, dir_flat, name_flat, dir_trace, name_trace, tpl):
        """Running the esorex muse_flat recipe
        """
        self.run_oscommand(("{esorex} --log-file=flat_{tpl}.log muse_flat " 
                "--nifu={nifu} {merge} {sof}").format(esorex=self.esorex, 
                    nifu=self.nifu, merge=self.merge, sof=sof, tpl=tpl))
        # Moving the MASTER FLAT and TRACE_TABLE
        self.run_oscommand("{nocache} mv {namein}.fits {nameout}_{tpl}.fits".format(nocache=self.nocache, 
            namein=self.joinprod(name_flat), nameout=joinpath(dir_flat, name_flat), tpl=tpl))
        self.run_oscommand("{nocache} mv {namein}.fits {nameout}_{tpl}.fits".format(nocache=self.nocache, 
            namein=self.joinprod(name_trace), nameout=joinpath(dir_trace, name_trace), tpl=tpl))

    def recipe_wave(self, sof, dir_wave, name_wave, tpl):
        """Running the esorex muse_wavecal recipe
        """
        self.run_oscommand(("{esorex} --log-file=wave_{tpl}.log muse_wavecal --nifu={nifu} "
                "--resample --residuals {merge} {sof}").format(esorex=self.esorex, 
                    nifu=self.nifu, merge=self.merge, sof=sof, tpl=tpl))
        # Moving the MASTER WAVE
        self.run_oscommand("{nocache} mv {namein}.fits {nameout}_{tpl}.fits".format(nocache=self.nocache, 
            namein=self.joinprod(name_wave), nameout=joinpath(dir_wave, name_wave), tpl=tpl))
    
    def recipe_lsf(self, sof, dir_lsf, name_lsf, tpl):
        """Running the esorex muse_lsf recipe
        """
        self.run_oscommand("{esorex} --log-file=wave_{tpl}.log muse_lsf --nifu={nifu} {merge} {sof}".format(esorex=self.esorex,
            nifu=self.nifu, merge=self.merge, sof=sof, tpl=tpl))
        # Moving the MASTER LST PROFILE
        self.run_oscommand("{nocache} mv {namein}.fits {nameout}_{tpl}.fits".format(nocache=self.nocache, 
            namein=self.joinprod(name_lsf), nameout=joinpath(dir_lsf, name_lsf), tpl=tpl))
    
    def recipe_twilight(self, sof, dir_twilight, name_twilight, tpl):
        """Running the esorex muse_twilight recipe
        """
        self.run_oscommand("{esorex} --log-file=twilight_{tpl}.log muse_twilight {sof}".format(esorex=self.esorex, 
            sof=sof, tpl=tpl))
        # Moving the TWILIGHT CUBE
        for name_prod in name_twilight:
            self.run_oscommand("{nocache} mv {namein}.fits {nameout}_{tpl}.fits".format(nocache=self.nocache, 
                namein=self.joinprod(name_prod), nameout=joinpath(dir_twilight, name_prod), tpl=tpl))

    def recipe_std(self, sof, dir_std, name_std, tpl):
        """Running the esorex muse_stc recipe
        """
        [name_cube, name_flux, name_response, name_telluric] = name_std
        self.run_oscommand("{esorex} --log-file=std_{tpl}.log muse_standard --filter=white {sof}".format(esorex=self.esorex,
                sof=sof, tpl=tpl))

        for name_prod in name_std:
            self.run_oscommand('{nocache} mv {name_prodin}_0001.fits {name_prodout}_{tpl}.fits'.format(nocache=self.nocache,
                name_prodin=self.joinprod(name_prod), name_prodout=joinpath(dir_std, name_prod), tpl=tpl))

    def recipe_sky(self, sof, dir_sky, name_sky, tpl, iexpo=1, fraction=0.8):
        """Running the esorex muse_stc recipe
        """
        self.run_oscommand("{esorex} --log-file=sky_{tpl}.log muse_create_sky --fraction={fraction} {sof}".format(esorex=self.esorex,
                sof=sof, fraction=fraction, tpl=tpl))

        for name_prod in name_sky:
            self.run_oscommand('{nocache} mv {name_prodin}.fits {name_prodout}_{iexpo:04d}_{tpl}.fits'.format(nocache=self.nocache,
                name_prodin=self.joinprod(name_prod), name_prodout=joinpath(dir_sky, name_prod), iexpo=iexpo, tpl=tpl))

    def recipe_scibasic(self, sof, tpl, expotype, dir_products=None, name_products=[], suffix=""):
        """Running the esorex muse_scibasic recipe
        """
        self.run_oscommand("{esorex} --log-file=scibasic_{expotype}_{tpl}.log muse_scibasic --nifu={nifu} "
                "--saveimage=FALSE {merge} {sof}".format(esorex=self.esorex, 
                    nifu=self.nifu, merge=self.merge, sof=sof, tpl=tpl, expotype=expotype))

        suffix_out = "{0}_{1}".format(suffix, tpl)
        for name_prod in name_products :
            self.run_oscommand('{nocache} mv {prod} {newprod}'.format(nocache=self.nocache,
                prod=self.joinprod("{0}_{1}".format(suffix, name_prod)), newprod=joinpath(dir_products, 
                    "{0}_{1}".format(suffix_out, name_prod))))
   
    def recipe_scipost(self, sof, tpl, expotype, dir_products=None, name_products=[], 
            suffix_products=[], suffix_finalnames=[], list_expo=[1], save='cube,skymodel', filter_list='white', skymethod='model',
            pixfrac=0.8, darcheck='none', skymodel_frac=0.05, astrometry='TRUE',
            lambdamin=4000., lambdamax=10000., suffix="", autocalib='none'):
        """Running the esorex muse_scipost recipe
        """
        self.run_oscommand("{esorex} --log-file=scipost_{expotype}_{tpl}.log muse_scipost  "
                "--astrometry={astro} --save={save} "
                "--pixfrac={pixfrac}  --filter={filt} --skymethod={skym} "
                "--darcheck={darcheck} --skymodel_frac={model:02f} "
                "--lambdamin={lmin} --lambdamax={lmax} --autocalib={autocalib}"
                "{sof}".format(esorex=self.esorex, astro=astrometry, save=save, 
                    pixfrac=pixfrac, filt=filter_list, skym=skymethod, 
                    darcheck=darcheck, model=skymodel_frac, lmin=lambdamin,
                    lmax=lambdamax, autocalib=autocalib, sof=sof, expotype=expotype, tpl=tpl))

        for name_prod, suff_prod, suff_final in zip(name_products, suffix_products, suffix_finalnames) :
            self.run_oscommand('{nocache} mv {name_imain}.fits {name_imaout}{suffix}_{tpl}.fits'.format(nocache=self.nocache,
                name_imain=self.joinprod(name_prod+suff_prod), name_imaout=joinpath(dir_products, name_prod+suff_final), tpl=tpl,
                suffix=suffix))
   
    def recipe_align(self, sof, dir_products, name_products, tpl, 
            suffix="", threshold=10.0, srcmin=3, srcmax=80, fwhm=5.0):
        """Running the muse_exp_align recipe
        """
        self.run_oscommand("{esorex} --log-file=exp_align_{tpl}.log muse_exp_align --srcmin={srcmin} "
                "--srcmax={srcmax} --threshold={threshold} --fwhm={fwhm} {sof}".format(esorex=self.esorex, 
                    srcmin=srcmin, srcmax=srcmax, threshold=threshold, fwhm=fwhm, sof=sof, tpl=tpl))
    
        for name_prod in name_products :
            self.run_oscommand('{nocache} mv {name_imain}.fits {name_imaout}{suffix}_{tpl}.fits'.format(nocache=self.nocache,
                name_imain=self.joinprod(name_prod), name_imaout=joinpath(dir_products, name_prod), tpl=tpl,
                suffix=suffix))

    def recipe_cube(self, sof, save='cube', pixfrac=0.8, format_out='Cube', filter_FOV='SDSS_g,SDSS_r,SDSS_i'):
        """Running the muse_exp_combine recipe
        """
        self.run_oscommand("{esorex}  --log-file=exp_combine_cube.log muse_exp_combine--save={save} --pixfrac={pixfrac:0.2f} "
        "--format={form} --filter={filt} {sof}".format(esorex=self.esorex, save=save, 
            pixfrac=pixfrac, form=format_out, filt=filter_FOV, sof=sof))


