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
import musepipe as mpipe

# Likwid command
default_likwid = "likwid-pin -c N:"

class PipeRecipes(object) :
    """PipeRecipes class containing all the esorex recipes for MUSE data reduction
    """
    def __init__(self, nifu=-1, first_cpu=0, ncpu=24, list_cpu=[], likwid=default_likwid,
            fakemode=True, domerge=True) :
        """Initialisation of PipeRecipes
        """
        self.fakemode = fakemode
        if self.verbose :
            if fakemode : mpipe.print_warning("WARNING: running in FAKE mode")
            else : mpipe.print_warning("WARNING: running actual recipes")

        # Addressing CPU by number (cpu0=start, cpu1=end)
        self.first_cpu = first_cpu
        self.ncpu = ncpu

        self.likwid = likwid
        self.nifu = nifu
        self._set_cpu(first_cpu, ncpu, list_cpu)
        self.domerge = domerge

    @property
    def esorex(self):
        return ("{likwid}{list_cpu} {nocache} esorex --output-dir={outputdir} {checksum}" 
                    " --log-dir={logdir}").format(likwid=self.likwid, 
                    list_cpu=self.list_cpu, nocache=self.nocache, outputdir=self.paths.products, 
                    checksum=self.checksum, logdir=self.paths.esorexlog)

    @property
    def checksum(self):
        if self.nochecksum:
            return "--no-checksum"
        else : 
            return ""
    @property
    def merge(self):
        if self.domerge: 
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
            mpipe.print_info("LIST_CPU: {0}".format(self.list_cpu))

    def run_oscommand(self, command, log=True) :
        """Running an os.system shell command
        Fake mode will just spit out the command but not actually do it.
        """
        if self.verbose : 
            print(command)
    
        if log :
            fout = open(self.logfile, 'a')
            text = "# At : " + mpipe.formatted_time()
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
        return joinpath(self.paths.products, name)

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
        self.run_oscommand("{nocache} cp {namein}.fits {nameout}_{tpl}.fits".format(nocache=self.nocache, 
            namein=self.joinprod(name_lsf), nameout=joinpath(dir_lsf, name_lsf), tpl=tpl))
    
    def recipe_twilight(self, sof, dir_twilight, name_twilight, name_products, tpl):
        """Running the esorex muse_twilight recipe
        """
        self.run_oscommand("{esorex} --log-file=twilight_{tpl}.log muse_twilight {sof}".format(esorex=self.esorex, 
            sof=sof, tpl=tpl))
        # Moving the TWILIGHT CUBE
        self.run_oscommand("{nocache} cp {namein}.fits {nameout}_{tpl}.fits".format(nocache=self.nocache, 
            namein=self.joinprod(name_twilight), nameout=joinpath(dir_twilight, name_twilight), tpl=tpl))
        [name_cube_skyflat, name_twilight_cube] = name_products
        self.run_oscommand('rm {name}*.fits'.format(name=self.joinprod(name_cube_skyflat)))
        self.run_oscommand('rm {name}*.fits'.format(name=self.joinprod(name_twilight_cube)))

    def recipe_std(self, sof, dir_std, name_std, tpl):
        """Running the esorex muse_stc recipe
        """
        [name_cube, name_flux, name_response, name_telluric] = name_std
        self.run_oscommand("{esorex} --log-file=std_{tpl}.log muse_standard --filter=white {sof}".format(esorex=self.esorex,
                sof=sof, tpl=tpl))

        self.run_oscommand('{nocache} mv {name_cubein}_0001.fits {name_cubeout}_{tpl}.fits'.format(nocache=self.nocache,
            name_cubein=self.joinprod(name_cube), name_cubeout=joinpath(dir_std, name_cube), tpl=tpl))
        self.run_oscommand('{nocache} mv {name_fluxin}_0001.fits {name_fluxout}_{tpl}.fits'.format(nocache=self.nocache,
            name_fluxin=self.joinprod(name_flux), name_fluxout=joinpath(dir_std, name_flux), tpl=tpl))
        self.run_oscommand('{nocache} mv {name_responsein}_0001.fits {name_responseout}_{tpl}.fits'.format(nocache=self.nocache,
            name_responsein=self.joinprod(name_response), name_responseout=joinpath(dir_std, name_response), tpl=tpl))
        self.run_oscommand('{nocache} mv {name_telluricin}_0001.fits {name_telluricout}_{tpl}.fits'.format(nocache=self.nocache,
            name_telluricin=self.joinprod(name_telluric), name_telluricout=joinpath(dir_std, name_telluric), tpl=tpl))

    def recipe_sky(self, sof, dir_sky, name_sky, tpl, fraction=0.8):
        """Running the esorex muse_stc recipe
        """
        [name_spec, name_pixtable] = name_sky
        self.run_oscommand("{esorex} --log-file=sky_{tpl}.log muse_create_sky --fraction={fraction} {sof}".format(esorex=self.esorex,
                sof=sof, fraction=fraction, tpl=tpl))

        self.run_oscommand('{nocache} cp {name_spec}_0001.fits {name_spec}_{tpl}.fits'.format(nocache=self.nocache,
            name_specin=self.joinprod(name_spec), name_specout=joinpath(dir_sky, name_spec), tpl=tpl))
        self.run_oscommand('{nocache} cp {name_pixtable}_0001.fits {name_pixtable}_{tpl}.fits'.format(nocache=self.nocache,
            name_pixtablein=self.joinprod(name_pixtable), name_pixtableout=joinpath(dir_sky, name_pixtable), tpl=tpl))

    def recipe_scibasic(self, sof, tpl, expotype, dir_products=None, name_products=None):
        """Running the esorex muse_scibasic recipe
        """
        self.run_oscommand("{esorex} --log-file=scibasic_expotype_{tpl}.log muse_scibasic --nifu={nifu} "
                "--saveimage=FALSE {merge} {sof}".format(esorex=self.esorex, 
                    nifu=self.nifu, merge=self.merge, sof=sof, tpl=tpl))

        if name_products is not None :
            for prod in name_products :
                # newprod = prod.replace("0001", tpl)
                self.run_oscommand('{nocache} mv {prod} {newprod}'.format(nocache=self.nocache,
                    prod=self.joinprod(prod), newprod=joinpath(dir_products, prod)))
    
    def recipe_scipost(self, sof, save='cube', filter_list='white', skymethod='model', pixfrac=0.8, darcheck='none', skymodel_frac=0.05, astrometry='TRUE'):
        """Running the esorex muse_scipost recipe
        """
        self.run_oscommand("{esorex} muse_scipost --astrometry={astro} --save={save} "
                "--pixfrac={pixfrac}  --filter={filt} --skymethod={skym} "
                "--darcheck={darkcheck} --skymodel_frac={model:02f} "
                "{sof}".format(esorex=self.esorex, astro=astrometry, save=save, 
                    pixfrac=pixfrac, filt=filter_list, sky=skymethod, 
                    darkcheck=darcheck, model=skymodel_frac, sof=sof))
    
    def recipe_align(self, sof, srcmin=1, srcmax=10):
        """Running the muse_exp_align recipe
        """
        os.system("{esorex} muse_exp_align --srcmin={srcmin} "
                "--srcmax={srcmax} {sof}".format(esorex=self.esorex, 
                    srcmin=srcmin, srcmax=srcmax, sof=sof))
    
    def recipe_cube(self, sof, save='cube', pixfrac=0.8, format_out='Cube', filter_FOV='SDSS_g,SDSS_r,SDSS_i'):
        """Running the muse_exp_combine recipe
        """
        os.system("{esorex} muse_exp_combine --save={save} --pixfrac={pixfrac:0.2f} "
        "--format={form} --filter={filt} {sof}".format(esorex=self.esorex, save=save, 
            pixfrac=pixfrac, form=format_out, filt=filter_FOV, sof=sof))


