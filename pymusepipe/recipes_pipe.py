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
    def __init__(self, nifu=-1, first_cpu=0, ncpu=24, list_cpu=[], nocache=True, likwid=default_likwid,
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

        if nocache : nocache = "nocache"
        else : nocache = ""
        self.nocache = nocache
        self.likwid = likwid
        self.nifu = nifu
        self._set_cpu(first_cpu, ncpu, list_cpu)
        self.domerge = domerge

    @property
    def esorex(self):
        return "{likwid}{list_cpu} {nocache} esorex".format(likwid=self.likwid, 
                list_cpu=self.list_cpu, nocache=self.nocache)

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

    def recipe_bias(self, sof, name_master, tpl):
        """Running the esorex muse_bias recipe
        """
        # Runing the recipe
        self.run_oscommand("{esorex} muse_bias --nifu={nifu} {merge} {sof}".format(esorex=self.esorex,
            nifu=self.nifu, merge=self.merge, sof=sof))
        # Moving the MASTER BIAS
        self.run_oscommand("{nocache} mv {name}.fits {name}_{tpl}.fits".format(nocache=self.nocache, 
            name=name_master, tpl=tpl))

    def recipe_flat(self, sof, name_master, name_trace, tpl):
        """Running the esorex muse_flat recipe
        """
        self.run_oscommand("{esorex} muse_flat --nifu={nifu} {merge} {sof}".format(esorex=self.esorex,
            nifu=self.nifu, merge=self.merge, sof=sof))
        # Moving the MASTER FLAT and TRACE_TABLE
        self.run_oscommand("{nocache} mv {name}.fits {name}_{tpl}.fits".format(nocache=self.nocache, 
            name=name_master, tpl=tpl))
        self.run_oscommand("{nocache} mv {name}.fits {name}_{tpl}.fits".format(nocache=self.nocache, 
            name=name_trace, tpl=tpl))

    def recipe_wave(self, sof, name_master, tpl):
        """Running the esorex muse_wavecal recipe
        """
        self.run_oscommand("{esorex} muse_wavecal --nifu={nifu} --resample --residuals {merge} {sof}".format(esorex=self.esorex,
            nifu=self.nifu, merge=self.merge, sof=sof))
        # Moving the MASTER WAVE
        self.run_oscommand("{nocache} mv {name}.fits {name}_{tpl}.fits".format(nocache=self.nocache, 
            name=name_master, tpl=tpl))
    
    def recipe_lsf(self, sof, name_master, tpl):
        """Running the esorex muse_lsf recipe
        """
        self.run_oscommand("{esorex} muse_lsf --nifu={nifu} {merge} {sof}".format(esorex=self.esorex,
            nifu=self.nifu, merge=self.merge, sof=sof))
        # Moving the MASTER LST PROFILE
        self.run_oscommand("{nocache} cp {name}.fits {name}_{tpl}.fits".format(nocache=self.nocache, 
            name=name_master, tpl=tpl))
    
    def recipe_twilight(self, sof, name_master, dir_products, name_products, tpl):
        """Running the esorex muse_twilight recipe
        """
        self.run_oscommand("{esorex} muse_twilight {sof}".format(esorex=self.esorex, sof=sof))
        # Moving the TWILIGHT CUBE
        self.run_oscommand("{nocache} cp {name}.fits {name}_{tpl}.fits".format(nocache=self.nocache, 
            name=name_master, tpl=tpl))
        [name_cube_skyflat, name_twilight_cube] = name_products
        self.run_oscommand('rm {name}*.fits'.format(name=joinpath(dir_products, name_cube_skyflat)))
        self.run_oscommand('rm {name}*.fits'.format(name=joinpath(dir_products, name_twilight_cube)))

    def recipe_std(self, sof, dir_products, name_products, tpl):
        """Running the esorex muse_stc recipe
        """
        [name_cube, name_flux, name_response, name_telluric] = name_products
        self.run_oscommand("{esorex} muse_standard --filter=white {sof}".format(esorex=self.esorex,
                sof=sof))
        self.run_oscommand('{nocache} mv {name_cube}_0001.fits {name_cube}_{tpl}.fits'.format(nocache=self.nocache,
            name_cube=joinpath(dir_products, name_cube), tpl=tpl))
        self.run_oscommand('{nocache} mv {name_flux}_0001.fits {name_flux}_{tpl}.fits'.format(nocache=self.nocache,
            name_flux=joinpath(dir_products, name_flux), tpl=tpl))
        self.run_oscommand('{nocache} mv {name_response}_0001.fits {name_response}_{tpl}.fits'.format(nocache=self.nocache,
            name_response=joinpath(dir_products, name_response), tpl=tpl))
        self.run_oscommand('{nocache} mv {name_telluric}_0001.fits {name_telluric}_{tpl}.fits'.format(nocache=self.nocache,
            name_telluric=joinpath(dir_products, name_telluric), tpl=tpl))

    def recipe_sky(self, sof, dir_products, name_products, tpl, fraction=0.8):
        """Running the esorex muse_stc recipe
        """
        [name_spec, name_pixtable] = name_products
        self.run_oscommand("{esorex} muse_create_sky --fraction={fraction} {sof}".format(esorex=self.esorex,
                sof=sof, fraction=fraction))
        self.run_oscommand('{nocache} cp {name_spec}_0001.fits {name_spec}_{tpl}.fits'.format(nocache=self.nocache,
            name_spec=joinpath(dir_products, name_spec), tpl=tpl))
        self.run_oscommand('{nocache} cp {name_pixtable}_0001.fits {name_pixtable}_{tpl}.fits'.format(nocache=self.nocache,
            name_pixtable=joinpath(dir_products, name_pixtable), tpl=tpl))

    def recipe_scibasic(self, sof, dir_products=None, name_products=None, tpl=None):
        """Running the esorex muse_scibasic recipe
        """
        self.run_oscommand("{esorex} muse_scibasic --nifu={nifu} "
                "--saveimage=FALSE {merge} {sof}".format(esorex=self.esorex, 
                    nifu=self.nifu, merge=self.merge, sof=sof))

        if name_products is not None :
            for prod in name_products :
                newprod = prod.replace("0001", tpl)
                self.run_oscommand('{nocache} mv {prod} {newprod}'.format(nocache=self.nocache,
                    prod=joinpath(dir_products, prod), newprod=joinpath(dir_products, newprod)))

    
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


