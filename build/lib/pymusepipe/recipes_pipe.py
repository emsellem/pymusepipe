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

# Likwid command
default_likwid = "likwid-pin -c N:"

def prepare_recipe(func) :
    """Used as decorator for joining suffix command
    """
    def wrapper(*args, **kwargs) :
        self.esorex = "{likwid}{list_cpu} {nocache} exorex".format(likwid=self.likwid, 
                list_cpu=self.list_cpu, nocache=self.nocache) 
        sof = joinpath(self.paths.sof, sof)
        func(*args, **kwargs)

    return wrapper

class PipeRecipes(object) :
    def __init__(self, nifu=-1, first_cpu=0, ncpu=24, list_cpu=[], nocache=True, likwid=default_likwid) :
        # Addressing CPU by number (cpu0=start, cpu1=end)
        self.cpu0 = cpu0
        self.cpu1 = cpu1

        if nocache : nocache = "nocache"
        else : nocache = ""
        self.nocache = nocache
        self.likwid = likwid
        self.nifu = nifu
        self.set_cpu(first_cpu, ncpu, list_cpu)

    def set_cpu(self, first_cpu=0, ncpu=24, list_cpu=None) :
        if list_cpu is None :
            self.list_cpu = "{0}:{1}".format(first_cpu, first_cpu + ncpu - 1)
        else :
            self.list_cpu = "{0}".format(list_cpu[0])
            for i in range(1, len(list_cpu)) :
                self.list_cpu += ":{0}".format(list_cpu[i])
        if self.verbose:
            print("LIST_CPU: {0}".format(list_cpu))

    @prepare_recipe
    def run_bias(self, sof):
        """Running the esorex muse_bias recipe
        """
        # Runing the recipe
        os.system('{esorex} muse_bias --nifu={nifu} {merge} {sof}'.format(exorex=self.esorex, 
            nifu=self.nifu, merge=self.merge, sof=sof))
        attr = self.dic_attr_master['BIAS']
        # Moving the MASTER BIAS
        os.system('{nocache} cp {name}.fits {name}_{times}.fits'.format(nocache=self.nocache, 
            name=getattr(self.filenames, attr), times=times[i]))

    @prepare_recipe
    def run_flat(self, sof):
        """Running the esorex muse_flat recipe
        """
        os.system('{esorex} muse_flat --nifu={nifu} --merge={merge} {sof}'.format(exorex=self.esorex, 
            nifu=self.nifu, merge=self.merge, sof=sof))
    
    @prepare_recipe
    def run_wave(self, sof):
        """Running the esorex muse_wavecal recipe
        """
        os.system('{esorex} muse_wavecal --nifu={nifu} --merge={merge} {sof}'.format(exorex=self.esorex, 
            nifu=self.nifu, merge=self.merge, sof=sof))
    
    @prepare_recipe
    def run_lsf(self, sof):
        """Running the esorex muse_lsf recipe
        """
        os.system('{esorex} muse_lsf --nifu={nifu} --merge={merge} {sof}'.format(exorex=self.esorex, 
            nifu=self.nifu, merge=self.merge, sof=sof))
    
    @prepare_recipe
    def run_twilight(self, sof):
        """Running the esorex muse_twilight recipe
        """
        os.system('{esorex} muse_twilight sof'.format(exorex=self.esorex, sof=sof))
    
    @prepare_recipe
    def run_std(self, sof):
        """Running the esorex muse_stc recipe
        """
        os.system("{esorex} muse_scibasic --nifu={nifu} --saveimage=FALSE "
            "--merge={merge} {sof}".format(exorex=self.esorex, nifu=self.nifu, 
                merge=self.merge, sof=sof))
        create_sof_standard('std.sof')
        os.system('{esorex} muse_standard std.sof'.format(esorex=self.esorex))
    
    @prepare_recipe
    def run_scibasic(self, sof):
        """Running the esorex muse_scibasic recipe
        """
        os.system("{esorex} muse_scibasic --nifu={nifu} "
                "--saveimage=FALSE --merge={merge} {sof}".format(esorex=self.esorex, 
                    nifu=self.nifu, merge=self.merge, sof=sof))
    
    @prepare_recipe
    def run_scipost(self, sof, save='cube', filter_list='white', skymethod='model', pixfrac=0.8, darcheck='none', skymodel_frac=0.05, astrometry='TRUE'):
        """Running the esorex muse_scipost recipe
        """
        os.system("{esorex} muse_scipost --astrometry={astro} --save={save} "
                "--pixfrac={pixfrac}  --filter={filt} --skymethod={skym} "
                "--darcheck={darkcheck} --skymodel_frac={model:02f} "
                "{sof}".format(esorex=self.esorex, astro=astrometry, save=save, 
                    pixfrac=pixfrac, filt=filter_list, sky=skymethod, 
                    darkcheck=darcheck, model=skymodel_frac, sof=sof))
    
    @prepare_recipe
    def run_align(self, sof, srcmin=1, srcmax=10):
        """Running the muse_exp_align recipe
        """
        os.system("{esorex} muse_exp_align --srcmin={srcmin} "
                "--srcmax={srcmax} {sof}".format(exorex=self.esorex, 
                    srcmin=srcmin, srcmax=srcmax, sof=sof))
    
    @prepare_recipe
    def run_cube(self, sof, save='cube', pixfrac=0.8, format_out='Cube', filter_FOV='SDSS_g,SDSS_r,SDSS_i'):
        """Running the muse_exp_combine recipe
        """
        os.system("{esorex} muse_exp_combine --save={save} --pixfrac={pixfrac:0.2f} "
        "--format={form} --filter={filt} {sof}".format(esorex=self.esorex, save=save, 
            pixfrac=pixfrac, form=format_out, filt=filter_FOV, sof=sof))


