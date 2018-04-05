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

def create_time_name() :
    """Create a time-link name for file saving purposes

    Return: a string including the time, hence a priori unique
    """
    return str(time.time())

class PipeRecipes(object) :
    """PipeRecipes class containing all the esorex recipes for MUSE data reduction
    """
    def __init__(self, nifu=-1, first_cpu=0, ncpu=24, list_cpu=[], nocache=True, likwid=default_likwid,
            fakemode=True, merge=True) :
        """Initialisation of PipeRecipes
        """

        self.fakemode = fakemode
        if self.verbose :
            if fakemode : print("WARNING: running in FAKE mode")
            else : print("WARNING: running actual recipes")

        # Addressing CPU by number (cpu0=start, cpu1=end)
        self.first_cpu = first_cpu
        self.ncpu = ncpu

        if nocache : nocache = "nocache"
        else : nocache = ""
        self.nocache = nocache
        self.likwid = likwid
        self.nifu = nifu
        self._set_cpu(first_cpu, ncpu, list_cpu)
        self.merge = merge

    def esorex(self) :
        return "{likwid}{list_cpu} {nocache} exorex".format(likwid=self.likwid, 
                list_cpu=self.list_cpu, nocache=self.nocache)

    def get_merge(self) :
        if self.merge : 
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
            print("LIST_CPU: {0}".format(list_cpu))

    def run_oscommand(self, text, log=True) :
        """Running an os.system shell command
        Fake mode will just spit out the command but not actually do it.
        """
        if self.verbose : 
            print(text)
    
        if log :
            fout = open(self.logile, 'a')
            time_name = create_time_name()
            fout.write("At : " + timename + "\n")
            fout.write(text + "\n")
            fout.close()

        if not fakemode :
            os.system(command)

    def recipe_bias(self, sof, tpl):
        """Running the esorex muse_bias recipe
        """
        # Runing the recipe
        self.run_oscommand('{esorex} muse_bias --nifu={nifu} {merge} {sof}'.format(exorex=self.esorex(), 
            nifu=self.nifu, merge=self.get_merge(), sof=sof))
        attr = self.dic_attr_master['BIAS']
        # Moving the MASTER BIAS
        self.run_oscommand('{nocache} cp {name}.fits {name}_{tpl}.fits'.format(nocache=self.nocache, 
            name=getattr(self.masterfiles, attr), times=tpl))

    def recipe_flat(self, sof):
        """Running the esorex muse_flat recipe
        """
        os.system('{esorex} muse_flat --nifu={nifu} --merge={merge} {sof}'.format(exorex=self.esorex(), 
            nifu=self.nifu, merge=self.get_merge(), sof=sof))
    
    def recipe_wave(self, sof):
        """Running the esorex muse_wavecal recipe
        """
        os.system('{esorex} muse_wavecal --nifu={nifu} --merge={merge} {sof}'.format(exorex=self.esorex(), 
            nifu=self.nifu, merge=self.get_merge(), sof=sof))
    
    def recipe_lsf(self, sof):
        """Running the esorex muse_lsf recipe
        """
        os.system('{esorex} muse_lsf --nifu={nifu} --merge={merge} {sof}'.format(exorex=self.esorex(), 
            nifu=self.nifu, merge=self.get_merge(), sof=sof))
    
    def recipe_twilight(self, sof):
        """Running the esorex muse_twilight recipe
        """
        os.system('{esorex} muse_twilight sof'.format(exorex=self.esorex(), sof=sof))

    def recipe_std(self, sof):
        """Running the esorex muse_stc recipe
        """
        os.system("{esorex} muse_scibasic --nifu={nifu} --saveimage=FALSE "
            "--merge={merge} {sof}".format(exorex=self.esorex(), nifu=self.nifu, 
                merge=self.get_merge, sof=sof))
        create_sof_standard('std.sof')
        os.system('{esorex} muse_standard std.sof'.format(esorex=self.esorex()))
    
    def recipe_scibasic(self, sof):
        """Running the esorex muse_scibasic recipe
        """
        os.system("{esorex} muse_scibasic --nifu={nifu} "
                "--saveimage=FALSE --merge={merge} {sof}".format(esorex=self.esorex(), 
                    nifu=self.nifu, merge=self.get_merge(), sof=sof))
    
    def recipe_scipost(self, sof, save='cube', filter_list='white', skymethod='model', pixfrac=0.8, darcheck='none', skymodel_frac=0.05, astrometry='TRUE'):
        """Running the esorex muse_scipost recipe
        """
        os.system("{esorex} muse_scipost --astrometry={astro} --save={save} "
                "--pixfrac={pixfrac}  --filter={filt} --skymethod={skym} "
                "--darcheck={darkcheck} --skymodel_frac={model:02f} "
                "{sof}".format(esorex=self.esorex(), astro=astrometry, save=save, 
                    pixfrac=pixfrac, filt=filter_list, sky=skymethod, 
                    darkcheck=darcheck, model=skymodel_frac, sof=sof))
    
    def recipe_align(self, sof, srcmin=1, srcmax=10):
        """Running the muse_exp_align recipe
        """
        os.system("{esorex} muse_exp_align --srcmin={srcmin} "
                "--srcmax={srcmax} {sof}".format(exorex=self.esorex(), 
                    srcmin=srcmin, srcmax=srcmax, sof=sof))
    
    def recipe_cube(self, sof, save='cube', pixfrac=0.8, format_out='Cube', filter_FOV='SDSS_g,SDSS_r,SDSS_i'):
        """Running the muse_exp_combine recipe
        """
        os.system("{esorex} muse_exp_combine --save={save} --pixfrac={pixfrac:0.2f} "
        "--format={form} --filter={filt} {sof}".format(esorex=self.esorex(), save=save, 
            pixfrac=pixfrac, form=format_out, filt=filter_FOV, sof=sof))


