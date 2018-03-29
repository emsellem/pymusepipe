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
        self.command_suffix = "{likwid}{list_cpu} {nocache}".format(likwid= self.likwid, self.list_cpu= self.list_cpu,
                nocache= nocache) 
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
        self.likwid = likwid
        self.nifu = nifu
        self.set_cpu(first_cpu, ncpu, list_cpu)

    def set_cpu(self, first_cpu=0, ncpu=24, list_cpu=None)) :
        if list_cpu is None :
            self.list_cpu = "{0}:{1}".format(first_cpu, first_cpu + ncpu - 1)
        else :
            self.list_cpu = "{0}".format(list_cpu[0])
            for i in range(1, len(list_cpu)) :
                self.list_cpu += ":{0}".format(list_cpu[i])
        if self.verbose:
            print("LIST_CPU: {0}".format(list_cpu))

    @prepare_recipe
    def run_bias(self, sof=bias.sof):
        """Running the esorex muse_bias recipe
        """
        # Runing the recipe
        os.system('{likwid}{list_cpu} {nocache} esorex muse_bias --nifu={nifu} {merge} {sof}'%(likwid=likwid,
                list_cpu=self.list_cpu, nocache=nocache, nifu=self.nifu, merge=self.merge, sof=sof))
        attr = self.dic_attr_master['BIAS']
        # Moving the MASTER BIAS
        os.system('{nocache} cp {name}.fits {name}_%s.fits'%(nocache=nocache, 
            name=getattr(self.filenames, attr), times[i]))

    @prepare_recipe
    def run_flat(self, sof):
        """Running the esorex muse_flat recipe
        """
        os.system('{likwid}{list_cpu} {nocache} esorex muse_flat --nifu=%d --merge=%s %s'%(likwid=likwid,
                list_cpu=self.list_cpu, nocache=nocache, nifu=self.nifu, merge=self.merge, sof=sof))
    
    @prepare_recipe
    def run_wave(self, sof):
        """Running the esorex muse_wavecal recipe
        """
        os.system('{likwid}{list_cpu} {nocache} esorex muse_wavecal --nifu=%d --merge=%s %s'%(likwid=likwid,
                list_cpu=self.list_cpu, nocache=nocache, nifu=self.nifu, merge=self.merge, sof=sof))
    
    @prepare_recipe
    def run_lsf(self, sof):
        """Running the esorex muse_lsf recipe
        """
        os.system('{likwid}{list_cpu} {nocache} esorex muse_lsf --nifu=%d --merge=%s %s'%(likwid=likwid,
                list_cpu=self.list_cpu, nocache=nocache, nifu=self.nifu, merge=self.merge, sof=sof))
    
    @prepare_recipe
    def run_twilight(self, sof):
        """Running the esorex muse_twilight recipe
        """
        os.system('{likwid}{list_cpu} {nocache} esorex muse_twilight %s'%(likwid=likwid,
                list_cpu=self.list_cpu, nocache=nocache, sof=sof))
    
    @prepare_recipe
    def run_std(self, sof):
        """Running the esorex muse_stc recipe
        """
        os.system('{likwid}{list_cpu} {nocache} esorex muse_scibasic --nifu=%d --saveimage=FALSE --merge=%s %s'%(cpu_process,cores-1,nifu,merge,sof))
        create_sof_standard('std.sof')
        os.system('{likwid}{list_cpu} {nocache} esorex muse_standard  std.sof'%(cpu_process,cores-1))
    
    @prepare_recipe
    def run_scibasic(self, sof):
        """Running the esorex muse_scibasic recipe
        """
        os.system('{likwid}{list_cpu} {nocache} esorex muse_scibasic --nifu=%d --saveimage=FALSE --merge=%s %s'%(cpu_process,cores-1,nifu,merge,sof))
    
    @prepare_recipe
    def run_scipost(self, sof, save='cube', filter_list='white', skymethod='model', darcheck='none', skymodel_frac=0.05, astrometry='TRUE'):
        """Running the esorex muse_scipost recipe
        """
        os.system('{likwid}{list_cpu} {nocache} esorex muse_scipost --astrometry=%s --save=%s --pixfrac=0.8  --filter=%s --skymethod=%s --darcheck=%s --skymodel_frac=%.02f %s'%(cpu_process,cores-1,astrometry,save,filter_list,skymethod,darcheck,skymodel_frac,sof))
    
    @prepare_recipe
    def run_align(self, sof, srcmin=1, srcmax=10):
        """Running the muse_exp_align recipe
        """
        os.system('{likwid}{list_cpu} {nocache} esorex muse_exp_align --srcmin=%d --srcmax=%d  %s'%(srcmin,srcmax,sof))
    
    def run_cube(sof,cores,save='cube',pixfrac=0.8,format='Cube',filter_FOV='SDSS_g,SDSS_r,SDSS_i'):
        os.system('{likwid}{list_cpu} {nocache} esorex muse_exp_combine --save=%s --pixfrac=%.1f --format=%s --filter=%s %s'%(cpu_process,cores-1,save,pixfrac,format,filter_FOV,sof))


