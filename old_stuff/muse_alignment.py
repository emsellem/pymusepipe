# hellp Rebecca master`
import astropy.io.fits as pyfits
import cpl
import math
import numpy
from astroquery.ned import Ned

from drizzlepac import tweakreg
#from drizzlepac import astrodrizzle
#from drizzlepac import tweakback

import os

# MdB: we're not running this as an executable.
# MdB: we're making another class of this.

class MuseAlign():
    """This contains the modules to align muse cubes with each other
    ma=MuseAlign(refcubedir='', cubedirs=[''], pixtabledirs=[''], objectname="NGC7162")
    """
    
    def __init__(self, refcubedir='', cubedirs=[''], pixtabledirs=[''], objectname=None, line=6562.8, linewidth=10., verbose=1, method='tweakreg',apply=True,fitgeometry='shift',subtract_continuum=False):
        self.refcubedir=refcubedir
        self.cubedirs=cubedirs
        self.pixtabledirs=pixtabledirs
        self.scratch='/scratch/denbrokm/reduction/tmp/'
        self.fov_template_pipeline='/scratch/denbrokm/reduction/calibration_files/current/IMAGE_FOV.fits'
        self.fov_template_tweakreg='/scratch/denbrokm/reduction/calibration_files/current/narrowband_hst.fits'
        # MdB: we need a template, since we're not going to individually
        # MdB: define every Halpha filter in the calibration filter list for
        # MdB: every galaxy.


        # MdB: method='tweakreg' or 'muse_exp_align'. It is my impression that
        # MdB: muse_exp_align does not work and produces bunk. I'd strongly
        # MdB: advice against using it.  
        self.method=method
        self.subtract_continuum=subtract_continuum
        self.objectname=objectname
        # MdB: we need this for the redshift info
    
        self.verbose=verbose
        
        self.z=0.0
        # MdB: in case of tweaking

        # MdB: the wavelength of the line on which the image is made
        # MdB: and its width [in Angstrom]
        self.line=line
        self.linewidth=linewidth
        self.apply=apply
        self.fitgeometry=fitgeometry

    def run(self):
        # MdB: Get the redshift from NED
        self.determine_redshift()

        # MdB: Write the narrowband images in the original cube dir
        self.create_narrowbandimage(self.refcubedir+"DATACUBE_FINAL.fits",self.refcubedir+"narrowband.fits")
        for dir in self.cubedirs:
            self.create_narrowbandimage(dir+"DATACUBE_FINAL.fits",dir+"narrowband.fits")

        # MdB: Use the tweakreg/pipeline module to determine the offsets
        # MdB: Put the offsets in the cube dir
        for dir in self.cubedirs:
            self.run_alignment(self.refcubedir+"narrowband.fits",dir+"narrowband.fits",dir)

        # MdB: to be sure, make a backup of the pixel tables, but only if we apply corrections
        if self.apply==True:
            for dir in self.pixtabledirs:
                self.backup_pixeltables(dir, dir+'backup/')

        
        
        # MdB: now make the correction
        for i in range(len(self.pixtabledirs)):
            if self.method=='muse_exp_align':
                referencefile=self.cubedirs[i]+"OFFSET_LIST.fits"
            if self.method=='tweakreg':
                referencefile=self.cubedirs[i]+"shift.txt"
            origcube=self.cubedirs[i]+"DATACUBE_FINAL.fits"
            dra,ddec,drot=self.read_alignment_from_file(referencefile, origcube)
            
            pixtabledir=self.pixtabledirs[i]
            if self.apply==True:
                self.apply_alignment(pixtabledir,dra,ddec,rot=drot)
            else:
                print "Measured offsets of dra=%f ddec=%f"%(dra,ddec)
            
    # MdB: We try to determine the redshift from NED
    # MdB: If that fails we use z=0.
    # MdB: We assume that the LSR correction is small?
    def determine_redshift(self):
        if self.objectname!=None:
            try: 
                table = Ned.query_object(self.objectname)
                self.z = table['Redshift'][0]
                
            except Exception as e:
                print "Could not determine redshift: ", e
                self.z = 0.0

        if self.verbose:
            print "Using redshift z = ",self.z

    # MdB: select those few frames around Halpha (or other line)
    # MdB: to have point sources
    # MdB: axis 3 is the spectral axis

    def create_narrowbandimage(self, infile, outfile):
        if self.method=='muse_exp_align':
            self.create_narrowbandimage_pipeline(infile, outfile)
        if self.method=='tweakreg':
            self.create_narrowbandimage_tweakreg(infile, outfile)
    
    def create_narrowbandimage_pipeline(self, infile, outfile):
        if self.verbose:
            print "infile: ",infile
            print "outfile: ",outfile
        hdu=pyfits.open(infile)
        cd3_3=hdu[1].header['CD3_3']
        crval3=hdu[1].header['CRVAL3']
        crpix3=hdu[1].header['CRPIX3']
        wav_min=(1.+self.z)*self.line - self.linewidth
        wav_max=(1.+self.z)*self.line + self.linewidth
        
        # MdB: int((wav_min-crval3)/cd3_3 + crpix3 -1.0 + 0.5)
        # MdB: the + 0.5 is for rounding to the closest int
        # MdB: the -1.0 is for converting pixel to array index
        arr_min=int((wav_min-crval3)/cd3_3 + crpix3 -0.5) 
        arr_max=int((wav_max-crval3)/cd3_3 + crpix3 -0.5) 

        # MdB: we sum the image slices in the cube to create
        # MdB: a point source image.
        # MdB: this should do the trick, but I haven't tested it
        image=hdu[1].data[arr_min:arr_max,:,:].sum(axis=0)

        out_hdu=pyfits.open(self.fov_template_pipeline)
        
        for i in range(2):
            for key in out_hdu[i].header.keys():
                if (key != 'COMMENT'):
                    try:
                        out_hdu[i].header[key]=hdu[i].header[key]
                    except KeyError:
                        #print key
                        try:
                            out_hdu[i].header[key]=hdu[1].header[key]
                        except KeyError:
                            print "No final key for ",key
        
        # MdB: we use the same cube skeleton to write the fits file
        # MdB: we need to change
        # MdB: the header to the following format though.

        
        out_hdu[1].data=image
        out_hdu[1].verify('fix')
        
        out_hdu[2].data=numpy.array(image*0,numpy.int32)
        out_hdu[2].verify('fix')
        out_hdu[3].verify('fix')
        out_hdu.writeto(outfile,clobber=True)
     
    def create_narrowbandimage_tweakreg(self, infile, outfile):
        if self.verbose:
            print "infile: ",infile
            print "outfile: ",outfile
        hdu=pyfits.open(infile)
        cd3_3=hdu[1].header['CD3_3']
        crval3=hdu[1].header['CRVAL3']
        crpix3=hdu[1].header['CRPIX3']
        wav_min=(1.+self.z)*self.line - self.linewidth
        wav_max=(1.+self.z)*self.line + self.linewidth
        
        # MdB: int((wav_min-crval3)/cd3_3 + crpix3 -1.0 + 0.5)
        # MdB: the + 0.5 is for rounding to the closest int
        # MdB: the -1.0 is for converting pixel to array index
        arr_min=int((wav_min-crval3)/cd3_3 + crpix3 -0.5) 
        arr_max=int((wav_max-crval3)/cd3_3 + crpix3 -0.5) 

        # MdB: we sum the image slices in the cube to create
        # MdB: a point source image.
        # MdB: this should do the trick, but I haven't tested it
        image=hdu[1].data[arr_min:arr_max,:,:].sum(axis=0)

        
        if (self.subtract_continuum):
            nframes=arr_max-arr_min
            aver_frame=nframes*0.5*(hdu[1].data[arr_min-1,:,:]+hdu[1].data[arr_max,:,:])
            image=image-aver_frame
            med=numpy.nanpercentile(image,99.)
            print "median: ",med
            q=numpy.where(image < med)
            image[q]=0.
        
        # MdB: we fix the image for NaNs
        q=numpy.where(numpy.isnan(image))
        image[q]=0.
        #q=numpy.where(image == 0.)
        #image[q]=1000.#numpy.min(image)

        out_hdu=pyfits.open(self.fov_template_tweakreg)
        keys=['CD1_1','CD2_1','CD1_2','CD2_2', 'CRVAL1','CRVAL2','CRPIX1','CRPIX2']

        for key in keys:
            out_hdu[0].header[key]=hdu[1].header[key]

        # MdB: not sure if this necessary, but just to be sure
        out_hdu[0].header['ORIENTAT']=0.
        out_hdu[0].header['RA_APER']=hdu[1].header['CRVAL1']
        out_hdu[0].header['DEC_APER']=hdu[1].header['CRVAL2']

        # MdB: we use the same cube skeleton to write the fits file
        # MdB: we need to change
        # MdB: the header to the following format though.
        out_hdu[0].data=image
        out_hdu[0].verify('fix')
        out_hdu.writeto(outfile,clobber=True)

    def run_alignment(self, reffile, tweakfile, workdir):         
        if self.method=='muse_exp_align':
            self.run_pipeline_alignment( reffile, tweakfile, workdir)
        if self.method=='tweakreg':
            self.run_tweakreg_alignment( reffile, tweakfile, workdir)
    

        
    def run_tweakreg_alignment(self, reffile, tweakfile, workdir):
        cwd=os.getcwd()
        # MdB: do the tweakreg in a temp directory
        os.chdir(workdir)
        try:
            os.mkdir('shifts')
        except Exception as e:
            print e
        os.chdir('shifts')

        # MdB: not sure what should have been put in here
        imagefind_dict = {}
        fitgeometry=self.fitgeometry
        # MdB: now we finally run tweakreg
        if self.verbose:
            print "Running tweakreg on ",tweakfile
            print "and reference ",reffile
        #tweakreg.TweakReg(tweakfile,minobj=10,xoffset=0,yoffset=0, imagefindcfg=imagefind_dict, refimage=reffile, searchrad=10.,  shiftfile=True, outshifts='shift.txt',  see2dplot=False, residplot='No plot')
        tweakreg.TweakReg(tweakfile, refimage=reffile, searchrad=2.,  shiftfile=True, outshifts='shift.txt',  see2dplot=False, residplot='No plot', updatehdr=False, fitgeometry=fitgeometry,minobj=4,tolerance=3.0)#,tolerance=100,rfluxunits='cps')
        
        os.system('cp shift.txt ..')
        # MdB: move back to the original directory 
        os.chdir(cwd)

    def run_pipeline_alignment(self,reffile, tweakfile, workdir):
        cwd=os.getcwd()
        # MdB: do the tweakreg in a temp directory
        os.chdir(workdir)
        cpl.esorex.init()
        muse_align= cpl.Recipe('muse_exp_align')
        muse_align.output_dir=workdir
        muse_align.param.weight=False
        # MdB: these are the parameters set such that they worked
        # MdB: for NGC 7162. They might require tweaking
        muse_align.param.step=1.
        muse_align.param.fwhm=5.
        muse_align.param.srcmax=2000
        alignlist=[reffile,tweakfile]
        
        muse_align(reffile,tweakfile)
        os.chdir(cwd)

    def backup_pixeltables(self, pixeltabledir, backupdir):
        try:
            os.mkdir(backupdir)
            # MdB: if the dir is already there, a backup has already been made?
            os.system("cp  %sPIXTABLE_OBJECT*fits   %s "%(pixeltabledir,backupdir))
            # MdB: I know that os.system is deprecated
        except:
            pass
        
    def recover_orig_pixeltables(self, pixeltabledir, backupdir):
        try:
            os.system("cp  %sPIXTABLE_OBJECT*fits   %s "%(backupdir,pixeltabledir))
            # MdB: I know that os.system is deprecated
        except:
            pass

    def read_alignment_from_file(self,referencefile, origcube):
        if self.method=='muse_exp_align':
            hdu=pyfits.open(referencefile)
            dra=hdu[1].data[1][2]
            ddec=hdu[1].data[1][3]
            return dra, ddec, 0.0
        if self.method=='tweakreg':
            # MdB: we read the file with coordinates
            f=open(referencefile,'r')
            lines=f.readlines()

            # MdB: we remove all lines that start with '#'
            removelist=[]
            for line in lines:
                if line.startswith('#'):
                    removelist.append(line)
            for line in removelist:
                lines.remove(line)

            # MdB: and we assume all the info is in line 0.
            hdu=pyfits.open(origcube)
            cd1_1=hdu[1].header['CD1_1']
            cd1_2=hdu[1].header['CD1_2']
            cd2_1=hdu[1].header['CD2_1']
            cd2_2=hdu[1].header['CD2_2']
            crval1=hdu[1].header['CRVAL1']
            crval2=hdu[1].header['CRVAL2']
            
            line=lines[0]
            line=line.strip()
            fields=line.split()
            deltax=float(fields[1])
            deltay=float(fields[2])
            deltarot=float(fields[3])
            deltascale=float(fields[4])
            dra=(deltax*cd1_1+deltay*cd1_2)/math.cos(crval2*math.pi/180.)
            ddec=(deltay*cd2_2+deltax*cd2_1)
            return dra,ddec,deltarot
        
    def apply_alignment(self, pixeltabledir,dra,ddec,rot=0.0):
        #dra=self.dra
        #ddec=self.ddec
        
        for i in range(1,25):
            hdu=pyfits.open(pixeltabledir+'PIXTABLE_OBJECT_0001-%02i.fits'%i,mode='update')
            hdu[0].header['RA']=hdu[0].header['RA']-dra
            hdu[0].header['DEC']=hdu[0].header['DEC']-ddec
            if (rot != 0.):
                print "Correcting for difference in PA by: ",rot 
                hdu[0].header['HIERARCH ESO INS DROT POSANG']=hdu[0].header['HIERARCH ESO INS DROT POSANG'] - rot
                if hdu[0].header['HIERARCH ESO INS DROT POSANG'] < 0.:
                    hdu[0].header['HIERARCH ESO INS DROT POSANG'] = hdu[0].header['HIERARCH ESO INS DROT POSANG'] + 360.
            hdu.flush()
            
        

        
     
        
