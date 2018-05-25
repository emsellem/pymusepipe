# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MUSE-PHANGS preparation recipe module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2017, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# Importing modules
import os
from os.path import join as joinpath

# Numpy
import numpy as np

# pymusepipe modules
import musepipe as mpipe

# =======================================================
# List of recipes
# =======================================================
list_recipes = ['bias', 'flat', 'wave', 'lsf', 
        'twilight', 'scibasic_all', 'std', 'sky']
        

dic_files_products = {
        'STD': ['DATACUBE_STD', 'STD_FLUXES', 
            'STD_RESPONSE', 'STD_TELLURIC'],
        'TWILIGHT': ['DATACUBE_SKYFLAT', 'TWILIGHT_CUBE'],
        'SKY': ['SKY_SPECTRUM', 'PIXTABLE_REDUCED']
        }
# =======================================================
# Few useful functions
# =======================================================
def add_listpath(suffix, paths) :
    """Add a suffix to a list of path
    and normalise them
    """
    newlist = []
    for mypath in paths:
        newlist.append(mpipe.normpath(joinpath(suffix, mypath)))
    return newlist

def norm_listpath(paths) :
    """Normalise the path for a list of paths
    """
    newlist = []
    for mypath in paths:
        newlist.append(mpipe.normpath(mypath))
    return newlist

###################################################################
# Class for preparing the launch of recipes
###################################################################
class PipePrep(object) :
    """PipePre class prepare the SOF files and launch the recipes
    """
    def __init__(self, nocache=True):
        """Initialisation of PipePrep
        """
        self.list_recipes = list_recipes
        if nocache : nocache = "nocache"
        else : nocache = ""
        self.nocache = nocache

    def run_all_recipes(self, fraction=0.8):
        """Running all recipes in one shot
        """
        self.run_bias()
        self.run_flat()
        self.run_wave()
        self.run_lsf()
        self.run_twilight()
        self.run_scibasic_all()
        self.run_std()
        self.run_sky(fraction=fraction)

    def run_bias(self, sof_filename='bias', tpl="ALL", fits_tablename=None):
        """Reducing the Bias files and creating a Master Bias
        Will run the esorex muse_bias command on all Biases

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='BIAS', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                mpipe.print_warning("No BIAS recovered from the astropy Table - Aborting")
                return
        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the BIASES including
        # the list of files to be processed for one MASTER BIAS
        # Adding the bad pixel table
        self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['BIAS'] = add_listpath(self.paths.rawfiles,
                    gtable['filename'].data.astype(np.object))
            # extract the tpl (string)
            tpl = gtable['tpls'][0]
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Bias
            name_bias = self._get_suffix_expo('BIAS')
            dir_bias = self._get_fullpath_expo('BIAS')
            # Run the recipe
            self.recipe_bias(self.current_sof, dir_bias, name_bias, tpl)

        # Write the MASTER BIAS Table and save it
        self.save_master_table('BIAS', tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_flat(self, sof_filename='flat', tpl="ALL", fits_tablename=None):
        """Reducing the Flat files and creating a Master Flat
        Will run the esorex muse_flat command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='FLAT,LAMP', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                mpipe.print_warning("No FLAT recovered from the astropy Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the FLATs including
        # the list of files to be processed for one MASTER Flat
        # Adding the bad pixel table
        self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['FLAT'] = add_listpath(self.paths.rawfiles,
                    gtable['filename'].data.astype(np.object))
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self.get_tpl_meanmjd(gtable)
            # Adding the best tpc MASTER_BIAS
            self.add_tplmaster_to_sofdict(mean_mjd, 'BIAS')
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Flat and Trace Table
            dir_flat = self._get_fullpath_expo('FLAT')
            name_flat =  self._get_suffix_expo('FLAT')

            dir_trace = self._get_fullpath_expo('TRACE')
            name_tracetable = self._get_suffix_expo('TRACE')
            # Run the recipe
            self.recipe_flat(self.current_sof, dir_flat, name_flat, dir_trace, name_tracetable, tpl)

        # Write the MASTER FLAT Table and save it
        self.save_master_table('FLAT', tpl_gtable, fits_tablename)
        self.save_master_table('TRACE', tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_wave(self, sof_filename='wave', tpl="ALL", fits_tablename=None):
        """Reducing the WAVE-CAL files and creating the Master Wave
        Will run the esorex muse_wave command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='WAVE', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                mpipe.print_warning("No WAVE recovered from the astropy file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the FLATs including
        # the list of files to be processed for one WAVECAL
        # Adding Badpix table and Line Catalog
        self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        self.add_calib_to_sofdict("LINE_CATALOG")
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['ARC'] = add_listpath(self.paths.rawfiles,
                    gtable['filename'].data.astype(np.object))
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self.get_tpl_meanmjd(gtable)
            # Finding the best tpl for BIAS + TRACE
            self.add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'TRACE'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Wave
            dir_wave = self._get_fullpath_expo('WAVE')
            name_wave = self._get_suffix_expo('WAVE')
            # Run the recipe
            self.recipe_wave(self.current_sof, dir_wave, name_wave, tpl)

        # Write the MASTER WAVE Table and save it
        self.save_master_table('WAVE', tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_lsf(self, sof_filename='lsf', tpl="ALL", fits_tablename=None):
        """Reducing the LSF files and creating the LSF PROFILE
        Will run the esorex muse_lsf command on all Flats

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='WAVE', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                mpipe.print_warning("No WAVE recovered from the astropy file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        # Adding Badpix table and Line Catalog
        self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        self.add_calib_to_sofdict("LINE_CATALOG")
        for gtable in tpl_gtable.groups:
            # Provide the list of files to the dictionary
            self._sofdict['ARC'] = add_listpath(self._get_fullpath_expo('WAVE'),
                    gtable['filename'].data.astype(np.object))
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self.get_tpl_meanmjd(gtable)
            # Finding the best tpl for BIAS, TRACE, WAVE
            self.add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'TRACE', 'WAVE'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Name of final Master Wave
            dir_lsf = self._get_fullpath_expo('LSF')
            name_lsf = self._get_suffix_expo('LSF')
            # Run the recipe
            self.recipe_lsf(self.current_sof, dir_lsf, name_lsf, tpl)

        # Write the MASTER LSF PROFILE Table and save it
        self.save_master_table('LSF', tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_twilight(self, sof_filename='twilight', tpl="ALL", fits_tablename=None):
        """Reducing the  files and creating the TWILIGHT CUBE.
        Will run the esorex muse_twilight command on all TWILIGHT

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype='FLAT,SKY', tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                mpipe.print_warning("No TWILIGHT recovered from the astropy file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
        self.add_calib_to_sofdict("VIGNETTING_MASK")
        for gtable in tpl_gtable.groups:
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self.get_tpl_meanmjd(gtable)
            self.add_geometry_to_sofdict(tpl)
            # Provide the list of files to the dictionary
            self._sofdict['TWILIGHT'] = add_listpath(self._get_fullpath_expo('TWILIGHT'),
                    gtable['filename'].data.astype(np.object))
            # Finding the best tpl for BIAS, ILLUM, FLAT, TRACE, WAVE
            self.add_tplraw_to_sofdict(mean_mjd, "ILLUM")
            self.add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'FLAT', 'TRACE', 'WAVE'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Names and folder of final Master Wave
            name_products = dic_files_products['TWILIGHT']
            dir_twilight = self._get_fullpath_expo('TWILIGHT')
            name_twilight = self._get_suffix_expo('TWILIGHT')
            self.recipe_twilight(self.current_sof, dir_twilight, name_twilight, name_products, tpl)

        # Write the MASTER TWILIGHT Table and save it
        self.save_master_table('TWILIGHT', tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_scibasic_all(self, list_object=['OBJECT', 'SKY', 'STD'], tpl="ALL"):
        """Running scibasic for all objects in list_object
        Making different sof for each category
        """
        for expotype in list_object:
            sof_filename = 'scibasic_{0}'.format(expotype.lower())
            self.run_scibasic(sof_filename=sof_filename, expotype=expotype, tpl=tpl)

    def run_scibasic(self, sof_filename='scibasic', expotype="OBJECT", tpl="ALL", fits_tablename=None):
        """Reducing the files of a certain category and creating the PIXTABLES
        Will run the esorex muse_scibasic 

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        tpl_gtable = self.select_tpl_files(expotype=expotype, tpl=tpl)
        if len(tpl_gtable) == 0:
            if self.verbose :
                mpipe.print_warning("No {0} recovered from the astropy file Table - Aborting".format(expotype))
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        for gtable in tpl_gtable.groups:
            self.add_calib_to_sofdict("BADPIX_TABLE", reset=True)
            self.add_calib_to_sofdict("LINE_CATALOG")
            # extract the tpl (string) and mean mjd (float) 
            tpl, mean_mjd = self.get_tpl_meanmjd(gtable)
            self.add_geometry_to_sofdict(tpl)
            # Provide the list of files to the dictionary
            self._sofdict[expotype] = add_listpath(self._get_path_files(expotype),
                    gtable['filename'].data.astype(np.object))
            # Number of objects
            Nexpo = len(self._sofdict[expotype])
            if self.verbose:
                mpipe.print_info("Number of expo is {Nexpo} for {expotype}".format(Nexpo=Nexpo, expotype=expotype))
            # Finding the best tpl for BIAS
            self.add_tplraw_to_sofdict(mean_mjd, "ILLUM") 
            self.add_list_tplmaster_to_sofdict(mean_mjd, ['BIAS', 'FLAT', 
                'TRACE', 'WAVE', 'TWILIGHT'])
            # Writing the sof file
            self.write_sof(sof_filename=sof_filename + "_" + tpl, new=True)
            # Run the recipe to reduce the standard (muse_scibasic)
            dir_products = self._get_fullpath_expo(expotype)
            suffix = self._get_suffix_expo(expotype)
            name_products = []
            for i in range(Nexpo):
                name_products += ['{0}_{1:04d}-{2:02d}.fits'.format(suffix, i+1, j+1) for j in range(24)]
            self.recipe_scibasic(self.current_sof, tpl, expotype, dir_products, name_products)

        # Write the MASTER files Table and save it
        self.save_master_table(expotype, tpl_gtable, fits_tablename)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_std(self, sof_filename='standard', tpl="ALL"):
        """Reducing the STD files after they have been obtained
        Running the muse_standard routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        std_table = self.Tables.masterstd
        if len(std_table) == 0:
            if self.verbose :
                mpipe.print_warning("No STD recovered from the astropy file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the STD Sof
        for i in range(len(std_table)):
            mytpl = std_table['tpls'][i]
            if tpl != "ALL" and tpl != mytpl :
                continue
            # Now starting with the standard recipe
            self.add_calib_to_sofdict("EXTINCT_TABLE", reset=True)
            self.add_calib_to_sofdict("STD_FLUX_TABLE")
            dir_std = self._get_fullpath_expo('STD')
            self._sofdict['PIXTABLE_STD'] = [joinpath(dir_std, 
                'PIXTABLE_STD_{0}-{1:02d}.fits'.format(mytpl, j+1)) for j in range(24)]
            self.write_sof(sof_filename=sof_filename + "_" + mytpl, new=True)
            name_std = dic_files_products['STD']
            self.recipe_std(self.current_sof, dir_std, name_std, mytpl)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

    def run_sky(self, sof_filename='sky', tpl="ALL", fraction=0.8):
        """Reducing the SKY after they have been scibasic reduced
        Will run the esorex muse_create_sky routine

        Parameters
        ----------
        sof_filename: string (without the file extension)
            Name of the SOF file which will contain the Bias frames
        tpl: ALL by default or a special tpl time

        """
        # First selecting the files via the grouped table
        sky_table = self.Tables.mastersky
        if len(sky_table) == 0:
            if self.verbose :
                mpipe.print_warning("No SKY recovered from the astropy file Table - Aborting")
                return

        # Go to the data folder
        self.goto_folder(self.paths.data, logfile=True)

        # Create the dictionary for the LSF including
        # the list of files to be processed for one MASTER Flat
        for i in range(len(sky_table)):
            mytpl = sky_table['tpls'][i]
            mymjd = sky_table['mjd'][i]
            if tpl != "ALL" and tpl != mytpl :
                continue
            # Now starting with the standard recipe
            self.add_calib_to_sofdict("EXTINCT_TABLE", reset=True)
            self.add_calib_to_sofdict("SKY_LINES")
            self.add_skycalib_to_sofdict("STD_RESPONSE", mymjd, 'STD')
            self.add_skycalib_to_sofdict("STD_TELLURIC", mymjd, 'STD')
            self.add_tplmaster_to_sofdict(mymjd, 'LSF')
            dir_sky = self._get_fullpath_expo('SKY')
            self._sofdict['PIXTABLE_SKY'] = [joinpath(dir_sky,
                'PIXTABLE_SKY_{0:04d}-{1:02d}.fits'.format(i+1,j+1)) for j in range(24)]
            self.write_sof(sof_filename=sof_filename + "{0:02d}".format(i+1) + "_" + mytpl, new=True)
            name_sky = dic_files_products['SKY']
            self.recipe_sky(self.current_sof, dir_sky, name_sky, mytpl, fraction=fraction)

        # Go back to original folder
        self.goto_prevfolder(logfile=True)

################### OLD ############################################
#    def create_calibrations(self):
#        self.check_for_calibrations()
#        # MdB: this creates the standard calibrations, if needed
#        if not ((self.redo==0) and (self.Master['BIAS']==1)):
#            self.run_bias()
#        if not ((self.redo==0) and (self.Master['DARK']==1)):
#            self.run_dark()
#        if not ((self.redo==0) and (self.Master['FLAT']==1)):
#            self.run_flat()
#        if not ((self.redo==0) and (self.Master['WAVE']==1)):
#            self.run_wavecal()
#        if not ((self.redo==0) and (self.Master['TWILIGHT']==1)):
#            self.run_twilight()    
#        if not ((self.redo==0) and (self.Master['STD']==1)):
#            self.create_standard_star()
#            self.run_standard_int()
#
#    def check_for_calibrations(self, verbose=None):
#        """This function checks which calibration file are present, just in case
#        you do not wish to redo them. Variables are named: 
#        masterbias, masterdark, standard, masterflat, mastertwilight
#        """
#        
#        outcal = getattr(self.my_params, "master" + suffix_folder)
#        if verbose is None: verbose = self.verbose
#
#        for mastertype in dic_listMaster.keys() :
#            [fout, suffix] = dic_listMaster[mastertype]
#            # If TWILIGHT = only one cube
#            if mastertype == "TWILIGHT" :
#                if not os.path.isfile(joinpath(outcal, fout, "{suffix}.fits".format(suffix=suffix))):
#                    self.Master[mastertype] = False
#
#                if verbose :
#                    if self.Master[mastertype] :
#                        print("WARNING: Twilight flats already made")
#                    else :
#                        print("WARNING: Twilight flats NOT there yet")
#
#            # Others = 24 IFU exposures
#            else :
#                for ifu in range(1,25):
#                    if not os.path.isfile(joinpath(outcal, fout, "{0}-{1:02d}.fits".format(suffix, ifu))):
#                        self.Master[mastertype] = False
#                        break
#                if verbose :
#                    if self.Master[mastertype]:
#                        print("WARNING: Master {0} already in place".format(mastertype))
#                    else :
#                        print("WARNING: Master {0} NOT yet there".format(mastertype))
#
#    def select_stdfile(self):
#        """Minimise the time difference between the object and the illumination
#        """
#        
#        if len(self.stdlist) == 0:
#            self.stdfile=''
#            return
#        
#        # Initialisation of the time array
#        std_times = []
#        # Transforming the list into an array
#        stdlist = np.array(self.stdlist)
#        for std in stdlist:
#            std_times.append(np.datetime64(std[-28:-5]))
#        # Transforming the list into an array
#        std_times = np.array(std_times)
#
#        for object in self.objectlist: 
#            object_time = np.datetime64(object[-28:-5])
#            DeltaTime = np.abs(std_times - object_time)
#            mask = (DeltaTime == np.min(DeltaTime))
#            self.stdfile = stdlist[mask][0]
#
#    def select_run_tables(self, runname="Run01", verbose=None):
#        """
#        """
#        if len(self.objectlist) < 1 : 
#            print("WARNING: objectlist is empty, hence no CAL tables defined")
#            return
#
#        filename = self.objectlist[0]
#        hdu = pyfits.open(filename)
#        filetime = np.datetime64(hdu[0].header['DATE-OBS']).astype('datetime64[D]')
#        for runs in MUSEPIPE_runs.keys():
#            Pdates = MUSEPIPE_runs[runs]
#            date1 = np.datetime64(Pdates[1]).astype('datetime64[D]')
#            date2 = np.datetime64(Pdates[2]).astype('datetime64[D]')
#            if ((filetime >= date1) and (filetime <= date2)):
#                runname = runs
#                break
#        if verbose is None: verbose = self.verbose
#        if verbose:
#            print "Using geometry calibration data from MUSE runs %s\n"%finalkey
#        self.geo_table = joinpath(getattr(self.my_params, "musecalib" + suffix_folder), 
#                "geometry_table_wfm_{runname}.fits".format(runname=runname)) 
#        self.astro_table = joinpath(getattr(self.my_params, "musecalib" + suffix_folder), 
#                "astrometry_wcs_wfm_{runname}.fits".format(runname=runname))
#
#    def select_illumfiles(self):
#        """Minimise the difference between the OBJECT and ILLUM files
#        """
#        self.final_illumlist=[]
#        if len(self.illumlist) == 0:
#            return
#        illum_times=[]
#        illumlist=np.array(self.illumlist)
#        for illum in illumlist:
#            illum_times.append(np.datetime64(illum[-28:-5]))
#        illum_times=np.array(illum_times)
#        for object in self.objectlist: 
#            object_time=np.datetime64(object[-28:-5])
#            DeltaTime=np.abs(illum_times - object_time)
#            mask = (DeltaTime==np.min(DeltaTime))
#            self.final_illumlist.append(illumlist[mask][0])
#        
