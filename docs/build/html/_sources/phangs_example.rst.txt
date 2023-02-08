========================
PHANGS pipeline example
========================

This is an example::

    # import relevant modules
    import pymusepipe
    # Define relevant folders (data etc)
    livefold = "/data/beegfs/astro-storage/groups/schinnerer/PHANGS/MUSE/live/"
    folder_offset_tables = f"{livefold}Alignment_tables/"
    folder_align_scripts = f"{livefold}Alignment_scripts/"
    conf_folder = livefold + "Config/"
    # importing the input files
    rc_all = conf_folder + "rc_phangs.dic"
    cal_all = conf_folder + "calib_tables_phangs.dic"
    # Importing the PHANGS dictionaries
    import sys
    sys.path.append(conf_folder)
    # Has the phangs sample, phangs exposures and periods
    # And which tls expo are to be used
    from PHANGSdictionaries import *
    # Has psf_150pc (to be changed), optimal psf size and 
    # Psf characteristics for each pointing
    from PSF_dictionary import *
    # not sure what this is (expo time per expo)
    from PHANGS_EXPTIME import *
    # Setting the Sample
    def set_reduct(targetnames=None):
        # Data reduction of all PHANGS galaxies
        if targetnames is not None:
            thisdic = dict((k, dict_PHANGS_sample[k]) for k in targetnames)
        else:
            thisdic = dict_PHANGS_sample
        phangs = MusePipeSample(thisdic, rc_filename=rc_all, cal_filename=cal_all, PHANGS=True)
        print("=================================")
        print("Galaxies in the PHANGS sample ---")
        for i, name in enumerate(phangs.targetnames):
            print("Target {0:02d}: {1:10s}".format(i+1, name))
        print("=================================")
        return phangs
    # =============== REDUCTION UP TO ALIGNMENT ========================================
    # Loop over the targetnames - all included in phangs class
    for targetname in phangs.targetnames:
        # Set up the reduction for that target
        phangs = set_reduct([targetname])
        # Reduce up to pre-align
        phangs.reduce_target_prealign(targetname, filter_for_alignment=dict_filter_for_alignment[targetname])