============
Targets and Mosaicking
============

#align target from Eric about GECKOS
from astropy import units as u
from pymusepipe.align_pipe import AlignMuseDataset
myalign = AlignMuseDataset(folder_reference=ref_fold, folder_muse_images=foldalign,
                           folder_offset_table=foldtab, filter_name="LEGACY_R",
                           filter_suffix="LEGACY_R", ref_unit=53 * u.microJansky,
                           folder_output_table=foldotab, name_reference=imaref,
                           firstguess=None, name_offset_table=name_offset_table)
myalign.run_optical_flow(provide_header=True)
myalign.save_fits_offset_table(overwrite=True)


#align target from Enrico for NGC253
# set up the alignment. In this case I am using a previous reference table as first guesses for the
# optical flow. If no table exist, you have to change firstguess to something that I do not remember
# and you can remove folder_offset_table and name_offset_table.
myalign = AlignMuseDataset(folder_reference=wfifold, folder_muse_images=foldmuse,
                           folder_offset_table=foldtab, filter_name="WFI_BB",
                           folder_output_table=foldtab, name_reference=imaref,
                           firstguess="fits", name_offset_table=tabref)

# running the optical flow the first time, using the guesses provided during the setup
myalign.run_optical_flow(list_nima=None, provide_header=True, mask_stars=True)

# in this case, I knew these exposures were very badly aligned, so I am running the OF again,
# this time manually providing a guess offset I know is correct
myalign.run_optical_flow(list_nima=[138, 139, 140, 141, 142, 143, 144, 145],
                         provide_header=True, mask_stars=True, guess_offset_arcsec=[30,22])


# saving the final table.
myalign.save_fits_offset_table(name_output_table=taboref,
                               folder_output_table=foldtab, overwrite=True)