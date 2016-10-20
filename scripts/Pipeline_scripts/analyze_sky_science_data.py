"""
End to End pipeline for Sky / Science data. (Cubes containing sky frames and cubes containing the
star) NACO/AGPM
"""

from PynPoint import Pypeline

from PynPoint.processing_modules import ReadFitsSkyDirectory, MeanSkyCubes, DarkSubtractionModule, \
    SkySubtraction, BadPixelCleaningSigmaFilterModule, FlatSubtractionModule, CutAroundCenterModule,\
    AngleCalculationModule, AlignmentSkyAndScienceDataModule, CutAroundPositionModule, \
    StackAndSubsetModule, PSFSubtractionModule
from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile


pipeline = Pypeline("/Volumes/Seagate/NEW_DATA",
                    "/Volumes/Seagate/NEW_DATA",
                    "/Volumes/Seagate/NEW_DATA")

# ---- Read Sky data
sky_reading = ReadFitsSkyDirectory(name_in="sky_reading",
                                   input_dir="/Volumes/Seagate/NEW_DATA/Sky",
                                   sky_tag="sky_raw_arr")
pipeline.add_module(sky_reading)

# ---- Read dark
dark_reading = ReadFitsCubesDirectory(name_in="dark_reading",
                                      input_dir="/Volumes/Seagate/NEW_DATA/dark",
                                      image_tag="dark_arr")
pipeline.add_module(dark_reading)

# ---- Read sky flat
flat_reading = ReadFitsCubesDirectory(name_in="flat_reading",
                                      input_dir="/Volumes/Seagate/NEW_DATA/Sky_flat",
                                      image_tag="flat_arr")
pipeline.add_module(flat_reading)

# --- Read Science data

science_reading = ReadFitsCubesDirectory(name_in="science_reading",
                                         input_dir="/Volumes/Seagate/NEW_DATA/Science",
                                         image_tag="im_arr")
pipeline.add_module(science_reading)

# ---- average all sky images
sky_mean = MeanSkyCubes(sky_in_tag="sky_raw_arr",
                        sky_out_tag="sky_arr")
pipeline.add_module(sky_mean)

# ---- subtract dark form sky
dark_subtraction1 = DarkSubtractionModule(name_in="dark_subtraction1",
                                          image_in_tag="sky_arr",
                                          dark_in_tag="dark_arr",
                                          image_out_tag="sky_arr",
                                          number_of_images_in_memory=None)

pipeline.add_module(dark_subtraction1)

# --- subtract dark from science

dark_subtraction2 = DarkSubtractionModule(name_in="dark_subtraction2",
                                          image_in_tag="im_arr",
                                          dark_in_tag="dark_arr",
                                          image_out_tag="im_arr",
                                          number_of_images_in_memory=100)

pipeline.add_module(dark_subtraction2)

# --- cut sky flat to dark resolution

cutting_sky_flat = CutAroundCenterModule((776, 768),
                                         image_in_tag="flat_arr",
                                         image_out_tag="flat_arr",
                                         number_of_images_in_memory=None)
pipeline.add_module(cutting_sky_flat)


# --- subtract dark from sky flat

dark_subtraction3 = DarkSubtractionModule(name_in="dark_subtraction3",
                                          image_in_tag="flat_arr",
                                          dark_in_tag="dark_arr",
                                          image_out_tag="flat_arr",
                                          number_of_images_in_memory=100)

pipeline.add_module(dark_subtraction3)

# --- subtract flat from sky

flat_subtraction1 = FlatSubtractionModule(name_in="flat_subtraction1",
                                          image_in_tag="sky_arr",
                                          flat_in_tag="flat_arr",
                                          image_out_tag="sky_arr_clean",
                                          number_of_images_in_memory=100)

pipeline.add_module(flat_subtraction1)

# --- subtract flat from science

flat_subtraction2 = FlatSubtractionModule(name_in="flat_subtraction2",
                                          image_in_tag="im_arr",
                                          flat_in_tag="flat_arr",
                                          image_out_tag="im_arr_clean",
                                          number_of_images_in_memory=100)

pipeline.add_module(flat_subtraction2)

# --- run bad pixel cleaning on Science data
bp_cleaning1 = BadPixelCleaningSigmaFilterModule(name_in="sigma_filtering_science",
                                                 image_in_tag="im_arr_clean",
                                                 image_out_tag="im_arr_clean")
pipeline.add_module(bp_cleaning1)

# --- run bad pixel cleaning on Sky data
bp_cleaning2 = BadPixelCleaningSigmaFilterModule(name_in="sigma_filtering_sky",
                                                 image_in_tag="sky_arr_clean",
                                                 image_out_tag="sky_arr_clean")
pipeline.add_module(bp_cleaning2)

# --- align Sky and Science Data

align_sky = AlignmentSkyAndScienceDataModule(position_of_center=(390, 440),
                                             science_in_tag="im_arr_clean",
                                             science_out_tag="im_arr_align",
                                             sky_in_tag="sky_arr_clean",
                                             sky_out_tag="sky_arr_align",
                                             size_of_center=(100, 100),
                                             accuracy=100)

pipeline.add_module(align_sky)

# --- run background subtraction
sky_subtraction = SkySubtraction(name_in="sky_subtraction",
                                 sky_in_tag="sky_arr_align",
                                 science_data_in_tag="im_arr_align",
                                 science_data_out_tag="bg_cleaned_arr",
                                 mode="next")

pipeline.add_module(sky_subtraction)


# --- bp cleaning 3

bp_cleaning3 = BadPixelCleaningSigmaFilterModule(name_in="sigma_filtering_final",
                                                 image_in_tag="bg_cleaned_arr",
                                                 image_out_tag="bg_cleaned_arr")
pipeline.add_module(bp_cleaning3)

# --- cut Star

cut_star = CutAroundPositionModule(name_in="star_extraction",
                                   new_shape=(110, 110),
                                   center_of_cut=(394, 440),
                                   image_in_tag="bg_cleaned_arr",
                                   image_out_tag="star_cut_arr")
pipeline.add_module(cut_star)

# --- anlge calculation

angle_calc = AngleCalculationModule(name_in="angle_calculation",
                                    data_tag="star_cut_arr")
pipeline.add_module(angle_calc)

# --- stacking

subset = StackAndSubsetModule(name_in="stacking_subset",
                              image_in_tag="star_cut_arr",
                              image_out_tag="star_stacked",
                              random_subset=None,
                              stacking=None)

pipeline.add_module(subset)

# --- PCA

psf_sub = PSFSubtractionModule(pca_number=20,
                               cent_size=0.05,
                               name_in="PSF_subtraction",
                               images_in_tag="star_stacked",
                               reference_in_tag="star_stacked",
                               res_mean_tag="res_mean")
pipeline.add_module(psf_sub)

# --- writing out

writing = WriteAsSingleFitsFile(name_in="Fits_writing",
                                file_name="results/planet.fits",
                                data_tag="res_mean")
pipeline.add_module(writing)

writing2 = WriteAsSingleFitsFile(name_in="Fits_writing_input",
                                 file_name="results/data.fits",
                                 data_tag="star_stacked")
pipeline.add_module(writing2)

pipeline.run()
