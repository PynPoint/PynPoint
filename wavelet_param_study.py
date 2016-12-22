from PynPoint import Pypeline
import numpy as np

from PynPoint.io_modules import ReadFitsCubesDirectory

from PynPoint.processing_modules import BadPixelCleaningSigmaFilterModule, \
DarkSubtractionModule, FlatSubtractionModule, CutTopTwoLinesModule, \
AngleCalculationModule, MeanBackgroundSubtractionModule, \
StarExtractionModule, StarAlignmentModule


# 00 reading the data

pipeline = Pypeline("/scratch/user/mbonse/Beta_Pic_2009_12_26/working_files/",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_26/data/00_raw_data/",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_26/results/")

'''
pipeline = Pypeline("/Volumes/Seagate/Beta_Pic02/Working_files/",
                    "/Volumes/Seagate/Beta_Pic02/01_raw_part/",
                    "/Volumes/Seagate/Beta_Pic02/results")

'''
reading_data = ReadFitsCubesDirectory(name_in="Fits_reading",
                                      image_tag="00_raw_data")
pipeline.add_module(reading_data)

reading_dark = ReadFitsCubesDirectory(name_in="Dark_reading",
                                      input_dir="/scratch/user/mbonse/Beta_Pic_2009_12_26/data/00_dark/",
                                      image_tag="00_dark_arr")
pipeline.add_module(reading_dark)

reading_flat = ReadFitsCubesDirectory(name_in="Flat_reading",
                                      input_dir="/scratch/user/mbonse/Beta_Pic_2009_12_26/data/00_flat/",
                                      image_tag="00_flat_arr")
pipeline.add_module(reading_flat)

# 01 NACO cutting

cutting = CutTopTwoLinesModule(name_in="NACO_cutting",
                               image_in_tag="00_raw_data",
                               image_out_tag="01_raw_data_cut")
pipeline.add_module(cutting)

# 02 Dark and Flat Subtraction

dark_sub = DarkSubtractionModule(name_in="dark_subtraction",
                                 image_in_tag="01_raw_data_cut",
                                 dark_in_tag="00_dark_arr",
                                 image_out_tag="02_dark_sub")

flat_sub = FlatSubtractionModule(name_in="flat_subtraction",
                                 image_in_tag="02_dark_sub",
                                 flat_in_tag="00_flat_arr",
                                 image_out_tag="02_dark_flat_sub")

pipeline.add_module(dark_sub)
pipeline.add_module(flat_sub)

# 03 Bad Pixel

bp_cleaning = BadPixelCleaningSigmaFilterModule(name_in="Bad_Pixel_filtering",
                                                image_in_tag="02_dark_flat_sub",
                                                image_out_tag="03_bad_pixel_clean")
pipeline.add_module(bp_cleaning)

# 04 Background Subtraction

bg_subtraction = MeanBackgroundSubtractionModule(star_pos_shift=602,
                                                 name_in="mean_background_subtraction",
                                                 image_in_tag="03_bad_pixel_clean",
                                                 image_out_tag="04_mean_background_sub")
pipeline.add_module(bg_subtraction)

# 05 Star extraction

star_cut = StarExtractionModule(name_in="star_cutting",
                                image_in_tag="04_mean_background_sub",
                                image_out_tag="05_star_arr_cut",
                                psf_size=3.0,
                                num_images_in_memory=None,
                                fwhm_star=7)
pipeline.add_module(star_cut)

# 06 Alignment

alignment = StarAlignmentModule(name_in="star_alignment",
                                image_in_tag="05_star_arr_cut",
                                image_out_tag="06_star_arr_aligned",
                                interpolation="spline",
                                accuracy=10,
                                resize=2.0,
                                num_images_in_memory=None)
pipeline.add_module(alignment)

# 06 Angle Calculation

angle_calc = AngleCalculationModule(name_in="angle_calculation",
                                    data_tag="06_star_arr_aligned")
pipeline.add_module(angle_calc)

# xx run Pipeline

pipeline.run()
