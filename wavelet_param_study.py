from PynPoint import Pypeline

from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile

from PynPoint.processing_modules import BadPixelCleaningSigmaFilterModule, \
DarkSubtractionModule, FlatSubtractionModule, CutTopTwoLinesModule, \
AngleCalculationModule, SimpleBackgroundSubtractionModule, \
StarExtractionModule, StarAlignmentModule, PSFSubtractionModule, \
StackAndSubsetModule


# 00 reading the data

pipeline = Pypeline("/scratch/user/mbonse/Working_files/",
                    "/scratch/user/mbonse/Data/00_raw_Data/",
                    "/scratch/user/mbonse/results/")

reading_data = ReadFitsCubesDirectory(name_in="Fits_reading",
                                      image_tag="00_raw_data")
pipeline.add_module(reading_data)

reading_dark = ReadFitsCubesDirectory(name_in="Dark_reading",
                                      input_dir="/scratch/user/mbonse/Data/00_Dark_and_Flat/Dark",
                                      image_tag="00_dark_arr")
pipeline.add_module(reading_dark)

reading_flat = ReadFitsCubesDirectory(name_in="Flat_reading",
                                      input_dir="/scratch/user/mbonse/Data/00_Dark_and_Flat/Flat",
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
                                 image_in_tag="02_raw_data_dark_sub",
                                 flat_in_tag="00_flat_arr",
                                 image_out_tag="02_dark_flat_sub")

pipeline.add_module(dark_sub)
pipeline.add_module(flat_sub)

# 03 Bad Pixel

bp_cleaning = BadPixelCleaningSigmaFilterModule(name_in="Bad_Pixel_filtering",
                                                image_in_tag="02_dark_flat_sub",
                                                image_out_tag="03_bad_pixel_clean")
pipeline.add_module(bp_cleaning)


# xx run Pipeline

pipeline.run()
