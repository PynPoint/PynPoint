from PynPoint import Pypeline

from PynPoint.processing_modules import ReadFitsSkyDirectory, MeanSkyCubes, DarkSubtractionModule, \
    SkySubtraction, BadPixelCleaningSigmaFilterModule
from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile


pipeline = Pypeline("/Volumes/Seagate/NEW_DATA_PART",
                    "/Volumes/Seagate/NEW_DATA_PART",
                    "/Volumes/Seagate/NEW_DATA_PART")

# ---- Read Sky data
sky_reading = ReadFitsSkyDirectory(name_in="sky_reading",
                                   input_dir="/Volumes/Seagate/NEW_DATA_PART/Sky",
                                   sky_tag="sky_raw_arr")
pipeline.add_module(sky_reading)

# ---- Read dark
dark_reading = ReadFitsCubesDirectory(name_in="dark_reading",
                                      image_tag="dark_arr")
pipeline.add_module(dark_reading)

# --- Read Science data

science_reading = ReadFitsCubesDirectory(name_in="science_reading",
                                         input_dir="/Volumes/Seagate/NEW_DATA_PART/Science",
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

# --- run background subtraction
sky_subtraction = SkySubtraction(name_in="sky_subtraction",
                                 sky_in_tag="sky_arr",
                                 science_data_in_tag="im_arr",
                                 science_data_out_tag="bg_cleaned_arr",
                                 mode="both")

pipeline.add_module(sky_subtraction)

# --- run bad pixel cleaning
bp_cleaning = BadPixelCleaningSigmaFilterModule(name_in="sigma_filtering",
                                                image_in_tag="bg_cleaned_arr",
                                                image_out_tag="bg_cleaned_arr")
pipeline.add_module(bp_cleaning)

# --- write out

writing = WriteAsSingleFitsFile("results/result.fits",
                                name_in="writing_results",
                                data_tag="bg_cleaned_arr")

pipeline.add_module(writing)

pipeline.run()

#pipeline.run_module("writing_results")