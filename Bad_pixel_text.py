from PynPoint import Pypeline
from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile
from PynPoint.processing_modules import BadPixelMapCreationModule, BadPixelInterpolationModule, \
    CutTopTwoLinesModule

pipeline = Pypeline("/Users/markusbonse/Desktop/Bad_Pixels/Working_Files",
                    "/Users/markusbonse/Desktop/Bad_Pixels/Data/Raw",
                    output_place_in="/Users/markusbonse/Desktop/Bad_Pixels/results")

'''
# reading The data
reading = ReadFitsCubesDirectory(name_in="reading_raw",
                                 image_tag="00_raw_data")

pipeline.add_module(reading)

# read Dark

read_dark = ReadFitsCubesDirectory(name_in="reading_dark",
                                   input_dir="/Users/markusbonse/Desktop/Bad_Pixels/Data/Darks",
                                   image_tag="00_dark")
pipeline.add_module(read_dark)

# read Flat

read_flat = ReadFitsCubesDirectory(name_in="reading_flat",
                                   input_dir="/Users/markusbonse/Desktop/Bad_Pixels/Data/Flats",
                                   image_tag="00_flat")
pipeline.add_module(read_flat)

# Create Bad Pixel Map
bp_map_creation = BadPixelMapCreationModule(name_in="Bad_Pixel_Map_creation",
                                            dark_in_tag="00_dark",
                                            flat_in_tag="00_flat",
                                            bp_map_out_tag="bp_map")
pipeline.add_module(bp_map_creation)'''

# Interpolation

interpolation = BadPixelInterpolationModule(name_in="Bad_Pixel_Interpolation",
                                            image_in_tag="00_raw_data",
                                            bad_pixel_map_tag="bp_map",
                                            image_out_tag="01_bp_cleaed",
                                            iterations=1000,
                                            number_of_images_in_memory=5)
pipeline.add_module(interpolation)


# writing
writing = WriteAsSingleFitsFile("cleaned.fits",
                                data_tag="01_bp_cleaed")
pipeline.add_module(writing)

pipeline.run()
