from PynPoint import Pypeline
import numpy as np

from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile

from PynPoint.processing_modules import BadPixelCleaningSigmaFilterModule, \
DarkSubtractionModule, FlatSubtractionModule, CutTopTwoLinesModule, \
AngleCalculationModule, MeanBackgroundSubtractionModule, \
StarExtractionModule, StarAlignmentModule


# 00 reading the data

pipeline = Pypeline("/Users/markusbonse/Desktop/",
                    "/Users/markusbonse/Desktop/Science",
                    "/Users/markusbonse/Desktop/results")

'''
reading_data = ReadFitsCubesDirectory(name_in="Fits_reading",
                                      image_tag="00_raw_data")
pipeline.add_module(reading_data)'''

# star cut

cut = StarExtractionModule(image_in_tag="02_bad_pixel_clean",
                           image_out_tag="03_star_cut",
                           psf_size=3.0)
pipeline.add_module(cut)

# align

align = StarAlignmentModule(image_in_tag="03_star_cut",
                            image_out_tag="04_star_aligned",
                            accuracy=100)
pipeline.add_module(align)


w = WriteAsSingleFitsFile("cleaned.fits",
                          data_tag="04_star_aligned")
pipeline.add_module(w)

# xx run Pipeline

pipeline.run()
