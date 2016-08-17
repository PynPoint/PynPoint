from PynPoint import Pypeline
from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile
from PynPoint.processing_modules import BadPixelCleaningSigmaFilterModule, DarkSubtractionModule,\
    FlatSubtractionModule, CutTopTwoLinesModule, AngleCalculationModule, \
    SimpleBackgroundSubtractionModule, StarExtractionModule, StarAlignmentModule, \
    PSFSubtractionModule, StackAndSubsetModule


pype = Pypeline("/Volumes/Seagate/Beta_Pic2/",
                "/Volumes/Seagate/Beta_Pic2/00_raw_Data",
                "/Volumes/Seagate/Beta_Pic2/")

reading_data = ReadFitsCubesDirectory(name_in="Fits_reading",
                                      image_tag="im_ar")


reading_dark = ReadFitsCubesDirectory("Dark_reading",
                                      input_dir="/Volumes/Seagate/Beta_Pic2/00_Dark_and_Flat/Dark",
                                      image_tag="dark_arr")

reading_flat = ReadFitsCubesDirectory("Flat_reading",
                                      input_dir="/Volumes/Seagate/Beta_Pic2/00_Dark_and_Flat/Flat",
                                      image_tag="flat_arr")

bp_cleaning = BadPixelCleaningSigmaFilterModule(name_in="sigma_filtering",
                                                image_in_tag="flat_sub_arr",
                                                image_out_tag="im_arr_bp_clean")

dark_sub = DarkSubtractionModule(name_in="dark_subtraction",
                                 image_in_tag="im_arr_cut",
                                 dark_in_tag="dark_arr",
                                 image_out_tag="dark_sub_arr")

flat_sub = FlatSubtractionModule(name_in="flat_subtraction",
                                 image_in_tag="dark_sub_arr",
                                 flat_in_tag="flat_arr",
                                 image_out_tag="flat_sub_arr")

k4 = CutTopTwoLinesModule(name_in="dark_cutting",
                          image_in_tag="dark_arr",
                          image_out_tag="dark_arr",
                          num_images_in_memory=None)

cutting = CutTopTwoLinesModule(name_in="NACO_cutting",
                               image_in_tag="im_arr",
                               image_out_tag="im_arr_cut",
                               num_images_in_memory=1000)

bg_subtraction = SimpleBackgroundSubtractionModule(star_pos_shift=602,
                                                   image_in_tag="im_arr_bp_clean",
                                                   image_out_tag="bg_cleaned_arr")

extraction = StarExtractionModule(name_in="star_cutting",
                                  image_in_tag="bg_cleaned_arr",
                                  image_out_tag="im_arr_cut",
                                  psf_size=4,
                                  fwhm_star=7)

alignment = StarAlignmentModule(name_in="star_align",
                                image_in_tag="im_arr_cut",
                                image_out_tag="im_arr_aligned",
                                accuracy=100,
                                resize=2)

angle_calc = AngleCalculationModule(name_in="angle_calculation",
                                    data_tag="im_arr_aligned")

subset = StackAndSubsetModule(name_in="stacking_subset",
                              image_in_tag="im_arr_aligned",
                              image_out_tag="im_stacked",
                              random_subset=None,
                              stacking=20)

psf_sub = PSFSubtractionModule(pca_number=10,
                               name_in="PSF_subtraction",
                               images_in_tag="im_stacked",
                               reference_in_tag="im_stacked",
                               res_mean_tag="res_mean")

writing = WriteAsSingleFitsFile(name_in="Fits_writing",
                                file_name="test.fits",
                                data_tag="res_mean")


pype.add_module(reading_data)
pype.add_module(reading_dark)
pype.add_module(reading_flat)

pype.add_module(cutting)

pype.add_module(dark_sub)
pype.add_module(flat_sub)
pype.add_module(bp_cleaning)
pype.add_module(bg_subtraction)
pype.add_module(extraction)
pype.add_module(alignment)

pype.add_module(subset)

pype.add_module(psf_sub)

pype.add_module(writing)


#pype.add_module(j)

pype.run()

'''

import PynPoint
from PynPoint import Pypeline
from PynPoint.io_modules.Hdf5Reading import Hdf5ReadingModule
from PynPoint.processing_modules import PSFSubtractionModule
from matplotlib import pyplot as plt


pipeline = Pypeline("/Users/markusbonse/Desktop",
                    PynPoint.get_data_dir(),
                    "/Users/markusbonse/Desktop")

reading_dict = {"im_arr": "im_arr"}

reading = Hdf5ReadingModule(name_in="hdf5_reading",
                            tag_dictionary=reading_dict)

pipeline.add_module(reading)


subtraction = PSFSubtractionModule(6,
                                   name_in="PSF_subtraction",
                                   images_in_tag="im_arr",
                                   reference_in_tag="im_arr",
                                   res_mean_tag="result",
                                   cent_remove=True,
                                   cent_size=0.07)

pipeline.add_module(subtraction)

pipeline.run()

result = pipeline.get_data("result")

plt.imshow(result,
           origin='lower',
           interpolation='nearest')
plt.title("Residual Image: mean")
plt.colorbar()

plt.savefig("/Users/markusbonse/Desktop/result.png")'''




