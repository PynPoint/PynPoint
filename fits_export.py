from PynPoint import Pypeline
from PynPoint.io_modules import WriteAsSingleFitsFile
from PynPoint.processing_modules import RemoveMeanModule, RotateFramesModule, CombineADIModule

pipeline = Pypeline("/Volumes/Seagate/0_PCA_study",
                    "/Volumes/Seagate/0_PCA_study",
                    "/Volumes/Seagate/0_PCA_study")

# 08_wavelet_denoised_mirror_soft_1_0
# 06_star_arr_aligned

tag = "08_wavelet_denoised_mirror_soft_1_0"


mean_removal = RemoveMeanModule(name_in="remove_mean",
                                image_in_tag=tag,
                                image_out_tag=tag + "_no_mean",
                                number_of_images_in_memory=1000)
pipeline.add_module(mean_removal)

rotate = RotateFramesModule(name_in="rotation",
                            image_in_tag=tag,
                            rot_out_tag=tag + "_rotate")
pipeline.add_module(rotate)



'''
# --- mean ---
combine_mean = CombineADIModule(type="mean",
                                name_in="combine_mean",
                                image_in_tag=tag + "_rotate",
                                image_out_tag=tag + "_final_mean")

pipeline.add_module(combine_mean)

writing_mean = WriteAsSingleFitsFile("ADI_mean_wavelets.fits",
                                     name_in="fits_writing_mean",
                                     data_tag=tag + "_final_mean")

pipeline.add_module(writing_mean)'''

# --- median ---
combine_median = CombineADIModule(type="median",
                                  name_in="combine_median",
                                  image_in_tag=tag + "_rotate",
                                  image_out_tag=tag + "_final_median")

pipeline.add_module(combine_median)

writing_median = WriteAsSingleFitsFile("ADI_median_wavelets.fits",
                                       name_in="fits_writing_median",
                                       data_tag=tag + "_final_median")

pipeline.add_module(writing_median)

pipeline.run()
