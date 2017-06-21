from PynPoint import Pypeline
from PynPoint.io_modules import WriteAsSingleFitsFile
from PynPoint.processing_modules import RemoveMeanOrMedianModule, RotateFramesModule, \
    CombineADIModule

pipeline = Pypeline("/Volumes/Seagate/0_PCA_study_HR8799",
                    "/Volumes/Seagate/0_PCA_study_HR8799",
                    "/Volumes/Seagate/0_PCA_study_HR8799")

#tag = "08_wavelet_denoised_mirror_hard_1_0"
tag = "08_wavelet_denoised_mirror_soft_1_0"
#tag = "06_star_arr_aligned"

# --- 01 median / mean removal

removal = RemoveMeanOrMedianModule(mode="mean",
                                   name_in="01_remove_mean_median",
                                   image_in_tag=tag,
                                   image_out_tag=tag + "_normal_01",
                                   number_of_images_in_memory=1000)
pipeline.add_module(removal)

writing = WriteAsSingleFitsFile("no_mean_wavelets_time.fits",
                                data_tag=tag + "_normal_01")
pipeline.add_module(writing)

'''

# --- 02 rotate ---

rotate = RotateFramesModule(name_in="02_rotation",
                            image_in_tag=tag + "_normal_01",
                            rot_out_tag=tag + "_normal_02")
pipeline.add_module(rotate)


# --- 03 combine mean ---
combine_mean = CombineADIModule(type_in="mean",
                                name_in="combine_mean",
                                image_in_tag=tag + "_normal_02",
                                image_out_tag=tag + "_normal_03_final_mean")

pipeline.add_module(combine_mean)

writing_mean = WriteAsSingleFitsFile("03_Median_ADI_mean_hard_wavelets.fits",
                                     name_in="fits_writing_mean",
                                     data_tag=tag + "_normal_03_final_mean")

pipeline.add_module(writing_mean)

# ---  03 combine median ---
combine_median = CombineADIModule(type_in="median",
                                  name_in="combine_median",
                                  image_in_tag=tag + "_normal_02",
                                  image_out_tag=tag + "_normal_03_final_median")

pipeline.add_module(combine_median)

writing_median = WriteAsSingleFitsFile("04_Median_ADI_median_hard_wavelets.fits",
                                       name_in="fits_writing_median",
                                       data_tag=tag + "_normal_03_final_median")

pipeline.add_module(writing_median)'''

pipeline.run()
