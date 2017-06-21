from PynPoint import Pypeline
from PynPoint.io_modules import WriteAsSingleFitsFile
from PynPoint.processing_modules import RemoveMeanOrMedianModule, RotateFramesModule, \
    CombineADIModule, SimpleSpeckleSubtraction

pipeline = Pypeline("/Volumes/Seagate/0_PCA_study_HR8799",
                    "/Volumes/Seagate/0_PCA_study_HR8799",
                    "/Volumes/Seagate/0_PCA_study_HR8799")

tag = "08_wavelet_denoised_mirror_hard_1_0"
#tag = "08_wavelet_denoised_mirror_soft_1_0"
#tag = "06_star_arr_aligned"

# --- 00 Remove Median ---
median_removal = RemoveMeanOrMedianModule(mode="median",
                                          name_in="00_remove_median",
                                          image_in_tag=tag,
                                          image_out_tag=tag + "_00",
                                          number_of_images_in_memory=1000)

pipeline.add_module(median_removal)

# --- 01 Rotate ---
rotate = RotateFramesModule(name_in="01_rotation",
                            image_in_tag=tag + "_00",
                            rot_out_tag=tag + "_01")
pipeline.add_module(rotate)

# --- 02 Remove Mean ---
mean_removal = RemoveMeanOrMedianModule(mode="mean",
                                        name_in="02_remove_mean",
                                        image_in_tag=tag + "_01",
                                        image_out_tag=tag + "_02",
                                        number_of_images_in_memory=1000)

pipeline.add_module(mean_removal)

# --- 03 Derotate ---

rotate_inv = RotateFramesModule(mode="inverse",
                                name_in="03_inverse_rotation",
                                image_in_tag=tag + "_02",
                                rot_out_tag=tag + "_03")
pipeline.add_module(rotate_inv)

# --- 04 Speckle Subtraction ---

subtraction = SimpleSpeckleSubtraction(name_in="04_speckle_subtraction",
                                       image_in_tag=tag,
                                       speckle_in_tag=tag + "_03",
                                       image_out_tag=tag + "_04")

pipeline.add_module(subtraction)

# --- 05 Median Subtraction

median_removal = RemoveMeanOrMedianModule(mode="median",
                                          name_in="05_remove_median",
                                          image_in_tag=tag + "_04",
                                          image_out_tag=tag + "_05",
                                          number_of_images_in_memory=1000)

pipeline.add_module(median_removal)


writing_04 = WriteAsSingleFitsFile("Planet.fits",
                                   name_in="writing_planet",
                                   data_tag=tag + "_05")

pipeline.add_module(writing_04)

# --- 06 Rotate ---
rotate_planet = RotateFramesModule(name_in="06_rotation",
                                   image_in_tag=tag + "_05",
                                   rot_out_tag=tag + "_06")
pipeline.add_module(rotate_planet)

# --- 07 Combine ---

# --- mean ---
combine_mean = CombineADIModule(type_in="mean",
                                name_in="07_combine_mean",
                                image_in_tag=tag + "_06",
                                image_out_tag=tag + "_07_mean")

pipeline.add_module(combine_mean)

writing_mean = WriteAsSingleFitsFile("05_inv_ADI_mean_hard_wavelets.fits",
                                     name_in="fits_writing_mean",
                                     data_tag=tag + "_07_mean")

pipeline.add_module(writing_mean)

# --- median ---
combine_median = CombineADIModule(type_in="median",
                                  name_in="07_combine_median",
                                  image_in_tag=tag + "_06",
                                  image_out_tag=tag + "_07_median")

pipeline.add_module(combine_median)

writing_median = WriteAsSingleFitsFile("06_inv_ADI_median_hard_wavelets.fits",
                                       name_in="fits_writing_median",
                                       data_tag=tag + "_07_median")

pipeline.add_module(writing_median)


pipeline.run()
