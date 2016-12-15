from PynPoint import Pypeline
import numpy as np

from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile

from PynPoint.processing_modules import BadPixelCleaningSigmaFilterModule, \
DarkSubtractionModule, FlatSubtractionModule, CutTopTwoLinesModule, \
AngleCalculationModule, MeanBackgroundSubtractionModule, \
StarExtractionModule, StarAlignmentModule, PSFSubtractionModule, \
StackAndSubsetModule, WaveletTimeDenoisingModule, CwtWaveletConfiguration, DwtWaveletConfiguration

from PynPoint.processing_modules.PSFSubtraction import MakePSFModelModule, CreateResidualsModule


# 00 reading the data

pipeline = Pypeline("/scratch/user/mbonse/Working_files/08_klein4/",
                    "/scratch/user/mbonse/Data/00_raw_Data/",
                    "/scratch/user/mbonse/results/")

'''
pipeline = Pypeline("/Volumes/Seagate/Beta_Pic02/Working_files/",
                    "/Volumes/Seagate/Beta_Pic02/01_raw_part/",
                    "/Volumes/Seagate/Beta_Pic02/results")


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
                                num_images_in_memory=1000)
pipeline.add_module(alignment)

# 06 Angle Calculation

angle_calc = AngleCalculationModule(name_in="angle_calculation",
                                    data_tag="06_star_arr_aligned")
pipeline.add_module(angle_calc)


# 07 Wavelet Analysis
#wavelet = DwtWaveletConfiguration()

wavelet = CwtWaveletConfiguration(wavelet="dog",
                                  wavelet_order=2.0,
                                  keep_mean=True,
                                  resolution=0.2)

#wavelet_names = []
k = 1
for j in [list(np.arange(0.0, 2.1, 0.2)),
          list(np.arange(2.2, 4.1, 0.2)),
          list(np.arange(4.2, 6.1, 0.2))]:

    wavelet_thresholds = j

    wavelet_names = []
    for i in wavelet_thresholds:
        wavelet_names.append("07_wavelet_denoised_large_" + str(int(i)) + "_" + str(int((i % 1.0)*10)))

    denoising = WaveletTimeDenoisingModule(wavelet_configuration=wavelet,
                                           name_in="wavelet_time_denoising" + str(k),
                                           image_in_tag="06_star_arr_aligned",
                                           image_out_tag=wavelet_names,
                                           denoising_threshold=wavelet_thresholds,
                                           padding="const_mean",
                                           num_rows_in_memory=30)
    pipeline.add_module(denoising)
    k += 1


'''

# 08 PSF Subtraction and preparation

'''
wavelet_names = ["07_wavelet_denoised_0_0",
                 "07_wavelet_denoised_4_0",
                 "07_wavelet_denoised_8_0"]'''

wavelet_names = ["07_wavelet_denoised_4_0",
                 "07_wavelet_denoised_4_2",
                 "07_wavelet_denoised_4_4",
                 "07_wavelet_denoised_4_6",
                 "07_wavelet_denoised_4_8"]

pca_numbers = range(1, 20, 1)
pca_numbers.extend(range(25, 90, 5))

#pca_numbers = range(1, 5)

for denoising_result in wavelet_names:

    psf_subtraction = PSFSubtractionModule(0,
                                           name_in="PSF_sub_for_" + denoising_result,
                                           images_in_tag=denoising_result,
                                           reference_in_tag=denoising_result,
                                           res_mean_tag="08_res_mean_for_" + denoising_result +
                                                        "_" + str(0).zfill(2),  # 0 PCAs
                                           res_median_tag="08_res_median_for_" + denoising_result +
                                                        "_" + str(0).zfill(2),  # 0 PCAs
                                           res_rot_mean_clip_tag="08_res_rot_mean_clip_for_"
                                                                 + denoising_result +
                                                                 "_" + str(0).zfill(2),  # 0 PCAs
                                           prep_tag="prep_data",
                                           basis_out_tag="pca_basis",
                                           image_ave_tag="im_ave",
                                           cent_mask_tag="cent_mask",
                                           cent_size=0.05)
    pipeline.add_module(psf_subtraction)

    writing = WriteAsSingleFitsFile("08_res_mean_for_" + denoising_result +
                                    "_" + str(0).zfill(2) + ".fits",
                                    name_in="writing_" + "08_res_mean_for_" + denoising_result +
                                            "_" + str(0).zfill(2),
                                    data_tag="08_res_mean_for_" + denoising_result +
                                             "_" + str(0).zfill(2))
    pipeline.add_module(writing)

    for pca_number in pca_numbers:

        tmp_make_psf_model = MakePSFModelModule(num=pca_number,
                                                name_in="PSF_model_creation_for_" +
                                                        denoising_result + "_" + str(pca_number),
                                                im_arr_in_tag="prep_data",
                                                basis_in_tag="pca_basis",
                                                basis_average_in_tag="im_ave",
                                                psf_basis_out_tag="tmp_psf_model")

        tmp_residuals_module = \
            CreateResidualsModule(name_in="Creating_residuals_for_" +
                                          denoising_result + "_" + str(pca_number),
                                  im_arr_in_tag="prep_data",
                                  psf_im_in_tag="tmp_psf_model",
                                  mask_in_tag="cent_mask",
                                  res_mean_tag="08_res_mean_for_" + denoising_result +
                                               "_" + str(pca_number).zfill(2),
                                  res_median_tag="08_res_median_for_" + denoising_result +
                                               "_" + str(pca_number).zfill(2),
                                  res_rot_mean_clip_tag="08_res_rot_mean_clip_for_" +
                                                        denoising_result + "_" +
                                                        str(pca_number).zfill(2),)

        pipeline.add_module(tmp_make_psf_model)
        pipeline.add_module(tmp_residuals_module)

        # write out result as fits

        writing = WriteAsSingleFitsFile("08_res_mean_for_" + denoising_result +
                                        "_" + str(pca_number).zfill(2) + ".fits",
                                        name_in="writing_" + "08_res_mean_for_" + denoising_result +
                                                "_" + str(pca_number).zfill(2),
                                        data_tag="08_res_mean_for_" + denoising_result +
                                                 "_" + str(pca_number).zfill(2))
        pipeline.add_module(writing)

# xx run Pipeline

pipeline.run()
