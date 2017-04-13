from PynPoint import Pypeline
import numpy as np
from PynPoint.processing_modules import CwtWaveletConfiguration, WaveletTimeDenoisingModule


pipeline = Pypeline("/scratch/user/mbonse/Beta_Pic_2009_12_26_norm/Workplace",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_26_norm/Data/00_raw_Data",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_26_norm/Results")


# 07 Wavelet Analysis
wavelet = CwtWaveletConfiguration(wavelet="dog",
                                  wavelet_order=2.0,
                                  keep_mean=True,
                                  resolution=0.2)

k = 1

# hard threshold
for j in [[1.0], ]:

    wavelet_thresholds = j

    wavelet_names = []
    for i in wavelet_thresholds:
        wavelet_names.append("08_wavelet_denoised_planet_save_" + str(int(i)) + "_" + str(int(round((i % 1.0)*10))))

    denoising = WaveletTimeDenoisingModule(wavelet_configuration=wavelet,
                                           name_in="wavelet_time_denoising" + str(k),
                                           image_in_tag="07_star_arr_normalized",
                                           image_out_tag=wavelet_names,
                                           denoising_threshold=wavelet_thresholds,
                                           padding="zero",
                                           median_filter=False,
                                           threshold_function="hard",
                                           num_rows_in_memory=30)
    pipeline.add_module(denoising)
    k += 1

'''
# soft threshold
for j in [[1.0], ]:

    wavelet_thresholds = j

    wavelet_names = []
    for i in wavelet_thresholds:
        wavelet_names.append("08_wavelet_denoised_mirror_soft_" + str(int(i)) + "_" + str(int(round((i % 1.0)*10))))

    denoising = WaveletTimeDenoisingModule(wavelet_configuration=wavelet,
                                           name_in="wavelet_time_denoising" + str(k),
                                           image_in_tag="07_star_arr_normalized",
                                           image_out_tag=wavelet_names,
                                           denoising_threshold=wavelet_thresholds,
                                           padding="mirror",
                                           median_filter=False,
                                           threshold_function="soft",
                                           num_rows_in_memory=30)
    pipeline.add_module(denoising)
    k += 1

# ---------- zeros padding -------

# hard threshold
for j in [[1.0], ]:

    wavelet_thresholds = j

    wavelet_names = []
    for i in wavelet_thresholds:
        wavelet_names.append("08_wavelet_denoised_zero_hard_" + str(int(i)) + "_" + str(int(round((i % 1.0)*10))))

    denoising = WaveletTimeDenoisingModule(wavelet_configuration=wavelet,
                                           name_in="wavelet_time_denoising" + str(k),
                                           image_in_tag="07_star_arr_normalized",
                                           image_out_tag=wavelet_names,
                                           denoising_threshold=wavelet_thresholds,
                                           padding="zero",
                                           median_filter=False,
                                           threshold_function="hard",
                                           num_rows_in_memory=30)
    pipeline.add_module(denoising)
    k += 1


# soft threshold
for j in [[1.0], ]:

    wavelet_thresholds = j

    wavelet_names = []
    for i in wavelet_thresholds:
        wavelet_names.append("08_wavelet_denoised_zero_soft_" + str(int(i)) + "_" + str(int(round((i % 1.0)*10))))

    denoising = WaveletTimeDenoisingModule(wavelet_configuration=wavelet,
                                           name_in="wavelet_time_denoising" + str(k),
                                           image_in_tag="07_star_arr_normalized",
                                           image_out_tag=wavelet_names,
                                           denoising_threshold=wavelet_thresholds,
                                           padding="zero",
                                           median_filter=False,
                                           threshold_function="soft",
                                           num_rows_in_memory=30)
    pipeline.add_module(denoising)
    k += 1'''


pipeline.run()
