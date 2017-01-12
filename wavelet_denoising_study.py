from PynPoint import Pypeline
import numpy as np
from PynPoint.processing_modules import CwtWaveletConfiguration, WaveletTimeDenoisingModule


pipeline = Pypeline("/scratch/user/mbonse/Beta_Pic_2009_12_29_small/working_files",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_29_small/Data/00_raw_data/",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_29_small/results/")


# 07 Wavelet Analysis
wavelet = CwtWaveletConfiguration(wavelet="dog",
                                  wavelet_order=2.0,
                                  keep_mean=True,
                                  resolution=0.2)

k = 1
'''
for j in [list(np.arange(0.2, 2.1, 0.2)),
          list(np.arange(2.2, 4.1, 0.2)),
          list(np.arange(4.2, 6.0, 0.2))]:'''

for j in [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
          [0.2, 1.2, 2.2, 3.2, 4.2, 5.2],
          [0.4, 1.4, 2.4, 3.4, 4.4, 5.4],
          [0.6, 1.6, 2.6, 3.6, 4.6, 5.6],
          [0.8, 1.8, 2.8, 3.8, 4.8, 5.8]]:

    wavelet_thresholds = j

    wavelet_names = []
    for i in wavelet_thresholds:
        wavelet_names.append("07_wavelet_denoised_" + str(int(i)) + "_" + str(int(round((i % 1.0)*10))))

    denoising = WaveletTimeDenoisingModule(wavelet_configuration=wavelet,
                                           name_in="wavelet_time_denoising" + str(k),
                                           image_in_tag="06_star_arr_aligned",
                                           image_out_tag=wavelet_names,
                                           denoising_threshold=wavelet_thresholds,
                                           padding="const_mean",
                                           num_rows_in_memory=30)
    pipeline.add_module(denoising)
    k += 1

pipeline.run()
