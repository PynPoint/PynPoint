from PynPoint import Pypeline
import numpy as np
from PynPoint.processing_modules import CwtWaveletConfiguration, WaveletTimeDenoisingModule


pipeline = Pypeline("/scratch/user/mbonse/Beta_Pic_2009_12_26/working_files/",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_26/data/00_raw_data/",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_26/results/")


# 07 Wavelet Analysis
wavelet = CwtWaveletConfiguration(wavelet="dog",
                                  wavelet_order=2.0,
                                  keep_mean=True,
                                  resolution=0.2)

k = 1
for j in [list(np.arange(0.2, 2.1, 0.2)),
          list(np.arange(2.2, 4.1, 0.2)),
          list(np.arange(4.2, 6.0, 0.2))]:

    wavelet_thresholds = j

    wavelet_names = []
    for i in wavelet_thresholds:
        wavelet_names.append("07_wavelet_denoised_" + str(int(i)) + "_" + str(int((i % 1.0)*10)))

    denoising = WaveletTimeDenoisingModule(wavelet_configuration=wavelet,
                                           name_in="wavelet_time_denoising" + str(k),
                                           image_in_tag="06_star_arr_aligned",
                                           image_out_tag=wavelet_names,
                                           denoising_threshold=wavelet_thresholds,
                                           padding="const_mean",
                                           num_rows_in_memory=60)
    pipeline.add_module(denoising)
    k += 1

pipeline.run()
