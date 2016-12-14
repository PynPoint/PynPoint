from PynPoint import Pypeline
from PynPoint.io_modules import Hdf5WritingModule


pipeline = Pypeline("/scratch/user/mbonse/Working_files/",
                    "/scratch/user/mbonse/Data/00_raw_Data/",
                    "/scratch/user/mbonse/results/")

ff = {"07_wavelet_denoised_0_0" : "07_wavelet_denoised_0_0",
      "07_wavelet_denoised_0_2" : "07_wavelet_denoised_0_2",
      "07_wavelet_denoised_0_4" : "07_wavelet_denoised_0_4",
      "07_wavelet_denoised_0_6" : "07_wavelet_denoised_0_6",
      "07_wavelet_denoised_0_8" : "07_wavelet_denoised_0_8",
      "07_wavelet_denoised_1_0" : "07_wavelet_denoised_1_0",
      "07_wavelet_denoised_1_2" : "07_wavelet_denoised_1_2",
      "07_wavelet_denoised_1_4" : "07_wavelet_denoised_1_4",
      "07_wavelet_denoised_1_6" : "07_wavelet_denoised_1_6",
      "07_wavelet_denoised_1_8" : "07_wavelet_denoised_1_8",
      "07_wavelet_denoised_2_0" : "07_wavelet_denoised_2_0",
      "07_wavelet_denoised_2_2" : "07_wavelet_denoised_2_2",
      "07_wavelet_denoised_2_4" : "07_wavelet_denoised_2_4",
      "07_wavelet_denoised_2_6" : "07_wavelet_denoised_2_6",
      "07_wavelet_denoised_2_8" : "07_wavelet_denoised_2_8",
      "07_wavelet_denoised_3_0" : "07_wavelet_denoised_3_0",
      "07_wavelet_denoised_3_2" : "07_wavelet_denoised_3_2",
      "07_wavelet_denoised_3_4" : "07_wavelet_denoised_3_4",
      "07_wavelet_denoised_3_6" : "07_wavelet_denoised_3_6",
      "07_wavelet_denoised_3_8" : "07_wavelet_denoised_3_8",
      "07_wavelet_denoised_4_0" : "07_wavelet_denoised_4_0",
      "07_wavelet_denoised_4_2" : "07_wavelet_denoised_4_2",
      "07_wavelet_denoised_4_4" : "07_wavelet_denoised_4_4",
      "07_wavelet_denoised_4_6" : "07_wavelet_denoised_4_6",
      "07_wavelet_denoised_4_8" : "07_wavelet_denoised_4_8",
      "07_wavelet_denoised_5_0" : "07_wavelet_denoised_5_0",
      "07_wavelet_denoised_5_2" : "07_wavelet_denoised_5_2",
      "07_wavelet_denoised_5_4" : "07_wavelet_denoised_5_4",
      "07_wavelet_denoised_5_6" : "07_wavelet_denoised_5_6",
      "07_wavelet_denoised_5_8" : "07_wavelet_denoised_5_8",
      "07_wavelet_denoised_6_0" : "07_wavelet_denoised_6_0",
      "07_wavelet_denoised_6_2" : "07_wavelet_denoised_6_2",
      "07_wavelet_denoised_6_4" : "07_wavelet_denoised_6_4",
      "07_wavelet_denoised_6_6" : "07_wavelet_denoised_6_6",
      "07_wavelet_denoised_6_8" : "07_wavelet_denoised_6_8",
      "07_wavelet_denoised_7_0" : "07_wavelet_denoised_7_0",
      "07_wavelet_denoised_7_2" : "07_wavelet_denoised_7_2",
      "07_wavelet_denoised_7_4" : "07_wavelet_denoised_7_4",
      "07_wavelet_denoised_7_6" : "07_wavelet_denoised_7_6",
      "07_wavelet_denoised_7_8" : "07_wavelet_denoised_7_8",
      "07_wavelet_denoised_8_0" : "07_wavelet_denoised_8_0"}

saving = Hdf5WritingModule("PynPoint_database2.hdf5",
                           tag_dictionary=ff)

pipeline.add_module(saving)
pipeline.run()