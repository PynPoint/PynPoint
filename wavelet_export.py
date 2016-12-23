from PynPoint import Pypeline
from PynPoint.io_modules import Hdf5WritingModule


pipeline = Pypeline("/scratch/user/mbonse/Beta_Pic_2009_12_29/Working_files/08_gross/",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_29/Working_files/08_gross/",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_29/Working_files/08_gross/")

# ---------------------------

ff = {"06_star_arr_aligned" : "07_wavelet_denoised_0_0",
      "07_wavelet_denoised_0_2" : "07_wavelet_denoised_0_2",
      "07_wavelet_denoised_0_4" : "07_wavelet_denoised_0_4",
      "07_wavelet_denoised_0_6" : "07_wavelet_denoised_0_6",
      "07_wavelet_denoised_0_8" : "07_wavelet_denoised_0_8"}

saving1 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_01",
                            output_dir="/scratch/user/mbonse/Beta_Pic_2009_12_29/Working_files/08_gross_1",
                            tag_dictionary=ff)

pipeline.add_module(saving1)

# ---------------------------

ff = {"07_wavelet_denoised_1_0" : "07_wavelet_denoised_1_0",
      "07_wavelet_denoised_1_2" : "07_wavelet_denoised_1_2",
      "07_wavelet_denoised_1_4" : "07_wavelet_denoised_1_4",
      "07_wavelet_denoised_1_6" : "07_wavelet_denoised_1_6",
      "07_wavelet_denoised_1_8" : "07_wavelet_denoised_1_8"}

saving2 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_02",
                            output_dir="/scratch/user/mbonse/Beta_Pic_2009_12_29/Working_files/08_gross_2",
                            tag_dictionary=ff)

pipeline.add_module(saving2)

# ---------------------------

ff = {"07_wavelet_denoised_2_0" : "07_wavelet_denoised_2_0",
      "07_wavelet_denoised_2_2" : "07_wavelet_denoised_2_2",
      "07_wavelet_denoised_2_4" : "07_wavelet_denoised_2_4",
      "07_wavelet_denoised_2_6" : "07_wavelet_denoised_2_6",
      "07_wavelet_denoised_2_8" : "07_wavelet_denoised_2_8"}

saving3 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_03",
                            output_dir="/scratch/user/mbonse/Beta_Pic_2009_12_29/Working_files/08_gross_3",
                            tag_dictionary=ff)

pipeline.add_module(saving3)

# ---------------------------

ff = {"07_wavelet_denoised_3_0" : "07_wavelet_denoised_3_0",
      "07_wavelet_denoised_3_2" : "07_wavelet_denoised_3_2",
      "07_wavelet_denoised_3_4" : "07_wavelet_denoised_3_4",
      "07_wavelet_denoised_3_6" : "07_wavelet_denoised_3_6",
      "07_wavelet_denoised_3_8" : "07_wavelet_denoised_3_8"}

saving4 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_04",
                            output_dir="/scratch/user/mbonse/Beta_Pic_2009_12_29/Working_files/08_gross_4",
                            tag_dictionary=ff)

pipeline.add_module(saving4)

# ---------------------------

ff = {"07_wavelet_denoised_4_0" : "07_wavelet_denoised_4_0",
      "07_wavelet_denoised_4_2" : "07_wavelet_denoised_4_2",
      "07_wavelet_denoised_4_4" : "07_wavelet_denoised_4_4",
      "07_wavelet_denoised_4_6" : "07_wavelet_denoised_4_6",
      "07_wavelet_denoised_4_8" : "07_wavelet_denoised_4_8"}

saving5 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_05",
                            output_dir="/scratch/user/mbonse/Beta_Pic_2009_12_29/Working_files/08_gross_5",
                            tag_dictionary=ff)

pipeline.add_module(saving5)

# ---------------------------

ff = {"07_wavelet_denoised_5_0" : "07_wavelet_denoised_5_0",
      "07_wavelet_denoised_5_2" : "07_wavelet_denoised_5_2",
      "07_wavelet_denoised_5_4" : "07_wavelet_denoised_5_4",
      "07_wavelet_denoised_5_6" : "07_wavelet_denoised_5_6",
      "07_wavelet_denoised_5_8" : "07_wavelet_denoised_5_8",
      "07_wavelet_denoised_6_0" : "07_wavelet_denoised_6_0"}

saving6 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_06",
                            output_dir="/scratch/user/mbonse/Beta_Pic_2009_12_29/Working_files/08_gross_6",
                            tag_dictionary=ff)

pipeline.add_module(saving6)

# ---------------------------


pipeline.run()
