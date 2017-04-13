from PynPoint import Pypeline
from PynPoint.io_modules import Hdf5WritingModule


pipeline = Pypeline("/scratch/user/mbonse/Beta_Pic_2009_12_29_norm/Workplace",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_29_norm/Data/00_raw_Data",
                    "/scratch/user/mbonse/Beta_Pic_2009_12_29_norm/Results")

# ---------------------------

ff = {"06_star_arr_aligned" : "06_star_arr_aligned",
      "07_star_arr_normalized": "07_star_arr_normalized"}

saving1 = Hdf5WritingModule("PynPoint_database_06_07.hdf5",
                            name_in="hdf5_writing_01",
                            output_dir="/scratch/user/mbonse/Beta_Pic_2009_12_29_norm/Workplace",
                            tag_dictionary=ff)

pipeline.add_module(saving1)

'''
# ---------------------------

ff = {"08_wavelet_denoised_mirror_soft_1_0" : "08_wavelet_denoised_mirror_soft_1_0",
      "08_wavelet_denoised_mirror_soft_median_1_0" : "08_wavelet_denoised_mirror_soft_median_1_0"}

saving2 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_02",
                            output_dir="/scratch/user/mbonse/HR8799_2012_08_25_norm/Workplace/08_klein_1",
                            tag_dictionary=ff)

pipeline.add_module(saving2)

# ---------------------------

ff = {"08_wavelet_denoised_zero_hard_1_0" : "08_wavelet_denoised_zero_hard_1_0",
      "08_wavelet_denoised_zero_hard_median_1_0" : "08_wavelet_denoised_zero_hard_median_1_0"}

saving3 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_03",
                            output_dir="/scratch/user/mbonse/HR8799_2012_08_25_norm/Workplace/08_klein_2",
                            tag_dictionary=ff)

pipeline.add_module(saving3)



# ---------------------------


ff = {"08_wavelet_denoised_zero_soft_1_0" : "08_wavelet_denoised_zero_soft_1_0",
      "08_wavelet_denoised_zero_soft_median_1_0" : "08_wavelet_denoised_zero_soft_median_1_0"}

saving4 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_04",
                            output_dir="/scratch/user/mbonse/HR8799_2012_08_25_norm/Workplace/08_klein_3",
                            tag_dictionary=ff)

pipeline.add_module(saving4)'''

# ---------------------------

pipeline.run()
