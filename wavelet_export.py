from PynPoint import Pypeline
from PynPoint.io_modules import Hdf5WritingModule


pipeline = Pypeline("/scratch/user/mbonse/HR8799_2012_08_25/working_files/normalized/",
                    "/scratch/user/mbonse/HR8799_2012_08_25/working_files/normalized/",
                    "/scratch/user/mbonse/HR8799_2012_08_25/working_files/normalized/results")

# ---------------------------

ff = {"06_star_arr_aligned" : "08_wavelet_denoised_0_0"}

saving1 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_01",
                            output_dir="/scratch/user/mbonse/HR8799_2012_08_25/working_files/normalized/08_klein_0",
                            tag_dictionary=ff)

pipeline.add_module(saving1)


# ---------------------------

ff = {"08_wavelet_denoised_1_0" : "08_wavelet_denoised_1_0"}

saving2 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_02",
                            output_dir="/scratch/user/mbonse/HR8799_2012_08_25/working_files/normalized/08_klein_1",
                            tag_dictionary=ff)

pipeline.add_module(saving2)

# ---------------------------

ff = {"08_wavelet_denoised_2_0" : "08_wavelet_denoised_2_0"}

saving3 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_03",
                            output_dir="/scratch/user/mbonse/HR8799_2012_08_25/working_files/normalized/08_klein_2",
                            tag_dictionary=ff)

pipeline.add_module(saving3)


'''
# ---------------------------

ff = {"07_wavelet_denoised_1_2" : "07_wavelet_denoised_1_2"}

saving4 = Hdf5WritingModule("PynPoint_database.hdf5",
                            name_in="hdf5_writing_04",
                            output_dir="/scratch/user/mbonse/HR8799_2012_08_26_ND/working_files/08_klein_3",
                            tag_dictionary=ff)

pipeline.add_module(saving4)

# ---------------------------'''

pipeline.run()
