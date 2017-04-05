from PynPoint import Pypeline
from PynPoint.io_modules import WriteAsSingleFitsFile

pipeline = Pypeline("/Volumes/Seagate/ETH/normalized",
                    "/Volumes/Seagate/ETH/normalized",
                    "/Volumes/Seagate/ETH/normalized/results")

writing1 = WriteAsSingleFitsFile("no_wavelets.fits",
                                 name_in="fits_writing_1",
                                 data_tag="07_star_arr_normalized")

pipeline.add_module(writing1)

writing2 = WriteAsSingleFitsFile("wavelets.fits",
                                 name_in="fits_writing_2",
                                 data_tag="08_wavelet_denoised_1_0")

pipeline.add_module(writing2)

pipeline.run()
