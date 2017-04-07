from PynPoint import Pypeline
from PynPoint.io_modules import WriteAsSingleFitsFile

pipeline = Pypeline("/Users/markusbonse/Desktop",
                    "/Users/markusbonse/Desktop",
                    "/Users/markusbonse/Desktop")

writing1 = WriteAsSingleFitsFile("soft_wavelets.fits",
                                 name_in="fits_writing_1",
                                 data_tag="08_wavelet_denoised_soft_median_1_0")

pipeline.add_module(writing1)

pipeline.run()
