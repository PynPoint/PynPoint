import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from PynPoint import Pypeline
from PynPoint.processing_modules import WaveletTimeDenoisingModule, CwtWaveletConfiguration

pipeline = Pypeline("/scratch/user/mbonse/FastPca",
                    "/scratch/user/mbonse/FastPca",
                    "/scratch/user/mbonse/FastPca")

from PynPoint.processing_modules import PSFSubtractionPCA

psf_subtraction = PSFSubtractionPCA(range(1000),
                                    name_in="PSF_subtraction",
                                    images_in_tag="10_stacked",
                                    reference_in_tag="10_stacked")
pipeline.add_module(psf_subtraction)

pipeline.run_module("PSF_subtraction")

from PynPoint.io_modules import WriteAsSingleFitsFile

write = WriteAsSingleFitsFile("PCA_result.fits",
                              name_in="wr",
                              data_tag="res_mean")
pipeline.add_module(write)
pipeline.run_module("wr")
