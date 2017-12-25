import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from PynPoint import Pypeline
from PynPoint.processing_modules import WaveletTimeDenoisingModule, CwtWaveletConfiguration

pipeline = Pypeline("/Users/markusbonse/Desktop/BetaPic",
                    "/Users/markusbonse/Desktop/BetaPic",
                    "/Users/markusbonse/Desktop/Results")

wv = CwtWaveletConfiguration(wavelet="dog",
                             wavelet_order=2,
                             keep_mean=False,
                             resolution=0.5)

time_denoising = WaveletTimeDenoisingModule(wv,
                                            name_in="time_denoising",
                                            image_in_tag="10_stacked",
                                            image_out_tag="star_arr_denoised",
                                            denoising_threshold=1.0,
                                            padding="zero",
                                            median_filter=True,
                                            threshold_function="soft")

pipeline.add_module(time_denoising)

pipeline.run_module("time_denoising")
