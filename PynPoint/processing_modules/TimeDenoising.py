from statsmodels.robust import mad
import numpy as np
import pywt
from PynPoint.util import WaveletAnalysisCapsule

from PynPoint.core.Processing import ProcessingModule

import matplotlib.pyplot as plt


class WaveletConfiguration(object):

    def __init__(self,
                 wavelet):

        self.m_wavelet = wavelet


class CwtWaveletConfiguration(WaveletConfiguration):

    def __init__(self,
                 wavelet="dog",
                 wavelet_order=2,
                 keep_mean=False,
                 resolution=0.1):

        if not wavelet in ["dog", "morlet"]:
            raise ValueError("CWT supports only dog and morlet wavelets")

        super(CwtWaveletConfiguration, self).__init__(wavelet)
        self.m_wavelet_order = wavelet_order
        self.m_keep_mean = keep_mean
        self.m_resolution = resolution


class DwtWaveletConfiguration(WaveletConfiguration):

    def __init__(self,
                 wavelet="db8"):

        # create list of supported wavelets
        supported = []
        for family in pywt.families():
            supported += pywt.wavelist(family)

        # check if wavelet is supported
        if wavelet not in supported:
            raise ValueError("DWT supports only " + str(supported) + " as input wavelet")

        super(DwtWaveletConfiguration, self).__init__(wavelet)


class WaveletTimeDenoisingModule(ProcessingModule):

    def __init__(self,
                 wavelet_configuration,
                 name_in="time_denoising",
                 image_in_tag="star_arr",
                 image_out_tag="star_arr_denoised",
                 denoising_threshold=1.0,
                 padding="zero"):

        super(WaveletTimeDenoisingModule, self).__init__(name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        # Parameters
        self.m_wavelet_configuration = wavelet_configuration
        self.m_denoising_threshold = denoising_threshold
        self.m_padding = padding  # TODO check paddings

    def run(self):

        if type(self.m_wavelet_configuration) is DwtWaveletConfiguration:
            # use DWT denoising
            def denoise_line_in_time(signal_in):
                # TODO add padding
                coef = pywt.wavedec(signal_in,
                                    wavelet=self.m_wavelet_configuration.m_wavelet,
                                    level=6,
                                    mode="constant")

                sigma = mad(coef[-1])
                uthresh = sigma * np.sqrt(2 * np.log(len(signal_in))) * self.m_denoising_threshold
                denoised = coef[:]
                denoised[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in denoised[1:])
                return pywt.waverec(denoised,
                                    wavelet=self.m_wavelet_configuration.m_wavelet,
                                    mode="constant")

        elif type(self.m_wavelet_configuration) is CwtWaveletConfiguration:
            # use CWT denoising
            def denoise_line_in_time(signal_in):

                # we need have length of the signal as border right and left
                line_size = int(len(signal_in) * 0.5)
                if self.m_padding == "zeros":
                    line = np.zeros(line_size)
                elif self.m_padding == "const_mean":
                    line = np.ones(line_size)
                else:
                    return

                mean = np.mean(signal_in)
                border = line * mean
                signal_in = np.append(border, signal_in)
                signal_in = np.append(signal_in, border)

                cwt_capsule = WaveletAnalysisCapsule(
                    signal_in,
                    wavelet_in=self.m_wavelet_configuration.m_wavelet,
                    order=self.m_wavelet_configuration.m_wavelet_order,
                    frequency_resolution=self.m_wavelet_configuration.m_resolution)

                cwt_capsule.compute_cwt()

                cwt_capsule.denoise_spectrum_universal_threshold(
                    padded_input_signal=True,
                    threshold=self.m_denoising_threshold)

                cwt_capsule.update_signal()

                # remove padding
                res_signal = cwt_capsule.get_signal()
                res_signal = res_signal[line_size: -line_size]

                if self.m_wavelet_configuration.m_keep_mean:
                    mean_after = np.mean(res_signal)
                    diff = mean - mean_after
                    res_signal += np.ones(len(res_signal)) * diff

                return res_signal

        else:
            return

        self.apply_function_to_line_in_time_multi_processing(denoise_line_in_time,
                                                             self.m_image_in_port,
                                                             self.m_image_out_port)

        '''self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_history_information("Time denoising",
                                                      "Using Wavelet analysis")
        # TODO add more information here
        self.m_image_out_port.close_port()'''
