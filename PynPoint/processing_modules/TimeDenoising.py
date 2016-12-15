from statsmodels.robust import mad
import numpy as np
import pywt
from PynPoint.util import WaveletAnalysisCapsule
from copy import deepcopy

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
                 padding="zero",
                 num_rows_in_memory=40):
        """
        :type image_in_tag: list and str
        :type denoising_threshold: list and float
        """

        super(WaveletTimeDenoisingModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        # Ports
        if type(denoising_threshold) is list:
            assert type(image_out_tag) is list
            assert len(image_out_tag) == len(denoising_threshold)

            self.m_list_mode = True
            self.m_port_dict = dict()

            for i in range(len(image_out_tag)):
                self.m_port_dict[denoising_threshold[i]] = self.add_output_port(image_out_tag[i])

            # create tmp data port
            self.m_tmp_data_port_denoising = self.add_output_port("tmp_data_port_denoising2")
            self.m_tmp_data_port_denoising_in = self.add_input_port("tmp_data_port_denoising2")

        elif type(denoising_threshold) is float and type(image_in_tag) is str:
            self.m_list_mode = False
            self.m_image_out_port = self.add_output_port(image_out_tag)

        else:
            raise ValueError("image_in_tag needs to be a list or a string.")

        self.m_num_rows_in_memory = num_rows_in_memory

        # Parameters
        self.m_wavelet_configuration = wavelet_configuration
        self.m_denoising_threshold = denoising_threshold
        assert padding in ["zeros", "const_mean"]
        self.m_padding = padding  # TODO check paddings

    def run(self):

        if type(self.m_wavelet_configuration) is DwtWaveletConfiguration:
            # use DWT denoising
            if self.m_padding == "const_mean":
                self.m_padding = "constant"

            def denoise_line_in_time(signal_in):
                coef = pywt.wavedec(signal_in,
                                    wavelet=self.m_wavelet_configuration.m_wavelet,
                                    level=6,
                                    mode=self.m_padding)

                sigma = mad(coef[-1])
                uthresh = sigma * np.sqrt(2 * np.log(len(signal_in)))\

                if self.m_list_mode:
                    tmp_res = []
                    for threshold in self.m_denoising_threshold:
                        current_threshold = threshold * uthresh
                        tmp_denoised = deepcopy(coef[:])
                        tmp_denoised[1:] = (pywt.threshold(i, value=current_threshold, mode="soft")
                                            for i in tmp_denoised[1:])

                        tmp_res += list(pywt.waverec(tmp_denoised,
                                                     wavelet=self.m_wavelet_configuration.m_wavelet,
                                                     mode=self.m_padding))
                    return np.asarray(tmp_res)

                else:
                    threshold = uthresh * self.m_denoising_threshold
                    denoised = coef[:]
                    denoised[1:] = (pywt.threshold(i, value=threshold, mode="soft")
                                    for i in denoised[1:])
                    return pywt.waverec(denoised,
                                        wavelet=self.m_wavelet_configuration.m_wavelet,
                                        mode=self.m_padding)

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

                def denoise_one_thresold(capsule_in,
                                         threshold_in):
                    capsule_in.denoise_spectrum_universal_threshold(
                        padded_input_signal=True,
                        threshold=threshold_in)

                    capsule_in.update_signal()

                    # remove padding
                    res_signal = capsule_in.get_signal()
                    res_signal = res_signal[line_size: -line_size]

                    if self.m_wavelet_configuration.m_keep_mean:
                        mean_after = np.mean(res_signal)
                        diff = mean - mean_after
                        res_signal += np.ones(len(res_signal)) * diff

                    return res_signal

                if self.m_list_mode:
                    tmp_res = []
                    for threshold in self.m_denoising_threshold:
                        tmp_cwt_capsule = deepcopy(cwt_capsule)
                        tmp_res += list(denoise_one_thresold(tmp_cwt_capsule,
                                                             threshold))
                    return np.asarray(tmp_res)

                else:
                    return denoise_one_thresold(cwt_capsule,
                                                self.m_denoising_threshold)

        else:
            return

        if self.m_list_mode:
            # Calculate Results
            self.apply_function_to_line_in_time_multi_processing(denoise_line_in_time,
                                                                 self.m_image_in_port,
                                                                 self.m_tmp_data_port_denoising,
                                                                 num_rows_in_memory=
                                                                 self.m_num_rows_in_memory)
            print "Finished analyzing. Start splitting ..."

            tmp_num_elements_per_threshold = self.m_image_in_port.get_shape()[0]
            for i in range(0, len(self.m_denoising_threshold), 1):
                tmp_threshold = self.m_denoising_threshold[i]
                tmp_port = self.m_port_dict[tmp_threshold]

                print "splitting part "+str(i+1) + " of " + str(len(self.m_denoising_threshold)) \
                      + " parts"

                tmp_port.set_all(
                    self.m_tmp_data_port_denoising_in[(i + 0) * tmp_num_elements_per_threshold:
                                                      (i + 1) * tmp_num_elements_per_threshold,
                                                      :, :])
                # TODO create parameter for num frames in memory

                tmp_port.copy_attributes_from_input_port(self.m_image_in_port)
                tmp_port.add_history_information("Wavelet time denoising",
                                                 "threshold " + str(tmp_threshold))

            # clean up tmp data port
            self.m_tmp_data_port_denoising.del_all_attributes()
            self.m_tmp_data_port_denoising.del_all_data()
            self.m_image_in_port.close_port()

        else:
            self.apply_function_to_line_in_time_multi_processing(denoise_line_in_time,
                                                                 self.m_image_in_port,
                                                                 self.m_image_out_port,
                                                                 num_rows_in_memory=
                                                                 self.m_num_rows_in_memory)

            self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
            self.m_image_out_port.add_history_information("Wavelet time denoising",
                                                          "threshold " +
                                                          str(self.m_denoising_threshold))

            self.m_image_out_port.close_port()
