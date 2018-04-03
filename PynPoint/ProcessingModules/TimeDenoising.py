from copy import deepcopy

import pywt
import numpy as np

from statsmodels.robust import mad

from PynPoint.Util import WaveletAnalysisCapsule
from PynPoint.Core.Processing import ProcessingModule


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
                 median_filter=True,
                 threshold_function="soft"):
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
            self.m_tmp_data_port_denoising = self.add_output_port("tmp_data_port_denoising")
            self.m_tmp_data_port_denoising_in = self.add_input_port("tmp_data_port_denoising")

        elif type(denoising_threshold) is float and type(image_in_tag) is str:
            self.m_list_mode = False
            self.m_image_out_port = self.add_output_port(image_out_tag)

        else:
            raise ValueError("image_in_tag needs to be a list or a string.")

        # Parameters
        self.m_wavelet_configuration = wavelet_configuration
        self.m_denoising_threshold = denoising_threshold
        assert padding in ["zero", "mirror", "none"]
        self.m_padding = padding
        assert threshold_function in ["soft", "hard"]
        if threshold_function == "soft":
            self.m_threshold_function = True
        else:
            self.m_threshold_function = False
        self.m_median_filter = median_filter

    def run(self):

        if type(self.m_wavelet_configuration) is DwtWaveletConfiguration:
            # use DWT denoising
            if self.m_padding == "const_mean":
                self.m_padding = "constant"
            if self.m_padding == "none":
                self.m_padding = "periodic"

            def denoise_line_in_time(signal_in):
                coef = pywt.wavedec(signal_in,
                                    wavelet=self.m_wavelet_configuration.m_wavelet,
                                    level=6,
                                    mode=self.m_padding)

                sigma = mad(coef[-1])
                uthresh = sigma * np.sqrt(2 * np.log(len(signal_in)))

                if self.m_list_mode:
                    tmp_res = []
                    for threshold in self.m_denoising_threshold:
                        current_threshold = threshold * uthresh
                        tmp_denoised = deepcopy(coef[:])
                        tmp_denoised[1:] = (pywt.threshold(i,
                                                           value=current_threshold,
                                                           mode=self.m_threshold_function)
                                            for i in tmp_denoised[1:])

                        tmp_res += list(pywt.waverec(tmp_denoised,
                                                     wavelet=self.m_wavelet_configuration.m_wavelet,
                                                     mode=self.m_padding))
                    return np.asarray(tmp_res)

                else:
                    threshold = uthresh * self.m_denoising_threshold
                    denoised = coef[:]
                    denoised[1:] = (pywt.threshold(i,
                                                   value=threshold,
                                                   mode=self.m_threshold_function)
                                    for i in denoised[1:])
                    return pywt.waverec(denoised,
                                        wavelet=self.m_wavelet_configuration.m_wavelet,
                                        mode=self.m_padding)

        elif type(self.m_wavelet_configuration) is CwtWaveletConfiguration:
            # use CWT denoising
            def denoise_line_in_time(signal_in):

                cwt_capsule = WaveletAnalysisCapsule(
                    signal_in,
                    padding=self.m_padding,
                    wavelet_in=self.m_wavelet_configuration.m_wavelet,
                    order=self.m_wavelet_configuration.m_wavelet_order,
                    frequency_resolution=self.m_wavelet_configuration.m_resolution)

                cwt_capsule.compute_cwt()

                def denoise_one_threshold(capsule_in,
                                          threshold_in):

                    capsule_in.denoise_spectrum_universal_threshold(threshold=threshold_in,
                                                                    soft=self.m_threshold_function)

                    if self.m_median_filter:
                        capsule_in.median_filter()

                    capsule_in.update_signal()
                    res_signal = capsule_in.get_signal()

                    return res_signal

                if self.m_list_mode:
                    tmp_res = []
                    for threshold in self.m_denoising_threshold:
                        tmp_cwt_capsule = deepcopy(cwt_capsule)
                        tmp_res += list(denoise_one_threshold(tmp_cwt_capsule,
                                                              threshold))
                    return np.asarray(tmp_res)

                else:
                    return denoise_one_threshold(cwt_capsule,
                                                 self.m_denoising_threshold)

        else:
            return

        if self.m_list_mode:
            # Calculate Results
            self.apply_function_to_line_in_time_multi_processing(denoise_line_in_time,
                                                                 self.m_image_in_port,
                                                                 self.m_tmp_data_port_denoising)
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
                                                                 self.m_image_out_port)

            self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
            self.m_image_out_port.add_history_information("Wavelet time denoising",
                                                          "threshold " +
                                                          str(self.m_denoising_threshold))

            self.m_image_out_port.close_port()


class TimeNormalizationModule(ProcessingModule):

    def __init__(self,
                 name_in="normalization",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_normalized"):

        super(TimeNormalizationModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        def image_normalization(image_in):

            median = np.median(image_in)
            tmp_image = image_in - median

            return tmp_image

        self.apply_function_to_images(image_normalization,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running TimeNormalizationModule...")

        self.m_image_out_port.add_history_information("Frame normalization",
                                                      "using median")

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()
