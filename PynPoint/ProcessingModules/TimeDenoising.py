import pywt
import numpy as np

from statsmodels.robust import mad

from PynPoint.Util import WaveletAnalysisCapsule
from PynPoint.Core.Processing import ProcessingModule


class CwtWaveletConfiguration(object):

    def __init__(self,
                 wavelet="dog",
                 wavelet_order=2,
                 keep_mean=False,
                 resolution=0.5):

        if wavelet not in ["dog", "morlet"]:
            raise ValueError("CWT supports only dog and morlet wavelets")

        self.m_wavelet = wavelet
        self.m_wavelet_order = wavelet_order
        self.m_keep_mean = keep_mean
        self.m_resolution = resolution


class DwtWaveletConfiguration(object):

    def __init__(self,
                 wavelet="db8"):

        # create list of supported wavelets
        supported = []
        for family in pywt.families():
            supported += pywt.wavelist(family)

        # check if wavelet is supported
        if wavelet not in supported:
            raise ValueError("DWT supports only " + str(supported) + " as input wavelet")

        self.m_wavelet = wavelet


class WaveletTimeDenoisingModule(ProcessingModule):

    def __init__(self,
                 wavelet_configuration,
                 name_in="time_denoising",
                 image_in_tag="star_arr",
                 image_out_tag="star_arr_denoised",
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
        self.m_image_out_port = self.add_output_port(image_out_tag)

        # Parameters
        self.m_wavelet_configuration = wavelet_configuration

        assert padding in ["zero", "mirror", "none"]
        self.m_padding = padding

        assert threshold_function in ["soft", "hard"]
        self.m_threshold_function = threshold_function == "soft"

        self.m_median_filter = median_filter

    def run(self):

        if isinstance(self.m_wavelet_configuration, DwtWaveletConfiguration):
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
                threshold = sigma * np.sqrt(2 * np.log(len(signal_in)))

                denoised = coef[:]
                denoised[1:] = (pywt.threshold(i,
                                               value=threshold,
                                               mode=self.m_threshold_function)
                                for i in denoised[1:])
                return pywt.waverec(denoised,
                                    wavelet=self.m_wavelet_configuration.m_wavelet,
                                    mode=self.m_padding)

        elif isinstance(self.m_wavelet_configuration, CwtWaveletConfiguration):
            # use CWT denoising
            def denoise_line_in_time(signal_in):

                cwt_capsule = WaveletAnalysisCapsule(
                    signal_in,
                    padding=self.m_padding,
                    wavelet_in=self.m_wavelet_configuration.m_wavelet,
                    order=self.m_wavelet_configuration.m_wavelet_order,
                    frequency_resolution=self.m_wavelet_configuration.m_resolution)

                cwt_capsule.compute_cwt()
                cwt_capsule.denoise_spectrum(soft=self.m_threshold_function)

                if self.m_median_filter:
                    cwt_capsule.median_filter()

                cwt_capsule.update_signal()
                res_signal = cwt_capsule.get_signal()

                return res_signal

        else:
            return

        self.apply_function_to_line_in_time_multi_processing(denoise_line_in_time,
                                                             self.m_image_in_port,
                                                             self.m_image_out_port)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_history_information("Wavelet time de-noising",
                                                      "universal threshold")

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
