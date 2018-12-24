"""
CWT based wavelet de-noising for speckle suppression in the time domain. The module acts as an
additional pre-processing step. For more information see Bonse et al. 2018.
"""

from __future__ import absolute_import

import pywt
import numpy as np

from statsmodels.robust import mad

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.wavelets import WaveletAnalysisCapsule


class CwtWaveletConfiguration(object):
    """
    Configuration capsule for a CWT based time de-noising. Standard configuration as in the original
    paper.
    """

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
    """
    Configuration capsule for a DWT based time de-noising. A cheap alternative of the CWT based
    wavelet de-noising. However, the supported wavelets should perform worse compared to the
    CWT DOG wavelet.
    """

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
    """
    Module for speckle subtraction in time domain used CWT or DWT wavelet shrinkage.
    See Bonse et al 2018.
    """

    def __init__(self,
                 wavelet_configuration,
                 name_in="time_denoising",
                 image_in_tag="star_arr",
                 image_out_tag="star_arr_denoised",
                 padding="zero",
                 median_filter=False,
                 threshold_function="soft"):
        """
        Constructor of WaveletTimeDenoisingModule.

        :param wavelet_configuration: Instance of DwtWaveletConfiguration or CwtWaveletConfiguration
                                      which gives the parameters of the wavelet transformation to be
                                      used.
        :param name_in: Module name
        :param image_in_tag: Input tag in the central database
        :param image_out_tag: Output tag in the central database
        :param padding: Padding strategy can be (zero, mirror and none)
        :param median_filter: If true a median filter in time gets applied which removes outliers
                              in time like cosmic rays
        :param threshold_function: Threshold function used for wavelet shrinkage in the wavelet
                                   space. Can be soft or hard.

        :return: None
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
        """
        Run method of the module. Applies time de-noising using multiprocessing parallel on all
        lines in time.
        :return: None
        """

        if isinstance(self.m_wavelet_configuration, DwtWaveletConfiguration):
            # use DWT denoising
            if self.m_padding == "const_mean":
                self.m_padding = "constant"
            if self.m_padding == "none":
                self.m_padding = "periodic"

            def denoise_line_in_time(signal_in):
                """
                Definition of temporal de-noising for DWT
                :param signal_in: 1d signal
                :return:
                """

                if self.m_threshold_function:
                    threshold_mode = "soft"
                else:
                    threshold_mode = "hard"

                coef = pywt.wavedec(signal_in,
                                    wavelet=self.m_wavelet_configuration.m_wavelet,
                                    level=None,
                                    mode=self.m_padding)

                sigma = mad(coef[-1])
                threshold = sigma * np.sqrt(2 * np.log(len(signal_in)))

                denoised = coef[:]
                denoised[1:] = (pywt.threshold(i,
                                               value=threshold,
                                               mode=threshold_mode)
                                for i in denoised[1:])
                return pywt.waverec(denoised,
                                    wavelet=self.m_wavelet_configuration.m_wavelet,
                                    mode=self.m_padding)

        elif isinstance(self.m_wavelet_configuration, CwtWaveletConfiguration):
            def denoise_line_in_time(signal_in):
                """
                Definition of temporal de-noising for CWT
                :param signal_in: 1d signal
                :return:
                """

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

        self.apply_function_in_time(denoise_line_in_time,
                                    self.m_image_in_port,
                                    self.m_image_out_port)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_history_information("Wavelet time de-noising",
                                                      "universal threshold")

        self.m_image_out_port.close_port()


class TimeNormalizationModule(ProcessingModule):
    """
    Module for normalization of global brightness variations of the detector (see Bonse et al 2018.)
    """

    def __init__(self,
                 name_in="normalization",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_normalized"):

        super(TimeNormalizationModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module.
        :return: None
        """

        def image_normalization(image_in):
            """
            Subtract the median pixel value from the current image
            """

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
