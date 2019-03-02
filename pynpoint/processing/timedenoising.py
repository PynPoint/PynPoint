"""
Continuous wavelet transform (CWT) and discrete wavelet transform (DWT) denoising for speckle
suppression in the time domain. The module can be used as additional preprocessing step. See
Bonse et al. 2018 more information.
"""

from __future__ import absolute_import

import pywt
import numpy as np

from statsmodels.robust import mad

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.wavelets import WaveletAnalysisCapsule


class CwtWaveletConfiguration(object):
    """
    Configuration capsule for a CWT based time denoising. Standard configuration as in the
    original paper.
    """

    def __init__(self,
                 wavelet="dog",
                 wavelet_order=2,
                 keep_mean=False,
                 resolution=0.5):
        """
        :param wavelet: Wavelet.
        :type wavelet: str
        :param wavelet_order: Wavelet order.
        :type wavelet_order: int
        :param keep_mean: Keep mean.
        :type keep_mean: bool
        :param resolution: Resolution.
        :type resolution: float

        :return: None
        """

        if wavelet not in ["dog", "morlet"]:
            raise ValueError("CWT supports only 'dog' and 'morlet' wavelets.")

        self.m_wavelet = wavelet
        self.m_wavelet_order = wavelet_order
        self.m_keep_mean = keep_mean
        self.m_resolution = resolution


class DwtWaveletConfiguration(object):
    """
    Configuration capsule for a DWT based time denoising. A cheap alternative of the CWT based
    wavelet denoising. However, the supported wavelets should perform worse compared to the
    CWT DOG wavelet.
    """

    def __init__(self,
                 wavelet="db8"):
        """
        :param wavelet: Wavelet.
        :type wavelet: str

        :return: None
        """

        # create list of supported wavelets
        supported = []
        for family in pywt.families():
            supported += pywt.wavelist(family)

        # check if wavelet is supported
        if wavelet not in supported:
            raise ValueError("DWT supports only " + str(supported) + " as input wavelet.")

        self.m_wavelet = wavelet


class WaveletTimeDenoisingModule(ProcessingModule):
    """
    Module for speckle subtraction in the time domain by using CWT or DWT wavelet shrinkage
    (see Bonse et al. 2018).
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
        :type wavelet_configuration: pynpoint.processing.timedenoising.CwtWaveletConfiguration or
                                     pynpoint.processing.timedenoising.DwtWaveletConfiguration
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str
        :param padding: Padding method ("zero", "mirror", or "none").
        :type padding: str
        :param median_filter: If true a median filter in time is applied which removes outliers
                              in time like cosmic rays.
        :type median_filter: bool
        :param threshold_function: Threshold function used for wavelet shrinkage in the wavelet
                                   space ("soft" or "hard").
        :type threshold_function: str

        :return: None
        """

        super(WaveletTimeDenoisingModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_wavelet_configuration = wavelet_configuration
        self.m_median_filter = median_filter

        assert padding in ["zero", "mirror", "none"]
        self.m_padding = padding

        assert threshold_function in ["soft", "hard"]
        self.m_threshold_function = threshold_function == "soft"

    def run(self):
        """
        Run method of the module. Applies the time denoising for the lines in time in parallel.

        :return: None
        """

        if isinstance(self.m_wavelet_configuration, DwtWaveletConfiguration):

            if self.m_padding == "const_mean":
                self.m_padding = "constant"

            if self.m_padding == "none":
                self.m_padding = "periodic"

            def denoise_line_in_time(signal_in):
                """
                Definition of the temporal denoising for DWT.

                :param signal_in: 1D input signal.
                :type signal_in:

                :return: Multilevel 1D inverse discrete wavelet transform.
                :rtype: numpy.ndarray
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
                Definition of temporal denoising for CWT.

                :param signal_in: 1D input signal.
                :type signal_in:

                :return: 1D output signal.
                :rtype: numpy.ndarray
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

                return cwt_capsule.get_signal()

        else:
            return

        self.apply_function_in_time(denoise_line_in_time,
                                    self.m_image_in_port,
                                    self.m_image_out_port)

        if self.m_threshold_function:
            history = "threshold_function = soft"
        else:
            history = "threshold_function = hard"

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("WaveletTimeDenoisingModule", history)
        self.m_image_out_port.close_port()


class TimeNormalizationModule(ProcessingModule):
    """
    Module for normalization of global brightness variations of the detector
    (see Bonse et al. 2018).
    """

    def __init__(self,
                 name_in="normalization",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_normalized"):
        """
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str

        :return: None
        """

        super(TimeNormalizationModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module.

        :return: None
        """

        def _normalization(image_in):
            median = np.median(image_in)
            return image_in - median

        self.apply_function_to_images(_normalization,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running TimeNormalizationModule...")

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("TimeNormalizationModule", "normalization = median")
        self.m_image_out_port.close_port()
