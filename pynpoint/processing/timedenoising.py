"""
Continuous wavelet transform (CWT) and discrete wavelet transform (DWT) denoising for speckle
suppression in the time domain. The module can be used as additional preprocessing step. See
Bonse et al. (arXiv:1804.05063) more information.
"""

from typing import Union

import pywt

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.apply_func import cwt_denoise_line_in_time, dwt_denoise_line_in_time, \
                                     normalization


class CwtWaveletConfiguration:
    """
    Configuration capsule for a CWT based time denoising. Standard configuration as in the
    original paper.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 wavelet: str = 'dog',
                 wavelet_order: int = 2,
                 keep_mean: bool = False,
                 resolution: float = 0.5) -> None:
        """
        Parameters
        ----------
        wavelet : str
            Wavelet.
        wavelet_order : int
            Wavelet order.
        keep_mean : bool
            Keep mean.
        resolution : float
            Resolution.

        Returns
        -------
        NoneType
            None
        """

        if wavelet not in ['dog', 'morlet']:
            raise ValueError('CWT supports only \'dog\' and \'morlet\' wavelets.')

        self.m_wavelet = wavelet
        self.m_wavelet_order = wavelet_order
        self.m_keep_mean = keep_mean
        self.m_resolution = resolution


class DwtWaveletConfiguration:
    """
    Configuration capsule for a DWT based time denoising. A cheap alternative of the CWT based
    wavelet denoising. However, the supported wavelets should perform worse compared to the
    CWT DOG wavelet.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 wavelet: str = 'db8') -> None:
        """
        Parameters
        ----------
        wavelet : str
            Wavelet.

        Returns
        -------
        NoneType
            None
        """

        # create list of supported wavelets
        supported = []
        for item in pywt.families():
            supported += pywt.wavelist(item)

        # check if wavelet is supported
        if wavelet not in supported:
            raise ValueError(f'DWT supports only {supported} as input wavelet.')

        self.m_wavelet = wavelet


class WaveletTimeDenoisingModule(ProcessingModule):
    """
    Pipeline module for speckle subtraction in the time domain by using CWT or DWT wavelet
    shrinkage. See Bonse et al. (arXiv:1804.05063) for details.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 wavelet_configuration: Union[CwtWaveletConfiguration, DwtWaveletConfiguration],
                 padding: str = 'zero',
                 median_filter: bool = False,
                 threshold_function: str = 'soft') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name for the pipeline module.
        image_in_tag : str
            Database tag with the input data.
        image_out_tag : str
            Database tag for the output data.
        wavelet_configuration : pynpoint.processing.timedenoising.CwtWaveletConfiguration or \
                                pynpoint.processing.timedenoising.DwtWaveletConfiguration
            Instance of :class:`~pynpoint.processing.timedenoising.DwtWaveletConfiguration` or
            :class:`~pynpoint.processing.timedenoising.CwtWaveletConfiguration` which contains the
            parameters for the wavelet transformation.
        padding : str
            Padding method (``'zero'``, ``'mirror'``, or ``'none'``).
        median_filter : bool
            Apply a median filter in time to remove outliers, for example due to cosmic rays.
        threshold_function : str
            Threshold function that is used for wavelet shrinkage in the wavelet space
            (``'soft'`` or ``'hard'``).

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_wavelet_configuration = wavelet_configuration
        self.m_median_filter = median_filter

        assert padding in ['zero', 'mirror', 'none']
        self.m_padding = padding

        assert threshold_function in ['soft', 'hard']
        self.m_threshold_function = threshold_function == 'soft'

        assert isinstance(wavelet_configuration,
                          (DwtWaveletConfiguration, CwtWaveletConfiguration))

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Applies the time denoising for the lines in time in parallel.

        Returns
        -------
        NoneType
            None
        """

        if isinstance(self.m_wavelet_configuration, DwtWaveletConfiguration):

            if self.m_padding == 'const_mean':
                self.m_padding = 'constant'

            if self.m_padding == 'none':
                self.m_padding = 'periodic'

            self.apply_function_in_time(dwt_denoise_line_in_time,
                                        self.m_image_in_port,
                                        self.m_image_out_port,
                                        func_args=(self.m_threshold_function,
                                                   self.m_padding,
                                                   self.m_wavelet_configuration))

        elif isinstance(self.m_wavelet_configuration, CwtWaveletConfiguration):

            self.apply_function_in_time(cwt_denoise_line_in_time,
                                        self.m_image_in_port,
                                        self.m_image_out_port,
                                        func_args=(self.m_threshold_function,
                                                   self.m_padding,
                                                   self.m_median_filter,
                                                   self.m_wavelet_configuration))

        if self.m_threshold_function:
            history = 'threshold_function = soft'
        else:
            history = 'threshold_function = hard'

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('WaveletTimeDenoisingModule', history)
        self.m_image_out_port.close_port()


class TimeNormalizationModule(ProcessingModule):
    """
    Pipeline module for normalization of global brightness variations of the detector. See Bonse
    et al. (arXiv:1804.05063) for details.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name for the pipeline module.
        image_in_tag : str
            Database tag with the input data.
        image_out_tag : str
            Database tag for the output data.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module.

        Returns
        -------
        NoneType
            None
        """

        self.apply_function_to_images(normalization,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Time normalization')

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('TimeNormalizationModule', 'normalization = median')
        self.m_image_out_port.close_port()
