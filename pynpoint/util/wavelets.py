"""
Wrapper utils for the wavelet functions for the mlpy cwt implementation (see continous.py)
"""

from __future__ import absolute_import

import numpy as np

from numba import jit
from scipy.special import gamma, hermite
from scipy.signal import medfilt
from statsmodels.robust import mad
from six.moves import range

from pynpoint.util.continuous import autoscales, cwt, icwt
# from pynpoint.util.continuous import fourier_from_scales


@jit(cache=True)
def _fast_zeros(soft,
                spectrum,
                uthresh):
    """
    Fast numba method to modify values in the wavelet space by using a hard or soft threshold
    function.

    Parameters
    ----------
    soft : bool
        If True soft the threshold function will be used, otherwise a hard threshold is applied.
    spectrum : numpy.ndarray
        The input 2D wavelet space.
    uthresh : float
        Threshold used by the threshold function.

    Returns
    -------
    numpy.ndarray
        Modified spectrum.
    """

    if soft:
        for i in range(0, spectrum.shape[0], 1):
            for j in range(0, spectrum.shape[1], 1):
                tmp_value = spectrum[i, j].real
                if abs(spectrum[i, j]) > uthresh:
                    spectrum[i, j] = np.sign(tmp_value) * (abs(tmp_value) - uthresh)
                else:
                    spectrum[i, j] = 0
    else:
        for i in range(0, spectrum.shape[0], 1):
            for j in range(0, spectrum.shape[1], 1):
                if abs(spectrum[i, j]) < uthresh:
                    spectrum[i, j] = 0

    return spectrum


class WaveletAnalysisCapsule(object):
    """
    Capsule class to process one 1d time series using the CWT and wavelet de-nosing by wavelet
    shrinkage.
    """

    def __init__(self,
                 signal_in,
                 wavelet_in='dog',
                 order=2,
                 padding="none",
                 frequency_resolution=0.5):
        """
        Constructor of the WaveletAnalysisCapsule.

        Parameters
        ----------
        signal_in : numpy.ndarray
            1D input signal.
        wavelet_in : str
            Wavelet function ("dog" or "morlet").
        order : int
            Order of the wavelet function.
        padding : str
            Padding method ("zero", "mirror", or "none").
        frequency_resolution : float
            Wavelet space resolution in scale/frequency.

        Returns
        -------
        NoneType
            None
        """

        # save input data
        self.m_supported_wavelets = ['dog', 'morlet']

        # check supported wavelets
        if wavelet_in not in self.m_supported_wavelets:
            raise ValueError('Wavelet ' + str(wavelet_in) + ' is not supported')

        if wavelet_in == 'dog':
            self._m_c_reconstructions = {2: 3.5987,
                                         4: 2.4014,
                                         6: 1.9212,
                                         8: 1.6467,
                                         12: 1.3307,
                                         16: 1.1464,
                                         20: 1.0222,
                                         30: 0.8312,
                                         40: 0.7183,
                                         60: 0.5853}
        elif wavelet_in == 'morlet':
            self._m_c_reconstructions = {5: 0.9484,
                                         6: 0.7784,
                                         7: 0.6616,
                                         8: 0.5758,
                                         10: 0.4579,
                                         12: 0.3804,
                                         14: 0.3254,
                                         16: 0.2844,
                                         20: 0.2272}
        self.m_wavelet = wavelet_in

        if padding not in ["none", "zero", "mirror"]:
            raise ValueError("Padding can only be none, zero or mirror")

        self._m_data = signal_in - np.ones(len(signal_in)) * np.mean(signal_in)
        self.m_padding = padding
        self.__pad_signal()
        self._m_data_size = len(self._m_data)
        self._m_data_mean = np.mean(signal_in)

        if order not in self._m_c_reconstructions:
            raise ValueError('Wavelet ' + str(wavelet_in) + ' does not support order '
                             + str(order) + ". \n Only orders: " +
                             str(sorted(self._m_c_reconstructions.keys())).strip('[]') +
                             " are supported")
        self.m_order = order
        self._m_c_final_reconstruction = self._m_c_reconstructions[order]

        # create scales for wavelet transform
        self._m_scales = autoscales(N=self._m_data_size,
                                    dt=1,
                                    dj=frequency_resolution,
                                    wf=wavelet_in,
                                    p=order)

        self._m_number_of_scales = len(self._m_scales)
        self._m_frequency_resolution = frequency_resolution

        self.m_spectrum = None

    # --- functions for reconstruction value
    @staticmethod
    def _morlet_function(omega0,
                         x_in):
        """
        Returns
        -------
        numpy.complex128
            Morlet function.
        """

        return np.pi**(-0.25) * np.exp(1j * omega0 * x_in) * np.exp(-x_in**2/2.0)

    @staticmethod
    def _dog_function(order,
                      x_in):
        """
        Returns
        -------
        float
            DOG function.
        """

        p_hpoly = hermite(order)[int(x_in / np.power(2, 0.5))]
        herm = p_hpoly / (np.power(2, float(order) / 2))

        return ((-1)**(order+1)) / np.sqrt(gamma(order + 0.5)) * herm

    def __pad_signal(self):
        """
        Returns
        -------
        NoneType
            None
        """

        padding_length = int(len(self._m_data) * 0.5)

        if self.m_padding == "zero":
            new_data = np.append(self._m_data, np.zeros(padding_length, dtype=np.float64))
            self._m_data = np.append(np.zeros(padding_length, dtype=np.float64), new_data)

        elif self.m_padding == "mirror":
            left_half_signal = self._m_data[:padding_length]
            right_half_signal = self._m_data[padding_length:]
            new_data = np.append(self._m_data, right_half_signal[::-1])
            self._m_data = np.append(left_half_signal[::-1], new_data)

    def __compute_reconstruction_factor(self):
        """
        Computes the reconstruction factor.

        Returns
        -------
        float
            Reconstruction factor.
        """

        dj = self._m_frequency_resolution
        wavelet = self.m_wavelet
        order = self.m_order

        if wavelet == 'morlet':
            zero_function = self._morlet_function(order, 0)
        else:
            zero_function = self._dog_function(order, 0)

        c_delta = self._m_c_final_reconstruction

        reconstruction_factor = dj/(c_delta * zero_function)

        return reconstruction_factor.real

    def compute_cwt(self):
        """
        Compute the wavelet space of the given input signal.

        Returns
        -------
        NoneType
            None
        """

        self.m_spectrum = cwt(self._m_data,
                              dt=1,
                              scales=self._m_scales,
                              wf=self.m_wavelet,
                              p=self.m_order)

    def update_signal(self):
        """
        Updates the internal signal by the reconstruction of the current wavelet space.

        Returns
        -------
        NoneType
            None
        """

        self._m_data = icwt(self.m_spectrum, scales=self._m_scales)
        reconstruction_factor = self.__compute_reconstruction_factor()
        self._m_data *= reconstruction_factor

    def denoise_spectrum(self,
                         soft=False):
        """
        Applies wavelet shrinkage on the current wavelet space (m_spectrum) by either a hard of
        soft threshold function.

        Parameters
        ----------
        soft : bool
            If True a soft threshold is used, hard otherwise.

        Returns
        -------
        NoneType
            None
        """

        if self.m_padding != "none":
            # Python 2
            # noise_length_4 = len(self._m_data) / 4
            # Python 2+3?
            noise_length_4 = len(self._m_data) // 4
            noise_spectrum = self.m_spectrum[0, noise_length_4: (noise_length_4 * 3)].real

        else:
            noise_spectrum = self.m_spectrum[0, :].real

        sigma = mad(noise_spectrum)
        uthresh = sigma*np.sqrt(2.0*np.log(len(noise_spectrum)))

        self.m_spectrum = _fast_zeros(soft, self.m_spectrum, uthresh)

    def median_filter(self):
        """
        Applies a median filter on the internal 1d signal. Can be useful for cosmic ray correction
        after temporal de-noising

        Returns
        -------
        NoneType
            None
        """

        self._m_data = medfilt(self._m_data, 19)

    def get_signal(self):
        """
        Returns the current version of the 1d signal. Use update_signal() in advance in order to get
        the current reconstruction of the wavelet space. Removes padded values as well.

        Returns
        -------
        numpy.ndarray
            Current version of the 1D signal.
        """

        tmp_data = self._m_data + np.ones(len(self._m_data)) * self._m_data_mean

        if self.m_padding == "none":
            return tmp_data

        # Python 2
        # return tmp_data[len(self._m_data) / 4: 3 * (len(self._m_data) / 4)]

        # Python 2+3?
        return tmp_data[len(self._m_data) // 4: 3 * (len(self._m_data) // 4)]

    # def __transform_period(self,
    #                        period):
    #
    #     tmp_y = fourier_from_scales(self._m_scales,
    #                                 self.m_wavelet,
    #                                 self.m_order)
    #
    #     def __transformation(x):
    #         return np.log2(x + 1) * tmp_y[-1] / np.log2(tmp_y[-1] + 1)
    #
    #     cutoff_scaled = __transformation(period)
    #
    #     scale_new = tmp_y[-1] - tmp_y[0]
    #     scale_old = self.m_spectrum.shape[0]
    #
    #     factor = scale_old / scale_new
    #     cutoff_scaled *= factor
    #
    #     return cutoff_scaled

    # ----- plotting functions --------

    # def __plot_or_save_spectrum(self):
    #     plt.close()
    #
    #     plt.figure(figsize=(8, 6))
    #     plt.subplot(1, 1, 1)
    #
    #     tmp_y = fourier_from_scales(self._m_scales,
    #                                 self.m_wavelet,
    #                                 self.m_order)
    #
    #     tmp_x = np.arange(0, self._m_data_size + 1, 1)
    #
    #     scaled_spec = copy.deepcopy(self.m_spectrum.real)
    #     for i, _ in enumerate(scaled_spec):
    #         scaled_spec[i] /= np.sqrt(self._m_scales[i])
    #
    #     plt.imshow(abs(scaled_spec),
    #                aspect='auto',
    #                extent=[tmp_x[0],
    #                        tmp_x[-1],
    #                        tmp_y[0],
    #                        tmp_y[-1]],
    #                cmap=plt.get_cmap("gist_ncar"),
    #                origin='lower')
    #
    #     # COI first part (only for DOG) with padding
    #
    #     inner_frequency = 2.*np.pi/np.sqrt(self.m_order + 0.5)
    #     coi = np.append(np.zeros(len(tmp_x)/4),
    #                     tmp_x[0:len(tmp_x) / 4])
    #     coi = np.append(coi,
    #                     tmp_x[0:len(tmp_x) / 4][::-1])
    #     coi = np.append(coi,
    #                     np.zeros(len(tmp_x) / 4))
    #
    #     plt.plot(np.arange(0, len(coi), 1.0),
    #              inner_frequency * coi / np.sqrt(2),
    #              color="white")
    #
    #     plt.ylim([tmp_y[0],
    #               tmp_y[-1]])
    #
    #     plt.fill_between(np.arange(0, len(coi), 1.0),
    #                      inner_frequency * coi / np.sqrt(2),
    #                      np.ones(len(coi)) * tmp_y[-1],
    #                      facecolor="none",
    #                      edgecolor='white',
    #                      alpha=0.4,
    #                      hatch="x")
    #
    #     plt.yscale('log', basey=2)
    #     plt.ylabel("Period in [s]")
    #     plt.xlabel("Time in [s]")
    #     plt.title("Spectrum computed with CWT using '" + str(self.m_wavelet) +
    #               "' wavelet of order " + str(self.m_order))
    #
    # def plot_spectrum(self):
    #     """
    #     Shows a plot of the current wavelet space.
    #     :return: None
    #     """
    #
    #     self.__plot_or_save_spectrum()
    #     plt.show()
    #
    # def save_spectrum(self,
    #                   location):
    #     """
    #     Saves a plot of the current wavelet space to a given location.
    #     :param location: Save location
    #     :type location: str
    #     :return: None
    #     """
    #     self.__plot_or_save_spectrum()
    #     plt.savefig(location)
    #     plt.close()
    #
    # def __plot_or_save_signal(self):
    #     plt.close()
    #     plt.plot(self._m_data)
    #     plt.title("Signal")
    #     plt.ylabel("Value of the function")
    #     plt.xlim([0, self._m_data_size])
    #     plt.xlabel("Time in [s]")
    #
    # def plot_signal(self):
    #     """
    #     Plot the current signal.
    #     :return: None
    #     """
    #     self.__plot_or_save_signal()
    #     plt.show()
    #
    # def save_signal(self,
    #                 location):
    #     """
    #     Saves a plot of the current signal to a given location.
    #     :param location: Save location
    #     :type location: str
    #     :return: None
    #     """
    #     self.__plot_or_save_signal()
    #     plt.savefig(location)
