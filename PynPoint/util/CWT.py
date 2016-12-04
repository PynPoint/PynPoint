import numpy as np
#import mlpy.wavelet as wave
import continous as wave
from scipy.special import gamma, hermite
from statsmodels.robust import mad
from numba import jit
import copy

import matplotlib.pyplot as plt

# --- Wavelet analysis Capsule ---------
# TODO: Documentation

@jit(cache=True)
def _fast_zeros(soft,
                spectrum,
                uthresh):
        if soft:
            for i in range(0, spectrum.shape[0], 1):
                for j in range(0, spectrum.shape[1], 1):
                    if 1 - uthresh/abs(spectrum[i, j].real) < 0:
                        spectrum[i, j] *= 0
                    else:
                        spectrum[i, j] *= 1 - uthresh/abs(spectrum[i, j].real)
        else:
            for i in range(0, spectrum.shape[0], 1):
                for j in range(0, spectrum.shape[1], 1):
                    if abs(spectrum[i, j]) < uthresh:
                        spectrum[i, j] = 0

        return spectrum


class WaveletAnalysisCapsule:

    def __init__(self,
                 signal_in,
                 wavelet_in = 'dog',
                 order = 2,
                 frequency_resolution = 0.1):

        # save input data
        self.__m_supported_wavelets = ['dog', 'morlet']

        # check supported wavelets
        if not (wavelet_in in self.__m_supported_wavelets):
            raise ValueError('Wavelet ' + str(wavelet_in) + ' is not supported')

        if wavelet_in == 'dog':
            self.__m_C_reconstructions = {2: 3.5987,
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
            self.__m_C_reconstructions = {5: 0.9484,
                                          6: 0.7784,
                                          7: 0.6616,
                                          8: 0.5758,
                                          10: 0.4579,
                                          12: 0.3804,
                                          14: 0.3254,
                                          16: 0.2844,
                                          20: 0.2272}
        self.__m_data = signal_in
        self.__m_data_size = len(signal_in)
        self.__m_wavelet = wavelet_in

        if not order in self.__m_C_reconstructions:
            raise ValueError('Wavelet ' + str(wavelet_in) + ' does not support order ' + str(order) +
                             ". \n Only orders: " + str(sorted(self.__m_C_reconstructions.keys())).strip('[]') +
                             " are supported")
        self.__m_order = order
        self.__m_C_final_reconstruction = self.__m_C_reconstructions[order]

        # create scales for wavelet transform
        self.__m_scales = wave.autoscales(N = self.__m_data_size,
                                          dt=1,
                                          dj=frequency_resolution,
                                          wf=wavelet_in,
                                          p=order)

        self.__m_number_of_scales = len(self.__m_scales)
        self.__m_frequency_resolution = frequency_resolution

        self.__m_spectrum = None
        return

    # --- functions for reconstruction value
    @staticmethod
    def __morlet_function(omega0,
                          x):
        return np.pi**(-0.25) * np.exp(1j * omega0 * x) * np.exp(-x**2/2.0)

    @staticmethod
    def __dog_function(order,
                       x):
        pHpoly = hermite(order)[int(x / np.power(2, 0.5))]
        herm = pHpoly / (np.power(2, float(order) / 2))
        return ((-1)**(order+1)) / np.sqrt(gamma(order + 0.5)) * herm

    def __compute_reconstruction_factor(self):
        dj = self.__m_frequency_resolution
        wavelet = self.__m_wavelet
        order = self.__m_order

        if wavelet == 'morlet':
            zero_function = self.__morlet_function(order, 0)
        else:
            zero_function = self.__dog_function(order, 0)

        c_delta = self.__m_C_final_reconstruction

        reconstruction_factor = dj/(c_delta * zero_function)
        return reconstruction_factor.real

    def compute_cwt(self):
        self.__m_spectrum = wave.cwt(self.__m_data,
                                     dt=1,
                                     scales=self.__m_scales,
                                     wf=self.__m_wavelet,
                                     p=self.__m_order)

    def update_signal(self):
        self.__m_data = wave.icwt(self.__m_spectrum,
                                  dt=1,
                                  scales=self.__m_scales,
                                  wf=self.__m_wavelet,
                                  p=self.__m_order)
        reconstruction_factor = self.__compute_reconstruction_factor()
        self.__m_data *= reconstruction_factor

    def filter_periods(self,
                       cutoff):

        tmp_y = wave.fourier_from_scales(self.__m_scales, self.__m_wavelet,self.__m_order)

        def transformation(x):
            return np.log2(x + 1) * tmp_y[-1] / np.log2(tmp_y[-1] + 1)

        cutoff_scaled = transformation(cutoff)

        scale_new = tmp_y[-1] - tmp_y[0]
        scale_old = self.__m_spectrum.shape[0]

        factor = scale_old / scale_new

        cutoff_scaled *= factor

        self.__m_spectrum[cutoff_scaled::] = 0

    def block_period(self,
                     period,
                     cutoff_range,
                     inverse = False):

        # TODO handle Error cases with to large or to small period and range

        start_period = period - cutoff_range
        end_period = period + cutoff_range

        tmp_y = wave.fourier_from_scales(self.__m_scales, self.__m_wavelet,self.__m_order)

        def transformation(x):
            return np.log2(x + 1) * tmp_y[-1] / np.log2(tmp_y[-1] + 1)

        start_period_scaled = transformation(start_period)
        end_period_scaled = transformation(end_period)

        scale_new = tmp_y[-1] - tmp_y[0]
        scale_old = self.__m_spectrum.shape[0]

        factor = scale_old / scale_new

        start_period_scaled *= factor
        end_period_scaled *= factor

        # delete Region
        if inverse:
            self.__m_spectrum[0:start_period_scaled] = 0
            self.__m_spectrum[end_period_scaled::] = 0
        else:
            self.__m_spectrum[start_period_scaled:end_period_scaled] = 0

    def denoise_spectrum_universal_threshold(self,
                                             threshold=1.0,
                                             soft=False,
                                             padded_input_signal=False):

        if padded_input_signal:
            noise = self.__m_spectrum[0, :].real
            noise_length_3 = len(noise)/3
            noise = noise[noise_length_3: (noise_length_3*2)]
            sigma = mad(noise)
        else:
            sigma = mad(self.__m_spectrum[0, :].real)
        uthresh = sigma*np.sqrt(2.0*np.log(self.__m_data_size)) * threshold

        self.__m_spectrum = _fast_zeros(soft,
                                        self.__m_spectrum,
                                        uthresh)

    def get_signal(self):
        return self.__m_data

    # ----- plotting functions --------

    def __plot_or_save_spectrum(self):
        plt.close()
        tmp_y = wave.fourier_from_scales(self.__m_scales, self.__m_wavelet,self.__m_order)
        tmp_x = np.arange(0, self.__m_data_size, 1.0)

        scaled_spec = copy.deepcopy(self.__m_spectrum.real)
        for i in range(len(scaled_spec)):
            scaled_spec[i] /= np.sqrt(self.__m_scales[i])

        plt.imshow(abs(scaled_spec),
                   aspect='auto',
                   extent=[tmp_x[0],
                           tmp_x[-1],
                           tmp_y[0],
                           tmp_y[-1]],
                   cmap=plt.get_cmap("gray"),
                   origin='lower')

        plt.yscale('log', basey=2)
        plt.ylabel("Period in [s]")
        plt.xlabel("Time in [s]")
        plt.title("Spectrum computed with CWT using '" + str(self.__m_wavelet) +
                  "' wavelet of order " + str(self.__m_order))

    def plot_spectrum(self):
        self.__plot_or_save_spectrum()
        plt.show()

    def save_spectrum(self,
                      location):
        self.__plot_or_save_spectrum()
        plt.savefig(location)
        plt.close()

    def __plot_or_save_signal(self):
        plt.close()
        plt.plot(self.__m_data)
        plt.title("Signal")
        plt.ylabel("Value of the function")
        plt.xlim([0, self.__m_data_size])
        plt.xlabel("Time in [s]")

    def plot_signal(self):
        self.__plot_or_save_signal()
        plt.show()

    def save_signal(self,
                    location):
        self.__plot_or_save_signal()
        plt.savefig(location)

    # ---------------------------------
