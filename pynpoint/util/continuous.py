# This code is written by Davide Albanese, <albanese@fbk.eu>
# (C) 2011 mlpy Developers.

# See: Practical Guide to Wavelet Analysis - C. Torrence and G. P. Compo.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from scipy.special import gamma


def normalization(s, dt):
    """"
    Parameters
    ----------
    s : numpy.ndarray
        Scales.
    dt : int
        Time step.

    Returns
    -------
    numpy.ndarray
        Normalized data.
    """

    return np.sqrt((2 * np.pi * s) / dt)


def morletft(s, w, w0, dt):
    """"
    Fourier transformed morlet function.

    Parameters
    ----------
    s : numpy.ndarray
        Scales.
    w : numpy.ndarray
        Angular frequencies.
    w0 : int
        Omega0 frequency.
    dt : int
        Time step.

    Returns
    -------
    numpy.ndarray
        Normalized Fourier transformed morlet function
    """

    p = 0.75112554446494251  # pi**(-1.0/4.0)
    pos = w > 0
    wavelet = np.zeros((s.shape[0], w.shape[0]))

    for i in range(s.shape[0]):
        n = normalization(s[i], dt)
        wavelet[i][pos] = n * p * np.exp(-(s[i] * w[pos] - w0) ** 2 / 2.0)

    return wavelet


def dogft(s, w, order, dt):
    """
    Fourier transformed DOG function.

    Parameters
    ----------
    s : numpy.ndarray
        Scales.
    w : numpy.ndarray
        Angular frequencies.
    order : int
        Wavelet order.
    dt : int
        Time step.

    Returns
    -------
    numpy.ndarray
        Normalized Fourier transformed DOG function.
    """

    p = - (0.0 + 1.0j) ** order / np.sqrt(gamma(order + 0.5))
    wavelet = np.zeros((s.shape[0], w.shape[0]), dtype=np.complex128)

    for i in range(s.shape[0]):
        n = normalization(s[i], dt)
        h = s[i] * w
        wavelet[i] = n * p * h ** order * np.exp(-h ** 2 / 2.0)

    return wavelet


def angularfreq(N, dt):
    """
    Compute angular frequencies.

    Parameters
    ----------
    N : int
        Number of data samples.
    dt : int
        Time step.

    Returns
    -------
    numpy.ndarray
        Angular frequencies (1D).
    """

    # See (5) at page 64.

    N2 = int(N / 2.0)
    w = np.empty(N)

    for i in range(w.shape[0]):
        if i <= N2:
            w[i] = (2 * np.pi * i) / (N * dt)
        else:
            w[i] = (2 * np.pi * (i - N)) / (N * dt)

    return w


def autoscales(N, dt, dj, wf, p):
    """
    Compute scales as fractional power of two.

    Parameters
    ----------
    N : int
        Number of data samples.
    dt : int
        Time step.
    dj : float
        Scale resolution (smaller values of give finer resolution).
    wf : str
        Wavelet function ("morlet", "paul", or "dog").
    p : int
        omega0 ("morlet") or order ("paul", "dog").

    Returns
    -------
    numpy.ndarray
        Scales (1D).
    """

    if wf == 'dog':
        s0 = (dt * np.sqrt(p + 0.5)) / np.pi

    elif wf == 'morlet':
        s0 = (dt * (p + np.sqrt(2 + p ** 2))) / (2 * np.pi)

    else:
        raise ValueError('Wavelet function not available.')

    # See (9) and (10) at page 67.

    J = int(np.floor(dj ** -1 * np.log2((N * dt) / s0)))
    s = np.empty(J + 1)

    for i in range(s.shape[0]):
        s[i] = s0 * 2 ** (i * dj)

    return s


# def fourier_from_scales(scales, wf, p):
#     """Compute the equivalent fourier period
#     from scales.
#
#     :Parameters:
#        scales : list or 1d numpy array
#           scales
#        wf : string ('morlet', 'paul', 'dog')
#           wavelet function
#        p : float
#           wavelet function parameter ('omega0' for morlet, 'm' for paul
#           and dog)
#
#     :Returns:
#        fourier wavelengths
#     """
#
#     scales_arr = np.asarray(scales)
#
#     if wf == 'dog':
#         return (2 * np.pi * scales_arr) / np.sqrt(p + 0.5)
#     elif wf == 'morlet':
#         return (4 * np.pi * scales_arr) / (p + np.sqrt(2 + p ** 2))
#     else:
#         raise ValueError('wavelet function not available')


# def scales_from_fourier(f, wf, p):
#     """Compute scales from fourier period.
#
#     :Parameters:
#        f : list or 1d numpy array
#           fourier wavelengths
#        wf : string ('morlet', 'paul', 'dog')
#           wavelet function
#        p : float
#           wavelet function parameter ('omega0' for morlet, 'm' for paul
#           and dog)
#
#     :Returns:
#        scales
#     """
#
#     f_arr = np.asarray(f)
#
#     if wf == 'dog':
#         return (f_arr * np.sqrt(p + 0.5)) / (2 * np.pi)
#     elif wf == 'morlet':
#         return (f_arr * (p + np.sqrt(2 + p ** 2))) / (4 * np.pi)
#     else:
#         raise ValueError('wavelet function not available')


def cwt(x, dt, scales, wf="dog", p=2):
    """
    Continuous Wavelet Transform.

    Parameters
    ----------
    x : numpy.ndarray
        Data (1D).
    dt : int
        Time step.
    scales : numpy.ndarray
        Scales (1D).
    wf : str
        Wavelet function ("morlet", "paul", or "dog").
    p : int
        omega0 ("morlet") or order ("paul", "dog").

    Returns
    -------
    numpy.ndarray
        Transformed data (2D).
    """

    x_arr = np.asarray(x) - np.mean(x)
    scales_arr = np.asarray(scales)

    if x_arr.ndim != 1:
        raise ValueError('x must be an 1d numpy array of list')

    if scales_arr.ndim != 1:
        raise ValueError('scales must be an 1d numpy array of list')

    w = angularfreq(N=x_arr.shape[0], dt=dt)

    if wf == 'dog':
        wft = dogft(s=scales_arr, w=w, order=p, dt=dt)

    elif wf == 'morlet':
        wft = morletft(s=scales_arr, w=w, w0=p, dt=dt)

    else:
        raise ValueError('wavelet function is not available')

    X_ARR = np.empty((wft.shape[0], wft.shape[1]), dtype=np.complex128)

    x_arr_ft = np.fft.fft(x_arr)

    for i in range(X_ARR.shape[0]):
        X_ARR[i] = np.fft.ifft(x_arr_ft * wft[i])

    return X_ARR


def icwt(X, scales):
    """
    Inverse Continuous Wavelet Transform. The reconstruction factor is not applied.

    Parameters
    ----------
    X : numpy.ndarray
        Transformed data (2D).
    scales : numpy.ndarray
        Scales (1D).

    Returns
    -------
    numpy.ndarray
         1D data.
    """

    X_arr = np.asarray(X)
    scales_arr = np.asarray(scales)

    if X_arr.shape[0] != scales_arr.shape[0]:
        raise ValueError('X, scales: shape mismatch')

    # See (11), (13) at page 68
    X_ARR = np.empty_like(X_arr)
    for i in range(scales_arr.shape[0]):
        X_ARR[i] = X_arr[i] / np.sqrt(scales_arr[i])

    return np.sum(np.real(X_ARR), axis=0)
