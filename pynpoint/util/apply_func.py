"""
Functions that are executed with
:func:`~pynpoint.core.processing.ProcessingModule.apply_function_to_images` and
:func:`~pynpoint.core.processing.ProcessingModule.apply_function_in_time`. The functions are placed
here such that they are pickable by the multiprocessing functionalities. The first two parameters
are always the sliced data and the index in the dataset.

TODO Docstrings are missing for most of the functions.
"""

import copy
import math
import warnings

from typing import List, Optional, Union, Tuple

import cv2
import numpy as np
import pywt

from numba import jit
from photutils import aperture_photometry
from photutils.aperture import Aperture
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale
from statsmodels.robust import mad
from typeguard import typechecked

from pynpoint.core.dataio import InputPort, OutputPort
from pynpoint.util.image import center_pixel, crop_image, scale_image, shift_image
from pynpoint.util.star import locate_star
from pynpoint.util.wavelets import WaveletAnalysisCapsule


@typechecked
def image_scaling(image_in: np.ndarray,
                  im_index: int,
                  scaling_y: float,
                  scaling_x: float,
                  scaling_flux: float) -> np.ndarray:

    return scaling_flux * scale_image(image_in, scaling_y, scaling_x)


@typechecked
def subtract_line(image_in: np.ndarray,
                  im_index: int,
                  mask: np.ndarray,
                  combine: str,
                  im_shape: Tuple[int, int]) -> np.ndarray:

    image_tmp = np.copy(image_in)
    image_tmp[mask == 0.] = np.nan

    if combine == 'mean':
        row_mean = np.nanmean(image_tmp, axis=1)
        col_mean = np.nanmean(image_tmp, axis=0)

        x_grid, y_grid = np.meshgrid(col_mean, row_mean)
        subtract = (x_grid+y_grid)/2.

    elif combine == 'median':
        col_median = np.nanmedian(image_tmp, axis=0)
        col_2d = np.tile(col_median, (im_shape[1], 1))

        image_tmp -= col_2d
        image_tmp[mask == 0.] = np.nan

        row_median = np.nanmedian(image_tmp, axis=1)
        row_2d = np.tile(row_median, (im_shape[0], 1))
        row_2d = np.rot90(row_2d)  # 90 deg rotation in clockwise direction

        subtract = col_2d + row_2d

    return image_in - subtract


@typechecked
def align_image(image_in: np.ndarray,
                im_index: int,
                interpolation: str,
                accuracy: float,
                resize: Optional[float],
                num_references: int,
                subframe: Optional[float],
                ref_images_reshape: np.ndarray,
                ref_images_shape: Tuple[int, int, int]) -> np.ndarray:

    offset = np.array([0., 0.])

    # Reshape the reference images back to their original 3D shape
    # The original shape can not be used directly because of util.module.update_arguments
    ref_images = ref_images_reshape.reshape(ref_images_shape)

    for i in range(num_references):
        if subframe is None:
            tmp_offset, _, _ = phase_cross_correlation(ref_images[i, :, :],
                                                       image_in,
                                                       upsample_factor=accuracy)

        else:
            sub_in = crop_image(image_in, None, subframe)
            sub_ref = crop_image(ref_images[i, :, :], None, subframe)

            tmp_offset, _, _ = phase_cross_correlation(sub_ref,
                                                       sub_in,
                                                       upsample_factor=accuracy)
        offset += tmp_offset

    offset /= float(num_references)

    if resize is not None:
        offset *= resize

        sum_before = np.sum(image_in)

        tmp_image = rescale(image_in,
                            (resize, resize),
                            order=5,
                            mode='reflect',
                            multichannel=False,
                            anti_aliasing=True)

        sum_after = np.sum(tmp_image)

        # Conserve flux because the rescale function normalizes all values to [0:1].
        tmp_image = tmp_image*(sum_before/sum_after)

    else:
        tmp_image = image_in

    return shift_image(tmp_image, offset, interpolation)


@typechecked
def fit_2d_function(image: np.ndarray,
                    im_index: int,
                    mask_radii: Tuple[float, float],
                    sign: str,
                    model: str,
                    filter_size: Optional[float],
                    guess: Union[Tuple[float, float, float, float, float, float, float],
                                 Tuple[float, float, float, float, float, float, float, float]],
                    mask_out_port: Optional[OutputPort],
                    xx_grid: np.ndarray,
                    yy_grid: np.ndarray,
                    rr_ap: np.ndarray,
                    pixscale: float) -> np.ndarray:

    @typechecked
    def gaussian_2d(grid: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
                    x_center: float,
                    y_center: float,
                    fwhm_x: float,
                    fwhm_y: float,
                    amp: float,
                    theta: float,
                    offset: float) -> np.ndarray:
        """
        Function to create a 2D elliptical Gaussian model.

        Parameters
        ----------
        grid : tuple(np.ndarray, np.ndarray), np.ndarray
            A tuple of two 2D arrays with the mesh grid points in x and y
            direction, or an equivalent 3D numpy array with 2 elements
            along the first axis.
        x_center : float
            Offset of the model center along the x axis (pix).
        y_center : float
            Offset of the model center along the y axis (pix).
        fwhm_x : float
            Full width at half maximum along the x axis (pix).
        fwhm_y : float
            Full width at half maximum along the y axis (pix).
        amp : float
            Peak flux.
        theta : float
            Rotation angle in counterclockwise direction (rad).
        offset : float
            Flux offset.

        Returns
        -------
        np.ndimage
            Raveled 2D elliptical Gaussian model.
        """

        (xx_grid, yy_grid) = grid

        x_diff = xx_grid - x_center
        y_diff = yy_grid - y_center

        sigma_x = fwhm_x/math.sqrt(8.*math.log(2.))
        sigma_y = fwhm_y/math.sqrt(8.*math.log(2.))

        a_gauss = 0.5 * ((np.cos(theta)/sigma_x)**2 + (np.sin(theta)/sigma_y)**2)
        b_gauss = 0.5 * ((np.sin(2.*theta)/sigma_x**2) - (np.sin(2.*theta)/sigma_y**2))
        c_gauss = 0.5 * ((np.sin(theta)/sigma_x)**2 + (np.cos(theta)/sigma_y)**2)

        gaussian = offset + amp*np.exp(-(a_gauss*x_diff**2 + b_gauss*x_diff*y_diff +
                                         c_gauss*y_diff**2))

        return gaussian[(rr_ap > mask_radii[0]) & (rr_ap < mask_radii[1])]

    @typechecked
    def moffat_2d(grid: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
                  x_center: float,
                  y_center: float,
                  fwhm_x: float,
                  fwhm_y: float,
                  amp: float,
                  theta: float,
                  offset: float,
                  beta: float) -> np.ndarray:
        """
        Function to create a 2D elliptical Moffat model.

        The parametrization used here is equivalent to the one in AsPyLib:
        http://www.aspylib.com/doc/aspylib_fitting.html#elliptical-moffat-psf

        Parameters
        ----------
        grid : tuple(np.ndarray, np.ndarray), np.ndarray
            A tuple of two 2D arrays with the mesh grid points in x and y
            direction, or an equivalent 3D numpy array with 2 elements
            along the first axis.
        x_center : float
            Offset of the model center along the x axis (pix).
        y_center : float
            Offset of the model center along the y axis (pix).
        fwhm_x : float
            Full width at half maximum along the x axis (pix).
        fwhm_y : float
            Full width at half maximum along the y axis (pix).
        amp : float
            Peak flux.
        theta : float
            Rotation angle in counterclockwise direction (rad).
        offset : float
            Flux offset.
        beta : float
            Power index.

        Returns
        -------
        np.ndimage
            Raveled 2D elliptical Moffat model.
        """

        (xx_grid, yy_grid) = grid

        x_diff = xx_grid - x_center
        y_diff = yy_grid - y_center

        if 2.**(1./beta)-1. < 0.:
            alpha_x = np.nan
            alpha_y = np.nan

        else:
            alpha_x = 0.5*fwhm_x/np.sqrt(2.**(1./beta)-1.)
            alpha_y = 0.5*fwhm_y/np.sqrt(2.**(1./beta)-1.)

        if alpha_x == 0. or alpha_y == 0.:
            a_moffat = np.nan
            b_moffat = np.nan
            c_moffat = np.nan

        else:
            a_moffat = (np.cos(theta)/alpha_x)**2. + (np.sin(theta)/alpha_y)**2.
            b_moffat = (np.sin(theta)/alpha_x)**2. + (np.cos(theta)/alpha_y)**2.
            c_moffat = 2.*np.sin(theta)*np.cos(theta)*(1./alpha_x**2. - 1./alpha_y**2.)

        a_term = a_moffat*x_diff**2
        b_term = b_moffat*y_diff**2
        c_term = c_moffat*x_diff*y_diff

        moffat = offset + amp / (1.+a_term+b_term+c_term)**beta

        return moffat[(rr_ap > mask_radii[0]) & (rr_ap < mask_radii[1])]

    if filter_size:
        image = gaussian_filter(image, filter_size)

    if mask_out_port is not None:
        mask = np.copy(image)

        mask[(rr_ap < mask_radii[0]) | (rr_ap > mask_radii[1])] = 0.

        mask_out_port.append(mask, data_dim=3)

    if sign == 'negative':
        image = -1.*image + np.abs(np.min(-1.*image))

    image = image[(rr_ap > mask_radii[0]) & (rr_ap < mask_radii[1])]

    if model == 'gaussian':
        model_func = gaussian_2d

    elif model == 'moffat':
        model_func = moffat_2d

    try:
        popt, pcov = curve_fit(model_func,
                               (xx_grid, yy_grid),
                               image,
                               p0=guess,
                               sigma=None,
                               method='lm')

        perr = np.sqrt(np.diag(pcov))

    except RuntimeError:
        if model == 'gaussian':
            popt = np.zeros(7)
            perr = np.zeros(7)

        elif model == 'moffat':
            popt = np.zeros(8)
            perr = np.zeros(8)

        print(f'Fit could not converge on image number {im_index}. [WARNING]')

    if model == 'gaussian':

        best_fit = np.asarray((popt[0], perr[0],
                               popt[1], perr[1],
                               popt[2]*pixscale, perr[2]*pixscale,
                               popt[3]*pixscale, perr[3]*pixscale,
                               popt[4], perr[4],
                               math.degrees(popt[5]) % 360., math.degrees(perr[5]),
                               popt[6], perr[6]))

    elif model == 'moffat':

        best_fit = np.asarray((popt[0], perr[0],
                               popt[1], perr[1],
                               popt[2]*pixscale, perr[2]*pixscale,
                               popt[3]*pixscale, perr[3]*pixscale,
                               popt[4], perr[4],
                               math.degrees(popt[5]) % 360., math.degrees(perr[5]),
                               popt[6], perr[6],
                               popt[7], perr[7]))

    return best_fit


@typechecked
def crop_around_star(image: np.ndarray,
                     im_index: int,
                     position: Optional[Union[Tuple[int, int, float],
                                              Tuple[None, None, float]]],
                     im_size: int,
                     fwhm: int,
                     pixscale: float,
                     index_out_port: Optional[OutputPort],
                     image_out_port: OutputPort) -> np.ndarray:

    if position is None:
        center = None
        width = None

    else:
        if position[0] is None and position[1] is None:
            center = None
        else:
            center = (position[1], position[0])  # (y, x)

        width = int(math.ceil(position[2]/pixscale))

    starpos = locate_star(image, center, width, fwhm)

    try:
        im_crop = crop_image(image, tuple(starpos), im_size)

    except ValueError:
        warnings.warn(f'Chosen image size is too large to crop the image around the '
                      f'brightest pixel (image index = {im_index}, pixel [x, y] '
                      f'= [{starpos[0]}, {starpos[1]}]). Using the center of the '
                      f'image instead.')

        if index_out_port is not None:
            index_out_port.append(im_index, data_dim=1)

        starpos = center_pixel(image)
        im_crop = crop_image(image, tuple(starpos), im_size)

    return im_crop


@typechecked
def crop_rotating_star(image: np.ndarray,
                       im_index: int,
                       position: Union[Tuple[float, float], np.ndarray],
                       im_size: int,
                       filter_size: Optional[int],
                       search_size: int) -> np.ndarray:

    starpos = locate_star(image=image,
                          center=tuple(position),
                          width=search_size,
                          fwhm=filter_size)

    return crop_image(image=image,
                      center=tuple(starpos),
                      size=im_size)


@typechecked
def photometry(image: np.ndarray,
               im_index: int,
               aperture: Union[Aperture, List[Aperture]]) -> np.float64:
    # https://photutils.readthedocs.io/en/stable/overview.html
    # In Photutils, pixel coordinates are zero-indexed, meaning that (x, y) = (0, 0)
    # corresponds to the center of the lowest, leftmost array element. This means that
    # the value of data[0, 0] is taken as the value over the range -0.5 < x <= 0.5,
    # -0.5 < y <= 0.5. Note that this is the same coordinate system as used by PynPoint.

    return np.array(aperture_photometry(image, aperture, method='exact')['aperture_sum'])


@typechecked
def image_stat(image_in: np.ndarray,
               im_index: int,
               indices: Optional[np.ndarray]) -> np.ndarray:

    if indices is None:
        image_select = np.copy(image_in)

    else:
        image_reshape = np.reshape(image_in, (image_in.shape[0]*image_in.shape[1]))
        image_select = image_reshape[indices]

    nmin = np.nanmin(image_select)
    nmax = np.nanmax(image_select)
    nsum = np.nansum(image_select)
    mean = np.nanmean(image_select)
    median = np.nanmedian(image_select)
    std = np.nanstd(image_select)

    return np.asarray([nmin, nmax, nsum, mean, median, std])


@typechecked
def subtract_psf(image: np.ndarray,
                 im_index: int,
                 parang_thres: Optional[float],
                 nref: Optional[int],
                 reference: Optional[np.ndarray],
                 ang_diff: np.ndarray,
                 image_in_port: InputPort) -> np.ndarray:

    if parang_thres:
        index_thres = np.where(ang_diff > parang_thres)[0]

        if index_thres.size == 0:
            reference = image_in_port.get_all()

            warnings.warn('No images meet the rotation threshold. Creating a reference '
                          'PSF from the median of all images instead.')

        else:
            if nref:
                index_diff = np.abs(im_index - index_thres)
                index_near = np.argsort(index_diff)[:nref]
                index_sort = np.sort(index_thres[index_near])

                reference = image_in_port[index_sort, :, :]

            else:
                reference = image_in_port[index_thres, :, :]

        reference = np.median(reference, axis=0)

    return image-reference


@typechecked
def dwt_denoise_line_in_time(signal_in: np.ndarray,
                             im_index: int,
                             threshold_function: bool,
                             padding: str,
                             wavelet_conf) -> np.ndarray:
    """
    Definition of the temporal denoising for DWT.

    Parameters
    ----------
    signal_in : np.ndarray
        1D input signal.

    Returns
    -------
    np.ndarray
        Multilevel 1D inverse discrete wavelet transform.
    """

    if threshold_function:
        threshold_mode = 'soft'
    else:
        threshold_mode = 'hard'

    coef = pywt.wavedec(signal_in, wavelet=wavelet_conf.m_wavelet, level=None, mode=padding)

    sigma = mad(coef[-1])

    threshold = sigma * np.sqrt(2 * np.log(len(signal_in)))

    denoised = coef[:]

    denoised[1:] = (pywt.threshold(i, value=threshold, mode=threshold_mode) for i in denoised[1:])

    return pywt.waverec(denoised, wavelet=wavelet_conf.m_wavelet, mode=padding)


@typechecked
def cwt_denoise_line_in_time(signal_in: np.ndarray,
                             im_index: int,
                             threshold_function: bool,
                             padding: str,
                             median_filter: bool,
                             wavelet_conf) -> np.ndarray:
    """
    Definition of temporal denoising for CWT.

    Parameters
    ----------
    signal_in : np.ndarray
        1D input signal.

    Returns
    -------
    np.ndarray
        1D output signal.
    """

    cwt_capsule = WaveletAnalysisCapsule(signal_in=signal_in,
                                         padding=padding,
                                         wavelet_in=wavelet_conf.m_wavelet,
                                         order=wavelet_conf.m_wavelet_order,
                                         frequency_resolution=wavelet_conf.m_resolution)

    cwt_capsule.compute_cwt()

    cwt_capsule.denoise_spectrum(soft=threshold_function)

    if median_filter:
        cwt_capsule.median_filter()

    cwt_capsule.update_signal()

    return cwt_capsule.get_signal()


@typechecked
def normalization(image_in: np.ndarray,
                  im_index: int) -> np.ndarray:

    return image_in - np.median(image_in)


@typechecked
def time_filter(timeline: np.ndarray,
                im_index: int,
                sigma: Tuple[float, float]) -> np.ndarray:

    median = np.median(timeline)
    std = np.std(timeline)

    index_lower = np.argwhere(timeline < median-sigma[0]*std)
    index_upper = np.argwhere(timeline > median+sigma[1]*std)

    if index_lower.size > 0:
        mask = np.ones(timeline.shape, dtype=bool)
        mask[index_lower] = False
        timeline[index_lower] = np.mean(timeline[mask])

    if index_upper.size > 0:
        mask = np.ones(timeline.shape, dtype=bool)
        mask[index_upper] = False
        timeline[index_upper] = np.mean(timeline[mask])

    return timeline


# This function cannot by @typechecked because of a compatibility issue with numba
@jit(cache=True)
def calc_fast_convolution(F_roof_tmp: np.complex128,
                          W: np.ndarray,
                          tmp_s: tuple,
                          N_size: float,
                          tmp_G: np.ndarray,
                          N: Tuple[int, ...]) -> np.ndarray:

    new = np.zeros(N, dtype=np.complex64)

    if ((tmp_s[0] == 0) and (tmp_s[1] == 0)) or \
            ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == 0)) or \
            ((tmp_s[0] == 0) and (tmp_s[1] == N[1] / 2)) or \
            ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == N[1] / 2)):

        for m in range(0, N[0], 1):
            for j in range(0, N[1], 1):
                new[m, j] = F_roof_tmp * W[m - tmp_s[0], j - tmp_s[1]]

    else:

        for m in range(0, N[0], 1):
            for j in range(0, N[1], 1):
                new[m, j] = (F_roof_tmp * W[m - tmp_s[0], j - tmp_s[1]] +
                             np.conjugate(F_roof_tmp) * W[(m + tmp_s[0]) %
                             N[0], (j + tmp_s[1]) % N[1]])

    if ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == 0)) or \
            ((tmp_s[0] == 0) and (tmp_s[1] == N[1] / 2)) or \
            ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == N[1] / 2)):  # causes problems, unknown why

        res = new / float(N_size)

    else:

        res = new / float(N_size)

    tmp_G = tmp_G - res

    return tmp_G


@typechecked
def bad_pixel_interpolation(image_in: np.ndarray,
                            bad_pixel_map: np.ndarray,
                            iterations: int) -> np.ndarray:
    """
    Internal function to interpolate bad pixels.

    Parameters
    ----------
    image_in : np.ndarray
        Input image.
    bad_pixel_map : np.ndarray
        Bad pixel map.
    iterations : int
        Number of iterations.

    Returns
    -------
    np.ndarray
        Image in which the bad pixels have been interpolated.
    """

    image_in = image_in * bad_pixel_map

    # for names see ref paper
    g = copy.deepcopy(image_in)
    G = np.fft.fft2(g)
    w = copy.deepcopy(bad_pixel_map)
    W = np.fft.fft2(w)

    N = g.shape
    N_size = float(N[0] * N[1])
    F_roof = np.zeros(N, dtype=complex)
    tmp_G = copy.deepcopy(G)

    iteration = 0

    while iteration < iterations:
        # 1.) select line using max search and compute conjugate
        tmp_s = np.unravel_index(np.argmax(abs(tmp_G.real[:, 0: N[1] // 2])),
                                 (N[0], N[1] // 2))

        tmp_s_conjugate = (np.mod(N[0] - tmp_s[0], N[0]),
                           np.mod(N[1] - tmp_s[1], N[1]))

        # 2.) compute the new F_roof
        # special cases s = 0 or s = N/2 no conjugate line exists
        if ((tmp_s[0] == 0) and (tmp_s[1] == 0)) or \
                ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == 0)) or \
                ((tmp_s[0] == 0) and (tmp_s[1] == N[1] / 2)) or \
                ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == N[1] / 2)):
            F_roof_tmp = N_size * tmp_G[tmp_s] / W[(0, 0)]

            # 3.) update F_roof
            F_roof[tmp_s] += F_roof_tmp

        # conjugate line exists
        else:
            a = (np.power(np.abs(W[(0, 0)]), 2))
            b = np.power(np.abs(W[(2 * tmp_s[0]) % N[0], (2 * tmp_s[1]) % N[1]]), 2)

            if a == b:
                W[(2 * tmp_s[0]) % N[0], (2 * tmp_s[1]) % N[1]] += 0.00000000001

            a = (np.power(np.abs(W[(0, 0)]), 2))
            b = np.power(np.abs(W[(2 * tmp_s[0]) % N[0], (2 * tmp_s[1]) % N[1]]),
                         2.0) + 0.01
            c = a - b

            F_roof_tmp = N_size * (tmp_G[tmp_s] * W[(0, 0)] - np.conj(tmp_G[tmp_s]) *
                                   W[(2 * tmp_s[0]) % N[0], (2 * tmp_s[1]) % N[1]]) / c

            # 3.) update F_roof
            F_roof[tmp_s] += F_roof_tmp
            F_roof[tmp_s_conjugate] += np.conjugate(F_roof_tmp)

        # 4.) calc the new error spectrum using fast numba function
        tmp_G = calc_fast_convolution(F_roof_tmp, W, tmp_s, N_size, tmp_G, N)

        iteration += 1

    return image_in * bad_pixel_map + np.fft.ifft2(F_roof).real * (1 - bad_pixel_map)


@typechecked
def image_interpolation(image_in: np.ndarray,
                        im_index: int,
                        iterations: int,
                        bad_pixel_map: np.ndarray) -> np.ndarray:

    return bad_pixel_interpolation(image_in,
                                   bad_pixel_map,
                                   iterations)


@typechecked
def replace_pixels(image: np.ndarray,
                   im_index: int,
                   index: np.ndarray,
                   size: int,
                   replace: str) -> np.ndarray:

    im_mask = np.copy(image)

    for _, item in enumerate(index):
        im_mask[item[0], item[1]] = np.nan

    for _, item in enumerate(index):
        im_tmp = im_mask[item[0]-size:item[0]+size+1,
                         item[1]-size:item[1]+size+1]

        if np.size(np.where(im_tmp != np.nan)[0]) == 0:
            im_mask[item[0], item[1]] = image[item[0], item[1]]

        else:
            if replace == 'mean':
                im_mask[item[0], item[1]] = np.nanmean(im_tmp)

            elif replace == 'median':
                im_mask[item[0], item[1]] = np.nanmedian(im_tmp)

            elif replace == 'nan':
                im_mask[item[0], item[1]] = np.nan

    return im_mask


# This function cannot by @typechecked because of a compatibility issue with numba
@jit(cache=True)
def sigma_filter(dev_image: np.ndarray,
                 var_image: np.ndarray,
                 mean_image: np.ndarray,
                 source_image: np.ndarray,
                 out_image: np.ndarray,
                 bad_pixel_map: np.ndarray) -> None:

    for i in range(source_image.shape[0]):
        for j in range(source_image.shape[1]):

            if dev_image[i][j] < var_image[i][j]:
                out_image[i][j] = source_image[i][j]

            else:
                out_image[i][j] = mean_image[i][j]
                bad_pixel_map[i][j] = 0

    return out_image, bad_pixel_map


@typechecked
def bad_pixel_sigma_filter(image_in: np.ndarray,
                           im_index: int,
                           box: int,
                           sigma: float,
                           iterate: int,
                           map_out_port: Optional[OutputPort]) -> np.ndarray:

    # Algorithm adapted from http://idlastro.gsfc.nasa.gov/ftp/pro/image/sigma_filter.pro

    # Initialize bad pixel map

    bad_pixel_map = np.ones(image_in.shape)

    while iterate > 0:
        # Source image

        source_image = copy.deepcopy(image_in)

        source_blur = cv2.blur(copy.deepcopy(source_image), (box, box))

        # Mean image

        box2 = box * box

        mean_image = (source_blur * box2 - source_image) / (box2 - 1)

        # Squared deviation between mean and source image

        dev_image = (mean_image - source_image) ** 2

        dev_blur = cv2.blur(copy.deepcopy(dev_image), (box, box))

        # Compute variance by smoothing the image with the deviations from the mean

        fact = float(sigma ** 2) / (box2 - 2)

        var_image = fact * (dev_blur * box2 - dev_image)

        # Update image_in for the next iteration by setting out_image equal to image_in

        out_image = image_in

        # Apply the sigma filter

        out_image, bad_pixel_map = sigma_filter(dev_image,
                                                var_image,
                                                mean_image,
                                                source_image,
                                                out_image,
                                                bad_pixel_map)

        # Subtract 1 from the number of iterations

        iterate -= 1

    if map_out_port is not None:
        # Write bad pixel map to the database when CPU = 1
        map_out_port.append(bad_pixel_map, data_dim=3)

    return out_image


@typechecked
def apply_shift(image_in: np.ndarray,
                im_index: int,
                shift: Union[Tuple[float, float], np.ndarray],
                interpolation: str) -> np.ndarray:

    return shift_image(image_in, shift, interpolation)
