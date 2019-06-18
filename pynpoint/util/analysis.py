"""
Functions for point source analysis.
"""

import math

from typing import Tuple

import numpy as np

from typeguard import typechecked
from scipy.stats import t
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import hessian_matrix
from photutils import aperture_photometry, CircularAperture

from pynpoint.util.image import shift_image, center_subpixel, pixel_distance, select_annulus, \
                                cartesian_to_polar


@typechecked
def false_alarm(image: np.ndarray,
                x_pos: float,
                y_pos: float,
                size: float,
                ignore: bool) -> Tuple[float, float, float, float]:
    """
    Function for the formal t-test for high-contrast imaging at small working angles and the
    related false positive fraction (Mawet et al. 2014).

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D).
    x_pos : float
        Position (pix) along the horizontal axis. The pixel coordinates of the bottom-left
        corner of the image are (-0.5, -0.5).
    y_pos : float
        Position (pix) along the vertical axis. The pixel coordinates of the bottom-left corner
        of the image are (-0.5, -0.5).
    size : float
        Aperture radius (pix).
    ignore : bool
        Ignore the neighboring apertures for the noise estimate.

    Returns
    -------
    float
        Signal.
    float
        Noise level.
    float
        Signal-to-noise ratio.
    float
        False positive fraction (FPF).
    """

    center = center_subpixel(image)
    radius = math.sqrt((center[0]-y_pos)**2.+(center[1]-x_pos)**2.)

    num_ap = int(math.pi*radius/size)
    ap_theta = np.linspace(0, 2.*math.pi, num_ap, endpoint=False)

    if ignore:
        num_ap -= 2
        ap_theta = np.delete(ap_theta, [1, np.size(ap_theta)-1])

    if num_ap < 3:
        raise ValueError(f'Number of apertures (num_ap={num_ap}) is too small to calculate the '
                         'false positive fraction.')

    ap_phot = np.zeros(num_ap)

    for i, theta in enumerate(ap_theta):
        x_tmp = center[1] + (x_pos-center[1])*math.cos(theta) - \
                            (y_pos-center[0])*math.sin(theta)

        y_tmp = center[0] + (x_pos-center[1])*math.sin(theta) + \
                            (y_pos-center[0])*math.cos(theta)

        aperture = CircularAperture((x_tmp, y_tmp), size)
        phot_table = aperture_photometry(image, aperture, method='exact')
        ap_phot[i] = phot_table['aperture_sum']

    # Note: ddof=1 is a necessary argument in order to compute the *unbiased* estimate of the
    # standard deviation, as suggested by eq. 8 of Mawet et al. (2014).
    noise = np.std(ap_phot[1:], ddof=1) * math.sqrt(1.+1./float(num_ap-1))
    t_test = (ap_phot[0] - np.mean(ap_phot[1:])) / noise

    # Note that the number of degrees of freedom is given by nu = n-1 with n the number of samples.
    # The number of samples is equal to the number of apertures minus 1 (i.e. the planet aperture).
    # See Section 3 of Mawet et al. (2014) for more details on the Student's t distribution.
    return ap_phot[0], noise, t_test, 1.-t.cdf(t_test, num_ap-2)


@typechecked
def student_t(t_input: Tuple[str, float],
              radius: float,
              size: float,
              ignore: bool) -> float:
    """
    Function to calculate the false positive fraction for a given sigma level (Mawet et al. 2014).

    Parameters
    ----------
    t_input : tuple(str, float)
        Tuple with the input type ('sigma' or 'fpf') and the input value.
    radius : float
        Aperture radius (pix).
    size : float
        Separation of the aperture center (pix).
    ignore : bool
        Ignore neighboring apertures of the point source to exclude the self-subtraction lobes.

    Returns
    -------
    float
        False positive fraction (FPF).
    """

    num_ap = int(math.pi*radius/size)

    if ignore:
        num_ap -= 2

    # Note that the number of degrees of freedom is given by nu = n-1 with n the number of samples.
    # The number of samples is equal to the number of apertures minus 1 (i.e. the planet aperture).
    # See Section 3 of Mawet et al. (2014) for more details on the Student's t distribution.

    if t_input[0] == 'sigma':
        t_result = 1. - t.cdf(t_input[1], num_ap-2, loc=0., scale=1.)

    elif t_input[0] == 'fpf':
        t_result = t.ppf(1. - t_input[1], num_ap-2, loc=0., scale=1.)

    return t_result


@typechecked
def fake_planet(images: np.ndarray,
                psf: np.ndarray,
                parang: np.ndarray,
                position: Tuple[float, float],
                magnitude: float,
                psf_scaling: float,
                interpolation: str = 'spline') -> np.ndarray:
    """
    Function to inject artificial planets in a dataset.

    Parameters
    ----------
    images : numpy.ndarray
        Input images (3D).
    psf : numpy.ndarray
        PSF template (3D).
    parang : numpy.ndarray
        Parallactic angles (deg).
    position : tuple(float, float)
        Separation (pix) and position angle (deg) measured in counterclockwise with respect to the
        upward direction.
    magnitude : float
        Magnitude difference used to scale input PSF.
    psf_scaling : float
        Extra factor used to scale input PSF.
    interpolation : str
        Interpolation type ('spline', 'bilinear', or 'fft').

    Returns
    -------
    numpy.ndarray
        Images with artificial planet injected.
    """

    sep = position[0]
    ang = np.radians(position[1] + 90. - parang)

    flux_ratio = 10. ** (-magnitude / 2.5)
    psf = psf*psf_scaling*flux_ratio

    x_shift = sep*np.cos(ang)
    y_shift = sep*np.sin(ang)

    im_shift = np.zeros(images.shape)

    for i in range(images.shape[0]):
        if psf.shape[0] == 1:
            im_shift[i, ] = shift_image(psf[0, ],
                                        (y_shift[i], x_shift[i]),
                                        interpolation,
                                        mode='reflect')

        else:
            im_shift[i, ] = shift_image(psf[i, ],
                                        (y_shift[i], x_shift[i]),
                                        interpolation,
                                        mode='reflect')

    return images + im_shift


@typechecked
def merit_function(residuals: np.ndarray,
                   merit: str,
                   aperture: Tuple[int, int, float],
                   sigma: float) -> float:

    """
    Function to calculate the figure of merit at a given position in the image residuals.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residuals of the PSF subtraction (2D).
    merit : str
        Figure of merit for the chi-square function ('hessian', 'poisson', or 'gaussian').
    aperture : tuple(int, int, float)
        Position (y, x) of the aperture center (pix) and aperture radius (pix).
    sigma : float
        Standard deviation (pix) of the Gaussian kernel which is used to smooth the residuals
        before the chi-square is calculated.

    Returns
    -------
    float
        Chi-square ('poisson' and 'gaussian') or sum of the absolute values ('hessian').
    """

    rr_grid = pixel_distance(im_shape=residuals.shape,
                             position=(aperture[0], aperture[1]))

    indices = np.where(rr_grid < aperture[2])

    if merit == 'hessian':

        # This is not the chi-square but simply the sum of the absolute values

        hessian_rr, hessian_rc, hessian_cc = hessian_matrix(image=residuals,
                                                            sigma=sigma,
                                                            mode='constant',
                                                            cval=0.,
                                                            order='rc')

        hes_det = (hessian_rr*hessian_cc) - (hessian_rc*hessian_rc)

        chi_square = np.sum(np.abs(hes_det[indices]))

    elif merit == 'poisson':

        if sigma > 0.:
            residuals = gaussian_filter(input=residuals, sigma=sigma)

        chi_square = np.sum(np.abs(residuals[indices]))

    elif merit == 'gaussian':

        # separation (pix) and position angle (deg)
        sep_ang = cartesian_to_polar(center=center_subpixel(residuals),
                                     y_pos=aperture[0],
                                     x_pos=aperture[1])

        if sigma > 0.:
            residuals = gaussian_filter(input=residuals, sigma=sigma)

        selected = select_annulus(image_in=residuals,
                                  radius_in=sep_ang[0]-aperture[2],
                                  radius_out=sep_ang[0]+aperture[2],
                                  mask_position=aperture[0:2],
                                  mask_radius=aperture[2])

        chi_square = np.sum(residuals[indices]**2)/np.var(selected)

    else:

        raise ValueError('Figure of merit not recognized. Please use \'hessian\', \'poisson\' '
                         'or \'gaussian\'. Previous use of \'sum\' should now be set as '
                         '\'poisson\'.')

    return chi_square
