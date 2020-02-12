"""
Functions for point source analysis.
"""

import math

from typing import Tuple, Union

import numpy as np

from typeguard import typechecked
from scipy.stats import t
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import hessian_matrix
from photutils import aperture_photometry, CircularAperture

from pynpoint.util.image import shift_image, center_subpixel, pixel_distance, select_annulus, \
                                cartesian_to_polar, create_mask
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals


@typechecked
def false_alarm(image: np.ndarray,
                x_pos: float,
                y_pos: float,
                size: float,
                ignore: bool) -> Tuple[float, float, float, float]:
    """
    Compute the signal-to-noise ratio (SNR), which is formally defined as the test statistic of a
    two-sample t-test, and related quantities (such as the FPF) at a given position in an image.

    For more detailed information about the definition of the signal-to-noise ratio and the
    motivation behind it, please see the following paper:

        Mawet, D. et al. (2014): "Fundamental limitations of high contrast imaging set by small
        sample statistics". *The Astrophysical Journal*, 792(2), 97.
        DOI: `10.1088/0004-637X/792/2/97 <https://dx.doi.org/10.1088/0004-637X/792/2/97>`_.

    Parameters
    ----------
    image : numpy.ndarray
        The input image as a 2D numpy array. For example, this could be a residual frame returned by
        a :class:`.PcaPsfSubtractionModule`.
    x_pos : float
        The planet position (in pixels) along the horizontal axis. The pixel coordinates of the
        bottom-left corner of the image are (-0.5, -0.5).
    y_pos : float
        The planet position (pix) along the vertical axis. The pixel coordinates of the bottom-left
        corner of the image are (-0.5, -0.5).
    size : float
        The radius of the references apertures (in pixels). Usually, this values should be chosen
        close to lambda over D, that is, the typical FWHM of the PSF.
    ignore : bool
        Whether or not to ignore the immediate neighboring apertures for the noise estimate. This is
        desirable in case there are "self-subtraction wings" left and right of the planet which
        would bias the estimation of the noise level at the separation of the planet if not ignored.

    Returns
    -------
    signal_sum :
        The integrated (summed up) flux inside the signal aperture.

        Please note that this is **not** identical to the numerator of the fraction defining the SNR
        (which is given by the `signal_sum` minus the mean of the noise apertures).
    noise :
        The denominator of the SNR, i.e., the standard deviation of the integrated flux of the noise
        apertures, times a correction factor that accounts for small sample statistics.
    snr :
        The signal-to-noise ratio (SNR) as defined by Mawet et al. (2014) in eq. (8).
    fpf :
        The false positive fraction (FPF) as defined by Mawet et al. (2014) in eq. (10).
    """

    # Compute the center of the current frame (with subpixel precision) and use it to compute the
    # radius of the given position in polar coordinates (with the origin at the center of the frame)
    center = center_subpixel(image)
    radius = math.sqrt((center[0] - y_pos)**2 + (center[1] - x_pos)**2)

    # Compute the number of apertures which we can place at the separation of  the given position
    num_ap = int(math.pi * radius / size)

    # Compute the angles at which to place the reference apertures
    ap_theta = np.linspace(0, 2 * math.pi, num_ap, endpoint=False)

    # If ignore is True, delete the apertures immediately right and left of the aperture placed on
    # the planet signal. These apertures often contain "self-subtraction wings", which means they
    # cannot be considered to originate from the same distribution. In accordance with section 3.2
    # of Mawet et al. (2014), such apertures are ignored to prevent bias.
    if ignore:
        num_ap -= 2
        ap_theta = np.delete(ap_theta, [1, np.size(ap_theta) - 1])

    # If the number of apertures is 2 or less, we cannot compute the false positive fraction
    if num_ap < 3:
        raise ValueError(f'Number of apertures (num_ap={num_ap}) is too small to calculate the '
                         'false positive fraction.')

    # Initialize a numpy array in which we will store the integrated flux of all reference apertures
    ap_phot = np.zeros(num_ap)

    # Loop over all reference apertures and measure the integrated flux
    for i, theta in enumerate(ap_theta):

        # Compute the position of the current aperture in polar coordinates and convert to Cartesian
        x_tmp = center[1] + (x_pos - center[1]) * math.cos(theta) - \
                            (y_pos - center[0]) * math.sin(theta)
        y_tmp = center[0] + (x_pos - center[1]) * math.sin(theta) + \
                            (y_pos - center[0]) * math.cos(theta)

        # Place a circular aperture at this position and sum up the flux inside the aperture
        aperture = CircularAperture((x_tmp, y_tmp), size)
        phot_table = aperture_photometry(image, aperture, method='exact')
        ap_phot[i] = phot_table['aperture_sum']

    # Define shortcuts to the signal and the noise aperture sums
    signal_aperture = ap_phot[0]
    noise_apertures = ap_phot[1:]

    # Compute the "signal", that is, the numerator of the signal-to-noise ratio: According to
    # eq. (8) in Mawet et al. (2014), this is given by the difference between the integrated flux
    # in the signal aperture and the mean of the integrated flux in the noise apertures
    signal = signal_aperture - np.mean(noise_apertures)

    # Compute the "noise", that is, the denominator of the signal-to-noise-ratio: According to
    # eq. (8) in Mawet et al. (2014), this is given by the standard deviation of the integrated flux
    # in the noise apertures times a correction factor to account for the small sample statistics.
    # NOTE: `ddof=1` is a necessary argument for np.std() in order to compute the *unbiased*
    #       estimate (i.e., including Bessel's corrections) of the standard deviation.
    noise = np.std(ap_phot[1:], ddof=1) * math.sqrt(1 + 1 / (num_ap - 1))

    # Compute the signal-to-noise ratio by dividing the "signal" through the "noise"
    snr = signal / noise

    # Compute the false positive fraction (FPF). According to eq. (10) in Mawet et al. (2014), the
    # FPF is given by 1 - F_nu(SNR), where F_nu is the cumulative distribution function (CDF) of a
    # t-distribution with `nu = n-1` degrees of freedom (see Section 3 of Mawet et al. (2014) for
    # more details on the Student's t distribution).
    # For numerical reasons, we use the survival function (SF), which is defined precisely as 1-CDF,
    # but may give more accurate results according to the scipy documentation.
    fpf = t.sf(snr, df=(num_ap - 2))

    return signal_aperture, noise, snr, fpf


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
                   sigma: float,
                   noise: Union[float, None]) -> float:
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
    noise : float, None
        Variance of the noise which is required when `merit` is set to 'gaussian'.

    Returns
    -------
    float
        Chi-square ('poisson' and 'gaussian') or sum of the absolute values ('hessian').
    """

    rr_grid = pixel_distance(im_shape=residuals.shape,
                             position=(aperture[0], aperture[1]))

    indices = np.where(rr_grid <= aperture[2])

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

        chi_square = np.sum(residuals[indices]**2)/noise

    else:

        raise ValueError('Figure of merit not recognized. Please use \'hessian\', \'poisson\' '
                         'or \'gaussian\'. Previous use of \'sum\' should now be set as '
                         '\'poisson\'.')

    return chi_square


@typechecked
def gaussian_noise(images: np.ndarray,
                   parang: np.ndarray,
                   cent_size: Union[float, None],
                   edge_size: Union[float, None],
                   pca_number: int,
                   residuals: str,
                   aperture: Tuple[int, int, float]) -> float:
    """
    Function to calculate the variance of the noise. After the PSF subtraction, images are rotated
    in opposite direction of the regular derotation, therefore dispersing any companion or disk
    signal. The noise is measured within an annulus.

    Parameters
    ----------
    images : numpy.ndarray
        Input images (3D).
    parang : numpy.ndarray
        Parallactic angles.
    cent_size : float, None
        Radius of the central mask (pix). No mask is used when set to None.
    edge_size : float, None
        Outer radius (pix) beyond which pixels are masked. No outer mask is used when set to
        None.
    pca_number : int
        Number of principal components (PCs) used for the PSF subtraction.
    residuals : str
        Method for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
    aperture : tuple(int, int, float)
        Aperture position (y, x) and radius (pix).

    Returns
    -------
    float
        Variance of the pixel values.
    """

    mask = create_mask(images.shape[-2:], (cent_size, edge_size))

    _, im_res_derot = pca_psf_subtraction(images*mask, parang, pca_number)

    res_noise = combine_residuals(residuals, im_res_derot)

    sep_ang = cartesian_to_polar(center_subpixel(res_noise), aperture[0], aperture[1])

    selected = select_annulus(res_noise[0, ], sep_ang[0]-aperture[2], sep_ang[0]+aperture[2])

    return np.var(selected)
