"""
Functions for analysis of a point source.
"""

from __future__ import absolute_import

import math

import numpy as np

from scipy.stats import t
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import hessian_matrix
from photutils import aperture_photometry, CircularAperture, EllipticalAperture
from six.moves import range

from pynpoint.util.image import shift_image, center_subpixel


def false_alarm(image,
                x_pos,
                y_pos,
                size,
                ignore):
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
        raise ValueError("Number of apertures (num_ap=%s) is too small to calculate the "
                         "false positive fraction." % num_ap)

    ap_phot = np.zeros(num_ap)

    for i, theta in enumerate(ap_theta):
        x_tmp = center[1] + (x_pos-center[1])*math.cos(theta) - \
                            (y_pos-center[0])*math.sin(theta)

        y_tmp = center[0] + (x_pos-center[1])*math.sin(theta) + \
                            (y_pos-center[0])*math.cos(theta)

        aperture = CircularAperture((x_tmp, y_tmp), size)
        phot_table = aperture_photometry(image, aperture, method='exact')
        ap_phot[i] = phot_table['aperture_sum']

    noise = np.std(ap_phot[1:]) * math.sqrt(1.+1./float(num_ap-1))
    t_test = (ap_phot[0] - np.mean(ap_phot[1:])) / noise

    # Note that the number of degrees of freedom is given by nu = n-1 with n the number of samples.
    # The number of samples is equal to the number of apertures minus 1 (i.e. the planet aperture).
    # See Section 3 of Mawet et al. (2014) for more details on the Student's t distribution.
    return ap_phot[0], noise, t_test, 1.-t.cdf(t_test, num_ap-2)

def student_t(t_input,
              radius,
              size,
              ignore):
    """
    Function to calculate the false positive fraction for a given sigma level (Mawet et al. 2014).

    Parameters
    ----------
    t_input : tuple(str, float)
        Tuple with the input type ("sigma" or "fpf") and the input value.
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

    if t_input[0] == "sigma":
        t_result = 1. - t.cdf(t_input[1], num_ap-2, loc=0., scale=1.)

    elif t_input[0] == "fpf":
        t_result = t.ppf(1. - t_input[1], num_ap-2, loc=0., scale=1.)

    return t_result

def fake_planet(images,
                psf,
                parang,
                position,
                magnitude,
                psf_scaling,
                interpolation="spline"):
    """
    Function to inject artificial planets in a dataset.

    Parameters
    ----------
    images : numpy.ndarray
        Input images (3D).
    psf : numpy.ndarray
        PSF template (3D).
    parang : float
        Parallactic angles (deg).
    position : tuple(float, float)
        Separation (pix) and position angle (deg) measured in counterclockwise with respect to the
        upward direction.
    magnitude : float
        Magnitude difference used to scale input PSF.
    psf_scaling : float
        Extra factor used to scale input PSF.
    interpolation : str
        Interpolation type ("spline", "bilinear", or "fft").

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

def merit_function(residuals,
                   function,
                   variance,
                   aperture,
                   sigma):

    """
    Function to calculate the merit function at a given position in the image residuals.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residuals of the PSF subtraction (2D).
    function : str
        Figure of merit ("hessian" or "sum").
    variance : tuple(str, float)
        Variance type and value for the likelihood function. The value is set to None in case a
        Poisson distribution is assumed.
    aperture : dict
        Dictionary with the aperture properties. See for more information
        :func:`~pynpoint.util.analysis.create_aperture`.
    sigma : float
        Standard deviation (pix) of the Gaussian kernel which is used to smooth the residuals
        before the function of merit is calculated.

    Returns
    -------
    float
        Merit value.
    """

    if function == "hessian":

        if aperture['type'] != 'circular':
            raise ValueError("Measuring the Hessian is only possible with a circular aperture.")

        npix = residuals.shape[-1]

        pos_x = aperture['pos_x']
        pos_y = aperture['pos_y']

        x_grid = np.linspace(-(pos_x+0.5), npix-(pos_x+0.5), npix)
        y_grid = np.linspace(-(pos_y+0.5), npix-(pos_y+0.5), npix)

        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        rr_grid = np.sqrt(xx_grid*xx_grid+yy_grid*yy_grid)

        hessian_rr, hessian_rc, hessian_cc = hessian_matrix(image=residuals,
                                                            sigma=sigma,
                                                            mode='constant',
                                                            cval=0.,
                                                            order='rc')

        hes_det = (hessian_rr*hessian_cc) - (hessian_rc*hessian_rc)
        hes_det[rr_grid > aperture['radius']] = 0.
        merit = np.sum(np.abs(hes_det))

    elif function == "sum":

        if sigma > 0.:
            residuals = gaussian_filter(input=residuals, sigma=sigma)

        # https://photutils.readthedocs.io/en/stable/overview.html
        # In Photutils, pixel coordinates are zero-indexed, meaning that (x, y) = (0, 0)
        # corresponds to the center of the lowest, leftmost array element. This means that
        # the value of data[0, 0] is taken as the value over the range -0.5 < x <= 0.5,
        # -0.5 < y <= 0.5. Note that this is the same coordinate system as used by PynPoint.

        phot_table = aperture_photometry(np.abs(residuals),
                                         create_aperture(aperture),
                                         method='exact')

        merit = phot_table['aperture_sum'][0]

        if variance[0] == "gaussian":
            merit = merit**2/variance[1]

    else:

        raise ValueError("Merit function not recognized.")

    return merit

def create_aperture(aperture):
    """
    Function to create a circular or elliptical aperture.

    Parameters
    ----------
    aperture : dict
        Dictionary with the aperture properties. The aperture 'type' can be 'circular' or
        'elliptical' (str). Both types of apertures require a position, 'pos_x' and 'pos_y'
        (float), where the aperture is placed. The circular aperture requires a 'radius'
        (in pixels, float) and the elliptical aperture requires a 'semimajor' and 'semiminor'
        axis (in pixels, float), and an 'angle' (deg). The rotation angle in degrees of the
        semimajor axis from the positive x axis. The rotation angle increases counterclockwise.

    Returns
    -------
    photutils.aperture.circle.CircularAperture or photutils.aperture.circle.EllipticalAperture
        Aperture object.
    """

    if aperture['type'] == "circular":

        phot_ap = CircularAperture((aperture['pos_x'], aperture['pos_y']),
                                   aperture['radius'])

    elif aperture['type'] == "elliptical":

        phot_ap = EllipticalAperture((aperture['pos_x'], aperture['pos_y']),
                                     aperture['semimajor'],
                                     aperture['semiminor'],
                                     math.radians(aperture['angle']))

    else:

        raise ValueError("Aperture type not recognized.")

    return phot_ap
