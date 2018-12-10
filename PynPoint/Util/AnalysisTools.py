"""
Functions for analysis of a planet signal.
"""

from __future__ import absolute_import

import math

import numpy as np

from scipy.stats import t
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import hessian_matrix
from photutils import aperture_photometry, CircularAperture, EllipticalAperture
from six.moves import range

from PynPoint.Util.ImageTools import shift_image


def false_alarm(image,
                x_pos,
                y_pos,
                size,
                ignore):
    """
    Function for the formal t-test for high-contrast imaging at small working angles, as well as
    the related false positive fraction (Mawet et al. 2014).

    :param image: Input image.
    :type image: numpy.ndarray
    :param x_pos: Position (pix) along the x-axis.
    :type x_pos: float
    :param y_pos: Position (pix) along the y-axis.
    :type y_pos: float
    :param size: Aperture radius (pix).
    :type size: float
    :param ignore: Ignore neighboring aperture for the noise.
    :type ignore: bool

    :return: Noise level, SNR, FPF
    :rtype: float, float, float
    """

    center = (np.size(image, 0)/2., np.size(image, 1)/2.)
    radius = math.sqrt((center[0]-y_pos)**2.+(center[1]-x_pos)**2.)

    num_ap = int(math.pi*radius/size)
    ap_theta = np.linspace(0, 2.*math.pi, num_ap, endpoint=False)

    if ignore:
        num_ap -= 2
        ap_theta = np.delete(ap_theta, [1, np.size(ap_theta)-1])

    if num_ap < 3:
        raise ValueError("Number of apertures (num_ap=%s) is too small to calculate the "
                         "false positive fraction. Increase the lower limit of the "
                         "separation argument." % num_ap)

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

    return noise, t_test, 1.-t.cdf(t_test, num_ap-2)

def student_fpf(sigma,
                radius,
                size,
                ignore):
    """
    Function to calculate the false positive fraction for a given sigma level (Mawet et al. 2014).
    """

    num_ap = int(math.pi*radius/size)
    if ignore:
        num_ap -= 2

    return 1. - t.cdf(sigma, num_ap-2, loc=0., scale=1.)

def fake_planet(images,
                psf,
                parang,
                position,
                magnitude,
                psf_scaling,
                interpolation="spline"):
    """
    Function to inject artificial planets in a dataset.

    :param images: Input images.
    :type images: numpy.ndarray
    :param psf: PSF template.
    :type psf: numpy.ndarray
    :param parang: Parallactic angles (deg).
    :type parang: float
    :param position: Separation (pix) and position angle (deg) measured in counterclockwise
                     with respect to the upward direction.
    :type position: (float, float)
    :param magnitude: Magnitude difference used to scale input PSF.
    :type magnitude: float
    :param psf_scaling: Extra factor used to scale input PSF.
    :type psf_scaling: float
    :param interpolation: Interpolation type (spline, bilinear, fft)
    :type interpolation: str

    :return: Images with artificial planet injected.
    :rtype: numpy.ndarray
    """

    sep = position[0]
    ang = np.radians(position[1] + 90. - parang)

    flux_ratio = 10. ** (-magnitude / 2.5)
    psf = psf*psf_scaling*flux_ratio

    x_shift = sep*np.cos(ang)
    y_shift = sep*np.sin(ang)

    im_shift = np.zeros(images.shape)

    if images.ndim == 2:
        im_shift = shift_image(psf, (y_shift, x_shift), interpolation, mode='reflect')

    elif images.ndim == 3:
        for i in range(images.shape[0]):
            if psf.ndim == 2:
                im_shift[i, ] = shift_image(psf,
                                            (y_shift[i], x_shift[i]),
                                            interpolation,
                                            mode='reflect')

            elif psf.ndim == 3:
                im_shift[i, ] = shift_image(psf[i, ],
                                            (y_shift[i], x_shift[i]),
                                            interpolation,
                                            mode='reflect')

    return images + im_shift

def merit_function(residuals,
                   function,
                   aperture,
                   sigma):

    """
    Function to calculate the merit function at a given position in the image residuals.

    :param residuals: Residuals of the PSF subtraction.
    :type residuals: ndarray
    :param function: Figure of merit ("hessian" or "sum").
    :type function: str
    :param aperture: Dictionary with the aperture properties. See
                     Util.AnalysisTools.create_aperture for details.
    :type aperture: dict
    :param sigma: Standard deviation (pix) of the Gaussian kernel which is used to smooth
                  the residuals before the function of merit is calculated.
    :type sigma: float

    :return: Merit value.
    :rtype: float
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

        phot_table = aperture_photometry(np.abs(residuals),
                                         create_aperture(aperture),
                                         method='exact')

        merit = phot_table['aperture_sum']

    else:

        raise ValueError("Merit function not recognized.")

    return merit

def create_aperture(aperture):
    """
    Function to create a circular or elliptical aperture.

    :param aperture: Dictionary with the aperture properties. The aperture 'type' can be
                     'circular' or 'elliptical' (str). Both types of apertures require a position,
                     'pos_x' and 'pos_y' (float), where the aperture is placed. The circular
                     aperture requires a 'radius' (in pixels, float) and the elliptical
                     aperture requires a 'semimajor' and 'semiminor' axis (in pixels, float),
                     and an 'angle' (deg). The rotation angle in degrees of the semimajor axis
                     from the positive x axis. The rotation angle increases counterclockwise.
    :type aperture: dict

    :return: Aperture object.
    :rtype: photutils.aperture.circle.CircularAperture or
            photutils.aperture.circle.EllipticalAperture
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
