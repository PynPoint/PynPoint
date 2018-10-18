"""
Functions for analysis of a planet signal.
"""

import math

import numpy as np

from photutils import aperture_photometry, CircularAperture
from scipy.stats import t

from PynPoint.Util.ImageTools import shift_image


def false_alarm(image, x_pos, y_pos, size, ignore):
    """
    Function for the formal t-test for high-contrast imaging at small working angles, as well as
    the related false positive fraction (Mawet et al. 2014).

    :param image: Input image.
    :type image: ndarray
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

    return noise, t_test, 1. - t.cdf(t_test, num_ap-2)

def student_fpf(sigma, radius, size, ignore):
    """
    Function to calculate the false positive fraction for a given sigma level (Mawet et al. 2014).
    """

    num_ap = int(math.pi*radius/size)
    if ignore:
        num_ap -= 2

    return 1. - t.cdf(sigma, num_ap-2, loc=0., scale=1.)

def fake_planet(science,
                psf,
                parang,
                position,
                magnitude,
                psf_scaling,
                interpolation="spline"):
    """
    Function to inject artificial planets in a dataset.

    :param image: Input images.
    :type image: ndarray
    :param psf: PSF template.
    :type psf: ndarray
    :param parang: Parallactic angles (deg).
    :type parang: float
    :param position: Separation (pix) and position angle (deg) measured in counterclockwise
                     with respect to the upward direction.
    :type position: (float, float)
    :param interpolation: Interpolation type (spline, bilinear, fft)
    :type interpolation: str

    :return: Images with artificial planet injected.
    :rtype: ndarray
    """

    radial = position[0]
    theta = position[1]*math.pi/180. + math.pi/2.
    flux_ratio = 10.**(-magnitude/2.5)

    psf_size = (psf.shape[-2], psf.shape[-1])

    if psf_size != (science.shape[1], science.shape[2]):
        raise ValueError("The science images should have the same dimensions as the PSF template.")

    if psf.ndim == 3 and psf.shape[0] == 1:
        psf = np.squeeze(psf, axis=0)
    elif psf.ndim == 3 and psf.shape[0] != science.shape[0]:
        psf = np.mean(psf, axis=0)

    fake = np.copy(science)

    for i in range(fake.shape[0]):
        x_shift = radial*math.cos(theta-math.radians(parang[i]))
        y_shift = radial*math.sin(theta-math.radians(parang[i]))

        if psf.ndim == 2:
            psf_tmp = np.copy(psf)
        elif psf.ndim == 3:
            psf_tmp = np.copy(psf[i, ])

        psf_tmp = shift_image(psf_tmp, (y_shift, x_shift), interpolation, mode='reflect')

        fake[i, ] += psf_scaling*flux_ratio*psf_tmp

    return fake
