"""
Functions for image processing.
"""

import math

import numpy as np

from photutils import aperture_photometry

from PynPoint.Util.ImageTools import shift_image
from PynPoint.Util.PSFSubtractionTools import pca_psf_subtraction


def fake_planet(science,
                psf,
                parang,
                position,
                magnitude,
                psf_scaling,
                pixscale):
    """
    Function to inject fake planets into a cube of images. This function is similar to
    FakePlanetModule but does not require access to the PynPoint database.

    :return: Images with a fake planet injected.
    :rtype: ndarray
    """

    radial = position[0]/pixscale
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

        psf_tmp = shift_image(psf_tmp, (y_shift, x_shift), interpolation="spline", mode='reflect')

        fake[i, ] += psf_scaling*flux_ratio*psf_tmp

    return fake

def lnprob(param,
           bounds,
           images,
           psf,
           mask,
           parang,
           psf_scaling,
           pixscale,
           pca_number,
           extra_rot,
           aperture):
    """
    Function for the log posterior function. Should be placed at the highest level of the
    Python module in order to be pickled.

    :param param: Tuple with the separation (arcsec), angle (deg), and contrast
                  (mag). The angle is measured in counterclockwise direction with
                  respect to the upward direction (i.e., East of North).
    :type param: (float, float, float)
    :param bounds: Tuple with the boundaries of the separation (arcsec), angle (deg),
                   and contrast (mag). Each set of boundaries is specified as a tuple.
    :type bounds: ((float, float), (float, float), (float, float))
    :param images: Stack with images.
    :type images: ndarray
    :param psf: PSF template, either a single image (2D) or a cube (3D) with the dimensions
                equal to *image_in_tag*.
    :type psf: ndarray
    :param mask: Array with the circular mask (zeros) of the central and outer regions.
    :type mask: ndarray
    :param parang: Array with the angles for derotation.
    :type parang: ndarray
    :param psf_scaling: Additional scaling factor of the planet flux (e.g., to correct for a
                        neutral density filter). Should be negative in order to inject negative
                        fake planets.
    :type psf_scaling: float
    :param pixscale: Additional scaling factor of the planet flux (e.g., to correct for a neutral
                     density filter). Should be negative in order to inject negative fake planets.
    :type pixscale: float
    :param pca_number: Number of principle components used for the PSF subtraction.
    :type pca_number: int
    :param extra_rot: Additional rotation angle of the images (deg).
    :type extra_rot: float
    :param aperture: Circular aperture at the position specified in *param*.
    :type aperture: photutils.CircularAperture

    :return: Log posterior.
    :rtype: float
    """

    def _lnprior():
        """
        Internal function for the log prior function.

        :return: Log prior.
        :rtype: float
        """

        if bounds[0][0] <= param[0] <= bounds[0][1] and \
           bounds[1][0] <= param[1] <= bounds[1][1] and \
           bounds[2][0] <= param[2] <= bounds[2][1]:

            ln_prior = 0.

        else:

            ln_prior = -np.inf

        return ln_prior

    def _lnlike():
        """
        Internal function for the log likelihood function. Noise of each pixel is assumed to be
        given by photon noise only (see Wertz et al. 2017 for details).

        :return: Log likelihood.
        :rtype: float
        """

        sep, ang, mag = param

        fake = fake_planet(images,
                           psf,
                           parang-extra_rot,
                           (sep, ang),
                           mag,
                           psf_scaling,
                           pixscale)

        fake *= mask

        im_res = pca_psf_subtraction(fake,
                                     parang,
                                     pca_number,
                                     extra_rot)

        phot_table = aperture_photometry(np.abs(im_res), aperture, method='exact')

        return -0.5*phot_table['aperture_sum'][0]

    ln_prior = _lnprior()

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + _lnlike()

    return ln_prob
