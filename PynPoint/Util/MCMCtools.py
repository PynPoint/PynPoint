"""
Functions for image processing.
"""

import math

import numpy as np

from photutils import aperture_photometry

from PynPoint.Util.AnalysisTools import fake_planet
from PynPoint.Util.PSFSubtractionTools import pca_psf_subtraction

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
                           (sep/pixscale, ang),
                           mag,
                           psf_scaling)

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
