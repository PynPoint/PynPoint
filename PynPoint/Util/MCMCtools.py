"""
Functions for MCMC sampling.
"""

from __future__ import absolute_import

import math

import numpy as np

from PynPoint.Util.AnalysisTools import fake_planet, merit_function
from PynPoint.Util.PSFSubtractionTools import pca_psf_subtraction
from PynPoint.Util.Residuals import combine_residuals


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
           aperture,
           indices):
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
    :param pca_number: Number of principal components used for the PSF subtraction.
    :type pca_number: int
    :param extra_rot: Additional rotation angle of the images (deg).
    :type extra_rot: float
    :param aperture: Dictionary with the aperture properties. See
                     Util.AnalysisTools.create_aperture for details.
    :type aperture: dict
    :param indices: Non-masked image indices.
    :type indices: numpy.ndarray

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
        Internal function for the log likelihood function. Noise of each pixel is assumed to follow
        a Poisson distribution (see Wertz et al. 2017 for details).

        :return: Log likelihood.
        :rtype: float
        """

        sep, ang, mag = param

        fake = fake_planet(images=images,
                           psf=psf,
                           parang=parang-extra_rot,
                           position=(sep/pixscale, ang),
                           magnitude=mag,
                           psf_scaling=psf_scaling)

        fake *= mask

        _, im_res = pca_psf_subtraction(images=fake,
                                        angles=-1.*parang+extra_rot,
                                        pca_number=pca_number,
                                        indices=indices)

        stack = combine_residuals(method="mean", res_rot=im_res)

        merit = merit_function(residuals=stack,
                               function="sum",
                               aperture=aperture,
                               sigma=0.)

        return -0.5*merit

    ln_prior = _lnprior()

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + _lnlike()

    return ln_prob
