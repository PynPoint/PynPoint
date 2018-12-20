"""
Functions for MCMC sampling.
"""

from __future__ import absolute_import

import math

import numpy as np

from pynpoint.util.analysis import fake_planet, merit_function
from pynpoint.util.image import polar_to_cartesian
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals


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
           indices,
           prior):
    """
    Function for the log posterior function. Should be placed at the highest level of the
    Python module to be pickable for the multiprocessing.

    :param param: Tuple with the separation (arcsec), angle (deg), and contrast
                  (mag). The angle is measured in counterclockwise direction with
                  respect to the positive y-axis.
    :type param: (float, float, float)
    :param bounds: Tuple with the boundaries of the separation (arcsec), angle (deg),
                   and contrast (mag). Each set of boundaries is specified as a tuple.
    :type bounds: ((float, float), (float, float), (float, float))
    :param images: Stack with images.
    :type images: numpy.ndarray
    :param psf: PSF template, either a single image (2D) or a cube (3D) with the dimensions
                equal to *images*.
    :type psf: numpy.ndarray
    :param mask: Array with the circular mask (zeros) of the central and outer regions.
    :type mask: numpy.ndarray
    :param parang: Array with the angles for derotation.
    :type parang: numpy.ndarray
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
    :param prior: Prior can be set to "flat" or "aperture". With "flat", the values of *bounds*
                  are used as uniform priors. With "aperture", the prior probability is set to
                  zero beyond the aperture and unity within the aperture.
    :type prior: str

    :return: Log posterior probability.
    :rtype: float
    """

    def _lnprior():
        """
        Internal function for the log prior function.

        :return: Log prior.
        :rtype: float
        """

        if prior == "flat":

            if bounds[0][0] <= param[0] <= bounds[0][1] and \
               bounds[1][0] <= param[1] <= bounds[1][1] and \
               bounds[2][0] <= param[2] <= bounds[2][1]:

                ln_prior = 0.

            else:

                ln_prior = -np.inf

        elif prior == "aperture":

            x_pos, y_pos = polar_to_cartesian(images, param[0]/pixscale, param[1])

            delta_x = x_pos - aperture['pos_x']
            delta_y = y_pos - aperture['pos_y']

            if aperture['type'] == "circular":

                if math.sqrt(delta_x**2+delta_y**2) < aperture['radius'] and \
                   bounds[2][0] <= param[2] <= bounds[2][1]:

                    ln_prior = 0.

                else:

                    ln_prior = -np.inf

            elif aperture['type'] == "elliptical":

                cos_ang = math.cos(math.radians(180.-aperture['angle']))
                sin_ang = math.sin(math.radians(180.-aperture['angle']))

                x_rot = delta_x*cos_ang - delta_y*sin_ang
                y_rot = delta_x*sin_ang + delta_y*cos_ang

                r_check = (x_rot/aperture['semimajor'])**2 + (y_rot/aperture['semiminor'])**2

                if r_check <= 1. and bounds[2][0] <= param[2] <= bounds[2][1]:
                    ln_prior = 0.

                else:
                    ln_prior = -np.inf


        else:
            raise ValueError("Prior type not recognized.")

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

        _, im_res = pca_psf_subtraction(images=fake*mask,
                                        angles=-1.*parang+extra_rot,
                                        pca_number=pca_number,
                                        indices=indices)

        stack = combine_residuals(method="mean", res_rot=im_res)

        merit = merit_function(residuals=stack,
                               function="sum",
                               aperture=aperture,
                               sigma=0.)[0]

        return -0.5*merit

    ln_prior = _lnprior()

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + _lnlike()

    return ln_prob
