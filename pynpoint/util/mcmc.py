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
           prior,
           variance,
           residuals):
    """
    Function for the log posterior function. Should be placed at the highest level of the
    Python module to be pickable for the multiprocessing.

    param : tuple(float, float, float)
        Tuple with the separation (arcsec), angle (deg), and contrast (mag). The angle is measured
        in counterclockwise direction with respect to the positive y-axis.
    bounds : tuple(tuple(float, float), tuple(float, float), tuple(float, float))
        Tuple with the boundaries of the separation (arcsec), angle (deg), and contrast (mag). Each
        set of boundaries is specified as a tuple.
    images : numpy.ndarray
        Stack with images.
    psf : numpy.ndarray
        PSF template, either a single image (2D) or a cube (3D) with the dimensions equal to
        *images*.
    mask : numpy.ndarray
        Array with the circular mask (zeros) of the central and outer regions.
    parang : numpy.ndarray
        Array with the angles for derotation.
    psf_scaling : float
        Additional scaling factor of the planet flux (e.g., to correct for a neutral density
        filter). Should be negative in order to inject negative fake planets.
    pixscale : float
        Additional scaling factor of the planet flux (e.g., to correct for a neutral density
        filter). Should be negative in order to inject negative fake planets.
    pca_number : int
        Number of principal components used for the PSF subtraction.
    extra_rot : float
        Additional rotation angle of the images (deg).
    aperture : dict
        Dictionary with the aperture properties. See for more information
        :func:`~pynpoint.util.analysis.create_aperture`.
    indices : numpy.ndarray
        Non-masked image indices.
    prior : str
        Prior can be set to "flat" or "aperture". With "flat", the values of *bounds* are used
        as uniform priors. With "aperture", the prior probability is set to zero beyond the
        aperture and unity within the aperture.
    variance : tuple(str, float)
        Variance type and value for the likelihood function. The value is set to None in case
        a Poisson distribution is assumed.
    residuals : str
        Method used for combining the residuals ("mean", "median", "weighted", or "clipped").

    Returns
    -------
    float
        Log posterior probability.
    """

    def _lnprior():
        """
        Internal function for the log prior function.

        Returns
        -------
        float
            Log prior.
        """

        if prior == "flat":

            if bounds[0][0] <= param[0] <= bounds[0][1] and \
               bounds[1][0] <= param[1] <= bounds[1][1] and \
               bounds[2][0] <= param[2] <= bounds[2][1]:

                ln_prior = 0.

            else:

                ln_prior = -np.inf

        elif prior == "aperture":

            xy_pos = polar_to_cartesian(images, param[0]/pixscale, param[1])

            delta_x = xy_pos[0] - aperture['pos_x']
            delta_y = xy_pos[1] - aperture['pos_y']

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
        either a Poisson distribution (see Wertz et al. 2017) or a Gaussian distribution with a
        correction for small sample statistics (see Mawet et al. 2014).

        Returns
        -------
        float
            Log likelihood.
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

        stack = combine_residuals(method=residuals, res_rot=im_res)

        merit = merit_function(residuals=stack[0, ],
                               function="sum",
                               variance=variance,
                               aperture=aperture,
                               sigma=0.)

        return -0.5*merit

    ln_prior = _lnprior()

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + _lnlike()

    return ln_prob
