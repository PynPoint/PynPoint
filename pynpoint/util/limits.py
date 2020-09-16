"""
Functions for calculating detection limits.
"""
import math
import os
from typing import Tuple
import numpy as np

from photutils import aperture_photometry, CircularAperture
from typeguard import typechecked

from pynpoint.util.analysis import student_t, fake_planet, false_alarm
from pynpoint.util.image import polar_to_cartesian, center_subpixel
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals
from pynpoint.util.paco import PACO, FastPACO, FullPACO
<<<<<<< HEAD
from pynpoint.util.pacomath import *
=======
>>>>>>> master

@typechecked
def contrast_limit(path_images: str,
                   path_psf: str,
                   noise: np.ndarray,
                   mask: np.ndarray,
                   parang: np.ndarray,
                   psf_scaling: float,
                   extra_rot: float,
                   pca_number: int,
                   threshold: Tuple[str, float],
                   aperture: float,
                   residuals: str,
                   snr_inject: float,
                   position: Tuple[float, float]) -> Tuple[float, float, float, float]:

    """
    Function for calculating the contrast limit at a specified position for a given sigma level or
    false positive fraction, both corrected for small sample statistics.

    Parameters
    ----------
    path_images : str
        System location of the stack of images (3D).
    path_psf : str
        System location of the PSF template for the fake planet (3D). Either a single image or a
        stack of images equal in size to science data.
    noise : numpy.ndarray
        Residuals of the PSF subtraction (3D) without injection of fake planets. Used to measure
        the noise level with a correction for small sample statistics.
    mask : numpy.ndarray
        Mask (2D).
    parang : numpy.ndarray
        Derotation angles (deg).
    psf_scaling : float
        Additional scaling factor of the planet flux (e.g., to correct for a neutral density
        filter). Should have a positive value.
    extra_rot : float
        Additional rotation angle of the images in clockwise direction (deg).
    pca_number : int
        Number of principal components used for the PSF subtraction.
    threshold : tuple(str, float)
        Detection threshold for the contrast curve, either in terms of 'sigma' or the false
        positive fraction (FPF). The value is a tuple, for example provided as ('sigma', 5.) or
        ('fpf', 1e-6). Note that when sigma is fixed, the false positive fraction will change with
        separation. Also, sigma only corresponds to the standard deviation of a normal distribution
        at large separations (i.e., large number of samples).
    aperture : float
        Aperture radius (pix) for the calculation of the false positive fraction.
    residuals : str
        Method used for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
    snr_inject : float
        Signal-to-noise ratio of the injected planet signal that is used to measure the amount
        of self-subtraction.
    position : tuple(float, float)
        The separation (pix) and position angle (deg) of the fake planet.

    Returns
    -------
    float
        Separation (pix).
    float
        Position angle (deg).
    float
        Contrast (mag).
    float
        False positive fraction.
    """

    images = np.load(path_images)
    psf = np.load(path_psf)

    if threshold[0] == 'sigma':
        sigma = threshold[1]

        # Calculate the FPF for a given sigma level
        fpf = student_t(t_input=threshold,
                        radius=position[0],
                        size=aperture,
                        ignore=False)

    elif threshold[0] == 'fpf':
        fpf = threshold[1]

        # Calculate the sigma level for a given FPF
        sigma = student_t(t_input=threshold,
                          radius=position[0],
                          size=aperture,
                          ignore=False)

    else:
        raise ValueError('Threshold type not recognized.')

    # Cartesian coordinates of the fake planet
    yx_fake = polar_to_cartesian(images, position[0], position[1]-extra_rot)

    # Determine the noise level
    _, t_noise, _, _ = false_alarm(image=noise[0, ],
                                   x_pos=yx_fake[1],
                                   y_pos=yx_fake[0],
                                   size=aperture,
                                   ignore=False)

    # Aperture properties
    im_center = center_subpixel(images)

    # Measure the flux of the star
    ap_phot = CircularAperture((im_center[1], im_center[0]), aperture)
    phot_table = aperture_photometry(psf_scaling*psf[0, ], ap_phot, method='exact')
    star = phot_table['aperture_sum'][0]

    # Magnitude of the injected planet
    flux_in = snr_inject*t_noise
    mag = -2.5*math.log10(flux_in/star)

    # Inject the fake planet
    fake = fake_planet(images=images,
                       psf=psf,
                       parang=parang,
                       position=(position[0], position[1]),
                       magnitude=mag,
                       psf_scaling=psf_scaling)

    # Run the PSF subtraction
    _, im_res = pca_psf_subtraction(images=fake*mask,
                                    angles=-1.*parang+extra_rot,
                                    pca_number=pca_number)

    # Stack the residuals
    im_res = combine_residuals(method=residuals, res_rot=im_res)

    # Measure the flux of the fake planet
    flux_out, _, _, _ = false_alarm(image=im_res[0, ],
                                    x_pos=yx_fake[1],
                                    y_pos=yx_fake[0],
                                    size=aperture,
                                    ignore=False)

    # Calculate the amount of self-subtraction
    attenuation = flux_out/flux_in

    # Calculate the detection limit
    contrast = sigma*t_noise/(attenuation*star)

    # The flux_out can be negative, for example if the aperture includes self-subtraction regions
    if contrast > 0.:
        contrast = -2.5*math.log10(contrast)
    else:
        contrast = np.nan

    # Separation [pix], position antle [deg], contrast [mag], FPF
    return  position[0], position[1], contrast, fpf


def paco_contrast_limit(path_images,
                        path_psf,
                        noise,
                        parang,
                        psf_rad,
                        psf_scaling,
                        res_scaling,
                        pixscale,
                        extra_rot,
                        threshold,
                        aperture,
                        snr_inject,
                        position,
                        algorithm):

    """
    Function for calculating the contrast limit at a specified position for a given sigma level or
    false positive fraction, both corrected for small sample statistics.

    Parameters
    ----------
    path_images : str
        System location of the stack of images (3D).
    path_psf : str
        System location of the PSF template for the fake planet (3D). Either a single image or a
        stack of images equal in size to science data.
    noise : numpy.ndarray
        Residuals of the PSF subtraction (3D) without injection of fake planets. Used to measure
        the noise level with a correction for small sample statistics.
    parang : numpy.ndarray
        Derotation angles (deg).
    psf_scaling : float
        Additional scaling factor of the planet flux (e.g., to correct for a neutral density
        filter). Should have a positive value.
    extra_rot : float
        Additional rotation angle of the images in clockwise direction (deg).
    threshold : tuple(str, float)
        Detection threshold for the contrast curve, either in terms of "sigma" or the false
        positive fraction (FPF). The value is a tuple, for example provided as ("sigma", 5.) or
        ("fpf", 1e-6). Note that when sigma is fixed, the false positive fraction will change with
        separation. Also, sigma only corresponds to the standard deviation of a normal distribution
        at large separations (i.e., large number of samples).
    aperture : float
        Aperture radius (pix) for the calculation of the false positive fraction.
    position : tuple(float, float)
        The separation (pix) and position angle (deg) of the fake planet.
    snr_inject : float
        Signal-to-noise ratio of the injected planet signal that is used to measure the amount
        of self-subtraction.

    Returns
    -------
    NoneType
        None
    """
    images = np.load(path_images)
    psf = np.load(path_psf)

    if threshold[0] == "sigma":
        sigma = threshold[1]

        # Calculate the FPF for a given sigma level
        fpf = student_t(t_input=threshold,
                        radius=position[0],
                        size=aperture,
                        ignore=False)

    elif threshold[0] == "fpf":
        fpf = threshold[1]

        # Calculate the sigma level for a given FPF
        sigma = student_t(t_input=threshold,
                          radius=position[0],
                          size=aperture,
                          ignore=False)

    else:
        raise ValueError("Threshold type not recognized.")

    # Cartesian coordinates of the fake planet
    xy_fake = polar_to_cartesian(images, position[0], position[1]-extra_rot)

    # Determine the noise level
    _, t_noise, _, _ = false_alarm(image=noise,
                                   x_pos=xy_fake[0],
                                   y_pos=xy_fake[1],
                                   size=aperture,
                                   ignore=False)

    # Aperture properties
    im_center = center_subpixel(images)

    # Measure the flux of the star
    ap_phot = CircularAperture((im_center[1], im_center[0]), aperture)
    phot_table = aperture_photometry(psf_scaling*psf[0, ], ap_phot, method='exact')
    star = phot_table['aperture_sum'][0]

    # Magnitude of the injected planet
    flux_in = snr_inject*t_noise
    mag = -2.5*math.log10(flux_in/star)

    # Inject the fake planet
    fake = fake_planet(images=images,
                       psf=psf,
                       parang=parang,
                       position=(position[0], position[1]),
                       magnitude=mag,
                       psf_scaling=psf_scaling)
    path_fake_planet = os.path.split(path_images)[0] = "/"
    np.save(path_fake_planet+"injected.npy", fake)
    # Run the PSF subtraction
    #_, im_res = pca_psf_subtraction(images=fake*mask,
    #                                angles=-1.*parang+extra_rot,
    #                                pca_number=pca_number)

    # Stack the residuals
    #im_res = combine_residuals(method=residuals, res_rot=im_res)

    # Measure the flux of the fake planet
    #flux_out, _, _, _ = false_alarm(image=im_res[0, ],
    #                                x_pos=xy_fake[0],
    #                                y_pos=xy_fake[1],
    #                                size=aperture,
    #                                ignore=False)
    # Setup PACO
    if algorithm == "fastpaco":
        fp = FastPACO(image_file=path_fake_planet + "injected.npy",
                      angles=parang,
                      psf=psf,
                      psf_rad=psf_rad,
                      px_scale=pixscale,
                      res_scale=res_scaling,
                      verbose=False)
    elif algorithm == "fullpaco":
        fp = FullPACO(image_file=path_fake_planet + "injected.npy",
                      angles=parang,
                      psf=psf,
                      psf_rad=psf_rad,
                      px_scale=pixscale,
                      res_scale=res_scaling,
                      verbose=False)

    # Run PACO
    # Restrict to 1 processor since this module is called from a child process
    a, b = fp.PACO(cpu=1)

    # Should do something with these?
    snr = b/np.sqrt(a)
    flux_residual = b/a
    flux_out, _, _, _ = false_alarm(image=flux_residual,
                                    x_pos=xy_fake[0],
                                    y_pos=xy_fake[1],
                                    size=aperture,
                                    ignore=False)

    # Iterative, unbiased flux estimation
    # This doesn't seem to give the correct results yet?
    #if self.m_flux_calc:
    #    phi0s = fp.thresholdDetection(snr,self.m_threshold)
    #    init = np.array([flux[int(phi0[0]),int(phi0[1])] for phi0 in phi0s])
    #    ests =  np.array(fp.fluxEstimate(phi0s,self.m_eps,init))

    # Calculate the amount of self-subtraction
    attenuation = flux_out/flux_in

    # Calculate the detection limit
    contrast = sigma*t_noise/(attenuation*star)

    # The flux_out can be negative, for example if the aperture includes self-subtraction regions
    if contrast > 0.:
        contrast = -2.5*math.log10(contrast)
    else:
        contrast = np.nan

    # Separation [pix], position antle [deg], contrast [mag], FPF
    return (position[0], position[1], contrast, fpf)
