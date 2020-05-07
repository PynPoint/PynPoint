#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 08:27:30 2019

@author: kiefer
"""

from typing import Union

import numpy as np

from typeguard import typechecked
from sklearn.decomposition import PCA
from scipy.ndimage import rotate

from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.ifs import sdi_scaling


@typechecked
def postprocessor(images: np.ndarray,
                  angles: np.ndarray,
                  scales: np.ndarray,
                  pca_number: Union[int, np.int64, tuple],
                  pca_sklearn: PCA = None,
                  im_shape: Union[None, tuple] = None,
                  indices: np.ndarray = None,
                  mask: np.ndarray = None,
                  processing_type: str = 'Oadi'):

    """
    Function to apply different kind of post processings. If processing_type = \'Cadi\'
    and mask = None, it is equivalent to pynpoint.util.psf.pca_psf_subtraction.

    Parameters
    ----------
    images : numpy.array
        Input images which should be reduced.
    angles : numpy.ndarray
        Derotation angles (deg).
    scales : numpy.array
        Scaling factors
    pca_number : Union[int, np.int64, tuple]
        Number of principal components used for the PSF subtraction.
    pca_sklearn : sklearn.decomposition.pca.PCA, None
        PCA object with the basis if not set to None.
    im_shape : tuple(int, int, int), None
        Original shape of the stack with images. Required if `pca_sklearn` is not set to None.
    indices : numpy.ndarray, None
        Non-masked image indices. All pixels are used if set to None.
    mask : numpy.ndarray
        Mask (2D).
    processing_type : str
        Type of post processing. Currently supported:
            Tnan: Applaying no PCA reduction and returning one wavelength avaraged image
            Wnan: Applaying no PCA reduction and returing one image per Wavelengths
            Oadi: Applaying ADI without using the wavelength informtation
            Tadi: Applaying ADI and creturning one wavelength avaraged image
            Wadi: Applaying ADI and returing one image per Wavelengths
            Tsdi: Applaying SDI and returning one wavelength avaraged image
            Wsdi: Applaying SDI and returing one image per Wavelengths
            Tsaa: Applaying SDI and ADI simultaniously and returning one wavelength avaraged image
            Wsaa: Applaying SDI and ADI simultaniously and returing one image per Wavelengths
            Tsap: Applaying SDI then ADI and returning one wavelength avaraged image
            Wsap: Applaying SDI then ADI and returing one image per Wavelengths
            Tasp: Applaying ADI then SDI and returning one wavelength avaraged image
            Wasp: Applaying ADI then SDI and returing one image per Wavelengths

    Returns
    -------
    numpy.ndarray
        Residuals of the PSF subtraction.
    numpy.ndarray
        Derotated residuals of the PSF subtraction.

    """

    # set up pca_numbers and check if they have the right dimensions
    if not isinstance(pca_number, tuple):
        pca_number = (pca_number, -1)
        if processing_type in ['Wsap', 'Tsap', 'Wasp', 'Tasp']:
            raise ValueError('The processing type ' + processing_type +
                             'requires a tuple of pca numbers.')

    # fall back to default mask if none is given
    if mask is None:
        mask = 1.

    # Set up output arrays
    res_raw = np.zeros(images.shape)
    res_rot = np.zeros(images.shape)

    # ----------------------------------------- List of different processing
    # No reduction
    if processing_type in ['Wnan', 'Tnan']:
        res_raw = images
        for j, item in enumerate(angles):
            if images.ndim == 4:
                for k, ima in enumerate(images[:, j]):
                    res_rot[k, j] = rotate(ima, item, reshape=False)
            else:
                res_rot[j] = rotate(images[j], item, reshape=False)

    # Wavelength independendt adi
    elif processing_type in ['Oadi']:
        res_raw, res_rot = pca_psf_subtraction(images=images*mask,
                                               angles=angles,
                                               scales=np.array([None]),
                                               pca_number=int(pca_number[0]),
                                               pca_sklearn=pca_sklearn,
                                               im_shape=im_shape,
                                               indices=indices)

    # Wavelength specific adi
    elif processing_type in ['Wadi', 'Tadi']:
        for i, _ in enumerate(images):
            res_raw_i, res_rot_i = pca_psf_subtraction(images=images[i]*mask,
                                                       angles=angles,
                                                       scales=np.array([None]),
                                                       pca_number=int(pca_number[0]),
                                                       pca_sklearn=pca_sklearn,
                                                       im_shape=im_shape,
                                                       indices=indices)
            res_raw[i] = res_raw_i
            res_rot[i] = res_rot_i

    # time specific sdi
    elif processing_type in ['Wsdi', 'Tsdi']:
        for i, _ in enumerate(images[0]):
            im_scaled, _, _ = sdi_scaling(images[:, i], scales)
            res_raw_i, res_rot_i = pca_psf_subtraction(images=im_scaled*mask,
                                                       angles=angles[i]*np.ones_like(scales),
                                                       scales=scales,
                                                       pca_number=int(pca_number[0]),
                                                       pca_sklearn=pca_sklearn,
                                                       im_shape=im_shape,
                                                       indices=indices)
            res_raw[:, i] = res_raw_i
            res_rot[:, i] = res_rot_i

    # SDI and ADI simultaniously
    elif processing_type in ['Wsaa', 'Tsaa']:
        im_scaled = np.zeros((im_shape[0]*im_shape[1], im_shape[2], im_shape[3]))
        scales_flat = np.zeros((len(images)*len(images[0])))
        angles_flat = np.zeros((len(images)*len(images[0])))

        # alligne data, wavlengths and parangs on one axis as it is required for the
        # pca_psf_reduction function.
        for i, _ in enumerate(images[0]):
            im_scaled[i*len(images):(i+1)*len(images)] = sdi_scaling(images[:, i], scales)[0]
            angles_flat[i*len(images):(i+1)*len(images)] = angles[i]*np.ones_like(scales)
            scales_flat[i*len(images):(i+1)*len(images)] = scales

        # apply PCA PSF subtraction using the whole datacube at once
        res_raw_i, res_rot_i = pca_psf_subtraction(images=im_scaled*mask,
                                                   angles=angles_flat,
                                                   scales=scales_flat,
                                                   pca_number=int(pca_number[0]),
                                                   pca_sklearn=pca_sklearn,
                                                   im_shape=im_shape,
                                                   indices=indices)

        image_shape = images.shape
        res_raw = res_raw_i.reshape(image_shape)
        res_rot = res_rot_i.reshape(image_shape)

    # SDI then ADI
    elif processing_type in ['Wsap', 'Tsap']:
        # SDI step
        res_raw_int = np.zeros(res_raw.shape)
        for i, _ in enumerate(images[0]):
            im_scaled, _, _ = sdi_scaling(images[:, i], scales)
            res_raw_i, _ = pca_psf_subtraction(images=im_scaled*mask,
                                               angles=np.array([None]),
                                               scales=scales,
                                               pca_number=int(pca_number[0]),
                                               pca_sklearn=pca_sklearn,
                                               im_shape=im_shape,
                                               indices=indices)
            res_raw_int[:, i] = res_raw_i

        # ADI step
        for j, _ in enumerate(images):
            res_raw_i, res_rot_i = pca_psf_subtraction(images=res_raw_int[j]*mask,
                                                       angles=angles,
                                                       scales=np.array([None]),
                                                       pca_number=int(pca_number[1]),
                                                       pca_sklearn=pca_sklearn,
                                                       im_shape=im_shape,
                                                       indices=indices)
            res_raw[j] = res_raw_i
            res_rot[j] = res_rot_i

    # ADI then SDI
    elif processing_type in ['Wasp', 'Tasp']:
        # ADI step
        res_raw_int = np.zeros(res_raw.shape)
        for j, _ in enumerate(images):
            res_raw_i, _ = pca_psf_subtraction(images=images[j]*mask,
                                               angles=np.array([None]),
                                               scales=np.array([None]),
                                               pca_number=int(pca_number[0]),
                                               pca_sklearn=pca_sklearn,
                                               im_shape=im_shape,
                                               indices=indices)
            res_raw_int[j] = res_raw_i

        # SDI step
        for i, _ in enumerate(images[0]):
            im_scaled, _, _ = sdi_scaling(res_raw_int[:, i], scales)
            res_raw_i, res_rot_i = pca_psf_subtraction(images=im_scaled*mask,
                                                       angles=angles[i]*np.ones_like(scales),
                                                       scales=scales,
                                                       pca_number=int(pca_number[1]),
                                                       pca_sklearn=pca_sklearn,
                                                       im_shape=im_shape,
                                                       indices=indices)
            res_raw[:, i] = res_raw_i
            res_rot[:, i] = res_rot_i

    else:
        # Error message if unknown processing type
        error_msg = 'Processing type ' + processing_type + ' is not supported'
        raise ValueError(error_msg)

    return res_raw, res_rot
