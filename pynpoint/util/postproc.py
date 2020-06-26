"""
Functions for post-processing.
"""

from typing import Union, Optional, Tuple

import numpy as np

from typeguard import typechecked
from sklearn.decomposition import PCA

from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.sdi import sdi_scaling


@typechecked
def postprocessor(images: np.ndarray,
                  angles: np.ndarray,
                  scales: Optional[np.ndarray],
                  pca_number: Union[int, Tuple[Union[int, np.int64], Union[int, np.int64]]],
                  pca_sklearn: PCA = None,
                  im_shape: Union[None, tuple] = None,
                  indices: np.ndarray = None,
                  mask: np.ndarray = None,
                  processing_type: str = 'ADI'):

    """
    Function to apply different kind of post processings. It is equivalent to
    :func:`~pynpoint.util.psf.pca_psf_subtraction` if ``processing_type='ADI'` and
    ``mask=None``.

    Parameters
    ----------
    images : np.array
        Input images which should be reduced.
    angles : np.ndarray
        Derotation angles (deg).
    scales : np.array
        Scaling factors
    pca_number : tuple(int, int)
        Number of principal components used for the PSF subtraction.
    pca_sklearn : sklearn.decomposition.pca.PCA, None
        PCA object with the basis if not set to None.
    im_shape : tuple(int, int, int), None
        Original shape of the stack with images. Required if ``pca_sklearn`` is not set to None.
    indices : np.ndarray, None
        Non-masked image indices. All pixels are used if set to None.
    mask : np.ndarray
        Mask (2D).
    processing_type : str
        Post-processing type:
            - ADI: Angular differential imaging.
            - SDI: Spectral differential imaging.
            - SDI+ADI: Spectral and angular differential imaging.
            - ADI+SDI: Angular and spectral differential imaging.

    Returns
    -------
    np.ndarray
        Residuals of the PSF subtraction.
    np.ndarray
        Derotated residuals of the PSF subtraction.
    """

    if not isinstance(pca_number, tuple):
        pca_number = (pca_number, -1)

    if mask is None:
        mask = 1.

    res_raw = np.zeros(images.shape)
    res_rot = np.zeros(images.shape)

    if processing_type == 'ADI':
        if images.ndim == 2:
            res_raw, res_rot = pca_psf_subtraction(images=images*mask,
                                                   angles=angles,
                                                   scales=None,
                                                   pca_number=pca_number[0],
                                                   pca_sklearn=pca_sklearn,
                                                   im_shape=im_shape,
                                                   indices=indices)

        elif images.ndim == 4:
            for i in range(images.shape[0]):
                res_raw[i, ], res_rot[i, ] = pca_psf_subtraction(images=images[i, ]*mask,
                                                                 angles=angles,
                                                                 scales=None,
                                                                 pca_number=pca_number[0],
                                                                 pca_sklearn=pca_sklearn,
                                                                 im_shape=im_shape,
                                                                 indices=indices)

    elif processing_type == 'SDI':
        for i in range(images.shape[1]):
            im_scaled = sdi_scaling(images[:, i, :, :], scales)

            res_raw[:, i], res_rot[:, i] = pca_psf_subtraction(images=im_scaled*mask,
                                                               angles=np.full(scales.size,
                                                                              angles[i]),
                                                               scales=scales,
                                                               pca_number=pca_number[0],
                                                               pca_sklearn=pca_sklearn,
                                                               im_shape=im_shape,
                                                               indices=indices)

    elif processing_type == 'SDI+ADI':
        # SDI
        res_raw_int = np.zeros(res_raw.shape)

        for i in range(images.shape[1]):
            im_scaled = sdi_scaling(images[:, i], scales)

            res_raw_int[:, i], _ = pca_psf_subtraction(images=im_scaled*mask,
                                                       angles=None,
                                                       scales=scales,
                                                       pca_number=pca_number[0],
                                                       pca_sklearn=pca_sklearn,
                                                       im_shape=im_shape,
                                                       indices=indices)

        # ADI
        for i in range(images.shape[0]):
            res_raw[i], res_rot[i] = pca_psf_subtraction(images=res_raw_int[i]*mask,
                                                         angles=angles,
                                                         scales=None,
                                                         pca_number=pca_number[1],
                                                         pca_sklearn=pca_sklearn,
                                                         im_shape=im_shape,
                                                         indices=indices)

    elif processing_type == 'ADI+SDI':
        # ADI
        res_raw_int = np.zeros(res_raw.shape)

        for i in range(images.shape[0]):
            res_raw_int[i], _ = pca_psf_subtraction(images=images[i, ]*mask,
                                                    angles=None,
                                                    scales=None,
                                                    pca_number=pca_number[0],
                                                    pca_sklearn=pca_sklearn,
                                                    im_shape=im_shape,
                                                    indices=indices)

        # SDI
        for i in range(images.shape[1]):
            im_scaled = sdi_scaling(res_raw_int[:, i], scales)

            res_raw[:, i], res_rot[:, i] = pca_psf_subtraction(images=im_scaled*mask,
                                                               angles=np.full(scales.size,
                                                                              angles[i]),
                                                               scales=scales,
                                                               pca_number=pca_number[1],
                                                               pca_sklearn=pca_sklearn,
                                                               im_shape=im_shape,
                                                               indices=indices)

    return res_raw, res_rot
