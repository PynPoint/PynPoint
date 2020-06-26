"""
Functions for PSF subtraction.
"""

from typing import Optional, Union, Tuple

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from typeguard import typechecked

from pynpoint.util.image import scale_image, shift_image


@typechecked
def pca_psf_subtraction(images: np.ndarray,
                        angles: Optional[np.ndarray],
                        pca_number: Union[int, np.int64],
                        scales: Optional[np.ndarray] = None,
                        pca_sklearn: Optional[PCA] = None,
                        im_shape: Optional[tuple] = None,
                        indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for PSF subtraction with PCA.

    Parameters
    ----------
    images : np.ndarray
        Stack of images. Also used as reference images if `pca_sklearn` is set to None. Should be
        in the original 3D shape if `pca_sklearn` is set to None or in the 2D reshaped format if
        `pca_sklearn` is not set to None.
    angles : np.ndarray, None
        Derotation angles (deg). The images are not derotated (e.g. for SDI) if set to None.
    pca_number : int
        Number of principal components used for the PSF model.
    scales : np.ndarray, None
        Scaling factors for SDI. Not used if set to None.
    pca_sklearn : sklearn.decomposition.pca.PCA, None
        PCA decomposition of the input data.
    im_shape : tuple(int, int, int), None
        Original shape of the stack with images. Required if `pca_sklearn` is not set to None.
    indices : np.ndarray, None
        Non-masked image indices. All pixels are used if set to None.

    Returns
    -------
    np.ndarray
        Residuals of the PSF subtraction.
    np.ndarray
        Derotated residuals of the PSF subtraction.
    """

    if pca_sklearn is None:
        pca_sklearn = PCA(n_components=pca_number, svd_solver='arpack')

        im_shape = images.shape

        if indices is None:
            # select the first image and get the unmasked image indices
            im_star = images[0, ].reshape(-1)
            indices = np.where(im_star != 0.)[0]

        # reshape the images and select the unmasked pixels
        im_reshape = images.reshape(im_shape[0], im_shape[1]*im_shape[2])
        im_reshape = im_reshape[:, indices]

        # subtract mean image
        im_reshape -= np.mean(im_reshape, axis=0)

        # create pca basis
        pca_sklearn.fit(im_reshape)

    else:
        im_reshape = np.copy(images)

    # create pca representation
    zeros = np.zeros((pca_sklearn.n_components - pca_number, im_reshape.shape[0]))
    pca_rep = np.matmul(pca_sklearn.components_[:pca_number], im_reshape.T)
    pca_rep = np.vstack((pca_rep, zeros)).T

    # create psf model
    psf_model = pca_sklearn.inverse_transform(pca_rep)

    # create original array size
    residuals = np.zeros((im_shape[0], im_shape[1]*im_shape[2]))

    # subtract the psf model
    if indices is None:
        indices = np.arange(0, im_reshape.shape[1], 1)

    residuals[:, indices] = im_reshape - psf_model

    # reshape to the original image size
    residuals = residuals.reshape(im_shape)

    # ----------- back scale images
    scal_cor = np.zeros(residuals.shape)

    if scales is not None:

        # check if the number of parang is equal to the number of images
        if residuals.shape[0] != scales.shape[0]:
            raise ValueError(f'The number of images ({residuals.shape[0]}) is not equal to the '
                             f'number of wavelengths ({scales.shape[0]}).')

        for i, _ in enumerate(scales):
            # rescaling the images
            swaps = scale_image(residuals[i, ], 1/scales[i], 1/scales[i])

            npix_del = scal_cor.shape[-1] - swaps.shape[-1]

            if npix_del == 0:
                scal_cor[i, ] = swaps

            else:
                if npix_del % 2 == 0:
                    npix_del_a = int(npix_del/2)
                    npix_del_b = int(npix_del/2)

                else:
                    npix_del_a = int((npix_del-1)/2)
                    npix_del_b = int((npix_del+1)/2)

                scal_cor[i, npix_del_a:-npix_del_b, npix_del_a:-npix_del_b] = swaps

                if npix_del % 2 == 1:
                    scal_cor[i, ] = shift_image(scal_cor[i, ], (0.5, 0.5), interpolation='spline')

    else:
        scal_cor = residuals

    res_rot = np.zeros(residuals.shape)

    if angles is not None:

        # Check if the number of parang is equal to the number of images
        if residuals.shape[0] != angles.shape[0]:
            raise ValueError(f'The number of images ({residuals.shape[0]}) is not equal to the '
                             f'number of parallactic angles ({angles.shape[0]}).')

        for j, item in enumerate(angles):
            res_rot[j, ] = rotate(scal_cor[j, ], item, reshape=False)

    else:
        res_rot = scal_cor

    return scal_cor, res_rot
