"""
Functions for PSF subtraction.
"""

from typing import Optional, Tuple

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from typeguard import typechecked

from pynpoint.util.image import scale_image, center_pixel


@typechecked
def pca_psf_subtraction(images: np.ndarray,
                        angles: np.ndarray,
                        pca_number: int,
                        scales: np.ndarray = np.array([None]),
                        pca_sklearn: Optional[PCA] = None,
                        im_shape: Optional[tuple] = None,
                        indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for PSF subtraction with PCA.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of images. Also used as reference images if `pca_sklearn` is set to None. Should be
        in the original 3D shape if `pca_sklearn` is set to None or in the 2D reshaped format if
        `pca_sklearn` is not set to None.
    angles : numpy.ndarray
        Derotation angles (deg).
    pca_number : int
        Number of principal components used for the PSF model.
    pca_sklearn : sklearn.decomposition.pca.PCA, None
        PCA decomposition of the input data.
    im_shape : tuple(int, int, int), None
        Original shape of the stack with images. Required if `pca_sklearn` is not set to None.
    indices : numpy.ndarray, None
        Non-masked image indices. All pixels are used if set to None.

    Returns
    -------
    numpy.ndarray
        Residuals of the PSF subtraction.
    numpy.ndarray
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

    if scales[0] is not None:

        # check if the number of parang is equal to the number of images
        if residuals.shape[0] != scales.shape[0]:
            raise ValueError(f'The number of images ({residuals.shape[0]}) is not equal to the '
                             f'number of wavelengths ({scales.shape[0]}).')
        for i, _ in enumerate(scales):

            data_shape = min(len(residuals[i, 0]), len(residuals[i, :, 0]))
            down_shape = data_shape/scales[i]
            if down_shape >= data_shape-2:
                scal = 1
            else:
                if np.floor(down_shape) % 2 == 1:
                    scal = data_shape/np.floor(down_shape)
                else:
                    scal = data_shape/(np.floor(down_shape)+1)

            # rescaling the images
            swaps = scale_image(residuals[i, ], 1/scal, 1/scal)

            x_0, y_0 = center_pixel(scal_cor)
            x_1, y_1 = center_pixel(swaps)
            x_2 = x_0 - x_1
            y_2 = y_0 - y_1

            if y_2 == 0 or x_2 == 0:
                scal_cor[i] = swaps
            else:
                scal_cor[i, -y_2-swaps.shape[-2]:-y_2, -x_2-swaps.shape[-1]:-x_2] = swaps

    else:
        scal_cor = residuals

    # ----------- derotate the images
    res_rot = np.zeros(residuals.shape)

    if angles[0] is not None:

        # check if the number of parang is equal to the number of images
        if residuals.shape[0] != angles.shape[0]:
            raise ValueError(f'The number of images ({residuals.shape[0]}) is not equal to the '
                             f'number of parallactic angles ({angles.shape[0]}).')

        for j, item in enumerate(angles):
            res_rot[j, ] = rotate(scal_cor[j, ], item, reshape=False)

    else:
        res_rot = scal_cor

    return scal_cor, res_rot
