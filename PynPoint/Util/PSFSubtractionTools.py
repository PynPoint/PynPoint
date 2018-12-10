"""
Functions for PSF subtraction.
"""

from __future__ import absolute_import

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA


def pca_psf_subtraction(images,
                        angles,
                        pca_number,
                        pca_sklearn=None,
                        im_shape=None,
                        indices=None):
    """
    Function for PSF subtraction with PCA.

    :param images: Stack of images. Also used as reference images if pca_sklearn is set to None.
                   Should be in the original 3D shape if pca_sklearn is set to None or in the 2D
                   reshaped format if pca_sklearn is not None.
    :type images: numpy.ndarray
    :param parang: Derotation angles (deg).
    :type parang: numpy.ndarray
    :param pca_number: Number of principal components used for the PSF model.
    :type pca_number: int
    :param pca_sklearn: PCA object with the basis if not set to None.
    :type pca_sklearn: sklearn.decomposition.pca.PCA
    :param im_shape: Original shape of the stack with images. Required if pca_sklearn is not
                     set to None.
    :type im_shape: tuple(int, int, int)
    :param indices: Non-masked image indices, required if pca_sklearn is not set to None. Optional
                    if pca_sklearn is set to None.
    :type indices: numpy.ndarray

    :return: Mean residuals of the PSF subtraction.
    :rtype: ndarray
    """

    if pca_sklearn is None:
        pca_sklearn = PCA(n_components=pca_number, svd_solver="arpack")

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
    residuals[:, indices] = im_reshape - psf_model

    # reshape to the original image size
    residuals = residuals.reshape(im_shape)

    # derotate the images
    res_rot = np.zeros(residuals.shape)
    for j, item in enumerate(angles):
        res_rot[j, ] = rotate(residuals[j, ], item, reshape=False)

    return residuals, res_rot
