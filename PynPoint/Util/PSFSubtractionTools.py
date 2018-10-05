"""
Functions for PSF subtraction.
"""

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA


def pca_psf_subtraction(images,
                        parang,
                        pca_number,
                        extra_rot):
    """
    Function for PSF subtraction with PCA.

    :param images: Stack of images, also used as reference images.
    :type images: ndarray
    :param parang: Angles (deg) for derotation of the images.
    :type parang: ndarray
    :param pca_number: Number of PCA components used for the PSF model.
    :type pca_number: int
    :param extra_rot: Additional rotation angle of the images (deg).
    :type extra_rot: float

    :return: Mean residuals of the PSF subtraction.
    :rtype: ndarray
    """

    pca = PCA(n_components=pca_number, svd_solver="arpack")

    images -= np.mean(images, axis=0)
    images_reshape = images.reshape((images.shape[0], images.shape[1]*images.shape[2]))

    pca.fit(images_reshape)

    pca_rep = np.matmul(pca.components_[:pca_number], images_reshape.T)
    pca_rep = np.vstack((pca_rep, np.zeros((0, images.shape[0])))).T

    model = pca.inverse_transform(pca_rep)
    model = model.reshape(images.shape)

    residuals = images - model

    for j, item in enumerate(-1.*parang):
        residuals[j, ] = rotate(residuals[j, ], item+extra_rot, reshape=False)

    return np.mean(residuals, axis=0)
