"""
Functions for PSF subtraction.
"""

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA


def pca_psf_subtraction(images,
                        angles,
                        pca_number):
    """
    Function for PSF subtraction with PCA.

    :param images: Stack of images, also used as reference images.
    :type images: ndarray
    :param parang: Derotation angles (deg).
    :type parang: ndarray
    :param pca_number: Number of principle components used for the PSF model.
    :type pca_number: int

    :return: Mean residuals of the PSF subtraction.
    :rtype: ndarray
    """

    pca = PCA(n_components=pca_number, svd_solver="arpack")

    # original image shape
    im_shape = images.shape

    # select the first image and get the unmasked image indices
    im_star = images[0, ].reshape(-1)
    indices = np.where(im_star != 0.)[0]

    # reshape the images and select the unmasked pixels
    im_reshape = images.reshape(im_shape[0], im_shape[1]*im_shape[2])
    im_reshape = im_reshape[:, indices]

    # subtract mean image
    im_reshape -= np.mean(im_reshape, axis=0)

    pca.fit(im_reshape)

    # create pca representation
    pca_rep = np.matmul(pca.components_[:pca_number], im_reshape.T)
    pca_rep = np.vstack((pca_rep, np.zeros((0, im_reshape.shape[0])))).T

    # create PSF model
    psf_model = pca.inverse_transform(pca_rep)

    # create original array size
    residuals = np.zeros((im_shape[0], im_shape[1]*im_shape[2]))

    # subtract the psf model
    residuals[:, indices] = im_reshape - psf_model

    # reshape to the original image size
    residuals = residuals.reshape(im_shape)

    for j, item in enumerate(angles):
        residuals[j, ] = rotate(residuals[j, ], item, reshape=False)

    return np.mean(residuals, axis=0)
