"""
Functions for PSF subtraction.
"""

from typing import Tuple

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA


#@profile
def pca_psf_subtraction(images: np.ndarray,
                        angles: np.ndarray,
                        pca_number: int,
                        pca_sklearn: PCA = None,
                        im_shape: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Function for PSF subtraction with PCA.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of images. Also used as reference images if `pca_sklearn` is set to None. Should be
        in the original 3D shape if `pca_sklearn` is set to None or in the 2D reshaped format if
        `pca_sklearn` is not set to None.
    parang : numpy.ndarray
        Derotation angles (deg).
    pca_number : int
        Number of principal components used for the PSF model.
    pca_sklearn : sklearn.decomposition.pca.PCA, None
        PCA object with the basis if not set to None.
    im_shape : tuple(int, int, int), None
        Original shape of the stack with images. Required if `pca_sklearn` is not set to None.

    Returns
    -------
    numpy.ndarray
        Derotated residuals of the PSF subtraction.
    numpy.ndarray
        Variance of the residuals before derotation for use in weighted combination.
    """

    if pca_sklearn is None:
        pca_sklearn = PCA(n_components=pca_number, svd_solver='arpack')

        im_shape = images.shape

        # reshape the images and select the unmasked pixels
        im_reshape = images.reshape(im_shape[0], im_shape[1]*im_shape[2])

        # subtract mean image
        im_reshape -= np.mean(im_reshape, axis=0)

        # create pca basis
        pca_sklearn.fit(im_reshape)

    else:
        im_reshape = images

    # create pca representation
    zeros = np.zeros((pca_sklearn.n_components - pca_number, im_reshape.shape[0]))
    #print('pca basis:',pca_sklearn.components_[:pca_number].flags)
    #print('data:',im_reshape.T.flags)
    #print()
    pca_rep = np.matmul(pca_sklearn.components_[:pca_number], im_reshape.T)
    pca_rep = np.vstack((pca_rep, zeros)).T

    # create original array size
    residuals = np.zeros((im_shape[0], im_shape[1]*im_shape[2]))

    # subtract the psf model

    # create psf model and subtract
    residuals = im_reshape - pca_sklearn.inverse_transform(pca_rep)

    # reshape to the original image size
    residuals = residuals.reshape(im_shape)

    # get variance of residuals for weighted combination
    res_var = np.var(residuals, axis=0)

    # derotate the images
    #res_rot = np.zeros(residuals.shape)
    for j, item in enumerate(angles):
        residuals[j, ] = rotate(residuals[j, ], item, reshape=False)

    return residuals, res_var
