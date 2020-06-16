"""
Functions for PSF subtraction.
"""

from typing import Optional, Tuple

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from typeguard import typechecked


@typechecked
def pca_psf_subtraction(images: np.ndarray,
                        angles: np.ndarray,
                        pca_number: int,
                        pca_sklearn: Optional[PCA] = None,
                        im_shape: Optional[Tuple[int, int, int]] = None,
                        indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for PSF subtraction with PCA.

    Parameters
    ----------
    images : np.ndarray
        Stack of images. Also used as reference images if ```pca_sklearn``` is set to None. The
        data should have the original 3D shape if ``pca_sklearn`` is set to None or it should be
        in a 2D reshaped format if ``pca_sklearn`` is not set to None.
    angles : np.ndarray
        Parallactic angles (deg).
    pca_number : int
        Number of principal components.
    pca_sklearn : sklearn.decomposition.pca.PCA, None
        PCA object with the principal components.
    im_shape : tuple(int, int, int), None
        The original 3D shape of the stack with images. Only required if ``pca_sklearn`` is not set
        to None.
    indices : np.ndarray, None
        Array with the indices of the pixels that are used for the PSF subtraction. All pixels are
        used if set to None.

    Returns
    -------
    np.ndarray
        Residuals of the PSF subtraction.
    np.ndarray
        Derotated residuals of the PSF subtraction.
    """

    if pca_sklearn is None:
        # Create a PCA object if not provided as argument
        pca_sklearn = PCA(n_components=pca_number, svd_solver='arpack')

        # The 3D shape of the array with images
        im_shape = images.shape

        if indices is None:
            # Select the first image and get the unmasked image indices
            im_star = images[0, ].reshape(-1)
            indices = np.where(im_star != 0.)[0]

        # Reshape the images and select the unmasked pixels
        im_reshape = images.reshape(im_shape[0], im_shape[1]*im_shape[2])
        im_reshape = im_reshape[:, indices]

        # Subtract the mean image
        # This is also done by sklearn.decomposition.PCA.fit()
        im_reshape -= np.mean(im_reshape, axis=0)

        # Fit the principal components
        pca_sklearn.fit(im_reshape)

    else:
        # If the PCA object is already there then so are the reshaped data
        im_reshape = np.copy(images)

    # Project the data on the principal components
    # Note that this is the same as sklearn.decomposition.PCA.transform()
    # It is harcoded because the number of components has been adjusted
    pca_rep = np.matmul(pca_sklearn.components_[:pca_number], im_reshape.T)

    # The zeros are added with vstack to account for the components that have not been used for the
    # transformation to the lower-dimensional space, while they were initiated with the PCA object.
    # Since inverse_transform uses the number of initial components, the zeros are added for
    # components > pca_number. These components do not impact the inverse transformation.
    zeros = np.zeros((pca_sklearn.n_components - pca_number, im_reshape.shape[0]))
    pca_rep = np.vstack((pca_rep, zeros)).T

    # Transform the data back to the original space
    psf_model = pca_sklearn.inverse_transform(pca_rep)

    # Create an array with the original shape
    residuals = np.zeros((im_shape[0], im_shape[1]*im_shape[2]))

    # Select all pixel indices if set to None
    if indices is None:
        indices = np.arange(0, im_reshape.shape[1], 1)

    # Subtract the PSF model
    residuals[:, indices] = im_reshape - psf_model

    # Reshape the residuals to the original shape
    residuals = residuals.reshape(im_shape)

    # Check if the number of PARANG is equal to the number of images
    if residuals.shape[0] != angles.shape[0]:
        raise ValueError(f'The number of images ({residuals.shape[0]}) is not equal to the '
                         f'number of parallactic angles ({angles.shape[0]}).')

    # Derotate the images
    res_derot = np.zeros(residuals.shape)
    for j, item in enumerate(angles):
        res_derot[j, ] = rotate(residuals[j, ], item, reshape=False)

    return residuals, res_derot
