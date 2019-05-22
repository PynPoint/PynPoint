"""
Functions for PSF subtraction.
"""

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
    pca_sklearn : sklearn.decomposition.pca.PCA
        PCA object with the basis if not set to None.
    im_shape : tuple(int, int, int)
        Original shape of the stack with images. Required if `pca_sklearn` is not set to None.
    indices : numpy.ndarray
        Non-masked image indices, required if `pca_sklearn` is not set to None. Optional if
        `pca_sklearn` is set to None.

    Returns
    -------
    numpy.ndarray
        Residuals of the PSF subtraction.
    numpy.ndarray
        Derotated residuals of the PSF subtraction.
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


def iterative_pca_psf_subtraction(images,
                        angles,
                        pca_number,
                        pca_number_init,
                        indices=None):
    """
    Function for PSF subtraction with  iterative PCA.

    :param images: Stack of images. Also used as reference images if pca_sklearn is set to None.
                   Should be in the original 3D shape if pca_sklearn is set to None or in the 2D
                   reshaped format if pca_sklearn is not None.
    :type images: numpy.ndarray
    :param parang: Derotation angles (deg).
    :type parang: numpy.ndarray
    :param pca_number: Number of principal components used for the PSF model.
    :type pca_number: int
    :param pca_number_init: Number of principal component of first iteration
    :type pca_number_init: int
    :param indices: Non-masked image indices, required if pca_sklearn is not set to None. Optional
                    if pca_sklearn is set to None.
    :type indices: numpy.ndarray

    :return: Mean residuals of the PSF subtraction and the derotated but non-stacked residuals.
    :rtype: numpy.ndarray, numpy.ndarray
    """



    #pca_sklearn = PCA(n_components=pca_number, svd_solver="arpack")

    im_shape = images.shape
    

    if indices is None:
        # select the first image and get the unmasked image indices
        im_star = images[0, ].reshape(-1)
        indices = np.where(im_star != 0.)[0]

    # reshape the images and select the unmasked pixels
    im_reshape = images.reshape(im_shape[0], im_shape[1]*im_shape[2])
    #im_reshape = im_reshape[:, indices]

    # subtract mean image
    #im_reshape -= np.mean(im_reshape, axis=0)

    # create first iteration
    S = im_reshape - LRA(im_reshape, pca_number_init)
    for i in range(pca_number_init, pca_number+1):
        S = im_reshape - LRA(im_reshape-theta(red(S, im_shape, angles), images, angles), i)

    #
    # create original array size
    #residuals = np.zeros((im_shape[0], im_shape[1]*im_shape[2]))

    # subtract the psf model
    residuals = copy(S)

    # reshape to the original image size
    residuals = residuals.reshape(im_shape)


    # derotate the images
    res_rot = np.zeros(residuals.shape)
    for j, item in enumerate(angles):
        res_rot[j, ] = rotate(residuals[j, ], item, reshape=False)
    
    return residuals, res_rot#red(S, im_shape, angles), why does this not work?
    

def SVD(A):
    U, sigma, Vh = svd(A)
    #create corresponding matrix Sigma from list sigma
    Sigma = np.zeros((len(A), len(A[0])))
    for i in range(len(sigma)):
        Sigma[i][i] = sigma[i]
    return U, Sigma, Vh

def LRA(A, rank):
    U, Sigma, Vh = SVD(A)
    L = trimatmul(U[:, :rank], Sigma[:rank, :rank], Vh[:rank, :])
    return L

def red(S, im_shape, angles = None): #takes t x n^2 matrix S, reshapes it to cube S_cube and rotates each frame if angles list is given and returns mean of cube, i.e. processed frame
    S = np.reshape(S, (im_shape[0], im_shape[1], im_shape[2]))
    if angles is not None:
        for i in range(len(S)):
            S[i] = rotate(S[i], angles[i], reshape = False)
    return np.mean(S, axis = 0)

def theta(frame, original_cube, angles = None): #takes a (PCA processed) frame, sets negative parts of it to zero, reshapes it into t x n x n cube, rotates frames according to list and returns t x n^2 matrix
    d = frame.clip(min = 0)
    d = frame2cube(d, original_cube)
    if angles is not None:
        for i in range(len(d)):
            d[i] = rotate(d[i], -1*angles[i], reshape = False)
    d_shape = np.shape(d)
    d = np.reshape(d,(d_shape[0], d_shape[1]*d_shape[2]))
    return d

def trimatmul(A, B, C):
    return np.matmul(A, np.matmul(B, C))

def frame2cube(frame, original_cube): #takes a (PCA processed) n x n frame and returns a t x n x n cube with t copies of it, gets value of t from length of original cube
    t = len(original_cube)
    return np.stack([frame]*t, axis = 0)

def cube2mat(A): #take t x n x n data cube and reshape it to t x n^2 matrix
   return A.reshape((len(A), len(A[0])* len(A)[1]))
