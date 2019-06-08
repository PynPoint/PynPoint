"""
Functions for PSF subtraction.
"""

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from scipy.linalg import svd
from copy import deepcopy, copy
from pynpoint.util.module import progress


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
    pca_sklearn : sklearn.decomposition.pca.PCA, None
        PCA object with the basis if not set to None.
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
    if indices is None:
        indices = np.arange(0, im_reshape.shape[1], 1)

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
                        pca_numbers,
                        pca_number_init,
                        interval=1,
                        indices=None):
    """
    Function for PSF subtraction with  iterative PCA.

    :param images: Stack of images. Also used as reference images if pca_sklearn is set to None.
                   Should be in the original 3D shape if pca_sklearn is set to None or in the 2D
                   reshaped format if pca_sklearn is not None.
    :type images: numpy.ndarray
    :param parang: Derotation angles (deg).
    :type parang: numpy.ndarray
    :param pca_numbers: Number or list of numbers of principal components used for the PSF model.
    :type pca_numbers: int
    :param pca_number_init: Number of principal component of first iteration
    :type pca_number_init: int
    :param indices: Non-masked image indices, required if pca_sklearn is not set to None. Optional
                    if pca_sklearn is set to None.
    :type indices: numpy.ndarray

    :return: Mean residuals of the PSF subtraction and the derotated but non-stacked residuals.
    :rtype: numpy.ndarray, numpy.ndarray
    """

    residuals_list = []
    res_rot_list = []

    #set pca_number to maximum number if multiple are given
    if isinstance(pca_numbers, int):
        pca_number = pca_numbers
        pca_numbers = np.asarray([pca_numbers])
    elif len(pca_numbers) > 1:
        pca_number = max(pca_numbers)
    else:
        pca_number = int(pca_numbers)

    im_shape = images.shape
    
    
    if indices is None:
        # select the first image and get the unmasked image indices
        im_star = images[0, ].reshape(-1)
        indices = np.where(im_star != 0.)[0]

    # reshape the images and select the unmasked pixels
    im_reshape = images.reshape(im_shape[0], im_shape[1]*im_shape[2])
    
    '''what does this do? causes problem in contrastcurvemodule if active'''
    #im_reshape = im_reshape[:, indices]
    # subtract mean image
    #im_reshape -= np.mean(im_reshape, axis=0)

    # create first iteration
    S = im_reshape - LRA(im_reshape, pca_number_init)
  
    #iterate through all values between initial and final pca number
    pca_numbers_range = range(pca_number_init, pca_number+1)
    
    for counter, i in enumerate(pca_numbers_range):
        progress(counter, len(pca_numbers_range), "Creating residuals...")
        S = im_reshape - LRA(im_reshape-theta(red(S, im_shape, angles), im_shape, angles), i)
        #save intermediate results to lists if the current pca_number corresponds to a final pca_number in pca_numbers
        if i in pca_numbers:
            '''implement this?'''
            # subtract the psf model IMPLEMENT THIS?
        
            residuals = deepcopy(S)
            
            # reshape to the original image size
            residuals = residuals.reshape(im_shape)
            
            # derotate the images
            res_rot = np.zeros(residuals.shape)
            for j, item in enumerate(angles):
                res_rot[j, ] = rotate(residuals[j, ], item, reshape=False)
                
            #append the results to the lists   
            residuals_list.append(residuals)
            res_rot_list.append(res_rot)
            
    residuals = np.asarray(residuals_list)
    res_rot = np.asarray(res_rot_list)
    
    return residuals, res_rot
    

def SVD(A):
    U, sigma, Vh = svd(A)
    #create corresponding matrix Sigma from list sigma
    Sigma = np.zeros((len(A), len(A[0])))
    for i in range(len(sigma)):
        Sigma[i][i] = sigma[i]
    return U, Sigma, Vh

def LRA(A, pca_number):
    U, Sigma, Vh = SVD(A)
    L = np.matmul(U[:, :pca_number], np.matmul(Sigma[:pca_number, :pca_number], Vh[:pca_number, :]))
    return L

def red(S, im_shape, angles): #takes t x n^2 matrix S, reshapes it to cube S_cube, rotates each frame, and returns mean of cube, i.e. processed frame
    S = np.reshape(S, (im_shape[0], im_shape[1], im_shape[2]))
    if angles is not None:
        for i in range(len(S)):
            S[i] = rotate(S[i], angles[i], reshape = False)
    return np.mean(S, axis = 0)

def theta(frame, im_shape, angles = None): #takes a (PCA processed) frame, sets negative parts of it to zero, reshapes it into t x n x n cube, rotates frames according to list and returns t x n^2 matrix
    d = frame.clip(min = 0)
    d = np.stack([d]*im_shape[0], axis = 0)
    if angles is not None:
        for i in range(len(d)):
            d[i] = rotate(d[i], -1*angles[i], reshape = False)
    d_shape = np.shape(d)
    d = np.reshape(d,(d_shape[0], d_shape[1]*d_shape[2]))
    return d