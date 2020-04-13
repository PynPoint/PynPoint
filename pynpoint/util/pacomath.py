"""
This module contains basic utility functions

- Rotations
- Coordinate transformations
- Any other commonly used calculations

Most variables names correspond to naming in
Flasseur et al 2018.
"""
import numpy as np
import cv2

def rotateImage(image, angle):
    """
    Rotate an image about its center by a given angle
    Parameters
    ------------
    image : arr
        A 2D numpy array
    angle : float
        Clockwise rotation by this angle in degrees.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    return result

def getRotatedPixels(x, y, p0, angles):
    """
    For a given pixel, find the new pixel location after a rotation for each angle in angles
    Parameters
    --------------
    x : arr
        Grid of x components of pixel coordinates
    y : arr
        Grid of y components of pixel coordinates
    p0 : (int,int)
        Initial pixel location
    angles : arr
        List of angles for which to compute the new pixel location
    """
    # Current pixel
    phi0 = np.array([x[int(p0[0]), int(p0[1])], y[int(p0[0]), int(p0[1])]])
    # Convert to polar coordinates
    rphi0 = cartToPol(phi0)
    angles_rad = rphi0[1] - np.array([a*np.pi/180 for a in angles])

    # Rotate the polar coordinates by each frame angle
    angles_ind = [[rphi0[0], phi] for phi in angles_rad]
    angles_pol = np.array(list(zip(*angles_ind)))

    # Convert from polar to cartesian and pixel coordinates
    angles_px = np.array(gridPolToCart(angles_pol[0], angles_pol[1]))+int(x.shape[0]/2)
    angles_px = angles_px.T
    angles_px = np.fliplr(angles_px)
    return angles_px

def createCircularMask(shape, radius=4, center=None):
    """
    Returns a 2D boolean mask given some radius and location
    Parameters
    -------------
    shape : arr
        Shape of a 2D numpy array
    radius : int
        Radius of the mask in pixels
    center : (int,int)
        Pixel coordinates denoting the center of the mask,
        None defaults to center of shape
    """
    w = shape[0]
    h = shape[1]
    if center is None:
        center = [int(w/2), int(h/2)]
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    X, Y = np.ogrid[:w, :h]
    dist2 = (X - center[0])**2 + (Y-center[1])**2
    mask = dist2 <= radius**2
    return mask

def resizeImage(image, scaleFactor):
    """
    Rescale an image in both directions by scaleFactor
    """
    if scaleFactor == 1:
        return image
    return cv2.resize(image, (0, 0),
                      fx=scaleFactor, fy=scaleFactor,
                      interpolation=cv2.INTER_NEAREST)

def gaussian2d(x, y, A, sigma):
    """
    A 2D gaussian over range x,y with amplitude A and width sigma
    """
    return A*np.exp(-(x**2+y**2)/(2*sigma**2))

def gaussian2dModel(n, params):
    """
    Model function to create a PSF template
    """
    sigma = params["sigma"]
    dim = int(n/2)
    x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
    return 1.0/(2.0*np.pi*sigma**2)*np.exp(-((x+0.5)**2+(y+0.5)**2)/(2*sigma**2))

def psfTemplateModel(n, params):
    """
    Model using a psf template directly from the data.
    Template should be normalized such that the sum equals 1.

    If model needs rescaling it is done here
    """
    psf_template = params["psf_template"]
    print("PSF template shape", np.shape(psf_template))
    #dim = int(n)
    #m = np.shape(psf_template)[0]
    #if m != dim:
    #    raise ValueError("PSF template dimension not equal patch size")

    if np.sum(psf_template) != 1:
        print("Normalizing PSF template to sum = 1")
        psf_template = psf_template/np.sum(psf_template)
    return psf_template

def cartToPol(coords):
    """
    Takes cartesian (2D) coordinates and transforms them into polar.
    """
    if len(coords.shape) == 1:
        rho = np.sqrt(coords[0]**2 + coords[1]**2)
        phi = np.arctan2(coords[1], coords[0])
        return np.array((rho, phi))
    else:
        rho = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        phi = np.arctan2(coords[:, 1], coords[:, 0])
        return np.column_stack((rho, phi))

def polToCart(coords):
    """
    Takes polar coordinates and transforms them to cartesian
    """
    if len(coords.shape) == 1:
        x = coords[0]*np.cos(coords[1])
        y = coords[0]*np.sin(coords[0])
        return np.array((x, y))
    else:
        x = coords[:, 0]*np.cos(coords[:, 1])
        y = coords[:, 0]*np.sin(coords[:, 1])
        return np.column_stack((x, y))

def intPolToCart(coords):
    """
    Enforce integer (pixel) coordinates
    Possibly unnecessary, float coordinates work
    Output shapes necessary for PACO, but I don't remember why
    """
    if len(coords.shape) == 1:
        x = int(coords[0]*np.cos(coords[1]))
        y = int(coords[0]*np.sin(coords[0]))
        return np.array((x, y))
    else:
        x = coords[:, 0]*np.cos(coords[:, 1]).astype(int)
        y = coords[:, 0]*np.sin(coords[:, 1]).astype(int)
        return np.column_stack((x, y))

def gridCartToPol(x, y):
    """
    Takes cartesian (2D) coordinates and transforms them into polar.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def gridPolToCart(r, phi):
    """
    Takes polar (2D) coordinates and transforms them into cartesian.
    """
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return (x, y)

"""
Math functions for computing patch covariance
"""
def pixelCalc(patch):
    """
    Calculate the mean and inverse covariance within a patch
    Reimplemented in PACO class, can probably be deleted
    Parameters
    -------------
    patch : arr
        Array of circular (flattened) patches centered on the same physical
        pixel vertically throughout the image stack
    """
    if patch is None:
        return None, None
    T = patch.shape[0]
    #size = patch.shape[1]

    # Calculate the mean of the column
    m = np.mean(patch, axis=0)
    # Calculate the covariance matrix
    S = sampleCovariance(patch, m, T)
    rho = shrinkageFactor(S, T)
    F = diagSampleCovariance(S)
    C = covariance(rho, S, F)
    Cinv = np.linalg.inv(C)
    return m, Cinv

def covariance(rho, S, F):
    """
    Ĉ: Shrinkage covariance matrix
    Parameters
    -------------
    rho : float
        Shrinkage factor weight
    S : arr
        Sample covariance matrix
    F : arr
        Diagonal of sample covariance matrix
    """
    C = (1.0-rho)*S + rho*F
    return C

def sampleCovariance(r, m, T):
    """
    Ŝ: Sample covariance matrix
    Parameters
    ------------
    r : arr
        Observed intensity at position θk and time tl
    m : arr
        Mean of all background patches at position θk
    T : int
        Number of temporal frames
    """
    #S = (1.0/T)*np.sum([np.outer((p-m).ravel(),(p-m).ravel().T) for p in r], axis=0)
    S = (1.0/T)*np.sum([np.cov(np.stack((p, m)),\
                               rowvar=False, bias=False) for p in r], axis=0)
    return S

def diagSampleCovariance(S):
    """
    F: Diagonal elements of the sample covariance matrix
    Parameters
    ----------
    S : arr
        Sample covariance matrix
    """
    return np.diag(np.diag(S))

def shrinkageFactor(S, T):
    """
    ρ: Shrinkage factor to regularize covariant matrix
    Parameters
    -----------
    S : arr
        Sample covariance matrix
    T : int
        Number of temporal frames
    """
    top = (np.trace(np.dot(S, S)) + np.trace(S)**2 -\
           2.0*np.sum(np.array([d**2.0 for d in np.diag(S)])))
    bot = ((T+1.0)*(np.trace(np.dot(S, S))-\
                    np.sum(np.array([d**2.0 for d in np.diag(S)]))))
    p = top/bot
    return max(min(p, 1.0), 0.0)
