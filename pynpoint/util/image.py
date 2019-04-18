"""
Functions for image processing.
"""

from __future__ import absolute_import

import math

import numpy as np

from skimage.transform import rescale
from scipy.ndimage import fourier_shift, shift, rotate


def center_pixel(image):
    """
    Function to get the pixel position of the image center. Note that this position
    can not be unambiguously defined for an even-sized image. Python indexing starts
    at 0 so the coordinates of the pixel in the bottom-left corner are (0, 0).

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D or 3D).

    Returns
    -------
    tuple(int, int)
        Pixel position (y, x) of the image center.
    """

    if image.shape[-2]%2 == 0 and image.shape[-1]%2 == 0:
        center = (image.shape[-2] // 2 - 1, image.shape[-1] // 2 - 1)

    elif image.shape[-2]%2 == 0 and image.shape[-1]%2 == 1:
        center = (image.shape[-2] // 2 - 1, (image.shape[-1]-1) // 2)

    elif image.shape[-2]%2 == 1 and image.shape[-1]%2 == 0:
        center = ((image.shape[-2]-1) // 2, image.shape[-1] // 2 - 1)

    elif image.shape[-2]%2 == 1 and image.shape[-1]%2 == 1:
        center = ((image.shape[-2]-1) // 2, (image.shape[-1]-1) // 2)

    return center

def center_subpixel(image):
    """
    Function to get the precise position of the image center. The center of the pixel in the
    bottom left corner of the image is defined as (0, 0), so the bottom left corner of the
    image is located at (-0.5, -0.5).

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D or 3D).

    Returns
    -------
    tuple(float, float)
        Subpixel position (y, x) of the image center.
    """

    center_x = float(image.shape[-1])/2. - 0.5
    center_y = float(image.shape[-2])/2. - 0.5

    return (center_y, center_x)

def crop_image(image,
               center,
               size):
    """
    Function to crop square images around a specified position.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D or 3D).
    center : tuple(int, int)
        The new image center (y, x). The center of the image is used if set to None.
    size : int
        Image size (pix) for both dimensions. Increased by 1 pixel if size is an even number.

    Returns
    -------
    numpy.ndarray
        Cropped odd-sized image (2D or 3D).
    """

    if center is None or (center[0] is None and center[1] is None):
        center = center_pixel(image)

    if size%2 == 0:
        size += 1

    x_start = center[1] - (size-1)//2
    x_end = center[1] + (size-1)//2 + 1

    y_start = center[0] - (size-1)//2
    y_end = center[0] + (size-1)//2 + 1

    if x_start < 0 or y_start < 0 or x_end > image.shape[-1] or y_end > image.shape[-2]:
        raise ValueError("Target image resolution does not fit inside the input image resolution.")

    if image.ndim == 2:
        im_return = np.copy(image[y_start:y_end, x_start:x_end])
    if image.ndim == 3:
        im_return = np.copy(image[:, y_start:y_end, x_start:x_end])

    return im_return

def rotate_images(images,
                  angles):
    """
    Function to rotate all images in clockwise direction.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of images (3D).
    angle : numpy.ndarray
        Rotation angles (deg).

    Returns
    -------
    numpy.ndarray
        Rotated images.
    """

    im_rot = np.zeros(images.shape)

    for i, item in enumerate(angles):
        im_rot[i, ] = rotate(input=images[i, ], angle=item, reshape=False)

    return im_rot

def create_mask(im_shape,
                size):
    """
    Function to create a mask for the central and outer image regions.

    Parameters
    ----------
    im_shape : tuple(int, int)
        Image size in both dimensions.
    size : tuple(float, float)
        Size (pix) of the inner and outer mask.

    Returns
    -------
    numpy.ndarray
        Image mask.
    """

    mask = np.ones(im_shape)
    npix = im_shape[0]

    if size[0] is not None or size[1] is not None:

        if npix%2 == 0:
            x_grid = y_grid = np.linspace(-npix/2+0.5, npix/2-0.5, npix)
        elif npix%2 == 1:
            x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)

        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        rr_grid = np.sqrt(xx_grid**2+yy_grid**2)

    if size[0] is not None:
        mask[rr_grid < size[0]] = 0.

    if size[1] is not None:
        if size[1] > npix/2.:
            size[1] = npix/2.

        mask[rr_grid > size[1]] = 0.

    return mask

def shift_image(image,
                shift_yx,
                interpolation,
                mode='constant'):
    """
    Function to shift an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D).
    shift_yx : tuple(float, float)
        Shift (y, x) to be applied (pix).
    interpolation : str
        Interpolation type ("spline", "bilinear", or "fft").

    Returns
    -------
    numpy.ndarray
        Shifted image.
    """

    if interpolation == "spline":
        im_center = shift(image, shift_yx, order=5, mode=mode)

    elif interpolation == "bilinear":
        im_center = shift(image, shift_yx, order=1, mode=mode)

    elif interpolation == "fft":
        fft_shift = fourier_shift(np.fft.fftn(image), shift_yx)
        im_center = np.fft.ifftn(fft_shift).real

    return im_center

def scale_image(image,
                scaling_x,
                scaling_y):
    """
    Function to spatially scale an image.

    Parameters
    ----------
    images : numpy.ndarray
        Input image (2D).
    scaling_x : float
        Scaling factor x.
    scaling_y : float
        Scaling factor y.

    Returns
    -------
    numpy.ndarray
        Shifted image (2D).
    """

    sum_before = np.sum(image)

    im_scale = rescale(image=np.asarray(image, dtype=np.float64),
                       scale=(scaling_y, scaling_x),
                       order=5,
                       mode="reflect",
                       anti_aliasing=True,
                       multichannel=False)

    sum_after = np.sum(im_scale)

    return im_scale * (sum_before / sum_after)

def cartesian_to_polar(center,
                       x_pos,
                       y_pos):
    """
    Function to convert pixel coordinates to polar coordinates.

    Parameters
    ----------
    center : tuple(float, float)
        Image center (y, x) from :func:`~pynpoint.util.image.center_subpixel`.
    x_pos : float
        Pixel coordinate along the horizontal axis. The bottom left corner of the image is
        (-0.5, -0.5).
    y_pos : float
        Pixel coordinate along the vertical axis. The bottom left corner of the image is
        (-0.5, -0.5).

    Returns
    -------
    tuple(float, float)
        Separation (pix) and position angle (deg). The angle is measured counterclockwise with
        respect to the positive y-axis.
    """

    sep = math.sqrt((center[1]-x_pos)**2.+(center[0]-y_pos)**2.)
    ang = math.atan2(y_pos-center[1], x_pos-center[0])
    ang = (math.degrees(ang)-90.)%360.

    return tuple([sep, ang])

def polar_to_cartesian(image,
                       sep,
                       ang):
    """
    Function to convert polar coordinates to pixel coordinates.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D or 3D).
    sep : float
        Separation (pix).
    ang : float
        Position angle (deg), measured counterclockwise with respect to the positive y-axis.

    Returns
    -------
    tuple(float, float)
        Cartesian coordinates (x, y). The bottom left corner of the image is (-0.5, -0.5).
    """

    center = center_subpixel(image) # (y, x)

    x_pos = center[1] + sep*math.cos(math.radians(ang+90.))
    y_pos = center[0] + sep*math.sin(math.radians(ang+90.))

    return tuple([x_pos, y_pos])
