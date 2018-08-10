"""
Functions for image processing.
"""

import math

import cv2
import numpy as np

from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from scipy.ndimage import rotate


def image_center(image):
    """
    Function to get the pixel position of the center of an image. Note that this position
    can not be unambiguously defined for an even-sized image.

    :param image: Input image (2D or 3D).
    :type image: ndarray

    :return: Pixel position (y, x) of the image center.
    :rtype: (int, int)
    """

    if image.shape[-2]%2 == 0 and image.shape[-1]%2 == 0:
        center = (image.shape[-2]/2-1, image.shape[-1]/2-1)

    elif image.shape[-2]%2 == 0 and image.shape[-1]%2 == 1:
        center = (image.shape[-2]/2-1, (image.shape[-1]-1)/2)

    elif image.shape[-2]%2 == 1 and image.shape[-1]%2 == 0:
        center = ((image.shape[-2]-1)/2, image.shape[-1]/2-1)

    elif image.shape[-2]%2 == 1 and image.shape[-1]%2 == 1:
        center = ((image.shape[-2]-1)/2, (image.shape[-1]-1)/2)

    return center

def crop_image(image, center, size):
    """
    Function to crop square images around a specified position.

    :param image: Input image (2D or 3D).
    :type image: ndarray
    :param center: Tuple (y, x) with the new image center. The center of the image is used if
                   set to None.
    :type center: (int, int)
    :param size: Image size (pix) for both dimensions. Increased by 1 pixel if size is an even
                 number.
    :type size: int

    :return: Cropped odd-sized image.
    :rtype: ndarray
    """

    if center is None or (center[0] is None and center[1] is None):
        if image.ndim == 2:
            center = image_center(image)

        elif image.ndim == 3:
            center = image_center(image[0, ])

    if size%2 == 0:
        size += 1

    x_start = center[1] - (size-1)/2
    x_end = center[1] + (size-1)/2 + 1

    y_start = center[0] - (size-1)/2
    y_end = center[0] + (size-1)/2 + 1

    if x_start < 0 or y_start < 0 or x_end > image.shape[1] or y_end > image.shape[0]:
        raise ValueError("Target image resolution does not fit inside the input image resolution.")

    if image.ndim == 2:
        im_crop = np.copy(image[y_start:y_end, x_start:x_end])

    elif image.ndim == 2:
        im_crop = np.copy(image[:, y_start:y_end, x_start:x_end])

    return im_crop

def rotate_images(images, angles):
    """
    Function to rotate all images in clockwise direction.

    :param images: Stack of images.
    :type images: ndarray
    :param angle: Rotation angles (deg).
    :type angle: ndarray

    :return: Rotated images.
    :rtype: ndarray
    """

    im_rot = np.zeros(images.shape)

    if images.ndim == 2:
        im_rot = rotate(input=images, angle=angles, reshape=False)

    elif images.ndim == 3:
        for i, item in enumerate(angles):
            im_rot[i, ] = rotate(input=images[i, ], angle=item, reshape=False)

    return im_rot

def create_mask(im_shape, size):
    """
    Function to create a mask for the central and outer image regions.
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

def locate_star(image, center, width, fwhm):
    """
    Function to locate the star by finding the brightest pixel.

    :param image: Input image.
    :type image: ndarray
    :param center: Pixel center (y, x) of the subframe. The full image is used if set to None.
    :type center: (int, int)
    :param width: The width (pixel) of the subframe. The full image is used if set to None.
    :type fwhm: Full width at half maximum of the Gaussian kernel.
    :param fwhm: int

    :return: Position (y, x) of the brightest pixel.
    :rtype: (int, int)
    """

    if center is not None and width is not None:
        if center[0] is None and center[1] is None:
            center = image_center(image)

        image = crop_image(image, center, width)

    sigma = fwhm/math.sqrt(8.*math.log(2.))
    kernel = (fwhm*2+1, fwhm*2+1)
    smooth = cv2.GaussianBlur(image, kernel, sigma)

    # argmax[0] is the y position and argmax[1] is the y position
    argmax = np.asarray(np.unravel_index(smooth.argmax(), smooth.shape))

    if center is not None and width is not None:
        argmax[0] += center[0] - (image.shape[0]-1)/2 # y
        argmax[1] += center[1] - (image.shape[1]-1)/2 # x

    return argmax

def shift_image(image, shift_yx, interpolation, mode='constant'):
    """
    Function to shift an image.

    :param images: Input image.
    :type images: ndarray
    :param shift_yx: Shift (y, x) to be applied (pixel).
    :type shift_yx: (float, float)
    :param interpolation: Interpolation type (spline, bilinear, fft)
    :type interpolation: str

    :return: Shifted image.
    :rtype: ndarray
    """

    if interpolation == "spline":
        im_center = shift(image, shift_yx, order=5, mode=mode)

    elif interpolation == "bilinear":
        im_center = shift(image, shift_yx, order=1, mode=mode)

    elif interpolation == "fft":
        fft_shift = fourier_shift(np.fft.fftn(image), shift_yx)
        im_center = np.fft.ifftn(fft_shift).real

    return im_center
