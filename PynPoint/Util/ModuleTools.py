"""
Functions for Pypeline modules.
"""

import sys
import numpy as np

from scipy.ndimage import rotate


def progress(current, total, message):
    """
    Function to show and update the progress as standard output.
    """

    fraction = float(current)/float(total)
    percentage = round(fraction*100., 1)

    sys.stdout.write("%s %s%s \r" % (message, percentage, "%"))
    sys.stdout.flush()

def memory_frames(memory, nimages):
    """
    Function to subdivide the input images is in quantities of MEMORY.
    """

    if memory == 0 or memory >= nimages:
        frames = [0, nimages]

    else:
        frames = np.linspace(0,
                             nimages-nimages%memory,
                             int(float(nimages)/float(memory))+1,
                             endpoint=True,
                             dtype=np.int)

        if nimages%memory > 0:
            frames = np.append(frames, nimages)

    return frames

def crop_image(image, center, size):
    """
    Function to crop a square image around a specified position.

    :param image: Input image.
    :type image: ndarray
    :param center: Tuple (x0, y0) with the new image center.
    :type center: tuple, int
    :param size: Image size (pix) for both dimensions.
    :type size: int

    :return: Cropped odd-sized image.
    :rtype: ndarray
    """

    if size%2 == 0:
        size += 1

    x_start = center[0] - (size-1)/2
    x_end = center[0] + (size-1)/2 + 1

    y_start = center[1] - (size-1)/2
    y_end = center[1] + (size-1)/2 + 1

    return image[y_start:y_end, x_start:x_end]

def number_images_port(port):
    """
    Function to get the number of images of an input port.
    """

    if port.get_ndim() == 2:
        nimages = 1

    elif port.get_ndim() == 3:
        nimages = port.get_shape()[0]

    return nimages

def image_size_port(port):
    """
    Function to get the image size of an input port.
    """

    if port.get_ndim() == 2:
        size = port.get_shape()

    elif port.get_ndim() == 3:
        size = (port.get_shape()[1], port.get_shape()[2])

    return size

def image_size(images):
    """
    Function to get the image size of an array.
    """

    if images.ndim == 2:
        size = images.shape

    elif images.ndim == 3:
        size = (images.shape[1], images.shape[2])

    return size

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
