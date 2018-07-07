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
    :param size: Image size (pix) for both dimensions.
    :type size: int
    :param center: Tuple (x0, y0) with the new image center.
    :type center: tuple, int

    :return: Cropped image.
    :rtype: ndarray
    """

    x_off = int(center[0] - size/2)
    y_off = int(center[1] - size/2)

    return image[y_off:y_off+size, x_off:x_off+size]

def number_images(port):
    """
    Function to get the number of images of an input port.
    """

    if port.get_ndim() == 2:
        nimages = 1

    elif port.get_ndim() == 3:
        nimages = port.get_shape()[0]

    return nimages

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
