"""
Functions for Pypeline modules.
"""

from __future__ import absolute_import

import sys
import math

import cv2
import numpy as np

from pynpoint.util.image import crop_image, center_pixel


def progress(current,
             total,
             message):
    """
    Function to show and update the progress as standard output.

    Parameters
    ----------
    current : int
        Current index.
    total : int
        Total index number.
    message : str
        Message that is printed.

    Returns
    -------
    NoneType
        None
    """

    fraction = float(current)/float(total)
    percentage = round(fraction*100., 1)

    sys.stdout.write("%s %s%s \r" % (message, percentage, "%"))
    sys.stdout.flush()

def memory_frames(memory,
                  nimages):
    """
    Function to subdivide the input images is in quantities of MEMORY.

    Parameters
    ----------
    memory : int
        Number of images that is simultaneously loaded into the memory.
    nimages : int
        Number of images in the stack.

    Returns
    -------
    numpy.ndarray
    """

    if memory == 0 or memory >= nimages:
        frames = np.asarray([0, nimages])

    else:
        frames = np.linspace(0,
                             nimages-nimages%memory,
                             int(float(nimages)/float(memory))+1,
                             endpoint=True,
                             dtype=np.int)

        if nimages%memory > 0:
            frames = np.append(frames, nimages)

    return frames

def locate_star(image,
                center,
                width,
                fwhm):
    """
    Function to locate the star by finding the brightest pixel.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D).
    center : tuple(int, int)
        Pixel center (y, x) of the subframe. The full image is used if set to None.
    width : int
        The width (pix) of the subframe. The full image is used if set to None.
    fwhm : int
        Full width at half maximum of the Gaussian kernel.

    Returns
    -------
    tuple(int, int)
        Position (y, x) of the brightest pixel.
    """

    if width is not None:
        if center is None:
            center = center_pixel(image)

        image = crop_image(image, center, width)

    sigma = fwhm/math.sqrt(8.*math.log(2.))
    kernel = (fwhm*2+1, fwhm*2+1)
    smooth = cv2.GaussianBlur(image, kernel, sigma)

    # argmax[0] is the y position and argmax[1] is the y position
    argmax = np.asarray(np.unravel_index(smooth.argmax(), smooth.shape))

    if center is not None and width is not None:
        argmax[0] += center[0] - (image.shape[0]-1) // 2 # y
        argmax[1] += center[1] - (image.shape[1]-1) // 2 # x

    return argmax

def rotate_coordinates(center,
                       position,
                       angle):
    """
    Function to rotate coordinates around the image center.

    Parameters
    ----------
    center : tuple(float, float)
        Image center (y, x).
    position : tuple(float, float)
        Position (y, x) in the image.
    angle : float
        Angle (deg) to rotate in counterclockwise direction.

    Returns
    -------
    tuple(float, float)
        New position (y, x).
    """

    pos_x = (position[1]-center[1])*math.cos(np.radians(angle)) - \
            (position[0]-center[0])*math.sin(np.radians(angle))

    pos_y = (position[1]-center[1])*math.sin(np.radians(angle)) + \
            (position[0]-center[0])*math.cos(np.radians(angle))

    return (center[0]+pos_y, center[1]+pos_x)
