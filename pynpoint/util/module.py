"""
Functions for Pypeline modules.
"""

from __future__ import absolute_import

import sys
import math

import cv2
import numpy as np

from pynpoint.util.image import crop_image, image_center_pixel


def progress(current,
             total,
             message):
    """
    Function to show and update the progress as standard output.
    """

    fraction = float(current)/float(total)
    percentage = round(fraction*100., 1)

    sys.stdout.write("%s %s%s \r" % (message, percentage, "%"))
    sys.stdout.flush()

def memory_frames(memory,
                  nimages):
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

def locate_star(image,
                center,
                width,
                fwhm):
    """
    Function to locate the star by finding the brightest pixel.

    :param image: Input image.
    :type image: ndarray
    :param center: Pixel center (y, x) of the subframe. The full image is used if set to None.
    :type center: (int, int)
    :param width: The width (pixel) of the subframe. The full image is used if set to None.
    :type width: int
    :param fwhm: Full width at half maximum of the Gaussian kernel.
    :type fwhm: int

    :return: Position (y, x) of the brightest pixel.
    :rtype: (int, int)
    """

    if width is not None:
        if center is None:
            center = image_center_pixel(image)

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

    :param center: Image center (y, x).
    :type center: (float, float)
    :param position: Position (y, x) in the image.
    :type position: (float, float)
    :param angle: Angle (deg) to rotate in counterclockwise direction.
    :type angle: float

    :return: Position (y, x) of the brightest pixel.
    :rtype: (int, int)
    """

    pos_x = (position[1]-center[1])*math.cos(np.radians(angle)) - \
            (position[0]-center[0])*math.sin(np.radians(angle))

    pos_y = (position[1]-center[1])*math.sin(np.radians(angle)) + \
            (position[0]-center[0])*math.cos(np.radians(angle))

    return (center[0]+pos_y, center[1]+pos_x)
