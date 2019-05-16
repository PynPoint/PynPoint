"""
Functions for Pypeline modules.
"""

import sys
import time
import math

import cv2
import numpy as np

from pynpoint.util.image import crop_image, center_pixel


def progress(current,
             total,
             message,
             start_time=None):
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
    start_time : float, None, optional
        Start time in seconds. Not used if set to None.

    Returns
    -------
    NoneType
        None
    """

    def time_string(delta_time):
        """
        Converts to input time in seconds to a string which displays as hh:mm:ss.
        
        Parameters
        ----------
        delta_time : float
            Input time in seconds.

        Returns
        -------
        str:
            String with the formatted time.
        """

        hours = int(delta_time / 3600.)
        minutes = int((delta_time%3600.) / 60.)
        seconds = int(delta_time%60.)

        return f'{hours:>02}:{minutes:>02}:{seconds:>02}'

    fraction = float(current) / float(total)
    percentage = 100.*fraction

    if start_time is None:
        sys.stdout.write(f'\r{message} {percentage:4.1f}% \r')

    else:
        if fraction > 0. and current+1 != total:
            time_taken = time.time() - start_time
            time_left = time_taken / fraction * (1. - fraction)
            sys.stdout.write(f'{message} {percentage:4.1f}% - Time: {time_string(time_left)}\r')

    if current+1 == total:
        sys.stdout.write(' ' * (29+len(message)) + '\r')

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
        frames = np.linspace(start=0,
                             stop=nimages-nimages%memory,
                             num=int(float(nimages)/float(memory))+1,
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
    fwhm : int, None
        Full width at half maximum (pix) of the Gaussian kernel. Not used if set to None.

    Returns
    -------
    tuple(int, int)
        Position (y, x) of the brightest pixel.
    """

    if width is not None:
        if center is None:
            center = center_pixel(image)

        image = crop_image(image, center, width)

    if fwhm is None:
        smooth = np.copy(image)

    else:
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

def update_arguments(index,
                     nimages,
                     args_in):
    """
    Function to update the arguments of an input function. Specifically, arguments which contain an
    array with the first dimension equal in size to the total number of images will be substituted
    by the array element of the image index.

    Parameters
    ----------
    index : int
        Image index in the stack.
    nimages : int
        Total number of images in the stack.
    args_in : tuple
        Function arguments that have to be updated.

    Returns
    -------
    tuple
        Updated function arguments.
    """

    if args_in is None:
        args_out = None

    else:
        args_out = []

        for item in args_in:
            if isinstance(item, np.ndarray) and item.shape[0] == nimages:
                args_out.append(item[index])

            else:
                args_out.append(item)

        args_out = tuple(args_out)

    return args_out
