"""
Functions for stellar extraction.
"""

import math
import time

from typing import Union, Tuple

import cv2
import numpy as np

from typeguard import typechecked

from pynpoint.core.dataio import InputPort
from pynpoint.util.image import crop_image, center_pixel
from pynpoint.util.module import progress


@typechecked
def locate_star(image: np.ndarray,
                center: Union[tuple, None],
                width: Union[int, None],
                fwhm: Union[int, None]) -> np.ndarray:
    """
    Function to locate the star by finding the brightest pixel.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D).
    center : tuple(int, int), None
        Pixel center (y, x) of the subframe. The full image is used if set to None.
    width : int, None
        The width (pix) of the subframe. The full image is used if set to None.
    fwhm : int, None
        Full width at half maximum (pix) of the Gaussian kernel. Not used if set to None.

    Returns
    -------
    numpy.ndarray
        Position (y, x) of the brightest pixel.
    """

    if width is not None:
        if center is None:
            center = center_pixel(image)

        image = crop_image(image, center, width)

    if fwhm is None:
        smooth = np.copy(image)

    else:
        sigma = fwhm / math.sqrt(8. * math.log(2.))
        kernel = (fwhm * 2 + 1, fwhm * 2 + 1)
        smooth = cv2.GaussianBlur(image, kernel, sigma)

    # argmax[0] is the y position and argmax[1] is the y position
    argmax = np.asarray(np.unravel_index(smooth.argmax(), smooth.shape))

    if center is not None and width is not None:
        argmax[0] += center[0] - (image.shape[0] - 1) // 2  # y
        argmax[1] += center[1] - (image.shape[1] - 1) // 2  # x

    return argmax


@typechecked
def star_positions(input_port: InputPort,
                   fwhm: float,
                   position: Union[Tuple[int, int, float], Tuple[None, None, float],
                                   Tuple[int, int, None]] = None) -> np.ndarray:
    """
    Function to return the position of the star in each image.

    Parameters
    ----------
    input_port : pynpoint.core.dataio.InputPort
        Input port where the images are stored.
    fwhm : int
        The FWHM (pix) of the Gaussian kernel that is used to smooth the images before the
        brightest pixel is located.
    position : tuple(int, int, int), None
        Subframe that is selected to search for the star. The tuple contains the center (pix)
        and size (pix) (pos_x, pos_y, size). Setting `position` to None will use the full
        image to search for the star. If `position=(None, None, size)` then the center of the
        image will be used. If `position=(pos_x, pos_y, None)` then a fixed position is used
        for the aperture.

    Returns
    -------
    numpy.ndarray
        Positions (y, x) of the brightest pixel.
    """

    nimages = input_port.get_shape()[0]
    starpos = np.zeros((nimages, 2), dtype=np.int64)

    if position is not None and position[2] is None:
        # [y. x] position
        starpos[:, 0] = position[1]
        starpos[:, 1] = position[0]

    else:
        center = None
        width = None

        if position is not None:
            width = position[2]

            if position[0] is not None and position[1] is not None:
                center = position[0:2]

        start_time = time.time()

        for i in range(nimages):
            progress(i, nimages, 'Locating stellar position...', start_time)

            # [y. x] position
            starpos[i, :] = locate_star(input_port[i, ], center, width, fwhm)

    return starpos
