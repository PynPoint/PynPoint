"""
Functions for image processing.
"""

import math

from typing import Union, Tuple

import numpy as np

from typeguard import typechecked
from skimage.transform import rescale
from scipy.ndimage import fourier_shift, shift, rotate


@typechecked
def center_pixel(image: np.ndarray) -> Tuple[int, int]:
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

    if image.shape[-2] % 2 == 0 and image.shape[-1] % 2 == 0:
        center = (image.shape[-2] // 2 - 1, image.shape[-1] // 2 - 1)

    elif image.shape[-2] % 2 == 0 and image.shape[-1] % 2 == 1:
        center = (image.shape[-2] // 2 - 1, (image.shape[-1]-1) // 2)

    elif image.shape[-2] % 2 == 1 and image.shape[-1] % 2 == 0:
        center = ((image.shape[-2] - 1) // 2, image.shape[-1] // 2 - 1)

    elif image.shape[-2] % 2 == 1 and image.shape[-1] % 2 == 1:
        center = ((image.shape[-2] - 1) // 2, (image.shape[-1] - 1) // 2)

    else:
        raise RuntimeError('Unexpected image shape. This error should not occur.')

    return center


@typechecked
def center_subpixel(image: np.ndarray) -> Tuple[float, float]:
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

    center_x = float(image.shape[-1]) / 2 - 0.5
    center_y = float(image.shape[-2]) / 2 - 0.5

    return center_y, center_x


@typechecked
def crop_image(image: np.ndarray,
               center: Union[tuple, None],
               size: int,
               copy: bool = True) -> np.ndarray:
    """
    Function to crop square images around a specified position.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D or 3D).
    center : tuple(int, int), None
        The new image center (y, x). The center of the image is used if set to None.
    size : int
        Image size (pix) for both dimensions. Increased by 1 pixel if size is an even number.
    copy : bool
        Whether or not to return a copy (instead of a view) of the cropped image (default: True).

    Returns
    -------
    numpy.ndarray
        Cropped odd-sized image (2D or 3D).
    """

    if center is None or (center[0] is None and center[1] is None):
        center = center_pixel(image)

        # if image.shape[-1] % 2 == 0:
        #     warnings.warn('The image is even-size so there is not a uniquely defined pixel in '
        #                   'the center of the image. The image center is determined (with pixel '
        #                   'precision) with the pynpoint.util.image.center_pixel function.')

    if size % 2 == 0:
        size += 1

    x_start = center[1] - (size - 1) // 2
    x_end = center[1] + (size - 1) // 2 + 1

    y_start = center[0] - (size - 1) // 2
    y_end = center[0] + (size - 1) // 2 + 1

    if x_start < 0 or y_start < 0 or x_end > image.shape[-1] or y_end > image.shape[-2]:
        raise ValueError('Target image resolution does not fit inside the input image resolution.')

    return np.array(image[..., y_start:y_end, x_start:x_end], copy=copy)


@typechecked
def rotate_images(images: np.ndarray,
                  angles: np.ndarray) -> np.ndarray:
    """
    Function to rotate all images in clockwise direction.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of images (3D).
    angles : numpy.ndarray
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


@typechecked
def create_mask(im_shape: Tuple[int, int],
                size: Union[Tuple[float, float],
                            Tuple[float, None],
                            Tuple[None, float],
                            Tuple[None, None]]) -> np.ndarray:
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

        if npix % 2 == 0:
            x_grid = y_grid = np.linspace(-npix / 2 + 0.5, npix / 2 - 0.5, npix)
        else:
            x_grid = y_grid = np.linspace(-(npix - 1) / 2, (npix - 1) / 2, npix)

        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        rr_grid = np.sqrt(xx_grid**2 + yy_grid**2)

        if size[0] is not None:
            mask[rr_grid < size[0]] = 0.

        if size[1] is not None:
            if size[1] > npix / 2:
                size = (size[0], npix / 2)
            mask[rr_grid > size[1]] = 0.

    return mask


@typechecked
def shift_image(image: np.ndarray,
                shift_yx: Union[Tuple[float, float], np.ndarray],
                interpolation: str,
                mode: str = 'constant') -> np.ndarray:
    """
    Function to shift an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D or 3D). If 3D the image is not shifted along the 0th axis.
    shift_yx : tuple(float, float), np.ndarray
        Shift (y, x) to be applied (pix). An additional shift of zero pixels will be added
        for the first dimension in case the input image is 3D.
    interpolation : str
        Interpolation type ('spline', 'bilinear', or 'fft').
    mode : str
        Interpolation mode.

    Returns
    -------
    numpy.ndarray
        Shifted image.
    """

    if image.ndim == 2:
        shift_val = (shift_yx[0], shift_yx[1])
    elif image.ndim == 3:
        shift_val = (0, shift_yx[0], shift_yx[1])
    else:
        raise ValueError('Invalid number of dimensions for image: must be 2 or 3')

    if interpolation == 'spline':
        im_center = shift(image, shift_val, order=5, mode=mode)

    elif interpolation == 'bilinear':
        im_center = shift(image, shift_val, order=1, mode=mode)

    elif interpolation == 'fft':
        fft_shift = fourier_shift(np.fft.fftn(image), shift_val)
        im_center = np.fft.ifftn(fft_shift).real

    else:
        raise ValueError('interpolation must be one of the following: spline, bilinear, fft')

    return im_center


@typechecked
def scale_image(image: np.ndarray,
                scaling_y: float,
                scaling_x: float) -> np.ndarray:
    """
    Function to spatially scale an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D).
    scaling_y : float
        Scaling factor y.
    scaling_x : float
        Scaling factor x.

    Returns
    -------
    numpy.ndarray
        Shifted image (2D).
    """

    sum_before = np.sum(image)

    im_scale = rescale(image=np.asarray(image, dtype=np.float64),
                       scale=(scaling_y, scaling_x),
                       order=5,
                       mode='reflect',
                       anti_aliasing=True,
                       multichannel=False)

    sum_after = np.sum(im_scale)

    return im_scale * (sum_before / sum_after)


@typechecked
def cartesian_to_polar(center: Tuple[float, float],
                       y_pos: float,
                       x_pos: float) -> Tuple[float, float]:
    """
    Function to convert pixel coordinates to polar coordinates.

    Parameters
    ----------
    center : tuple(float, float)
        Image center (y, x) from :func:`~pynpoint.util.image.center_subpixel`.
    y_pos : float
        Pixel coordinate along the vertical axis. The bottom left corner of the image is
        (-0.5, -0.5).
    x_pos : float
        Pixel coordinate along the horizontal axis. The bottom left corner of the image is
        (-0.5, -0.5).

    Returns
    -------
    tuple(float, float)
        Separation (pix) and position angle (deg). The angle is measured counterclockwise with
        respect to the positive y-axis.
    """

    sep = math.sqrt((center[1] - x_pos)**2 + (center[0] - y_pos)**2)
    ang = math.atan2(y_pos-center[1], x_pos-center[0])
    ang = (math.degrees(ang) - 90) % 360

    return sep, ang


@typechecked
def polar_to_cartesian(image: np.ndarray,
                       sep: float,
                       ang: float) -> Tuple[float, float]:
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
        Cartesian coordinates (y, x). The bottom left corner of the image is (-0.5, -0.5).
    """

    center = center_subpixel(image)  # (y, x)

    x_pos = center[1] + sep * math.cos(math.radians(ang + 90))
    y_pos = center[0] + sep * math.sin(math.radians(ang + 90))

    return y_pos, x_pos


@typechecked
def pixel_distance(im_shape: Tuple[int, int],
                   position: Tuple[int, int] = None) -> np.ndarray:
    """
    Function to calculate the distance of each pixel with respect to a given pixel position.

    Parameters
    ----------
    im_shape : tuple(int, int)
        Image shape (y, x).
    position : tuple(int, int)
        Pixel center (y, x) from which the distance is calculated. The image center is used if
        set to None. Python indexing starts at zero so the bottom left pixel is (0, 0).

    Returns
    -------
    numpy.ndarray
        2D array with the distances of each pixel from the provided pixel position.
    """

    if im_shape[0] % 2 == 0:
        y_grid = np.linspace(-im_shape[0] / 2 + 0.5, im_shape[0] / 2 - 0.5, im_shape[0])
    else:
        y_grid = np.linspace(-(im_shape[0] - 1) / 2, (im_shape[0] - 1) / 2, im_shape[0])

    if im_shape[1] % 2 == 0:
        x_grid = np.linspace(-im_shape[1] / 2 + 0.5, im_shape[1] / 2 - 0.5, im_shape[1])
    else:
        x_grid = np.linspace(-(im_shape[1] - 1) / 2, (im_shape[1] - 1) / 2, im_shape[1])

    if position is not None:
        y_shift = y_grid[position[0]]
        x_shift = x_grid[position[1]]

        y_grid -= y_shift
        x_grid -= x_shift

    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

    return np.sqrt(xx_grid**2 + yy_grid**2)


@typechecked
def subpixel_distance(im_shape: Tuple[int, int],
                      position: Tuple[float, float]) -> np.ndarray:
    """
    Function to calculate the distance of each pixel with respect to a given subpixel position.

    Parameters
    ----------
    im_shape : tuple(int, int)
        Image shape (y, x).
    position : tuple(float, float)
        Pixel center (y, x) from which the distance is calculated. Python indexing starts at zero
        so the bottom left image corner is (-0.5, -0.5).

    Returns
    -------
    numpy.ndarray
        2D array with the distances of each pixel from the provided pixel position.
    """

    if im_shape[0] % 2 == 0:
        raise ValueError('The subpixel_distance function has only been implemented for '
                         'odd-sized images.')

    y_size = (im_shape[0] - 1) / 2
    x_size = (im_shape[1] - 1) / 2

    y_grid = np.linspace(-y_size, y_size, im_shape[0])
    x_grid = np.linspace(-x_size, x_size, im_shape[1])

    y_pos = position[0] - y_size
    x_pos = position[1] - x_size

    y_grid -= y_pos
    x_grid -= x_pos

    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

    return np.sqrt(xx_grid**2 + yy_grid**2)


@typechecked
def select_annulus(image_in: np.ndarray,
                   radius_in: float,
                   radius_out: float,
                   mask_position: Tuple[float, float] = None,
                   mask_radius: float = None) -> np.ndarray:
    """
    image_in : numpy.ndarray
        Input image.
    radius_in : float
        Inner radius of the annulus (pix).
    radius_out : float
        Outer radius of the annulus (pix).
    mask_position : tuple(float, float), None
        Center (pix) position (y, x) in of the circular region that is excluded. Not used
        if set to None.
    mask_radius : float, None
        Radius (pix) of the circular region that is excluded. Not used if set to None.
    """

    im_shape = image_in.shape

    if im_shape[0] % 2 == 0:
        y_grid = np.linspace(-im_shape[0] / 2 + 0.5, im_shape[0] / 2 - 0.5, im_shape[0])
    else:
        y_grid = np.linspace(-(im_shape[0] - 1) / 2, (im_shape[0] - 1) / 2, im_shape[0])

    if im_shape[1] % 2 == 0:
        x_grid = np.linspace(-im_shape[1] / 2 + 0.5, im_shape[1] / 2 - 0.5, im_shape[1])
    else:
        x_grid = np.linspace(-(im_shape[1] - 1) / 2, (im_shape[1] - 1) / 2, im_shape[1])

    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
    rr_grid = np.sqrt(xx_grid**2 + yy_grid**2)

    mask = np.ones(im_shape)

    indices = np.where((rr_grid < radius_in) | (rr_grid > radius_out))
    mask[indices[0], indices[1]] = 0.

    if mask_position is not None and mask_radius is not None:
        distance = subpixel_distance(im_shape=im_shape, position=mask_position)
        indices = np.where(distance < mask_radius)
        mask[indices[0], indices[1]] = 0.

    indices = np.where(mask == 1.)

    return image_in[indices[0], indices[1]]


@typechecked
def rotate_coordinates(center: Tuple[float, float],
                       position: Tuple[float, float],
                       angle: float) -> Tuple[float, float]:
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

    pos_y = (position[1] - center[1]) * math.sin(np.radians(angle)) + \
            (position[0] - center[0]) * math.cos(np.radians(angle))

    pos_x = (position[1] - center[1]) * math.cos(np.radians(angle)) - \
            (position[0] - center[0]) * math.sin(np.radians(angle))

    return center[0] + pos_y, center[1] + pos_x
