"""
Functions for spectral differential imaging.
"""

import numpy as np

from typeguard import typechecked

from pynpoint.util.image import scale_image, shift_image


@typechecked
def sdi_scaling(image_in: np.ndarray,
                scaling: np.ndarray) -> np.ndarray:

    """
    Function to rescale the images by their wavelength ratios.

    Parameters
    ----------
    image_in : np.ndarray
        Data to rescale
    scaling : np.ndarray
        Scaling factors.

    Returns
    -------
    np.ndarray
        Rescaled images with the same shape as ``image_in``.
    """

    if image_in.shape[0] != scaling.shape[0]:
        raise ValueError('The number of wavelengths is not equal to the number of available '
                         'scaling factors.')

    image_out = np.zeros(image_in.shape)

    for i in range(image_in.shape[0]):
        swaps = scale_image(image_in[i, ], scaling[i], scaling[i])

        npix_del = swaps.shape[-1] - image_out.shape[-1]

        if npix_del == 0:
            image_out[i, ] = swaps

        else:
            if npix_del % 2 == 0:
                npix_del_a = int(npix_del/2)
                npix_del_b = int(npix_del/2)

            else:
                npix_del_a = int((npix_del-1)/2)
                npix_del_b = int((npix_del+1)/2)

            image_out[i, ] = swaps[npix_del_a:-npix_del_b, npix_del_a:-npix_del_b]

        if npix_del % 2 == 1:
            image_out[i, ] = shift_image(image_out[i, ], (-0.5, -0.5), interpolation='spline')

    return image_out


@typechecked
def scaling_factors(wavelengths: np.ndarray) -> np.ndarray:
    """
    Function to calculate the scaling factors for SDI.

    Parameters
    ----------
    wavelengths : np.ndarray
        Array with the wavelength of each frame.

    Returns
    -------
    np.ndarray
        Scaling factors.
    """

    return max(wavelengths) / wavelengths
