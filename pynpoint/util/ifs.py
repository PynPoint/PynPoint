# -*- coding: utf-8 -*-
"""
Various function to support IFS reduction

@author: Sven Kiefer
"""

import numpy as np

from typeguard import typechecked

from pynpoint.util.image import scale_image, center_pixel


@typechecked
def sdi_scaling(data_ns: np.ndarray,
                scaling: np.ndarray):

    """
        Function to rescale images according to their wavelength. Used for SDI.

        Parameters
        ----------
        data_ns : numpy.ndarray
            Data to rescale
        lam : numpy.ndarray
            Wavelength of each frame in data_ns.

        Returns
        -------
        numpy.ndarray
            Rescaled data with the same shpae as data_ns
        int
            Lower bound index of non filed area of data_ns
        int
            Upper bound index of non filed area of data_ns

    """

    # Check shape
    if not data_ns.shape[0] == scaling.shape[0]:
        raise ValueError('Data and lambda do not have the same length')

    # prepare scaling
    frame_nr = len(data_ns[:, 0, 0])
    data = np.full_like(data_ns, 0)

    # scale images
    for i in range(frame_nr):

        swaps = scale_image(data_ns[i, :, :], scaling[i], scaling[i])

        x_0, y_0 = center_pixel(data)
        x_1, y_1 = center_pixel(swaps)
        x_2 = x_1 - x_0
        y_2 = y_1 - y_0

        if y_2 == 0 or x_2 == 0:
            data[i] = swaps
        else:
            data[i] = swaps[-y_2-data.shape[-2]:-y_2, -x_2-data.shape[-1]:-x_2]

    return data


@typechecked
def scaling_calculation(data_shape: float,
                        lam: np.ndarray):

    """
        Function to calculate the rescaling factors according to lambda.
        Currently only works for SPHERE/IFS

        Parameters
        ----------
        lam : numpy.ndarray
            Wavelength of each frame in data_ns.

        Returns
        -------
        numpy.ndarray
            Scaling factors calculated from lam

    """

    # physical scaling factor
    scaling = max(lam) / lam

    # adjustments due to low image resolution to reduce jitter
    for i, scal in enumerate(scaling):
        up_shape = data_shape*scal
        if up_shape <= data_shape+2:
            # if the scaling is to small, do not scale the image
            scaling[i] = 1
        else:
            # create always an odd sized image after down scaling
            if np.floor(up_shape) % 2 == 0:
                scaling[i] = np.floor(up_shape)/data_shape
            else:
                scaling[i] = (np.floor(up_shape)+1)/data_shape

    return scaling


@typechecked
def i_want_to_seperate_wavelengths(processing_type: str):
    """
    Returns True if processing_type suggests wavelength specific output'

    Parameters
    ----------
    processing_type : str
        processing type (type how to process)

    Returns
    -------
    bool
        True if wavelength specific output is wished

    """

    return processing_type[0] == 'W'
