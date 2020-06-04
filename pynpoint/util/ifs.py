# -*- coding: utf-8 -*-
"""
Various function to support IFS reduction

@author: Sven Kiefer
"""

import numpy as np

from typeguard import typechecked

from pynpoint.util.image import scale_image, center_pixel, shift_image


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
            
        npix_del = swaps.shape[-1] - data.shape[-1]

        if npix_del == 0:
            data[i, ] = swaps
        else:
            if npix_del % 2 == 0:
                npix_del_a = int(npix_del/2)
                npix_del_b = int(npix_del/2)

            else:
                npix_del_a = int((npix_del-1)/2)
                npix_del_b = int((npix_del+1)/2)

            data[i, ] = swaps[npix_del_a:-npix_del_b, npix_del_a:-npix_del_b]

        if npix_del % 2 == 1:
            data[i, ] = shift_image(data[i, ], (-0.5, -0.5), interpolation='spline')

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
