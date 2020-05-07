# -*- coding: utf-8 -*-
"""
Various function to support IFS reduction

@author: Sven Kiefer
"""

import numpy as np

from typeguard import typechecked

from pynpoint.util.image import scale_image


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
    max_f1 = 0
    min_f2 = len(data_ns[0, :, 0])

    # scale images
    for i in range(frame_nr):

        swaps = scale_image(data_ns[i, :, :], scaling[i], scaling[i])

        # Calculate assignemnd legnths (all frmaes are centerd after rescaling)
        side = len(swaps[0, :])
        siye = len(data[0, :, 0])
        f_1 = (side - siye)//2
        f_2 = (side + siye)//2

        data[i] = swaps[f_1:f_2, f_1:f_2]

        # Set lower and upper bound
        if max_f1 < f_1:
            max_f1 = f_1

        if min_f2 > f_2:
            min_f2 = f_2

    return data, max_f1, min_f2


@typechecked
def scaling_calculation(pixscale: float,
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
