"""
Functions for combining the residuals of the PSF subtraction.
"""

from __future__ import absolute_import

import numpy as np

from scipy.ndimage import rotate
from six.moves import range


def combine_residuals(method, res_rot, residuals=None, angles=None):
    """
    Function for combining the derotated residuals of the PSF subtraction.

    :param method: Method used for combining the residuals ("mean", "median", "weighted", or
                   "clipped").
    :type method: str
    :param res_rot: Derotated residuals of the PSF subtraction (3D).
    :type res_rot: numpy.ndimage
    :param residuals: Non-derotated residuals of the PSF subtraction (3D). Only required for the
                      noise-weighted residuals.
    :type residuals: numpy.ndimage
    :param angles: Derotation angles (deg). Only required for the noise-weighted residuals.
    :type angles: numpy.ndimage

    :return: Combined residuals (2D).
    :rtype: numpy.ndimage
    """

    if method == "mean":
        stack = np.mean(res_rot, axis=0)

    elif method == "median":
        stack = np.median(res_rot, axis=0)

    elif method == "weighted":
        tmp_res_var = np.var(residuals, axis=0)

        res_repeat = np.repeat(tmp_res_var[np.newaxis, :, :],
                               repeats=residuals.shape[0],
                               axis=0)

        res_var = np.zeros(res_repeat.shape)
        for j, angle in enumerate(angles):
            # scipy.ndimage.rotate rotates in clockwise direction for positive angles
            res_var[j, ] = rotate(input=res_repeat[j, ],
                                  angle=angle,
                                  reshape=False)

        weight1 = np.divide(res_rot,
                            res_var,
                            out=np.zeros_like(res_var),
                            where=(np.abs(res_var) > 1e-100) & (res_var != np.nan))

        weight2 = np.divide(1.,
                            res_var,
                            out=np.zeros_like(res_var),
                            where=(np.abs(res_var) > 1e-100) & (res_var != np.nan))

        sum1 = np.sum(weight1, axis=0)
        sum2 = np.sum(weight2, axis=0)

        stack = np.divide(sum1,
                          sum2,
                          out=np.zeros_like(sum2),
                          where=(np.abs(sum2) > 1e-100) & (sum2 != np.nan))

    elif method == "clipped":
        stack = np.zeros(res_rot.shape[-2:])

        for i in range(stack.shape[0]):
            for j in range(stack.shape[1]):
                temp = res_rot[:, i, j]

                if temp.var() > 0.0:
                    no_mean = temp - temp.mean()

                    part1 = no_mean.compress((no_mean < 3.0*np.sqrt(no_mean.var())).flat)
                    part2 = part1.compress((part1 > (-1.0)*3.0*np.sqrt(no_mean.var())).flat)

                    stack[i, j] = temp.mean() + part2.mean()


    return stack
