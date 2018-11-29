"""
Modules for the detection and interpolation of bad pixels.
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import copy

import cv2
import numpy as np

from numba import jit
from six.moves import range

from PynPoint.Core.Processing import ProcessingModule


@jit(cache=True)
def _calc_fast_convolution(F_roof_tmp, W, tmp_s, N_size, tmp_G, N):
    new = np.zeros(N, dtype=np.complex64)

    if ((tmp_s[0] == 0) and (tmp_s[1] == 0)) or \
            ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == 0)) or \
            ((tmp_s[0] == 0) and (tmp_s[1] == N[1] / 2)) or \
            ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == N[1] / 2)):

        for m in range(0, N[0], 1):
            for j in range(0, N[1], 1):
                new[m, j] = F_roof_tmp * W[m - tmp_s[0], j - tmp_s[1]]

    else:

        for m in range(0, N[0], 1):
            for j in range(0, N[1], 1):
                new[m, j] = (F_roof_tmp * W[m - tmp_s[0], j - tmp_s[1]] +
                             np.conjugate(F_roof_tmp) * W[(m + tmp_s[0]) % N[0], (j + \
                             tmp_s[1]) % N[1]])

    if ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == 0)) or \
            ((tmp_s[0] == 0) and (tmp_s[1] == N[1] / 2)) or \
            ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == N[1] / 2)): # causes problems, unknown why

        res = new / float(N_size)

    else:

        res = new / float(N_size)

    tmp_G = tmp_G - res

    return tmp_G


def _bad_pixel_interpolation(image_in,
                             bad_pixel_map,
                             iterations):
    """"
    Internal function to interpolate bad pixels.

    :param image_in: Input image.
    :type image_in: ndarray
    :param bad_pixel_map: Bad pixel map.
    :type bad_pixel_map: ndarray
    :param iterations: Number of iterations.
    :type iterations: int

    :return: Image in which the bad pixels have been interpolated.
    :rtype: ndarray
    """

    image_in = image_in * bad_pixel_map

    # for names see ref paper
    g = copy.deepcopy(image_in)
    G = np.fft.fft2(g)
    w = copy.deepcopy(bad_pixel_map)
    W = np.fft.fft2(w)

    N = g.shape
    N_size = float(N[0] * N[1])
    F_roof = np.zeros(N, dtype=complex)
    tmp_G = copy.deepcopy(G)

    iteration = 0

    while iteration < iterations:
        # 1.) select line using max search and compute conjugate
        tmp_s = np.unravel_index(np.argmax(abs(tmp_G.real[:, 0: N[1] // 2])),
                                 (N[0], N[1] // 2))

        tmp_s_conjugate = (np.mod(N[0] - tmp_s[0], N[0]),
                           np.mod(N[1] - tmp_s[1], N[1]))

        # 2.) compute the new F_roof
        # special cases s = 0 or s = N/2 no conjugate line exists
        if ((tmp_s[0] == 0) and (tmp_s[1] == 0)) or \
                ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == 0)) or \
                ((tmp_s[0] == 0) and (tmp_s[1] == N[1] / 2)) or \
                ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == N[1] / 2)):
            F_roof_tmp = N_size * tmp_G[tmp_s] / W[(0, 0)]

            # 3.) update F_roof
            F_roof[tmp_s] += F_roof_tmp

        # conjugate line exists
        else:
            a = (np.power(np.abs(W[(0, 0)]), 2))
            b = np.power(np.abs(W[(2 * tmp_s[0]) % N[0], (2 * tmp_s[1]) % N[1]]), 2)

            if a == b:
                W[(2 * tmp_s[0]) % N[0], (2 * tmp_s[1]) % N[1]] += 0.00000000001

            a = (np.power(np.abs(W[(0, 0)]), 2))
            b = np.power(np.abs(W[(2 * tmp_s[0]) % N[0], (2 * tmp_s[1]) % N[1]]),
                         2.0) + 0.01
            c = a - b

            F_roof_tmp = N_size * (tmp_G[tmp_s] * W[(0, 0)] - np.conj(tmp_G[tmp_s]) *
                                   W[(2 * tmp_s[0]) % N[0], (2 * tmp_s[1]) % N[1]]) / c

            # 3.) update F_roof
            F_roof[tmp_s] += F_roof_tmp
            F_roof[tmp_s_conjugate] += np.conjugate(F_roof_tmp)

        # 4.) calc the new error spectrum using fast numba function
        tmp_G = _calc_fast_convolution(F_roof_tmp, W, tmp_s, N_size, tmp_G, N)

        iteration += 1

    return image_in * bad_pixel_map + np.fft.ifft2(F_roof).real * (1 - bad_pixel_map)


# @jit(cache=True)
# def _sigma_detection(dev_image,
#                      var_image,
#                      source_image,
#                      out_image):
#     """"
#     Internal function to create a map with ones and zeros.
#
#     :param dev_image: Image of pixel deviations from neighborhood means, squared.
#     :type dev_image: ndarray
#     :param var_image: Image of pixel neighborhood variances * (N_sigma)^2.
#     :type var_image: ndarray
#     :param source_image: Input image.
#     :type source_image: ndarray
#     :param out_image: Bad pixel map.
#     :type out_image: ndarray
#
#     :return: None
#     """
#
#     for i in range(source_image.shape[0]):
#         for j in range(source_image.shape[1]):
#             if dev_image[i][j] < var_image[i][j]:
#                 out_image[i][j] = 1
#             else:
#                 out_image[i][j] = 0


class BadPixelSigmaFilterModule(ProcessingModule):
    """
    Module for finding bad pixels with a sigma filter and replacing them with the mean value of
    the surrounding pixels.
    """

    @staticmethod
    @jit(cache=True)
    def _sigma_filter(dev_image,
                      var_image,
                      mean_image,
                      source_image,
                      out_image,
                      bad_pixel_map):

        for i in range(source_image.shape[0]):
            for j in range(source_image.shape[1]):

                if dev_image[i][j] < var_image[i][j]:
                    out_image[i][j] = source_image[i][j]

                else:
                    out_image[i][j] = mean_image[i][j]
                    bad_pixel_map[i][j] = 0

    def __init__(self,
                 name_in="sigma_filtering",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_bp_clean",
                 map_out_tag=None,
                 box=9,
                 sigma=5,
                 iterate=1):
        """
        Constructor of BadPixelSigmaFilterModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str
        :param map_out_tag: Tag of the database entry with the bad pixel map that is written as
                            output. No data is written if set to None.
        :type map_out_tag: str
        :param box: Size of the sigma filter.
        :type box: int
        :param sigma: Sigma threshold.
        :type sigma: float
        :param iterate: Number of iterations.
        :type iterate: int

        :return: None
        """

        super(BadPixelSigmaFilterModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        if map_out_tag is None:
            self.m_map_out_port = None
        else:
            self.m_map_out_port = self.add_output_port(map_out_tag)

        self.m_box = box
        self.m_sigma = sigma
        self.m_iterate = iterate

    def run(self):
        """
        Run method of the module. Finds bad pixels with a sigma filter, replaces bad pixels with
        the mean value of the surrounding pixels, and writes the cleaned images to the database.

        :return: None
        """

        def _bad_pixel_sigma_filter(image_in,
                                    box,
                                    sigma,
                                    iterate):

            # algorithm adapted from http://idlastro.gsfc.nasa.gov/ftp/pro/image/sigma_filter.pro

            bad_pixel_map = np.ones(image_in.shape)

            if iterate < 1:
                iterate = 1

            while iterate > 0:
                box2 = box * box

                source_image = copy.deepcopy(image_in)

                mean_image = (cv2.blur(copy.deepcopy(source_image),
                                       (box, box)) * box2 - source_image) / (box2 - 1)

                dev_image = (mean_image - source_image) ** 2

                fact = float(sigma ** 2) / (box2 - 2)
                var_image = fact * (cv2.blur(copy.deepcopy(dev_image),
                                             (box, box)) * box2 - dev_image)

                out_image = image_in

                self._sigma_filter(dev_image,
                                   var_image,
                                   mean_image,
                                   source_image,
                                   out_image,
                                   bad_pixel_map)

                iterate -= 1

            if self.m_map_out_port is not None:
                self.m_map_out_port.append(bad_pixel_map, data_dim=3)

            return out_image

        if self.m_map_out_port is not None:
            self.m_map_out_port.del_all_data()
            self.m_map_out_port.del_all_attributes()

        self.apply_function_to_images(_bad_pixel_sigma_filter,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running BadPixelSigmaFilterModule...",
                                      func_args=(self.m_box,
                                                 self.m_sigma,
                                                 self.m_iterate))

        self.m_image_out_port.add_history_information("Bad pixel cleaning",
                                                      "Sigma filter = " + str(self.m_sigma))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        if self.m_map_out_port is not None:

            self.m_map_out_port.add_history_information("Bad pixel cleaning",
                                                        "Sigma filter = " + str(self.m_sigma))

            self.m_map_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class BadPixelMapModule(ProcessingModule):
    """
    Module to create a bad pixel map from the dark frames and flat fields.
    """

    def __init__(self,
                 name_in="bad_pixel_map",
                 dark_in_tag="dark",
                 flat_in_tag="flat",
                 bp_map_out_tag="bp_map",
                 dark_threshold=0.2,
                 flat_threshold=0.2):
        """
        Constructor of BadPixelMapModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param dark_in_tag: Tag of the database entry with the dark frames that are read as input.
                            Not read if set to None.
        :type dark_in_tag: str
        :param flat_in_tag: Tag of the database entry with the flat fields that are read as input.
                            Not read if set to None.
        :type flat_in_tag: str
        :param bp_map_out_tag: Tag of the database entry with the bad pixel map that is written as
                               output.
        :type bp_map_out_tag: str
        :param dark_threshold: Fractional threshold with respect to the maximum pixel value in the
                               dark frame to flag bad pixels.
        :type dark_threshold: float
        :param flat_threshold: Fractional threshold with respect to the maximum pixel value in the
                               flat field to flag bad pixels.
        :type flat_threshold: float

        :return: None
        """

        super(BadPixelMapModule, self).__init__(name_in)

        if dark_in_tag is None:
            self.m_dark_port = None
        else:
            self.m_dark_port = self.add_input_port(dark_in_tag)

        if flat_in_tag is None:
            self.m_flat_port = None
        else:
            self.m_flat_port = self.add_input_port(flat_in_tag)

        self.m_bp_map_out_port = self.add_output_port(bp_map_out_tag)

        self.m_dark_threshold = dark_threshold
        self.m_flat_threshold = flat_threshold

    def run(self):
        """
        Run method of the module. Collapses a cube of dark frames and flat fields if needed, flags
        bad pixels by comparing the pixel values with the threshold times the maximum value, and
        writes a bad pixel map to the database. For the dark frame, pixel values larger than the
        threshold will be flagged while for the flat frame pixel values smaller than the threshold
        will be flagged.

        :return: None
        """

        if self.m_dark_port is not None:
            dark = self.m_dark_port.get_all()

            if dark.ndim == 3:
                dark = np.mean(dark, axis=0)

            max_dark = np.max(dark)
            print("Threshold dark frame [counts] =", max_dark*self.m_dark_threshold)

            bpmap = np.ones(dark.shape)
            bpmap[np.where(dark > max_dark*self.m_dark_threshold)] = 0

        if self.m_flat_port is not None:
            flat = self.m_flat_port.get_all()

            if flat.ndim == 3:
                flat = np.mean(flat, axis=0)

            max_flat = np.max(flat)
            print("Threshold flat field [counts] =", max_flat*self.m_flat_threshold)

            if self.m_dark_port is None:
                bpmap = np.ones(flat.shape)

            bpmap[np.where(flat < max_flat*self.m_flat_threshold)] = 0

        if self.m_dark_port is not None and self.m_flat_port is not None:
            if not dark.shape == flat.shape:
                raise ValueError("Dark and flat images should have the same shape.")

        self.m_bp_map_out_port.set_all(bpmap, data_dim=2)

        if self.m_dark_port is not None:
            self.m_bp_map_out_port.copy_attributes_from_input_port(self.m_dark_port)
        elif self.m_flat_port is not None:
            self.m_bp_map_out_port.copy_attributes_from_input_port(self.m_flat_port)

        self.m_bp_map_out_port.close_port()


class BadPixelInterpolationModule(ProcessingModule):
    """
    Module to interpolate bad pixels with spectral deconvolution.
    """

    def __init__(self,
                 name_in="bad_pixel_interpolation",
                 image_in_tag="im_arr",
                 bad_pixel_map_tag="bp_map",
                 image_out_tag="im_arr_bp_clean",
                 iterations=1000):
        """
        Constructor of BadPixelInterpolationModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with the images that are read as input.
        :type image_in_tag: str
        :param bad_pixel_map_tag: Tag of the database entry with the bad pixel map that is read
                                  as input.
        :type bad_pixel_map_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str
        :param iterations: Number of iterations of the spectral deconvolution.
        :type iterations: int

        :return: None
        """

        super(BadPixelInterpolationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_bp_map_in_port = self.add_input_port(bad_pixel_map_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_iterations = iterations

    def run(self):
        """
        Run method of the module. Interpolates bad pixels with an iterative spectral deconvolution.

        :return: None
        """

        bad_pixel_map = self.m_bp_map_in_port.get_all()
        im_shape = self.m_image_in_port.get_shape()

        if self.m_iterations > im_shape[1]*im_shape[2]:
            raise ValueError("Maximum number of iterations needs to be smaller than the number of "
                             "pixels in the image.")

        if bad_pixel_map.shape[0] != im_shape[-2] or bad_pixel_map.shape[1] != im_shape[-1]:
            raise ValueError("The shape of the bad pixel map does not match the shape of the "
                             "images.")

        def _image_interpolation(image_in):
            return _bad_pixel_interpolation(image_in,
                                            bad_pixel_map,
                                            self.m_iterations)

        self.apply_function_to_images(_image_interpolation,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running BadPixelInterpolationModule...")

        self.m_image_out_port.add_history_information("Bad pixel interpolation",
                                                      "Iterations = " + str(self.m_iterations))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()


class BadPixelTimeFilterModule(ProcessingModule):
    """
    Module for finding bad pixels with a sigma filter along a pixel line in time. This module is
    suitable for removing bad pixels that are only present at a position in a small number of
    images, for example because a dither pattern has been applied. Pixel lines can be processed
    in parallel by setting the CPU keyword in the configuration file.
    """

    def __init__(self,
                 name_in="bp_time",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_bp_time",
                 sigma=(5., 5.)):
        """
        Constructor of BadPixelTimeFilterModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str
        :param sigma: Lower and upper sigma threshold (lower, upper).
        :type sigma: (float, float)

        :return: None
        """

        super(BadPixelTimeFilterModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_sigma = sigma

    def run(self):
        """
        Run method of the module. Finds bad pixels along a pixel line, replaces the bad pixels with
        the mean value of the pixels (excluding the bad pixels), and writes the cleaned images to
        the database.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        def _time_filter(timeline, sigma):
            median = np.median(timeline)
            std = np.std(timeline)

            index_lower = np.argwhere(timeline < median-sigma[0]*std)
            index_upper = np.argwhere(timeline > median+sigma[1]*std)

            if index_lower.size > 0:
                mask = np.ones(timeline.shape, dtype=bool)
                mask[index_lower] = False
                timeline[index_lower] = np.mean(timeline[mask])

            if index_upper.size > 0:
                mask = np.ones(timeline.shape, dtype=bool)
                mask[index_upper] = False
                timeline[index_upper] = np.mean(timeline[mask])

            return timeline

        sys.stdout.write("Running BadPixelTimeFilterModule...")
        sys.stdout.flush()

        self.apply_function_in_time(_time_filter,
                                    self.m_image_in_port,
                                    self.m_image_out_port,
                                    func_args=(self.m_sigma, ))

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.add_history_information("BadPixelTimeFilterModule",
                                                      "sigma = " + str(self.m_sigma))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()


class ReplaceBadPixelsModule(ProcessingModule):
    """
    Module for replacing bad pixels with the mean are median value of the surrounding pixels. The
    bad pixels are selected from the input bad pixel map.
    """

    def __init__(self,
                 name_in="bp_replace",
                 image_in_tag="im_arr",
                 map_in_tag="bp_map",
                 image_out_tag="im_arr_bp_replace",
                 size=2,
                 replace="mean"):
        """
        Constructor of ReplaceBadPixelsModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str
        :param sigma: Lower and upper sigma threshold (lower, upper).
        :type sigma: (float, float)
        :param size: Number of pixel lines around the bad pixel that is used to calculate the
                     mean or median replacement value. For example, a 5x5 window is used if
                     _size_=2.
        :type size: int
        :param replace: Replace the bad pixel with the mean ('mean'), median ('media'), or
                        NaN ('nan').
        :type replace: str

        :return: None
        """

        super(ReplaceBadPixelsModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_map_in_port = self.add_input_port(map_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_size = size
        self.m_replace = replace

    def run(self):
        """
        Run method of the module. Masks the bad pixels with NaN and replaces the bad pixels with the
        mean or median value (excluding the bad pixels) within a window centered on the bad pixel.
        The original value is used if there are only NaNs within the window.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        bpmap = self.m_map_in_port.get_all()
        index = np.argwhere(bpmap == 0)

        def _replace_pixels(image, index):

            im_mask = np.copy(image)

            for _, item in enumerate(index):
                im_mask[item[0], item[1]] = np.nan

            for _, item in enumerate(index):
                im_tmp = im_mask[item[0]-self.m_size:item[0]+self.m_size+1,
                                 item[1]-self.m_size:item[1]+self.m_size+1]

                if np.size(np.where(im_tmp != np.nan)[0]) == 0:
                    im_mask[item[0], item[1]] = image[item[0], item[1]]
                else:
                    if self.m_replace == "mean":
                        im_mask[item[0], item[1]] = np.nanmean(im_tmp)
                    elif self.m_replace == "median":
                        im_mask[item[0], item[1]] = np.nanmedian(im_tmp)
                    elif self.m_replace == "nan":
                        im_mask[item[0], item[1]] = np.nan

            return im_mask

        self.apply_function_to_images(_replace_pixels,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running ReplaceBadPixelsModule...",
                                      func_args=(index, ))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()
