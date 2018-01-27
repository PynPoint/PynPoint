"""
Pypline Modules for the detection and interpolation of bad pixels
"""
import numpy as np
import copy
import cv2
from numba import jit

from PynPoint.core.Processing import ProcessingModule

# Function needed in multiple moduels
# - fast numba methods ----------------------- #


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
                             np.conjugate(F_roof_tmp) * W[
                                 (m + tmp_s[0]) % N[0], (j + tmp_s[1]) % N[1]])

    if ((tmp_s[0] == N[0] / 2) and (tmp_s[1] == 0)) or \
            ((tmp_s[0] == 0) and (tmp_s[1] == N[1] / 2)) or \
            ((tmp_s[0] == N[0] / 2) and (
                        tmp_s[1] == N[1] / 2)):  # seems to make problems ...  no idea why ;)
        res = new / float(N_size)
    else:
        res = new / float(N_size)

    tmp_G = tmp_G - res
    return tmp_G
# -------------------------------------------- #


def _bad_pixel_interpolation_image(image_in,
                                   bad_pixel_map,
                                   iterations):
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
        tmp_s = np.unravel_index(np.argmax(abs(tmp_G.real[:, 0: N[1] / 2])),
                                 (N[0], N[1] / 2))

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


@jit(cache=True)
def _sigma_detection(dev_image,
                     var_image,
                     source_image,
                     out_image):
    for i in range(source_image.shape[0]):
        for j in range(source_image.shape[1]):
            if dev_image[i][j] < var_image[i][j]:
                out_image[i][j] = 1
            else:
                out_image[i][j] = 0


class BadPixelCleaningSigmaFilterModule(ProcessingModule):

    # ------ fast numba functions -------
    @staticmethod
    @jit(cache=True)
    def __sigma_filter(dev_image,
                       var_image,
                       mean_image,
                       source_image,
                       out_image):

        for i in range(source_image.shape[0]):
            for j in range(source_image.shape[1]):
                if dev_image[i][j] < var_image[i][j]:
                    out_image[i][j] = source_image[i][j]
                else:
                    out_image[i][j] = mean_image[i][j]

    # -----------------------------------

    def __init__(self,
                 name_in="sigma_filtering",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_bp_clean",
                 box=9,
                 sigma=5,
                 iterate=1):

        super(BadPixelCleaningSigmaFilterModule, self).__init__(name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)

        self.m_image_out_port = self.add_output_port(image_out_tag)

        # Properties
        self.m_box = box
        self.m_sigma = sigma
        self.m_iterate = iterate

    def run(self):

        def bad_pixel_clean_sigmafilter_image(image_in,
                                              box,
                                              sigma,
                                              iterate):
            """
            This function filters a single image using sigmafilter.
            :param image_in: Input image
            :type image_in: ndarray
            :param box: The filter Mask size of the sigma filter
            :type box: int
            :param sigma: The sigma Threshold
            :type sigma: int
            :param iterate: Number of iterations.
            :type iterate: int
            :return: The filtered image.
            :rtype: ndarray
            """

            if iterate < 1:
                iterate = 1

            while iterate > 0:
                box2 = box * box

                # copy algorithm from http://idlastro.gsfc.nasa.gov/ftp/pro/image/sigma_filter.pro

                source_image = copy.deepcopy(image_in)

                mean_image = (cv2.blur(copy.deepcopy(source_image),
                                       (box, box)) * box2 - source_image) / (box2 - 1)

                dev_image = (mean_image - source_image) ** 2

                fact = float(sigma ** 2) / (box2 - 2)
                var_image = fact * (cv2.blur(copy.deepcopy(dev_image),
                                             (box, box)) * box2 - dev_image)

                out_image = image_in

                self.__sigma_filter(dev_image,
                                    var_image,
                                    mean_image,
                                    source_image,
                                    out_image)
                iterate -= 1

            return out_image

        self.m_num_images_in_memory = self._m_config_port.get_attribute("MEMORY")

        self.apply_function_to_images(bad_pixel_clean_sigmafilter_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running BadPixelCleaningSigmaFilterModule...",
                                      func_args=(self.m_box,
                                                 self.m_sigma,
                                                 self.m_iterate),
                                      num_images_in_memory=self.m_num_images_in_memory)

        history = "Sigma filtering with Sigma = " + str(self.m_sigma)
        self.m_image_out_port.add_history_information("bp_cleaning",
                                                      history)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class BadPixelMapCreationModule(ProcessingModule):

    def __init__(self,
                 name_in="Bad_Pixel_Map_creation",
                 dark_in_tag="dark",
                 flat_in_tag="flat",
                 bp_map_out_tag="bp_map",
                 dark_threshold=0.2,
                 flat_threshold=.2):

        super(BadPixelMapCreationModule, self).__init__(name_in)

        # Ports
        self.m_dark_port = self.add_input_port(dark_in_tag)
        self.m_flat_port = self.add_input_port(flat_in_tag)

        self.m_bp_map_out_port = self.add_output_port(bp_map_out_tag)

        # Properties
        self.m_dark_threshold = dark_threshold
        self.m_flat_threshold = flat_threshold

    def run(self):
        # load dark and flat
        dark = self.m_dark_port.get_all()
        flat = self.m_flat_port.get_all()

        if dark.ndim == 3:
            dark = np.mean(dark, axis=0)
        if flat.ndim == 3:
            flat = np.mean(flat, axis=0)

        if not dark.shape == flat.shape:
            raise ValueError("Dark and Flat need to have the same resolution")

        max_val_dark = np.max(dark)
        max_val_flat = np.max(flat)

        bpmap = np.ones(dark.shape)
        bpmap[np.where(dark > max_val_dark * self.m_dark_threshold)] = 0
        bpmap[np.where(flat < max_val_flat * self.m_flat_threshold)] = 0

        self.m_bp_map_out_port.set_all(bpmap)

        self.m_bp_map_out_port.close_port()


class BadPixelInterpolationModule(ProcessingModule):

    def __init__(self,
                 name_in="Bad_Pixel_Interpolation",
                 image_in_tag="im_arr",
                 bad_pixel_map_tag="bp_map",
                 image_out_tag="im_arr_bp_clean",
                 iterations=1000):

        super(BadPixelInterpolationModule, self).__init__(name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_bp_map_in_port = self.add_input_port(bad_pixel_map_tag)

        self.m_image_out_port = self.add_output_port(image_out_tag)

        # Properties
        self.m_iterations = iterations

    def run(self):

        self.m_num_images_in_memory = self._m_config_port.get_attribute("MEMORY")

        # load bad pixel map
        bad_pixel_map = self.m_bp_map_in_port.get_all()

        # error cases
        if self.m_iterations > self.m_image_in_port.get_shape()[1] \
            * self.m_image_in_port.get_shape()[2]:
            raise ValueError(
                "Maximum number of iterations needs to be smaller than the number of pixels in the "
                "given image")

        if bad_pixel_map.shape[0] != self.m_image_in_port.get_shape()[1] or \
           bad_pixel_map.shape[1] != self.m_image_in_port.get_shape()[2]:
           raise ValueError("The given size of the input bad pixel map does not fit the dataset "
                            "size. Beforehand using this module run one of the Simple Tools to "
                            "fit the size.")

        def bad_pixel_interpolation_image(image_in):
            return _bad_pixel_interpolation_image(image_in,
                                                  bad_pixel_map,
                                                  self.m_iterations)

        self.apply_function_to_images(bad_pixel_interpolation_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running BadPixelInterpolationModule...",
                                      num_images_in_memory=self.m_num_images_in_memory)

        history = "Iterations = " + str(self.m_iterations)
        self.m_image_out_port.add_history_information("Bad Pixel Interpolation",
                                                      history)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()


class BadPixelInterpolationRefinementModule(ProcessingModule):

    def __init__(self,
                 name_in="Bad_Pixel_Interpolation_Refinement",
                 image_in_tag="im_arr",
                 bad_pixel_map_tag="bp_map",
                 star_pos_tag="star_positions",
                 image_out_tag="im_arr_bp_ref",
                 box_size=5,
                 sigma=4.,
                 iterations=5000):

        super(BadPixelInterpolationRefinementModule, self).__init__(name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_bp_map_in_port = self.add_input_port(bad_pixel_map_tag)
        self.m_pos_in_port = self.add_input_port(star_pos_tag)

        self.m_image_out_port = self.add_output_port(image_out_tag)

        # Properties
        self.m_iterations = iterations
        self.m_current_pos = 0  # TODO fix this for multi threading
        self.m_box_size = box_size
        self.m_sigma = sigma

    def run(self):

        self.m_num_images_in_memory = self._m_config_port.get_attribute("MEMORY")

        # load bad pixel map
        bad_pixel_map = self.m_bp_map_in_port.get_all()
        new_bp_shape = (self.m_image_in_port.get_shape()[1], self.m_image_in_port.get_shape()[2])

        # error cases
        if self.m_iterations > new_bp_shape[0] * new_bp_shape[1]:
            raise ValueError(
                "Maximum number of iterations needs to be smaller than the number of pixels in the "
                "given image")

        positions = self.m_pos_in_port.get_all()

        def bad_pixel_int_ref_image(image_in):
            current_pos = positions[self.m_current_pos]
            self.m_current_pos += 1

            start_x = int(current_pos[0] - new_bp_shape[0]/2)
            end_x = int(current_pos[0] + new_bp_shape[0]/2)
            start_y = int(current_pos[1] - new_bp_shape[1]/2)
            end_y = int(current_pos[1] + new_bp_shape[1]/2)

            tmp_bad_pixel_map = bad_pixel_map[start_x: end_x,
                                              start_y: end_y]

            # sigma detection
            source_image = copy.deepcopy(image_in)

            mean_image = (cv2.blur(copy.deepcopy(source_image),
                                   (self.m_box_size,
                                    self.m_box_size)) * self.m_box_size**2 - source_image) / \
                         (self.m_box_size**2 - 1)

            dev_image = (mean_image - source_image) ** 2

            fact = float(self.m_sigma ** 2) / (self.m_box_size**2 - 2)
            var_image = fact * (cv2.blur(copy.deepcopy(dev_image),
                                         (self.m_box_size,
                                          self.m_box_size)) * self.m_box_size**2 - dev_image)

            out_image = np.ones(image_in.shape)

            _sigma_detection(dev_image,
                             var_image,
                             source_image,
                             out_image)

            tmp_bad_pixel_map = np.minimum(out_image, tmp_bad_pixel_map)

            return _bad_pixel_interpolation_image(image_in,
                                                  tmp_bad_pixel_map,
                                                  self.m_iterations)

        self.apply_function_to_images(bad_pixel_int_ref_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running BadPixelInterpolationRefinementModule...",
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_image_out_port.add_history_information("Bad Pixel Interpolation Refinement",
                                                      "Iterations = "+str(self.m_iterations))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()
