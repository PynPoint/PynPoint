"""
Pypline Modules for the detection and interpolation of bad pixels
"""
import numpy as np
import copy
import cv2
from numba import jit

from PynPoint.core.Processing import ProcessingModule


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
                 iterate=1,
                 number_of_images_in_memory=100):

        super(BadPixelCleaningSigmaFilterModule, self).__init__(name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)

        self.m_image_out_port = self.add_output_port(image_out_tag)

        # Properties
        self.m_box = box
        self.m_sigma = sigma
        self.m_iterate = iterate

        self.m_number_of_images_in_memory = number_of_images_in_memory

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
                source_image = np.array(source_image, dtype=np.float32)

                mean_image = (cv2.blur(copy.deepcopy(source_image),
                                       (box, box)) * box2 - source_image) / (box2 - 1)
                mean_image = np.array(mean_image,
                                      dtype=np.float32)

                dev_image = (mean_image - source_image) ** 2
                dev_image = np.array(dev_image,
                                     dtype=np.float32)

                fact = float(sigma ** 2) / (box2 - 2)
                var_image = fact * (cv2.blur(copy.deepcopy(dev_image),
                                             (box, box)) * box2 - dev_image)
                var_image = np.array(var_image,
                                     dtype=np.float32)

                out_image = image_in
                out_image = np.array(out_image,
                                     dtype=np.float32)

                self.__sigma_filter(dev_image,
                                    var_image,
                                    mean_image,
                                    source_image,
                                    out_image)
                iterate -= 1

            return out_image

        self.apply_function_to_images(bad_pixel_clean_sigmafilter_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(self.m_box,
                                                 self.m_sigma,
                                                 self.m_iterate),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

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

        assert dark.shape == flat.shape

        max_val_dark = np.max(dark)
        max_val_flat = np.max(flat)
        bpmap = np.ones(dark.shape)

        for i in range(dark.shape[0]):
            for j in range(dark.shape[1]):
                if dark[i, j] > max_val_dark * self.m_dark_threshold:
                    bpmap[i, j] = 0
                if flat[i, j] < max_val_flat * self.m_flat_threshold:
                    bpmap[i, j] = 0

        plt.imshow(bpmap)
        plt.show()

        self.m_bp_map_out_port.set_all(bpmap)

        self.m_bp_map_out_port.close_port()


class BadPixelInterpolationModule(ProcessingModule):

    # - fast numba methods ----------------------- #
    @staticmethod
    @jit(cache=True)
    def __calc_fast_convolution(F_roof_tmp, W, tmp_s, N_size, tmp_G, N):
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

    def __init__(self,
                 name_in="Bad_Pixel_Interpolation",
                 image_in_tag="im_arr",
                 bad_pixel_map_tag="bp_map",
                 image_out_tag="im_arr_bp_clean",
                 iterations=5000,
                 number_of_images_in_memory=100):

        super(BadPixelInterpolationModule, self).__init__(name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_bp_map_in_port = self.add_input_port(bad_pixel_map_tag)

        self.m_image_out_port = self.add_output_port(image_out_tag)

        # Properties
        self.m_iterations = iterations

        self.m_number_of_images_in_memory = number_of_images_in_memory

    def run(self):
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
           print "Error missing here"

        def bad_pixel_interpolation_image(image_in):
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

            while iteration < self.m_iterations:
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
                tmp_G = self.__calc_fast_convolution(F_roof_tmp, W, tmp_s, N_size, tmp_G, N)

                iteration += 1

            return image_in * bad_pixel_map + np.fft.ifft2(F_roof).real * (1 - bad_pixel_map)

        self.apply_function_to_images(bad_pixel_interpolation_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        history = "Iterations = " + str(self.m_iterations)
        self.m_image_out_port.add_history_information("Bad Pixel Interpolation",
                                                      history)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()
