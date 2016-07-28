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
        self.m_image_out_port.add_attribute("history: bp_cleaning",
                                            history)

        self.m_image_out_port.close_port()
