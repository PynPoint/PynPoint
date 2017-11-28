"""
Modules with simple pre-processing tools.
"""

import math

from PynPoint.core import ProcessingModule

from skimage.transform import rescale
from scipy.ndimage import shift
from scipy.ndimage.filters import gaussian_filter
import numpy as np


class CutAroundCenterModule(ProcessingModule):
    """
    Module for cropping around the center of an image.
    """

    def __init__(self,
                 new_shape,
                 name_in="cut_around_center",
                 image_in_tag="im_arr",
                 image_out_tag="cut_im_arr",
                 number_of_images_in_memory=100):
        """
        Constructor of CutAroundCenterModule.

        :param new_shape: Tuple (delta_x, delta_y) with the new image size.
        :type new_shape: tuple, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str
        :param num_image_in_memory: Number of frames that are simultaneously loaded into the memory.
        :type num_image_in_memory: int
        :return: None
        """

        super(CutAroundCenterModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_shape = new_shape

    def run(self):
        """
        Run method of the module. Reduces the image size by cropping around the center of the
        original image.

        :return: None
        """

        def image_cutting(image_in,
                          shape_in):

            shape_of_input = image_in.shape

            if shape_in[0] > shape_of_input[0] or shape_in[1] > shape_of_input[1]:
                raise ValueError("Input frame resolution smaller than target image resolution.")

            x_off = (shape_of_input[0] - shape_in[0]) / 2
            y_off = (shape_of_input[1] - shape_in[1]) / 2

            return image_in[y_off: shape_in[1] + y_off, x_off: shape_in[0] + x_off]

        self.apply_function_to_images(image_cutting,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(self.m_shape,),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        self.m_image_out_port.add_history_information("Cropped image size to",
                                                      str(self.m_shape))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class CutAroundPositionModule(ProcessingModule):
    """
    Module for cropping around a given position of an image.
    """

    def __init__(self,
                 new_shape,
                 center_of_cut,
                 name_in="cut_around_position",
                 image_in_tag="im_arr",
                 image_out_tag="cut_im_arr",
                 number_of_images_in_memory=100):
        """
        Constructor of CutAroundPositionModule.

        :param new_shape: Tuple (delta_x, delta_y) with the new image size.
        :type new_shape: tuple, int
        :param center_of_cut: Tuple (x0, y0) with the new image center. Python indexing starts at 0.
        :type center_of_cut: tuple, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str
        :param num_image_in_memory: Number of frames that are simultaneously loaded into the memory.
        :type num_image_in_memory: int
        :return: None
        """

        super(CutAroundPositionModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_shape = new_shape
        self.m_center_of_cut = center_of_cut

    def run(self):
        """
        Run method of the module. Reduces the image size by cropping around an given position.

        :return: None
        """

        def image_cutting(image_in,
                          shape_in,
                          center_of_cut_in):

            shape_of_input = image_in.shape

            if shape_in[0] > shape_of_input[0] or shape_in[1] > shape_of_input[1]:
                raise ValueError("Input frame resolution smaller than target image resolution.")

            x_off = center_of_cut_in[0] - (shape_in[0] / 2)
            y_off = center_of_cut_in[1] - (shape_in[1] / 2)

            return image_in[y_off: shape_in[1] + y_off, x_off: shape_in[0] + x_off]

        self.apply_function_to_images(image_cutting,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(self.m_shape, self.m_center_of_cut),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        self.m_image_out_port.add_history_information("Cropped image size to",
                                                      str(self.m_shape))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class ScaleFramesModule(ProcessingModule):
    """
    Module for rescaling of an image.
    """

    def __init__(self,
                 scaling_factor,
                 name_in="scaling",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_scaled",
                 number_of_images_in_memory=100):
        """
        Constructor of ScaleFramesModule.

        :param scaling_factor: Scaling factor for upsampling (*scaling_factor* > 1) and downsampling
                               (0 < *scaling_factor* < 1).
        :type scaling_factor: float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str
        :param num_image_in_memory: Number of frames that are simultaneously loaded into the memory.
        :type num_image_in_memory: int
        :return: None
        """

        super(ScaleFramesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_scaling = scaling_factor

    def run(self):
        """
        Run method of the module. Rescales an image with a fifth order spline interpolation and a
        reflecting boundary condition.

        :return: None
        """

        def image_scaling(image_in,
                          scaling):

            sum_before = np.sum(image_in)
            tmp_image = rescale(image=np.asarray(image_in,
                                                 dtype=np.float64),
                                scale=(scaling,
                                       scaling),
                                order=5,
                                mode="reflect")

            sum_after = np.sum(tmp_image)
            return tmp_image * (sum_before / sum_after)

        self.apply_function_to_images(image_scaling,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(self.m_scaling,),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        self.m_image_out_port.add_history_information("Scaled by a factor of",
                                                      str(self.m_scaling))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class ShiftForCenteringModule(ProcessingModule):
    """
    Module for shifting of an image.
    """

    def __init__(self,
                 shift_vector,
                 name_in="shift",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_shifted",
                 number_of_images_in_memory=100):
        """
        Constructor of ShiftForCenteringModule.

        :param shift_vector: Tuple (delta_y, delta_x) with the shift in both directions.
        :type new_shape: tuple, float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str
        :param num_image_in_memory: Number of frames that are simultaneously loaded into the memory.
        :type num_image_in_memory: int
        :return: None
        """

        super(ShiftForCenteringModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_shift_vector = shift_vector

    def run(self):
        """
        Run method of the module. Shifts an image with a fifth order spline interpolation.

        :return: None
        """

        def image_shift(image_in):

            return shift(image_in, self.m_shift_vector, order=5)

        self.apply_function_to_images(image_shift,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        self.m_image_out_port.add_history_information("Shifted by",
                                                      str(self.m_shift_vector))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()

class LocateStarModule(ProcessingModule):
    """
    Module for locating the position of the star.
    """

    def __init__(self,
                 name_in="locate_star",
                 data_tag="im_arr",
                 gaussian_fwhm=7):
        """
        Constructor of LocateStarModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param data_tag: Tag of the database entry for which the star positions are written as
                         attributes.
        :type data_tag: str
        :param gaussian_fwhm: Full width at half maximum (pix) of the Gaussian kernel that is used
                              to smooth the image before the star is located.
        :type gaussian_fwhm: float
        :return: None
        """

        super(LocateStarModule, self).__init__(name_in)

        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

        self.m_gaussian_fwhm = gaussian_fwhm

    def run(self):
        """
        Run method of the module. Smooths the image with a Gaussian kernel, finds the largest
        pixel value, and writes the STAR_POSITION attribute.

        :return: None
        """

        sigma = self.m_gaussian_fwhm/math.sqrt(8.*math.log(2.))

        star_position = np.zeros((self.m_data_in_port.get_shape()[0], 2))

        for i in range(self.m_data_in_port.get_shape()[0]):
            im_smooth = gaussian_filter(self.m_data_in_port[i],
                                        sigma,
                                        truncate=4.)

            star_position[i, :] = np.unravel_index(im_smooth.argmax(), im_smooth.shape)

        self.m_data_out_port.add_attribute("STAR_POSITION_X",
                                           star_position[:, 1],
                                           static=False)

        self.m_data_out_port.add_attribute("STAR_POSITION_Y",
                                           star_position[:, 0],
                                           static=False)
