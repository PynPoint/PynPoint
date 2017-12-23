"""
Modules for dark frame and flat field corrections.
"""

import warnings
import numpy as np

from PynPoint.core.Processing import ProcessingModule


def _image_cutting(data,
                   image_in_port,
                   dark=True):
    """
    Internal function which creates a master dark/flat by calculating the mean (3D data) and
    cropping the frames to the dimensions of the science data if needed.

    :param data: 2D or 3D array of dark frames.
    :type data: numpy array
    :param image_in_port: Port with the science data.
    :type image_in_port: numpy array
    :param dark: Tags which exist in the database.
    :type dark: InputPort
    :return: Dark/flat frame.
    :rtype: numpy array
    """

    if dark:
        name = "dark"
    else:
        name = "flat"

    if data.ndim == 3:
        tmp_data = np.mean(data, axis=0)
    elif data.ndim == 2:
        tmp_data = data
    else:
        raise ValueError("Dimension of input %s not supported. "
                         "Only 2D and 3D dark array can be used." % name)

    shape_of_input = (image_in_port.get_shape()[1],
                      image_in_port.get_shape()[2])

    if tmp_data.shape[0] < shape_of_input[0] or tmp_data.shape[1] < shape_of_input[1]:
        raise ValueError("Input %s resolution smaller than image resolution." % name)

    if not tmp_data.shape == shape_of_input:
        dark_shape = tmp_data.shape
        x_off = (dark_shape[0] - shape_of_input[0]) / 2
        y_off = (dark_shape[1] - shape_of_input[1]) / 2
        tmp_data = tmp_data[x_off: shape_of_input[0] + x_off, y_off:shape_of_input[1] + y_off]

        warnings.warn("The given %s has a different shape than the input images and was cut "
                      "around the center in order to get the same resolution. If the position "
                      "of the %s does not match the input frame you have to cut the %s "
                      "before using this module." % (name, name, name))

    return tmp_data


class DarkSubtractionModule(ProcessingModule):
    """
    Module to subtract a dark frame from the science data.
    """

    def __init__(self,
                 name_in="dark_subtraction",
                 image_in_tag="im_arr",
                 dark_in_tag="dark_arr",
                 image_out_tag="dark_sub_arr"):
        """
        Constructor of DarkSubtractionModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the science database that is read as input.
        :type image_in_tag: str
        :param dark_in_tag: Tag of the dark frame database that is read as input.
        :type dark_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str

        :return: None
        """

        super(DarkSubtractionModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_dark_in_port = self.add_input_port(dark_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Creates a master dark with the same dimensions as the science
        data and subtracts the dark frame from the science data.

        :return: None
        """

        self.m_num_images_in_memory = self._m_config_port.get_attribute("MEMORY")

        def dark_subtraction_image(image_in,
                                   dark_in):
            return image_in - dark_in

        dark = self.m_dark_in_port.get_all()

        tmp_dark = _image_cutting(dark,
                                  self.m_image_in_port)

        self.apply_function_to_images(dark_subtraction_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(tmp_dark,),
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_image_out_port.add_history_information("Dark subtraction",
                                                      "simple")

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class FlatSubtractionModule(ProcessingModule):
    """
    Module to apply a flat field correction to the science data.
    """

    def __init__(self,
                 name_in="flat_subtraction",
                 image_in_tag="dark_sub_arr",
                 flat_in_tag="flat_arr",
                 image_out_tag="flat_sub_arr"):
        """
        Constructor of FlatSubtractionModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the science database that is read as input.
        :type image_in_tag: str
        :param dark_in_tag: Tag of the flat field database that is read as input.
        :type dark_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str

        :return: None
        """

        super(FlatSubtractionModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_flat_in_port = self.add_input_port(flat_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Creates a master flat with the same dimensions as the science
        data and divides the science data by the flat field.

        :return: None
        """

        self.m_num_images_in_memory = self._m_config_port.get_attribute("MEMORY")

        def flat_subtraction_image(image_in,
                                   flat_in):

            return image_in / flat_in

        flat = self.m_flat_in_port.get_all()

        tmp_flat = _image_cutting(flat,
                                  self.m_image_in_port,
                                  dark=False)

        # shift all values to positive
        flat_min = np.min(tmp_flat)

        # +1 and -1 to prevent division by zero
        if flat_min < 0:
            tmp_flat -= np.ones(tmp_flat.shape) * (flat_min - 1)
        else:
            tmp_flat -= np.ones(tmp_flat.shape) * (flat_min + 1)

        flat_median = np.median(np.median(tmp_flat))

        # normalization
        tmp_flat /= float(flat_median)

        self.apply_function_to_images(flat_subtraction_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(tmp_flat, ),
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_image_out_port.add_history_information("Flat correction",
                                                      "simple")

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()
