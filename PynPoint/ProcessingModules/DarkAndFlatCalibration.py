"""
Modules for dark frame and flat field calibrations.
"""

from __future__ import absolute_import

import warnings

import numpy as np

from PynPoint.Core.Processing import ProcessingModule


def _master_frame(data,
                  image_in_port):
    """
    Internal function which creates a master dark/flat by calculating the mean (3D data) and
    cropping the frames to the shape of the science images if needed.

    :param data: Input array (2D or 3D) with dark or flat frames.
    :type data: ndarray
    :param image_in_port: Input port with the science images.
    :type image_in_port: ndarray

    :return: Master dark/flat frame.
    :rtype: ndarray
    """

    if data.ndim == 3:
        data = np.mean(data, axis=0)
    elif data.ndim != 2:
        raise ValueError("Dimension of input %s not supported. Only 2D and 3D arrays can be "
                         "used." % image_in_port.tag)

    shape_in = (image_in_port.get_shape()[1], image_in_port.get_shape()[2])

    if data.shape[0] < shape_in[0] or data.shape[1] < shape_in[1]:
        raise ValueError("Shape of the calibration images is smaller than the science images.")

    if data.shape != shape_in:
        cal_shape = data.shape

        x_off = (cal_shape[0] - shape_in[0]) / 2
        y_off = (cal_shape[1] - shape_in[1]) / 2

        data = data[x_off:x_off+shape_in[0], y_off:y_off+shape_in[1]]

        warnings.warn("The calibration images were cropped around their center to match the shape "
                      "of the science images.")

    return data


class DarkCalibrationModule(ProcessingModule):
    """
    Module to subtract a master dark from the science data.
    """

    def __init__(self,
                 name_in="dark_calibration",
                 image_in_tag="im_arr",
                 dark_in_tag="dark_arr",
                 image_out_tag="dark_cal_arr"):
        """
        Constructor of DarkCalibrationModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with the science images that are read as
                             input.
        :type image_in_tag: str
        :param dark_in_tag: Tag of the database with the dark frames that are read as input.
        :type dark_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str

        :return: None
        """

        super(DarkCalibrationModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_dark_in_port = self.add_input_port(dark_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Creates a master dark with the same shape as the science
        data and subtracts the dark frame from the science data.

        :return: None
        """

        def _dark_calibration(image_in, dark_in):
            return image_in - dark_in

        dark = self.m_dark_in_port.get_all()
        master = _master_frame(dark, self.m_image_in_port)

        self.apply_function_to_images(_dark_calibration,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running DarkCalibrationModule...",
                                      func_args=(master, ))

        self.m_image_out_port.add_history_information("Calibration", "dark")
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()


class FlatCalibrationModule(ProcessingModule):
    """
    Module to apply a flat field correction to the science data.
    """

    def __init__(self,
                 name_in="flat_calibration",
                 image_in_tag="dark_cal_arr",
                 flat_in_tag="flat_arr",
                 image_out_tag="flat_cal_arr"):
        """
        Constructor of FlatCalibrationModule.

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

        super(FlatCalibrationModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_flat_in_port = self.add_input_port(flat_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Creates a master flat with the same shape as the science
        image and divides the science images by the flat field.

        :return: None
        """

        def _flat_calibration(image_in, flat_in):
            return image_in / flat_in

        flat = self.m_flat_in_port.get_all()
        master = _master_frame(flat, self.m_image_in_port)

        # shift all values to greater or equal to +1.0
        flat_min = np.amin(master)
        master -= flat_min - 1.

        # normalization
        master /= np.median(master)

        if not np.allclose(np.median(master), 1., rtol=1e-6, atol=0.):
            raise ValueError("Median of the master flat should be equal to unity (value=%s)."
                             % np.median(master))

        self.apply_function_to_images(_flat_calibration,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running FlatCalibrationModule...",
                                      func_args=(master, ))

        self.m_image_out_port.add_history_information("Calibration", "flat")
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()
