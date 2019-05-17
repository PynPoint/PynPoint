"""
Pipeline modules for dark frame and flat field calibrations.
"""

import warnings

import numpy as np

from pynpoint.core.processing import ProcessingModule


def _master_frame(data,
                  im_shape):
    """
    Internal function which creates a master dark/flat by calculating the mean (3D data) and
    cropping the frames to the shape of the science images if needed.

    Parameters
    ----------
    data : numpy.ndarray
        Input array (2D) with mean of the dark or flat frames.
    im_shape : numpy.ndarray
        Shape of the science images (3D).

    Returns
    -------
    numpy.ndarray
        Master dark/flat frame.
    """

    shape_in = (im_shape[1], im_shape[2])

    if data.shape[0] < shape_in[0] or data.shape[1] < shape_in[1]:
        raise ValueError('Shape of the calibration images is smaller than the science images.')

    if data.shape != shape_in:
        cal_shape = data.shape

        x_off = (cal_shape[0] - shape_in[0]) // 2
        y_off = (cal_shape[1] - shape_in[1]) // 2

        data = data[x_off:x_off+shape_in[0], y_off:y_off+shape_in[1]]

        warnings.warn('The calibration images were cropped around their center to match the shape '
                      'of the science images.')

    return data


class DarkCalibrationModule(ProcessingModule):
    """
    Pipeline module to subtract a master dark from the science data.
    """

    def __init__(self,
                 name_in='dark_calibration',
                 image_in_tag='im_arr',
                 dark_in_tag='dark_arr',
                 image_out_tag='dark_cal_arr'):
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images that are read as input.
        dark_in_tag : str
            Tag of the database with the dark frames that are read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.

        Returns
        -------
        NoneType
            None
        """

        super(DarkCalibrationModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_dark_in_port = self.add_input_port(dark_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Creates a master dark with the same shape as the science
        data and subtracts the dark frame from the science data.

        Returns
        -------
        NoneType
            None
        """

        def _dark_calibration(image_in, dark_in):
            return image_in - dark_in

        dark = self.m_dark_in_port.get_all()

        master = _master_frame(data=np.mean(dark, axis=0),
                               im_shape=self.m_image_in_port.get_shape())

        self.apply_function_to_images(_dark_calibration,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Running DarkCalibrationModule',
                                      func_args=(master, ))

        history = f'dark_in_tag = {self.m_dark_in_port.tag}'
        self.m_image_out_port.add_history('DarkCalibrationModule', history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()


class FlatCalibrationModule(ProcessingModule):
    """
    Pipeline module to apply a flat field correction to the science data.
    """

    def __init__(self,
                 name_in='flat_calibration',
                 image_in_tag='dark_cal_arr',
                 flat_in_tag='flat_arr',
                 image_out_tag='flat_cal_arr'):
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the science database that is read as input.
        dark_in_tag : str
            Tag of the flat field database that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.

        Returns
        -------
        NoneType
            None
        """

        super(FlatCalibrationModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_flat_in_port = self.add_input_port(flat_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Creates a master flat with the same shape as the science
        image and divides the science images by the flat field.

        Returns
        -------
        NoneType
            None
        """

        def _flat_calibration(image_in, flat_in):
            return image_in / flat_in

        flat = self.m_flat_in_port.get_all()

        master = _master_frame(data=np.mean(flat, axis=0),
                               im_shape=self.m_image_in_port.get_shape())

        # shift all values to greater or equal to +1.0
        flat_min = np.amin(master)
        master -= flat_min - 1.

        # normalization, median value is 1 afterwards
        master /= np.median(master)

        self.apply_function_to_images(_flat_calibration,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Running FlatCalibrationModule',
                                      func_args=(master, ))

        history = f'flat_in_tag = {self.m_flat_in_port.tag}'
        self.m_image_out_port.add_history('FlatCalibrationModule', history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()
