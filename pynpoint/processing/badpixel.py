"""
Pipeline modules for the detection and interpolation of bad pixels.
"""

import warnings

from typing import Optional, Tuple

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.apply_func import bad_pixel_sigma_filter, image_interpolation, \
                                     replace_pixels, time_filter


class BadPixelSigmaFilterModule(ProcessingModule):
    """
    Pipeline module for finding bad pixels with a sigma filter and replacing them with the mean
    value of the surrounding pixels.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 map_out_tag: Optional[str] = None,
                 box: int = 9,
                 sigma: float = 5.,
                 iterate: int = 1) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        map_out_tag : str, None
            Tag of the database entry with the bad pixel map that is written as output. No data
            is written if set to None. This output port can not be used if CPU > 1.
        box : int
            Size of the sigma filter. The area of the filter is equal to the squared value of
            *box*.
        sigma : float
            Sigma threshold.
        iterate : int
            Number of iterations.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if map_out_tag is None:
            self.m_map_out_port = None
        else:
            self.m_map_out_port = self.add_output_port(map_out_tag)

        self.m_box = box
        self.m_sigma = sigma
        self.m_iterate = iterate

        if self.m_iterate < 1:
            raise ValueError('The argument of \'iterate\' should be 1 or larger.')

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Finds bad pixels with a sigma filter, replaces bad pixels with
        the mean value of the surrounding pixels, and writes the cleaned images to the database.

        Returns
        -------
        NoneType
            None
        """

        cpu = self._m_config_port.get_attribute('CPU')

        if cpu > 1 and self.m_map_out_port is not None:
            warnings.warn('The \'map_out_port\' can only be used if CPU = 1. No data will '
                          'be stored to this output port.')

            del self._m_output_ports[self.m_map_out_port.tag]
            self.m_map_out_port = None

        self.apply_function_to_images(bad_pixel_sigma_filter,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Bad pixel sigma filter',
                                      func_args=(self.m_box,
                                                 self.m_sigma,
                                                 self.m_iterate,
                                                 self.m_map_out_port))

        history = f'sigma = {self.m_sigma}'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('BadPixelSigmaFilterModule', history)

        if self.m_map_out_port is not None:
            self.m_map_out_port.copy_attributes(self.m_image_in_port)
            self.m_map_out_port.add_history('BadPixelSigmaFilterModule', history)

        self.m_image_out_port.close_port()


class BadPixelMapModule(ProcessingModule):
    """
    Pipeline module to create a bad pixel map from the dark frames and flat fields.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 dark_in_tag: Optional[str],
                 flat_in_tag: Optional[str],
                 bp_map_out_tag: str,
                 dark_threshold: float = 0.2,
                 flat_threshold: float = 0.2) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        dark_in_tag : str, None
            Tag of the database entry with the dark frames that are read as input. Not read if set
            to None.
        flat_in_tag : str, None
            Tag of the database entry with the flat fields that are read as input. Not read if set
            to None.
        bp_map_out_tag : str
            Tag of the database entry with the bad pixel map that is written as output.
        dark_threshold : float
            Fractional threshold with respect to the maximum pixel value in the dark frame to flag
            bad pixels. Pixels `brighter` than the fractional threshold are flagged as bad.
        flat_threshold : float
            Fractional threshold with respect to the maximum pixel value in the flat field to flag
            bad pixels. Pixels `fainter` than the fractional threshold are flagged as bad.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

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

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Collapses a cube of dark frames and flat fields if needed, flags
        bad pixels by comparing the pixel values with the threshold times the maximum value, and
        writes a bad pixel map to the database. For the dark frame, pixel values larger than the
        threshold will be flagged while for the flat frame pixel values smaller than the threshold
        will be flagged.

        Returns
        -------
        NoneType
            None
        """

        if self.m_dark_port is not None:
            dark = self.m_dark_port.get_all()

            if dark.ndim == 3:
                dark = np.mean(dark, axis=0)

            max_dark = np.max(dark)

            print(f'Threshold dark frame = {max_dark*self.m_dark_threshold}')

            bpmap = np.ones(dark.shape)
            bpmap[np.where(dark > max_dark*self.m_dark_threshold)] = 0

        if self.m_flat_port is not None:
            flat = self.m_flat_port.get_all()

            if flat.ndim == 3:
                flat = np.mean(flat, axis=0)

            max_flat = np.max(flat)

            print(f'Threshold flat field (ADU) = {max_flat*self.m_flat_threshold:.2e}')

            if self.m_dark_port is None:
                bpmap = np.ones(flat.shape)

            bpmap[np.where(flat < max_flat*self.m_flat_threshold)] = 0

        if self.m_dark_port is not None and self.m_flat_port is not None:
            if not dark.shape == flat.shape:
                raise ValueError('Dark and flat images should have the same shape.')

        self.m_bp_map_out_port.set_all(bpmap, data_dim=3)

        if self.m_dark_port is not None:
            self.m_bp_map_out_port.copy_attributes(self.m_dark_port)
        elif self.m_flat_port is not None:
            self.m_bp_map_out_port.copy_attributes(self.m_flat_port)

        history = f'dark = {self.m_dark_threshold}, flat = {self.m_flat_threshold}'
        self.m_bp_map_out_port.add_history('BadPixelMapModule', history)

        self.m_bp_map_out_port.close_port()


class BadPixelInterpolationModule(ProcessingModule):
    """
    Pipeline module to interpolate bad pixels with spectral deconvolution.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 bad_pixel_map_tag: str,
                 image_out_tag: str,
                 iterations: int = 1000) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the images that are read as input.
        bad_pixel_map_tag : str
            Tag of the database entry with the bad pixel map that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        iterations : int
            Number of iterations of the spectral deconvolution.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_bp_map_in_port = self.add_input_port(bad_pixel_map_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_iterations = iterations

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Interpolates bad pixels with an iterative spectral deconvolution.

        Returns
        -------
        NoneType
            None
        """

        bad_pixel_map = self.m_bp_map_in_port.get_all()[0, ]
        im_shape = self.m_image_in_port.get_shape()

        if self.m_iterations > im_shape[1]*im_shape[2]:
            raise ValueError('Maximum number of iterations needs to be smaller than the number of '
                             'pixels in the image.')

        if bad_pixel_map.shape[0] != im_shape[-2] or bad_pixel_map.shape[1] != im_shape[-1]:
            raise ValueError('The shape of the bad pixel map does not match the shape of the '
                             'images.')

        self.apply_function_to_images(image_interpolation,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Bad pixel interpolation',
                                      func_args=(self.m_iterations,
                                                 bad_pixel_map))

        history = f'iterations = {self.m_iterations}'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('BadPixelInterpolationModule', history)
        self.m_image_out_port.close_port()


class BadPixelTimeFilterModule(ProcessingModule):
    """
    Pipeline module for finding bad pixels with a sigma filter along a pixel line in time. This
    module is suitable for removing bad pixels that are only present at a position in a small
    number of images, for example because a dither pattern has been applied. Pixel lines can be
    processed in parallel by setting the CPU keyword in the configuration file.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 sigma: Tuple[float, float] = (5., 5.)) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        sigma : tuple(float, float)
            Lower and upper sigma threshold as (lower, upper).

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_sigma = sigma

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Finds bad pixels along a pixel line, replaces the bad pixels with
        the mean value of the pixels (excluding the bad pixels), and writes the cleaned images to
        the database.

        Returns
        -------
        NoneType
            None
        """

        print('Temporal filtering of bad pixels ...', end='')

        self.apply_function_in_time(time_filter,
                                    self.m_image_in_port,
                                    self.m_image_out_port,
                                    func_args=(self.m_sigma, ))

        print(' [DONE]')

        history = f'sigma = {self.m_sigma}'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('BadPixelTimeFilterModule', history)
        self.m_image_out_port.close_port()


class ReplaceBadPixelsModule(ProcessingModule):
    """
    Pipeline module for replacing bad pixels with the mean are median value of the surrounding
    pixels. The bad pixels are selected from the input bad pixel map.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 map_in_tag: str,
                 image_out_tag: str,
                 size: int = 2,
                 replace: str = 'median') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        size : int
            Number of pixel lines around the bad pixel that are used to calculate the median or mean
            replacement value. For example, a 5x5 window is used if ``size=2``.
        replace : str
            Replace the bad pixel with the 'median', 'mean' or 'nan'.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_map_in_port = self.add_input_port(map_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_size = size
        self.m_replace = replace

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Masks the bad pixels with NaN and replaces the bad pixels with the
        mean or median value (excluding the bad pixels) within a window centered on the bad pixel.
        The original value is used if there are only NaNs within the window.

        Returns
        -------
        NoneType
            None
        """

        bpmap = self.m_map_in_port.get_all()[0, ]
        index = np.argwhere(bpmap == 0)

        self.apply_function_to_images(replace_pixels,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Running ReplaceBadPixelsModule',
                                      func_args=(index,
                                                 self.m_size,
                                                 self.m_replace))

        history = f'replace = {self.m_replace}'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('ReplaceBadPixelsModule', history)
        self.m_image_out_port.close_port()
