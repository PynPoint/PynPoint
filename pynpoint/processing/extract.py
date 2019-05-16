"""
Pipeline modules for locating, aligning, and centering of the star.
"""

import math
import warnings

# from typing import Union, Tuple

import numpy as np

# from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import locate_star, rotate_coordinates
from pynpoint.util.image import crop_image, center_pixel


class StarExtractionModule(ProcessingModule):
    """
    Pipeline module to locate the position of the star in each image and to crop all the images
    around this position.
    """

    def __init__(self,
                 name_in='star_cutting',
                 image_in_tag='im_arr',
                 image_out_tag='im_arr_crop',
                 index_out_tag=None,
                 image_size=2.,
                 fwhm_star=0.2,
                 position=None):
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.
        index_out_tag : str
            List with image indices for which the image size is too large to be cropped around the
            brightest pixel. No data is written if set to None. This tag name can be provided to
            the *frames* parameter in
            :class:`~pynpoint.processing.frameselection.RemoveFramesModule`. This argument is
            ignored if ``CPU`` is set to a value larger than 1.
        image_size : float
            Cropped image size (arcsec).
        fwhm_star : float
            Full width at half maximum (arcsec) of the Gaussian kernel that is used to smooth the
            images to lower contributions of bad pixels.
        position : tuple(int, int, float)
            Subframe that is selected to search for the star. The tuple can contain a single
            position (pix) and size (arcsec) as (pos_x, pos_y, size), or the position and size can
            be defined for each image separately in which case the tuple should be 2D
            (nframes x 3). Setting *position* to None will use the full image to search for the
            star. If *position=(None, None, size)* then the center of the image will be used.

        Returns
        -------
        NoneType
            None
        """

        super(StarExtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if index_out_tag is None:
            self.m_index_out_port = None
        else:
            self.m_index_out_port = self.add_output_port(index_out_tag)

        self.m_image_size = image_size
        self.m_fwhm_star = fwhm_star
        self.m_position = position

        self.m_count = 0

    def run(self):
        """
        Run method of the module. Locates the position of the star (only pixel precision) by
        selecting the highest pixel value. A Gaussian kernel with a FWHM similar to the PSF is
        used to lower the contribution of bad pixels which may have higher values than the
        peak of the PSF. Images are cropped and written to an output port. The position of the
        star is attached to the input images as the non-static attribute STAR_POSITION (y, x).

        Returns
        -------
        NoneType
            None
        """

        cpu = self._m_config_port.get_attribute('CPU')

        if cpu > 1:
            self.m_index_out_port = None

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        nimages = self.m_image_in_port.get_shape()[0]

        if self.m_position is not None:
            self.m_position = np.asarray(self.m_position)

            if self.m_position.ndim == 2 and self.m_position.shape[0] != nimages:
                raise ValueError('Either a single \'position\' should be specified or an array '
                                 'equal in size to the number of images in \'image_in_tag\'.')

            if self.m_position.ndim == 2 and cpu > 1:
                raise ValueError('Multiprocessing is only implemented for a single position. '
                                 'Set CPU=1 in the configuration file or use a single '
                                 'value for the \'position\' argument.')

        self.m_image_size = int(math.ceil(self.m_image_size/pixscale))
        self.m_fwhm_star = int(math.ceil(self.m_fwhm_star/pixscale))

        star = []
        index = []

        def _crop_around_star(image, position, im_size, fwhm):

            if position is None:
                center = None
                width = None

            else:
                if position.ndim == 1:
                    if position[0] is None and position[1] is None:
                        center = None
                    else:
                        center = (int(position[1]), int(position[0]))

                    width = int(math.ceil(position[2]/pixscale))

                elif position.ndim == 2:
                    center = (int(position[self.m_count, 1]), int(position[self.m_count, 0]))
                    width = int(math.ceil(position[self.m_count, 2]/pixscale))

            starpos = locate_star(image, center, width, fwhm)

            try:
                im_crop = crop_image(image, starpos, im_size)

            except ValueError:
                if cpu == 1:
                    warnings.warn('PSF size is too large to crop the image around the brightest '
                                  'pixel (image index = '+str(self.m_count)+', pixel [x, y] = '
                                  +str([starpos[0]]+[starpos[1]])+'). Using the center of the '
                                  'image instead.')

                    index.append(self.m_count)

                else:
                    warnings.warn('PSF size is too large to crop the image around the brightest '
                                  'pixel. Using the center of the image instead.')

                starpos = center_pixel(image)
                im_crop = crop_image(image, starpos, im_size)

            if cpu == 1:
                star.append((starpos[1], starpos[0]))
                self.m_count += 1

            return im_crop

        self.apply_function_to_images(_crop_around_star,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Running StarExtractionModule',
                                      func_args=(self.m_position,
                                                 self.m_image_size,
                                                 self.m_fwhm_star))

        history = f'fwhm_star [pix] = {self.m_fwhm_star}'

        if self.m_index_out_port is not None:
            self.m_index_out_port.set_all(np.transpose(np.asarray(index)))
            self.m_index_out_port.copy_attributes(self.m_image_in_port)
            self.m_index_out_port.add_history('StarExtractionModule', history)

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('StarExtractionModule', history)

        if cpu == 1:
            self.m_image_out_port.add_attribute('STAR_POSITION', np.asarray(star), static=False)

        self.m_image_out_port.close_port()


class ExtractBinaryModule(ProcessingModule):
    """
    Pipeline module to extract a binary star (or another point source) which is rotating across the
    image stack.
    """

    def __init__(self,
                 pos_center,
                 pos_binary,
                 name_in='extract_binary',
                 image_in_tag='im_arr',
                 image_out_tag='im_arr_crop',
                 image_size=2.,
                 search_size=0.1):
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.
        pos_center : tuple(int, int)
            Approximate position (x, y) of the center of rotation (pix).
        pos_binary : tuple(int, int)
            Approximate position (x, y) of the binary star in the first image (pix).
        image_size : float
            Cropped image size (arcsec).
        search_size : float
            Window size (arcsec) in which the brightest pixel is selected as position of the binary
            star. The search window is centered on the position that for each image is calculated
            from the `pos_center`, `pos_binary`, and parallactic angle (`PARANG`) of the image.

        Returns
        -------
        NoneType
            None
        """

        super(ExtractBinaryModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_pos_center = (pos_center[1], pos_center[0]) # (y, x)
        self.m_pos_binary = (pos_binary[1], pos_binary[0]) # (y, x)

        self.m_image_size = image_size
        self.m_search_size = search_size

    def run(self):
        """
        Run method of the module. Locates the position of a binary star (or another point source)
        which rotates across the stack of images due to parallactic rotation. The approximate
        position of the binary star is calculated by taking into account the parallactic angle
        of each image separately. The brightest pixel is then selected as center around which
        the image is cropped.

        Returns
        -------
        NoneType
            None
        """

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        parang = self.m_image_in_port.get_attribute('PARANG')

        positions = np.zeros((parang.shape[0], 2), dtype=np.int)

        for i, item in enumerate(parang):
            # rotates in counterclockwise direction, hence the minus sign in angle
            positions[i, :] = rotate_coordinates(center=self.m_pos_center,
                                                 position=self.m_pos_binary,
                                                 angle=-item+parang[0])

        self.m_image_size = int(math.ceil(self.m_image_size/pixscale))
        self.m_search_size = math.ceil(self.m_search_size/pixscale)

        def _crop_rotating_star(image, position, im_size):

            starpos = locate_star(image=image,
                                  center=position,
                                  width=self.m_search_size,
                                  fwhm=None)

            return crop_image(image=image,
                              center=starpos,
                              size=im_size)

        self.apply_function_to_images(_crop_rotating_star,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Running ExtractBinaryModule',
                                      func_args=(positions,
                                                 self.m_image_size))

        history = f'binary (x, y) = ({self.m_pos_binary[0]}, {self.m_pos_binary[1]})'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('ExtractBinaryModule', history)
        self.m_image_out_port.close_port()
