"""
Pipeline modules for spatial filtering of images.
"""

import math
import time

from typeguard import typechecked

from scipy.ndimage import gaussian_filter

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import memory_frames, progress


class GaussianFilterModule(ProcessingModule):
    """
    Pipeline module for applying a Gaussian filter.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 fwhm: float = 1.) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tags : tuple(str, str)
            Tuple with two tags of the database entry that are read as input.
        image_out_tag : str
            Tag of the database entry with the subtracted images that are written as output.
        fwhm : float
            Full width at half maximum (arcsec) of the Gaussian kernel.

        Returns
        -------
        NoneType
            None
        """

        super(GaussianFilterModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_fwhm = fwhm

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Applies a Gaussian filter to the spatial dimensions of the
        images.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        memory = self._m_config_port.get_attribute('MEMORY')
        pixscale = self._m_config_port.get_attribute('PIXSCALE')

        nimages = self.m_image_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        sigma = (self.m_fwhm/pixscale) / (2.*math.sqrt(2.*math.log(2.)))  # [pix]

        start_time = time.time()

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), 'Applying Gaussian filter...', start_time)

            images = self.m_image_in_port[frames[i]:frames[i+1], ]
            im_filter = gaussian_filter(images, (0, sigma, sigma))

            self.m_image_out_port.append(im_filter, data_dim=3)

        history = f'fwhm [arcsec] = {self.m_fwhm}'
        self.m_image_out_port.add_history('GaussianFilterModule', history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()
