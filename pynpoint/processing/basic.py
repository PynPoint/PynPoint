"""
Pipeline modules for basic image operations.
"""

import time

from typing import Tuple

from typeguard import typechecked
from scipy.ndimage import rotate

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress, memory_frames


class SubtractImagesModule(ProcessingModule):
    """
    Pipeline module for subtracting two sets of images.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tags: Tuple[str, str],
                 image_out_tag: str,
                 scaling: float = 1.) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tags : tuple(str, str)
            Tuple with two tags of the database entry that are read as input.
        image_out_tag : str
            Tag of the database entry with the subtracted images that are written as output.
        scaling : float
            Additional scaling factor.

        Returns
        -------
        NoneType
            None
        """

        super(SubtractImagesModule, self).__init__(name_in)

        self.m_image_in1_port = self.add_input_port(image_in_tags[0])
        self.m_image_in2_port = self.add_input_port(image_in_tags[1])
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_scaling = scaling

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Subtracts the images from the second database tag from the images
        of the first database tag, on a frame-by-frame basis.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        if self.m_image_in1_port.get_shape() != self.m_image_in2_port.get_shape():
            raise ValueError('The shape of the two input tags has to be the same.')

        memory = self._m_config_port.get_attribute('MEMORY')
        nimages = self.m_image_in1_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        start_time = time.time()

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), 'Subtracting images...', start_time)

            images1 = self.m_image_in1_port[frames[i]:frames[i+1], ]
            images2 = self.m_image_in2_port[frames[i]:frames[i+1], ]

            self.m_image_out_port.append(self.m_scaling*(images1-images2), data_dim=3)

        history = f'scaling = {self.m_scaling}'
        self.m_image_out_port.add_history('SubtractImagesModule', history)
        self.m_image_out_port.copy_attributes(self.m_image_in1_port)
        self.m_image_out_port.close_port()


class AddImagesModule(ProcessingModule):
    """
    Pipeline module for adding two sets of images.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tags: Tuple[str, str],
                 image_out_tag: str,
                 scaling: float = 1.) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tags : tuple(str, str)
            Tuple with two tags of the database entry that are read as input.
        image_out_tag : str
            Tag of the database entry with the added images that are written as output.
        scaling: float
            Additional scaling factor.

        Returns
        -------
        NoneType
            None
        """

        super(AddImagesModule, self).__init__(name_in)

        self.m_image_in1_port = self.add_input_port(image_in_tags[0])
        self.m_image_in2_port = self.add_input_port(image_in_tags[1])
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_scaling = scaling

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Add the images from the two database tags on a frame-by-frame
        basis.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        if self.m_image_in1_port.get_shape() != self.m_image_in2_port.get_shape():
            raise ValueError('The shape of the two input tags has to be the same.')

        memory = self._m_config_port.get_attribute('MEMORY')
        nimages = self.m_image_in1_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        start_time = time.time()

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), 'Adding images...', start_time)

            images1 = self.m_image_in1_port[frames[i]:frames[i+1], ]
            images2 = self.m_image_in2_port[frames[i]:frames[i+1], ]

            self.m_image_out_port.append(self.m_scaling*(images1+images2), data_dim=3)

        history = f'scaling = {self.m_scaling}'
        self.m_image_out_port.add_history('AddImagesModule', history)
        self.m_image_out_port.copy_attributes(self.m_image_in1_port)
        self.m_image_out_port.close_port()


class RotateImagesModule(ProcessingModule):
    """
    Pipeline module for rotating images.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 angle: float) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        angle : float
            Rotation angle (deg). Rotation is clockwise for positive values.
        Returns
        -------
        NoneType
            None
        """

        super(RotateImagesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_angle = angle

    def run(self) -> None:
        """
        Run method of the module. Rotates all images by a constant angle.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        memory = self._m_config_port.get_attribute('MEMORY')
        nimages = self.m_image_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        start_time = time.time()

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), 'Rotating images...', start_time)

            images = self.m_image_in_port[frames[i]:frames[i+1], ]

            for j in range(frames[i+1]-frames[i]):
                im_tmp = images[j, ]

                # ndimage.rotate rotates in clockwise direction for positive angles
                im_tmp = rotate(im_tmp, self.m_angle, reshape=False)

                self.m_image_out_port.append(im_tmp, data_dim=3)

        history = f'angle [deg] = {self.m_angle}'
        self.m_image_out_port.add_history('RotateImagesModule', history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()


class RepeatImagesModule(ProcessingModule):
    """
    Pipeline module for repeating the images from a dataset.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 repeat: int) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry with the added images that are written as output.
        repeat: int
            The number of times the input images get repeated.

        Returns
        -------
        NoneType
            None
        """

        super(RepeatImagesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_repeat = repeat

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Repeats the stack of input images a specified number of times.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        nimages = self.m_image_in_port.get_shape()[0]
        memory = self._m_config_port.get_attribute('MEMORY')

        frames = memory_frames(memory, nimages)

        start_time = time.time()

        for i in range(self.m_repeat):
            progress(i, self.m_repeat, 'Repeating images...', start_time)

            for j, _ in enumerate(frames[:-1]):
                images = self.m_image_in_port[frames[j]:frames[j+1], ]
                self.m_image_out_port.append(images, data_dim=3)

        history = f'repeat = {self.m_repeat}'
        self.m_image_out_port.add_history('RepeatImagesModule', history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()
