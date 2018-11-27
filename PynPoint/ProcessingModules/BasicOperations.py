"""
Modules for basic image operations.
"""

from __future__ import absolute_import

import sys

from scipy.ndimage import rotate
from six.moves import range

from PynPoint.Core.Processing import ProcessingModule
from PynPoint.Util.ModuleTools import progress, memory_frames, number_images_port


class SubtractImagesModule(ProcessingModule):
    """
    Module for subtracting two sets of images.
    """

    def __init__(self,
                 image_in_tags,
                 name_in="subtract_images",
                 image_out_tag="im_arr_subtract",
                 scaling=1.):
        """
        Constructor of SubtractImagesModule.

        :param image_in_tags: Tuple with two tags of the database entry that are read as input.
        :type image_in_tags: (str, str)
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_out_tag: Tag of the database entry with the subtracted images that are written
                              as output. Should be different from *image_in_tags*.
        :type image_out_tag: str
        :param scaling: Additional scaling factor.
        :type scaling: float

        :return: None
        """

        super(SubtractImagesModule, self).__init__(name_in=name_in)

        self.m_image_in1_port = self.add_input_port(image_in_tags[0])
        self.m_image_in2_port = self.add_input_port(image_in_tags[1])
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_scaling = scaling

    def run(self):
        """
        Run method of the module. Subtracts the images from the second database tag from the images
        of the first database tag, on a frame-by-frame basis.

        :return: None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        if self.m_image_in1_port.get_shape() != self.m_image_in2_port.get_shape():
            raise ValueError("The shape of the two input tags have to be equal.")

        memory = self._m_config_port.get_attribute("MEMORY")
        nimages = self.m_image_in1_port.get_shape()[0]

        frames = memory_frames(memory, nimages)

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Running SubtractImagesModule...")

            images1 = self.m_image_in1_port[frames[i]:frames[i+1], ]
            images2 = self.m_image_in2_port[frames[i]:frames[i+1], ]

            self.m_image_out_port.append(self.m_scaling*(images1-images2))

        sys.stdout.write("Running SubtractImagesModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.add_history_information("Images subtracted", "")
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in1_port)
        self.m_image_out_port.close_port()


class AddImagesModule(ProcessingModule):
    """
    Module for adding two sets of images.
    """

    def __init__(self,
                 image_in_tags,
                 name_in="add_images",
                 image_out_tag="im_arr_add",
                 scaling=1.):
        """
        Constructor of AddImagesModule.

        :param image_in_tags: Tuple with two tags of the database entry that are read as input.
        :type image_in_tags: (str, str)
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_out_tag: Tag of the database entry with the added images that are written as
                              output. Should be different from *image_in_tags*.
        :type image_out_tag: str
        :param scaling: Additional scaling factor.
        :type scaling: float

        :return: None
        """

        super(AddImagesModule, self).__init__(name_in=name_in)

        self.m_image_in1_port = self.add_input_port(image_in_tags[0])
        self.m_image_in2_port = self.add_input_port(image_in_tags[1])
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_scaling = scaling

    def run(self):
        """
        Run method of the module. Add the images from the two database tags on a frame-by-frame
        basis.

        :return: None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        if self.m_image_in1_port.get_shape() != self.m_image_in2_port.get_shape():
            raise ValueError("The shape of the two input tags have to be equal.")

        nimages = self.m_image_in1_port.get_shape()[0]
        memory = self._m_config_port.get_attribute("MEMORY")

        frames = memory_frames(memory, nimages)

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Running AddImagesModule...")

            images1 = self.m_image_in1_port[frames[i]:frames[i+1], ]
            images2 = self.m_image_in2_port[frames[i]:frames[i+1], ]

            self.m_image_out_port.append(self.m_scaling*(images1+images2))

        sys.stdout.write("Running AddImagesModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.add_history_information("Images added", "")
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in1_port)
        self.m_image_out_port.close_port()


class RotateImagesModule(ProcessingModule):
    """
    Module for rotating images.
    """

    def __init__(self,
                 angle,
                 name_in="rotate_images",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_rot"):
        """
        Constructor of RotateImagesModule.

        :param scaling: Rotation angle (deg). Rotation is clockwise for positive values.
        :type scaling: float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tags*.
        :type image_out_tag: str

        :return: None
        """

        super(RotateImagesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_angle = angle

    def run(self):
        """
        Run method of the module. Rotates all images by a constant angle.

        :return: None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        memory = self._m_config_port.get_attribute("MEMORY")
        ndim = self.m_image_in_port.get_ndim()
        nimages = number_images_port(self.m_image_in_port)
        frames = memory_frames(memory, nimages)

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Running RotateImagesModule...")

            if nimages == 1:
                images = self.m_image_in_port.get_all()
            else:
                images = self.m_image_in_port[frames[i]:frames[i+1], ]

            for j in range(frames[i+1]-frames[i]):

                if nimages == 1:
                    im_tmp = images
                else:
                    im_tmp = images[j, ]

                # ndimage.rotate rotates in clockwise direction for positive angles
                im_tmp = rotate(im_tmp, self.m_angle, reshape=False)

                self.m_image_out_port.append(im_tmp, data_dim=ndim)

        sys.stdout.write("Running RotateImagesModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.add_history_information("Images rotated", self.m_angle)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()
