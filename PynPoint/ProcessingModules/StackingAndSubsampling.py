"""
Modules for stacking and subsampling of images.
"""

import sys

import numpy as np

from scipy.ndimage import rotate

from PynPoint.Util.Progress import progress
from PynPoint.Core.Processing import ProcessingModule


class StackAndSubsetModule(ProcessingModule):
    """
    Module for stacking subsets of images and/or selecting a random sample of images.
    """

    def __init__(self,
                 name_in="stacking_subset",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr",
                 random=None,
                 stacking=None):
        """
        Constructor of StackAndSubsetModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str
        :param random: Number of random frames.
        :type random: int
        :param stacking: Number of stacked images per subset.
        :type stacking: int

        :return: None
        """

        super(StackAndSubsetModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_random = random
        self.m_stacking = stacking

    def run(self):
        """
        Run method of the module. Stacks subsets of images and/or selects a random subset.

        :return: None
        """

        if self.m_stacking in (None, False) and self.m_subset in (None, False):
            return

        tmp_data_shape = self.m_image_in_port.get_shape()

        if self.m_stacking is None and tmp_data_shape[0] < self.m_random:
            raise ValueError("The number of images of the destination subset is larger than the "
                             "number of images in the source.")

        elif self.m_stacking is not None and \
                     int(float(tmp_data_shape[0])/float(self.m_stacking)) < self.m_random:
            raise ValueError("The number of images of the destination subset is larger than the "
                             "number of images in the stacked source.")

        parang = self.m_image_in_port.get_attribute("PARANG")

        if self.m_stacking not in (False, None):
            # Stack subsets of frames
            num_new = int(np.floor(float(tmp_data_shape[0])/float(self.m_stacking)))
            tmp_parang = np.zeros(num_new)
            tmp_data = np.zeros([num_new, tmp_data_shape[1], tmp_data_shape[2]])

            for i in range(num_new):
                progress(i, num_new, "Running StackAndSubsetModule...")

                tmp_parang[i] = np.mean(parang[i*self.m_stacking:(i+1)*self.m_stacking])
                tmp_data[i, ] = np.mean(self.m_image_in_port[i*self.m_stacking: \
                                        (i+1)*self.m_stacking, ],
                                        axis=0)

            tmp_data_shape = tmp_data.shape

        else:
            tmp_parang = np.copy(parang)

        sys.stdout.write("Running StackAndSubsetModule... [DONE]\n")
        sys.stdout.flush()

        if self.m_random is not None:
            tmp_choice = np.random.choice(tmp_data_shape[0], self.m_random, replace=False)
            tmp_choice = np.sort(tmp_choice)
            tmp_parang = tmp_parang[tmp_choice]

            if self.m_stacking is None:
                # This will cause memory problems for large values of random
                tmp_data = self.m_image_in_port[tmp_choice, :, :]
            else:
                # Possibly also here depending on the stacking value
                tmp_data = tmp_data[tmp_choice, ]

        self.m_image_out_port.set_all(tmp_data, keep_attributes=True)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_attribute("PARANG", tmp_parang, static=False)

        if self.m_stacking is None:
            history_stacked = "Nothing Stacked and "
        else:
            history_stacked = "Stacked every " + str(self.m_stacking) + " and "

        if self.m_random is None:
            history_subset = "no subset chosen"
        else:
            history_subset = "picked randomly " + str(self.m_random) + " frames."

        history = history_stacked + history_subset
        self.m_image_out_port.add_history_information("Subset", history)
        self.m_image_out_port.close_database()


class MeanCubeModule(ProcessingModule):
    """
    Module for calculating the mean of each individual cube associated with a database tag.
    """

    def __init__(self,
                 name_in="mean_cube",
                 image_in_tag="im_arr",
                 image_out_tag="im_mean"):
        """
        Constructor of MeanCubeModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry with the mean collapsed images that are
                              written as output. Should be different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(MeanCubeModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Uses the NFRAMES attribute to select the images of each cube,
        calculates the mean of each cube, and saves the data and attributes.

        :return: None
        """

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        nframes = self.m_image_in_port.get_attribute("NFRAMES")

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        current = 0

        for i, frames in enumerate(nframes):
            progress(i, len(nframes), "Running MeanCubeModule...")

            mean_frame = np.mean(self.m_image_in_port[current:current+frames, ],
                                 axis=0)

            self.m_image_out_port.append(mean_frame,
                                         data_dim=3)

            current += frames

        sys.stdout.write("Running MeanCubeModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_database()


class DerotateAndStackModule(ProcessingModule):
    """
    Module for derotating the images and optional stacking.
    """

    def __init__(self,
                 name_in="rotate_stack",
                 image_in_tag="im_arr",
                 image_out_tag="im_stack",
                 stack=False,
                 extra_rot=0.):
        """
        Constructor of DerotateAndStackModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. The output is
                              either 2D (*stack=False*) or 3D (*stack=True*).
        :type image_out_tag: str
        :param stack: Apply a mean stacking after derotation.
        :type stack: bool
        :param extra_rot: Additional rotation angle of the images in clockwise direction (deg).
        :type extra_rot: float

        :return: None
        """

        super(DerotateAndStackModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_extra_rot = extra_rot
        self.m_stack = stack

    def run(self):
        """
        Run method of the module. Uses the PARANG attributes to derotate the images and applies
        an optional mean stacking afterwards.

        :return: None
        """

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        parang = self.m_image_in_port.get_attribute("PARANG")

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_stack:
            stack = np.zeros((self.m_image_in_port.get_shape()[1],
                              self.m_image_in_port.get_shape()[2]))

        elif not self.m_stack:
            stack = np.zeros(self.m_image_in_port.get_shape())

        count = 0.
        for i, ang in enumerate(parang):
            progress(i, len(parang), "Running RotateAndStackModule...")

            im_rot = rotate(self.m_image_in_port[i, ], -ang+self.m_extra_rot, reshape=False)

            if self.m_stack:
                stack += im_rot
            elif not self.m_stack:
                stack[i, ] = im_rot

            count += 1.

        if self.m_stack:
            stack /= count

        sys.stdout.write("Running RotateAndStackModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.set_all(stack)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_database()
