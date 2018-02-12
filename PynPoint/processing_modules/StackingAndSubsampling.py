"""
Modules for stacking and subsampling.
"""

import sys

import numpy as np

from PynPoint.util.Progress import progress
from PynPoint.core.Processing import ProcessingModule


class StackAndSubsetModule(ProcessingModule):
    """
    Module for stacking subsets of images and/or selecting a random sample of images.
    """

    def __init__(self,
                 name_in="stacking_subset",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr",
                 random_subset=None,
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
        :param random_subset: Number of random frames.
        :type random_subset: int
        :param stacking: Number of stacked images per subset.
        :type stacking: int
        :return: None
        """

        super(StackAndSubsetModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_subset = random_subset
        self.m_stacking = stacking

    def run(self):
        """
        Run method of the module. Stacks subsets of images and/or selects a random subset.

        :return: None
        """

        if self.m_stacking in (None, False) and self.m_subset in (None, False):
            return

        tmp_data_shape = self.m_image_in_port.get_shape()

        if self.m_stacking is None and tmp_data_shape[0] < self.m_subset:
            raise ValueError("The number of images of the destination subset is bigger than the "
                             "number of images in the source.")

        elif self.m_stacking is not None and \
                     int(float(tmp_data_shape[0])/float(self.m_stacking)) < self.m_subset:
            raise ValueError("The number of images of the destination subset is bigger than the "
                             "number of images in the stacked source.")

        tmp_files = self.m_image_in_port.get_attribute("Used_Files")
        num_images = self.m_image_in_port.get_attribute("NFRAMES")
        para_angles = self.m_image_in_port.get_attribute("NEW_PARA")

        if tmp_files is None:
            raise ValueError("No files are listed in Used_Files.")

        if num_images is None:
            raise ValueError("No images are present, NAXIS3 is empty.")

        if self.m_stacking not in (False, None):
            # Stack subsets of frames
            num_new = int(np.floor(float(tmp_data_shape[0])/float(self.m_stacking)))
            tmp_parang = np.zeros(num_new)
            tmp_data = np.zeros([num_new, tmp_data_shape[1], tmp_data_shape[2]])

            for i in range(num_new):
                progress(i, num_new, "Running StackAndSubsetModule...")

                tmp_parang[i] = np.mean(para_angles[i*self.m_stacking:(i+1)*self.m_stacking])
                tmp_data[i, ] = np.mean(self.m_image_in_port[i*self.m_stacking: \
                                        (i+1)*self.m_stacking, ],
                                        axis=0)

            tmp_data_shape = tmp_data.shape

        else:
            tmp_parang = np.copy(para_angles)

        sys.stdout.write("Running StackAndSubsetModule... [DONE]\n")
        sys.stdout.flush()

        if self.m_subset not in (None, False):
            # Random selection of frames
            tmp_choice = np.random.choice(tmp_data_shape[0],
                                          self.m_subset,
                                          replace=False)

            tmp_choice = np.sort(tmp_choice)
            tmp_parang = tmp_parang[tmp_choice]

            if self.m_stacking is None:
                # This will cause memory problems for large values of random_subset
                tmp_data = self.m_image_in_port[tmp_choice, :, :]
            else:
                # Possibly also here depending on the stacking value
                tmp_data = tmp_data[tmp_choice, ]

            # Check which files are used
            frames_cumulative = np.zeros(np.size(num_images))
            for i, item in enumerate(num_images):
                if i == 0:
                    frames_cumulative[i] = item
                else:
                    frames_cumulative[i] = frames_cumulative[i-1] + item

            files_out = []
            for i, item in enumerate(frames_cumulative):
                if self.m_stacking is None:
                    if i == 0:
                        index_check = np.logical_and(tmp_choice >= 0,
                                                     tmp_choice < frames_cumulative[i])
                    else:
                        index_check = np.logical_and(tmp_choice >= frames_cumulative[i-1],
                                                     tmp_choice < frames_cumulative[i])

                elif self.m_stacking is not None:
                    # Only the first frame of a stacked subset is considered for Used_Files
                    if i == 0:
                        index_check = np.logical_and(tmp_choice*self.m_stacking >= 0, \
                                      tmp_choice*self.m_stacking < frames_cumulative[i])

                    else:
                        index_check = np.logical_and(tmp_choice*self.m_stacking >= \
                                      frames_cumulative[i-1], tmp_choice*self.m_stacking < \
                                      frames_cumulative[i])

                if True in index_check:
                    files_out.append(tmp_files[i])

            tmp_files = files_out

        self.m_image_out_port.set_all(tmp_data,
                                      keep_attributes=True)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_attribute("NEW_PARA",
                                            tmp_parang,
                                            static=False)

        self.m_image_out_port.add_attribute("Used_Files",
                                            tmp_files,
                                            static=False)

        self.m_image_out_port.add_attribute("Num_Files",
                                            len(tmp_files),
                                            static=True)

        if self.m_stacking is None:
            history_stacked = "Nothing Stacked and "
        else:
            history_stacked = "Stacked every " + str(self.m_stacking) + " and "

        if self.m_subset is None:
            history_subset = "no subset chosen"
        else:
            history_subset = "picked randomly " + str(self.m_subset) + " frames."

        history = history_stacked + history_subset
        self.m_image_out_port.add_history_information("Subset",
                                                      history)

        self.m_image_out_port.close_port()
