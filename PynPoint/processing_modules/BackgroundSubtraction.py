"""
Modules with background subtraction routines.
"""

import numpy as np
import warnings
import sys
from PynPoint.core.Processing import ProcessingModule


class MeanBackgroundSubtractionModule(ProcessingModule):

    def __init__(self,
                 star_pos_shift,
                 name_in="mean_background_subtraction",
                 image_in_tag="im_arr",
                 image_out_tag="bg_cleaned_arr"):

        super(MeanBackgroundSubtractionModule, self).__init__(name_in)

        # add Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_star_prs_shift = star_pos_shift

    def run(self):

        number_of_frames = self.m_image_in_port.get_shape()[0]

        # Check size of the input
        if number_of_frames < self.m_star_prs_shift*2.0:
            raise ValueError("The input stack is to small for mean background subtraction. At least"
                             "one star position shift is needed.")

        # first subtraction is used to set up the output port array
        # calc mean
        tmp_data = self.m_image_in_port[self.m_star_prs_shift: self.m_star_prs_shift*2, :, :]
        tmp_mean = np.mean(tmp_data, axis=0)

        # init result port data
        tmp_res = self.m_image_in_port[0, :, :] - tmp_mean

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise NotImplementedError("Same input and output port not implemented yet.")
        else:
            self.m_image_out_port.set_all(tmp_res, data_dim=3)

        # clean first stack
        tmp_data = self.m_image_in_port[1:self.m_star_prs_shift, :, :]
        tmp_data = tmp_data - tmp_mean
        self.m_image_out_port.append(tmp_data)  # TODO This will not work for same in and out port

        # the last and the one before will be performed afterwards
        top = int(np.ceil(number_of_frames /
                          self.m_star_prs_shift)) - 2

        # process the rest of the stack
        for i in range(1, top, 1):
            print "Subtracting background from stack-part " + str(i) + " of " + \
                  str(int(np.floor(number_of_frames/self.m_star_prs_shift))) + " stack-parts"
            # calc the mean (next)
            tmp_data = self.m_image_in_port[(i+1) * self.m_star_prs_shift:
                                            (i+2) * self.m_star_prs_shift,
                                            :, :]
            tmp_mean = np.mean(tmp_data, axis=0)
            # calc the mean (previous)
            tmp_data = self.m_image_in_port[(i-1) * self.m_star_prs_shift:
                                            (i+0) * self.m_star_prs_shift, :, :]
            tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

            # subtract mean
            tmp_data = self.m_image_in_port[(i+0) * self.m_star_prs_shift:
                                            (i+1) * self.m_star_prs_shift, :, :]
            tmp_data = tmp_data - tmp_mean
            self.m_image_out_port.append(tmp_data)

        # last and the one before
        # 1. ------------------------------- one before -------------------
        # calc the mean (previous)
        tmp_data = self.m_image_in_port[(top - 1) * self.m_star_prs_shift:
                                        (top + 0) * self.m_star_prs_shift, :, :]
        tmp_mean = np.mean(tmp_data, axis=0)
        # calc the mean (next)
        # "number_of_frames" is important if the last step is to huge
        tmp_data = self.m_image_in_port[(top + 1) * self.m_star_prs_shift:
                                        number_of_frames, :, :]

        tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

        # subtract mean
        tmp_data = self.m_image_in_port[top * self.m_star_prs_shift:
                                        (top + 1) * self.m_star_prs_shift, :, :]
        tmp_data = tmp_data - tmp_mean
        self.m_image_out_port.append(tmp_data)

        # 2. ------------------------------- last -------------------
        # calc the mean (previous)
        tmp_data = self.m_image_in_port[(top + 0) * self.m_star_prs_shift:
                                        (top + 1) * self.m_star_prs_shift, :, :]
        tmp_mean = np.mean(tmp_data, axis=0)

        # subtract mean
        tmp_data = self.m_image_in_port[(top + 1) * self.m_star_prs_shift:
                                        number_of_frames, :, :]
        tmp_data = tmp_data - tmp_mean
        self.m_image_out_port.append(tmp_data)
        # -----------------------------------------------------------

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_history_information("Background",
                                                      "mean subtraction")

        self.m_image_out_port.close_port()


class SimpleBackgroundSubtractionModule(ProcessingModule):
    """
    Module for simple background subtraction, only applicable for dithered data.
    """

    def __init__(self,
                 star_pos_shift,
                 name_in="background_subtraction",
                 image_in_tag="im_arr",
                 image_out_tag="bg_cleaned_arr"):
        """
        Constructor of SimpleBackgroundSubtractionModule.

        :param star_pos_shift: Frame number offset for the background used for subtraction.
                               Typically equal to the number of frames per dither location.
                               If set to *None*, the (non-static) NAXIS3 values from the
                               FITS headers will be used.
        :type star_pos_shift: int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str
        :return: None
        """

        super(SimpleBackgroundSubtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_star_prs_shift = star_pos_shift

    def run(self):
        """
        Run method of the module. Simple background subtraction which uses a constant index offset. 

        :return: None
        """

        # Use NAXIS3 values if star_pos_shift is None
        if self.m_star_prs_shift is None:
            self.m_star_prs_shift = self.m_image_in_port.get_attribute("NAXIS3")

        number_of_frames = self.m_image_in_port.get_shape()[0]

        # First subtraction to set up the output port array
        if isinstance(self.m_star_prs_shift, np.ndarray):
            # Modulo is needed when the index offset exceeds the total number of frames
            tmp_res = self.m_image_in_port[0] - \
                      self.m_image_in_port[(0 + self.m_star_prs_shift[0]) % number_of_frames]

        else:
            tmp_res = self.m_image_in_port[0] - \
                      self.m_image_in_port[(0 + self.m_star_prs_shift) % number_of_frames]

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            self.m_image_out_port[0] = tmp_res
        else:
            self.m_image_out_port.set_all(tmp_res, data_dim=3)

        # Background subtraction of the rest of the data
        if isinstance(self.m_star_prs_shift, np.ndarray):
            frame_count = 1
            for i, naxis_three in enumerate(self.m_star_prs_shift):
                for j in range(naxis_three):
                    if i == 0 and j == 0:
                        continue
                    else:
                        # TODO This will cause problems if the NAXIS3 value decreases and the
                        # amount of dithering positions is small, e.g. two dithering positions
                        # with subsequent NAXIS3 values of 20, 10, and 10. Also, the modulo does
                        # not guarentee to give a correct background frame. With the current
                        # implementation some background frames are used twice when the NAXIS3
                        # values changes which is not necessarily wrong but something to be
                        # aware of.
                        if j == 0 and i < np.size(self.m_star_prs_shift)-1 and \
                                  self.m_star_prs_shift[i+1] > naxis_three:
                            warnings.warn("A small number of dither positions and variations of "
                                          "NAXIS3 may give incorrect results with the current "
                                          "implementation.")

                        tmp_res = self.m_image_in_port[frame_count] - \
                                  self.m_image_in_port[(frame_count + naxis_three) % number_of_frames]

                    frame_count += 1

                    if self.m_image_in_port.tag == self.m_image_out_port.tag:
                        self.m_image_out_port[i] = tmp_res
                    else:
                        self.m_image_out_port.append(tmp_res)
        else:
            for i in range(1, number_of_frames):
                tmp_res = self.m_image_in_port[i] - \
                          self.m_image_in_port[(i + self.m_star_prs_shift) % number_of_frames]

                if self.m_image_in_port.tag == self.m_image_out_port.tag:
                    self.m_image_out_port[i] = tmp_res
                else:
                    self.m_image_out_port.append(tmp_res)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_history_information("Background",
                                                      "simple subtraction")

        self.m_image_out_port.close_port()
