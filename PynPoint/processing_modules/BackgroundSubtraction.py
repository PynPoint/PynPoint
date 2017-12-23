"""
Modules with background subtraction routines.
"""

import numpy as np

from PynPoint.core.Processing import ProcessingModule


class MeanBackgroundSubtractionModule(ProcessingModule):
    """
    Module for mean background subtraction, only applicable for dithered data.
    """

    def __init__(self,
                 star_pos_shift=None,
                 cubes_per_position=1,
                 name_in="mean_background_subtraction",
                 image_in_tag="im_arr",
                 image_out_tag="bg_cleaned_arr"):
        """
        Constructor of MeanBackgroundSubtractionModule.

        :param star_pos_shift: Frame index offset for the background subtraction. Typically equal
                               to the number of frames per dither location. If set to *None*, the
                               (non-static) NAXIS3 values from the FITS headers will be used.
        :type star_pos_shift: int
        :param cube_per_position: Number of consecutive cubes per dithering position.
        :type cube_per_position: int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(MeanBackgroundSubtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_star_prs_shift = star_pos_shift
        self.m_cubes_per_position = cubes_per_position

    def run(self):
        """
        Run method of the module. Mean background subtraction which uses either a constant index
        offset or the (non-static) NAXIS3 values for the headers. The mean background is calculated
        from the cubes before and after the science cube(s).

        :return: None
        """

        # Use NAXIS3 values if star_pos_shift is None
        if self.m_star_prs_shift is None:
            self.m_star_prs_shift = self.m_image_in_port.get_attribute("NFRAMES")

        number_of_frames = self.m_image_in_port.get_shape()[0]

        # Check size of the input, only needed when a manual star_pos_shift is provided
        if not isinstance(self.m_star_prs_shift, np.ndarray) and \
               number_of_frames < self.m_star_prs_shift*2.0:
            raise ValueError("The input stack is to small for mean background subtraction. At least"
                             "one star position shift is needed.")

        # Number of substacks
        if isinstance(self.m_star_prs_shift, np.ndarray):
            num_stacks = np.size(self.m_star_prs_shift)
        else:
            num_stacks = int(np.floor(number_of_frames/self.m_star_prs_shift))

        # First mean subtraction to set up the output port array
        if isinstance(self.m_star_prs_shift, np.ndarray):
            next_start = np.sum(self.m_star_prs_shift[0:self.m_cubes_per_position])
            next_end = np.sum(self.m_star_prs_shift[0:2*self.m_cubes_per_position])

            if 2*self.m_cubes_per_position > np.size(self.m_star_prs_shift):
                raise ValueError("Not enough frames available for the background subtraction.")

            # Calculate the mean background of cubes_per_position number of cubes
            tmp_data = self.m_image_in_port[next_start:next_end,]
            tmp_mean = np.mean(tmp_data, axis=0)

        else:
            tmp_data = self.m_image_in_port[self.m_star_prs_shift:2*self.m_star_prs_shift,]
            tmp_mean = np.mean(tmp_data, axis=0)

        # Initiate the result port data with the first frame
        tmp_res = self.m_image_in_port[0,] - tmp_mean

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise NotImplementedError("Same input and output port not implemented yet.")
        else:
            self.m_image_out_port.set_all(tmp_res, data_dim=3)

        print "Subtracting background from stack-part " + str(1) + " of " + \
              str(num_stacks) + " stack-parts"

        # Mean subtraction of the first stack (minus the first frame)
        if isinstance(self.m_star_prs_shift, np.ndarray):
            for i in range(1, self.m_cubes_per_position):
                print "Subtracting background from stack-part " + str(i+1) + " of " + \
                      str(num_stacks) + " stack-parts"

            tmp_data = self.m_image_in_port[1:next_start,]
            tmp_data = tmp_data - tmp_mean

            self.m_image_out_port.append(tmp_data)

        else:
            tmp_data = self.m_image_in_port[1:self.m_star_prs_shift, :, :]
            tmp_data = tmp_data - tmp_mean

            # TODO This will not work for same in and out port
            self.m_image_out_port.append(tmp_data)

        # Processing of the rest of the data
        if isinstance(self.m_star_prs_shift, np.ndarray):
            for i in range(self.m_cubes_per_position, num_stacks, self.m_cubes_per_position):
                prev_start = np.sum(self.m_star_prs_shift[0:i-self.m_cubes_per_position])
                prev_end = np.sum(self.m_star_prs_shift[0:i])

                next_start = np.sum(self.m_star_prs_shift[0:i+self.m_cubes_per_position])
                next_end = np.sum(self.m_star_prs_shift[0:i+2*self.m_cubes_per_position])

                for j in range(self.m_cubes_per_position):
                    print "Subtracting background from stack-part " + str(i+j+1) + " of " + \
                          str(num_stacks) + " stack-parts"

                # calc the mean (previous)
                tmp_data = self.m_image_in_port[prev_start:prev_end,]
                tmp_mean = np.mean(tmp_data, axis=0)

                if i < num_stacks-self.m_cubes_per_position:
                    # calc the mean (next)
                    tmp_data = self.m_image_in_port[next_start:next_end,]
                    tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

                # subtract mean
                tmp_data = self.m_image_in_port[prev_end:next_start,]
                tmp_data = tmp_data - tmp_mean
                self.m_image_out_port.append(tmp_data)

        else:
            # the last and the one before will be performed afterwards
            top = int(np.ceil(number_of_frames /
                              self.m_star_prs_shift)) - 2

            for i in range(1, top, 1):
                print "Subtracting background from stack-part " + str(i+1) + " of " + \
                      str(num_stacks) + " stack-parts"
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
            print "Subtracting background from stack-part " + str(top+1) + " of " + \
                  str(num_stacks) + " stack-parts"
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
            print "Subtracting background from stack-part " + str(top+2) + " of " + \
                  str(num_stacks) + " stack-parts"
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

        :param star_pos_shift: Frame index offset for the background subtraction. Typically equal
                               to the number of frames per dither location.
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

        # add Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_star_prs_shift = star_pos_shift

    def run(self):
        """
        Run method of the module. Simple background subtraction which uses either a constant
        index offset.

        :return: None
        """

        number_of_frames = self.m_image_in_port.get_shape()[0]

        # first subtraction is used to set up the output port array
        tmp_res = self.m_image_in_port[0] - \
                  self.m_image_in_port[(0 + self.m_star_prs_shift) % number_of_frames]

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            self.m_image_out_port[0] = tmp_res
        else:
            self.m_image_out_port.set_all(tmp_res, data_dim=3)

        # process with the rest of the data
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
