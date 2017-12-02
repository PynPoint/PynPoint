"""
Modules for pre-processing of NACO data sets.
"""

import numpy as np

from PynPoint.core.Processing import ProcessingModule


class CutTopLinesModule(ProcessingModule):
    """
    Module to equalize the number of pixels in horizontal and vertical direction by
    removing several rows of pixels at the top of each frame.
    """

    def __init__(self,
                 name_in="NACO_cutting",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_cut",
                 num_lines=2,
                 num_images_in_memory=100):
        """
        Constructor of CutTopLinesModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag* unless *number_of_images_in_memory*
                              is set to *None*.
        :type image_out_tag: str
        :param num_lines: Number of top rows to delete from each frame.
        :type num_lines: int
        :param num_image_in_memory: Number of frames that are simultaneously loaded into the memory.
        :type num_image_in_memory: int
        :return: None
        """

        super(CutTopLinesModule, self).__init__(name_in)

        if image_in_tag == image_out_tag and num_images_in_memory is not None:
            raise ValueError("Input and output tags need to be different since the "
                             "CutTopLinesModule changes the size of the frames. The database can"
                             " not update existing frames with smaller new frames. The only way to "
                             "use the same input and output tags is to update all frames at once"
                             "(i.e. loading all frames to the memory). Set number_of_images_in_"
                             "memory to None to do this (Note this needs a lot of memory).")

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_num_images_in_memory = num_images_in_memory
        self.m_num_lines = num_lines

    def run(self):
        """
        Run method of the module. Removes the top *num_lines* lines from each frame.

        :return: None
        """

        def cut_top_lines(image_in):
            return image_in[:-int(self.m_num_lines), :]

        self.apply_function_to_images(cut_top_lines,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_image_out_port.add_history_information("NACO preparation",
                                                      "cut top lines")

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class AngleCalculationModule(ProcessingModule):
    """
    Module for calculating the parallactic angle values by interpolating between the begin and end
    value of a data cube.
    """

    def __init__(self,
                 name_in="angle_calculation",
                 data_tag="im_arr"):
        """
        Constructor of AngleCalculationModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param data_tag: Tag of the database entry for which the parallactic angles are written as
                         attributes.
        :type data_tag: str
        """

        super(AngleCalculationModule, self).__init__(name_in)

        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

    def run(self):
        """
        Run method of the module. Calculates the parallactic angles of each frame by linearly
        interpolating between the start and end values of the data cubes. The values are written
        as attributes to *data_tag*.

        :return: None
        """

        input_angles_start = self.m_data_in_port.get_attribute("ESO TEL PARANG START")
        input_angles_end = self.m_data_in_port.get_attribute("ESO TEL PARANG END")

        steps = self.m_data_in_port.get_attribute("NAXIS3")
        ndit = self.m_data_in_port.get_attribute("ESO DET NDIT")

        if False in (ndit == steps):
            raise ValueError("Parallactic angles should be calculated when NAXIS3 is equal to "
                             "NDIT. This implies for NACO cube data that the last frame (NDIT+1) "
                             "from each cube should have been removed while additional frame "
                             "selection should be applied after the parallactic angles have been "
                             "calculated.")

        new_angles = []

        for i in range(0, len(input_angles_start)):
            new_angles = np.append(new_angles,
                                   np.linspace(input_angles_start[i],
                                               input_angles_end[i],
                                               num=steps[i]))

        self.m_data_out_port.add_attribute("NEW_PARA",
                                           new_angles,
                                           static=False)


class RemoveLastFrameModule(ProcessingModule):
    """
    Module for removing every NDIT+1 frame from NACO data obtained in cube mode. This frame contains
    the average pixel values of the cube.
    """

    def __init__(self,
                 name_in="remove_last_frame",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_last"):
        """
        Constructor of RemoveLastFrameModule.

        :param name_in: Name of the module instance. Used as unique identifier in the Pypeline
                        dictionary.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str
        :return: None
        """

        super(RemoveLastFrameModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Removes every NDIT+1 frame and saves the data and attributes.

        :return: None
        """

        if self.m_image_out_port.tag == self.m_image_in_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        ndit = self.m_image_in_port.get_attribute("ESO DET NDIT")
        size = self.m_image_in_port.get_attribute("NAXIS3")

        if False in (size == ndit+1):
            raise ValueError("This module should be used when NAXIS3 = NDIT + 1.")

        ndit_tot = 0
        for i, _ in enumerate(ndit):
            tmp_in = self.m_image_in_port[ndit_tot:ndit_tot+ndit[i]+1,]
            tmp_out = np.delete(tmp_in, ndit[i], axis=0)

            if ndit_tot == 0:
                self.m_image_out_port.set_all(tmp_out, keep_attributes=True)
            else:
                self.m_image_out_port.append(tmp_out)

            ndit_tot += ndit[i]+1

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_attribute("NAXIS3", size-1, static=False)

        self.m_image_out_port.add_history_information("NACO preparation",
                                                      "remove every NDIT+1 frame")

        self.m_image_out_port.close_port()
