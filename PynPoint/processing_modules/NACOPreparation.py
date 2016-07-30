import numpy as np

from PynPoint.core.Processing import ProcessingModule


class CutTopTwoLinesModule(ProcessingModule):

    def __init__(self,
                 name_in="NACO_cutting",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_cut",
                 num_images_in_memory=100):

        super(CutTopTwoLinesModule, self).__init__(name_in)

        if image_in_tag == image_out_tag and num_images_in_memory is not None:
            raise ValueError("Input and output tags need to be different since the "
                             "CutTopTwoLinesModule changes the size of the frames. The database can"
                             " not update existing frames with smaller new frames. The only way to "
                             "use the same input and output tags is to update all frames at once"
                             "(i.e. loading all frames to the memory). Set number_of_images_in_"
                             "memory to None to do this (Note this needs a lot of memory).")

        # add Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_num_images_in_memory = num_images_in_memory

    def run(self):

        def cut_top_two_lines(image_in):
            return image_in[:-2, :]

        self.apply_function_to_images(cut_top_two_lines,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_image_out_port.add_history_information("NACO_preparation",
                                                      "cut top two lines")

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class AngleCalculationModule(ProcessingModule):

    def __init__(self,
                 name_in="angle_calcuation",
                 data_tag="im_arr"):

        super(AngleCalculationModule, self).__init__(name_in)

        # Ports
        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

    def run(self):

        input_angles_start = self.m_data_in_port.get_attribute("ESO TEL PARANG START")
        input_angles_end = self.m_data_in_port.get_attribute("ESO TEL PARANG END")

        steps = self.m_data_in_port.get_attribute("NAXIS3")

        new_angles = []

        for i in range(0, len(input_angles_start)):
            new_angles = np.append(new_angles,
                                   np.linspace(input_angles_start[i],
                                               input_angles_end[i],
                                               num=steps))

        self.m_data_out_port.add_attribute("NEW_PARA",
                                           new_angles,
                                           static=False)
