

from PynPoint.core.Processing import ProcessingModule


class SimpleBackgroundSubtractionModule(ProcessingModule):

    def __init__(self,
                 star_pos_shift,
                 name_in="background_subtraction",
                 image_in_tag = "im_arr",
                 image_out_tag = "bg_cleaned_arr"):

        super(SimpleBackgroundSubtractionModule, self).__init__(name_in)

        # add Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_star_prs_shift = star_pos_shift

    def run(self):

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

        self.m_image_out_port.add_history_information("background sub",
                                                      "simple frame - frame")

        self.m_image_out_port.close_port()



